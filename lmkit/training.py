"""Shared training machinery used by both the pretrain and SFT loops:
LR schedule, optimizer, checkpoint IO, metrics, device-aware autocast, and the
token-shard dataset. Everything is device-agnostic so the loops run on CPU
(for the quickstart and tests) or CUDA/ROCm unchanged.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch


def resolve_dtype(cfg) -> torch.dtype:
    return torch.bfloat16 if getattr(cfg, "dtype", "float32") == "bfloat16" else torch.float32


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    """bf16 autocast on CUDA; a no-op elsewhere (CPU runs in fp32)."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def get_lr(step: int, cfg) -> float:
    """WSD: linear warmup -> constant (stable) lr -> cosine decay to min_lr over the
    final `decay_frac` of max_steps. `decay_frac == 0` is a pure stable trunk (no
    decay tail), resumable indefinitely for chunked continued pretraining."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    decay_steps = int(cfg.decay_frac * cfg.max_steps)
    decay_start = cfg.max_steps - decay_steps
    if step < decay_start:
        return cfg.lr
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - decay_start) / max(1, decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def build_optimizer(model, cfg):
    """AdamW with weight decay on 2D+ params only; fused on CUDA."""
    decay = [p for _, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for _, p in model.named_parameters() if p.dim() < 2]
    fused = next(model.parameters()).device.type == "cuda"
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=fused,
    )


def save_ckpt(path: Path, model, optimizer, step: int, best_val: float,
              mcfg, compile: bool = False) -> None:
    """Atomic checkpoint: weights + optimizer + step + model config."""
    raw = getattr(model, "_orig_mod", model) if compile else model
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "model_cfg": mcfg.to_dict() if hasattr(mcfg, "to_dict") else None,
    }, tmp)
    os.replace(tmp, path)


def emit(metrics_path: Path, payload: dict) -> None:
    payload["ts"] = time.time()
    with open(metrics_path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def log_metrics(run, metrics_path: Path, event: str, step: int, metrics: dict) -> None:
    """Write one event to metrics.jsonl and track every numeric value to Aim (when
    a run is active). One call site per logging point keeps both in lockstep."""
    emit(metrics_path, {"event": event, "step": step, **metrics})
    if run:
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                run.track(v, name=k, step=step)


def peak_vram_gb(device: torch.device) -> float:
    """Peak allocated VRAM since the last reset (CUDA/ROCm), in GB; 0 on CPU."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1e9


def achieved_tflops(n_params: int, tokens_per_sec: float) -> float:
    """Rough achieved compute: ~6 FLOPs/param/token (fwd+bwd), in TFLOP/s. Divide
    by your accelerator's peak to get MFU."""
    return 6.0 * n_params * tokens_per_sec / 1e12


def prune_snapshots(out_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    snaps = sorted(p for p in Path(out_dir).glob("step_*") if p.is_dir())
    for old in snaps[:-keep_last]:
        shutil.rmtree(old, ignore_errors=True)


class TokenDataset(torch.utils.data.IterableDataset):
    """Streams (x, y) next-token windows from uint16 ``{split}_*.bin`` shards in
    ``data_dir`` (numpy memmap). A single ``{split}.bin`` is also accepted."""

    def __init__(self, data_dir: str, block_size: int, split: str = "train"):
        import numpy as np

        d = Path(data_dir)
        files = sorted(d.glob(f"{split}_*.bin")) or sorted(d.glob(f"{split}.bin"))
        if not files:
            raise FileNotFoundError(f"no {split}_*.bin or {split}.bin in {data_dir}")
        self.shards = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.block_size = block_size

    def __iter__(self):
        import numpy as np

        rng = np.random.default_rng()
        bs = self.block_size
        while True:
            shard = self.shards[rng.integers(len(self.shards))]
            idx = rng.integers(0, len(shard) - bs - 1)
            chunk = torch.from_numpy(shard[idx: idx + bs + 1].astype("int64"))
            yield chunk[:-1], chunk[1:]
