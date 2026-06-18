"""Supervised fine-tuning loop (assistant-only ChatML masking).

`run(model, cfg, mcfg)` fine-tunes a model that satisfies `lmkit.protocol.LMModel`
on a ChatML JSONL dataset (`cfg.data_file`), initialized from `cfg.init_from`.
The caller builds and places the model (device + dtype); the loop owns the data,
optimizer, schedule, checkpointing, metrics, and optional Aim tracking.
Device-agnostic. Returns 0 on completion/clean SIGTERM, 1 if no windows build,
2 on non-finite loss.
"""
from __future__ import annotations

import math
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .observability import make_run
from .sft_data import SFTDataset, build_sft_windows
from .training import (
    achieved_tflops, autocast_ctx, build_optimizer, emit, get_lr, log_metrics,
    peak_vram_gb, resolve_dtype, save_ckpt,
)

_STOP = False


def _request_stop(signum, frame):
    global _STOP
    _STOP = True


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Cross-entropy over assistant tokens only; -100 positions are ignored. A
    batch with zero trainable targets would make cross_entropy average over zero
    elements -> NaN, which aborts the run; return a grad-connected 0 instead."""
    flat = labels.reshape(-1)
    if (flat != -100).sum() == 0:
        return logits.reshape(-1, vocab_size).sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, vocab_size), flat, ignore_index=-100)


def compute_schedule(n_windows: int, cfg) -> tuple[int, int, int]:
    """(steps_per_epoch, max_steps, warmup_steps) from data size + epochs."""
    eff_batch = cfg.batch_size * cfg.grad_accum
    steps_per_epoch = max(1, n_windows // eff_batch)
    max_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(cfg.warmup_frac * max_steps))
    return steps_per_epoch, max_steps, warmup_steps


def run(model, cfg, mcfg, *, experiment: str = "sft") -> int:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"

    try:
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
    except ValueError:
        pass

    torch.manual_seed(cfg.seed)
    device = next(model.parameters()).device
    dtype = resolve_dtype(cfg)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(cfg.tokenizer_path)
    windows = build_sft_windows(cfg.data_file, tok, mcfg.block_size)
    if not windows:
        print("ERROR: no training windows built from", cfg.data_file, file=sys.stderr)
        return 1

    n_val = max(1, len(windows) // 100)
    val_windows, train_windows = windows[:n_val], windows[n_val:]
    train_ds, val_ds = SFTDataset(train_windows), SFTDataset(val_windows)
    steps_per_epoch, cfg.max_steps, cfg.warmup_steps = compute_schedule(len(train_ds), cfg)

    if getattr(cfg, "init_from", ""):
        ckpt = torch.load(cfg.init_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"init from {cfg.init_from} (pretrain step {ckpt.get('step', '?')})")

    optimizer = build_optimizer(model, cfg)
    if cfg.compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SFT: {len(train_ds):,} train / {len(val_ds):,} val windows, "
          f"{steps_per_epoch:,} steps/epoch x {cfg.epochs} = {cfg.max_steps:,} steps on {device.type}")
    emit(metrics_path, {"event": "start", "step": 0, "max_steps": cfg.max_steps,
                        "train_windows": len(train_ds), "init_from": getattr(cfg, "init_from", "")})
    run_ = make_run(experiment, hparams={"params": n_params, **cfg.to_dict(), **mcfg.to_dict()})

    DataLoader = torch.utils.data.DataLoader
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    def _batches():
        while True:
            for b in train_loader:
                yield b

    batches = _batches()
    best_val = float("inf")
    t0 = time.time()
    step = 0
    while step < cfg.max_steps:
        if _STOP:
            save_ckpt(latest_path, model, optimizer, step, best_val, mcfg, cfg.compile)
            emit(metrics_path, {"event": "sigterm", "step": step})
            if run_:
                run_.close()
            return 0

        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if step % cfg.eval_interval == 0 and len(val_ds) >= cfg.batch_size:
            model.eval()
            vl, n = 0.0, 0
            with torch.no_grad():
                for vb, (x, y) in enumerate(val_loader):
                    if vb >= cfg.eval_iters:
                        break
                    x, y = x.to(device), y.to(device)
                    with autocast_ctx(device, dtype):
                        vl += masked_loss(model(x), y, mcfg.vocab_size).item()
                    n += 1
            vl = vl / max(1, n)
            improved = vl < best_val
            if improved:
                best_val = vl
                save_ckpt(best_path, model, optimizer, step, best_val, mcfg, cfg.compile)
            log_metrics(run_, metrics_path, "eval", step, {
                "val_loss": vl, "val_perplexity": math.exp(min(vl, 20)),
                "best_val": best_val, "lr": lr, "improved": improved,
                "epoch": step / max(steps_per_epoch, 1)})
            print(f"step {step:6d} | val {vl:.4f}{' (best)' if improved else ''} | lr {lr:.2e}")
            model.train()

        if step > 0 and step % cfg.save_interval == 0:
            save_ckpt(latest_path, model, optimizer, step, best_val, mcfg, cfg.compile)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        aux_accum = 0.0
        has_aux = hasattr(model, "aux_loss")
        for _ in range(cfg.grad_accum):
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            with autocast_ctx(device, dtype):
                loss = masked_loss(model(x), y, mcfg.vocab_size)
                if has_aux:
                    aux = model.aux_loss()
                    aux_accum += float(aux)
                    loss = loss + aux
                loss = loss / cfg.grad_accum
            loss.backward()
            loss_accum += loss.item()

        if not (loss_accum == loss_accum and abs(loss_accum) != float("inf")):
            emit(metrics_path, {"event": "nan", "step": step})
            print(f"NON-FINITE LOSS at step {step}", file=sys.stderr)
            if run_:
                run_.close()
            return 2

        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        optimizer.step()

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            tok_s = cfg.batch_size * cfg.grad_accum * mcfg.block_size * max(cfg.log_interval, 1) / max(dt, 1e-9)
            m = {
                "train_loss": loss_accum, "lr": lr, "grad_norm": gnorm,
                "tok_per_sec": tok_s, "step_time_ms": 1000 * dt / max(cfg.log_interval, 1),
                "tokens_seen": (step + 1) * cfg.batch_size * cfg.grad_accum * mcfg.block_size,
                "tflops": achieved_tflops(n_params, tok_s),
                "peak_vram_gb": peak_vram_gb(device), "epoch": step / max(steps_per_epoch, 1)}
            if has_aux:
                m["aux_loss"] = aux_accum / max(cfg.grad_accum, 1)
            log_metrics(run_, metrics_path, "train", step, m)
            print(f"step {step:6d} | loss {loss_accum:.4f} | {tok_s/1e3:.1f}k tok/s | gnorm {gnorm:.2f} | lr {lr:.2e}")
            t0 = time.time()
        step += 1

    save_ckpt(latest_path, model, optimizer, step, best_val, mcfg, cfg.compile)
    emit(metrics_path, {"event": "done", "step": step, "best_val": best_val})
    if run_:
        run_.track(best_val, name="best_val_final")
        run_.close()
    print(f"done. {step} steps, best_val {best_val:.4f}")
    return 0
