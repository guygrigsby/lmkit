"""Pretraining loop (WSD: warmup -> stable -> optional cosine anneal tail).

`run(model, tcfg, mcfg)` trains a model that satisfies `lmkit.protocol.LMModel`
on token shards in `tcfg.data_dir`. The caller builds and places the model
(device + dtype); the loop owns the optimizer, resume, checkpointing, metrics,
and optional Aim tracking. Device-agnostic: runs on CPU or CUDA/ROCm.

Returns 0 on completion or clean SIGTERM, 2 on non-finite loss (resume from
`latest.pt`).
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
from .training import (
    TokenDataset, achieved_tflops, autocast_ctx, build_optimizer, emit, get_lr,
    log_metrics, peak_vram_gb, prune_snapshots, resolve_dtype, save_ckpt,
)

_STOP = False


def _request_stop(signum, frame):
    global _STOP
    _STOP = True


def run(model, tcfg, mcfg, *, experiment: str = "pretrain") -> int:
    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"

    try:
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
    except ValueError:
        pass  # not in the main thread; skip handlers

    torch.manual_seed(tcfg.seed)
    device = next(model.parameters()).device
    dtype = resolve_dtype(tcfg)
    optimizer = build_optimizer(model, tcfg)

    step, best_val = 0, float("inf")
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        emit(metrics_path, {"event": "resume", "step": step})

    if tcfg.compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params/1e6:.2f}M params | {mcfg.n_layer}L {mcfg.hidden_size}h | "
          f"target {tcfg.max_steps} steps on {device.type}")
    emit(metrics_path, {"event": "start", "step": step, "params": n_params,
                        "max_steps": tcfg.max_steps})
    run_ = make_run(experiment, hparams={"params": n_params, **tcfg.to_dict(),
                                         **mcfg.to_dict()})

    train_loader = iter(torch.utils.data.DataLoader(
        TokenDataset(tcfg.data_dir, mcfg.block_size, "train"), batch_size=tcfg.batch_size))
    val_loader = iter(torch.utils.data.DataLoader(
        TokenDataset(tcfg.data_dir, mcfg.block_size, "val"), batch_size=tcfg.batch_size))

    t0 = time.time()
    last_train_loss = 0.0
    while step < tcfg.max_steps:
        if _STOP:
            save_ckpt(latest_path, model, optimizer, step, best_val, mcfg, tcfg.compile)
            emit(metrics_path, {"event": "sigterm", "step": step})
            if run_:
                run_.close()
            return 0

        lr = get_lr(step, tcfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if step % tcfg.eval_interval == 0:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for _ in range(tcfg.eval_iters):
                    x, y = next(val_loader)
                    x, y = x.to(device), y.to(device)
                    with autocast_ctx(device, dtype):
                        vl += F.cross_entropy(
                            model(x).reshape(-1, mcfg.vocab_size), y.reshape(-1)).item()
            vl /= tcfg.eval_iters
            improved = vl < best_val
            if improved:
                best_val = vl
                save_ckpt(best_path, model, optimizer, step, best_val, mcfg, tcfg.compile)
            print(f"step {step:6d} | val {vl:.4f}{' (best)' if improved else ''} | lr {lr:.2e}")
            log_metrics(run_, metrics_path, "eval", step, {
                "val_loss": vl, "val_perplexity": math.exp(min(vl, 20)),
                "best_val": best_val, "train_loss": last_train_loss,
                "lr": lr, "improved": improved})
            model.train()

        if step > 0 and step % tcfg.save_interval == 0:
            save_ckpt(latest_path, model, optimizer, step, best_val, mcfg, tcfg.compile)

        if step > 0 and getattr(tcfg, "snapshot_interval", 0) and step % tcfg.snapshot_interval == 0:
            snap = out_dir / f"step_{step:06d}"
            snap.mkdir(parents=True, exist_ok=True)
            save_ckpt(snap / "model.pt", model, optimizer, step, best_val, mcfg, tcfg.compile)
            prune_snapshots(out_dir, getattr(tcfg, "keep_last_snapshots", 0))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        aux_accum = 0.0
        has_aux = hasattr(model, "aux_loss")
        for _ in range(tcfg.grad_accum):
            x, y = next(train_loader)
            x, y = x.to(device), y.to(device)
            with autocast_ctx(device, dtype):
                loss = F.cross_entropy(model(x).reshape(-1, mcfg.vocab_size), y.reshape(-1))
                if has_aux:  # auxiliary loss (MoD router / MoE load-balance / ...)
                    aux = model.aux_loss()
                    aux_accum += float(aux)
                    loss = loss + aux
                loss = loss / tcfg.grad_accum
            loss.backward()
            loss_accum += loss.item()

        if not math.isfinite(loss_accum):
            emit(metrics_path, {"event": "nan", "step": step, "loss": loss_accum})
            print(f"NON-FINITE LOSS at step {step}", file=sys.stderr)
            if run_:
                run_.close()
            return 2

        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip))
        optimizer.step()
        last_train_loss = loss_accum

        if step % tcfg.log_interval == 0:
            dt = time.time() - t0
            tok_s = (tcfg.batch_size * tcfg.grad_accum * mcfg.block_size
                     * max(tcfg.log_interval, 1) / max(dt, 1e-9))
            print(f"step {step:6d} | loss {loss_accum:.4f} | {tok_s/1e3:.1f}k tok/s | "
                  f"gnorm {gnorm:.2f} | lr {lr:.2e}")
            m = {
                "train_loss": loss_accum, "lr": lr, "grad_norm": gnorm,
                "tok_per_sec": tok_s, "step_time_ms": 1000 * dt / max(tcfg.log_interval, 1),
                "tokens_seen": (step + 1) * tcfg.batch_size * tcfg.grad_accum * mcfg.block_size,
                "tflops": achieved_tflops(n_params, tok_s),
                "peak_vram_gb": peak_vram_gb(device),
            }
            if has_aux:
                m["aux_loss"] = aux_accum / max(tcfg.grad_accum, 1)
            log_metrics(run_, metrics_path, "train", step, m)
            t0 = time.time()
        step += 1

    final = out_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    save_ckpt(final / "model.pt", model, optimizer, step, best_val, mcfg, tcfg.compile)
    save_ckpt(latest_path, model, optimizer, step, best_val, mcfg, tcfg.compile)
    emit(metrics_path, {"event": "done", "step": step, "best_val": best_val})
    if run_:
        run_.close()
    print(f"done. {step} steps, best_val {best_val:.4f}")
    return 0
