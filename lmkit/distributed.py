"""Data-parallel plumbing for the pretrain loop (torchrun / env-var rendezvous).

Deliberately not `DistributedDataParallel`: the loop averages gradients itself, once
per optimizer step, in one coalesced all-reduce. That keeps models unwrapped (so
`aux_loss`, `state_dict` and checkpoints are untouched), tolerates parameters that
receive no gradient on some rank (MoE experts nobody routed to), and cannot hang on
DDP's unused-parameter bookkeeping. The cost is no comm/compute overlap, which is
small next to a multi-second optimizer step.

`init()` is a no-op for a plain `python train.py` (world size 1), so single-process
callers are unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

# Rank 0 can spend minutes writing a multi-GB checkpoint while the others wait at
# the next collective; the 10 min default is too tight for that.
_TIMEOUT = timedelta(minutes=60)


@dataclass(frozen=True)
class Dist:
    rank: int = 0
    world: int = 1
    local_rank: int = 0

    @property
    def active(self) -> bool:
        return self.world > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def mean(self, values: list[float], device) -> list[float]:
        """Mean of per-rank scalars. Every rank must call with the same length."""
        if not self.active:
            return list(values)
        t = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(t)
        return (t / self.world).tolist()

    def any(self, flag: bool, device) -> bool:
        if not self.active:
            return flag
        t = torch.tensor([int(flag)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    def barrier(self) -> None:
        if self.active:
            dist.barrier()

    def broadcast_params(self, model) -> None:
        """Make every replica equal to rank 0 (fresh starts build the model from
        per-rank RNG state; resumes already agree and this is a cheap no-op)."""
        if not self.active:
            return
        tensors = [t.data for t in (*model.parameters(), *model.buffers())]
        _coalesced(tensors, lambda flat: dist.broadcast(flat, src=0))

    def sync_grads(self, params) -> None:
        """Replace each gradient with its mean over ranks. A parameter with no
        gradient on this rank contributes zeros, so all ranks issue identical
        collectives."""
        if not self.active:
            return
        grads = []
        for p in params:
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)

        def reduce(flat):
            dist.all_reduce(flat)
            flat.div_(self.world)

        _coalesced(grads, reduce)


def _coalesced(tensors, collective) -> None:
    """Run `collective` in place over one flat buffer per (device, dtype)."""
    groups: dict[tuple, list[torch.Tensor]] = {}
    for t in tensors:
        groups.setdefault((t.device, t.dtype), []).append(t)
    with torch.no_grad():
        for group in groups.values():
            flat = torch.cat([t.reshape(-1) for t in group])
            collective(flat)
            offset = 0
            for t in group:
                n = t.numel()
                t.copy_(flat[offset:offset + n].view_as(t))
                offset += n


def init() -> Dist:
    """Join the process group described by torchrun's env vars, pin this process
    to its local GPU, and return the topology. Idempotent. Call before building
    the model so `.to("cuda")` lands on the right card."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return Dist()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # ROCm builds expose RCCL under the "nccl" backend name.
        dist.init_process_group("nccl" if cuda else "gloo", rank=rank,
                                world_size=world, timeout=_TIMEOUT)
    return Dist(rank=rank, world=world, local_rank=local_rank)
