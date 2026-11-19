"""Data-parallel pretraining: gradient sync, rank-0-only side effects, synchronized
stop. Workers are real processes joined over gloo (CPU) using the same env-var
rendezvous torchrun uses, so `distributed.init()` is exercised as in production."""
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from lmkit import distributed, pretrain


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(fn, world: int, *args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    port = _free_port()
    mp.start_processes(_entry, args=(fn, world, port, *args), nprocs=world,
                       start_method="spawn", join=True)


def _entry(rank, fn, world, port, *args):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    try:
        fn(rank, world, *args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


# --- tiny model/config satisfying lmkit.protocol.LMModel ----------------------

class TinyLM(nn.Module):
    def __init__(self, vocab=32, hidden=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.out = nn.Linear(hidden, vocab, bias=False)
        self.unused = nn.Linear(hidden, hidden, bias=False)  # never in the graph
        self.calls = 0
        self.on_call = None

    def forward(self, x):
        self.calls += 1
        if self.on_call:
            self.on_call(self.calls)
        return self.out(self.emb(x))


def _cfgs(tmp: Path, **over):
    d = tmp / "data"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        rng.integers(0, 32, 4096, dtype=np.uint16).tofile(d / f"{split}_00000.bin")
    t = SimpleNamespace(
        out_dir=str(tmp / "out"), data_dir=str(d), batch_size=2, grad_accum=4,
        max_steps=6, eval_interval=3, eval_iters=4, log_interval=2, save_interval=3,
        snapshot_interval=0, keep_last_snapshots=0, lr=1e-2, min_lr=1e-3,
        warmup_steps=1, decay_frac=0.0, weight_decay=0.0, beta1=0.9, beta2=0.95,
        grad_clip=1.0, dtype="float32", compile=False, seed=7, device="cpu")
    t.to_dict = lambda: {k: v for k, v in vars(t).items() if k != "to_dict"}
    for k, v in over.items():
        setattr(t, k, v)
    m = SimpleNamespace(n_layer=1, hidden_size=8, block_size=16, vocab_size=32,
                        to_dict=lambda: {})
    return t, m


# --- sync_grads ---------------------------------------------------------------

def _w_sync(rank, world, outdir):
    d = distributed.init()
    assert d.active and d.world == world and d.rank == rank
    torch.manual_seed(100 + rank)  # different data per rank
    model = TinyLM()
    d.broadcast_params(model)
    x = torch.randint(0, 32, (4, 8))
    model(x).sum().backward()
    local = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    d.sync_grads(model.parameters())
    torch.save({"synced": {n: (p.grad.clone() if p.grad is not None else None)
                           for n, p in model.named_parameters()},
                "local": local}, Path(outdir) / f"r{rank}.pt")


def test_sync_grads_is_the_mean_across_ranks_and_unused_params_get_zeros(tmp_path):
    _spawn(_w_sync, 2, str(tmp_path))
    r0, r1 = (torch.load(tmp_path / f"r{i}.pt") for i in range(2))
    for n in r0["synced"]:
        assert torch.equal(r0["synced"][n], r1["synced"][n]), n
    for n, local0 in r0["local"].items():
        assert torch.allclose(r0["synced"][n], (local0 + r1["local"][n]) / 2, atol=1e-6), n
    assert torch.count_nonzero(r0["synced"]["unused.weight"]) == 0


# --- end-to-end pretrain.run --------------------------------------------------

def _w_run(rank, world, tmp, stop_rank):
    d = distributed.init()
    torch.manual_seed(rank)  # ranks start from DIFFERENT weights: run() must reconcile
    model = TinyLM()
    if stop_rank == rank:
        model.on_call = lambda n: setattr(pretrain, "_STOP", True) if n == 5 else None
    tcfg, mcfg = _cfgs(Path(tmp), max_steps=50 if stop_rank is not None else 6)
    rc = pretrain.run(model, tcfg, mcfg)
    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(world)]
    torch.distributed.all_gather(gathered, flat)
    Path(tmp, f"rc{rank}").write_text(str(rc))
    if rank == 0:
        torch.save(gathered, Path(tmp) / "weights.pt")


def _events(tmp):
    return [json.loads(l) for l in (Path(tmp) / "out" / "metrics.jsonl").read_text().splitlines()]


def test_two_rank_run_keeps_replicas_identical_and_logs_once(tmp_path):
    _spawn(_w_run, 2, str(tmp_path), None)
    w = torch.load(tmp_path / "weights.pt")
    assert torch.equal(w[0], w[1])
    ev = _events(tmp_path)
    assert [e["event"] for e in ev].count("start") == 1
    assert [e["event"] for e in ev].count("done") == 1
    assert (tmp_path / "out" / "final" / "model.pt").exists()
    assert (tmp_path / "out" / "latest.pt").exists()


def _w_stop(rank, world, tmp):
    _w_run(rank, world, tmp, 0)


def test_sigterm_on_one_rank_stops_all_ranks_at_the_same_step_without_hanging(tmp_path):
    _spawn(_w_stop, 2, str(tmp_path))
    assert Path(tmp_path, "rc0").read_text() == Path(tmp_path, "rc1").read_text() == "0"
    stops = [e for e in _events(tmp_path) if e["event"] == "sigterm"]
    assert len(stops) == 1
    pretrain._STOP = False


def _w_accum(rank, world, tmp):
    distributed.init()
    tcfg, mcfg = _cfgs(Path(tmp), grad_accum=3)
    with pytest.raises(ValueError, match="grad_accum"):
        pretrain.run(TinyLM(), tcfg, mcfg)


def test_grad_accum_must_divide_across_ranks(tmp_path):
    _spawn(_w_accum, 2, str(tmp_path))


def test_single_process_run_is_unchanged(tmp_path):
    tcfg, mcfg = _cfgs(tmp_path)
    assert pretrain.run(TinyLM(), tcfg, mcfg) == 0
    assert [e["event"] for e in _events(tmp_path)].count("done") == 1
