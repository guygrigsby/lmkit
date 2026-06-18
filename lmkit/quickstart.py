"""A 60-second, CPU-only demo of the real lmkit pretrain loop.

`lmkit quickstart` char-tokenizes a tiny bundled corpus, writes token shards, and
trains a small model with ``lmkit.pretrain.run`` — the same loop used for real
training, just tiny. Watch the loss fall, then it samples a few lines. No GPU, no
downloads, no data to find.

The demo model here is intentionally minimal; for a real architecture see
``examples/llama``. lmkit trains either one through the same loop.
"""
from __future__ import annotations

import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pretrain

# A few KB of plain prose — enough structure for a tiny model to visibly learn on
# a CPU in seconds. Generic on purpose; bring your own corpus for real runs.
CORPUS = """
A language model learns to predict the next token from the ones before it.
Train it on enough text and the predictions start to look like language: words
follow words, clauses close, sentences end where sentences should. None of this
is memorized in any simple sense. The model holds a compressed picture of how the
text tends to go, and it draws on that picture one token at a time.

Training from scratch is humbler than it sounds. You begin with noise. The first
samples are gibberish, a scatter of characters with no shape. Then the loss
falls, slowly at first, and the shape arrives: spaces in the right places, common
words, the rhythm of real sentences. A small model on a small corpus will overfit
quickly, learning the text almost by heart, and that is fine for a demo. The point
is to watch the thing move, to see learning happen on hardware you already have.

Bigger runs trade patience for quality. More data, more parameters, more steps,
and the model stops parroting and starts generalizing. But the loop is the same
loop you are running right now: read a batch, predict the next token, measure the
error, nudge the weights, repeat. Everything else is scale and care.

Local training is a good teacher. When the whole pipeline fits on one machine you
can see each part, change it, and watch what happens. That is the spirit here:
small, legible, and yours to take apart.
""".strip()


@dataclass
class _MCfg:
    vocab_size: int = 0
    hidden_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    block_size: int = 64

    def to_dict(self):
        return asdict(self)


@dataclass
class _TCfg:
    out_dir: str = ""
    data_dir: str = ""
    batch_size: int = 16
    grad_accum: int = 1
    max_steps: int = 600
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 50
    save_interval: int = 100000        # don't bother saving in the demo
    snapshot_interval: int = 0
    keep_last_snapshots: int = 0
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 50
    decay_frac: float = 0.5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "float32"
    compile: bool = False
    seed: int = 1337
    device: str = "cpu"

    def to_dict(self):
        return asdict(self)


class _DemoModel(nn.Module):
    """A minimal GPT-style decoder, just enough to demo the loop. Real
    architectures (RoPE/GQA/SwiGLU) live in examples/llama; this stays tiny."""

    def __init__(self, cfg: _MCfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        self.tok = nn.Embedding(cfg.vocab_size, h)
        self.pos = nn.Embedding(cfg.block_size, h)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(h),
                "attn": nn.MultiheadAttention(h, cfg.n_head, batch_first=True),
                "ln2": nn.LayerNorm(h),
                "mlp": nn.Sequential(nn.Linear(h, 4 * h), nn.GELU(), nn.Linear(4 * h, h)),
            }) for _ in range(cfg.n_layer)
        ])
        self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), 1)
        for b in self.blocks:
            h = b["ln1"](x)
            x = x + b["attn"](h, h, h, attn_mask=mask, need_weights=False)[0]
            x = x + b["mlp"](b["ln2"](x))
        return self.head(self.norm(x))


def _write_shards(tokens, data_dir: Path):
    import numpy as np

    arr = np.array(tokens, dtype=np.uint16)
    n_val = max(len(arr) // 10, 256)
    arr[n_val:].tofile(data_dir / "train.bin")
    arr[:n_val].tofile(data_dir / "val.bin")


@torch.no_grad()
def _sample(model, stoi, itos, block_size, prompt="A language model", n=200):
    model.eval()
    ids = [stoi.get(c, 0) for c in prompt]
    for _ in range(n):
        x = torch.tensor([ids[-block_size:]])
        logits = model(x)[0, -1]
        probs = F.softmax(logits, dim=-1)
        ids.append(int(torch.multinomial(probs, 1)))
    return "".join(itos[i] for i in ids)


def main() -> int:
    torch.manual_seed(1337)
    chars = sorted(set(CORPUS))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    tokens = [stoi[c] for c in CORPUS]

    work = Path(tempfile.mkdtemp(prefix="lmkit-quickstart-"))
    _write_shards(tokens, work)

    mcfg = _MCfg(vocab_size=len(chars))
    tcfg = _TCfg(out_dir=str(work / "out"), data_dir=str(work))
    model = _DemoModel(mcfg)  # CPU, fp32

    print(f"lmkit quickstart: {len(chars)}-char vocab, {len(tokens)} tokens, "
          f"{sum(p.numel() for p in model.parameters())/1e3:.0f}k params, CPU\n")
    pretrain.run(model, tcfg, mcfg, experiment="quickstart")

    print("\n--- sample after training ---")
    print(_sample(model, stoi, itos, mcfg.block_size))
    print("\ndone. This is the real lmkit pretrain loop; scale the model + corpus for real runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
