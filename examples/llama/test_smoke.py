"""Smoke test: the example Llama satisfies lmkit's training contract — forward to
logits, masked loss over lmkit-packed SFT windows, one backward with finite grads.
Runs on CPU. Proves the example model is trainable by lmkit unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))  # resolve `config` + `model`

import torch
import torch.nn.functional as F

from config import ModelConfig
from model import Llama
from lmkit.sft_data import tokenize_chatml, pack_sequences


def _small_cfg() -> ModelConfig:
    # tiny so it's fast on CPU
    return ModelConfig(vocab_size=256, hidden_size=128, n_layer=4, n_head=8,
                       n_kv_heads=2, intermediate_size=256, block_size=64)


def test_forward_shape():
    cfg = _small_cfg()
    m = Llama(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    with torch.no_grad():
        logits = m(x)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)


def test_masked_loss_and_backward():
    torch.manual_seed(0)
    cfg = _small_cfg()
    m = Llama(cfg).train()
    enc = lambda s: [min(ord(c), cfg.vocab_size - 1) for c in s]
    # long enough to fill at least one block_size window with trainable targets
    ex = tokenize_chatml([{"role": "user", "content": "u" * 30},
                          {"role": "assistant", "content": "a" * 100}], enc)
    windows = list(pack_sequences([ex], block_size=cfg.block_size))
    assert windows, "expected at least one packed window"
    x = torch.tensor([w[0] for w in windows])
    y = torch.tensor([w[1] for w in windows])
    logits = m(x)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1),
                           ignore_index=-100)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters())
