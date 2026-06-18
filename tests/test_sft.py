"""SFT loss + schedule unit tests. The full loop is exercised end-to-end against
real data on a GPU box; here we lock the pure pieces."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from lmkit.sft import compute_schedule, masked_loss


def test_masked_loss_matches_manual_over_valid_positions():
    torch.manual_seed(0)
    V = 10
    logits = torch.randn(1, 4, V)
    labels = torch.tensor([[-100, 3, -100, 5]])
    got = masked_loss(logits, labels, V)
    ref = F.cross_entropy(logits.reshape(-1, V)[[1, 3]], labels.reshape(-1)[[1, 3]])
    assert torch.allclose(got, ref)


def test_masked_loss_all_ignored_is_finite_zero():
    # zero trainable targets must NOT NaN (that aborted the whole run); a real,
    # grad-connected 0 instead.
    V = 10
    logits = torch.randn(1, 3, V, requires_grad=True)
    labels = torch.full((1, 3), -100)
    loss = masked_loss(logits, labels, V)
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_compute_schedule_math():
    cfg = SimpleNamespace(batch_size=4, grad_accum=2, epochs=3, warmup_frac=0.1)
    spe, max_steps, warmup = compute_schedule(100, cfg)  # eff batch 8 -> 12 spe -> 36 steps
    assert spe == 12
    assert max_steps == 36
    assert warmup == max(1, int(0.1 * 36))
