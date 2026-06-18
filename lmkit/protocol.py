"""The contracts lmkit trains against.

The framework never imports a concrete architecture; it works through these so
the same loops fit any model. A project constructs its own model + config and
hands them to the loops.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LMModel(Protocol):
    """A language model lmkit can train: a torch.nn.Module whose call maps a
    (batch, seq) LongTensor of token ids to (batch, seq, vocab) logits. The
    standard nn.Module surface (parameters/to/train/eval/state_dict) is assumed.
    """

    def __call__(self, idx: torch.Tensor) -> torch.Tensor: ...


# The config fields each loop reads. A project's config dataclass must provide at
# least these (extra fields are fine). Kept as documentation, not enforced, so
# projects own their dataclass and lmkit stays decoupled from it.
PRETRAIN_CONFIG_FIELDS = (
    "lr", "min_lr", "warmup_steps", "max_steps", "decay_frac",
    "beta1", "beta2", "weight_decay", "grad_clip",
    "batch_size", "grad_accum", "block_size",
    "device", "dtype", "seed", "compile",
    "eval_interval", "eval_iters", "log_interval", "save_interval",
)
SFT_CONFIG_FIELDS = PRETRAIN_CONFIG_FIELDS + ("epochs", "warmup_frac")


# A model config supplies at least the shape the loops need to build batches and
# compute the loss.
MODEL_CONFIG_FIELDS = ("block_size", "vocab_size")
