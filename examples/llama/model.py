"""A small vanilla Llama-style decoder — the lmkit example architecture.

Standard Llama: RMSNorm + SwiGLU + RoPE + grouped-query attention, no biases,
pre-norm, tied embeddings. `forward(input_ids)` returns logits of shape
(B, T, vocab_size); the loss is computed by the training loop, so this satisfies
`lmkit.protocol.LMModel` and can be trained by lmkit unchanged.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 internal compute for stability, cast back at the end.
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight


def build_rope_cache(
    seq_len: int, head_dim: int, base: float, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (seq_len, head_dim/2), cast to the activation dtype."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotate-halves RoPE to x of shape (B, H, T, D). cos/sin: (T, D/2)."""
    d_half = x.size(-1) // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


class Attention(nn.Module):
    """GQA via F.scaled_dot_product_attention (uses FlashAttention when available)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_head // self.n_kv
        assert self.n_head % self.n_kv == 0, "n_head must be divisible by n_kv_heads"

        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, h, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, Hk, D = self.n_head, self.n_kv, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)   # (B, H,  T, D)
        k = self.k_proj(x).view(B, T, Hk, D).transpose(1, 2)  # (B, Hk, T, D)
        v = self.v_proj(x).view(B, T, Hk, D).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_rep > 1:  # GQA: expand K/V to Q's head count
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(y)


class MLP(nn.Module):
    """SwiGLU."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Llama(nn.Module):
    """Vanilla Llama-style decoder. forward(input_ids) -> logits (B, T, vocab_size)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if not cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self._rope_len = 0
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self._init_weights()

    def _init_weights(self):
        """N(0, 0.02); residual projections scaled by 1/sqrt(2*n_layer) for depth
        stability (GPT-2 / Llama convention)."""
        residual_scale = 1.0 / math.sqrt(2 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if "weight" not in name:
                continue
            if "norm" in name:
                nn.init.ones_(p)
            elif "embed_tokens" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 * residual_scale)
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def _get_rope(self, seq_len: int, device, dtype):
        if seq_len > self._rope_len:
            self._cos, self._sin = build_rope_cache(
                max(seq_len, self.cfg.block_size),
                self.cfg.head_dim, self.cfg.rope_base, device, dtype,
            )
            self._rope_len = self._cos.size(0)
        return self._cos[:seq_len], self._sin[:seq_len]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.embed_tokens(input_ids)
        cos, sin = self._get_rope(T, h.device, h.dtype)
        for block in self.blocks:
            h = block(h, cos, sin)
        h = self.norm(h)
        if self.cfg.tie_embeddings:
            logits = h @ self.embed_tokens.weight.T
        else:
            logits = self.lm_head(h)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
