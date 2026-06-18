"""Autoregressive generation: no-cache, full-prefix decode.

Re-forwarding the whole prefix each step is correct for any architecture by
construction — it sidesteps the cached-decode coherence trap that can bite
dynamic-compute models. At small scale and short eval generations it's plenty
fast. Greedy (temperature 0) by default for reproducible eval; nucleus sampling
is available.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F


def render_chatml(messages: list[dict]) -> str:
    """ChatML prompt ending with an open assistant turn for the model to complete."""
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + "<|im_start|>assistant\n"


def sample_next(logits_row: torch.Tensor, temperature: float = 0.0, top_p: float = 0.95) -> int:
    """Next token id from a (vocab,) logits row. temperature 0 -> greedy argmax;
    otherwise temperature-scaled nucleus (top-p) sampling."""
    if temperature <= 0.0:
        return int(torch.argmax(logits_row).item())
    probs = F.softmax(logits_row / temperature, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    keep = torch.cumsum(sp, dim=-1) <= top_p
    keep[0] = True
    sp, si = sp[keep], si[keep]
    return int(si[torch.multinomial(sp / sp.sum(), 1)].item())


@torch.no_grad()
def generate(model, tok, messages: list[dict], *, block_size: int | None = None,
             max_new: int = 128, temperature: float = 0.0, top_p: float = 0.95,
             device: torch.device | None = None) -> str:
    """Generate the assistant completion for a ChatML conversation. Stops on
    <|im_end|>/<|endoftext|>. Returns the decoded assistant text only."""
    device = device or next(model.parameters()).device
    if block_size is None:
        block_size = getattr(getattr(model, "cfg", None), "block_size", 2048)
    im_end = tok.token_to_id("<|im_end|>")
    eot = tok.token_to_id("<|endoftext|>")

    ids = tok.encode(render_chatml(messages)).ids
    base = len(ids)
    model.eval()
    for _ in range(max_new):
        window = ids[-block_size:]
        logits = model(torch.tensor([window], device=device))
        nxt = sample_next(logits[0, -1].float(), temperature, top_p)
        if nxt in (im_end, eot):
            break
        ids.append(nxt)
    return tok.decode(ids[base:]).strip()


def load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def generate_completions(model, tok, prompts: list[dict], *, max_new: int = 200,
                         label: str = "", step: int = -1, device=None) -> list[dict]:
    """Greedy completion of each prompt's `messages`; returns rows for the judge."""
    out = []
    for p in prompts:
        completion = generate(model, tok, p["messages"], max_new=max_new,
                              temperature=0.0, device=device)
        out.append({"id": p["id"], "label": label, "step": step,
                    "messages": p["messages"], "completion": completion})
    return out
