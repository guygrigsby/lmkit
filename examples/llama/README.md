# Example: a vanilla Llama

A ~100M-parameter Llama-style decoder (RMSNorm + SwiGLU + RoPE + GQA, tied
embeddings) — a well-known architecture, included so you have a concrete,
runnable model to train with lmkit.

It's deliberately thin: just `model.py` (the architecture) and `config.py`
(model + train + SFT dataclasses). That's all a project needs to bring — lmkit
supplies the loops, tokenizer, sharding, eval, push, and tracking.

## Why it works with lmkit unchanged

`Llama.forward(input_ids)` returns `(B, T, vocab_size)` logits and nothing else —
the loss lives in the training loop. That's exactly `lmkit.protocol.LMModel`, so
lmkit trains it without knowing it's a Llama. Swap in any module with the same
contract and the harness is identical.

## Run the smoke test

```
pip install -e ".[dev]"        # from the lmkit repo root
cd examples/llama
PYTHONPATH=../.. python -m pytest test_smoke.py -q
```

It builds a tiny model, packs a ChatML example with `lmkit.sft_data`, and does one
masked forward+backward — proving the model is trainable by lmkit on CPU.

## Sizing

`ModelConfig.param_count()` reports the parameter count; the defaults are ~100M.
Scale `hidden_size` / `n_layer` / `vocab_size` to your budget.
