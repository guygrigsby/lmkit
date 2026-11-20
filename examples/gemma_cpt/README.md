# gemma_cpt

`probe.py`: throughput and VRAM probe for LoRA continued pretraining of a Gemma 4 12B base,
data parallel through `lmkit.distributed`. It trains on random tokens, so it measures speed
and memory, not learning. `--check` compares logits against stock attention on real text.

Measured on one H100 SXM (bf16 adapters, per-layer `torch.compile`, gradient checkpointing):

| config | tok/s |
|---|---|
| `--attn sdpa --seq 1024 --bs 4 --rank 64` | 4,270 |
| `--attn fast --seq 1024 --bs 4 --rank 64` | 5,420 |
| `--attn sdpa --seq 4096 --bs 1 --rank 128` | 2,020 |
| `--attn fast --seq 4096 --bs 1 --rank 128` | 3,870 |

`--attn fast` sends head_dim-256 layers to fused flash (plain causal when the 1024 window
covers the sequence, `flash-attn` with a sliding window beyond it). The head_dim-512 global
layers stay on `sdpa`.
