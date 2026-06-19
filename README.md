# lmkit

A small toolkit for training language models from scratch on a single machine.

`lmkit` is the reusable machinery — you bring the architecture and the data
recipe. It stays out of the business of *what* you train: model, hyperparameters,
data sources, and any push targets are yours to supply via a config object and
environment variables.

## What's in it

- **pretrain** — a WSD (warmup-stable-decay) next-token loop with a separable
  cosine anneal tail, checkpointing, and clean resume.
- **sft** — supervised fine-tuning with ChatML rendering and assistant-only loss
  masking; windows with no trainable target are dropped at pack time.
- **tokenizer / shard** — train a ByteLevel BPE and pack a corpus into token shards.
- **eval** — greedy/no-cache generation and a position-swapped pairwise judge.
- **push** — upload checkpoints + code to a model hub (repo id supplied by you).
- **observability** — optional experiment tracking; logs loss/lr/throughput to an
  Aim server when `AIM_REPO` is set, and no-ops otherwise.

## Design

`lmkit` operates against a small protocol (`lmkit.protocol`): any `torch.nn.Module`
that maps token ids to logits, plus a config dataclass with the fields the loops
read. It never imports a concrete architecture, so the same harness trains a
vanilla transformer or a custom one without change.

## Quickstart (60 seconds, CPU, no data)

```
pip install -e .
lmkit quickstart
```

Trains a tiny model on a small bundled corpus and shows the loss falling, then
samples a few lines. No GPU, no downloads — it runs the real pretrain loop, just
tiny.

## Setup

Python **3.10–3.12** (3.12 recommended — a few dependencies don't yet ship 3.13/
3.14 wheels). [`uv`](https://docs.astral.sh/uv/) is the easiest way to pin the
version and install; plain `venv` + `pip` works too.

### Linux (NVIDIA GPU)

```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,track]"
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # True with an NVIDIA driver
```

`pip install torch` pulls the CUDA build on Linux x86_64, so GPU training works
out of the box. **AMD / ROCm:** install the ROCm torch wheel instead —
`uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` (match
your ROCm version) — the rest is unchanged.

### macOS (Apple Silicon)

```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,track]"
```

You get the CPU/MPS torch build (there's no CUDA on Mac). The quickstart, tests,
and small experiments run fine; real from-scratch pretraining wants a Linux NVIDIA
box (local or rented). Mac is for development.

### Plain pip (either OS)

```
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,track]"
```

Extras: `hub` (HuggingFace push), `track` (Aim tracking), `dev` (pytest).

## Tests

```
pytest                                          # core: tokenization, masking, packing
PYTHONPATH=. pytest examples/llama/test_smoke.py  # the example model, CPU
```

## CLI (fleet ops)

This repo also ships a small **Go** command-line tool, `lmkit`, for running and
watching training workers across machines from your workstation. It is a separate
artifact from the Python library above: the **library** trains models (installed
with `pip`, runs on your GPU boxes); the **CLI** is an ops front-end (a static
binary, runs on your laptop, reaches the boxes over ssh).

```
lmkit status                 # roster of every worker across the fleet
lmkit logs <project/run> -f  # follow a worker's journal
lmkit run  <project/run>     # deploy a persistent systemd unit and start it
lmkit start|stop --all       # control every worker
```

It reads a fleet config (`~/.config/lmkit/fleet.toml`, the boxes) and per-project
manifests (`lmkit.toml`, how each worker launches). See
`docs/specs/2026-06-19-lmkit-fleet-ops-cli-design.md`.

**Install the CLI** — download a binary from the
[releases](https://github.com/guygrigsby/lmkit/releases) (each tagged
`cli/vX.Y.Z`; the assets are the CLI, not the Python library) or build it:

```
go install github.com/guygrigsby/lmkit/cmd/lmkit@latest   # or: make install-cli
```

## License

MIT.
