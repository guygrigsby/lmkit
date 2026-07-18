# lmkit

A small toolkit for training language models from scratch on a single machine.

`lmkit` is the reusable machinery. You bring the architecture and the data
recipe. It stays out of the business of *what* you train: model, hyperparameters,
data sources, and any push targets are yours to supply via a config object and
environment variables.

## What's in it

- **pretrain**: a WSD (warmup-stable-decay) next-token loop with a separable
  cosine anneal tail, checkpointing, and clean resume.
- **sft**: supervised fine-tuning with ChatML rendering and assistant-only loss
  masking; windows with no trainable target are dropped at pack time.
- **tokenizer / shard**: train a ByteLevel BPE and pack a corpus into token shards.
- **eval**: greedy/no-cache generation and a position-swapped pairwise judge.
- **push**: upload checkpoints + code to a model hub (repo id supplied by you).
- **observability**: optional experiment tracking; logs loss/lr/throughput to
  [Aim](https://github.com/aimhubio/aim) when `AIM_REPO` is set (e.g.
  `aim://host:53800`) and/or [MLflow](https://mlflow.org) when
  `MLFLOW_TRACKING_URI` is set (e.g. `http://bee:5000`). Either, both, or
  neither; it no-ops when neither is set.

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
samples a few lines. No GPU, no downloads. Runs the real pretrain loop, just
tiny.

## Setup

Python **3.10-3.12** (3.12 recommended; a few dependencies don't yet ship 3.13/
3.14 wheels). [`uv`](https://docs.astral.sh/uv/) is the easiest way to pin the
version and install; plain `venv` + `pip` works too.

### Linux (NVIDIA GPU)

```
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,track]"
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # True with an NVIDIA driver
```

`pip install torch` pulls the CUDA build on Linux x86_64, so GPU training works
out of the box. **AMD / ROCm:** install the ROCm torch wheel instead:
`uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` (match
your ROCm version); the rest is unchanged.

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

Extras: `hub` (HuggingFace push), `track` (Aim + MLflow tracking), `dev` (pytest).

## Using the library

The Python `lmkit` binary only runs `quickstart`. Everything else is a library
API you call from your own training script. There is no `lmkit pretrain`
subcommand on purpose: you own the model, the configs, and the driver, and
lmkit supplies the loops. A real run is a short script that walks the pipeline
below. The `examples/llama` model is a complete, runnable reference for the
`model` + `config` you bring.

**1. Train a tokenizer** (once per corpus family):

```python
from lmkit.tokenizer import train_tokenizer

train_tokenizer(["corpus/*.jsonl"], "tokenizer.json", vocab_size=32_000)
```

**2. Pack the corpus into token shards:**

```python
from tokenizers import Tokenizer
from lmkit.shard import shard_corpus

tok = Tokenizer.from_file("tokenizer.json")
counts = shard_corpus(["corpus/*.jsonl"], "data/", tok)   # writes train/val .bin shards
```

**3. Pretrain.** Bring a model (any `torch.nn.Module` mapping token ids →
logits) and two config dataclasses: a model config (`block_size`,
`vocab_size`, …) and a training config (the fields in
`lmkit.protocol.PRETRAIN_CONFIG_FIELDS`: `lr`, `min_lr`, `warmup_steps`,
`max_steps`, `batch_size`, `out_dir`, `data_dir`, …). lmkit never imports a
concrete architecture; it reads these fields and nothing else.

```python
from lmkit import pretrain
from examples.llama.model import Llama                  # your architecture
from examples.llama.config import ModelConfig, TrainConfig

mcfg = ModelConfig(vocab_size=32_000, block_size=1024)
tcfg = TrainConfig(out_dir="runs/base", data_dir="data/", max_steps=50_000)
model = Llama(mcfg).to("cuda")

pretrain.run(model, tcfg, mcfg, experiment="base")
```

The loop checkpoints to `out_dir` (`latest.pt`, `best.pt`), streams metrics to
`out_dir/metrics.jsonl`, and resumes cleanly: re-running the same script picks
up from `latest.pt` at the right step. `SIGTERM`/`SIGINT` checkpoint and exit,
so it is safe under systemd or a scheduler.

**4. SFT** (optional): same shape, plus a `tokenizer_path` and `data_file` on
the config; ChatML rendering and assistant-only loss masking are handled for
you.

```python
from lmkit import sft
sft.run(model, cfg, mcfg, experiment="chat")
```

**5. Eval and push:**

```python
from lmkit.eval.gen import generate
from lmkit.push import push

out = generate(model, tok, [{"role": "user", "content": "hello"}])
push("you/my-model", {"runs/base/best.pt": "model.pt", "tokenizer.json": "tokenizer.json"})
```

## Experiment tracking

Tracking is automatic and opt-in through the environment: the pretrain and SFT
loops call the tracker for you, so you never instrument anything. Install the
extra, set one or both endpoints, and the metrics you already see in
`metrics.jsonl` (loss, lr, throughput) also stream to your tracker. Set
neither and training runs exactly the same, logging only to the JSONL file.

```
uv pip install -e ".[track]"        # installs both aim and mlflow
```

**Aim**: set `AIM_REPO` to a running Aim server (or a local repo path):

```
export AIM_REPO=aim://your-host:53800     # or: aim://127.0.0.1:53800
# start a server with:  aim server --repo /path/to/aim-repo
```

**MLflow**: set `MLFLOW_TRACKING_URI` to your tracking server:

```
export MLFLOW_TRACKING_URI=http://your-host:5000
# start a server with:  mlflow server --host 0.0.0.0 --port 5000
```

If you only log to a remote MLflow server (no local UI), the client-only
`mlflow-skinny` package is enough and lighter than full `mlflow`.

Both can be set at once; metrics then go to both backends. A few things the
loops do for you:

- **Run naming and hparams follow fleet conventions.** Runs are named
  `lmkit-<out_dir basename>` (override with `run_name=`), and hyperparameters
  are logged under canonical cross-framework keys (`n_layers`, `n_heads`,
  `hidden_size`, …) so params line up in the tracker across different training
  frameworks.
- **A resumed run stays one run.** The MLflow run id is persisted to
  `out_dir/mlflow_run_id`; on resume the loop reattaches to it (flipping it back
  to `RUNNING`) instead of fragmenting one training across many runs. If the run
  is gone from the server, a fresh one is created.
- **Missing client libs are loud, not silent.** If an endpoint env var is set
  but the client package isn't installed, lmkit warns on stderr rather than
  quietly dropping your run.
- **Terminal status is recorded**: `KILLED` on SIGTERM, `FAILED` on a NaN loss,
  otherwise `FINISHED`.

Nothing about training depends on a tracker being installed or reachable: an
unreachable server or version skew disables tracking with a warning and the run
continues.

## Built for agents

lmkit is designed to be driven by a coding agent as much as by a human. The
training loops are non-interactive (config objects in, no prompts), emit
machine-readable progress to `metrics.jsonl`, resume idempotently, and check
their own work; `lmkit quickstart` is a real end-to-end run of the pretrain
loop that an agent can use as a smoke test. Repo-level agent instructions live in
[`AGENTS.md`](AGENTS.md). Point an agent at this README and it has everything it
needs to stand up a run.

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

**Install the CLI**: download a binary from the
[releases](https://github.com/guygrigsby/lmkit/releases) (each tagged
`cli/vX.Y.Z`; the assets are the CLI, not the Python library) or build it:

```
go install github.com/guygrigsby/lmkit/cmd/lmkit@latest   # or: make install-cli
```

## License

MIT.
