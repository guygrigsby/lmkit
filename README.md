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

## Install

```
pip install -e .            # core
pip install -e ".[hub,track,dev]"   # + hub push, Aim tracking, tests
```

Requires Python 3.10+. CUDA is used when available; everything runs on CPU for
tests.

## License

MIT.
