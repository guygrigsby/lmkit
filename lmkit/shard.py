"""Tokenize a corpus into uint16 token shards for the pretrain loop.

Reads docs from ``.jsonl(.gz)``/``.txt``, encodes with a trained tokenizer, and
writes ``<|endoftext|>``-separated uint16 streams to ``train_NNNNN.bin`` /
``val_NNNNN.bin`` (read back by ``lmkit.training.TokenDataset``). A deterministic
1/``val_every`` fraction of documents goes to val (by content hash), so the split
mirrors the source distribution and is stable across runs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np


class ShardWriter:
    """Append-only writer that rolls to a new file when the current shard fills."""

    def __init__(self, out_dir: Path, prefix: str, shard_tokens: int):
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.cap = shard_tokens
        self.idx = 0
        self.count = 0
        self.total = 0
        self.fh = None
        self._open_next()

    def _open_next(self) -> None:
        if self.fh is not None:
            self.fh.close()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{self.prefix}_{self.idx:05d}.bin"
        self.fh = open(path, "wb")
        self.count = 0

    def add(self, tokens: np.ndarray) -> None:
        self.fh.write(tokens.astype(np.uint16).tobytes())
        self.count += len(tokens)
        self.total += len(tokens)
        if self.count >= self.cap:
            self.idx += 1
            self._open_next()

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def _docs(path: Path, text_key: str) -> Iterator[str]:
    if path.name.endswith(".txt"):
        with _open(path) as f:
            yield f.read()
        return
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_key in obj:
                yield obj[text_key]
            elif "messages" in obj:
                yield "\n".join(m.get("content", "") for m in obj["messages"])


def _is_val(doc: str, val_every: int) -> bool:
    if val_every <= 0:
        return False
    h = int(hashlib.sha1(doc.encode("utf-8")).hexdigest()[:8], 16)
    return h % val_every == 0


def shard_corpus(files, out_dir, tokenizer, *, shard_tokens: int = 500_000_000,
                 val_every: int = 100, text_key: str = "text",
                 batch_size: int = 8192) -> dict:
    """Tokenize `files` into train/val shards under `out_dir`. Returns token counts.

    Docs are encoded in batches via ``tokenizer.encode_batch`` so the Rust
    tokenizer parallelizes across all cores — single-doc ``encode`` in a Python
    loop pins to ~one core (orders of magnitude slower on a big corpus). Output is
    byte-identical to the per-doc path: the val/train split is content-hashed
    (order-independent) and encode order is preserved.
    """
    out_dir = Path(out_dir)
    eot = tokenizer.token_to_id("<|endoftext|>")
    train = ShardWriter(out_dir, "train", shard_tokens)
    val = ShardWriter(out_dir, "val", shard_tokens)

    def _flush(docs: list) -> None:
        for doc, enc in zip(docs, tokenizer.encode_batch(docs)):
            ids = enc.ids
            ids.append(eot)
            (val if _is_val(doc, val_every) else train).add(np.array(ids, dtype=np.uint16))

    batch: list = []
    for fp in files:
        for doc in _docs(Path(fp), text_key):
            batch.append(doc)
            if len(batch) >= batch_size:
                _flush(batch)
                batch = []
    if batch:
        _flush(batch)

    train.close()
    val.close()
    return {"train_tokens": train.total, "val_tokens": val.total}
