"""Train a ByteLevel BPE tokenizer over a corpus.

Generic: reads text from ``.jsonl(.gz)`` (a configurable field, or ChatML
``messages``) or plain ``.txt`` files, optionally sampling up to ``max_per_file``
documents per file, and trains a ByteLevel BPE with the standard specials
(``<|endoftext|>``=0, ``<|pad|>``=1, ``<|im_start|>``=2, ``<|im_end|>``=3).
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Iterator

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")


def _docs(path: Path, text_key: str) -> Iterator[str]:
    name = path.name
    if name.endswith(".txt"):
        with _open(path) as f:
            yield f.read()
        return
    with _open(path) as f:  # jsonl(.gz)
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_key in obj:
                yield obj[text_key]
            elif "messages" in obj:
                yield "\n".join(m.get("content", "") for m in obj["messages"])


def _corpus(files, text_key: str, max_per_file: int | None, seed: int) -> Iterator[str]:
    rng = random.Random(seed)
    for fp in files:
        fp = Path(fp)
        docs = _docs(fp, text_key)
        if max_per_file is None:
            yield from docs
        else:
            # reservoir sample up to max_per_file to keep small sources represented
            sample: list[str] = []
            for i, d in enumerate(docs):
                if i < max_per_file:
                    sample.append(d)
                elif rng.random() < max_per_file / (i + 1):
                    sample[rng.randrange(max_per_file)] = d
            yield from sample


def train_tokenizer(files, out_path, *, vocab_size: int = 32_000,
                    max_per_file: int | None = None, text_key: str = "text",
                    seed: int = 1337) -> str:
    """Train a ByteLevel BPE over `files` and write `out_path` (tokenizer.json)."""
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token=None))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                         initial_alphabet=ByteLevel.alphabet())
    tok.train_from_iterator(_corpus(files, text_key, max_per_file, seed), trainer=trainer)
    out_path = str(out_path)
    tok.save(out_path)
    return out_path
