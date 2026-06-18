"""End-to-end data path: train a tiny BPE, shard a corpus, read it back."""
import json

from lmkit.shard import shard_corpus
from lmkit.tokenizer import train_tokenizer
from lmkit.training import TokenDataset


def test_tokenizer_shard_roundtrip(tmp_path):
    corpus = tmp_path / "c.jsonl"
    with open(corpus, "w") as f:
        for i in range(200):
            f.write(json.dumps({"text": f"the quick brown fox number {i} jumps over the lazy dog"}) + "\n")

    tok_path = train_tokenizer([corpus], tmp_path / "tok.json", vocab_size=300)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    assert tok.token_to_id("<|endoftext|>") == 0
    assert tok.token_to_id("<|im_end|>") == 3

    counts = shard_corpus([corpus], tmp_path / "shards", tok, shard_tokens=10**9, val_every=10)
    assert counts["train_tokens"] > 0 and counts["val_tokens"] > 0

    ds = TokenDataset(str(tmp_path / "shards"), block_size=8, split="train")
    x, y = next(iter(ds))
    assert x.shape[0] == 8 and y.shape[0] == 8
    assert int(y[0]) == int(x[1])  # next-token alignment
