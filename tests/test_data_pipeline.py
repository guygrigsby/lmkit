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


def test_shard_batching_is_byte_identical(tmp_path):
    """Batched encode (encode_batch, multi-core) must produce exactly the same
    shards as the per-doc path — same counts and byte-identical .bin files."""
    corpus = tmp_path / "c.jsonl"
    with open(corpus, "w") as f:
        for i in range(500):
            f.write(json.dumps({"text": f"sample document {i} with assorted words to tokenize here"}) + "\n")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(train_tokenizer([corpus], tmp_path / "tok.json", vocab_size=300))

    a, b = tmp_path / "a", tmp_path / "b"
    c1 = shard_corpus([corpus], a, tok, shard_tokens=10**9, val_every=7, batch_size=1)
    c2 = shard_corpus([corpus], b, tok, shard_tokens=10**9, val_every=7, batch_size=256)
    assert c1 == c2 and c1["train_tokens"] > 0 and c1["val_tokens"] > 0
    for split in ("train", "val"):
        fa, fb = sorted(a.glob(f"{split}_*.bin")), sorted(b.glob(f"{split}_*.bin"))
        assert [p.name for p in fa] == [p.name for p in fb]
        for pa, pb in zip(fa, fb):
            assert pa.read_bytes() == pb.read_bytes()
