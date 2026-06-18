"""Config for the example Llama model + its training.

A project supplies dataclasses like these; lmkit reads the fields documented in
`lmkit.protocol`. Defaults describe a ~100M-parameter model and a WSD schedule
(warmup -> stable -> optional cosine anneal tail). Tune to your hardware and data.
"""
from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    hidden_size: int = 768
    n_layer: int = 12
    n_head: int = 12               # head_dim = 64
    n_kv_heads: int = 4            # GQA 3:1
    intermediate_size: int = 2048  # SwiGLU: ~(2/3)*4*hidden, multiple of 64
    block_size: int = 2048

    rope_base: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True    # saves params + optimizer state at small scale

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.n_head == 0
        return self.hidden_size // self.n_head

    def param_count(self) -> int:
        h = self.hidden_size
        kvh = self.n_kv_heads * self.head_dim
        attn = h * h + 2 * h * kvh + h * h    # Q + K + V + O
        mlp = 3 * h * self.intermediate_size  # SwiGLU gate + up + down
        emb = self.vocab_size * h             # tied output head
        return emb + self.n_layer * (attn + mlp)

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    """Pretraining (WSD). `decay_frac=0` is a pure stable trunk; run an anneal
    branch separately (override `decay_frac` and `max_steps`) when ready to ship."""
    out_dir: str = "checkpoints"
    data_dir: str = "data"
    tokenizer_path: str = "data/tokenizer.json"

    batch_size: int = 2
    grad_accum: int = 32           # effective 64 sequences/step

    max_steps: int = 200_000       # an upper bound, not a target (resumable trunk)
    eval_interval: int = 2_000
    eval_iters: int = 100
    log_interval: int = 20
    save_interval: int = 200       # rolling latest.pt
    snapshot_interval: int = 25_000
    keep_last_snapshots: int = 3

    lr: float = 4e-4
    min_lr: float = 4e-5           # only used when an anneal branch runs
    warmup_steps: int = 1_000
    decay_frac: float = 0.0        # 0 = no decay tail (stable trunk)

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    dtype: str = "bfloat16"
    compile: bool = False
    seed: int = 1337
    device: str = "cuda"

    def to_dict(self):
        return asdict(self)


@dataclass
class SFTConfig:
    """Instruct SFT off a pretrained base. `max_steps`/`warmup_steps` are computed
    at runtime from the data (epochs * steps_per_epoch); the 0 defaults are
    placeholders."""
    init_from: str = ""            # base checkpoint to fine-tune (weights only)
    out_dir: str = "checkpoints-sft"
    data_file: str = "data/instruct/instruct.jsonl.gz"
    tokenizer_path: str = "data/tokenizer.json"

    batch_size: int = 2
    grad_accum: int = 16
    epochs: int = 3

    lr: float = 2e-5
    min_lr: float = 0.0
    warmup_frac: float = 0.03
    decay_frac: float = 1.0        # cosine the whole post-warmup tail
    max_steps: int = 0             # computed
    warmup_steps: int = 0          # computed

    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 10
    save_interval: int = 500

    dtype: str = "bfloat16"
    compile: bool = False
    seed: int = 1337
    device: str = "cuda"

    def to_dict(self):
        return asdict(self)
