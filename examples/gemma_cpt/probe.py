"""Throughput and VRAM probe: LoRA CPT of a Gemma 4 12B base, data parallel via lmkit.distributed.
torchrun --standalone --nproc-per-node N examples/gemma_cpt/probe.py [--model PATH] [--seq 4096] [--steps 200]
Random tokens: measures speed and memory only, not learning."""
import argparse, contextlib, json, time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from lmkit import distributed

p = argparse.ArgumentParser()
p.add_argument("--model", default="~/models/base/gemma-4-12B")
p.add_argument("--seq", type=int, default=4096)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--bs", type=int, default=1)
p.add_argument("--rank", type=int, default=128)
p.add_argument("--adapter", choices=["fp32", "bf16"], default="fp32", help="bf16 skips autocast casts and halves optimizer memory")
p.add_argument("--compile", action="store_true", help="torch.compile each decoder layer")
p.add_argument("--no-ckpt", action="store_true", help="skip gradient checkpointing (needs the VRAM)")
p.add_argument("--attn", default="sdpa", help="HF attn_implementation: sdpa, flex_attention, flash_attention_2")
p.add_argument("--check", action="store_true", help="compare logits against stock sdpa, then exit")
p.add_argument("--profile", action="store_true", help="phase timings and a one-step op profile, then exit")
a = p.parse_args()


_flash_attn_func = None


def _register_fast_attention():
    """Layers with head_dim <= 256 whose window covers the whole sequence are plain
    causal attention, so they take torch's fused flash kernel instead of the masked
    memory-efficient one (about 6x faster per layer on H100). Everything else (the
    head_dim-512 global layers, or sequences longer than the window) is unchanged."""
    global _flash_attn_func
    try:
        from flash_attn import flash_attn_func as _flash_attn_func
    except ImportError:
        _flash_attn_func = None
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    def fast_attention(module, q, k, v, attention_mask, dropout=0.0, scaling=None,
                       is_causal=None, sliding_window=None, **kw):
        if q.shape[-1] <= 256 and (sliding_window is None or q.shape[2] <= sliding_window):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scaling,
                                               enable_gqa=q.shape[1] != k.shape[1])
            return o.transpose(1, 2).contiguous(), None
        if q.shape[-1] <= 256 and _flash_attn_func is not None:
            o = _flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), softmax_scale=scaling,
                                 causal=True, window_size=(sliding_window - 1, 0))
            return o, None
        return sdpa_attention_forward(module, q, k, v, attention_mask, dropout=dropout, scaling=scaling,
                                      is_causal=is_causal, sliding_window=sliding_window, **kw)

    AttentionInterface.register("fast", fast_attention)
    AttentionMaskInterface.register("fast", sdpa_mask)


if a.attn == "fast":
    _register_fast_attention()

d = distributed.init()
dev = torch.device("cuda", d.local_rank)
model = AutoModelForCausalLM.from_pretrained(a.model.replace("~", __import__("os").path.expanduser("~")),
                                             dtype=torch.bfloat16, attn_implementation=a.attn).to(dev)
model.train()
if not a.no_ckpt:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, task_type="CAUSAL_LM",
                 target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
model = get_peft_model(model, cfg, autocast_adapter_dtype=(a.adapter == "fp32"))
params = [q for q in model.parameters() if q.requires_grad]
if a.adapter == "fp32":
    for q in params:
        q.data = q.data.float()
else:
    assert all(q.dtype == torch.bfloat16 for q in params), {q.dtype for q in params}
d.broadcast_params(model)
opt = torch.optim.AdamW(params, lr=1e-4)
vocab = model.config.get_text_config().vocab_size
g = torch.Generator(device="cpu").manual_seed(d.rank)
if d.is_main:
    print(f"trainable {sum(q.numel() for q in params)/1e6:.0f}M of {sum(q.numel() for q in model.parameters())/1e9:.1f}B", flush=True)
inner = model.get_base_model()
if a.compile:
    for i, layer in enumerate(inner.model.language_model.layers):
        inner.model.language_model.layers[i] = torch.compile(layer)
cap = inner.config.get_text_config().final_logit_softcapping


def _ce(h, y):
    logits = inner.lm_head(h).float()
    if cap:
        logits = cap * torch.tanh(logits / cap)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum")


def chunked_loss(x, chunk=512):
    """Next-token CE from hidden states in checkpointed chunks: full-vocab logits
    (262k) never exist for the whole sequence."""
    h = inner.model(input_ids=x).last_hidden_state[:, :-1]
    h = h.reshape(-1, h.size(-1))
    y = x[:, 1:].reshape(-1)
    tot = sum(checkpoint(_ce, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
              for i in range(0, h.size(0), chunk))
    return tot / y.numel()


if a.check:
    from transformers import AutoTokenizer
    mp = a.model.replace("~", __import__("os").path.expanduser("~"))
    text = open(mp + "/README.md").read()
    ids = AutoTokenizer.from_pretrained(mp)(text, return_tensors="pt").input_ids[:, :a.seq].to(dev)
    ids = ids.repeat(1, 1)
    lp, loss = {}, {}
    with torch.no_grad():
        for name in ("eager", "sdpa", a.attn):
            inner.set_attn_implementation(name)
            logits = inner(input_ids=ids).logits.float()
            lp[name] = torch.log_softmax(logits, -1)
            loss[name] = F.nll_loss(lp[name][:, :-1].reshape(-1, lp[name].size(-1)), ids[:, 1:].reshape(-1)).item()
    for name in loss:
        print("CHECK loss", name, round(loss[name], 4), flush=True)
    for name in ("sdpa", a.attn):
        kl = (lp["eager"].exp() * (lp["eager"] - lp[name])).sum(-1).mean().item()
        agree = (lp["eager"].argmax(-1) == lp[name].argmax(-1)).float().mean().item()
        print("CHECK", name, "vs eager: mean KL", round(kl, 5), "argmax agree", round(agree, 4), flush=True)
    kl = (lp["sdpa"].exp() * (lp["sdpa"] - lp[a.attn])).sum(-1).mean().item()
    print("CHECK", a.attn, "vs sdpa: mean KL", round(kl, 5), flush=True)
    raise SystemExit

phase = {}


def timed(name, fn):
    torch.cuda.synchronize(); t = time.time()
    out = fn()
    torch.cuda.synchronize(); phase[name] = phase.get(name, 0) + time.time() - t
    return out


def train_step():
    x = torch.randint(0, vocab, (a.bs, a.seq), generator=g).to(dev)

    def fwd():
        with (torch.autocast("cuda", dtype=torch.bfloat16) if a.adapter == "fp32" else contextlib.nullcontext()):
            return chunked_loss(x)

    loss = timed("forward+loss", fwd)
    timed("backward", loss.backward)
    timed("sync_grads", lambda: d.sync_grads(params))
    timed("clip", lambda: torch.nn.utils.clip_grad_norm_(params, 1.0))
    timed("optimizer", lambda: (opt.step(), opt.zero_grad(set_to_none=True)))
    return loss


if a.profile:
    from torch.profiler import profile, ProfilerActivity
    for _ in range(3):
        train_step()
    phase.clear()
    for _ in range(3):
        train_step()
    if d.is_main:
        print("PHASES (s/step):", {k: round(v / 3, 2) for k, v in phase.items()}, flush=True)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        train_step()
    if d.is_main:
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18, max_name_column_width=60), flush=True)
    raise SystemExit

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for step in range(1, a.steps + 1):
    loss = train_step()
    if step % 10 == 0:
        torch.cuda.synchronize(); dt = time.time() - t0; t0 = time.time()
        if d.is_main:
            print(json.dumps({"step": step, "loss": round(loss.item(), 3),
                              "tok_s_total": int(10 * a.bs * a.seq * d.world / dt),
                              "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}), flush=True)
