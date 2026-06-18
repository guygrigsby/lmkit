"""Position-swapped pairwise A/B judge.

For each prompt, asks a judge model twice with the two responses position-swapped,
and credits a win only when the same model wins both orders (else tie) — this
cancels the judge's position bias. The judge backend is pluggable: an
OpenAI-compatible HTTP endpoint (`judge_call`) or an in-process transformers model
(`make_hf_judge`, no server needed). Aggregation is pure and tested.
"""
from __future__ import annotations

import json
import urllib.request
from collections import Counter

JUDGE_SYSTEM = (
    "You are a strict, impartial judge of small AI assistant responses. Compare two "
    "responses to the same user prompt and decide which is the better assistant reply, "
    "judging coherence, how well it follows the instruction, factual correctness, and "
    "helpfulness. Ignore length unless one is empty or rambling. Think briefly (no more "
    "than three short sentences), then end your reply with a single final line: "
    "'VERDICT: A', 'VERDICT: B', or 'VERDICT: TIE'."
)


def build_judge_messages(user_prompt: str, resp_a: str, resp_b: str) -> list[dict]:
    content = (
        f"User prompt:\n{user_prompt}\n\n"
        f"Response A:\n{resp_a or '(empty)'}\n\n"
        f"Response B:\n{resp_b or '(empty)'}\n\n"
        "Which response is the better assistant reply? Think briefly, then end with "
        "'VERDICT: A', 'VERDICT: B', or 'VERDICT: TIE'."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": content}]


def extract_reply(resp_json: dict) -> str:
    """Pull judge text from an OpenAI-compatible response. Reasoning models put
    output in 'reasoning'/'reasoning_content' with null 'content'; fall back."""
    msg = resp_json["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or "")


def parse_verdict(text: str) -> str:
    """Map a judge reply to 'A'/'B'/'TIE'. Prefers the LAST 'VERDICT: X' line (so a
    reasoning model's final answer wins over stray letters in its thinking); falls
    back to a leading-token read, then a scan; defaults to TIE when ambiguous."""
    t = (text or "").strip().upper()
    if not t:
        return "TIE"
    idx = t.rfind("VERDICT:")
    if idx != -1:
        after = t[idx + len("VERDICT:"):].lstrip(" *_-.:\n\t")
        if after.startswith("TIE"):
            return "TIE"
        if after.startswith("A") and not after.startswith("AN"):
            return "A"
        if after.startswith("B"):
            return "B"
    head = t.lstrip(".:_-*# \n\t")
    if head.startswith("TIE"):
        return "TIE"
    if head.startswith("A") and not head.startswith("AN"):
        return "A"
    if head.startswith("B"):
        return "B"
    last = "TIE"
    for tok in t.replace("\n", " ").split():
        tok = tok.strip(".:,_-*#()[]")
        if tok in ("A", "B", "TIE"):
            last = tok
    return last


def resolve(verdict: str, slot_a_label: str, slot_b_label: str) -> str:
    return slot_a_label if verdict == "A" else slot_b_label if verdict == "B" else "tie"


def combine(winner_order1: str, winner_order2: str) -> str:
    """Credit a model only if it won BOTH position orders; else tie."""
    if winner_order1 == winner_order2 and winner_order1 != "tie":
        return winner_order1
    return "tie"


def judge_call(messages: list[dict], endpoint: str, model: str, timeout: float = 300.0,
               max_tokens: int = 768) -> str:
    """HTTP backend (OpenAI-compatible endpoint)."""
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.0,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return extract_reply(json.loads(r.read()))


def make_hf_judge(model_id: str, *, device: str = "cpu", dtype: str = "bfloat16",
                  max_new: int = 256):
    """In-process transformers judge — no server, no port. Loads the model once and
    returns a judge_fn(messages, *_) -> text."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()

    def judge_fn(messages, endpoint=None, model_name=None) -> str:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)

    return judge_fn


def load_completions(path: str) -> dict:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def _user_prompt(messages: list[dict]) -> str:
    users = [m["content"] for m in messages if m["role"] == "user"]
    return users[-1] if users else ""


def run(file_a: str, file_b: str, label_a: str, label_b: str, *,
        endpoint: str = "", model: str = "", judge_fn=judge_call) -> dict:
    """Pairwise-judge two completion files; returns a win/loss/tie report."""
    a_rows, b_rows = load_completions(file_a), load_completions(file_b)
    ids = [i for i in a_rows if i in b_rows]
    tally = Counter()
    per_prompt = []
    for pid in ids:
        ra, rb = a_rows[pid], b_rows[pid]
        user = _user_prompt(ra["messages"])
        w1 = resolve(parse_verdict(judge_fn(build_judge_messages(user, ra["completion"], rb["completion"]),
                                            endpoint, model)), label_a, label_b)
        w2 = resolve(parse_verdict(judge_fn(build_judge_messages(user, rb["completion"], ra["completion"]),
                                            endpoint, model)), label_b, label_a)
        winner = combine(w1, w2)
        tally[winner] += 1
        per_prompt.append({"id": pid, "winner": winner, "order1": w1, "order2": w2})
    wins_a, wins_b, ties = tally[label_a], tally[label_b], tally["tie"]
    decisive = wins_a + wins_b
    return {
        "n_prompts": len(ids), "label_a": label_a, "label_b": label_b,
        "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
        "win_rate_a": (wins_a / decisive) if decisive else None,
        "per_prompt": per_prompt,
    }
