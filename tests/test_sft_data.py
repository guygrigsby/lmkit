"""Tests for SFT ChatML tokenization + packing.

Contract: only assistant *content* and its closing <|im_end|> are trainable;
everything else (system/user turns, role headers, <|im_start|>, structural
newlines, the <|endoftext|> separator) is masked out of the loss. Verified
independent of a real BPE via a transparent per-codepoint encoder.
"""
from lmkit.sft_data import tokenize_chatml, pack_sequences

IM_START, IM_END, EOT = 2, 3, 0


def enc(s: str) -> list[int]:
    # each char -> its codepoint; content chars are >= 32 so they never collide
    # with the special ids 0..3, letting tests assert on exact spans.
    return [ord(c) for c in s]


def _spans(content: str) -> list[int]:
    return enc(content)


def test_lengths_match():
    ids, mask = tokenize_chatml(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], enc)
    assert len(ids) == len(mask)
    assert all(m in (0, 1) for m in mask)


def test_only_assistant_content_and_imend_trained():
    ids, mask = tokenize_chatml(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], enc)
    assert sum(mask) == len(_spans("yo")) + 1
    trained = [t for t, m in zip(ids, mask) if m == 1]
    assert trained == _spans("yo") + [IM_END]


def test_user_and_system_never_trained():
    msgs = [
        {"role": "system", "content": "ss"},
        {"role": "user", "content": "uuu"},
        {"role": "assistant", "content": "ok"},
    ]
    ids, mask = tokenize_chatml(msgs, enc)
    assert sum(mask) == len(_spans("ok")) + 1
    for forbidden in (_spans("ss") + _spans("uuu")):
        for t, m in zip(ids, mask):
            if t == forbidden:
                assert m == 0


def test_im_start_always_masked():
    ids, mask = tokenize_chatml(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], enc)
    for t, m in zip(ids, mask):
        if t == IM_START:
            assert m == 0


def test_multiturn_trains_each_assistant():
    msgs = [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "bb"},
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "ddd"},
    ]
    _, mask = tokenize_chatml(msgs, enc)
    assert sum(mask) == (len(_spans("bb")) + 1) + (len(_spans("ddd")) + 1)


def test_no_assistant_turn_all_masked():
    _, mask = tokenize_chatml([{"role": "user", "content": "hi"}], enc)
    assert sum(mask) == 0


def test_chatml_structure_order():
    ids, _ = tokenize_chatml(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], enc)
    expected = (
        [IM_START] + enc("user\n") + enc("hi") + [IM_END] + enc("\n")
        + [IM_START] + enc("assistant\n") + enc("yo") + [IM_END] + enc("\n")
    )
    assert ids == expected


# ---- packing ----

def test_pack_windows_and_eot_separator():
    ex1 = tokenize_chatml([{"role": "user", "content": "a"},
                           {"role": "assistant", "content": "b"}], enc)
    ex2 = tokenize_chatml([{"role": "user", "content": "c"},
                           {"role": "assistant", "content": "d"}], enc)
    out = list(pack_sequences([ex1, ex2], block_size=8, eot_id=EOT))
    assert out
    for x, labels in out:
        assert len(x) == 8 and len(labels) == 8
        assert all(l == -100 or l >= 0 for l in labels)


def test_pack_never_yields_all_masked_window():
    # A long prompt fills the first window(s) entirely (all masked); a long
    # assistant reply occupies later full windows. Pre-fix the first window was
    # emitted all-masked (-> NaN at train time); the packer must drop it.
    ex = tokenize_chatml([{"role": "user", "content": "u" * 40},
                          {"role": "assistant", "content": "a" * 40}], enc)
    out = list(pack_sequences([ex], block_size=8, eot_id=EOT))
    assert out, "expected at least one packed window"
    for _, labels in out:
        assert any(l != -100 for l in labels), "packer emitted an all-masked window"


def test_pack_label_shift_and_mask_alignment():
    ex = tokenize_chatml([{"role": "user", "content": "a"},
                          {"role": "assistant", "content": "b"}], enc)
    ids, mask = ex
    block = len(ids)
    x, labels = list(pack_sequences([ex], block_size=block, eot_id=EOT))[0]
    full_ids = ids + [EOT]
    full_mask = mask + [0]
    for i in range(block):
        if full_mask[i + 1] == 1:
            assert labels[i] == full_ids[i + 1]
        else:
            assert labels[i] == -100
