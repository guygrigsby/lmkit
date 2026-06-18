"""SFT data: ChatML tokenization with assistant-only loss masking, and packing.

For instruct SFT, conversations are rendered into ChatML and the loss is trained
on *only* the assistant's content and its closing ``<|im_end|>`` — never the
user/system turns or the role scaffolding. Masked target positions become -100 so
``F.cross_entropy(..., ignore_index=-100)`` skips them.

Specials follow the common convention ``<|endoftext|>``=0, ``<|pad|>``=1,
``<|im_start|>``=2, ``<|im_end|>``=3, but ids are passed in so nothing is hardcoded.

The functions take an ``encode: str -> list[int]`` callable so the masking logic
is testable without a real BPE. Wrap a tokenizers.Tokenizer as
``lambda s: tok.encode(s).ids``.
"""
from __future__ import annotations

from typing import Callable, Iterable, Iterator

Encode = Callable[[str], list[int]]


def tokenize_chatml(
    messages: list[dict],
    encode: Encode,
    *,
    im_start_id: int = 2,
    im_end_id: int = 3,
) -> tuple[list[int], list[int]]:
    """Render messages to ChatML token ids + a parallel loss mask.

    Each turn becomes: ``<|im_start|>{role}\\n{content}<|im_end|>\\n``.
    ``mask[i] == 1`` iff token i is assistant content or the ``<|im_end|>`` that
    closes an assistant turn; everything else is 0.
    """
    ids: list[int] = []
    mask: list[int] = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        train = role == "assistant"

        ids.append(im_start_id)
        mask.append(0)

        header = encode(f"{role}\n")
        ids.extend(header)
        mask.extend([0] * len(header))

        body = encode(content)
        ids.extend(body)
        mask.extend([1 if train else 0] * len(body))

        ids.append(im_end_id)
        mask.append(1 if train else 0)

        nl = encode("\n")
        ids.extend(nl)
        mask.extend([0] * len(nl))

    return ids, mask


def pack_sequences(
    examples: Iterable[tuple[list[int], list[int]]],
    block_size: int,
    eot_id: int = 0,
) -> Iterator[tuple[list[int], list[int]]]:
    """Concatenate (ids, mask) conversations into block_size training windows.

    Conversations are separated by ``<|endoftext|>`` (masked). Each yielded window
    is (x, labels) of length block_size, where labels is the next-token target with
    -100 wherever the target is not a trainable position. Windows step by
    block_size over a (block_size+1)-token frame so position i in x predicts i+1.

    Windows with no trainable target are dropped: a long prompt can fill a whole
    window, leaving all labels -100, and cross_entropy(ignore_index=-100) over such
    a window averages over zero elements -> NaN, which aborts training. Nothing to
    learn there anyway.
    """
    toks: list[int] = []
    masks: list[int] = []
    for ids, mask in examples:
        toks.extend(ids)
        toks.append(eot_id)
        masks.extend(mask)
        masks.append(0)

    frame = block_size + 1
    for start in range(0, len(toks) - frame + 1, block_size):
        window = toks[start : start + frame]
        wmask = masks[start : start + frame]
        x = window[:block_size]
        labels = [
            window[i + 1] if wmask[i + 1] == 1 else -100
            for i in range(block_size)
        ]
        if any(l != -100 for l in labels):
            yield x, labels
