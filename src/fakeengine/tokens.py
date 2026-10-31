"""Deterministic synthetic token generation.

The same prompt always yields the same output IDs, in aggregated and
disaggregated mode alike -- which makes "did disaggregation change the answer?"
a checkable question instead of a vibe. Nothing is sampled from a model; the
text Dynamo detokenizes is meaningless by construction.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Iterator, List, Sequence

# Low IDs are where tokenizers put BOS/EOS/pad/unk. Staying above them keeps
# synthetic output from accidentally terminating the frontend's stream.
_SPECIAL_TOKEN_CEILING = 256


def seed_for(prompt_token_ids: Sequence[int]) -> int:
    digest = hashlib.blake2b(
        b",".join(str(t).encode() for t in prompt_token_ids),
        digest_size=8,
        person=b"fake-sgl",
    ).digest()
    return int.from_bytes(digest, "little")


def token_stream(prompt_token_ids: Sequence[int], vocab_size: int) -> Iterator[int]:
    rng = random.Random(seed_for(prompt_token_ids))
    low = min(_SPECIAL_TOKEN_CEILING, max(vocab_size - 1, 1))
    high = max(vocab_size - 1, low + 1)
    while True:
        yield rng.randint(low, high)


def take(prompt_token_ids: Sequence[int], vocab_size: int, count: int) -> List[int]:
    stream = token_stream(prompt_token_ids, vocab_size)
    return [next(stream) for _ in range(count)]


def natural_length(
    prompt_token_ids: Sequence[int], median: float, sigma: float
) -> int:
    """How many tokens the answer to this prompt has before it would emit EOS.

    Lognormal, which fits chat output lengths: most answers are short and a
    few run long. Seeded by the prompt, so aggregated and disaggregated
    serving stop at the same token.
    """
    rng = random.Random(seed_for(prompt_token_ids) ^ 0x6F75_746C_656E)
    return max(1, round(rng.lognormvariate(math.log(median), sigma)))


def handoff_token(request_id: str, vocab_size: int) -> int:
    """The single token a disaggregated prefill hands to decode.

    Mirrors SGLang's ``_generate_fake_prefill_handoff_output_id``
    (``disaggregation/decode.py``): hash the request id so the token is
    request-diverse but identical on every rank.
    """
    digest = hashlib.blake2b(
        request_id.encode(), digest_size=8, person=b"fake-pd "
    ).digest()
    return int.from_bytes(digest, "little") % max(vocab_size, 1)
