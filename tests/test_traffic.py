import argparse
import io
import json
import time

import pytest

from scripts.traffic import percentiles, positive_float, read_stream


def stream(*events, done=True):
    data = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    return io.BytesIO(data + (b"data: [DONE]\n\n" if done else b""))


def test_stream_counts_content_chunks_not_tokens():
    response = stream(
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "several tokens"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    )
    result = read_stream(response, time.perf_counter())
    assert result["content_chunks"] == 1
    assert result["completion_tokens"] == 3
    assert result["prompt_tokens"] == 10
    assert result["latency_ms"] >= result["ttft_ms"] >= 0


@pytest.mark.parametrize(
    "response,message",
    [
        (stream({"error": {"message": "overloaded"}}), "server stream error"),
        (stream(done=False), "before \\[DONE\\]"),
        (stream(), "without a finish reason"),
        (
            stream({"choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]}),
            "omitted requested token usage",
        ),
    ],
)
def test_stream_failures_are_explicit(response, message):
    with pytest.raises(ValueError, match=message):
        read_stream(response, time.perf_counter())


def test_percentiles_and_invalid_rates():
    assert percentiles([]) is None
    assert percentiles([2, 1, 3]) == {"p50": 2, "p95": 3}
    for value in ("0", "-1", "nan", "inf"):
        with pytest.raises(argparse.ArgumentTypeError):
            positive_float(value)
