import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest

from scripts import mooncake_lab


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setitem(sys.modules, "mooncake_lab", mooncake_lab)
    spec = importlib.util.spec_from_file_location(
        "packed_probe", Path(__file__).resolve().parents[1] / "scripts/packed_probe.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", [None, "wrong-rank", "short-output", "missing-done"])
def test_probe_checks_actual_rank_and_complete_response(probe, monkeypatch, failure):
    requests = []

    def respond(request, timeout):
        requests.append(json.loads(request.data))
        event = {
            "nvext": {"worker_id": {
                "prefill_worker_id": 10, "prefill_dp_rank": 9 if failure == "wrong-rank" else 1,
                "decode_worker_id": 20, "decode_dp_rank": 2,
            }},
            "usage": {"prompt_tokens": 3, "completion_tokens": 3 if failure == "short-output" else 4},
            "choices": [{"finish_reason": "length"}],
        }
        body = b"data: " + json.dumps(event).encode() + b"\n\n"
        if failure != "missing-done":
            body += b"data: [DONE]\n\n"
        return io.BytesIO(body)

    monkeypatch.setattr(probe.urllib.request, "urlopen", respond)
    if failure:
        with pytest.raises((AssertionError, RuntimeError), match="Misrouted|Incomplete"):
            probe.send([1, 2, 3], ("prefill", 10, 1), ("decode", 20, 2))
    else:
        result = probe.send([1, 2, 3], ("prefill", 10, 1), ("decode", 20, 2))
        assert result["prefill_dp_rank"] == 1
    assert requests[0]["nvext"] == {
        "token_data": [1, 2, 3], "extra_fields": ["worker_id"],
        "prefill_worker_id": 10, "prefill_dp_rank": 1, "decode_worker_id": 20, "dp_rank": 2,
    }
