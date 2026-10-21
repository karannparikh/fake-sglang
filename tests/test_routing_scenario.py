import io
import json

import pytest
from prometheus_client.parser import text_string_to_metric_families

from scripts import routing_scenario as scenario


def test_corpus_is_exact_reproducible_and_prefix_shared():
    first = scenario.corpus(64, 16, 24)
    assert first == scenario.corpus(64, 16, 24)
    assert scenario.fingerprint(first) == scenario.fingerprint(scenario.corpus(64, 16, 24))
    assert len({scenario.fingerprint(q["tokens"]) for q in first}) == 24
    assert {len(q["tokens"]) for q in first} == {1040}
    assert first[0]["tokens"][:1024] == first[1]["tokens"][:1024]
    assert first[0]["tokens"][:1024] != first[2]["tokens"][:1024]
    assert scenario.fingerprint(first) != scenario.fingerprint(scenario.corpus(64, 8, 24))
    assert max(token for q in first for token in q["tokens"]) < 151936
    with pytest.raises(ValueError, match="vocabulary"):
        scenario.token_ids(100, 0, 64, 16)


def test_event_barrier_uses_success_not_component_as_worker_identity(monkeypatch):
    text = """
# TYPE dynamo_component_kv_cache_events_applied counter
dynamo_component_kv_cache_events_applied{dynamo_component="prefill",event_type="stored",status="ok"} 3
dynamo_component_kv_cache_events_applied{dynamo_component="backend",event_type="stored",status="ok"} 9
dynamo_component_kv_cache_events_applied{dynamo_component="prefill",event_type="stored",status="parent_block_not_found"} 2
dynamo_component_kv_cache_events_applied{dynamo_component="prefill",event_type="removed",status="ok"} 1
"""
    samples = [s for f in text_string_to_metric_families(text) for s in f.samples]
    monkeypatch.setattr(scenario, "metric_samples", lambda port: samples)
    assert scenario.prefill_events("stored") == 12
    assert scenario.prefill_events("removed") == 1


def test_worker_is_not_ready_until_capacity_metadata_arrives(monkeypatch):
    text = 'sglang:simulated_prefix_cache_query_tokens_total{worker_id="abc"} 0\n'

    def samples(port):
        return [s for f in text_string_to_metric_families(text) for s in f.samples]

    monkeypatch.setattr(scenario, "metric_samples", samples)
    assert scenario.worker_snapshot(9201) is None
    text += "dynamo_component_total_blocks 0\n"
    assert scenario.worker_snapshot(9201) is None
    text = text.replace("total_blocks 0", "total_blocks 96")
    assert scenario.worker_snapshot(9201) is None
    text += "".join(scenario.PREFIX + field + " " + ("1536" if field == "capacity_tokens" else "0") + "\n"
                    for field in scenario.CACHE_FIELDS)
    snapshot = scenario.worker_snapshot(9201)
    assert snapshot["capacity_blocks"] == 96
    assert snapshot["worker_id"] == 0xABC
    assert snapshot["capacity_tokens"] == 1536
    assert snapshot["resident_tokens"] == 0
    assert snapshot["observed_at_ms"] > 0


def test_long_workload_preserves_bounded_deterministic_request_population():
    args = scenario.parse_args(["--duration-seconds", "600"])
    assert args.duration_seconds == args.requests == 600
    assert args.pairs == 2
    rows = scenario.corpus(args.prefix_blocks, args.block_size, args.requests)
    assert len(rows) == 600
    assert len({scenario.fingerprint(row["tokens"]) for row in rows}) == 600
    assert scenario.fingerprint(rows) == scenario.fingerprint(scenario.corpus(64, 16, 600))
    assert scenario.parse_args([]).requests == 24
    for arguments in (["--duration-seconds", "0"], ["--duration-seconds", "601"],
                      ["--duration-seconds", "600", "--requests", "24"]):
        with pytest.raises(SystemExit):
            scenario.parse_args(arguments)


def test_ten_minute_replay_pacing_uses_the_same_600_requests_and_full_window(monkeypatch):
    clock = [1000.0]
    starts = []
    monkeypatch.setattr(scenario.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(scenario.time, "time_ns", lambda: round(clock[0] * 1e9))
    monkeypatch.setattr(scenario.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def measured(query, owner, stage, expect_full_prefix):
        starts.append((query["id"], clock[0]))
        clock[0] += 0.4

    args = scenario.parse_args(["--duration-seconds", "600"])
    start, finish = scenario.replay_queries(args, scenario.corpus(64, 16, 600), measured, "test")
    assert finish - start == 600_000
    assert starts == [(i, 1000 + i) for i in range(600)]


def stream(*events, done=True):
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    return io.BytesIO(body + (b"data: [DONE]\n\n" if done else b""))


def test_measured_http_requests_are_unpinned_and_exact(monkeypatch):
    requests = []

    def respond(request, timeout):
        requests.append(json.loads(request.data))
        return stream(
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": "length"}]},
            {"usage": {"prompt_tokens": 3, "completion_tokens": 8},
             "nvext": {"worker_id": {"prefill_worker_id": 42, "decode_worker_id": 43}}},
        )

    monkeypatch.setattr(scenario.urllib.request, "urlopen", respond)
    result = scenario.send([10, 11, 12])
    assert result["prefill_worker_id"] == 42
    assert result["decode_worker_id"] == 43
    assert result["output_sha256"] == scenario.fingerprint("hello")
    assert 0 < result["started_at_ms"] <= result["finished_at_ms"]
    assert requests[0]["nvext"]["token_data"] == [10, 11, 12]
    assert "prefill_worker_id" not in requests[0]["nvext"]
    assert "annotations" not in requests[0]["nvext"]
    scenario.send([10, 11, 12], pinned=42)
    assert requests[1]["nvext"]["prefill_worker_id"] == 42


def test_http_errors_and_inexact_preview_are_not_accepted(monkeypatch):
    monkeypatch.setattr(
        scenario.urllib.request, "urlopen", lambda *a, **k: stream({"error": "failed"}),
    )
    with pytest.raises(RuntimeError, match="Inference failed"):
        scenario.send([1])
    monkeypatch.setattr(
        scenario.urllib.request, "urlopen",
        lambda *a, **k: stream({"nvext": {"token_ids": [2], "worker_id": {
            "prefill_worker_id": 7, "decode_worker_id": 7,
        }}}),
    )
    with pytest.raises(AssertionError, match="exact token-ID"):
        scenario.send([1], probe=True)


def test_summary_weights_tokens_and_reports_regressions():
    def row(tokens, hits, kv, latency, owner):
        return {"stage": "replay", "input_tokens": tokens, "hit_tokens": hits,
                "kv_seconds": kv, "ttft_ms": latency, "latency_ms": latency + 10,
                "selected_seeded_owner": owner}

    cases = [
        {"metadata_enabled": True, "samples": [row(10, 10, 0, 30, True), row(30, 0, 3, 50, False)]},
        {"metadata_enabled": False, "samples": [row(10, 0, 1, 10, False), row(30, 0, 3, 30, True)]},
    ]
    result = scenario.summarize(cases)
    assert result["metadata_on"]["observed_hit_ratio"] == 0.25
    assert result["metadata_on"]["seeded_owner_fraction"] == 0.5
    assert result["change"]["hit_ratio_percentage_points"] == 25
    assert result["change"]["kv_generation_seconds_saved"] == 1
    assert result["change"]["ttft_p50_percent"] == 100
    assert result["change"]["mean_latency_percent"] == pytest.approx(100 * (50 / 30 - 1))
    with pytest.raises(ValueError, match="No replay samples"):
        scenario.summarize([])
