import copy
import csv
import io
import time
from types import SimpleNamespace

import pytest

from deploy.local.routing_dashboard import build_dashboard, iso_time
from scripts.routing_load import LoadMeasurements, load_query, run_load
from scripts.routing_scenario import fingerprint, parse_args, token_ids


def test_seeded_hot_long_tail_profile_varies_prefixes_and_rotates_without_changing_capacity():
    args = parse_args(["--duration-seconds", "600", "--rate", "100"])
    assert args.kv_blocks == 96
    assert args.kv_generation_ms == 250
    first = [load_query(i, args) for i in range(10000)]
    second = [load_query(i, args) for i in range(12000, 22000)]
    assert .78 < sum(family < 4 for family, _, _ in first) / len(first) < .82
    assert .78 < sum(4 <= family < 8 for family, _, _ in second) / len(second) < .82
    assert {row[2] for row in first} == {16, 64}
    assert {row[0] for row in first} == set(range(32))
    assert fingerprint(first) == fingerprint([load_query(i, args) for i in range(10000)])
    for index in (0, 11999, 12000, 59999):
        family, question, blocks = load_query(index, args)
        tokens = token_ids(family, question, blocks, 16)
        assert len(tokens) in (272, 1040)
        assert max(tokens) < 151936


def test_load_accounting_separates_rejection_failure_success_and_drain():
    measurements = LoadMeasurements(10000)
    measurements.window(10000)["offered"] += 2
    measurements.window(10000)["client_rejected"] += 1
    measurements.window(11000)["offered"] += 1
    measurements.completed({
        "started_at_ms": 10010, "finished_at_ms": 10500, "latency_ms": 490, "ttft_ms": 50,
    }, 10000)
    measurements.completed({
        "started_at_ms": 11010, "finished_at_ms": 12500, "error_type": "TimeoutError", "error": "timeout",
    }, 11000)
    result = measurements.finish(13000)
    assert result["totals"] == {"offered": 3, "sent": 2, "succeeded": 1, "failed": 1, "client_rejected": 1}
    assert [w["inflight"] for w in result["windows"]] == [0, 1, 0]
    assert result["windows"][1]["e2e_ms"] is None
    assert result["e2e_ms"] == {"p50": 490, "p95": 490, "p99": 490}
    assert result["errors"] == {"TimeoutError": 1}


def test_open_loop_offers_100_requests_without_hiding_concurrency_rejections():
    args = parse_args(["--duration-seconds", "1", "--rate", "100", "--concurrency", "1"])
    stages = []

    def send(tokens):
        start = time.time_ns() // 1_000_000
        time.sleep(.04)
        return {
            "started_at_ms": start, "finished_at_ms": time.time_ns() // 1_000_000,
            "latency_ms": 40, "ttft_ms": 10, "prefill_worker_id": 100, "decode_worker_id": 102,
            "output_sha256": fingerprint(tokens),
        }

    result = run_load(args, [100, 101, 102], stages.append, send, token_ids)
    assert result["totals"]["offered"] == 100
    assert result["totals"]["sent"] == result["totals"]["succeeded"]
    assert result["totals"]["failed"] == 0
    assert 0 < result["totals"]["client_rejected"] < 100
    assert result["finished_at_ms"] - result["started_at_ms"] >= 1000
    assert stages[0] == "load-baseline" and stages[-1] == "load-final"


def load_report():
    start = 1791562699000
    cases = []
    summary = {}
    for index, enabled in enumerate((True, False)):
        beginning = start + index * 10000
        workers = [{"worker_id": i} for i in (100, 101, 102)]
        states = []
        for stage, offset, amount in (("load-baseline", 1000, 0), ("load-final", 2000, 10),
                                      ("before-pressure", 2100, 10), ("eviction", 2200, 20)):
            states.append({"stage": stage, "workers": [{
                "worker_id": 100 + rank, "observed_at_ms": beginning + offset,
                "capacity_tokens": 1536, "resident_tokens": 1040 if rank < 2 else 0,
                "allocated_tokens": 1040 if rank < 2 else 0, "evictable_tokens": 1040 if rank < 2 else 0,
                "evicted_tokens_total": amount * 16, "cleared_tokens_total": 0,
                "generate_requests": amount, "input_tokens": amount * 1040 if rank < 2 else 0,
                "hit_tokens": amount * 1024 if enabled and rank < 2 else 0,
                "kv_seconds": amount * .004,
            } for rank in range(3)]})
        counts = {"offered": 100, "sent": 100, "succeeded": 98, "failed": 2, "client_rejected": 0}
        load = {
            "started_at_ms": beginning + 1000, "finished_at_ms": beginning + 2000,
            "offered_duration_seconds": 1, "totals": counts,
            "e2e_ms": {"p50": 20, "p95": 30, "p99": 40},
            "ttft_ms": {"p50": 10, "p95": 15, "p99": 20},
            "windows": [{
                **counts, "started_at_ms": beginning + 1000, "finished_at_ms": beginning + 2000,
                "inflight": 0, "e2e_ms": {"p50": 20, "p95": 30, "p99": 40},
                "ttft_ms": {"p50": 10, "p95": 15, "p99": 20},
                "schedule_lag_ms": {"p50": 0, "p95": 1, "p99": 2},
            }],
        }
        cases.append({
            "started_at_ms": beginning, "finished_at_ms": beginning + 3000,
            "name": str(index), "metadata_enabled": enabled, "workers": workers,
            "samples": [], "cache_samples": states, "load": load,
        })
        summary["metadata_on" if enabled else "metadata_off"] = {
            **counts, "completed_rps_including_drain": 98, "observed_hit_ratio": .98 if enabled else .5,
            "evicted_tokens": 160, "e2e_ms": copy.deepcopy(load["e2e_ms"]),
            "ttft_ms": copy.deepcopy(load["ttft_ms"]),
        }
    return {
        "schema_version": 3, "run_id": "b" * 32, "started_at_ms": start, "finished_at_ms": start + 20000,
        "configuration": {"rate": 100, "duration_seconds": 1}, "cases": cases, "summary": summary,
    }


def test_concurrent_dashboard_uses_aggregates_accessible_colors_and_shared_worker_panels():
    report = load_report()
    dashboard = build_dashboard(report)
    assert len(dashboard["panels"]) == 17
    panels = {p["id"]: p for p in dashboard["panels"]}
    assert len(panels) == len(dashboard["panels"])
    assert len(panels[2]["targets"]) == 8
    assert len(panels[3]["targets"]) == 4
    assert len(panels[4]["targets"]) == 2
    assert len(panels[6]["targets"]) == 4  # No fabricated decode hit-ratio series.
    assert len(panels[13]["targets"]) == 6
    assert ",100.0\n" in panels[2]["targets"][0]["csvContent"]
    for i, panel in enumerate(dashboard["panels"]):
        a = panel["gridPos"]
        for other in dashboard["panels"][i+1:]:
            b = other["gridPos"]
            assert (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"]
                    or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])
        for override in panel.get("fieldConfig", {}).get("overrides", []):
            fields = {p["id"]: p["value"] for p in override["properties"]}
            assert fields["color"]["fixedColor"] in ("#56B4E9", "#E69F00")
            assert "custom.lineStyle" in fields
    invalid = copy.deepcopy(report)
    invalid["cases"][0]["load"]["windows"][0]["sent"] = 1
    with pytest.raises(ValueError, match="totals"):
        build_dashboard(invalid)


def test_whole_case_comparison_uses_saved_percentiles_not_window_percentiles():
    report = load_report()
    for index, case in enumerate(report["cases"]):
        case["load"]["ttft_ms"]["p95"] = 400 + index
        case["load"]["e2e_ms"]["p95"] = 460 + index
    report["summary"]["metadata_on"]["ttft_ms"]["p95"] = 410
    report["summary"]["metadata_on"]["e2e_ms"]["p95"] = 469
    panels = {p["id"]: p for p in build_dashboard(report)["panels"]}
    text = panels[16]["options"]["content"]
    assert "| Case 1: metadata ON | 98 | 400.0 ms | 460.0 ms | 60.0 ms |" in text
    assert "| Case 2: metadata OFF | 98 | 401.0 ms | 461.0 ms | 60.0 ms |" in text
    assert "| Metadata ON (pooled) | 98 | 410.0 ms | 469.0 ms | 59.0 ms |" in text
    assert "not p95(E2E - TTFT)" in text
    assert "not averages of case/window percentiles" in text
    assert "do not change when zooming" in text
    references = panels[17]
    assert len(references["targets"]) == 4
    assert "Fixed reference lines" in references["description"]
    for index, case in enumerate(report["cases"]):
        for rank, field in enumerate(("ttft_ms", "e2e_ms")):
            target = references["targets"][index * 2 + rank]
            rows = list(csv.reader(io.StringIO(target["csvContent"])))
            assert "(whole case)" in rows[0][1]
            assert [row[0] for row in rows[1:]] == [
                iso_time(case["load"][key]) for key in ("started_at_ms", "finished_at_ms")
            ]
            assert [float(row[1]) for row in rows[1:]] == [case["load"][field]["p95"]] * 2
    for identifier, field in ((7, "e2e_ms"), (8, "ttft_ms")):
        assert "not the whole case" in panels[identifier]["description"]
        for index, case in enumerate(report["cases"]):
            for rank, percentile in enumerate(("p50", "p95", "p99")):
                rows = list(csv.reader(io.StringIO(panels[identifier]["targets"][index * 3 + rank]["csvContent"])))
                assert float(rows[1][1]) == case["load"]["windows"][0][field][percentile]


def test_whole_case_comparison_does_not_invent_latency_for_no_successes():
    report = load_report()
    load = report["cases"][0]["load"]
    for item in (load["totals"], load["windows"][0], report["summary"]["metadata_on"]):
        item["succeeded"] = 0
        item["failed"] = item["sent"]
    for item in (load, load["windows"][0], report["summary"]["metadata_on"]):
        item["e2e_ms"] = item["ttft_ms"] = None
    panels = {p["id"]: p for p in build_dashboard(report)["panels"]}
    assert "| Case 1: metadata ON | 0 | No successes | No successes | No successes |" in panels[16]["options"]["content"]
    assert len(panels[17]["targets"]) == 2
    assert all("metadata OFF" in target["csvContent"] for target in panels[17]["targets"])


@pytest.mark.parametrize("scope", ["case", "pooled"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 5])
def test_whole_case_comparison_rejects_invalid_latency(scope, value):
    report = load_report()
    result = report["cases"][0]["load"] if scope == "case" else report["summary"]["metadata_on"]
    result["e2e_ms"]["p95"] = value
    with pytest.raises(ValueError, match="p95"):
        build_dashboard(report)
