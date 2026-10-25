import json

import pytest

from scripts.mooncake_report import (
    CpuCounterRegression, cache_report, cpu_report, integrate, memory_report,
)


def test_cpu_regression_marks_only_affected_role_unavailable():
    series = {
        ("other-containers", "tempo", "main", "start"): {0: 20, 5: 0, 10: 10},
        ("prefill", "worker", "main", "start"): {0: 1, 10: 6},
    }
    result = cpu_report(series, 0, 10, {"prefill": 1})
    assert result["prefill"]["mean_cores"] == .5
    assert result["prefill"]["coverage_fraction"] == 1
    invalid = result["other-containers"]
    assert invalid["cpu_seconds"] is None
    assert invalid["mean_cores"] is None
    assert invalid["cpu_percent_one_core"] is None
    assert invalid["invalid_cpu_series"][0]["container_identity"] == ["tempo", "main", "start"]
    with pytest.raises(CpuCounterRegression):
        integrate(series[("other-containers", "tempo", "main", "start")], 0, 10)


def test_cpu_regression_outside_report_window_does_not_invalidate_window():
    assert integrate({0: 20, 5: 0, 10: 10}, 5, 10) == (10, 5)


def test_partial_cpu_coverage_is_not_filled_with_synthetic_samples():
    result = cpu_report({
        ("prefill", "worker", "main", "start"): {0: 1, 5: 6},
    }, 0, 10, {"prefill": 2, "decode": 2})
    assert result["prefill"]["covered_container_seconds"] == 5
    assert result["prefill"]["coverage_fraction"] == .25
    assert "decode" not in result


def test_memory_report_sums_contemporaneous_processes_not_per_rank(tmp_path):
    samples = []
    for batch, values in ((1, (10, 20)), (2, (15, 10))):
        for index, mib in enumerate(values):
            samples.append({
                "collected_at_ns": batch,
                "pods": [{
                    "podRef": {"name": f"fake-sglang-prefill-{index}", "uid": str(index),
                               "namespace": "dynamo-sim"},
                    "containers": [{
                        "name": "main", "memory": {
                            "time": "2026-01-01T00:00:00Z", "workingSetBytes": mib * 2**20,
                        },
                    }],
                }],
            })
    path = tmp_path / "cpu.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in samples))
    result = memory_report(path, 0, 2e9, {"prefill": 2})
    assert result["prefill"] == {
        "peak_single_container_working_set_mib": 20,
        "peak_observed_total_working_set_mib": 30,
        "complete_sample_batches": 2,
    }
    assert memory_report(path, 0, 2e9, {"prefill": 3})["prefill"][
        "peak_observed_total_working_set_mib"
    ] is None


def test_cache_report_excludes_retired_pods_and_decode_transfer_hits(tmp_path):
    pod = {"metadata": {"uid": "live", "name": "prefill-live"}, "status": {
        "containerStatuses": [{"name": "main", "restartCount": 0}],
    }}
    for name in ("initial-pods.json", "final-pods.json"):
        (tmp_path / name).write_text(json.dumps({"items": [pod]}))
    samples = []
    for name, role, rank, hits, queries in (
        ("prefill-live", "prefill", "0", 256, 512),
        ("prefill-live", "prefill", "1", 0, 512),
        ("prefill-retired", "prefill", "0", 999, 999),
        ("decode-live", "decode", "0", 999, 999),
    ):
        for field, value in (("hit", hits), ("query", queries)):
            samples.append({"metric": {
                "__name__": f"sglang:simulated_prefix_cache_{field}_tokens_total",
                "pod": name, "dynamo_component_type": role, "worker_id": name, "dp_rank": rank,
            }, "value": [0, str(value)]})
    (tmp_path / "cache-metrics.json").write_text(json.dumps({"data": {"result": samples}}))
    result = cache_report(tmp_path)
    assert result["prefill_ranks_observed"] == 2
    assert result["token_hit_fraction"] == .25
    assert result["fleet_stable"]
    incomplete = cache_report(tmp_path, expected_ranks=3)
    assert incomplete["rank_coverage_complete"] is False
    assert incomplete["token_hit_fraction"] is None
    pod["status"]["containerStatuses"][0]["restartCount"] = 1
    (tmp_path / "final-pods.json").write_text(json.dumps({"items": [pod]}))
    result = cache_report(tmp_path)
    assert result["fleet_stable"] is False
    assert result["token_hit_fraction"] is None


def test_historical_run_without_cache_snapshot_remains_readable(tmp_path):
    assert cache_report(tmp_path) is None


@pytest.mark.parametrize("fields", [[], ["query"]])
def test_missing_rank_counters_are_not_reported_as_zero_hits(tmp_path, fields):
    pod = {"metadata": {"uid": "live", "name": "prefill-live"}, "status": {
        "containerStatuses": [{"name": "main", "restartCount": 0}],
    }}
    for name in ("initial-pods.json", "final-pods.json"):
        (tmp_path / name).write_text(json.dumps({"items": [pod]}))
    samples = [{"metric": {
        "__name__": f"sglang:simulated_prefix_cache_{field}_tokens_total",
        "pod": "prefill-live", "dynamo_component_type": "prefill",
        "worker_id": "abc", "dp_rank": "0",
    }, "value": [0, "512"]} for field in fields]
    (tmp_path / "cache-metrics.json").write_text(json.dumps({"data": {"result": samples}}))
    with pytest.raises(ValueError, match="Incomplete per-rank"):
        cache_report(tmp_path, expected_ranks=1)
