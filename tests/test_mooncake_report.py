import json

import pytest

from scripts.mooncake_report import (
    CpuCounterRegression,
    cache_report,
    cpu_report,
    integrate,
    markdown_report,
    memory_report,
    REQUIRED_CASES,
)


def test_cpu_regression_marks_only_affected_role_unavailable():
    series = {
        ("other-containers", "otel", "main", "start"): {0: 20, 5: 0, 10: 10},
        ("prefill", "worker", "main", "start"): {0: 1, 10: 6},
    }
    result = cpu_report(series, 0, 10, {"prefill": 1})
    assert result["prefill"]["mean_cores"] == 0.5
    assert result["prefill"]["coverage_fraction"] == 1
    assert result["other-containers"]["mean_cores"] is None
    with pytest.raises(CpuCounterRegression):
        integrate(series[("other-containers", "otel", "main", "start")], 0, 10)


def test_memory_report_sums_contemporaneous_processes(tmp_path):
    samples = []
    for batch, values in ((1, (10, 20)), (2, (15, 10))):
        for index, mib in enumerate(values):
            samples.append(
                {
                    "collected_at_ns": batch,
                    "pods": [
                        {
                            "podRef": {
                                "name": f"fake-sglang-prefill-{index}",
                                "uid": str(index),
                                "namespace": "dynamo-sim",
                            },
                            "containers": [
                                {
                                    "name": "main",
                                    "memory": {
                                        "time": "2026-01-01T00:00:00Z",
                                        "workingSetBytes": mib * 2**20,
                                    },
                                }
                            ],
                        }
                    ],
                }
            )
    path = tmp_path / "cpu.jsonl"
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    result = memory_report(path, 0, 2e9, {"prefill": 2})
    assert result["prefill"] == {
        "peak_single_container_working_set_mib": 20,
        "peak_observed_total_working_set_mib": 30,
        "complete_sample_batches": 2,
    }


def metric_snapshot(pod_name, hit, query, *, stale=False):
    result = []
    names = [pod_name, "retired-prefill"] if stale else [pod_name]
    for name in names:
        for rank in ("0", "1"):
            for field, value in (("hit", hit), ("query", query)):
                result.append(
                    {
                        "metric": {
                            "__name__": (
                                f"sglang:simulated_prefix_cache_{field}_tokens_total"
                            ),
                            "pod": name,
                            "dynamo_component_type": "prefill",
                            "worker_id": name,
                            "dp_rank": rank,
                        },
                        "value": [0, str(999 if name != pod_name else value)],
                    }
                )
    return {"status": "success", "data": {"result": result}}


def test_cache_report_uses_per_run_deltas_and_current_pod_identity(tmp_path):
    pod = {
        "metadata": {
            "uid": "fresh",
            "name": "prefill-live",
            "labels": {"nvidia.com/dynamo-component-type": "prefill"},
        },
        "status": {
            "containerStatuses": [{"name": "main", "restartCount": 0}]
        },
    }
    (tmp_path / "initial-pods.json").write_text(json.dumps({"items": [pod]}))
    (tmp_path / "cache-metrics-before.json").write_text(
        json.dumps(metric_snapshot("prefill-live", 10, 20, stale=True))
    )
    (tmp_path / "cache-metrics-after.json").write_text(
        json.dumps(metric_snapshot("prefill-live", 30, 100, stale=True))
    )
    result = cache_report(tmp_path, expected_ranks=2, fleet_stable=True)
    assert result["hit_tokens_delta"] == 40
    assert result["query_tokens_delta"] == 160
    assert result["token_hit_fraction"] == 0.25
    assert "before/after counter deltas" in result["scope"]


def test_cache_counter_regression_is_rejected(tmp_path):
    pod = {
        "metadata": {
            "uid": "fresh",
            "name": "prefill-live",
            "labels": {"nvidia.com/dynamo-component-type": "prefill"},
        }
    }
    (tmp_path / "initial-pods.json").write_text(json.dumps({"items": [pod]}))
    (tmp_path / "cache-metrics-before.json").write_text(
        json.dumps(metric_snapshot("prefill-live", 30, 100))
    )
    (tmp_path / "cache-metrics-after.json").write_text(
        json.dumps(metric_snapshot("prefill-live", 10, 20))
    )
    with pytest.raises(ValueError, match="regressed"):
        cache_report(tmp_path, expected_ranks=2, fleet_stable=True)


def test_markdown_uses_po2c_report_label():
    matrix = {
        "offered_rps": 20,
        "h200_profile_from_application_defaults": {
            "kv_blocks_per_rank": 15_509,
            "tokens_per_page": 256,
            "token_slots_per_rank": 3_970_304,
            "context_length": 1_048_576,
            "kv_cache_dtype": "fp8_e4m3",
            "max_running_requests_per_rank": 32,
        },
        "topology": {
            "frontend_processes": 1,
            "planner_processes": 1,
            "prefill_processes": 4,
            "decode_processes": 4,
            "ranks_per_worker_process": 10,
        },
    }
    result = {
        "workload": "conversation",
        "routing": "PO2C",
        "successful": 100,
        "error_rate_pct": 0,
        "dispatch_rps": 20,
        "prefill_cache": {"token_hit_fraction": 0.5},
        "latency_ms_successful": {"p99": 100},
        "ttft_ms_successful": {"p99": 50},
        "fleet": {"stable": True},
        "cpu_dispatch_coverage_complete": True,
    }
    report = markdown_report(matrix, [result])
    assert "| conversation | PO2C |" in report
    assert "20 RPS" in report
    assert "15,509 blocks x 256 tokens/page" in report
    assert len(REQUIRED_CASES) == 6
    assert {item["routing_label"] for item in REQUIRED_CASES} == {
        "KV-aware",
        "load-aware",
        "PO2C",
    }
