import copy
import csv
from datetime import datetime
import io

import pytest

from deploy.local.routing_dashboard import annotations_for, build_dashboard, e2e_percentiles, iso_time, qps_buckets
from scripts.routing_scenario import summarize


@pytest.fixture
def report():
    start = 1791559121000
    cases = []
    for index, enabled in enumerate((True, False, False, True)):
        case_start = start + index * 10000
        rows = [{
            "id": i, "stage": "replay" if i < 2 else "eviction",
            "started_at_ms": case_start + 1000 + i * 200,
            "finished_at_ms": case_start + 1100 + i * 200,
            "input_tokens": 1040, "hit_tokens": 1024 if enabled else 0,
            "kv_seconds": 0.004 if enabled else 0.260,
            "ttft_ms": 30 if enabled else 340, "latency_ms": 90 if enabled else 400,
            "selected_worker": i % 2, "selected_seeded_owner": enabled,
            "decode_worker_id": 102,
            "router_predicted_ratio": 1024 / 1040 if enabled else 0,
        } for i in range(3)]
        cases.append({
            "name": f"{index}-metadata-{enabled}", "metadata_enabled": enabled,
            "started_at_ms": case_start, "finished_at_ms": case_start + 2000,
            "workers": [{"worker_id": worker} for worker in (100, 101, 102)],
            "samples": rows,
        })
    return {"schema_version": 2, "run_id": "a" * 32,
            "started_at_ms": start, "finished_at_ms": start + 40000,
            "cases": cases, "summary": summarize(cases)}


def test_dashboard_preserves_every_timestamp_and_value_without_synthetic_samples(report):
    dashboard = build_dashboard(report)
    assert dashboard["time"]["from"] == iso_time(report["started_at_ms"] - 2000)
    assert dashboard["time"]["to"] == iso_time(report["finished_at_ms"] + 2000)
    assert dashboard["refresh"] == ""
    assert dashboard["timezone"] == "browser"
    assert len({panel["id"] for panel in dashboard["panels"]}) == 13
    for panel in dashboard["panels"][1:7]:
        assert len(panel["targets"]) == 4
        for target, case in zip(panel["targets"], report["cases"]):
            rows = list(csv.reader(io.StringIO(target["csvContent"])))
            expected = [r for r in case["samples"] if r["stage"] == "replay"]
            assert target["scenarioId"] == "csv_content"
            assert len(rows) == len(expected) + 1
            assert [int(datetime.fromisoformat(r[0].replace("Z", "+00:00")).timestamp() * 1000)
                    for r in rows[1:]] == [r["finished_at_ms"] for r in expected]
    hit_panel, ttft_panel, kv_panel = dashboard["panels"][1:4]
    def values(panel):
        return [float(r[1]) for r in list(csv.reader(
            io.StringIO(panel["targets"][0]["csvContent"]),
        ))[1:]]
    assert values(hit_panel) == [1024 / 1040] * 2
    assert values(ttft_panel) == [30] * 2
    assert values(kv_panel) == [4] * 2


def test_worker_ratios_weight_tokens_reset_per_case_and_exclude_other_traffic(report):
    first, second = report["cases"][:2]
    first["samples"][0].update(input_tokens=10, hit_tokens=10, selected_worker=0)
    first["samples"][1].update(input_tokens=30, hit_tokens=0, selected_worker=0)
    second["samples"][0].update(input_tokens=20, hit_tokens=10, selected_worker=0)
    dashboard = build_dashboard(report)
    panel = next(p for p in dashboard["panels"] if p["id"] == 8)
    assert panel["id"] == 8
    assert "all prefill workers" in panel["title"]
    assert panel["gridPos"]["w"] == 24
    rows = {t["refId"]: list(csv.reader(io.StringIO(t["csvContent"])))
            for t in panel["targets"]}
    assert [float(row[1]) for row in rows["W0C1"][1:]] == [1.0, 0.25]
    assert rows["W0C1"][-1][0] == iso_time(first["samples"][1]["finished_at_ms"])
    assert [float(row[1]) for row in rows["W0C2"][1:]] == [0.5]
    assert "W1C1" not in rows
    assert all(f"W1C{i}" in rows for i in (2, 3, 4))
    assert panel["fieldConfig"]["defaults"]["unit"] == "percentunit"
    assert len(rows) == len(panel["targets"])
    assert len({table[0][1] for table in rows.values()}) == len(rows)
    overrides = {o["matcher"]["options"]: {p["id"]: p["value"] for p in o["properties"]}
                 for o in panel["fieldConfig"]["overrides"]}
    assert set(overrides) == {table[0][1] for table in rows.values()}
    assert overrides["Case 1: metadata ON / worker 0"]["color"]["fixedColor"] == "#56B4E9"
    assert overrides["Case 2: metadata OFF / worker 1"]["color"]["fixedColor"] == "#E69F00"
    assert overrides["Case 2: metadata OFF / worker 0"]["custom.lineStyle"]["fill"] == "solid"
    assert overrides["Case 2: metadata OFF / worker 1"]["custom.lineStyle"]["fill"] == "dash"


def test_worker_with_no_replay_observations_has_no_fabricated_zero_series(report):
    for case in report["cases"]:
        for row in case["samples"]:
            if row["stage"] == "replay":
                row["selected_worker"] = 0
    dashboard = build_dashboard(report)
    targets = next(p for p in dashboard["panels"] if p["id"] == 8)["targets"]
    assert len(targets) == 4
    assert all(t["refId"].startswith("W0C") for t in targets)


def test_shared_panel_includes_every_observed_worker_not_just_zero_and_one(report):
    report["cases"][0]["samples"][1]["selected_worker"] = 2
    panel = next(p for p in build_dashboard(report)["panels"] if p["id"] == 8)
    refs = {target["refId"] for target in panel["targets"]}
    assert "W2C1" in refs
    assert "W1C1" not in refs
    assert {"W0C1", "W1C2"} <= refs


def test_qps_counts_boundaries_idle_buckets_partial_windows_and_both_worker_roles():
    case = {
        "started_at_ms": 10000, "finished_at_ms": 16000,
        "workers": [{"worker_id": worker} for worker in (100, 101, 102, 103)],
        "samples": [
            {"stage": "replay", "started_at_ms": start, "finished_at_ms": end,
             "selected_worker": prefill, "decode_worker_id": decode}
            for start, end, prefill, decode in [
                (10000, 10100, 0, 102), (10900, 11000, 1, 103),
                (12000, 12500, 0, 102), (14000, 14500, 1, 103),
            ]
        ] + [{"stage": "eviction", "started_at_ms": 14501, "finished_at_ms": 16000}],
    }
    overall = qps_buckets(case)[None]
    assert overall == [(11000, 1), (12000, 1), (13000, 1), (14000, 0), (14500, 2)]
    for role, workers in (("prefill", {0, 1}), ("decode", {2, 3})):
        grouped = qps_buckets(case, role)
        assert set(grouped) == workers
        for index, (timestamp, qps) in enumerate(overall):
            assert all(points[index][0] == timestamp for points in grouped.values())
            assert sum(points[index][1] for points in grouped.values()) == qps
        durations = [1, 1, 1, 1, 0.5]
        assert sum(value * duration for (_, value), duration in zip(overall, durations)) == 4
    case["samples"] = [case["samples"][0]]
    assert qps_buckets(case)[None] == [(10100, 10)]
    case["samples"][0]["finished_at_ms"] = 10000
    with pytest.raises(ValueError, match="positive replay"):
        qps_buckets(case)


def test_qps_panels_are_standard_and_never_double_count_roles(report):
    dashboard = build_dashboard(report)
    qps = {p["id"]: p for p in dashboard["panels"] if p["id"] in (10, 11, 12)}
    assert len(qps) == 3
    assert [len(qps[i]["targets"]) for i in (10, 11, 12)] == [4, 8, 4]
    for identifier, panel in qps.items():
        assert panel["gridPos"] == {"x": (identifier - 10) * 8, "y": 11, "w": 8, "h": 9}
        assert panel["fieldConfig"]["defaults"]["unit"] == "reqps"
        assert len({t["refId"] for t in panel["targets"]}) == len(panel["targets"])
        assert "NOT engine-phase" in panel["description"]
    for index in range(4):
        def value(panel, suffix):
            target = next(t for t in panel["targets"] if t["refId"] == f"C{index + 1}W{suffix}")
            return float(list(csv.reader(io.StringIO(target["csvContent"])))[1][1])
        assert value(qps[10], "all") == pytest.approx(2 / 0.3)
        assert value(qps[10], "all") == value(qps[11], 0) + value(qps[11], 1)
        assert value(qps[10], "all") == value(qps[12], 2)


def test_duration_mode_qps_retains_full_observation_window_and_idle_tail(report):
    case = report["cases"][0]
    case["replay_started_at_ms"] = case["started_at_ms"]
    case["replay_finished_at_ms"] = case["finished_at_ms"]
    assert qps_buckets(case)[None] == [
        (case["started_at_ms"] + 1000, 0), (case["finished_at_ms"], 2),
    ]
    case["replay_finished_at_ms"] = case["started_at_ms"] + 1
    with pytest.raises(ValueError, match="QPS replay window"):
        qps_buckets(case)


def test_cache_history_is_explicitly_unavailable_in_old_reports(report):
    panel = next(p for p in build_dashboard(report)["panels"] if p["id"] == 13)
    assert panel["type"] == "text"
    assert "cannot be reconstructed" in panel["options"]["content"]


def test_cache_graphs_preserve_totals_capacity_eviction_and_sample_times(report):
    for case in report["cases"]:
        case["cache_samples"] = [
            {"stage": stage, "workers": [
                {"worker_id": worker["worker_id"], "observed_at_ms": case["started_at_ms"] + offset,
                 "capacity_tokens": 1536, "resident_tokens": 1040 if rank < 2 else 0,
                 "allocated_tokens": 1040 if rank < 2 else 0, "evictable_tokens": 1040 if rank < 2 else 0,
                 "evicted_tokens_total": evicted if rank == 0 else 0, "cleared_tokens_total": 0}
                for rank, worker in enumerate(case["workers"])
            ]}
            for stage, offset, evicted in (("before-pressure", 1550, 16), ("eviction", 1600, 1024))
        ]
    dashboard = build_dashboard(report)
    panels = {p["id"]: p for p in dashboard["panels"]}
    assert len(panels) == 15
    totals = [list(csv.reader(io.StringIO(t["csvContent"]))) for t in panels[13]["targets"][:3]]
    assert [[float(r[1]) for r in rows[1:]] for rows in totals] == [[2080]*2, [2080]*2, [4608]*2]
    evicted = list(csv.reader(io.StringIO(panels[15]["targets"][0]["csvContent"])))
    assert [float(r[1]) for r in evicted[1:]] == [16, 1024]
    assert [r[0] for r in evicted[1:]] == [
        iso_time(report["cases"][0]["started_at_ms"] + offset) for offset in (1550, 1600)
    ]
    assert len(annotations_for(report)) == 12
    test_every_recorded_series_uses_accessible_colors_and_explicit_labels(report)
    for i, a in enumerate(dashboard["panels"]):
        a = a["gridPos"]
        for b in dashboard["panels"][i+1:]:
            b = b["gridPos"]
            assert (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"]
                    or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])
    report["cases"][0]["cache_samples"][1]["workers"][0]["evicted_tokens_total"] = 0
    with pytest.raises(ValueError, match="inconsistently"):
        build_dashboard(report)


def test_every_recorded_series_uses_accessible_colors_and_explicit_labels(report):
    dashboard = build_dashboard(report)
    for panel in dashboard["panels"]:
        if panel["type"] != "timeseries":
            continue
        for override in panel["fieldConfig"]["overrides"]:
            label = override["matcher"]["options"]
            properties = {p["id"]: p["value"] for p in override["properties"]}
            assert "metadata ON" in label or "metadata OFF" in label
            assert properties["color"]["fixedColor"] == (
                "#56B4E9" if "metadata ON" in label else "#E69F00"
            )
            assert "custom.lineStyle" in properties
        assert panel["fieldConfig"]["defaults"]["custom"]["drawStyle"] == "line"


def test_e2e_percentile_definitions_and_small_sample_limits():
    assert e2e_percentiles([9]) == {"p50": 9, "p95": 9, "p99": 9}
    assert e2e_percentiles([10, 30]) == {"p50": 20, "p95": 30, "p99": 30}
    assert e2e_percentiles(range(1, 25)) == {"p50": 12.5, "p95": 23, "p99": 24}
    assert e2e_percentiles(reversed(range(1, 101))) == {"p50": 50.5, "p95": 95, "p99": 99}


@pytest.mark.parametrize("values", [[], [-1], [float("nan")], [1, float("inf")]])
def test_invalid_e2e_measurements_fail_explicitly(values):
    with pytest.raises(ValueError, match="latency observations"):
        e2e_percentiles(values)


def test_e2e_graph_is_running_per_case_and_summary_pools_requests(report):
    first = report["cases"][0]
    first["samples"][0]["latency_ms"] = 10
    first["samples"][1]["latency_ms"] = 30
    first["samples"][2]["latency_ms"] = 100000  # Eviction must not enter these percentiles.
    dashboard = build_dashboard(report)
    panel = next(p for p in dashboard["panels"] if p["id"] == 9)
    assert len(panel["targets"]) == 12
    assert panel["fieldConfig"]["defaults"]["unit"] == "ms"
    assert panel["gridPos"]["w"] == 24
    tables = {t["refId"]: list(csv.reader(io.StringIO(t["csvContent"]))) for t in panel["targets"]}
    assert len(tables) == 12
    assert len({rows[0][1] for rows in tables.values()}) == 12
    assert tables["P50C1"][0][1] == "Case 1: metadata ON / p50"
    assert [float(row[1]) for row in tables["P50C1"][1:]] == [10, 20]
    assert [float(row[1]) for row in tables["P95C1"][1:]] == [10, 30]
    assert [float(row[1]) for row in tables["P99C1"][1:]] == [10, 30]
    assert [float(row[1]) for row in tables["P50C2"][1:]] == [400, 400]
    for index, case in enumerate(report["cases"]):
        for percentile in ("P50", "P95", "P99"):
            assert [row[0] for row in tables[f"{percentile}C{index + 1}"][1:]] == [
                iso_time(row["finished_at_ms"]) for row in case["samples"] if row["stage"] == "replay"
            ]
    content = dashboard["panels"][0]["options"]["content"]
    assert "E2E p50 / p95 / p99" in content
    assert "60.0 / 90.0 / 90.0 ms" in content  # Pooled ON latencies: 10, 30, 90, 90.
    assert "400.0 / 400.0 / 400.0 ms" in content
    assert "p99 equals the observed maximum" in content
    for i, a in enumerate(dashboard["panels"]):
        a = a["gridPos"]
        for b in dashboard["panels"][i + 1:]:
            b = b["gridPos"]
            assert (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"]
                    or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])


def test_case_and_eviction_annotations_use_recorded_times(report):
    annotations = annotations_for(report)
    assert len(annotations) == 8
    for index, case in enumerate(report["cases"]):
        phase, eviction = annotations[index * 2:index * 2 + 2]
        assert phase["time"] == case["started_at_ms"]
        assert phase["timeEnd"] == case["finished_at_ms"]
        assert eviction["time"] == case["samples"][-1]["finished_at_ms"]
        assert "eviction-check" in eviction["tags"]
        assert ("metadata-on" in phase["tags"]) == case["metadata_enabled"]


@pytest.mark.parametrize("failure", ["legacy", "missing", "reversed", "out-of-case", "out-of-order", "nonfinite"])
def test_incomplete_or_invalid_recordings_fail_instead_of_fabricating_graphs(report, failure):
    report = copy.deepcopy(report)
    first = report["cases"][0]["samples"][0]
    if failure == "legacy":
        report.pop("schema_version")
    elif failure == "missing":
        first.pop("finished_at_ms")
    elif failure == "reversed":
        first["finished_at_ms"] = first["started_at_ms"] - 1
    elif failure == "out-of-case":
        first["finished_at_ms"] = report["finished_at_ms"] + 1
    elif failure == "out-of-order":
        report["cases"][1]["started_at_ms"] = report["started_at_ms"]
    else:
        first["kv_seconds"] = float("nan")
    with pytest.raises((ValueError, KeyError)):
        build_dashboard(report)
