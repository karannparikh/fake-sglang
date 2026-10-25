import json
from pathlib import Path
import sys

import pytest

from deploy.local import dashboard
from deploy.local.dashboard import build_dashboard, build_routing_live_dashboard, catalog


def test_catalog_covers_every_family_and_untyped_series_without_duplicate_buckets():
    metadata = {
        "dynamo_requests": [{"type": "counter"}],
        "dynamo_latency_seconds": [{"type": "histogram"}],
        "process_bytes": [{"type": "gauge"}],
        "gc_seconds": [{"type": "summary"}],
        "calls_created": [{"type": "counter"}],
    }
    names = [
        "dynamo_requests_total", "dynamo_latency_seconds_bucket",
        "dynamo_latency_seconds_count", "dynamo_latency_seconds_sum",
        "process_bytes", "gc_seconds", "gc_seconds_sum", "gc_seconds_count",
        "calls_created", "up",
    ]
    families = catalog(metadata, names)
    assert families == {
        "dynamo_requests_total": "counter", "dynamo_latency_seconds": "histogram",
        "process_bytes": "gauge", "gc_seconds": "summary", "calls_created": "gauge",
        "up": "gauge",
    }
    dashboard = build_dashboard(metadata, names)
    repeated = {
        metric
        for variable in dashboard["templating"]["list"]
        if variable["name"].startswith("catalog_")
        for metric in variable["query"].split(",")
    }
    assert repeated == set(families)
    all_panels = list(dashboard["panels"])
    all_panels += [child for row in dashboard["panels"] for child in row.get("panels", [])]
    assert len({p["id"] for p in all_panels}) == len(all_panels)
    assert all(row["collapsed"] for row in dashboard["panels"] if row["type"] == "row")
    for p in all_panels:
        if "repeat" in p:
            assert any(v["name"] == p["repeat"] for v in dashboard["templating"]["list"])
    histogram = next(p for p in all_panels if p.get("repeat", "").endswith("_histogram"))
    assert len(histogram["targets"]) == 4
    assert "histogram_quantile" in histogram["targets"][0]["expr"]
    counter = next(p for p in all_panels if p.get("repeat", "").endswith("_counter"))
    assert len(counter["targets"]) == 2
    assert counter["targets"][1]["expr"].startswith("rate(")


def test_invalid_metric_names_are_rejected():
    with pytest.raises(ValueError, match="Unsupported metric name"):
        catalog({'invalid"}': [{"type": "gauge"}]}, [])


def test_three_standard_qps_panels_filter_roles_and_generation_endpoints():
    dashboard = build_dashboard({}, [])
    qps = {p["title"]: p for p in dashboard["panels"] if "QPS -" in p["title"]}
    assert set(qps) == {
        "Overall QPS - completed requests", "Prefill QPS - by worker", "Decode QPS - by worker",
    }
    overall = qps["Overall QPS - completed requests"]
    assert overall["id"] == 2
    assert "sum(rate(dynamo_frontend_requests_total{" in overall["targets"][0]["expr"]
    assert 'dynamo_component_type="frontend"' in overall["targets"][0]["expr"]
    for index, role in enumerate(("prefill", "decode"), start=1):
        p = qps[role.capitalize() + " QPS - by worker"]
        query = p["targets"][0]["expr"]
        assert "sum by (pod, worker_id) (rate(dynamo_component_requests_total{" in query
        assert f'dynamo_component_type="{role}"' in query
        assert 'dynamo_endpoint="generate"' in query
        assert 'job=~"${job:regex}"' in query and 'instance=~"${instance:regex}"' in query
        assert p["gridPos"] == {"x": index * 8, "y": 4, "w": 8, "h": 8}
    assert all(p["fieldConfig"]["defaults"]["unit"] == "reqps" for p in qps.values())


def test_cache_capacity_and_evictions_use_engine_metrics_not_router_event_counts():
    panels = {p["title"]: p for p in build_dashboard({}, [])["panels"]}
    total = panels["Cache tokens - all workers"]
    assert [t["legendFormat"] for t in total["targets"]] == ["resident", "allocated", "capacity"]
    evicted = panels["Pressure-evicted tokens - by worker (cumulative)"]
    assert "sglang:simulated_prefix_cache_evicted_tokens_total" in evicted["targets"][0]["expr"]
    rates = panels["Cache eviction rate - tokens per second"]
    assert all(t["expr"].startswith("rate(sglang:simulated_prefix_cache_") for t in rates["targets"])
    assert "cleared_tokens_total" in rates["targets"][1]["expr"]


def test_live_experiment_keeps_workers_in_one_pod_distinct_and_filters_the_run():
    dashboard = build_routing_live_dashboard()
    assert dashboard["refresh"] == "5s"
    assert dashboard["uid"] == "dynamo-routing-live"
    panels = {p["id"]: p for p in dashboard["panels"]}
    assert panels[10]["gridPos"]["y"] == panels[23]["gridPos"]["y"] == 4
    assert "by (pod, worker_slot, dp_rank)" in panels[19]["targets"][0]["expr"]
    for panel in dashboard["panels"][1:]:
        for target in panel["targets"]:
            assert 'pod=~"${run:regex}"' in target["expr"]
            assert 'job="routing-experiment"' in target["expr"]
            assert "${instance" not in target["expr"]
            assert "{{pod}}" not in target.get("legendFormat", "")
        colors = [v["value"]["fixedColor"] for o in panel["fieldConfig"]["overrides"]
                  for v in o["properties"] if v["id"] == "color"]
        assert set(colors) <= {"#56B4E9", "#E69F00", "#CC79A7"}
    main = build_dashboard({}, [])
    for panel in main["panels"]:
        for target in panel.get("targets", []):
            assert 'job!="routing-experiment"' in target["expr"]


def test_observed_cache_panels_use_engine_counters_and_keep_layout_disjoint():
    dashboard = build_dashboard(
        {"sglang:simulated_prefix_cache_query_tokens": [{"type": "counter"}]},
        ["sglang:simulated_prefix_cache_query_tokens_total"],
    )
    panels = {p["title"]: p for p in dashboard["panels"]}
    fleet = panels["Observed prefix-cache hit ratio - token-weighted"]
    worker = panels["Observed prefix-cache hit ratio - per worker"]
    for p in (fleet, worker):
        expression = p["targets"][0]["expr"]
        assert "rate(sglang:simulated_prefix_cache_hit_tokens_total{" in expression
        assert "rate(sglang:simulated_prefix_cache_query_tokens_total{" in expression
        assert "dynamo_frontend_cached_tokens" not in expression
        assert "clamp_min" not in expression  # An idle denominator must not invent a ratio.
        assert p["fieldConfig"]["defaults"]["unit"] == "percentunit"
        assert expression.count('job=~"${job:regex}"') == 2
        assert expression.count('instance=~"${instance:regex}"') == 2
    assert worker["targets"][0]["expr"].count("sum by (pod, dp_rank)") == 2
    comparison = panels["Predicted / observed prefix-cache hit ratio - mean"]
    assert "dynamo_component_router_kv_hit_rate_sum" in comparison["targets"][0]["expr"]
    assert "sglang:simulated_prefix_cache_hit_ratio_sum" in comparison["targets"][1]["expr"]
    assert "Prefix-cache hit fraction" not in panels
    assert "NOT" in panels["Backend-reported cached fraction (includes transferred KV)"]["description"]
    assert any(title.startswith("Dynamo - observed prefix cache") for title in panels)

    top = dashboard["panels"]
    for i, a in enumerate(top):
        a = a["gridPos"]
        for b in top[i + 1:]:
            b = b["gridPos"]
            assert (a["x"] + a["w"] <= b["x"] or b["x"] + b["w"] <= a["x"]
                    or a["y"] + a["h"] <= b["y"] or b["y"] + b["h"] <= a["y"])


def test_offline_dashboards_are_reproducible(tmp_path, monkeypatch):
    output = tmp_path / "dashboard.json"
    monkeypatch.setattr(sys, "argv", ["dashboard.py", "--offline", "--output", str(output)])

    def reject_network(*args, **kwargs):
        pytest.fail("Offline generation must not contact Prometheus")

    monkeypatch.setattr(dashboard.urllib.request, "urlopen", reject_network)
    dashboard.main()
    expected = json.dumps(build_dashboard({}, []), separators=(",", ":")) + "\n"
    assert output.read_text() == expected
    committed = Path(dashboard.__file__).with_name("dashboard.json")
    assert committed.read_text() == expected
    live_expected = json.dumps(build_routing_live_dashboard(), separators=(",", ":")) + "\n"
    assert (tmp_path / "routing-live-dashboard.json").read_text() == live_expected
    assert committed.with_name("routing-live-dashboard.json").read_text() == live_expected
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dashboard.json", "routing-live-dashboard.json"]
