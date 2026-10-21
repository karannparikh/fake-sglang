#!/usr/bin/env python3
"""Generate a complete, type-aware Grafana catalog from live Prometheus metadata."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import urllib.request

HERE = Path(__file__).resolve().parent
PROM = {"type": "prometheus", "uid": "prometheus"}
FILTERS = 'job=~"${job:regex}",instance=~"${instance:regex}",job!="routing-experiment"'
WINDOW = "$__rate_interval"
METRIC_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")


def metric(name, labels=""):
    return name + "{" + FILTERS + ("," + labels if labels else "") + "}"


def rate(name, labels=""):
    return f"rate({metric(name, labels)}[{WINDOW}])"


def worker_qps(role):
    return "sum by (pod, worker_id) (" + rate(
        "dynamo_component_requests_total",
        f'dynamo_component_type="{role}",dynamo_endpoint="generate"',
    ) + ")"


def quantile(name, value):
    return f"histogram_quantile({value}, sum by (le) ({rate(name + '_bucket')}))"


def target(expression, ref="A", legend=None):
    result = {"refId": ref, "expr": expression, "range": True}
    if legend:
        result["legendFormat"] = legend
    return result


def panel(identifier, title, targets, x, y, unit="short", description=""):
    return {
        "id": identifier, "title": title, "type": "timeseries",
        "datasource": PROM, "gridPos": {"x": x, "y": y, "w": 12, "h": 8},
        "interval": "15s", "description": description,
        "targets": targets,
        "fieldConfig": {
            "defaults": {"unit": unit, "custom": {"lineWidth": 1, "fillOpacity": 5}},
            "overrides": [],
        },
        "options": {
            "legend": {"displayMode": "table", "placement": "bottom",
                       "calcs": ["lastNotNull"]},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def overview():
    definitions = [
        ("Overall QPS - completed requests", [target(
            "sum(" + rate("dynamo_frontend_requests_total", 'dynamo_component_type="frontend"') + ")",
            legend="overall",
        )], "reqps"),
        ("Output tokens / second", [target(f"sum({rate('dynamo_frontend_output_tokens_total')})")], "short"),
    ]
    for title, name in [
        ("Time to first token", "dynamo_frontend_time_to_first_token_seconds"),
        ("Inter-token latency", "dynamo_frontend_inter_token_latency_seconds"),
        ("Request duration", "dynamo_frontend_request_duration_seconds"),
    ]:
        definitions.append((
            title + " - p50 / p95 / p99",
            [target(quantile(name, q), ref, label)
             for q, ref, label in [(0.5, "A", "p50"), (0.95, "B", "p95"), (0.99, "C", "p99")]],
            "s",
        ))
    definitions.extend([
        ("Frontend active / queued requests", [
            target(metric("dynamo_frontend_active_requests"), "A", "active"),
            target(metric("dynamo_frontend_queued_requests"), "B", "HTTP queue"),
            target(metric("dynamo_frontend_router_queue_pending_requests"), "C", "router queue"),
        ], "short"),
        ("Input / output tokens per request - mean", [
            target(f"sum({rate('dynamo_frontend_input_sequence_tokens_sum')}) / sum({rate('dynamo_frontend_input_sequence_tokens_count')})", "A", "input"),
            target(f"sum({rate('dynamo_frontend_output_sequence_tokens_sum')}) / sum({rate('dynamo_frontend_output_sequence_tokens_count')})", "B", "output"),
        ], "short"),
        ("Backend-reported cached fraction (includes transferred KV)", [
            target(f"sum({rate('dynamo_frontend_cached_tokens_sum')}) / sum({rate('dynamo_frontend_input_sequence_tokens_sum')})"),
        ], "percentunit"),
        ("Prefill QPS - by worker", [
            target(worker_qps("prefill"), legend="{{pod}} / {{worker_id}}"),
        ], "reqps"),
        ("Simulated KV-cache occupancy", [
            target(metric("dynamo_component_gpu_cache_usage_percent")),
        ], "percentunit"),
        ("KV-router active decode blocks", [
            target(metric("dynamo_frontend_worker_active_decode_blocks")),
        ], "short"),
        ("Planner actual / desired replicas", [
            target(metric("dynamo_planner_num_prefill_replicas"), "A", "prefill actual"),
            target(metric("dynamo_planner_num_decode_replicas"), "B", "decode actual"),
            target(metric("dynamo_planner_predicted_num_prefill_replicas"), "C", "prefill desired"),
            target(metric("dynamo_planner_predicted_num_decode_replicas"), "D", "decode desired"),
        ], "short"),
        ("Planner FPM queued prefill tokens", [
            target(metric("dynamo_planner_engine_queued_prefill_tokens")),
        ], "short"),
        ("Planner FPM decode KV tokens", [
            target(metric("dynamo_planner_engine_inflight_decode_kv_tokens"), "A"),
            target(metric("dynamo_planner_engine_queued_decode_kv_tokens"), "B"),
        ], "short"),
        ("Request cancellation / rejection / TCP errors per second", [
            target(f"sum({rate('dynamo_component_cancellation_total')})", "A", "cancelled"),
            target(f"sum({rate('dynamo_rejection_request_total')})", "B", "rejected"),
            target(f"sum({rate('dynamo_transport_tcp_errors_total')})", "C", "TCP errors"),
        ], "ops"),
        ("Scrape health - all lab services", [target(metric("up"))], "short"),
        ("Observed prefix-cache hit ratio - token-weighted", [
            target(
                f"sum({rate('sglang:simulated_prefix_cache_hit_tokens_total')}) / "
                f"sum({rate('sglang:simulated_prefix_cache_query_tokens_total')})",
                legend="observed prefix reuse",
            ),
        ], "percentunit"),
        ("Observed prefix-cache hit ratio - per worker", [
            target(
                f"sum by (pod, dp_rank) ({rate('sglang:simulated_prefix_cache_hit_tokens_total')}) / "
                f"sum by (pod, dp_rank) ({rate('sglang:simulated_prefix_cache_query_tokens_total')})",
                legend="{{pod}} / DP {{dp_rank}}",
            ),
        ], "percentunit"),
        ("Predicted / observed prefix-cache hit ratio - mean", [
            target(
                f"sum({rate('dynamo_component_router_kv_hit_rate_sum')}) / "
                f"sum({rate('dynamo_component_router_kv_hit_rate_count')})",
                "A", "router predicted (routing observations)",
            ),
            target(
                f"sum({rate('sglang:simulated_prefix_cache_hit_ratio_sum')}) / "
                f"sum({rate('sglang:simulated_prefix_cache_hit_ratio_count')})",
                "B", "engine observed (initial prefill admissions)",
            ),
        ], "percentunit"),
        ("Observed prefix-cache reused / queried tokens per second", [
            target(f"sum({rate('sglang:simulated_prefix_cache_hit_tokens_total')})", "A", "reused"),
            target(f"sum({rate('sglang:simulated_prefix_cache_query_tokens_total')})", "B", "queried"),
        ], "short"),
        ("Simulated K/V-generation seconds per second", [
            target(rate("sglang:simulated_kv_generation_seconds_total"), legend="{{pod}} / DP {{dp_rank}}"),
        ], "short"),
        ("Decode QPS - by worker", [
            target(worker_qps("decode"), legend="{{pod}} / {{worker_id}}"),
        ], "reqps"),
        ("Cache tokens - all workers", [
            target(f"sum({metric('sglang:simulated_prefix_cache_' + field + '_tokens')})", ref, field)
            for field, ref in (("resident", "A"), ("allocated", "B"), ("capacity", "C"))
        ], "short"),
        ("Resident prefix-cache tokens / capacity - by worker", [
            target(metric("sglang:simulated_prefix_cache_" + field + "_tokens"), ref,
                   "{{pod}} / DP {{dp_rank}} / " + field)
            for field, ref in (("resident", "A"), ("capacity", "B"))
        ], "short"),
        ("Pressure-evicted tokens - by worker (cumulative)", [
            target(metric("sglang:simulated_prefix_cache_evicted_tokens_total"),
                   legend="{{pod}} / DP {{dp_rank}}"),
        ], "short"),
        ("Cache eviction rate - tokens per second", [
            target(rate("sglang:simulated_prefix_cache_" + field + "_tokens_total"), ref,
                   "{{pod}} / DP {{dp_rank}} / " + field)
            for field, ref in (("evicted", "A"), ("cleared", "B"))
        ], "short"),
    ])
    panels = [
        panel(i + 2, title, queries, (i % 2) * 12, 4 + (i // 2) * 8, unit)
        for i, (title, queries, unit) in enumerate(definitions)
    ]
    qps_index = other_index = 0
    for p in panels:
        if "QPS -" in p["title"]:
            p["gridPos"].update(x=qps_index * 8, y=4, w=8)
            qps_index += 1
            p["description"] = (
                "Completed frontend requests per second, including success/error outcomes. "
                "Counted once at the frontend; never sum prefill and decode phase counts for overall QPS."
                if p["title"].startswith("Overall") else
                "Request arrivals per second at each worker's generate endpoint, filtered by role. "
                "Excludes management and KV-query endpoints. Includes retries or canaries if they use "
                "generate; not a success-only counter. Arrival timing differs from frontend completion timing."
            )
        else:
            p["gridPos"].update(x=(other_index % 2) * 12, y=12 + (other_index // 2) * 8)
            other_index += 1
        if p["id"] >= 24:
            p["description"] = (
                "Engine measurements, independent of KV-event publication. Resident = reusable full-block "
                "prefix tokens; allocated = occupied token slots, including active/partial blocks; "
                "capacity = configured slots. Totals sum replicas, not deduplicated content. "
                "Pressure evictions exclude explicit cache clears (shown separately in the rate panel). "
                "Counters reset on worker restart. Cache is space-driven LRU, not TTL-based."
            )
        elif p["title"].startswith("Observed prefix-cache"):
            p["description"] = (
                "Engine-observed local prefix reuse at the first successful prefill admission. "
                "Includes all prompt tokens in the denominator, including partial blocks. "
                "Excludes transferred decode KV, no-peer probes/warmup, admission retries and preemption re-prefills. "
                "Counters reset on worker restart. No traffic / no local lookups gives no ratio, not zero."
            )
        elif p["title"].startswith("Predicted / observed"):
            p["description"] = (
                "Means of fractions, not token-weighted. Router predictions and engine observations "
                "are different populations: disaggregation can record multiple routing observations "
                "per frontend request; the engine records one initial prefill admission. "
                "Observed reuse excludes transferred decode KV."
            )
        elif p["title"].startswith("Backend-reported cached fraction"):
            p["description"] = (
                "Backend metadata counts transferred KV as cached in the fake decode engine. "
                "This is NOT an observed cross-request prefix-cache hit ratio."
            )
    return panels


def group_for(name):
    for prefixes, group in [
        (("sglang:simulated_prefix_cache_", "sglang:simulated_kv_generation_"), "Dynamo - observed prefix cache"),
        (("dynamo_frontend_worker_", "dynamo_component_gpu_", "dynamo_component_total_"), "Dynamo - simulated KV cache"),
        (("dynamo_component_router_", "dynamo_router_", "dynamo_component_kv_", "dynamo_component_ckf_"), "Dynamo - routing and KV events"),
        (("dynamo_frontend_",), "Dynamo - frontend"),
        (("dynamo_planner_",), "Dynamo - Planner (logical GPU accounting)"),
        (("dynamo_operator_", "controller_runtime_", "workqueue_", "leader_election_", "certwatcher_", "rest_client_"), "Kubernetes - Dynamo operator"),
        (("dynamo_tokio_", "dynamo_transport_"), "Dynamo - runtime and transport"),
        (("dynamo_", "sglang"), "Dynamo - workers and request queues"),
        (("etcd_",), "Infrastructure - etcd"),
        (("gnatsd_", "jetstream_",), "Infrastructure - NATS and JetStream"),
        (("otelcol_",), "Observability - OpenTelemetry Collector"),
        (("grafana_",), "Observability - Grafana"),
        (("tempo_",), "Observability - Tempo"),
        (("loki_", "cortex_",), "Observability - Loki"),
        (("prometheus_tsdb_",), "Prometheus - time-series storage"),
        (("prometheus_sd_",), "Prometheus - service discovery"),
        (("prometheus_target_", "prometheus_remote_", "prometheus_notifications_"), "Prometheus - scrape and remote delivery"),
        (("prometheus_", "promhttp_",), "Prometheus - query engine and server"),
        (("go_godebug_",), "Runtime - Go compatibility flags"),
        (("go_gc_",), "Runtime - Go garbage collection"),
        (("go_memory_", "go_memstats_",), "Runtime - Go memory"),
        (("go_",), "Runtime - Go scheduler and CPU"),
        (("process_", "os_", "python_",), "Runtime - process resources and Python"),
        (("grpc_", "net_",), "Runtime - network and gRPC"),
    ]:
        if name.startswith(prefixes):
            return group
    return "Other exporter and scrape metrics"


def catalog(metadata, names):
    families = {}
    covered = set()
    names = set(names)
    for name, records in metadata.items():
        if not METRIC_NAME.fullmatch(name):
            raise ValueError(f"Unsupported metric name: {name!r}")
        kind = records[0]["type"]
        actual_name = name
        if kind == "counter" and name not in names and name + "_total" in names:
            actual_name += "_total"
        families[actual_name] = "gauge" if actual_name.endswith("_created") else kind
        covered.add(name)
        if kind in ("histogram", "summary"):
            covered.update(name + suffix for suffix in ("_bucket", "_count", "_sum", "_created"))
        elif kind == "counter":
            covered.add(name + "_total")
    for name in names:
        if not METRIC_NAME.fullmatch(name):
            raise ValueError(f"Unsupported metric name: {name!r}")
        if name not in covered:
            families[name] = "gauge"
    return families


def repeat_variable(name, metrics):
    return {
        "name": name, "type": "custom", "hide": 2, "skipUrlSync": True,
        "query": ",".join(metrics), "multi": True, "includeAll": True,
        "current": {"text": ["All"], "value": ["$__all"]},
        "options": [{"text": "All", "value": "$__all", "selected": True}]
                   + [{"text": m, "value": m, "selected": False} for m in metrics],
    }


def catalog_panel(identifier, variable, kind, y):
    selector = '{__name__=~"${' + variable + ':regex}",' + FILTERS + "}"
    queries = [target(selector)]
    description = "Raw metric values. Labels distinguish every exported series; units are encoded in metric names."
    if kind == "counter":
        queries.append(target(f"rate({selector}[{WINDOW}])", "B"))
        description = "A: cumulative counter (left axis). B: reset-aware rate per second (right axis). All series labels are preserved."
    elif kind in ("histogram", "summary"):
        def part(suffix):
            return '{__name__=~"(${' + variable + ':regex})' + suffix + '",' + FILTERS + "}"
        count, total = part("_count"), part("_sum")
        if kind == "histogram":
            buckets = part("_bucket")
            queries = [
                target(
                    f'label_replace(histogram_quantile({q}, rate({buckets}[{WINDOW}])), '
                    f'"dynamo_lab_statistic", "{label}", "", "")', ref,
                )
                for q, ref, label in [(0.5, "A", "p50"), (0.95, "B", "p95"), (0.99, "C", "p99")]
            ]
            description = "A: p50. B: p95. C: p99. D: mean. Quantiles retain each series' labels. Raw buckets/count/sum are available in the Raw metric explorer."
        else:
            queries = [target(selector)]
            description = "A: exporter-provided quantiles. D: mean from sum/count rates. Raw count/sum are available in the Raw metric explorer."
        queries.append(target(
            f'label_replace(rate({total}[{WINDOW}]) / rate({count}[{WINDOW}]), '
            '"dynamo_lab_statistic", "mean", "", "")', "D",
        ))
    result = panel(identifier, "${" + variable + "}", queries, 0, y, description=description)
    result.update({"repeat": variable, "repeatDirection": "h", "maxPerRow": 2})
    if kind == "counter":
        result["fieldConfig"]["overrides"] = [{
            "matcher": {"id": "byFrameRefID", "options": "B"},
            "properties": [{"id": "custom.axisPlacement", "value": "right"},
                           {"id": "custom.axisLabel", "value": "rate / second"}],
        }]
    return result


def build_dashboard(metadata, names):
    families = catalog(metadata, names)
    groups = defaultdict(lambda: defaultdict(list))
    for name, kind in sorted(families.items()):
        groups[group_for(name)][kind].append(name)
    variables = [
        {"name": "job", "label": "Source", "type": "query", "datasource": PROM,
         "query": 'label_values(up{job!="routing-experiment"}, job)', "refresh": 1, "includeAll": True,
         "multi": True, "allValue": ".*", "current": {"text": "All", "value": "$__all"}},
        {"name": "instance", "label": "Instance", "type": "query", "datasource": PROM,
         "query": 'label_values(up{job=~"$job",job!="routing-experiment"}, instance)', "refresh": 1,
         "includeAll": True, "multi": True, "allValue": ".*",
         "current": {"text": "All", "value": "$__all"}},
        {"name": "metric", "label": "Raw metric", "type": "query", "datasource": PROM,
         "query": 'label_values({job=~"$job",instance=~"$instance",job!="routing-experiment"}, __name__)',
         "refresh": 1, "includeAll": True, "multi": True, "allValue": ".*",
         "current": {"text": "dynamo_frontend_requests_total",
                     "value": "dynamo_frontend_requests_total"}},
    ]
    panels = [{
        "id": 1, "type": "text", "title": "Real Dynamo / simulated SGLang",
        "gridPos": {"x": 0, "y": 0, "w": 24, "h": 4},
        "options": {"mode": "markdown", "content": (
            f"**{len(families)} metric families across {len(groups)} sections** "
            "in this catalog. Expand sections below the overview to see every family. "
            "Source and Instance filter the entire dashboard. The Raw metric explorer includes all live metric names, "
            "including histogram buckets, counts and sums, plus newly registered metrics.\n\n"
            "Isolated routing experiments are excluded here.\n\n"
            "**Simulation boundary:** engine timing, KV use and GPU-slot accounting are simulated. "
            "There is no physical GPU utilization, power or VRAM telemetry. SLA-regression and power-planning "
            "metrics are not calibrated for this lab. **No data** means no samples were emitted or the feature is inactive, not zero."
        )},
    }] + overview()
    y = max(p["gridPos"]["y"] + p["gridPos"]["h"] for p in panels)
    raw = '{__name__=~"${metric:regex}",' + FILTERS + "}"
    raw_chart = panel(101, "Raw metric - all matching series", [target(raw)], 0, y + 1)
    raw_table = {
        "id": 102, "type": "table", "title": "Raw metric - current values and labels",
        "datasource": PROM, "gridPos": {"x": 12, "y": y + 1, "w": 12, "h": 8},
        "targets": [{"refId": "A", "expr": raw, "format": "table", "instant": True, "range": False}],
        "options": {"showHeader": True},
    }
    panels.append({"id": 100, "type": "row", "title": "Raw metric explorer",
                   "collapsed": True, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
                   "panels": [raw_chart, raw_table]})
    for index, (group, kinds) in enumerate(sorted(groups.items()), start=1):
        y += 1
        children = []
        for offset, (kind, metrics) in enumerate(sorted(kinds.items())):
            var = f"catalog_{index}_{kind}"
            variables.append(repeat_variable(var, metrics))
            children.append(catalog_panel(1000 + index * 10 + offset, var, kind, y + 1 + offset * 8))
        count = sum(map(len, kinds.values()))
        panels.append({
            "id": 200 + index, "type": "row", "title": f"{group} - {count} families",
            "collapsed": True, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
            "panels": children,
        })
    return {
        "uid": "dynamo-local-sglang", "title": "Dynamo - mock SGLang lab",
        "description": "Full lab metric catalog, inference overview and live raw-series explorer. Real Dynamo, simulated inference; no physical GPU telemetry.",
        "schemaVersion": 39, "version": 2, "refresh": "10s",
        "time": {"from": "now-15m", "to": "now"},
        "tags": ["dynamo", "sglang", "simulation", "full-metric-catalog"],
        "templating": {"list": variables}, "panels": panels,
        "links": [
            {"title": "Prometheus", "type": "link", "url": "http://127.0.0.1:9090", "targetBlank": True},
        ],
    }


def build_routing_live_dashboard():
    selected = (2, 10, 23, 4, 6, 18, 19, 24, 25, 26, 27, 22)
    source = {p["id"]: p for p in overview()}
    panels = [{
        "id": 1, "type": "text", "title": "Live experiment - not a saved report",
        "gridPos": {"x": 0, "y": 0, "w": 24, "h": 4},
        "options": {"mode": "markdown", "content": (
            "**Live Prometheus data from the isolated experiment.** Refreshes every 5 seconds. "
            "Select a run below. Data begins when scraping starts; earlier history is not backfilled. "
            "Cases restart their worker processes, so worker IDs and cumulative counters change.\n\n"
            "**QPS:** frontend completions; prefill/decode arrivals at generate endpoints. "
            "Includes short setup/validation requests. Client-offered/rejected counts are not exposed live; "
            "the final recorded report contains those. Colors identify worker slots/statistics here, "
            "**not metadata ON/OFF**. Blue/orange/purple plus labels and line styles avoid red/green comparisons."
        )},
    }]
    for index, identifier in enumerate(selected):
        p = source[identifier]
        p["gridPos"] = (
            {"x": index * 8, "y": 4, "w": 8, "h": 8} if index < 3 else
            {"x": ((index - 3) % 2) * 12, "y": 12 + ((index - 3) // 2) * 8, "w": 12, "h": 8}
        )
        p["interval"] = "5s"
        p["fieldConfig"]["defaults"]["color"] = {"mode": "fixed", "fixedColor": "#56B4E9"}
        p["fieldConfig"]["defaults"]["custom"].update(lineWidth=2, fillOpacity=0)
        for target_index, query in enumerate(p["targets"]):
            query["expr"] = query["expr"].replace(
                FILTERS, 'job="routing-experiment",pod=~"${run:regex}"',
            ).replace("by (pod, dp_rank)", "by (pod, worker_slot, dp_rank)").replace(
                "by (pod, worker_id)", "by (pod, worker_slot, worker_id)",
            )
            if "legendFormat" in query:
                query["legendFormat"] = query["legendFormat"].replace("{{pod}}", "{{worker_slot}}")
            p["fieldConfig"]["overrides"].append({
                "matcher": {"id": "byFrameRefID", "options": query["refId"]},
                "properties": [
                    {"id": "color", "value": {
                        "mode": "fixed", "fixedColor": ("#56B4E9", "#E69F00", "#CC79A7")[target_index % 3],
                    }},
                    {"id": "custom.lineStyle", "value":
                     {"fill": "solid"} if target_index == 0 else {"fill": "dash", "dash": [6 * target_index, 3]}},
                ],
            })
        for rank, slot in enumerate(("prefill-0", "prefill-1", "decode-2")):
            p["fieldConfig"]["overrides"].append({
                "matcher": {"id": "byRegexp", "options": f"/^{slot} /"},
                "properties": [
                    {"id": "color", "value": {
                        "mode": "fixed", "fixedColor": ("#56B4E9", "#E69F00", "#CC79A7")[rank],
                    }},
                ],
            })
            if len(p["targets"]) == 1:
                p["fieldConfig"]["overrides"][-1]["properties"].append({
                    "id": "custom.lineStyle", "value":
                    {"fill": "solid"} if rank == 0 else {"fill": "dash", "dash": [3 * rank, 3]},
                })
        panels.append(p)
    return {
        "uid": "dynamo-routing-live", "title": "Dynamo - live routing experiment",
        "schemaVersion": 41, "version": 1, "refresh": "5s", "timezone": "browser",
        "time": {"from": "now-15m", "to": "now"},
        "tags": ["dynamo", "live-routing-experiment", "simulation"],
        "templating": {"list": [
            {"name": "run", "label": "Experiment run", "type": "query", "datasource": PROM,
             "query": 'label_values(up{job="routing-experiment"}, pod)', "refresh": 2,
             "includeAll": True, "allValue": ".*",
             "current": {"text": "All", "value": "$__all"}},
        ]},
        "panels": panels,
        "links": [{"title": "Completed recorded experiments", "type": "dashboards",
                   "tags": ["routing-experiment"], "asDropdown": True, "keepTime": False}],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--output", type=Path, default=HERE / "dashboard.json")
    parser.add_argument("--offline", action="store_true",
                        help="Generate the reproducible baseline overview without contacting Prometheus")
    args = parser.parse_args()
    def fetch(path):
        with urllib.request.urlopen(args.prometheus_url.rstrip("/") + path, timeout=30) as response:
            data = json.load(response)
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {data}")
        return data["data"]
    if args.offline:
        metadata, names = {}, []
    else:
        metadata = fetch("/api/v1/metadata")
        names = fetch("/api/v1/label/__name__/values")
        if not metadata or not names:
            raise RuntimeError("Prometheus has no metrics; refusing to replace the catalog")
    result = build_dashboard(metadata, names)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(f"Generated {len(catalog(metadata, names))} metric families in {args.output}")


if __name__ == "__main__":
    main()
