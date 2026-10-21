"""Recorded aggregate charts for concurrent routing experiments."""

import math

if __package__:
    from .routing_dashboard import (
        DATASOURCE, GRAFANA, METADATA_COLORS, cache_panels, case_title, csv_target,
        iso_time, validate_timestamps,
    )
else:
    from routing_dashboard import (
        DATASOURCE, GRAFANA, METADATA_COLORS, cache_panels, case_title, csv_target,
        iso_time, validate_timestamps,
    )


def p95_comparison(result, succeeded):
    distributions = [result[field] for field in ("ttft_ms", "e2e_ms")]
    if succeeded == 0:
        if any(value is not None for value in distributions):
            raise ValueError("Latency percentiles require successful responses")
        return None
    values = [value["p95"] if value is not None else None for value in distributions]
    if any(value is None or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Successful responses require finite, nonnegative p95 latencies")
    ttft, e2e = values
    if e2e < ttft:
        raise ValueError("E2E p95 cannot precede TTFT p95 for the same successful responses")
    return ttft, e2e, e2e - ttft


def build_dashboard(report):
    validate_timestamps(report)
    start, end = report["started_at_ms"], report["finished_at_ms"]
    text = (
        f"**Recorded concurrent A/B run:** {iso_time(start)} to {iso_time(end)} (UTC).\n\n"
        "**Blue = metadata ON; orange = OFF.** Okabe-Ito colors plus explicit labels and line styles. "
        "80% of arrivals use four hot prefix families; 20% use the remaining 28. Hot families change "
        "every 120 seconds of offered traffic. Prefix lengths are 256/1,024 tokens with a 16-token suffix "
        "and eight output tokens (default block size). The seeded offered sequence is identical per case; "
        "concurrent execution order is not deterministic.\n\n"
        f"**Target:** {report['configuration']['rate']} offered RPS for "
        f"{report['configuration']['duration_seconds']} seconds per case. "
        "Sent/completed rates can be lower. Concurrency-limit rejections are client-side, not server errors. "
        "Overloaded cases may admit different subsets of the same offered sequence; this is a saturation "
        "comparison, not a paired comparison of identical completed requests. "
        "Successful completions, failures and backlog are shown separately. Serial routing/eviction checks "
        "are excluded from load QPS, latency and hit-rate curves. "
        "The cache-size graphs include those checks and mark the deliberate pressure phase.\n\n"
        "| Condition | Offered | Sent | Success | Failed | Client rejected | Success RPS incl. drain | Cache hit ratio | Evicted tokens |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for key, title in (("metadata_on", "ON"), ("metadata_off", "OFF")):
        summary = report["summary"][key]
        ratio = summary["observed_hit_ratio"]
        hit = f"{ratio:.1%}" if ratio is not None else "No lookups"
        text += (
            f"| {title} | {summary['offered']} | {summary['sent']} | {summary['succeeded']} | "
            f"{summary['failed']} | {summary['client_rejected']} | "
            f"{summary['completed_rps_including_drain']:.2f} | {hit} | {summary['evicted_tokens']:g} |\n"
        )
    text += (
        "\nReal Dynamo, simulated SGLang compute/KV slots; this is not a production trace or GPU benchmark. "
        "Latency percentiles include successful responses only; always inspect failures alongside them. "
        "Hover points for recorded values. Empty latency windows have no point, not zero latency. "
        "Series connect observations only within a case."
    )
    comparison_text = (
        "**Whole-case statistics, not one-second percentiles.** All successful load requests, "
        "including drain; excludes setup and eviction checks. These fixed results do not change when zooming.\n\n"
        "| Scope | Success | TTFT p95 | E2E p95 | Difference |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    comparisons = [
        (case_title(index, case), case["load"], case["load"]["totals"]["succeeded"])
        for index, case in enumerate(report["cases"])
    ]
    comparisons.extend(
        (f"Metadata {title} (pooled)", report["summary"][key], report["summary"][key]["succeeded"])
        for key, title in (("metadata_on", "ON"), ("metadata_off", "OFF"))
    )
    for label, result, succeeded in comparisons:
        values = p95_comparison(result, succeeded)
        cells = [f"{value:.1f} ms" for value in values] if values is not None else ["No successes"] * 3
        comparison_text += f"| {label} | {succeeded:,} | " + " | ".join(cells) + " |\n"
    comparison_text += (
        "\n**Difference = E2E p95 - TTFT p95, not p95(E2E - TTFT) or measured decode p95.** "
        "Pooled rows use the saved request-level pooled percentiles, not averages of case/window percentiles. "
        "The separate one-second charts show short-window variation; their p95/p99 often equal the "
        "maximum when few requests finish in a window."
    )
    panels = [{
        "id": 1, "type": "text", "title": "Concurrent A/B - workload and outcomes",
        "gridPos": {"x": 0, "y": 0, "w": 24, "h": 13},
        "options": {"mode": "markdown", "content": text},
    }, {
        "id": 16, "type": "text", "title": "Whole-case latency - TTFT / E2E p95 comparison",
        "gridPos": {"x": 0, "y": 22, "w": 12, "h": 12},
        "options": {"mode": "markdown", "content": comparison_text},
    }]
    definitions = (
        (2, "Overall QPS - offered / sent / successful / failed", "reqps", 0, 13, 8),
        (3, "Prefill QPS - arrivals by worker", "reqps", 8, 13, 8),
        (4, "Decode QPS - arrivals by worker", "reqps", 16, 13, 8),
        (17, "Whole-case p95 references - TTFT / E2E", "ms", 12, 22, 12),
        (5, "Observed prefix-cache hit ratio - all prefill workers", "percentunit", 0, 34, 12),
        (6, "Observed prefix-cache hit ratio - by worker", "percentunit", 12, 34, 12),
        (7, "E2E latency - p50 / p95 / p99 per second", "ms", 0, 43, 12),
        (8, "TTFT - p50 / p95 / p99 per second", "ms", 12, 43, 12),
        (9, "Client in-flight requests", "short", 0, 52, 12),
        (10, "Client concurrency rejections / second", "reqps", 12, 52, 12),
        (11, "Request start lag - p95 per second", "ms", 0, 61, 12),
        (12, "Simulated K/V-generation seconds / second", "short", 12, 61, 12),
    )
    for identifier, title, unit, x, y, width in definitions:
        panels.append({
            "id": identifier, "type": "timeseries", "title": title, "datasource": DATASOURCE,
            "gridPos": {"x": x, "y": y, "w": width, "h": 9},
            "description": (
                "One-second client buckets, including drain after arrivals stop; rates use actual bucket duration. "
                "Worker QPS is the rate of generate-endpoint arrivals between engine metric snapshots, not "
                "client completions or management calls. Hit ratios are cumulative token-weighted reuse since "
                "the load baseline, excluding serial validation and eviction checks. Failed requests can "
                "contribute engine admissions but not successful-client latency. In-flight is client-side backlog, "
                "not a measurement of the server queue. Request start lag measures load-generator delay. "
                "p50 = median; p95/p99 = nearest rank. Small windows have unstable tail estimates."
            ),
            "targets": [], "fieldConfig": {
                "defaults": {"unit": unit, "min": 0, "custom": {
                    "drawStyle": "line", "lineWidth": 2, "showPoints": "auto",
                    "pointSize": 3, "spanNulls": False, "fillOpacity": 0,
                }}, "overrides": [],
            },
            "options": {"tooltip": {"mode": "multi"},
                        "legend": {"displayMode": "list", "placement": "bottom"}},
        })
    by_id = {panel["id"]: panel for panel in panels}
    by_id[17]["gridPos"]["h"] = 12
    by_id[17]["description"] = (
        "Fixed reference lines: each saved whole-case p95 is drawn across that case's load window, "
        "including drain. These are not instantaneous, rolling or cumulative measurements and do not "
        "change when zooming. Solid = TTFT; dashed = E2E. Blue = metadata ON; orange = OFF. "
        "The vertical difference is E2E p95 - TTFT p95, not p95 of per-request post-first-token duration. "
        "No line is drawn for a case with no successful responses."
    )
    by_id[17]["fieldConfig"]["defaults"]["custom"]["showPoints"] = "never"
    by_id[17]["options"]["legend"] = {
        "displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull"],
    }
    for identifier in (7, 8):
        by_id[identifier]["description"] = (
            "Each point uses only successful responses completed in that one-second bucket, not the "
            "whole case. Do not average these percentiles to obtain a case percentile; use the "
            "Whole-case latency comparison and reference panels for that. "
            + by_id[identifier]["description"]
        )

    def add(identifier, index, case, label, points, rank=0):
        if not points:
            return
        panel = by_id[identifier]
        label = case_title(index, case) + " / " + label
        panel["targets"].append(csv_target(
            index, case, points, label=label, ref_id=f"S{len(panel['targets'])}",
        ))
        panel["fieldConfig"]["overrides"].append({
            "matcher": {"id": "byName", "options": label},
            "properties": [
                {"id": "color", "value": {
                    "mode": "fixed", "fixedColor": METADATA_COLORS[case["metadata_enabled"]],
                }},
                {"id": "custom.lineStyle", "value":
                 {"fill": "solid"} if rank == 0 else {"fill": "dash", "dash": [rank * 3, 3]}},
            ],
        })

    for index, case in enumerate(report["cases"]):
        load = case["load"]
        windows = load["windows"]
        previous = load["started_at_ms"]
        totals = {field: 0 for field in ("offered", "sent", "succeeded", "failed", "client_rejected")}
        for window in windows:
            if not (window["started_at_ms"] == previous < window["finished_at_ms"] <= load["finished_at_ms"]
                    and case["started_at_ms"] <= previous < case["finished_at_ms"]):
                raise ValueError("Invalid concurrent measurement window")
            previous = window["finished_at_ms"]
            for field in totals:
                if window[field] < 0 or not math.isfinite(window[field]):
                    raise ValueError("Invalid concurrent request count")
                totals[field] += window[field]
        if previous != load["finished_at_ms"] or totals != load["totals"]:
            raise ValueError("Concurrent windows do not match reported request totals")
        values = p95_comparison(load, totals["succeeded"])
        if values is not None:
            for rank, (label, value) in enumerate(zip(("TTFT p95 (whole case)", "E2E p95 (whole case)"),
                                                     values[:2])):
                add(17, index, case, label, [
                    (load["started_at_ms"], value), (load["finished_at_ms"], value),
                ], rank)
        for rank, field in enumerate(("offered", "sent", "succeeded", "failed", "client_rejected")):
            add(10 if field == "client_rejected" else 2, index, case, field, [
                (w["finished_at_ms"], w[field] * 1000 / (w["finished_at_ms"] - w["started_at_ms"]))
                for w in windows
            ], rank)
        add(9, index, case, "inflight", [(w["finished_at_ms"], w["inflight"]) for w in windows])
        for identifier, field in ((7, "e2e_ms"), (8, "ttft_ms"), (11, "schedule_lag_ms")):
            for rank, percentile in enumerate(("p95",) if identifier == 11 else ("p50", "p95", "p99")):
                add(identifier, index, case, percentile, [
                    (w["finished_at_ms"], w[field][percentile]) for w in windows if w[field] is not None
                ], rank)
        snapshots = [s for s in case["cache_samples"]
                     if s["stage"] in ("load-baseline", "load", "load-final")]
        if len(snapshots) < 2:
            raise ValueError("Concurrent load needs baseline and final engine snapshots")
        baseline = snapshots[0]["workers"]
        for rank in range(3):
            arrivals, ratios, generation = [], [], []
            for before, after in zip(snapshots, snapshots[1:]):
                first, last = before["workers"][rank], after["workers"][rank]
                elapsed = (last["observed_at_ms"] - first["observed_at_ms"]) / 1000
                if elapsed > 0:
                    arrivals.append((last["observed_at_ms"],
                                     (last["generate_requests"] - first["generate_requests"]) / elapsed))
                    generation.append((last["observed_at_ms"],
                                       (last["kv_seconds"] - first["kv_seconds"]) / elapsed))
                queried = last["input_tokens"] - baseline[rank]["input_tokens"]
                if queried > 0:
                    ratios.append((last["observed_at_ms"],
                                   (last["hit_tokens"] - baseline[rank]["hit_tokens"]) / queried))
            add(3 if rank < 2 else 4, index, case, f"worker {rank}", arrivals, rank)
            add(6, index, case, f"worker {rank}", ratios, rank)
            if rank < 2:
                add(12, index, case, f"worker {rank}", generation, rank)
        ratios = []
        for snapshot in snapshots[1:]:
            queried = sum(w["input_tokens"] - b["input_tokens"] for w, b in zip(snapshot["workers"], baseline))
            hits = sum(w["hit_tokens"] - b["hit_tokens"] for w, b in zip(snapshot["workers"], baseline))
            if queried > 0:
                ratios.append((max(w["observed_at_ms"] for w in snapshot["workers"]), hits / queried))
        add(5, index, case, "token-weighted prefix reuse", ratios)
    panels.extend(cache_panels(report, 70))
    return {
        "uid": "routing-ab-" + report["run_id"][:16], "title": "Dynamo load A/B - " + iso_time(start),
        "schemaVersion": 41, "version": 1, "editable": True,
        "tags": ["routing-experiment", "recorded-measurements", "fake-sglang", "concurrent-load"],
        "timezone": "browser", "refresh": "",
        "time": {"from": iso_time(start - 2000), "to": iso_time(end + 2000)},
        "annotations": {"list": [{
            "builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True, "hide": False, "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Case windows, hot-set changes and eviction checks", "type": "dashboard",
        }]},
        "links": [{"title": "Live serving lab (not this experiment)", "type": "link",
                   "url": GRAFANA + "/d/dynamo-local-sglang", "keepTime": False}],
        "panels": panels,
    }
