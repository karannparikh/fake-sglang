#!/usr/bin/env python3
"""Publish recorded, timestamped routing results as a persistent Grafana dashboard."""

import argparse
import base64
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import statistics
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
GRAFANA = "http://127.0.0.1:3000"
DATASOURCE = {"type": "grafana-testdata-datasource", "uid": "routing-reports"}
METADATA_COLORS = {True: "#56B4E9", False: "#E69F00"}  # Okabe-Ito sky blue / orange.


def iso_time(milliseconds):
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat(
        timespec="milliseconds",
    ).replace("+00:00", "Z")


def validate_timestamps(report):
    if report.get("schema_version") not in (2, 3):
        raise ValueError("Report lacks exact timestamps; rerun routing-test rather than invent a timeline")
    if not re.fullmatch(r"[0-9a-f]{32}", report["run_id"]):
        raise ValueError("Invalid routing run ID")

    def bounds(item, lower, upper):
        start, end = item["started_at_ms"], item["finished_at_ms"]
        if (type(start) is not int or type(end) is not int
                or not lower <= start <= end <= upper):
            raise ValueError(f"Invalid or out-of-order timestamp bounds: {start}, {end}")
        return start, end

    start, end = bounds(report, 1, 2**53)
    if not report["cases"]:
        raise ValueError("Report has no cases")
    previous = start
    for case in report["cases"]:
        case_start, case_end = bounds(case, previous, end)
        if report["schema_version"] == 2 and not any(row["stage"] == "replay" for row in case["samples"]):
            raise ValueError("Case has no replay measurements")
        request_end = case_start
        for row in case["samples"]:
            _, request_end = bounds(row, request_end, case_end)
        previous = case_end


def case_title(index, case):
    return f"Case {index + 1}: metadata {'ON' if case['metadata_enabled'] else 'OFF'}"


def csv_target(index, case, measurements, *, label=None, ref_id=None):
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["time", label if label is not None else case_title(index, case)])
    for timestamp, number in measurements:
        if not math.isfinite(number):
            raise ValueError(f"Nonfinite measurement in {case['name']}")
        writer.writerow([iso_time(timestamp), number])
    return {
        "refId": ref_id if ref_id is not None else chr(ord("A") + index),
        "datasource": DATASOURCE,
        "scenarioId": "csv_content", "csvContent": output.getvalue(),
    }


def series_target(index, case, value):
    return csv_target(index, case, (
        (row["finished_at_ms"], value(row))
        for row in case["samples"] if row["stage"] == "replay"
    ))


def worker_ratio_target(index, case, worker):
    queried = hits = 0
    measurements = []
    for row in case["samples"]:
        if row["stage"] != "replay" or row["selected_worker"] != worker:
            continue
        if not 0 <= row["hit_tokens"] <= row["input_tokens"] or row["input_tokens"] <= 0:
            raise ValueError(f"Invalid worker cache observation in {case['name']}")
        queried += row["input_tokens"]
        hits += row["hit_tokens"]
        measurements.append((row["finished_at_ms"], hits / queried))
    if not measurements:
        return None
    return csv_target(
        index, case, measurements,
        label=f"{case_title(index, case)} / worker {worker}", ref_id=f"W{worker}C{index + 1}",
    )


def e2e_percentiles(latencies):
    values = sorted(latencies)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("E2E percentiles require finite, nonnegative latency observations")
    return {
        "p50": statistics.median(values),
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "p99": values[math.ceil(0.99 * len(values)) - 1],
    }


def e2e_percentile_targets(index, case):
    measurements = {"p50": [], "p95": [], "p99": []}
    latencies = []
    for row in case["samples"]:
        if row["stage"] != "replay":
            continue
        latencies.append(row["latency_ms"])
        for percentile, value in e2e_percentiles(latencies).items():
            measurements[percentile].append((row["finished_at_ms"], value))
    return [
        csv_target(
            index, case, points, label=f"{case_title(index, case)} / {percentile}",
            ref_id=f"{percentile.upper()}C{index + 1}",
        )
        for percentile, points in measurements.items()
    ]


def qps_buckets(case, role=None):
    rows = [row for row in case["samples"] if row["stage"] == "replay"]
    first = min(row["started_at_ms"] for row in rows)
    last = max(row["finished_at_ms"] for row in rows)
    start = case.get("replay_started_at_ms", first)
    end = case.get("replay_finished_at_ms", last)
    if not case["started_at_ms"] <= start <= first <= last <= end <= case["finished_at_ms"]:
        raise ValueError("QPS replay window must contain every replay request and stay within the case")
    if end <= start:
        raise ValueError("QPS requires a positive replay observation interval")
    if role not in (None, "prefill", "decode"):
        raise ValueError(f"Unsupported QPS worker role: {role}")
    aliases = ({worker["worker_id"]: index for index, worker in enumerate(case["workers"])}
               if role == "decode" else {})
    keys = [
        row["selected_worker"] if role == "prefill" else
        aliases[row["decode_worker_id"]] if role == "decode" else None
        for row in rows
    ]
    count = math.ceil((end - start) / 1000)
    totals = {worker: [0] * count for worker in set(keys)}
    for row, worker in zip(rows, keys):
        # Half-open one-second buckets; the final bucket also owns its right endpoint.
        bucket = min((row["finished_at_ms"] - start) // 1000, count - 1)
        totals[worker][bucket] += 1
    return {
        worker: [
            (min(start + (index + 1) * 1000, end),
             completed * 1000 / min(1000, end - start - index * 1000))
            for index, completed in enumerate(completions)
        ]
        for worker, completions in totals.items()
    }


def qps_panels(report, y):
    panels = []
    for column, (role, title) in enumerate((
        (None, "Overall QPS - completed replay requests"),
        ("prefill", "Prefill QPS - by worker"),
        ("decode", "Decode QPS - by worker"),
    )):
        targets, overrides = [], []
        for index, case in enumerate(report["cases"]):
            for rank, (worker, measurements) in enumerate(sorted(qps_buckets(case, role).items())):
                label = case_title(index, case)
                if worker is not None:
                    label += f" / {role} worker {worker}"
                targets.append(csv_target(
                    index, case, measurements, label=label,
                    ref_id=f"C{index + 1}W{worker if worker is not None else 'all'}",
                ))
                overrides.append({
                    "matcher": {"id": "byName", "options": label},
                    "properties": [
                        {"id": "color", "value": {
                            "mode": "fixed", "fixedColor": METADATA_COLORS[case["metadata_enabled"]],
                        }},
                        {"id": "custom.lineStyle", "value":
                         {"fill": "solid"} if rank == 0 else {"fill": "dash", "dash": [3 * rank, 3]}},
                    ],
                })
        panels.append({
            "id": 10 + column, "type": "timeseries", "title": title, "datasource": DATASOURCE,
            "gridPos": {"x": 8 * column, "y": y, "w": 8, "h": 9},
            "description": (
                "Completed replay requests / elapsed seconds in non-overlapping one-second buckets, "
                "anchored at each case's first replay request start. The final partial bucket is "
                "normalized by its actual duration. Points are at bucket ends. Empty buckets inside "
                "replay are zero; setup/teardown gaps are omitted. Seeds and eviction checks excluded. "
                "Worker panels attribute each completed request to its actual destination, using E2E "
                "completion time, NOT engine-phase completion timestamps. Sum workers within either "
                "role to recover overall QPS; do not add both roles. Worker aliases follow the report "
                "worker list (default: prefill 0/1, decode 2)."
            ),
            "targets": targets,
            "fieldConfig": {
                "defaults": {"unit": "reqps", "min": 0, "custom": {
                    "drawStyle": "line", "lineInterpolation": "stepBefore",
                    "showPoints": "always", "pointSize": 5,
                    "fillOpacity": 0, "lineWidth": 2, "spanNulls": False,
                }},
                "overrides": overrides,
            },
            "options": {"tooltip": {"mode": "multi"},
                        "legend": {"displayMode": "list", "placement": "bottom"}},
        })
    return panels


def cache_panels(report, y):
    recorded = ["cache_samples" in case for case in report["cases"]]
    if not any(recorded):
        config = report.get("configuration", {})
        capacity = config.get("kv_blocks", 0) * config.get("block_size", 0)
        return [{
            "id": 13, "type": "text", "title": "Cache occupancy history not recorded in this run",
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 4},
            "options": {"mode": "markdown", "content": (
                (f"Configured capacity: **{capacity:,} token slots per worker**. " if capacity else "")
                + "This report predates resident-token and engine-eviction measurements. "
                "Current occupancy cannot be reconstructed from cache-hit ratios or router event counts. "
                "New experiments include cache-state graphs; this historical run is unchanged."
            )},
        }]
    if not all(recorded):
        raise ValueError("Cache measurements must be present in every case, not a subset")
    panels = [{
        "id": 13 + index, "type": "timeseries", "title": title, "datasource": DATASOURCE,
        "gridPos": {"x": 0 if index < 2 else 12, "y": y if index == 0 else y + 9,
                    "w": 24 if index == 0 else 12, "h": 9},
        "description": (
            "Exact engine counters sampled after each seed, replay, pressure and relocation request. "
            "Blue = metadata ON; orange = OFF. Includes setup/pressure traffic, unlike replay-only QPS. "
            "Resident = reusable full-block prefix tokens; allocated = occupied token slots, including "
            "active/partial blocks; capacity = configured token slots. Totals sum replicas, not unique "
            "tokens across workers. Pressure evictions are cumulative per fresh worker; explicit flushes "
            "are excluded. Zero decode residency after a request does not mean transferred KV was unused. "
            "Sampling does not capture every transient allocation during a request."
        ),
        "targets": [], "fieldConfig": {
            "defaults": {"unit": "short", "min": 0, "custom": {
                "drawStyle": "line", "lineWidth": 2, "showPoints": "always",
                "pointSize": 3, "spanNulls": False, "fillOpacity": 0,
            }}, "overrides": [],
        },
        "options": {"tooltip": {"mode": "multi"},
                    "legend": {"displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull"]}},
    } for index, title in enumerate((
        "Cache tokens - all workers", "Resident prefix-cache tokens / capacity - by worker",
        "Pressure-evicted tokens - by worker (cumulative)",
    ))]
    fields = ("resident_tokens", "allocated_tokens", "capacity_tokens", "evictable_tokens",
              "evicted_tokens_total", "cleared_tokens_total")
    for index, case in enumerate(report["cases"]):
        snapshots = case["cache_samples"]
        if not snapshots:
            raise ValueError("Empty cache-state recording")
        ids = [worker["worker_id"] for worker in case["workers"]]
        previous = None
        for sample in snapshots:
            if [w["worker_id"] for w in sample["workers"]] != ids:
                raise ValueError("Cache snapshot worker identity/order mismatch")
            for rank, worker in enumerate(sample["workers"]):
                if (not case["started_at_ms"] <= worker["observed_at_ms"] <= case["finished_at_ms"]
                        or not all(math.isfinite(worker[key]) and worker[key] >= 0 for key in fields)
                        or not worker["evictable_tokens"] <= worker["resident_tokens"]
                        <= worker["allocated_tokens"] <= worker["capacity_tokens"]):
                    raise ValueError("Invalid cache-state measurement")
                if previous is not None:
                    prior = previous["workers"][rank]
                    if (worker["observed_at_ms"] < prior["observed_at_ms"]
                            or worker["capacity_tokens"] != prior["capacity_tokens"]
                            or any(worker[key] < prior[key] for key in fields[-2:])):
                        raise ValueError("Cache counters, time or capacity changed inconsistently")
            previous = sample

        def add(panel, label, measurements, style):
            label = case_title(index, case) + " / " + label
            panel["targets"].append(csv_target(
                index, case, measurements, label=label, ref_id=f"S{len(panel['targets'])}",
            ))
            panel["fieldConfig"]["overrides"].append({
                "matcher": {"id": "byName", "options": label},
                "properties": [
                    {"id": "color", "value": {
                        "mode": "fixed", "fixedColor": METADATA_COLORS[case["metadata_enabled"]],
                    }}, {"id": "custom.lineStyle", "value": style},
                ],
            })

        for rank, field in enumerate(fields[:3]):
            add(panels[0], field.removesuffix("_tokens") + " tokens", [
                (max(w["observed_at_ms"] for w in sample["workers"]),
                 sum(w[field] for w in sample["workers"]))
                for sample in snapshots
            ], {"fill": "solid"} if rank == 0 else {"fill": "dash", "dash": [10 if rank == 1 else 2, 4]})
        for rank in range(len(ids)):
            role = "decode" if rank == len(ids) - 1 else "prefill"
            label = f"{role} worker {rank}"
            style = {"fill": "solid"} if rank == 0 else {"fill": "dash", "dash": [3 * rank, 3]}
            for field in ("resident_tokens", "capacity_tokens"):
                add(panels[1], label + " / " + field.removesuffix("_tokens"), [
                    (sample["workers"][rank]["observed_at_ms"], sample["workers"][rank][field])
                    for sample in snapshots
                ], style if field == "resident_tokens" else {"fill": "dash", "dash": [12, 6]})
            add(panels[2], label, [
                (sample["workers"][rank]["observed_at_ms"], sample["workers"][rank]["evicted_tokens_total"])
                for sample in snapshots
            ], style)
    return panels


def build_dashboard(report):
    if report.get("schema_version") == 3:
        if __package__:
            from .routing_load_dashboard import build_dashboard as build_load_dashboard
        else:
            from routing_load_dashboard import build_dashboard as build_load_dashboard
        return build_load_dashboard(report)
    validate_timestamps(report)
    start, end = report["started_at_ms"], report["finished_at_ms"]
    summary = report["summary"]
    text = (
        f"**Recorded run:** {iso_time(start)} to {iso_time(end)} (UTC). "
        "The graph time axis uses your browser timezone. **Blue = metadata ON; orange = OFF.** "
        "Color-blind-friendly Okabe-Ito colors; legends explicitly name each condition, "
        "and line styles distinguish workers/statistics.\n\n"
        "Request values and running statistics are plotted at recorded request completion times. "
        "Lines connect measured points only within a case; gaps are setup/teardown, not zero load. "
        "Hover phase markers for case windows and eviction checks. "
        "Seeds and pressure requests are excluded from the curves. "
        "The QPS row counts completed replay requests in one-second buckets, "
        "attributed to each worker by E2E completion time.\n\n"
        "**Source:** saved experiment report, not live Prometheus scrapes. Grafana's built-in "
        "CSV reader plots explicit recorded timestamps/values; no random data is generated. "
        "Dynamo is real; SGLang compute and K/V costs are simulated.\n\n"
        "| Condition | Requests | Token-weighted cache hits | TTFT p50 / p95 | E2E p50 / p95 / p99 | Modeled K/V time |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for key, title in (("metadata_on", "ON"), ("metadata_off", "OFF")):
        values = summary[key]
        e2e = e2e_percentiles(
            row["latency_ms"] for case in report["cases"]
            if case["metadata_enabled"] == (key == "metadata_on")
            for row in case["samples"] if row["stage"] == "replay"
        )
        text += (f"| {title} | {values['requests']} | {values['observed_hit_ratio']:.1%} | "
                 f"{values['ttft_p50_ms']:.1f} / {values['ttft_p95_ms']:.1f} ms | "
                 f"{e2e['p50']:.1f} / {e2e['p95']:.1f} / {e2e['p99']:.1f} ms | "
                 f"{values['kv_generation_seconds']:.3f} s |\n")
    text += (
        "\nE2E summary percentiles pool replay requests across cases in each condition. "
        "The percentile graph is cumulative within each case, not a rolling-time window. "
        "p50 is the median; p95/p99 use nearest rank. With fewer than 100 requests, "
        "**p99 equals the observed maximum** and is not a stable tail-latency estimate."
    )
    overview_height = 11
    cache = cache_panels(report, overview_height + 9)
    graphs_y = max(p["gridPos"]["y"] + p["gridPos"]["h"] for p in cache)
    panels = [{
        "id": 1, "type": "text", "title": "Recorded A/B experiment - time window and method",
        "gridPos": {"x": 0, "y": 0, "w": 24, "h": overview_height},
        "options": {"mode": "markdown", "content": text},
    }]
    definitions = [
        ("Observed prefix-cache hit ratio / request", "percentunit",
         lambda r: r["hit_tokens"] / r["input_tokens"]),
        ("Time to first token / request", "ms", lambda r: r["ttft_ms"]),
        ("Simulated K/V-generation time / request", "ms", lambda r: r["kv_seconds"] * 1000),
        ("End-to-end request latency", "ms", lambda r: r["latency_ms"]),
        ("Selected prefill worker (0 or 1)", "short", lambda r: r["selected_worker"]),
        ("Router-predicted cache hit ratio / request", "percentunit",
         lambda r: r["router_predicted_ratio"]),
    ]
    series = [
        (title, unit, [series_target(i, case, value) for i, case in enumerate(report["cases"])])
        for title, unit, value in definitions
    ]
    workers = sorted({
        row["selected_worker"] for case in report["cases"]
        for row in case["samples"] if row["stage"] == "replay"
    })
    worker_targets = []
    worker_overrides = []
    for i, case in enumerate(report["cases"]):
        for rank, worker in enumerate(workers):
            target = worker_ratio_target(i, case, worker)
            if target is None:
                continue
            worker_targets.append(target)
            worker_overrides.append({
                "matcher": {"id": "byName", "options": f"{case_title(i, case)} / worker {worker}"},
                "properties": [
                    {"id": "color", "value": {
                        "mode": "fixed", "fixedColor": METADATA_COLORS[case["metadata_enabled"]],
                    }},
                    {"id": "custom.lineStyle", "value":
                     {"fill": "solid"} if rank == 0 else {"fill": "dash", "dash": [3 * rank, 3]}},
                    {"id": "custom.lineWidth", "value": 2},
                ],
            })
    series.append(("Observed cache-hit ratio - all prefill workers", "percentunit", worker_targets))
    percentile_targets = []
    percentile_overrides = []
    for i, case in enumerate(report["cases"]):
        percentile_targets.extend(e2e_percentile_targets(i, case))
        for rank, percentile in enumerate(("p50", "p95", "p99")):
            percentile_overrides.append({
                "matcher": {"id": "byName", "options": f"{case_title(i, case)} / {percentile}"},
                "properties": [
                    {"id": "color", "value": {
                        "mode": "fixed", "fixedColor": METADATA_COLORS[case["metadata_enabled"]],
                    }},
                    {"id": "custom.lineStyle", "value":
                     {"fill": "solid"} if rank == 0 else {"fill": "dash", "dash": [9 if rank == 1 else 2, 3]}},
                    {"id": "custom.lineWidth", "value": 2},
                ],
            })
    series.append(("E2E latency - running p50 / p95 / p99 per case", "ms", percentile_targets))
    for index, (title, unit, targets) in enumerate(series):
        worker_panel = index == len(definitions)
        percentile_panel = index == len(definitions) + 1
        full_width = worker_panel or percentile_panel
        y = graphs_y + (index // 2) * 9
        if full_width:
            y = graphs_y + (math.ceil(len(definitions) / 2) + index - len(definitions)) * 9
        defaults = {
            "unit": unit, "min": 0,
            "custom": {"drawStyle": "line", "lineWidth": 1, "showPoints": "always",
                       "pointSize": 5, "spanNulls": False, "fillOpacity": 0},
        }
        if unit == "percentunit" or index == 4:
            defaults["max"] = 1
        panels.append({
            "id": index + 2, "type": "timeseries", "title": title,
            "description": (
                "One series per prefill worker and case: running sum of reused tokens / queried tokens, "
                "reset at each case. Replay requests only; seeds, eviction checks and transferred "
                "decode KV are excluded. A worker with no observations has no series, not a zero hit rate. "
                "Blue = metadata ON; orange = OFF. Solid/dashed lines distinguish workers. "
                "Worker numbers are logical aliases; each case uses fresh worker processes."
                if worker_panel else
                "E2E is measured from sending the request until its response stream completes. "
                "Each point aggregates all replay requests completed so far in that case, across workers. "
                "Resets per case; excludes seeding and eviction checks. p50 = median; p95/p99 = nearest rank. "
                "Solid = p50; long dash = p95; short dash = p99. Blue = metadata ON; orange = OFF. "
                "For fewer than 100 requests p99 equals the maximum; small samples do not establish a stable tail."
                if percentile_panel else
                "Recorded replay samples at actual completion timestamps; phase markers include setup."
            ),
            "datasource": DATASOURCE,
            "gridPos": {"x": 0 if full_width else (index % 2) * 12,
                        "y": y, "w": 24 if full_width else 12, "h": 9},
            "targets": targets,
            "fieldConfig": {
                "defaults": defaults,
                "overrides": worker_overrides if worker_panel else percentile_overrides if percentile_panel else [{
                    "matcher": {"id": "byName", "options": case_title(i, case)},
                    "properties": [{"id": "color", "value": {
                        "mode": "fixed", "fixedColor": METADATA_COLORS[case["metadata_enabled"]],
                    }}, {"id": "custom.lineStyle", "value":
                         {"fill": "solid"} if case["metadata_enabled"] else {"fill": "dash", "dash": [6, 3]}}],
                } for i, case in enumerate(report["cases"])],
            },
            "options": {"tooltip": {"mode": "multi"},
                        "legend": {"displayMode": "table" if full_width else "list",
                                   "placement": "bottom",
                                   "calcs": ["lastNotNull"] if full_width else []}},
        })
    panels.extend(qps_panels(report, overview_height))
    panels.extend(cache)
    return {
        "uid": "routing-ab-" + report["run_id"][:16],
        "title": "Dynamo A/B - " + iso_time(start),
        "schemaVersion": 41, "version": 1, "editable": True,
        "tags": ["routing-experiment", "recorded-measurements", "fake-sglang"],
        "timezone": "browser", "refresh": "",
        "time": {"from": iso_time(start - 2000), "to": iso_time(end + 2000)},
        "annotations": {"list": [{
            "builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
            "enable": True, "hide": False, "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Case windows and eviction checks", "type": "dashboard",
        }]},
        "links": [{"title": "Live serving lab (not this experiment)", "type": "link",
                   "url": GRAFANA + "/d/dynamo-local-sglang", "keepTime": False}],
        "panels": panels,
    }


def annotations_for(report):
    markers = []
    for index, case in enumerate(report["cases"]):
        title = case_title(index, case)
        tags = ["routing-experiment", report["run_id"],
                "metadata-on" if case["metadata_enabled"] else "metadata-off"]
        markers.append({
            "time": case["started_at_ms"], "timeEnd": case["finished_at_ms"],
            "text": title + " (includes worker setup, seeding, replay and eviction check)",
            "tags": tags,
        })
        for row in case["samples"]:
            if row["stage"] == "eviction":
                markers.append({
                    "time": row["finished_at_ms"],
                    "text": (f"{title}: after eviction/reseeding, selected worker {row['selected_worker']}; "
                             f"reused {row['hit_tokens']:g}/{row['input_tokens']:g} tokens"),
                    "tags": [*tags, "eviction-check"],
                })
        pressure = [sample for sample in case.get("cache_samples", [])
                    if sample["stage"] in ("before-pressure", "eviction")]
        if len(pressure) == 2:
            markers.append({
                "time": max(w["observed_at_ms"] for w in pressure[0]["workers"]),
                "timeEnd": max(w["observed_at_ms"] for w in pressure[1]["workers"]),
                "text": title + ": deliberate pressure and cache-owner relocation (excluded from replay QPS)",
                "tags": [*tags, "cache-pressure"],
            })
        if "load" in case:
            for offset in range(120, case["load"]["offered_duration_seconds"], 120):
                markers.append({
                    "time": case["load"]["started_at_ms"] + offset * 1000,
                    "text": f"{title}: offered hot set changes at {offset}s",
                    "tags": [*tags, "hot-set-change"],
                })
    return markers


def publish(report_path):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    dashboard = build_dashboard(report)
    (report_path.parent / "dashboard.json").write_text(json.dumps(dashboard, indent=2) + "\n")
    password = (ROOT / ".local/grafana-password").read_text().strip()
    authorization = "Basic " + base64.b64encode(("dynamo:" + password).encode()).decode()

    def api(method, path, body=None):
        request = urllib.request.Request(
            GRAFANA + path, method=method,
            headers={"Authorization": authorization, "Content-Type": "application/json"},
            data=json.dumps(body).encode() if body is not None else None,
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)

    try:
        datasource = api("GET", "/api/datasources/uid/" + DATASOURCE["uid"])
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        datasource = api("POST", "/api/datasources", {
            **DATASOURCE, "name": "Recorded routing experiments", "access": "proxy", "isDefault": False,
        })["datasource"]
    if datasource["type"] != DATASOURCE["type"]:
        raise ValueError("The routing-reports datasource UID belongs to a different datasource type")

    api("POST", "/api/dashboards/db", {"dashboard": dashboard, "overwrite": True})
    saved = api("GET", "/api/dashboards/uid/" + dashboard["uid"])
    dashboard_id = saved["dashboard"]["id"]
    existing = api("GET", f"/api/annotations?dashboardId={dashboard_id}&limit=100")
    for marker in annotations_for(report):
        if not any(m["time"] == marker["time"] and m["text"] == marker["text"] for m in existing):
            api("POST", "/api/annotations", {"dashboardId": dashboard_id, **marker})
    query = urllib.parse.urlencode({
        "from": report["started_at_ms"] - 2000, "to": report["finished_at_ms"] + 2000,
    })
    url = GRAFANA + saved["meta"]["url"] + "?" + query
    receipt = {"url": url, "dashboard_uid": dashboard["uid"],
               "started_at": iso_time(report["started_at_ms"]),
               "finished_at": iso_time(report["finished_at_ms"])}
    (report_path.parent / "dashboard-link.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Experiment window (UTC): {receipt['started_at']} to {receipt['finished_at']}")
    print("Recorded A/B dashboard:", url)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Timestamped report.json produced by routing-test")
    publish(parser.parse_args().report)
