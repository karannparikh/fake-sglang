"""Summarize AIPerf accounting, kubelet CPU/memory, and per-rank cache reuse."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path


def role_of(pod):
    name = pod["podRef"]["name"]
    if pod["podRef"]["namespace"] == "dynamo-sim":
        for role in ("frontend", "prefill", "decode", "planner"):
            if name.startswith(f"fake-sglang-{role}-") or name == f"fake-sglang-{role}":
                return role
    return "other-containers"


def cpu_series(path):
    series = defaultdict(dict)
    for line in path.open():
        for pod in json.loads(line)["pods"]:
            for container in pod["containers"]:
                cpu = container.get("cpu", {})
                if "usageCoreNanoSeconds" not in cpu:
                    continue
                key = (role_of(pod), pod["podRef"]["uid"], container["name"], container["startTime"])
                timestamp = datetime.fromisoformat(cpu["time"].replace("Z", "+00:00")).timestamp()
                series[key][timestamp] = cpu["usageCoreNanoSeconds"] / 1e9
    return series


class CpuCounterRegression(ValueError):
    pass


def integrate(points, start, end):
    cpu_seconds = covered_seconds = 0.0
    values = sorted(points.items())
    for (t0, c0), (t1, c1) in zip(values, values[1:]):
        overlap = max(0, min(end, t1) - max(start, t0))
        if overlap:
            if c1 < c0:
                raise CpuCounterRegression(
                    f"CPU counter decreased within one container incarnation: "
                    f"{c0} at {t0} -> {c1} at {t1}"
                )
            cpu_seconds += (c1 - c0) * overlap / (t1 - t0)
            covered_seconds += overlap
    return cpu_seconds, covered_seconds


def cpu_report(series, start, end, replica_counts):
    groups = defaultdict(lambda: {"cpu_seconds": 0.0, "covered_container_seconds": 0.0,
                                  "container_incarnations": 0})
    for (role, *identity), points in series.items():
        try:
            cpu, covered = integrate(points, start, end)
        except CpuCounterRegression as error:
            groups[role].setdefault("invalid_cpu_series", []).append({
                "container_identity": identity, "error": str(error),
            })
            continue
        if covered:
            groups[role]["cpu_seconds"] += cpu
            groups[role]["covered_container_seconds"] += covered
            groups[role]["container_incarnations"] += 1
    for role, group in groups.items():
        group["mean_cores"] = group["cpu_seconds"] / (end - start)
        group["cpu_percent_one_core"] = 100 * group["mean_cores"]
        expected = replica_counts.get(role)
        if expected:
            group["coverage_fraction"] = group["covered_container_seconds"] / ((end - start) * expected)
        if group.get("invalid_cpu_series"):
            group["cpu_seconds"] = None
            group["mean_cores"] = None
            group["cpu_percent_one_core"] = None
    return dict(groups)


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def memory_report(path, start, end, replicas):
    batches = defaultdict(lambda: defaultdict(dict))
    peaks = defaultdict(dict)
    for line in path.open():
        sample = json.loads(line)
        for pod in sample["pods"]:
            role = role_of(pod)
            if role not in replicas:
                continue
            for container in pod["containers"]:
                memory = container.get("memory", {})
                if "workingSetBytes" not in memory or "time" not in memory:
                    continue
                timestamp = datetime.fromisoformat(memory["time"].replace("Z", "+00:00")).timestamp()
                if not start <= timestamp <= end:
                    continue
                key = (pod["podRef"]["uid"], container["name"])
                mib = memory["workingSetBytes"] / 2**20
                batches[sample["collected_at_ns"]][role][key] = mib
                peaks[role][key] = max(peaks[role].get(key, 0), mib)
    result = {}
    for role, containers in peaks.items():
        totals = [sum(batch[role].values()) for batch in batches.values()
                  if len(batch.get(role, {})) == replicas[role]]
        result[role] = {
            "peak_single_container_working_set_mib": max(containers.values()),
            "peak_observed_total_working_set_mib": max(totals) if totals else None,
            "complete_sample_batches": len(totals),
        }
    return result


def cache_report(case, expected_ranks=None):
    path = case / "cache-metrics.json"
    if not path.exists():
        return None
    initial = json.loads((case / "initial-pods.json").read_text())["items"]
    final = json.loads((case / "final-pods.json").read_text())["items"]

    def identities(pods):
        return {(p["metadata"]["uid"], s["name"], s["restartCount"])
                for p in pods for s in p["status"].get("containerStatuses", [])}

    stable = identities(initial) == identities(final)
    names = {p["metadata"]["name"] for p in initial}
    ranks = defaultdict(dict)
    for sample in json.loads(path.read_text())["data"]["result"]:
        labels = sample["metric"]
        if labels["pod"] not in names or labels["dynamo_component_type"] != "prefill":
            continue
        key = (labels["worker_id"], labels["dp_rank"])
        field = "hit_tokens" if "_hit_" in labels["__name__"] else "query_tokens"
        ranks[key][field] = float(sample["value"][1])
    if not ranks or any(set(v) != {"hit_tokens", "query_tokens"} for v in ranks.values()):
        raise ValueError(f"Incomplete per-rank cache counters: {case}")
    hits = sum(v["hit_tokens"] for v in ranks.values())
    queries = sum(v["query_tokens"] for v in ranks.values())
    complete = expected_ranks is None or len(ranks) == expected_ranks
    return {
        "fleet_stable": stable, "prefill_ranks_observed": len(ranks),
        "prefill_ranks_expected": expected_ranks, "rank_coverage_complete": complete,
        "hit_tokens": hits, "query_tokens": queries,
        "token_hit_fraction": hits / queries if stable and complete and queries else None,
        "scope": "admitted prefill requests; cold-reset counters, not all offered traffic",
    }


def report(case):
    invocation = json.loads((case / "invocation.json").read_text())
    deployment = json.loads((case / "deployment.json").read_text())
    replicas = {c["name"].lower(): c["replicas"] for c in deployment["spec"]["components"]}
    ranks = {}
    for component in deployment["spec"]["components"]:
        role = component["name"].lower()
        if role in ("prefill", "decode"):
            args = component["podTemplate"]["spec"]["containers"][0]["args"]
            ranks[role] = int(args[args.index("--dp-size") + 1]) if "--dp-size" in args else 1
    summary = json.loads((case / "aiperf/profile_export_aiperf.json").read_text())
    rows = [json.loads(line) for line in (case / "aiperf/profile_export.jsonl").open()]
    rows = [r for r in rows if r["metadata"]["benchmark_phase"] == "profiling"]
    metadata = [row["metadata"] for row in rows]
    starts = sorted(m["request_start_ns"] / 1e9 for m in metadata)
    start, last = starts[0], starts[-1]
    end = max(m["request_end_ns"] for m in metadata) / 1e9
    credits = sorted(m["credit_issued_ns"] / 1e9 for m in metadata)
    dispatch_lag = [(m["request_start_ns"] - m["credit_issued_ns"]) / 1e6 for m in metadata]
    schedule_lag = [(stamp - credits[0] - index / invocation["offered_rps"]) * 1000
                    for index, stamp in enumerate(credits)]
    series = cpu_series(case / "cpu.jsonl")
    errors = Counter(json.dumps(row.get("error"), sort_keys=True) for row in rows if row.get("error"))
    result = {
        "case": case.name,
        "replicas": replicas,
        "ranks_per_process": ranks,
        "logical_workers": {role: replicas[role] * count for role, count in ranks.items()},
        "expected": invocation["expected_requests"],
        "recorded": len(rows),
        "unique_conversations": len({m["conversation_id"] for m in metadata}),
        "completed": summary["completed_request_count"]["avg"],
        "successful": summary["request_count"]["avg"],
        "error_rate_pct": summary["request_error_rate"]["avg"],
        "errors": dict(errors),
        "error_summary": summary["error_summary"],
        "is_complete": summary["is_complete"],
        "dispatch_window_seconds": last - start,
        "dispatch_rps": (len(starts) - 1) / (last - start),
        "request_window_seconds": end - start,
        "first_request_epoch": start,
        "last_request_end_epoch": end,
        "credit_to_start_ms": {
            "p50": percentile(dispatch_lag, .5), "p99": percentile(dispatch_lag, .99),
            "max": max(dispatch_lag),
        },
        "cumulative_schedule_lag_ms": {
            "p50": percentile(schedule_lag, .5), "p99": percentile(schedule_lag, .99),
            "max": max(schedule_lag),
        },
        "cpu_dispatch_window": cpu_report(series, start, last, replicas),
        "cpu_request_window": cpu_report(series, start, end, replicas),
        "memory_request_window": memory_report(case / "cpu.jsonl", start, end, replicas),
        "prefill_cache": cache_report(
            case, replicas.get("prefill", 0) * ranks.get("prefill", 1),
        ),
        "latency_ms_successful": summary.get("request_latency"),
        "ttft_ms_successful": summary.get("time_to_first_token"),
        "output_length_mismatches": summary.get("osl_mismatch_count"),
        "input_length": summary.get("input_sequence_length"),
    }
    result["cpu_dispatch_coverage_complete"] = all(
        result["cpu_dispatch_window"].get(role, {}).get("coverage_fraction", 0) >= .99
        and not result["cpu_dispatch_window"].get(role, {}).get("invalid_cpu_series")
        for role in ("frontend", "prefill", "decode")
    )
    if result["recorded"] != result["expected"] or result["unique_conversations"] != result["expected"]:
        raise ValueError(f"Trace coverage mismatch: {case}")
    (case / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    cases = sorted(p.parent.parent for p in args.directory.glob("*/aiperf/profile_export_aiperf.json"))
    if not cases:
        raise ValueError("No completed AIPerf case artifacts found")
    results = [report(case) for case in cases]
    (args.directory / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    for row in results:
        print(row["case"], "sent", row["recorded"], "successful", row["successful"],
              "errors %", round(row["error_rate_pct"], 2), "dispatch RPS", round(row["dispatch_rps"], 2))
        if not row["cpu_dispatch_coverage_complete"]:
            print("  WARNING: CPU coverage is incomplete; these are not full-run CPU averages.")
        print("  CPU cores during dispatch:", {
            role: round(cpu["mean_cores"], 3) if cpu["mean_cores"] is not None else None
            for role, cpu in row["cpu_dispatch_window"].items()
        })
        for window in ("cpu_dispatch_window", "cpu_request_window"):
            for role, cpu in row[window].items():
                for invalid in cpu.get("invalid_cpu_series", []):
                    print(f"  WARNING: {window} {role} CPU aggregate unavailable: {invalid}")


if __name__ == "__main__":
    main()
