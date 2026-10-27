"""Validate and summarize one focused Mooncake routing-matrix run."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path


EXPECTED_ROLES = ("frontend", "planner", "prefill", "decode")
REQUIRED_CASES = [
    {
        "workload": workload,
        "routing_cli_mode": mode,
        "routing_label": label,
        "case": f"{workload}-{'po2c' if mode == 'power-of-two' else mode}",
    }
    for mode, label in (
        ("kv", "KV-aware"),
        ("load-aware", "load-aware"),
        ("power-of-two", "PO2C"),
    )
    for workload in ("conversation", "toolagent")
]


def role_of(pod):
    name = pod["podRef"]["name"]
    if pod["podRef"]["namespace"] == "dynamo-sim":
        for role in EXPECTED_ROLES:
            if name.startswith(f"fake-sglang-{role}-") or name == (
                f"fake-sglang-{role}"
            ):
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
                key = (
                    role_of(pod),
                    pod["podRef"]["uid"],
                    container["name"],
                    container["startTime"],
                )
                timestamp = datetime.fromisoformat(
                    cpu["time"].replace("Z", "+00:00")
                ).timestamp()
                series[key][timestamp] = cpu["usageCoreNanoSeconds"] / 1e9
    return series


class CpuCounterRegression(ValueError):
    pass


def integrate(points, start, end):
    cpu_seconds = covered_seconds = 0.0
    values = sorted(points.items())
    for (time_0, counter_0), (time_1, counter_1) in zip(values, values[1:]):
        overlap = max(0, min(end, time_1) - max(start, time_0))
        if not overlap:
            continue
        if counter_1 < counter_0:
            raise CpuCounterRegression(
                "CPU counter decreased within one container incarnation: "
                f"{counter_0} at {time_0} -> {counter_1} at {time_1}"
            )
        cpu_seconds += (counter_1 - counter_0) * overlap / (time_1 - time_0)
        covered_seconds += overlap
    return cpu_seconds, covered_seconds


def cpu_report(series, start, end, replica_counts):
    groups = defaultdict(
        lambda: {
            "cpu_seconds": 0.0,
            "covered_container_seconds": 0.0,
            "container_incarnations": 0,
        }
    )
    for (role, *identity), points in series.items():
        try:
            cpu, covered = integrate(points, start, end)
        except CpuCounterRegression as error:
            groups[role].setdefault("invalid_cpu_series", []).append(
                {"container_identity": identity, "error": str(error)}
            )
            continue
        if covered:
            groups[role]["cpu_seconds"] += cpu
            groups[role]["covered_container_seconds"] += covered
            groups[role]["container_incarnations"] += 1
    duration = end - start
    for role, group in groups.items():
        group["mean_cores"] = group["cpu_seconds"] / duration
        group["cpu_percent_one_core"] = 100 * group["mean_cores"]
        expected = replica_counts.get(role)
        if expected:
            group["coverage_fraction"] = (
                group["covered_container_seconds"] / (duration * expected)
            )
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
                timestamp = datetime.fromisoformat(
                    memory["time"].replace("Z", "+00:00")
                ).timestamp()
                if not start <= timestamp <= end:
                    continue
                key = (pod["podRef"]["uid"], container["name"])
                mib = memory["workingSetBytes"] / 2**20
                batches[sample["collected_at_ns"]][role][key] = mib
                peaks[role][key] = max(peaks[role].get(key, 0), mib)
    result = {}
    for role, containers in peaks.items():
        totals = [
            sum(batch[role].values())
            for batch in batches.values()
            if len(batch.get(role, {})) == replicas[role]
        ]
        result[role] = {
            "peak_single_container_working_set_mib": max(containers.values()),
            "peak_observed_total_working_set_mib": max(totals) if totals else None,
            "complete_sample_batches": len(totals),
        }
    return result


def fleet_report(case):
    stability = json.loads((case / "stability.json").read_text())
    initial = json.loads((case / "initial-pods.json").read_text())["items"]
    final = json.loads((case / "final-pods.json").read_text())["items"]

    def identities(pod_list):
        return {
            (
                pod["metadata"]["uid"],
                status["name"],
                status["restartCount"],
            )
            for pod in pod_list
            for status in pod["status"].get("containerStatuses", [])
        }

    stable = (
        stability["stable"]
        and not stability["violations"]
        and identities(initial) == identities(final)
    )
    return {
        "stable": stable,
        "traffic_observations": stability["observations"],
        "violations": stability["violations"],
    }


def cache_values(path, pod_names):
    snapshot = json.loads(path.read_text())
    ranks = defaultdict(dict)
    for sample in snapshot.get("data", {}).get("result", []):
        labels = sample["metric"]
        if (
            labels.get("pod") not in pod_names
            or labels.get("dynamo_component_type") != "prefill"
        ):
            continue
        metric_name = labels.get("__name__", "")
        if "_hit_" in metric_name:
            field = "hit_tokens"
        elif "_query_" in metric_name:
            field = "query_tokens"
        else:
            continue
        key = (labels["pod"], labels["worker_id"], labels["dp_rank"])
        if field in ranks[key]:
            raise ValueError(f"Duplicate {field} counter for {key}: {path}")
        ranks[key][field] = float(sample["value"][1])
    return dict(ranks)


def cache_report(case, expected_ranks, fleet_stable):
    initial = json.loads((case / "initial-pods.json").read_text())["items"]
    prefill_names = {
        pod["metadata"]["name"]
        for pod in initial
        if (
            pod["metadata"].get("labels", {}).get(
                "nvidia.com/dynamo-component-type"
            )
            or pod["metadata"].get("labels", {}).get(
                "nvidia.com/dynamo-component"
            )
        )
        == "prefill"
    }
    before = cache_values(case / "cache-metrics-before.json", prefill_names)
    after = cache_values(case / "cache-metrics-after.json", prefill_names)
    required_fields = {"hit_tokens", "query_tokens"}
    if set(before) != set(after):
        raise ValueError(f"Cache counter rank identities changed: {case}")
    if len(before) != expected_ranks or any(
        set(values) != required_fields for values in [*before.values(), *after.values()]
    ):
        raise ValueError(
            f"Expected complete cache counters for {expected_ranks} ranks: {case}"
        )
    deltas = {}
    for rank in before:
        deltas[rank] = {
            field: after[rank][field] - before[rank][field]
            for field in required_fields
        }
        if any(value < 0 for value in deltas[rank].values()):
            raise ValueError(f"Cache counter regressed for {rank}: {case}")
    hits = sum(values["hit_tokens"] for values in deltas.values())
    queries = sum(values["query_tokens"] for values in deltas.values())
    if queries <= 0:
        raise ValueError(f"No per-run prefill query-token delta was observed: {case}")
    return {
        "fleet_stable": fleet_stable,
        "prefill_ranks_observed": len(deltas),
        "prefill_ranks_expected": expected_ranks,
        "rank_coverage_complete": len(deltas) == expected_ranks,
        "hit_tokens_delta": hits,
        "query_tokens_delta": queries,
        "token_hit_fraction": hits / queries if fleet_stable else None,
        "scope": (
            "direct before/after counter deltas from the newly recreated "
            "prefill pod identities"
        ),
    }


def summary_average(summary, name):
    value = summary.get(name)
    if not isinstance(value, dict) or "avg" not in value:
        raise ValueError(f"AIPerf summary is missing {name}.avg")
    return value["avg"]


def report(case):
    invocation = json.loads((case / "invocation.json").read_text())
    deployment = json.loads((case / "deployment.json").read_text())
    replicas = {
        component["name"].lower(): component["replicas"]
        for component in deployment["spec"]["components"]
    }
    ranks = {}
    for component in deployment["spec"]["components"]:
        role = component["name"].lower()
        if role not in ("prefill", "decode"):
            continue
        args = component["podTemplate"]["spec"]["containers"][0]["args"]
        ranks[role] = (
            int(args[args.index("--dp-size") + 1]) if "--dp-size" in args else 1
        )
    summary = json.loads(
        (case / "aiperf/profile_export_aiperf.json").read_text()
    )
    rows = [
        json.loads(line)
        for line in (case / "aiperf/profile_export.jsonl").open()
    ]
    rows = [
        row
        for row in rows
        if row["metadata"]["benchmark_phase"] == "profiling"
    ]
    metadata = [row["metadata"] for row in rows]
    starts = sorted(item["request_start_ns"] / 1e9 for item in metadata)
    credits = sorted(item["credit_issued_ns"] / 1e9 for item in metadata)
    start, last = starts[0], starts[-1]
    end = max(item["request_end_ns"] for item in metadata) / 1e9
    dispatch_lag = [
        (item["request_start_ns"] - item["credit_issued_ns"]) / 1e6
        for item in metadata
    ]
    schedule_lag = [
        (stamp - credits[0] - index / invocation["offered_rps"]) * 1000
        for index, stamp in enumerate(credits)
    ]
    expected = invocation["expected_requests"]
    completed = summary_average(summary, "completed_request_count")
    successful = summary_average(summary, "request_count")
    error_rate = summary_average(summary, "request_error_rate")
    errors = Counter(
        json.dumps(row.get("error"), sort_keys=True)
        for row in rows
        if row.get("error")
    )
    fleet = fleet_report(case)
    strict_completion = (
        summary.get("is_complete") is True
        and len(rows) == expected
        and completed == expected
        and successful == expected
        and error_rate == 0
        and not errors
        and summary.get("error_summary") in (None, {}, [], "")
        and fleet["stable"]
    )
    if not strict_completion:
        raise ValueError(f"Case failed strict completion validation: {case}")
    if last <= start or end <= start:
        raise ValueError(f"Case has an invalid request timing window: {case}")
    series = cpu_series(case / "cpu.jsonl")
    result = {
        "case": case.name,
        "workload": invocation["workload"],
        "routing": invocation["routing_label"],
        "routing_cli_mode": invocation["routing_cli_mode"],
        "offered_rps": invocation["offered_rps"],
        "h200_profile_from_application_defaults": invocation[
            "h200_profile_from_application_defaults"
        ],
        "replicas": replicas,
        "ranks_per_process": ranks,
        "logical_workers": {
            role: replicas[role] * count for role, count in ranks.items()
        },
        "expected": expected,
        "recorded": len(rows),
        "completed": completed,
        "successful": successful,
        "error_rate_pct": error_rate,
        "errors": dict(errors),
        "error_summary": summary.get("error_summary"),
        "is_complete": summary["is_complete"],
        "strict_completion": strict_completion,
        "fleet": fleet,
        "dispatch_window_seconds": last - start,
        "dispatch_rps": (len(starts) - 1) / (last - start),
        "request_window_seconds": end - start,
        "first_request_epoch": start,
        "last_request_end_epoch": end,
        "credit_to_start_ms": {
            "p50": percentile(dispatch_lag, 0.5),
            "p99": percentile(dispatch_lag, 0.99),
            "max": max(dispatch_lag),
        },
        "cumulative_schedule_lag_ms": {
            "p50": percentile(schedule_lag, 0.5),
            "p99": percentile(schedule_lag, 0.99),
            "max": max(schedule_lag),
        },
        "cpu_dispatch_window": cpu_report(series, start, last, replicas),
        "cpu_request_window": cpu_report(series, start, end, replicas),
        "memory_request_window": memory_report(
            case / "cpu.jsonl", start, end, replicas
        ),
        "latency_ms_successful": summary.get("request_latency"),
        "ttft_ms_successful": summary.get("time_to_first_token"),
        "output_length_mismatches": summary.get("osl_mismatch_count"),
        "input_length": summary.get("input_sequence_length"),
    }
    result["prefill_cache"] = cache_report(
        case,
        replicas["prefill"] * ranks["prefill"],
        fleet["stable"],
    )
    result["cpu_dispatch_coverage_complete"] = all(
        result["cpu_dispatch_window"].get(role, {}).get("coverage_fraction", 0)
        >= 0.99
        and not result["cpu_dispatch_window"].get(role, {}).get(
            "invalid_cpu_series"
        )
        for role in EXPECTED_ROLES
    )
    (case / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def statistic(metric, name):
    if not isinstance(metric, dict):
        return None
    for key in (name, name.upper(), name.replace("p", "")):
        if key in metric:
            return metric[key]
    return None


def display(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def markdown_report(matrix, results):
    profile = matrix["h200_profile_from_application_defaults"]
    topology = matrix["topology"]
    lines = [
        "# Mooncake routing matrix",
        "",
        (
            f"All {len(results)} cases passed strict AIPerf completion, zero-error, "
            "fresh-pod, and in-traffic fleet-stability validation."
        ),
        "",
        "## Configuration",
        "",
        (
            f"- Offered load: {matrix['offered_rps']} RPS; workloads: conversation "
            "and toolagent; routing: KV-aware, load-aware, and PO2C."
        ),
        (
            "- Topology: "
            f"{topology['frontend_processes']} frontend, "
            f"{topology['planner_processes']} Planner, "
            f"{topology['prefill_processes']} prefill processes x "
            f"{topology['ranks_per_worker_process']} ranks, "
            f"{topology['decode_processes']} decode processes x "
            f"{topology['ranks_per_worker_process']} ranks."
        ),
        (
            "- H200 application defaults per rank: "
            f"{profile['kv_blocks_per_rank']:,} blocks x "
            f"{profile['tokens_per_page']} tokens/page = "
            f"{profile['token_slots_per_rank']:,} token slots; "
            f"{profile['context_length']:,}-token context; "
            f"{profile['kv_cache_dtype']} accounting; "
            f"{profile['max_running_requests_per_rank']} max running requests."
        ),
        "",
        "## Results",
        "",
        (
            "| Workload | Routing | Successful | Errors | Dispatch RPS | "
            "Cache hit | Latency p99 ms | TTFT p99 ms | Stable |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result["workload"],
                    result["routing"],
                    str(result["successful"]),
                    display(result["error_rate_pct"]),
                    display(result["dispatch_rps"]),
                    display(
                        100 * result["prefill_cache"]["token_hit_fraction"]
                        if result["prefill_cache"]["token_hit_fraction"] is not None
                        else None
                    )
                    + "%",
                    display(statistic(result["latency_ms_successful"], "p99")),
                    display(statistic(result["ttft_ms_successful"], "p99")),
                    "yes" if result["fleet"]["stable"] else "no",
                ]
            )
            + " |"
        )
    if any(not result["cpu_dispatch_coverage_complete"] for result in results):
        lines.extend(
            [
                "",
                (
                    "CPU averages with less than 99% container-time coverage are "
                    "retained in JSON but must not be used for cross-case conclusions."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    matrix_path = args.directory / "matrix.json"
    if not matrix_path.is_file():
        raise ValueError("matrix.json is missing")
    matrix = json.loads(matrix_path.read_text())
    if not matrix.get("cases_finished") and not matrix.get("comparison_complete"):
        raise ValueError("The benchmark matrix has not finished all cases")
    if matrix.get("expected_cases") != REQUIRED_CASES:
        raise ValueError("The benchmark is not the required two-by-three routing matrix")
    expected_cases = [item["case"] for item in matrix["expected_cases"]]
    case_paths = [
        path.parent.parent
        for path in args.directory.glob("*/aiperf/profile_export_aiperf.json")
    ]
    actual_cases = sorted(path.name for path in case_paths)
    if actual_cases != sorted(expected_cases):
        raise ValueError(
            f"Expected completed cases {sorted(expected_cases)}, got {actual_cases}"
        )
    by_name = {path.name: path for path in case_paths}
    results = [report(by_name[name]) for name in expected_cases]
    summary = {
        "matrix_complete": True,
        "offered_rps": matrix["offered_rps"],
        "h200_profile_from_application_defaults": matrix[
            "h200_profile_from_application_defaults"
        ],
        "topology": matrix["topology"],
        "cases": results,
    }
    (args.directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.directory / "report.md").write_text(markdown_report(matrix, results))
    for row in results:
        print(
            row["workload"],
            row["routing"],
            "successful",
            row["successful"],
            "errors %",
            row["error_rate_pct"],
            "dispatch RPS",
            round(row["dispatch_rps"], 2),
        )


if __name__ == "__main__":
    main()
