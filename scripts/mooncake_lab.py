"""Run the validated Mooncake routing matrix against the dedicated local lab."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".local"
KUBECTL = [
    "kubectl",
    "--kubeconfig",
    str(STATE / "kubeconfig"),
    "--context",
    "kind-dynamo-sim",
    "--request-timeout=30s",
]
SELECTOR = "nvidia.com/dynamo-graph-deployment-name=fake-sglang"
NODES = ("dynamo-sim-control-plane", "dynamo-sim-worker", "dynamo-sim-worker2")
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
EXPECTED_REPLICAS = {"frontend": 1, "planner": 1, "prefill": 4, "decode": 4}
EXPECTED_RANKS_PER_PROCESS = 10
EXPECTED_CONTEXT = 1_048_576
SETTLE_SECONDS = 15
WORKLOADS = ("conversation", "toolagent")
ROUTING_LABELS = {
    "kv": "KV-aware",
    "load-aware": "load-aware",
    "power-of-two": "PO2C",
}
H200_PROFILE = {
    "kv_blocks_per_rank": 15_509,
    "tokens_per_page": 256,
    "token_slots_per_rank": 3_970_304,
    "context_length": EXPECTED_CONTEXT,
    "kv_cache_dtype": "fp8_e4m3",
    "max_running_requests_per_rank": 32,
}


def kube(*args):
    return json.loads(subprocess.check_output([*KUBECTL, *args], text=True))


def graph():
    return kube("-n", "dynamo-sim", "get", "dgd", "fake-sglang", "-o", "json")


def pods():
    return kube(
        "-n",
        "dynamo-sim",
        "get",
        "pods",
        "-l",
        SELECTOR,
        "-o",
        "json",
    )


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def pod_role(pod):
    labels = pod["metadata"].get("labels", {})
    role = labels.get("nvidia.com/dynamo-component-type")
    if not role:
        role = labels.get("nvidia.com/dynamo-component")
    return role.lower() if role else "unknown"


def pod_ready(pod):
    return (
        pod.get("status", {}).get("phase") == "Running"
        and any(
            condition["type"] == "Ready" and condition["status"] == "True"
            for condition in pod.get("status", {}).get("conditions", [])
        )
        and all(
            status.get("ready") is True
            for status in pod.get("status", {}).get("containerStatuses", [])
        )
        and bool(pod.get("status", {}).get("containerStatuses"))
    )


def fleet_observation(pod_list):
    return sorted(
        [
            {
                "name": pod["metadata"]["name"],
                "uid": pod["metadata"]["uid"],
                "role": pod_role(pod),
                "ready": pod_ready(pod),
                "deleting": "deletionTimestamp" in pod["metadata"],
                "containers": sorted(
                    [
                        {
                            "name": status["name"],
                            "restart_count": status["restartCount"],
                        }
                        for status in pod.get("status", {}).get(
                            "containerStatuses", []
                        )
                    ],
                    key=lambda item: item["name"],
                ),
            }
            for pod in pod_list
        ],
        key=lambda item: item["name"],
    )


def validate_pods(pod_list, old_uids=()):
    observation = fleet_observation(pod_list)
    counts = Counter(item["role"] for item in observation)
    if dict(counts) != EXPECTED_REPLICAS:
        raise RuntimeError(
            f"Expected exact pod counts {EXPECTED_REPLICAS}, observed {dict(counts)}"
        )
    if any(not item["ready"] or item["deleting"] for item in observation):
        raise RuntimeError(f"Serving fleet is not fully ready and stable: {observation}")
    stale = set(old_uids).intersection(item["uid"] for item in observation)
    if stale:
        raise RuntimeError(f"Pod recreation reused stale UIDs: {sorted(stale)}")
    return observation


def validate_deployment(deployment):
    raw_components = deployment["spec"]["components"]
    components = {
        component["type"].lower(): component
        for component in raw_components
    }
    if len(raw_components) != len(EXPECTED_REPLICAS) or set(components) != set(
        EXPECTED_REPLICAS
    ):
        raise RuntimeError(
            f"Expected one component for each role {sorted(EXPECTED_REPLICAS)}"
        )
    replicas = {
        role: component["replicas"] for role, component in components.items()
    }
    if replicas != EXPECTED_REPLICAS:
        raise RuntimeError(
            f"Expected exact deployment topology {EXPECTED_REPLICAS}, observed {replicas}"
        )
    inherited = {
        "--context-length",
        "--page-size",
        "--fake-num-kv-blocks",
        "--kv-cache-dtype",
        "--max-running-requests",
    }
    for role in ("prefill", "decode"):
        args = components[role]["podTemplate"]["spec"]["containers"][0]["args"]
        if "--dp-size" not in args:
            raise RuntimeError(f"{role} is missing --dp-size")
        ranks = int(args[args.index("--dp-size") + 1])
        if ranks != EXPECTED_RANKS_PER_PROCESS:
            raise RuntimeError(
                f"Expected {EXPECTED_RANKS_PER_PROCESS} ranks per {role} process, got {ranks}"
            )
        overridden = sorted(inherited.intersection(args))
        if overridden:
            raise RuntimeError(
                f"{role} must inherit the merged H200 defaults, but overrides {overridden}"
            )
    return components


def validate_runtime_image(components):
    image_path = STATE / "image"
    if not image_path.is_file():
        raise RuntimeError("No content-addressed lab image is recorded; run lab.sh up")
    expected = image_path.read_text().strip()
    actual = {
        components[role]["podTemplate"]["spec"]["containers"][0]["image"]
        for role in ("frontend", "prefill", "decode")
    }
    if actual != {expected}:
        raise RuntimeError(
            f"Deployment images {sorted(actual)} do not match recorded image {expected}"
        )
    if not re.fullmatch(
        r"fake-sglang-dynamo:1\.6\.0-local-[0-9a-f]{16}", expected
    ):
        raise RuntimeError(f"Lab image is not content-addressed: {expected}")
    return expected


def routing_args(args, mode):
    if mode not in ROUTING_LABELS:
        raise ValueError(f"Unknown routing mode: {mode}")
    result = list(args)
    router_mode = "power-of-two" if mode == "power-of-two" else "kv"
    if "--router-mode" in result:
        result[result.index("--router-mode") + 1] = router_mode
    else:
        result.extend(["--router-mode", router_mode])
    if mode == "power-of-two":
        return [
            arg
            for arg in result
            if arg not in ("--router-kv-events", "--load-aware")
        ]
    wanted = "--router-kv-events" if mode == "kv" else "--load-aware"
    unwanted = "--load-aware" if mode == "kv" else "--router-kv-events"
    result = [arg for arg in result if arg != unwanted]
    return result if wanted in result else result + [wanted]


def graph_ready(deployment):
    return any(
        condition["type"] == "Ready"
        and condition["status"] == "True"
        and condition.get("observedGeneration")
        == deployment["metadata"]["generation"]
        for condition in deployment.get("status", {}).get("conditions", [])
    )


def wait_ready(old_uids=()):
    deadline = time.monotonic() + 420
    last_error = "deployment has not been observed"
    while time.monotonic() < deadline:
        try:
            deployment = graph()
            validate_deployment(deployment)
            if not graph_ready(deployment):
                raise RuntimeError("DynamoGraphDeployment is not Ready")
            current = pods()
            observation = validate_pods(current["items"], old_uids)
            return current, observation
        except (KeyError, RuntimeError, subprocess.SubprocessError) as error:
            last_error = str(error)
            time.sleep(3)
    raise RuntimeError(f"Local lab did not become ready: {last_error}")


def set_args(arguments):
    current = graph()
    validate_deployment(current)
    changes = [
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": current["metadata"]["resourceVersion"],
        }
    ]
    for index, component in enumerate(current["spec"]["components"]):
        if component["name"] not in arguments:
            continue
        args = arguments[component["name"]]
        if args != component["podTemplate"]["spec"]["containers"][0]["args"]:
            changes.append(
                {
                    "op": "replace",
                    "path": (
                        f"/spec/components/{index}/podTemplate/spec/"
                        "containers/0/args"
                    ),
                    "value": args,
                }
            )
    if len(changes) > 1:
        subprocess.run(
            [
                *KUBECTL,
                "-n",
                "dynamo-sim",
                "patch",
                "dgd",
                "fake-sglang",
                "--type=json",
                "-p",
                json.dumps(changes),
            ],
            check=True,
        )
    wait_ready()


def validate_model():
    with urllib.request.urlopen(
        "http://127.0.0.1:8000/v1/models", timeout=10
    ) as response:
        models = json.load(response)
    matching = [model for model in models["data"] if model["id"] == "fake-model"]
    if len(matching) != 1 or matching[0].get("context_window") != EXPECTED_CONTEXT:
        raise RuntimeError(
            f"Expected fake-model with context {EXPECTED_CONTEXT}, observed {models}"
        )


def cold_reset():
    current = pods()
    previous = validate_pods(current["items"])
    old_uids = {item["uid"] for item in previous}
    subprocess.run(
        [
            *KUBECTL,
            "-n",
            "dynamo-sim",
            "delete",
            "pod",
            "--wait=false",
            *[item["name"] for item in previous],
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    fresh_pods, fresh = wait_ready(old_uids)
    time.sleep(SETTLE_SECONDS)
    settled_pods = pods()
    settled = validate_pods(settled_pods["items"], old_uids)
    if settled != fresh:
        raise RuntimeError("Serving fleet changed during the 15-second settle period")
    validate_model()
    return fresh_pods, fresh


def sample_runtime(stop, cpu_path, stability_path):
    with (
        cpu_path.open("w") as cpu_output,
        stability_path.open("w") as stability_output,
        ThreadPoolExecutor(max_workers=3) as pool,
    ):
        while True:
            collected_at_ns = time.time_ns()

            def read_node(node):
                result = kube(
                    "get",
                    "--raw",
                    f"/api/v1/nodes/{node}/proxy/stats/summary",
                )
                return {
                    "node": node,
                    "collected_at_ns": collected_at_ns,
                    "pods": result["pods"],
                }

            for sample in pool.map(read_node, NODES):
                cpu_output.write(json.dumps(sample) + "\n")
            stability_output.write(
                json.dumps(
                    {
                        "collected_at_ns": collected_at_ns,
                        "pods": fleet_observation(pods()["items"]),
                    }
                )
                + "\n"
            )
            cpu_output.flush()
            stability_output.flush()
            if stop.wait(5):
                return


def validate_stability(case, expected):
    observations = [
        json.loads(line)
        for line in (case / "pod-stability.jsonl").read_text().splitlines()
        if line
    ]
    if not observations:
        raise RuntimeError("No serving-fleet observations were recorded during traffic")
    violations = [
        observation
        for observation in observations
        if observation["pods"] != expected
    ]
    final_pods = pods()
    final = validate_pods(final_pods["items"])
    if final != expected:
        violations.append({"collected_at_ns": time.time_ns(), "pods": final})
    result = {
        "stable": not violations,
        "observations": len(observations),
        "violations": violations,
    }
    write_json(case / "stability.json", result)
    write_json(case / "final-pods.json", final_pods)
    if violations:
        raise RuntimeError(f"Serving fleet changed during traffic: {case}")
    return final_pods


def cache_rank_fields(snapshot, pod_names):
    ranks = defaultdict(set)
    for sample in snapshot.get("data", {}).get("result", []):
        labels = sample["metric"]
        if (
            labels.get("pod") not in pod_names
            or labels.get("dynamo_component_type") != "prefill"
        ):
            continue
        name = labels.get("__name__", "")
        if "_hit_" in name:
            field = "hit_tokens"
        elif "_query_" in name:
            field = "query_tokens"
        else:
            continue
        ranks[
            (
                labels["pod"],
                labels["worker_id"],
                labels["dp_rank"],
            )
        ].add(field)
    return ranks


def query_cache_metrics(pod_names):
    pod_expression = "|".join(re.escape(name) for name in sorted(pod_names))
    query = (
        '{__name__=~"sglang:simulated_prefix_cache_(query|hit)_tokens_total",'
        f'pod=~"{pod_expression}"'
        "}"
    )
    url = "http://127.0.0.1:9090/api/v1/query?" + urllib.parse.urlencode(
        {"query": query}
    )
    with urllib.request.urlopen(url, timeout=15) as response:
        snapshot = json.load(response)
    if snapshot.get("status") != "success":
        raise RuntimeError(f"Cache metrics query failed: {snapshot}")
    return snapshot


def wait_cache_snapshot(pod_names, expected_ranks):
    deadline = time.monotonic() + 90
    last = {}
    while time.monotonic() < deadline:
        last = query_cache_metrics(pod_names)
        fields = cache_rank_fields(last, pod_names)
        if len(fields) == expected_ranks and all(
            value == {"hit_tokens", "query_tokens"} for value in fields.values()
        ):
            return last
        time.sleep(5)
    raise RuntimeError(
        f"Prometheus did not expose complete counters for {expected_ranks} "
        f"prefill ranks; observed {cache_rank_fields(last, pod_names)}"
    )


def summary_average(summary, name):
    value = summary.get(name)
    if not isinstance(value, dict) or "avg" not in value:
        raise RuntimeError(f"AIPerf summary is missing {name}.avg")
    return value["avg"]


def validate_aiperf(case, expected_requests):
    summary = json.loads(
        (case / "aiperf/profile_export_aiperf.json").read_text()
    )
    completed = summary_average(summary, "completed_request_count")
    successful = summary_average(summary, "request_count")
    error_rate = summary_average(summary, "request_error_rate")
    if summary.get("is_complete") is not True:
        raise RuntimeError("AIPerf did not mark the profile complete")
    if completed != expected_requests or successful != expected_requests:
        raise RuntimeError(
            f"AIPerf accounting mismatch: expected {expected_requests}, "
            f"completed {completed}, successful {successful}"
        )
    if error_rate != 0 or summary.get("error_summary") not in (None, {}, [], ""):
        raise RuntimeError(
            f"AIPerf reported request errors: rate={error_rate}, "
            f"summary={summary.get('error_summary')}"
        )
    mismatches = summary.get("osl_mismatch_count")
    if isinstance(mismatches, dict) and mismatches.get("avg", 0) != 0:
        raise RuntimeError(f"AIPerf reported output-length mismatches: {mismatches}")
    rows = [
        json.loads(line)
        for line in (case / "aiperf/profile_export.jsonl").read_text().splitlines()
        if line
    ]
    rows = [
        row
        for row in rows
        if row.get("metadata", {}).get("benchmark_phase") == "profiling"
    ]
    if len(rows) != expected_requests:
        raise RuntimeError(
            f"AIPerf trace has {len(rows)} profiling rows; expected {expected_requests}"
        )
    errors = [row.get("error") for row in rows if row.get("error")]
    if errors:
        raise RuntimeError(f"AIPerf trace contains request errors: {errors[:3]}")
    for row in rows:
        metadata = row["metadata"]
        if metadata["request_end_ns"] < metadata["request_start_ns"]:
            raise RuntimeError("AIPerf recorded a request ending before it started")
    return summary


def run_case(
    trace,
    mode,
    destination,
    original,
    request_rate,
    trace_dir,
    aiperf,
):
    routing_label = ROUTING_LABELS[mode]
    case_suffix = "po2c" if mode == "power-of-two" else mode
    case = destination / f"{trace}-{case_suffix}"
    case.mkdir()
    set_args(
        {
            name: routing_args(args, mode)
            for name, args in original.items()
        }
    )
    initial_pods, initial_fleet = cold_reset()
    deployment = graph()
    validate_deployment(deployment)
    write_json(case / "deployment.json", deployment)
    write_json(case / "initial-pods.json", initial_pods)
    prefill_names = {
        item["name"] for item in initial_fleet if item["role"] == "prefill"
    }
    expected_ranks = EXPECTED_REPLICAS["prefill"] * EXPECTED_RANKS_PER_PROCESS
    write_json(
        case / "cache-metrics-before.json",
        wait_cache_snapshot(prefill_names, expected_ranks),
    )
    source = trace_dir / f"{trace}_trace.jsonl"
    with source.open() as lines:
        count = sum(1 for _ in lines)
    if count <= 0:
        raise RuntimeError(f"Trace is empty: {source}")
    command = [
        str(aiperf),
        "profile",
        "--model",
        "fake-model",
        "--url",
        "http://127.0.0.1:8000",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--tokenizer",
        "Qwen/Qwen3-0.6B",
        "--tokenizer-revision",
        TOKENIZER_REVISION,
        "--input-file",
        str(source),
        "--custom-dataset-type",
        "mooncake_trace",
        "--no-fixed-schedule",
        "--request-count",
        str(count),
        "--num-dataset-entries",
        str(count),
        "--dataset-sampling-strategy",
        "sequential",
        "--request-rate",
        str(request_rate),
        "--arrival-pattern",
        "constant",
        "--isl-block-size",
        "512",
        "--use-server-token-count",
        "--extra-inputs",
        '{"temperature":0,"ignore_eos":true}',
        "--request-timeout-seconds",
        "120",
        "--random-seed",
        "42",
        "--workers-max",
        "4",
        "--ui-type",
        "none",
        "--artifact-dir",
        str(case / "aiperf"),
    ]
    write_json(
        case / "invocation.json",
        {
            "command": command,
            "trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "expected_requests": count,
            "workload": trace,
            "routing_cli_mode": mode,
            "routing_label": routing_label,
            "offered_rps": request_rate,
            "cold_cache": True,
            "fresh_pods": True,
            "settle_seconds": SETTLE_SECONDS,
            "original_timestamps_replaced": True,
            "h200_profile_from_application_defaults": H200_PROFILE,
        },
    )
    env = {
        **os.environ,
        "HF_HOME": str(STATE / "aiperf-hf"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": "42",
        "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "900",
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "960",
    }
    write_json(
        case / "aiperf-environment.json",
        {
            key: env[key]
            for key in (
                "AIPERF_DATASET_CONFIGURATION_TIMEOUT",
                "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT",
                "HF_HOME",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "TOKENIZERS_PARALLELISM",
                "PYTHONHASHSEED",
            )
        },
    )
    stop = threading.Event()
    print(
        f"Starting {trace} {routing_label}: {count} requests at "
        f"{request_rate} RPS; {case}",
        flush=True,
    )
    with (
        ThreadPoolExecutor(max_workers=1) as collector,
        (case / "console.log").open("w") as log,
    ):
        sampling = collector.submit(
            sample_runtime,
            stop,
            case / "cpu.jsonl",
            case / "pod-stability.jsonl",
        )
        try:
            result = subprocess.run(
                ["/usr/bin/time", "-l", *command],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=max(1800, 960 + count / request_rate + 240),
            )
        finally:
            time.sleep(15)
            stop.set()
            sampling.result()
    write_json(case / "exit.json", {"returncode": result.returncode})
    validate_stability(case, initial_fleet)
    write_json(
        case / "cache-metrics-after.json",
        wait_cache_snapshot(prefill_names, expected_ranks),
    )
    if result.returncode != 0:
        raise RuntimeError(f"AIPerf failed; see {case / 'console.log'}")
    summary = validate_aiperf(case, count)
    print(
        f"Finished {trace} {routing_label}: "
        f"{summary_average(summary, 'request_count')} successful requests",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-rate", type=float, default=20)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=STATE / "mooncake-100rps",
    )
    parser.add_argument(
        "--aiperf",
        type=Path,
        default=STATE / "aiperf-venv/bin/aiperf",
    )
    args = parser.parse_args()
    if args.request_rate <= 0:
        parser.error("--request-rate must be positive")
    args.trace_dir = args.trace_dir.resolve()
    args.aiperf = args.aiperf.resolve()
    for trace in WORKLOADS:
        source = args.trace_dir / f"{trace}_trace.jsonl"
        if not source.is_file():
            parser.error(f"missing trace: {source}")
    if not args.aiperf.is_file():
        parser.error(f"missing AIPerf executable: {args.aiperf}")
    server = kube("config", "view", "--minify", "-o", "json")["clusters"][0][
        "cluster"
    ]["server"]
    if not server.startswith("https://127.0.0.1:"):
        raise RuntimeError("Refusing to use a non-local cluster")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    initial = graph()
    components = validate_deployment(initial)
    runtime_image = validate_runtime_image(components)
    write_json(args.output / "initial-deployment.json", initial)
    original = {
        component["name"]: copy.deepcopy(
            component["podTemplate"]["spec"]["containers"][0]["args"]
        )
        for component in initial["spec"]["components"]
        if component["type"].lower() in ("frontend", "prefill", "decode")
    }
    expected_cases = [
        {
            "workload": trace,
            "routing_cli_mode": mode,
            "routing_label": ROUTING_LABELS[mode],
            "case": f"{trace}-{'po2c' if mode == 'power-of-two' else mode}",
        }
        for mode in ROUTING_LABELS
        for trace in WORKLOADS
    ]
    matrix = {
        "offered_rps": args.request_rate,
        "h200_profile_from_application_defaults": H200_PROFILE,
        "topology": {
            "frontend_processes": components["frontend"]["replicas"],
            "planner_processes": components["planner"]["replicas"],
            "prefill_processes": components["prefill"]["replicas"],
            "decode_processes": components["decode"]["replicas"],
            "ranks_per_worker_process": EXPECTED_RANKS_PER_PROCESS,
        },
        "runtime_image": runtime_image,
        "expected_cases": expected_cases,
        "cases_finished": False,
        "comparison_complete": False,
    }
    write_json(args.output / "matrix.json", matrix)
    try:
        for mode in ROUTING_LABELS:
            for trace in WORKLOADS:
                run_case(
                    trace,
                    mode,
                    args.output,
                    original,
                    args.request_rate,
                    args.trace_dir,
                    args.aiperf,
                )
        matrix["cases_finished"] = True
        write_json(args.output / "matrix.json", matrix)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/mooncake_report.py"), str(args.output)],
            check=True,
        )
        matrix["comparison_complete"] = True
        write_json(args.output / "matrix.json", matrix)
    except (RuntimeError, subprocess.SubprocessError, OSError, ValueError) as error:
        matrix["comparison_complete"] = False
        write_json(args.output / "matrix.json", matrix)
        write_json(
            args.output / "failure.json",
            {"error": str(error), "comparison_complete": False},
        )
        raise
    finally:
        try:
            set_args(original)
        except (RuntimeError, subprocess.SubprocessError, OSError, ValueError) as error:
            write_json(
                args.output / "restoration-failure.json",
                {"error": str(error), "original_arguments": original},
            )
            raise


if __name__ == "__main__":
    main()
