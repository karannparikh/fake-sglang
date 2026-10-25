"""Replay complete Mooncake traces with AIPerf against the dedicated local lab."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".local"
KUBECTL = [
    "kubectl", "--kubeconfig", str(STATE / "kubeconfig"),
    "--context", "kind-dynamo-sim", "--request-timeout=30s",
]
SELECTOR = "nvidia.com/dynamo-graph-deployment-name=fake-sglang"
NODES = ("dynamo-sim-control-plane", "dynamo-sim-worker", "dynamo-sim-worker2")
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def kube(*args):
    return json.loads(subprocess.check_output([*KUBECTL, *args], text=True))


def graph():
    return kube("-n", "dynamo-sim", "get", "dgd", "fake-sglang", "-o", "json")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def routing_args(args, mode):
    if mode not in ("kv", "load-aware", "power-of-two"):
        raise ValueError(f"Unknown routing mode: {mode}")
    result = list(args)
    router_mode = "power-of-two" if mode == "power-of-two" else "kv"
    if "--router-mode" in result:
        result[result.index("--router-mode") + 1] = router_mode
    else:
        result.extend(["--router-mode", router_mode])
    if mode == "power-of-two":
        return [arg for arg in result if arg not in ("--router-kv-events", "--load-aware")]
    wanted = "--router-kv-events" if mode == "kv" else "--load-aware"
    unwanted = "--load-aware" if mode == "kv" else "--router-kv-events"
    result = [arg for arg in result if arg != unwanted]
    return result if wanted in result else result + [wanted]


def set_args(arguments):
    current = graph()
    changes = [{"op": "test", "path": "/metadata/resourceVersion",
                "value": current["metadata"]["resourceVersion"]}]
    for i, component in enumerate(current["spec"]["components"]):
        if component["name"] not in arguments:
            continue
        args = arguments[component["name"]]
        if args != component["podTemplate"]["spec"]["containers"][0]["args"]:
            changes.append({
                "op": "replace",
                "path": f"/spec/components/{i}/podTemplate/spec/containers/0/args",
                "value": args,
            })
    if len(changes) > 1:
        subprocess.run([
            *KUBECTL, "-n", "dynamo-sim", "patch", "dgd", "fake-sglang",
            "--type=json", "-p", json.dumps(changes),
        ], check=True)
    wait_ready()


def wait_ready(old_uids=()):
    deadline = time.monotonic() + 420
    while time.monotonic() < deadline:
        dgd = graph()
        ready = any(
            c["type"] == "Ready" and c["status"] == "True"
            and c.get("observedGeneration") == dgd["metadata"]["generation"]
            for c in dgd.get("status", {}).get("conditions", [])
        )
        pods = kube("-n", "dynamo-sim", "get", "pods", "-l", SELECTOR, "-o", "json")["items"]
        counts = {}
        for pod in pods:
            role = pod["metadata"]["labels"]["nvidia.com/dynamo-component"].lower()
            if (pod["metadata"]["uid"] not in old_uids
                    and "deletionTimestamp" not in pod["metadata"]
                    and any(c["type"] == "Ready" and c["status"] == "True"
                            for c in pod.get("status", {}).get("conditions", []))):
                counts[role] = counts.get(role, 0) + 1
        expected = {c["name"].lower(): c["replicas"] for c in dgd["spec"]["components"]
                    if c["name"].lower() in ("frontend", "prefill", "decode")}
        if (ready and all(counts.get(role) == n for role, n in expected.items())
                and not any("deletionTimestamp" in p["metadata"] for p in pods)):
            return
        time.sleep(3)
    raise RuntimeError(f"Local lab did not reach its configured replica counts: {expected}")


def cold_reset():
    pods = kube("-n", "dynamo-sim", "get", "pods", "-l", SELECTOR, "-o", "json")["items"]
    targets = [p for p in pods if p["metadata"]["labels"]["nvidia.com/dynamo-component"].lower()
               in ("frontend", "prefill", "decode")]
    subprocess.run([
        *KUBECTL, "-n", "dynamo-sim", "delete", "pod", "--wait=false",
        *[p["metadata"]["name"] for p in targets],
    ], check=True, stdout=subprocess.DEVNULL)
    wait_ready({p["metadata"]["uid"] for p in targets})
    time.sleep(15)
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as response:
        models = json.load(response)
    if not any(m["id"] == "fake-model" and m["context_window"] == 131072 for m in models["data"]):
        raise RuntimeError(f"Unexpected lab model: {models}")


def sample_cpu(stop, path):
    with path.open("w") as output, ThreadPoolExecutor(max_workers=3) as pool:
        while True:
            collected_at_ns = time.time_ns()

            def read_node(node):
                result = kube("get", "--raw", f"/api/v1/nodes/{node}/proxy/stats/summary")
                return {"node": node, "collected_at_ns": collected_at_ns, "pods": result["pods"]}

            for sample in pool.map(read_node, NODES):
                output.write(json.dumps(sample) + "\n")
            output.flush()
            if stop.wait(5):
                return


def run_case(trace, mode, destination, original, request_rate):
    case = destination / f"{trace}-{mode}"
    case.mkdir()
    set_args({name: routing_args(args, mode) for name, args in original.items()})
    cold_reset()
    write_json(case / "deployment.json", graph())
    write_json(case / "initial-pods.json",
               kube("-n", "dynamo-sim", "get", "pods", "-l", SELECTOR, "-o", "json"))
    source = STATE / "mooncake-100rps" / f"{trace}_trace.jsonl"
    with source.open() as lines:
        count = sum(1 for _ in lines)
    command = [
        str(STATE / "aiperf-venv/bin/aiperf"), "profile",
        "--model", "fake-model", "--url", "http://127.0.0.1:8000",
        "--endpoint-type", "chat", "--streaming",
        "--tokenizer", "Qwen/Qwen3-0.6B", "--tokenizer-revision", TOKENIZER_REVISION,
        "--input-file", str(source), "--custom-dataset-type", "mooncake_trace",
        "--no-fixed-schedule", "--request-count", str(count),
        "--num-dataset-entries", str(count), "--dataset-sampling-strategy", "sequential",
        "--request-rate", str(request_rate), "--arrival-pattern", "constant",
        "--isl-block-size", "512", "--use-server-token-count",
        "--extra-inputs", '{"temperature":0,"ignore_eos":true}',
        "--request-timeout-seconds", "120",
        "--random-seed", "42", "--workers-max", "4", "--ui-type", "none",
        "--artifact-dir", str(case / "aiperf"),
    ]
    write_json(case / "invocation.json", {
        "command": command, "trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "expected_requests": count, "routing": mode, "offered_rps": request_rate,
        "cold_cache": True, "original_timestamps_replaced": True,
    })
    env = {
        **os.environ, "HF_HOME": str(STATE / "aiperf-hf"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": "42",
        "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "900",
        "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "960",
    }
    write_json(case / "aiperf-environment.json", {
        key: env[key] for key in (
            "AIPERF_DATASET_CONFIGURATION_TIMEOUT", "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT",
            "HF_HOME", "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "PYTHONHASHSEED",
        )
    })
    stop = threading.Event()
    print(f"Starting {trace} {mode}: {count} requests at {request_rate} RPS; {case}", flush=True)
    with ThreadPoolExecutor(max_workers=1) as collector, (case / "console.log").open("w") as log:
        sampling = collector.submit(sample_cpu, stop, case / "cpu.jsonl")
        try:
            result = subprocess.run(
                ["/usr/bin/time", "-l", *command], env=env, stdout=log, stderr=subprocess.STDOUT,
                timeout=max(1800, 960 + count / request_rate + 120 + 120),
            )
        finally:
            # Kubelet counters can lag collection time by several seconds.
            time.sleep(15)
            stop.set()
            sampling.result()
    write_json(case / "exit.json", {"returncode": result.returncode})
    write_json(case / "final-pods.json",
               kube("-n", "dynamo-sim", "get", "pods", "-l", SELECTOR, "-o", "json"))
    query = '{__name__=~"sglang:simulated_prefix_cache_(query|hit)_tokens_total"}'
    url = "http://127.0.0.1:9090/api/v1/query?" + urllib.parse.urlencode({"query": query})
    with urllib.request.urlopen(url, timeout=15) as response:
        cache_metrics = json.load(response)
    if cache_metrics["status"] != "success":
        raise RuntimeError(f"Cache metrics query failed: {cache_metrics}")
    write_json(case / "cache-metrics.json", cache_metrics)
    if result.returncode != 0:
        raise RuntimeError(f"AIPerf failed; see {case / 'console.log'}")
    summary = json.loads((case / "aiperf/profile_export_aiperf.json").read_text())
    completed = summary["completed_request_count"]["avg"]
    if not summary["is_complete"] or completed != count:
        raise RuntimeError(f"Incomplete replay: expected {count}, completed {completed}")
    print(f"Finished {trace} {mode}: {completed} requests; "
          f"errors {summary['request_error_rate']['avg']}%", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-rate", type=int, default=50)
    parser.add_argument("--traces", nargs="+", choices=("conversation", "toolagent"),
                        default=["conversation", "toolagent"])
    parser.add_argument("--modes", nargs="+", choices=("kv", "load-aware", "power-of-two"),
                        default=["kv", "load-aware"])
    args = parser.parse_args()
    if args.request_rate <= 0:
        parser.error("--request-rate must be positive")
    server = kube("config", "view", "--minify", "-o", "json")["clusters"][0]["cluster"]["server"]
    if not server.startswith("https://127.0.0.1:"):
        raise RuntimeError("Refusing to use a non-local cluster")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    initial = graph()
    write_json(args.output / "initial-deployment.json", initial)
    original = {
        c["name"]: copy.deepcopy(c["podTemplate"]["spec"]["containers"][0]["args"])
        for c in initial["spec"]["components"] if c["name"].lower() in ("frontend", "prefill", "decode")
    }
    try:
        for mode in args.modes:
            for trace in args.traces:
                run_case(trace, mode, args.output, original, args.request_rate)
    except (RuntimeError, subprocess.SubprocessError, OSError, ValueError) as error:
        write_json(args.output / "failure.json", {
            "error": str(error), "comparison_complete": False,
        })
        raise
    finally:
        try:
            set_args(original)
        except (RuntimeError, subprocess.SubprocessError, OSError, ValueError) as error:
            write_json(args.output / "restoration-failure.json", {
                "error": str(error), "original_arguments": original,
            })
            raise


if __name__ == "__main__":
    main()
