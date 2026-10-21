#!/usr/bin/env python3
"""Run the bounded routing scenario in an isolated pod; retain logs and report."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]


def run(arguments):
    duration_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    duration_parser.add_argument("--duration-seconds", type=int, default=0)
    duration_parser.add_argument("--pairs", type=int, default=2)
    duration_parser.add_argument("--requests", type=int, default=24)
    duration, _ = duration_parser.parse_known_args(arguments)
    if (not 0 <= duration.duration_seconds <= 600 or not 1 <= duration.pairs <= 4
            or not 1 <= duration.requests <= 600):
        raise ValueError("Require duration/requests <= 600 per case and 1-4 pairs")
    deadline_seconds = max(
        900, 2 * duration.pairs * max(duration.duration_seconds, duration.requests) + 300,
    )
    kube = ["kubectl", "--kubeconfig", str(ROOT / ".local/kubeconfig"),
            "--context", "kind-dynamo-sim"]

    def kubectl(*args, **kwargs):
        return subprocess.run(
            [*kube, *args], check=True, capture_output=True, text=True, timeout=30, **kwargs,
        ).stdout

    if kubectl("config", "current-context").strip() != "kind-dynamo-sim":
        raise RuntimeError("Refusing to run outside the dedicated kind-dynamo-sim cluster")
    config = json.loads(kubectl("config", "view", "--minify", "-o", "json"))
    if not config["clusters"][0]["cluster"]["server"].startswith("https://127.0.0.1:"):
        raise RuntimeError("The lab Kubernetes endpoint must be on localhost")
    image = (ROOT / ".local/image").read_text().strip()
    output = ROOT / ".local" / ("routing-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6])
    output.mkdir()
    manifest = {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"generateName": "fake-routing-test-", "namespace": "dynamo-sim",
                     "labels": {"app": "fake-routing-test"}},
        "spec": {
            "restartPolicy": "Never", "activeDeadlineSeconds": deadline_seconds,
            "automountServiceAccountToken": False,
            "containers": [{
                "name": "scenario", "image": image, "imagePullPolicy": "Never",
                "command": ["python", "scripts/routing_scenario.py", *arguments],
                "env": [{"name": "NATS_SERVER", "value": "nats://nats.dynamo-system.svc.cluster.local:4222"}],
                "resources": {"requests": {"cpu": "500m", "memory": "1Gi"},
                              "limits": {"cpu": "4", "memory": "4Gi"}},
                "securityContext": {"allowPrivilegeEscalation": False,
                                    "capabilities": {"drop": ["ALL"]}},
            }],
        },
    }
    (output / "pod.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pod = json.loads(kubectl("create", "-f", "-", "-o", "json", input=json.dumps(manifest)))
    name = pod["metadata"]["name"]
    print(f"Running {name}; results will be saved in {output}", flush=True)
    try:
        deadline = time.monotonic() + deadline_seconds + 20
        while time.monotonic() < deadline:
            pod = json.loads(kubectl("-n", "dynamo-sim", "get", "pod", name, "-o", "json"))
            if pod["status"].get("phase") in ("Succeeded", "Failed"):
                break
            time.sleep(2)
        else:
            raise TimeoutError(f"Routing test exceeded its deadline; inspect {output / 'pod.json'}")
        logs = kubectl("-n", "dynamo-sim", "logs", name)
        (output / "run.log").write_text(logs)
        reports = [line.removeprefix("ROUTING_REPORT ") for line in logs.splitlines()
                   if line.startswith("ROUTING_REPORT ")]
        if pod["status"]["phase"] != "Succeeded" or len(reports) != 1:
            raise RuntimeError(f"Routing test failed; retained logs: {output / 'run.log'}\n{logs[-10000:]}")
        report = json.loads(reports[0])
        report["image"] = image
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2))
        print(f"Report: {output / 'report.json'}")
        subprocess.run([
            sys.executable, str(ROOT / "deploy/local/routing_dashboard.py"),
            str(output / "report.json"),
        ], check=True, timeout=180)
    finally:
        (output / "pod-final.json").write_text(json.dumps(pod, indent=2) + "\n")
        if not (output / "run.log").exists():
            try:
                (output / "run.log").write_text(kubectl("-n", "dynamo-sim", "logs", name))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                print(f"Could not retrieve logs before cleanup: {exc}", file=sys.stderr)
        # This unique pod was created by this invocation; never touch serving pods.
        kubectl("-n", "dynamo-sim", "delete", "pod", name, "--wait=false", "--ignore-not-found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _, scenario_arguments = parser.parse_known_args()
    run(scenario_arguments)
