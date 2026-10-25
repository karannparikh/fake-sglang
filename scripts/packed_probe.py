"""Verify packed rank addressing and KV-cache isolation through the live frontend."""

import argparse
import json
from pathlib import Path
import random
import re
import time
import urllib.parse
import urllib.request

from mooncake_lab import kube, SELECTOR


def metric(name, pods):
    selector = json.dumps("|".join(re.escape(pod) for pod in pods))
    query = "sglang:simulated_prefix_cache_" + name + "{pod=~" + selector + "}"
    url = "http://127.0.0.1:9090/api/v1/query?" + urllib.parse.urlencode({"query": query})
    with urllib.request.urlopen(url, timeout=15) as response:
        result = json.load(response)
    if result["status"] != "success":
        raise RuntimeError(result)
    return {
        (s["metric"]["dynamo_component_type"], int(s["metric"]["worker_id"], 16),
         int(s["metric"]["dp_rank"])): float(s["value"][1])
        for s in result["data"]["result"]
    }


def send(tokens, prefill=None, decode=None):
    extension = {"token_data": tokens, "extra_fields": ["worker_id"]}
    if prefill is not None:
        extension.update(prefill_worker_id=prefill[1], prefill_dp_rank=prefill[2])
    if decode is not None:
        extension.update(decode_worker_id=decode[1], dp_rank=decode[2])
    payload = {
        "model": "fake-model", "messages": [{"role": "user", "content": "Packed rank probe"}],
        "nvext": extension, "temperature": 0, "max_tokens": 4, "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    workers = usage = finish_reason = None
    done = False
    with urllib.request.urlopen(request, timeout=30) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            workers = (event.get("nvext") or {}).get("worker_id") or workers
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                finish_reason = choice.get("finish_reason") or finish_reason
    if (not done or not workers or not usage or usage["completion_tokens"] != 4
            or finish_reason != "length"):
        raise RuntimeError(f"Incomplete packed response: {workers}, {usage}, done={done}")
    for role, expected in (("prefill", prefill), ("decode", decode)):
        if expected is not None and (
            workers[f"{role}_worker_id"], workers[f"{role}_dp_rank"]
        ) != expected[1:]:
            raise AssertionError(f"Misrouted {role}: {workers}, expected {expected}")
    if usage["prompt_tokens"] != len(tokens):
        raise AssertionError(f"Token count changed: {usage}")
    return workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-per-role", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    active = kube("-n", "dynamo-sim", "get", "pods", "-l", SELECTOR, "-o", "json")
    pods = [p["metadata"]["name"] for p in active["items"]
            if "deletionTimestamp" not in p["metadata"]]
    capacities = metric("capacity_tokens", pods)
    prefill = sorted(k for k in capacities if k[0] == "prefill")
    decode = sorted(k for k in capacities if k[0] == "decode")
    if len(prefill) != args.targets_per_role or len(decode) != args.targets_per_role:
        raise AssertionError(f"Expected {args.targets_per_role} targets per role: {prefill}, {decode}")
    if set(capacities.values()) != {131072.0}:
        raise AssertionError(f"Unexpected per-rank KV capacity: {capacities}")
    before = metric("query_tokens_total", pods)
    before_hits = metric("hit_tokens_total", pods)
    rng = random.Random(time.time_ns())
    responses = []
    for p, d in zip(prefill, decode):
        tokens = [rng.randrange(1000, 100000) for _ in range(256)]
        responses.append(send(tokens, p, d))

    same_process = next(p for p in prefill[1:] if p[1] == prefill[0][1])
    tokens = [rng.randrange(1000, 100000) for _ in range(512)]
    for target in (prefill[0], same_process, prefill[0]):
        responses.append(send(tokens, target, decode[0]))
    deadline = time.monotonic() + 30
    while True:
        after = metric("query_tokens_total", pods)
        hits = metric("hit_tokens_total", pods)
        if all(after.get(p, 0) > before[p] for p in prefill):
            if hits.get(prefill[0], 0) - before_hits[prefill[0]] >= 512:
                break
        if time.monotonic() >= deadline:
            raise AssertionError("Per-rank metrics did not reflect the routed requests")
        time.sleep(2)
    if hits[same_process] != before_hits[same_process]:
        raise AssertionError("A cold rank reused another rank's KV cache")

    exclusive = [rng.randrange(1000, 100000) for _ in range(1024)]
    responses.append(send(exclusive, prefill[0], decode[0]))
    time.sleep(3)
    selected = send(exclusive, decode=decode[0])
    if (selected["prefill_worker_id"], selected["prefill_dp_rank"]) != prefill[0][1:]:
        raise AssertionError(f"KV router did not choose the cached rank: {selected}")
    responses.append(selected)
    report = {
        "targets_per_role": args.targets_per_role,
        "prefill_processes": len({p[1] for p in prefill}),
        "decode_processes": len({d[1] for d in decode}),
        "per_rank_capacity_tokens": 131072, "responses": responses,
        "cache_isolation_verified": True, "cache_aware_rank_selection_verified": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "responses"}))


if __name__ == "__main__":
    main()
