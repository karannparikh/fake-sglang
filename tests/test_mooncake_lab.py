import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy.local import render


spec = importlib.util.spec_from_file_location(
    "mooncake_lab",
    Path(__file__).resolve().parents[1] / "scripts/mooncake_lab.py",
)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


@pytest.mark.parametrize("previous_mode", tuple(lab.ROUTING_LABELS))
@pytest.mark.parametrize("mode", tuple(lab.ROUTING_LABELS))
def test_all_routing_mode_transitions(previous_mode, mode):
    original = [
        "--router-mode",
        "kv",
        "--router-kv-events",
        "--dp-size",
        "10",
        "--router-policy-config",
        "/etc/lab-router/router-policy.yaml",
    ]
    previous = lab.routing_args(original, previous_mode)
    result = lab.routing_args(previous, mode)
    expected_router_mode = "power-of-two" if mode == "power-of-two" else "kv"
    assert result[result.index("--router-mode") + 1] == expected_router_mode
    assert ("--router-kv-events" in result) == (mode == "kv")
    assert ("--load-aware" in result) == (mode == "load-aware")
    assert lab.routing_args(result, mode) == result
    assert original[1:3] == ["kv", "--router-kv-events"]


def test_deployment_validation_requires_exact_focused_topology():
    deployment = render.dynamo()[0]
    components = lab.validate_deployment(deployment)
    assert set(components) == {"frontend", "prefill", "decode", "planner"}
    deployment["spec"]["components"][1]["replicas"] = 3
    with pytest.raises(RuntimeError, match="exact deployment topology"):
        lab.validate_deployment(deployment)


def pod(name, uid, role, *, ready=True, restarts=0):
    return {
        "metadata": {
            "name": name,
            "uid": uid,
            "labels": {"nvidia.com/dynamo-component-type": role},
        },
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "containerStatuses": [
                {"name": "main", "ready": ready, "restartCount": restarts}
            ],
        },
    }


def complete_pods():
    result = []
    for role, count in lab.EXPECTED_REPLICAS.items():
        for index in range(count):
            result.append(pod(f"fake-sglang-{role}-{index}", f"{role}-{index}", role))
    return result


def test_pod_validation_requires_exact_ready_fresh_stable_fleet():
    pod_list = complete_pods()
    observation = lab.validate_pods(pod_list)
    assert len(observation) == 10
    with pytest.raises(RuntimeError, match="stale UIDs"):
        lab.validate_pods(pod_list, {"prefill-0"})
    pod_list[-1]["status"]["containerStatuses"][0]["restartCount"] = 1
    assert lab.validate_pods(pod_list)[-1]["containers"][0]["restart_count"] in (0, 1)
    pod_list[-1]["status"]["conditions"][0]["status"] = "False"
    with pytest.raises(RuntimeError, match="not fully ready"):
        lab.validate_pods(pod_list)


def cache_snapshot(pod_names, value):
    samples = []
    for pod_name in pod_names:
        for rank in range(10):
            for field in ("query", "hit"):
                samples.append(
                    {
                        "metric": {
                            "__name__": (
                                f"sglang:simulated_prefix_cache_{field}_tokens_total"
                            ),
                            "pod": pod_name,
                            "dynamo_component_type": "prefill",
                            "worker_id": pod_name,
                            "dp_rank": str(rank),
                        },
                        "value": [0, str(value)],
                    }
                )
    return {"status": "success", "data": {"result": samples}}


def test_full_case_uses_direct_counter_baselines_and_strict_aiperf(
    tmp_path, monkeypatch
):
    state = tmp_path / "state"
    trace_dir = state / "traces"
    trace_dir.mkdir(parents=True)
    source = trace_dir / "conversation_trace.jsonl"
    source.write_text('{"input_length":512,"output_length":8}\n' * 3)
    aiperf = state / "aiperf"
    aiperf.write_text("")
    monkeypatch.setattr(lab, "STATE", state)
    deployment = render.dynamo()[0]
    pod_list = complete_pods()
    fleet = lab.validate_pods(pod_list)
    monkeypatch.setattr(lab, "set_args", lambda arguments: None)
    monkeypatch.setattr(
        lab,
        "cold_reset",
        lambda: ({"items": pod_list}, fleet),
    )
    monkeypatch.setattr(lab, "graph", lambda: deployment)
    snapshots = iter(
        [
            cache_snapshot(
                [item["name"] for item in fleet if item["role"] == "prefill"], 0
            ),
            cache_snapshot(
                [item["name"] for item in fleet if item["role"] == "prefill"], 100
            ),
        ]
    )
    monkeypatch.setattr(
        lab, "wait_cache_snapshot", lambda names, expected: next(snapshots)
    )

    def sample(stop, cpu_path, stability_path):
        cpu_path.write_text("")
        stability_path.write_text(
            json.dumps({"collected_at_ns": 1, "pods": fleet}) + "\n"
        )

    monkeypatch.setattr(lab, "sample_runtime", sample)
    monkeypatch.setattr(lab, "validate_stability", lambda case, expected: None)
    monkeypatch.setattr(lab.time, "sleep", lambda seconds: None)

    def run(command, **kwargs):
        destination = Path(command[command.index("--artifact-dir") + 1])
        destination.mkdir()
        lab.write_json(
            destination / "profile_export_aiperf.json",
            {
                "completed_request_count": {"avg": 3},
                "request_count": {"avg": 3},
                "request_error_rate": {"avg": 0},
                "error_summary": {},
                "osl_mismatch_count": {"avg": 0},
                "is_complete": True,
            },
        )
        rows = [
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "request_start_ns": index,
                    "request_end_ns": index + 1,
                },
                "error": None,
            }
            for index in range(3)
        ]
        (destination / "profile_export.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(lab.subprocess, "run", run)
    original = {
        "Frontend": ["--router-mode", "kv", "--router-kv-events"],
    }
    lab.run_case(
        "conversation",
        "power-of-two",
        tmp_path,
        original,
        20,
        trace_dir,
        aiperf,
    )
    case = tmp_path / "conversation-po2c"
    assert (case / "cache-metrics-before.json").is_file()
    assert (case / "cache-metrics-after.json").is_file()
    invocation = json.loads((case / "invocation.json").read_text())
    assert invocation["routing_label"] == "PO2C"
    assert invocation["offered_rps"] == 20
    assert invocation["h200_profile_from_application_defaults"] == lab.H200_PROFILE


def test_default_main_runs_two_by_three_matrix_at_20_rps(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    for trace in ("conversation", "toolagent"):
        (trace_dir / f"{trace}_trace.jsonl").write_text("{}\n")
    aiperf = tmp_path / "aiperf"
    aiperf.write_text("")
    output = tmp_path / "output"
    monkeypatch.setattr(
        lab,
        "kube",
        lambda *args: {
            "clusters": [{"cluster": {"server": "https://127.0.0.1:1234"}}]
        },
    )
    deployment = render.dynamo()[0]
    monkeypatch.setattr(lab, "graph", lambda: deployment)
    monkeypatch.setattr(
        lab,
        "validate_runtime_image",
        lambda components: "fake-sglang-dynamo:1.6.0-local-0123456789abcdef",
    )
    calls = []
    monkeypatch.setattr(
        lab,
        "run_case",
        lambda trace, mode, destination, original, rate, traces, executable: calls.append(
            (trace, mode, rate)
        ),
    )
    monkeypatch.setattr(lab, "set_args", lambda arguments: None)
    monkeypatch.setattr(lab.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mooncake_lab.py",
            "--output",
            str(output),
            "--trace-dir",
            str(trace_dir),
            "--aiperf",
            str(aiperf),
        ],
    )
    lab.main()
    assert calls == [
        ("conversation", "kv", 20),
        ("toolagent", "kv", 20),
        ("conversation", "load-aware", 20),
        ("toolagent", "load-aware", 20),
        ("conversation", "power-of-two", 20),
        ("toolagent", "power-of-two", 20),
    ]
    matrix = json.loads((output / "matrix.json").read_text())
    assert matrix["comparison_complete"] is True
    assert [case["routing_label"] for case in matrix["expected_cases"]][-2:] == [
        "PO2C",
        "PO2C",
    ]
    assert matrix["cases_finished"] is True
    assert matrix["comparison_complete"] is True


def test_report_failure_never_marks_matrix_complete(tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    for trace in lab.WORKLOADS:
        (trace_dir / f"{trace}_trace.jsonl").write_text("{}\n")
    aiperf = tmp_path / "aiperf"
    aiperf.write_text("")
    output = tmp_path / "output"
    monkeypatch.setattr(
        lab,
        "kube",
        lambda *args: {
            "clusters": [{"cluster": {"server": "https://127.0.0.1:1234"}}]
        },
    )
    monkeypatch.setattr(lab, "graph", lambda: render.dynamo()[0])
    monkeypatch.setattr(
        lab,
        "validate_runtime_image",
        lambda components: "fake-sglang-dynamo:1.6.0-local-0123456789abcdef",
    )
    monkeypatch.setattr(lab, "run_case", lambda *args: None)
    monkeypatch.setattr(lab, "set_args", lambda arguments: None)
    monkeypatch.setattr(
        lab.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            lab.subprocess.CalledProcessError(1, "report")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "mooncake_lab.py",
            "--output",
            str(output),
            "--trace-dir",
            str(trace_dir),
            "--aiperf",
            str(aiperf),
        ],
    )
    with pytest.raises(lab.subprocess.CalledProcessError):
        lab.main()
    matrix = json.loads((output / "matrix.json").read_text())
    assert matrix["cases_finished"] is True
    assert matrix["comparison_complete"] is False
    assert (output / "failure.json").is_file()
