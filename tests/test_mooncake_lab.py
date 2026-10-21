import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "mooncake_lab", Path(__file__).resolve().parents[1] / "scripts/mooncake_lab.py",
)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_routing_comparison_preserves_admission_and_engine_cache():
    original = [
        "--router-mode", "kv", "--router-kv-events",
        "--router-policy-config", "/etc/lab-router/router-policy.yaml",
        "--engine-request-limit", "8", "--kv-events-config", '{"publisher":"zmq"}',
    ]
    off = lab.routing_args(original, "load-aware")
    assert "--router-kv-events" not in off
    assert off[-1] == "--load-aware"
    assert off[:-1] == [a for a in original if a != "--router-kv-events"]
    assert "--router-kv-events" in original
    on = lab.routing_args(off, "kv")
    assert "--load-aware" not in on
    assert on[-1] == "--router-kv-events"
    assert on[:-1] == off[:-1]
    assert lab.routing_args(original, "kv") == original


@pytest.mark.parametrize("previous_mode", ["kv", "load-aware", "power-of-two"])
@pytest.mark.parametrize("mode", ["kv", "load-aware", "power-of-two"])
def test_all_routing_mode_transitions(previous_mode, mode):
    original = [
        "--router-mode", "kv", "--router-kv-events",
        "--dp-size", "10", "--engine-request-limit", "80",
        "--kv-events-config", '{"publisher":"zmq"}',
    ]
    previous = lab.routing_args(original, previous_mode)
    result = lab.routing_args(previous, mode)
    expected_router_mode = "power-of-two" if mode == "power-of-two" else "kv"
    assert result[result.index("--router-mode") + 1] == expected_router_mode
    assert ("--router-kv-events" in result) == (mode == "kv")
    assert ("--load-aware" in result) == (mode == "load-aware")
    assert [arg for arg in result[2:] if arg not in ("--router-kv-events", "--load-aware")] == [
        "--dp-size", "10", "--engine-request-limit", "80",
        "--kv-events-config", '{"publisher":"zmq"}',
    ]
    assert lab.routing_args(result, mode) == result
    assert original[1:3] == ["kv", "--router-kv-events"]


def test_routing_args_adds_missing_mode_and_rejects_unknown_mode():
    assert lab.routing_args([], "power-of-two") == ["--router-mode", "power-of-two"]
    with pytest.raises(ValueError, match="Unknown routing mode"):
        lab.routing_args([], "invalid")


def test_kubectl_always_targets_the_dedicated_local_context():
    assert lab.KUBECTL[lab.KUBECTL.index("--context") + 1] == "kind-dynamo-sim"
    assert lab.KUBECTL[lab.KUBECTL.index("--kubeconfig") + 1] == str(lab.STATE / "kubeconfig")


def test_cpu_interpolation_uses_sample_timestamps_not_poll_frequency():
    report_spec = importlib.util.spec_from_file_location(
        "mooncake_report", Path(__file__).resolve().parents[1] / "scripts/mooncake_report.py",
    )
    report = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(report)
    cpu, covered = report.integrate({0: 20, 10: 25, 20: 40}, 5, 15)
    assert cpu == 10
    assert covered == 10
    assert report.integrate({0: 20, 10: 25}, 20, 30) == (0, 0)
    result = report.cpu_report({
        ("prefill", "pod1", "main", "start"): {0: 0, 10: 10},
        ("prefill", "pod2", "main", "start"): {0: 0, 10: 10},
    }, 0, 10, {"prefill": 2})
    assert result["prefill"]["mean_cores"] == 2
    assert result["prefill"]["coverage_fraction"] == 1


def test_full_trace_replay_records_checksum_constant_rate_and_root_ignore_eos(tmp_path, monkeypatch):
    state = tmp_path / "state"
    source = state / "mooncake-100rps/conversation_trace.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"input_length":512,"output_length":8}\n' * 3)
    monkeypatch.setattr(lab, "STATE", state)
    original = {"frontend": ["--router-mode", "kv", "--router-kv-events"]}
    changed = []
    monkeypatch.setattr(lab, "set_args", changed.append)
    monkeypatch.setattr(lab, "cold_reset", lambda: None)
    monkeypatch.setattr(lab, "graph", lambda: {"spec": {"components": []}})
    monkeypatch.setattr(lab, "kube", lambda *args: {"items": []})
    monkeypatch.setattr(lab, "sample_cpu", lambda stop, path: path.write_text(""))
    monkeypatch.setattr(lab.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(lab.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(
        b'{"status":"success","data":{"result":[]}}',
    ))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        destination = Path(command[command.index("--artifact-dir") + 1])
        destination.mkdir()
        lab.write_json(destination / "profile_export_aiperf.json", {
            "completed_request_count": {"avg": 3}, "is_complete": True,
            "request_error_rate": {"avg": 0},
        })
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(lab.subprocess, "run", run)
    lab.run_case("conversation", "load-aware", tmp_path, original, 50)
    command = commands[0]
    for flag, value in {
        "--request-count": "3", "--num-dataset-entries": "3",
        "--dataset-sampling-strategy": "sequential", "--request-rate": "50",
        "--arrival-pattern": "constant", "--isl-block-size": "512",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert "--no-fixed-schedule" in command
    assert "--use-server-token-count" in command
    assert json.loads(command[command.index("--extra-inputs") + 1]) == {
        "temperature": 0, "ignore_eos": True,
    }
    invocation = json.loads((tmp_path / "conversation-load-aware/invocation.json").read_text())
    assert invocation["trace_sha256"] == lab.hashlib.sha256(source.read_bytes()).hexdigest()
    assert invocation["expected_requests"] == 3
    assert invocation["original_timestamps_replaced"] is True
    assert changed == [{"frontend": lab.routing_args(original["frontend"], "load-aware")}]


@pytest.mark.parametrize("fail_replay", [False, True])
@pytest.mark.parametrize("fail_restore", [False, True])
def test_original_dgd_arguments_restored_after_replay(tmp_path, monkeypatch, fail_replay, fail_restore):
    output = tmp_path / "comparison"
    original = ["--router-mode", "kv", "--router-kv-events"]
    monkeypatch.setattr(lab, "kube", lambda *args: {
        "clusters": [{"cluster": {"server": "https://127.0.0.1:1234"}}],
    })
    monkeypatch.setattr(lab, "graph", lambda: {"spec": {"components": [{
        "name": "frontend", "podTemplate": {"spec": {"containers": [{"args": original}]}},
    }]}})
    restored = []

    def restore(arguments):
        restored.append(arguments)
        if fail_restore:
            raise RuntimeError("restoration failed")

    monkeypatch.setattr(lab, "set_args", restore)

    def replay(*args):
        if fail_replay:
            raise RuntimeError("replay failed")

    monkeypatch.setattr(lab, "run_case", replay)
    monkeypatch.setattr("sys.argv", [
        "mooncake_lab.py", "--output", str(output), "--traces", "conversation", "--modes", "kv",
    ])
    if fail_restore:
        with pytest.raises(RuntimeError, match="restoration failed"):
            lab.main()
        failure = json.loads((output / "restoration-failure.json").read_text())
        assert failure["original_arguments"] == {"frontend": original}
    elif fail_replay:
        with pytest.raises(RuntimeError, match="replay failed"):
            lab.main()
    else:
        lab.main()
    if fail_replay:
        assert json.loads((output / "failure.json").read_text())["comparison_complete"] is False
    assert restored == [{"frontend": original}]
