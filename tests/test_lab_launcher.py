import os
from pathlib import Path
import subprocess

import pytest


def copy_launcher(tmp_path):
    source = Path(__file__).resolve().parents[1] / "deploy/local/lab.sh"
    launcher = tmp_path / "deploy/local/lab.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(source.read_text())
    launcher.chmod(0o755)
    return launcher


@pytest.mark.parametrize("command", ["benchmark", "report"])
def test_missing_benchmark_scripts_fail_before_tool_access(tmp_path, command):
    launcher = copy_launcher(tmp_path)
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("kubectl", "docker", "python3"):
        tool = tools / name
        tool.write_text("#!/bin/sh\necho unexpected >&2\nexit 99\n")
        tool.chmod(0o755)
    result = subprocess.run(
        ["bash", str(launcher), command],
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Missing required file:" in result.stderr
    assert "unexpected" not in result.stderr


def test_benchmark_setup_requires_both_validated_workloads_before_install(tmp_path):
    launcher = copy_launcher(tmp_path)
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "conversation_trace.jsonl").write_text("{}\n")
    tools = tmp_path / "bin"
    tools.mkdir()
    python = tools / "python3"
    python.write_text("#!/bin/sh\necho unexpected >&2\nexit 99\n")
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(launcher), "benchmark-setup", str(trace_dir)],
        env={
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "toolagent_trace.jsonl" in result.stderr
    assert "unexpected" not in result.stderr


def test_launcher_excludes_exploratory_commands():
    launcher = (
        Path(__file__).resolve().parents[1] / "deploy/local/lab.sh"
    ).read_text()
    for command in (
        "routing-dashboard",
        "routing-test",
        "traffic)",
        "packed_probe",
        "routing_scenario",
        "routing_load",
        "grafana",
    ):
        assert command not in launcher


def test_image_tag_is_content_addressed_to_repository_inputs():
    launcher = (
        Path(__file__).resolve().parents[1] / "deploy/local/lab.sh"
    ).read_text()
    assert "image=\"$(expected_image)\"" in launcher
    assert '[[ "$(cat "$STATE/image")" == "$(expected_image)" ]]' in launcher
