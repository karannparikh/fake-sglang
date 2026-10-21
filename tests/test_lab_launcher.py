import os
from pathlib import Path
import subprocess
import time

import pytest


@pytest.mark.parametrize("command", ["traffic", "routing-test", "routing-dashboard"])
def test_missing_benchmark_commands_fail_before_cluster_access(tmp_path, command):
    source = Path(__file__).resolve().parents[1] / "deploy/local/lab.sh"
    launcher = tmp_path / "deploy/local/lab.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(source.read_text())
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("kubectl", "docker", "python3"):
        tool = tools / name
        tool.write_text("#!/bin/sh\necho 'unexpected tool invocation' >&2\nexit 99\n")
        tool.chmod(0o755)
    result = subprocess.run(
        ["bash", str(launcher), command],
        env={**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"]},
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "requires the separate traffic/benchmark follow-up" in result.stderr
    assert "unexpected tool invocation" not in result.stderr


def test_long_routing_command_replaces_shell_and_preserves_exit_status_after_file_edit(tmp_path):
    source = Path(__file__).resolve().parents[1] / "deploy/local/lab.sh"
    launcher = tmp_path / "deploy/local/lab.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(source.read_text())
    (launcher.parent / "routing_compare.py").write_text("# Benchmark follow-up placeholder\n")
    tools = tmp_path / "bin"
    tools.mkdir()
    ready, release = tmp_path / "ready", tmp_path / "release"
    scripts = {
        "kubectl": '#!/bin/bash\nif [[ "$*" == "config current-context" ]]; then echo kind-dynamo-sim; '
                   'else echo https://127.0.0.1:12345; fi\n',
        "docker": "#!/bin/bash\necho dynamo-sim\n",
        "python3": '#!/bin/bash\nprintf "%s" "$$" > "$READY"\n'
                   'while [[ ! -f "$RELEASE" ]]; do sleep 0.01; done\nexit 23\n',
    }
    for name, content in scripts.items():
        script = tools / name
        script.write_text(content)
        script.chmod(0o755)
    process = subprocess.Popen(
        ["bash", str(launcher), "routing-test", "--rate", "10"],
        env={**os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
             "READY": str(ready), "RELEASE": str(release)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        assert int(ready.read_text()) == process.pid
        launcher.write_text(launcher.read_text() + '\necho "unfinished edit')
        release.touch()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 23, stdout + stderr
        assert "unexpected EOF" not in stderr
    finally:
        release.touch()
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=5)
