import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.python_runner import run_script_in_docker


def _docker_daemon_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def test_run_script_in_docker_falls_back_without_docker(tmp_path):
    """This environment likely doesn't have a usable Docker daemon
    (the `docker` CLI may be present but with no daemon running) --
    exercise the graceful-fallback path and confirm it still returns a
    valid ExecutionResult with sandboxed=False rather than crashing.
    """
    if _docker_daemon_available():
        import pytest

        pytest.skip("a working docker daemon is available in this environment; fallback path not exercised")

    result = run_script_in_docker("print(1)", str(tmp_path))

    assert result.success is True
    assert result.sandboxed is False
    assert "1" in result.stdout
