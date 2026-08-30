"""
tools/python_runner.py

Phase 4 - Python execution tool.

Executes a generated Python script as a subprocess (isolated process,
own interpreter) and captures stdout, stderr, exit code, execution
time, and any files it wrote into an artifacts directory.

NOTE: this runs on the local machine, not inside a container/sandbox.
That's an accepted limitation for a local prototype per SPEC.md /
project instructions ("For your first local prototype, you can keep
this controlled."). Swap `subprocess` for a Docker invocation later
without touching callers of `run_script`.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    artifacts: list[str] = field(default_factory=list)
    timed_out: bool = False
    sandboxed: bool = False


def run_script(
    code: str,
    workdir: str,
    filename: str = "generated_script.py",
    timeout: int = 120,
) -> ExecutionResult:
    """Write `code` to `workdir/filename` and execute it with the current
    Python interpreter as a subprocess, capturing everything.

    `workdir` doubles as the artifacts directory: any new files present
    there after execution (that weren't there before) are reported back
    as artifacts.
    """
    work = Path(workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    script_path = work / filename
    script_path.write_text(code, encoding="utf-8")

    before = {p.name for p in work.iterdir()}

    start = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(work),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n[python_runner] execution timed out after {timeout}s"
        exit_code = -1
    duration = time.time() - start

    after = {p.name for p in work.iterdir()}
    new_files = sorted(after - before - {filename})

    return ExecutionResult(
        success=(exit_code == 0 and not timed_out),
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        duration_seconds=round(duration, 3),
        artifacts=[str(work / f) for f in new_files],
        timed_out=timed_out,
        sandboxed=False,
    )


def run_script_in_docker(
    code: str,
    workdir: str,
    filename: str = "generated_script.py",
    timeout: int = 120,
    image: str = "ai-ds-agent-sandbox:latest",
    memory_limit: str = "512m",
    cpus: str = "1.0",
) -> ExecutionResult:
    """Write `code` to `workdir/filename` and execute it inside a
    throwaway Docker container built from docker/sandbox.Dockerfile,
    with no network access and bounded CPU/memory.

    Falls back to the plain subprocess `run_script` (with
    `sandboxed=False`) if the `docker` binary is missing or the
    container fails to even start (docker daemon not running, image
    not built, etc.) -- this keeps the rest of the pipeline runnable in
    dev/CI environments without Docker installed.
    """
    work = Path(workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    script_path = work / filename
    script_path.write_text(code, encoding="utf-8")

    # Pre-flight: confirm the docker binary exists AND a daemon is
    # actually reachable. `docker run` itself would also fail in these
    # cases, but with a docker-CLI error (nonzero exit, no exception) --
    # not something we can distinguish from the script's own nonzero
    # exit after the fact. Checking up front lets us cleanly fall back
    # to plain subprocess execution instead of misreporting a docker
    # infrastructure failure as sandboxed=True.
    try:
        preflight = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=10
        )
        if preflight.returncode != 0:
            return run_script(code, workdir, filename=filename, timeout=timeout)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return run_script(code, workdir, filename=filename, timeout=timeout)

    before = {p.name for p in work.iterdir()}

    docker_cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--memory",
        memory_limit,
        "--cpus",
        cpus,
        "-v",
        f"{work}:/workspace",
        "-w",
        "/workspace",
        image,
        "python",
        filename,
    ]

    start = time.time()
    timed_out = False
    try:
        proc = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n[python_runner] docker execution timed out after {timeout}s"
        exit_code = -1
    except (FileNotFoundError, OSError):
        # `docker` binary missing, or the run otherwise failed to start
        # (e.g. daemon not running). Fall back to plain subprocess
        # execution -- not isolated, so sandboxed stays False.
        return run_script(code, workdir, filename=filename, timeout=timeout)
    duration = time.time() - start

    after = {p.name for p in work.iterdir()}
    new_files = sorted(after - before - {filename})

    return ExecutionResult(
        success=(exit_code == 0 and not timed_out),
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        duration_seconds=round(duration, 3),
        artifacts=[str(work / f) for f in new_files],
        timed_out=timed_out,
        sandboxed=True,
    )
