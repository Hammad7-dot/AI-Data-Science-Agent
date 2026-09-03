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
import os
import sys
import time
import uuid
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


def _write_script(work, filename, code):
    if Path(filename.replace("\\", "/")).name != filename or filename in ("", ".", ".."):
        raise ValueError("Script filename must be a plain filename")
    script = work / filename
    temporary = work / (uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(code)
        # Replacing the directory entry avoids following planted output links.
        os.replace(temporary, script)
    finally:
        temporary.unlink(missing_ok=True)
    return script


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
    script_path = _write_script(work, filename, code)

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
        stdout = _as_text(e.stdout)
        stderr = _as_text(e.stderr) + f"\n[python_runner] execution timed out after {timeout}s"
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


def _as_text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")


def run_script_in_docker(
    code: str,
    workdir: str,
    filename: str = "generated_script.py",
    timeout: int = 120,
    image: str = "ai-ds-agent-sandbox:latest",
    memory_limit: str = "512m",
    cpus: str = "1.0",
    dataset_path: str | None = None,
    models_dir: str | None = None,
) -> ExecutionResult:
    """Run only in Docker. Infrastructure failures never execute code on the host."""
    work = Path(workdir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    script_path = _write_script(work, filename, code)

    def unavailable(reason):
        return ExecutionResult(False, "", f"Docker sandbox unavailable: {reason}", 125, 0)

    try:
        preflight = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        if preflight.returncode:
            return unavailable(preflight.stderr or "daemon not reachable")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return unavailable(str(exc))

    name = "ai-ds-" + uuid.uuid4().hex
    tools_dir = Path(__file__).resolve().parent
    docker_cmd = [
        "docker", "run", "--rm", "--name", name,
        "--network", "none", "--memory", memory_limit, "--cpus", cpus,
        "--pids-limit", "128", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,size=64m",
        "-v", f"{work}:/workspace",
        "-v", f"{script_path}:/workspace/{filename}:ro",
        "-v", f"{tools_dir}:/opt/agent/tools:ro",
        "-e", "AGENT_PROJECT_ROOT=/opt/agent",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
        "-w", "/workspace",
    ]
    if dataset_path:
        dataset = Path(dataset_path).resolve(strict=True)
        docker_cmd += ["-v", f"{dataset}:/input/dataset.csv:ro",
                       "-e", "AGENT_DATASET_PATH=/input/dataset.csv"]
    if models_dir:
        models = Path(models_dir).resolve()
        models.mkdir(parents=True, exist_ok=True)
        docker_cmd += ["-v", f"{models}:/models", "-e", "AGENT_MODELS_DIR=/models"]
    docker_cmd += [image, "python", filename]

    before = {p.name for p in work.iterdir()}
    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout)
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        if exit_code in (125, 126, 127):
            return unavailable(stderr or f"container could not start (exit {exit_code})")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr) + f"\nDocker sandbox timed out after {timeout}s"
        exit_code = -1
        # Killing the Docker CLI alone does not stop the container.
        try:
            cleanup = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=10)
            if cleanup.returncode:
                stderr += f"\nContainer cleanup failed: {cleanup.stderr}"
        except (OSError, subprocess.TimeoutExpired) as cleanup_error:
            stderr += f"\nContainer cleanup failed: {cleanup_error}"
    except OSError as exc:
        return unavailable(str(exc))
    artifacts = [str(p) for p in work.iterdir() if p.name not in before]
    return ExecutionResult(
        success=exit_code == 0 and not timed_out, stdout=stdout, stderr=stderr,
        exit_code=exit_code, duration_seconds=round(time.monotonic() - start, 3),
        artifacts=artifacts, timed_out=timed_out, sandboxed=True,
    )
