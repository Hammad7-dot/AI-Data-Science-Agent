"""
app/executor.py

Phase 4/7 - Executor.

Thin wrapper around tools.python_runner.run_script that names each
iteration's script deterministically so generated scripts and any
artifacts they write don't collide across iterations.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.python_runner import run_script, run_script_in_docker, ExecutionResult  # noqa: E402


def execute(code: str, workdir: str, iteration: int, use_docker: bool = False) -> ExecutionResult:
    """Write and run generated `code` for a given Ralph loop iteration.

    When `use_docker` is True, runs inside the Docker sandbox
    (tools.python_runner.run_script_in_docker), which gracefully falls
    back to the plain subprocess path if Docker isn't available.
    """
    filename = f"iteration_{iteration}.py"
    if use_docker:
        return run_script_in_docker(code, workdir, filename=filename)
    return run_script(code, workdir, filename=filename)
