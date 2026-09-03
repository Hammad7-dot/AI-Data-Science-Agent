"""
app/executor.py

Phase 4/7 - Executor.

Thin wrapper around tools.python_runner.run_script that names each
iteration's script deterministically so generated scripts and any
artifacts they write don't collide across iterations.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.python_runner import run_script, run_script_in_docker, ExecutionResult  # noqa: E402


def execute(code: str, workdir: str, iteration: int, use_docker: bool = False,
            dataset_path: str | None = None, models_dir: str | None = None) -> ExecutionResult:
    """Write and run generated `code` for a given Ralph loop iteration.

    When `use_docker` is True, runs inside the Docker sandbox
    (tools.python_runner.run_script_in_docker). Docker infrastructure failures
    return a failed result without executing anything on the host.
    """
    filename = f"iteration_{iteration}.py"
    if use_docker:
        result = run_script_in_docker(code, workdir, filename=filename,
                                      dataset_path=dataset_path, models_dir=models_dir)
        if result.success:
            try:
                lines = result.stdout.strip().splitlines()
                payload = json.loads(lines[-1])
                if payload.get("model_path"):
                    relative = PurePosixPath(payload["model_path"]).relative_to("/models")
                    if ".." in relative.parts or not models_dir:
                        raise ValueError("Model artifact escaped the output directory")
                    root = Path(models_dir).resolve()
                    model = root.joinpath(*relative.parts).resolve()
                    if not model.is_relative_to(root) or not model.is_file():
                        raise ValueError("Model artifact is outside the output directory or missing")
                    payload["model_path"] = str(model)
                lines[-1] = json.dumps(payload)
                result.stdout = "\n".join(lines)
            except (ValueError, IndexError, TypeError, AttributeError) as exc:
                result.success = False
                result.stderr = f"Invalid sandbox output: {exc}"
                result.exit_code = 1
        return result
    return run_script(code, workdir, filename=filename)


def evaluate_in_sandbox(model_path, dataset_path, plan, workdir):
    """Keep model deserialization and holdout prediction inside the sandbox."""
    code = f'''import os, sys, json
sys.path.insert(0, os.environ["AGENT_PROJECT_ROOT"])
from tools.validation import evaluate_holdout
result = evaluate_holdout("/models/best_model.joblib", os.environ["AGENT_DATASET_PATH"], {plan!r})
print(json.dumps(result))
'''
    result = run_script_in_docker(code, workdir, filename="evaluate_holdout.py",
                                  dataset_path=dataset_path, models_dir=str(Path(model_path).parent))
    if not result.success:
        raise ValueError(result.stderr or "Sandbox holdout evaluation failed")
    return json.loads(result.stdout.strip().splitlines()[-1])
