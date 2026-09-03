import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.python_runner import run_script_in_docker


@pytest.mark.parametrize("failure", [FileNotFoundError("docker missing"),
                                    subprocess.CompletedProcess(["docker"], 1, "", "daemon unavailable")])
def test_sandbox_never_falls_back_to_host(tmp_path, monkeypatch, failure):
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if isinstance(failure, Exception):
            raise failure
        return failure
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_script_in_docker("open('host-executed', 'w').write('unsafe')", str(tmp_path))
    assert not result.success
    assert "sandbox" in result.stderr.lower()
    assert all(command[0] == "docker" for command in calls)
    assert not (tmp_path / "host-executed").exists()


def test_sandbox_mounts_explicit_inputs_helpers_and_outputs(tmp_path, monkeypatch):
    dataset = tmp_path / "data.csv"
    dataset.write_text("x,y\n1,2\n")
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "{}", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_script_in_docker("print('{}')", str(tmp_path / "generated"),
                                  dataset_path=str(dataset), models_dir=str(tmp_path / "models"))
    assert result.success and result.sandboxed
    command = next(command for command in calls if command[1] == "run")
    joined = " ".join(command)
    assert f"{dataset.resolve()}:/input/dataset.csv:ro" in command
    assert any(part.endswith(":/opt/agent/tools:ro") for part in command)
    assert f"{(tmp_path / 'models').resolve()}:/models" in command
    assert "AGENT_DATASET_PATH=/input/dataset.csv" in command
    assert "AGENT_MODELS_DIR=/models" in command
    assert "AGENT_PROJECT_ROOT=/opt/agent" in command
    assert "--network none" in joined
    assert "--read-only" in command
    assert f"{(tmp_path / 'generated' / 'generated_script.py').resolve()}:/workspace/generated_script.py:ro" in command


def test_timeout_removes_container_and_decodes_partial_output(tmp_path, monkeypatch):
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, 1, output=b"partial", stderr=b"diagnostic")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_script_in_docker("print(1)", str(tmp_path), timeout=1)
    assert result.timed_out and not result.success
    assert result.stdout == "partial"
    assert "diagnostic" in result.stderr
    command = next(command for command in calls if command[1] == "run")
    name = command[command.index("--name") + 1]
    assert ["docker", "rm", "-f", name] in calls


def test_executor_translates_model_path_without_allowing_escape(tmp_path, monkeypatch):
    import app.executor as executor
    from tools.python_runner import ExecutionResult
    models = tmp_path / "models"
    models.mkdir()
    (models / "winner.joblib").write_bytes(b"model")
    for reported, valid in [("/models/winner.joblib", True), ("/models/../../secret", False)]:
        def fake(*args, **kwargs):
            return ExecutionResult(True, json.dumps({"model_path": reported}), "", 0, 0, sandboxed=True)
        monkeypatch.setattr(executor, "run_script_in_docker", fake)
        result = executor.execute("print(1)", str(tmp_path), 1, use_docker=True,
                                  dataset_path="data.csv", models_dir=str(models))
        assert result.success is valid
        if valid:
            assert json.loads(result.stdout)["model_path"] == str(models / "winner.joblib")


def test_real_docker_training_and_holdout(tmp_path):
    if not shutil.which("docker"):
        pytest.skip("Docker is not installed")
    if subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode:
        pytest.skip("Docker daemon unavailable")
    if subprocess.run(["docker", "image", "inspect", "ai-ds-agent-sandbox:latest"], capture_output=True).returncode:
        pytest.skip("Build docker/sandbox.Dockerfile to run the integration test")
    import numpy as np
    import pandas as pd
    from app.agent import run_agent
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"x": np.arange(60, dtype=float), "target": np.arange(60) * 2.0}).to_csv(dataset, index=False)
    result = run_agent(str(dataset), "Predict target", max_iterations=1,
                       workspace_dir=str(tmp_path / "workspace"), use_docker=True)
    assert result["best_model_path"], result["all_experiments"]
    assert Path(result["best_model_path"]).is_file()
    assert result["holdout_evaluation"]["score"] == pytest.approx(1.0)


def test_agent_stops_on_missing_sandbox(tmp_path, monkeypatch):
    from app.agent import run_agent
    import pandas as pd
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"x": [float(i) for i in range(60)], "target": [float(i * 2) for i in range(60)]}).to_csv(dataset, index=False)
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("docker missing")
    monkeypatch.setattr(subprocess, "run", unavailable)
    result = run_agent(str(dataset), "Predict target", max_iterations=5,
                       workspace_dir=str(tmp_path / "workspace"), use_docker=True)
    assert result["status"] == "sandbox_unavailable"
    assert result["iterations_run"] == 1
    assert not result["best_model_path"]
    history = Path(result["experiments_path"])
    assert not history.exists() or json.loads(history.read_text()) == []


def test_generated_code_uses_container_environment_paths(tmp_path, monkeypatch):
    import pandas as pd
    from app.coder import generate_code
    from tools.python_runner import run_script
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"x": [float(i) for i in range(60)], "target": [float(i * 2) for i in range(60)]}).to_csv(dataset, index=False)
    models = tmp_path / "models"
    monkeypatch.setenv("AGENT_DATASET_PATH", str(dataset))
    monkeypatch.setenv("AGENT_MODELS_DIR", str(models))
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    plan = {"task_type": "regression", "target": "target", "metric": "r2", "candidate_models": ["linear_regression"]}
    code = generate_code("/unavailable/host/data.csv", plan, 1,
                         feature_engineering="pipeline", models_dir="/unavailable/host/models")
    result = run_script(code, str(tmp_path / "generated"))
    assert result.success, result.stderr
    assert Path(json.loads(result.stdout)["model_path"]).is_relative_to(models)


def test_promoting_model_does_not_follow_preexisting_output_links(tmp_path, monkeypatch):
    import os
    import pandas as pd
    import app.ralph as ralph
    from app.agent import run_agent
    victim = tmp_path / "unrelated-file"
    victim.write_bytes(b"must remain unchanged")
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"x": [float(i) for i in range(60)], "target": [float(i * 2) for i in range(60)]}).to_csv(dataset, index=False)
    actual_execute = ralph.execute
    def execute_with_link(code, workdir, iteration, **kwargs):
        result = actual_execute(code, workdir, iteration, **kwargs)
        os.link(victim, Path(workdir).parent / "models" / "best_model.joblib")
        return result
    monkeypatch.setattr(ralph, "execute", execute_with_link)
    result = run_agent(str(dataset), "Predict target", max_iterations=1,
                       workspace_dir=str(tmp_path / "workspace"))
    assert result["best_model_path"]
    assert victim.read_bytes() == b"must remain unchanged"
