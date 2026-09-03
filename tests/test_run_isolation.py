import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agent import run_agent
from app.run_scope import scope_key


def dataset(path, offset=0):
    pd.DataFrame({"x": np.arange(60, dtype=float),
                  "target": np.arange(60) * 2.0 + offset}).to_csv(path, index=False)
    return str(path)


def test_scope_changes_when_bytes_change_at_same_path(tmp_path):
    path = dataset(tmp_path / "data.csv")
    first = scope_key(path, "regression", "target", "r2")
    dataset(Path(path), offset=5)
    assert scope_key(path, "regression", "target", "r2") != first


def test_identical_dataset_bytes_share_scope_across_names(tmp_path):
    first = dataset(tmp_path / "a.csv")
    second = tmp_path / "renamed.csv"
    second.write_bytes(Path(first).read_bytes())
    assert scope_key(first, "regression", "target", "r2") == scope_key(str(second), "regression", "target", "r2")


def test_reruns_preserve_artifacts_and_snapshot_inputs(tmp_path):
    path = dataset(tmp_path / "data.csv")
    kwargs = dict(dataset_path=path, objective="Predict target", max_iterations=1,
                  workspace_dir=str(tmp_path / "workspace"))
    first = run_agent(**kwargs)
    report = Path(first["report_path"]).read_bytes()
    model = Path(first["best_model_path"]).read_bytes()
    history = Path(first["run_experiments_path"]).read_bytes()
    second = run_agent(**kwargs)
    assert first["run_dir"] != second["run_dir"]
    assert first["experiments_path"] == second["experiments_path"]
    assert first["best_model_path"] != second["best_model_path"]
    assert Path(first["report_path"]).read_bytes() == report
    assert Path(first["best_model_path"]).read_bytes() == model
    assert Path(first["run_experiments_path"]).read_bytes() == history
    original = Path(path).read_bytes()
    dataset(Path(path), offset=100)
    assert Path(first["dataset_snapshot"]).read_bytes() == original
    for run in (first, second):
        for record in run["all_experiments"][-run["iterations_run"]:]:
            assert Path(record["code_path"]).is_file()
            assert Path(record["code_path"]).is_relative_to(Path(run["run_dir"]))


def test_concurrent_datasets_do_not_overwrite_reports(tmp_path):
    paths = [dataset(tmp_path / f"data{i}.csv", offset=i) for i in range(2)]
    def run(path):
        return run_agent(path, "Predict target", max_iterations=1,
                         workspace_dir=str(tmp_path / "workspace"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(run, paths))
    assert first["report_path"] != second["report_path"]
    assert first["best_model_path"] != second["best_model_path"]
    assert first["experiments_path"] != second["experiments_path"]


def test_explicit_history_rejects_different_dataset(tmp_path):
    path = dataset(tmp_path / "data.csv")
    kwargs = dict(dataset_path=path, objective="Predict target", max_iterations=1,
                  workspace_dir=str(tmp_path / "workspace"),
                  experiments_path=str(tmp_path / "history.json"))
    run_agent(**kwargs)
    original = Path(kwargs["experiments_path"]).read_bytes()
    dataset(Path(path), offset=50)
    with pytest.raises(ValueError, match="dataset|configuration"):
        run_agent(**kwargs)
    assert Path(kwargs["experiments_path"]).read_bytes() == original


def test_shared_history_is_locked_across_processes(tmp_path):
    import subprocess
    import sys
    from app.run_scope import history_lock
    path = str(tmp_path / "history.json")
    with history_lock(path):
        code = f"from app.run_scope import history_lock\nwith history_lock({path!r}):\n    pass"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode != 0
        assert "active run" in result.stderr
    with history_lock(path):
        pass


def test_distinct_explicit_histories_have_distinct_objective_files(tmp_path):
    from app.run_scope import default_objective_path
    assert default_objective_path(str(tmp_path / "a.json")) != default_objective_path(str(tmp_path / "b.json"))


def test_configuration_version_changes_scope(tmp_path, monkeypatch):
    import app.run_scope as scopes
    path = dataset(tmp_path / "data.csv")
    before = scope_key(path, "regression", "target", "r2")
    monkeypatch.setattr(scopes, "EXPERIMENT_CONFIG_VERSION", "next-version")
    assert scope_key(path, "regression", "target", "r2") != before


def test_uploads_with_same_filename_are_isolated(tmp_path):
    from app.run_scope import store_upload
    first = Path(store_upload(tmp_path, "../../data.csv", b"first"))
    second = Path(store_upload(tmp_path, "../../data.csv", b"second"))
    assert first != second
    assert first.is_relative_to(tmp_path) and second.is_relative_to(tmp_path)
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"


def test_copied_history_cannot_bypass_dataset_check(tmp_path):
    path = dataset(tmp_path / "data.csv")
    first = run_agent(path, "Predict target", max_iterations=1,
                      workspace_dir=str(tmp_path / "workspace"))
    copied = tmp_path / "copied.json"
    copied.write_bytes(Path(first["experiments_path"]).read_bytes())
    dataset(Path(path), offset=50)
    with pytest.raises(ValueError, match="dataset|configuration"):
        run_agent(path, "Predict target", max_iterations=1,
                  workspace_dir=str(tmp_path / "workspace"), experiments_path=str(copied))


def test_local_and_sandbox_histories_do_not_share_models(tmp_path):
    path = dataset(tmp_path / "data.csv")
    assert scope_key(path, "regression", "target", "r2", use_docker=True) != scope_key(path, "regression", "target", "r2")
