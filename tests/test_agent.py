from pathlib import Path

import numpy as np
import pandas as pd

from app.agent import _build_report, run_agent


def test_report_surfaces_model_trust_details(tmp_path):
    """Catches reports that hide cross-validation, schema, or explanations."""
    dataset = tmp_path / "training.csv"
    pd.DataFrame(
        {
            "age": np.arange(60, dtype=float),
            "city": ["north", "south"] * 30,
            "target": np.tile([0, 1], 30),
        }
    ).to_csv(dataset, index=False)

    result = run_agent(
        str(dataset),
        "Predict target",
        target_column="target",
        max_iterations=1,
        workspace_dir=str(tmp_path / "workspace"),
    )

    report = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "## Validation uncertainty" in report
    assert "95% descriptive interval" in report
    assert "## Raw input schema" in report
    assert "## Model explanation" in report
    assert result["model_schema"]["schema_version"] == 1
    assert "available" in result["explainability"]


def test_report_formats_trust_metadata_and_limits_explanations_to_ten_rows():
    """Catches a report that obscures validation detail or floods readers with features."""
    features = [
        {"feature": f"feature_{index}", "importance": index / 10, "coefficient": -index / 10}
        for index in range(12, 0, -1)
    ]
    result = {
        "profile": {"rows": 30, "columns": 3, "missing_values": 0, "duplicate_rows": 0},
        "plan": {
            "task_type": "classification",
            "metric": "f1",
            "target_score": 0.8,
            "direction": "maximize",
            "target": "target",
            "candidate_models": ["logistic_regression"],
            "steps": [],
        },
        "all_experiments": [],
        "best_experiment": {
            "model": "logistic_regression",
            "metric": "f1",
            "score": 0.9,
            "status": "success",
            "cv_scores": [0.8, 0.9, 1.0],
            "cv_mean": 0.9,
            "cv_std": 0.1,
            "cv_interval_95": [0.7868, 1.0132],
        },
        "model_schema": {
            "schema_version": 1,
            "raw_columns": ["age", "city"],
            "dtype_families": {"age": "numeric", "city": "categorical"},
            "target_name": "target",
        },
        "explainability": {"available": True, "method": "coefficient", "features": features, "reason": None},
        "status": "success",
        "iterations_run": 0,
    }

    report = _build_report("Predict target", 0.8, result)

    assert "- Fold scores: 0.8000, 0.9000, 1.0000" in report
    assert "- Mean ± standard deviation: 0.9000 ± 0.1000" in report
    assert "- 95% descriptive interval: [0.7868, 1.0132]" in report
    assert "| age | numeric |" in report
    assert "| city | categorical |" in report
    assert "| 1 | feature_12 | 1.2000 | -1.2000 |" in report
    assert "| 10 | feature_3 | 0.3000 | -0.3000 |" in report
    assert "feature_2" not in report


def test_cli_prints_validation_and_promoted_trust_artifact_paths(tmp_path, monkeypatch, capsys):
    """Catches a CLI summary that omits the metadata needed to inspect a winner."""
    from app import main

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    best_model_path = models_dir / "best_model.joblib"
    best_model_path.write_bytes(b"model")
    schema_path = models_dir / "schema.json"
    explanation_path = models_dir / "explainability.json"
    schema_path.write_text("{}", encoding="utf-8")
    explanation_path.write_text("{}", encoding="utf-8")
    result = {
        "status": "success",
        "iterations_run": 1,
        "stop_mode": "target",
        "best_experiment": {
            "model": "logistic_regression",
            "metric": "f1",
            "score": 0.9,
            "cv_scores": [0.8, 0.9, 1.0],
            "cv_mean": 0.9,
            "cv_std": 0.1,
            "cv_interval_95": [0.7868, 1.0132],
        },
        "plan": {"task_type": "classification", "metric": "f1", "target_score": 0.8, "direction": "maximize"},
        "all_experiments": [],
        "report_path": str(tmp_path / "report.md"),
        "best_model_path": str(best_model_path),
    }
    monkeypatch.setattr(main, "run_agent", lambda **_: result)

    assert main.main(["--dataset", str(tmp_path / "training.csv"), "--objective", "Predict target"]) == 0

    output = capsys.readouterr().out
    assert "Validation uncertainty:" in output
    assert "Fold scores: 0.8000, 0.9000, 1.0000" in output
    assert "95% descriptive interval: [0.7868, 1.0132]" in output
    assert f"Raw input schema artifact: {schema_path}" in output
    assert f"Model explanation artifact: {explanation_path}" in output
