import copy
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _trust_result(best_model_path=None, *, model_schema=None, explainability=None):
    return {
        "status": "success",
        "iterations_run": 1,
        "stop_mode": "target",
        "sandbox_requested": bool(best_model_path),
        "best_model_path": str(best_model_path) if best_model_path else None,
        "best_experiment": {
            "iteration": 1,
            "model": "logistic_regression",
            "metric": "f1",
            "score": 0.9,
            "status": "success",
            "feature_engineering": "pipeline",
            "hyperparams": {},
            "cv_scores": [0.8, 0.9, 1.0],
            "cv_mean": 0.9,
            "cv_std": 0.1,
            "cv_interval_95": [0.7868, 1.0132],
        },
        "model_schema": model_schema,
        "explainability": explainability,
        "holdout_evaluation": {"metric": "f1", "score": 0.88, "samples": 12},
        "all_experiments": [],
        "plan": {
            "task_type": "classification",
            "metric": "f1",
            "target_score": 0.8,
            "direction": "maximize",
            "target": "target",
        },
    }


def _run_with_result(monkeypatch, result):
    import app.agent
    import app.run_scope

    monkeypatch.setattr(app.agent, "run_agent", lambda **_: copy.deepcopy(result))
    monkeypatch.setattr(app.run_scope, "store_upload", lambda *_: "training.csv")
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    at.file_uploader[0].set_value(("training.csv", b"age,target\n21,1\n", "text/csv"))
    at.button[0].click().run(timeout=30)
    assert not at.exception
    return at


def test_streamlit_renders_validation_schema_and_explanation_from_result(monkeypatch):
    """Catches an interface that discards trust metadata returned by the agent."""
    schema = {
        "schema_version": 1,
        "raw_columns": ["age", "city"],
        "dtype_families": {"age": "numeric", "city": "categorical"},
        "target_name": "target",
    }
    explanation = {
        "available": True,
        "method": "coefficient",
        "features": [
            {"feature": "age", "importance": 0.8, "coefficient": 0.8},
            {"feature": "city_north", "importance": 0.2, "coefficient": -0.2},
        ],
        "reason": None,
    }

    at = _run_with_result(monkeypatch, _trust_result(model_schema=schema, explainability=explanation))

    assert {metric.label for metric in at.metric} >= {
        "CV mean",
        "CV std",
        "95% descriptive interval",
        "Final holdout f1",
    }
    assert {heading.value for heading in at.subheader} >= {"Raw input schema", "Model explanation"}
    assert any(list(frame.value["column"]) == ["age", "city"] for frame in at.dataframe)
    assert any(
        "feature" in frame.value and list(frame.value["feature"]) == ["age", "city_north"]
        for frame in at.dataframe
    )


def test_streamlit_uses_promoted_json_when_result_lacks_trust_metadata(tmp_path, monkeypatch):
    """Catches fallback paths that would otherwise hide sandbox-produced trust metadata."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    best_model_path = models_dir / "best_model.joblib"
    best_model_path.write_bytes(b"not deserialized")
    (models_dir / "schema.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "raw_columns": ["age"],
                "dtype_families": {"age": "numeric"},
                "target_name": "target",
            }
        ),
        encoding="utf-8",
    )
    (models_dir / "explainability.json").write_text(
        json.dumps(
            {
                "available": False,
                "method": None,
                "features": [],
                "reason": "The estimator has no native importance values.",
            }
        ),
        encoding="utf-8",
    )

    at = _run_with_result(monkeypatch, _trust_result(best_model_path))

    assert any(list(frame.value["column"]) == ["age"] for frame in at.dataframe)
    assert any("The estimator has no native importance values." in caption.value for caption in at.caption)
