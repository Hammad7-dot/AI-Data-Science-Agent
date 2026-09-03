"""Behavioral regressions for validation boundaries and portable pipelines."""
import contextlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from app.coder import generate_code


def train(tmp_path, frame, mode, task="regression", dropped=None, metric=None):
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    model_name = "random_forest" if task == "classification" else "random_forest_regressor"
    plan = {
        "task_type": task, "target": "target",
        "metric": metric or ("f1" if task == "classification" else "r2"),
        "candidate_models": [model_name],
        "medium_cardinality_columns": ["city"],
        "dropped_high_cardinality_columns": dropped or [],
    }
    code = generate_code(str(path), plan, 1, model_name=model_name,
                         feature_engineering=mode, hyperparams={"n_estimators": 5},
                         models_dir=str(tmp_path / "models"))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "generated_contract.py", "exec"), {})
    result = json.loads(output.getvalue().strip().splitlines()[-1])
    return joblib.load(result["model_path"]), result


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_feature_selection_runs_with_more_than_ten_features(tmp_path, task):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(80, 12)), columns=[f"x{i}" for i in range(12)])
    frame["target"] = np.tile([0, 1], 40) if task == "classification" else frame.x0 * 3 + frame.x1
    model, result = train(tmp_path, frame, "pipeline", task)
    assert np.isfinite(model.predict(frame.drop(columns="target"))).all()
    assert result["n_features"] == 10
    assert len(result["feature_names"]) == 10
    if task == "regression":
        from sklearn.feature_selection import f_regression
        assert model.named_steps["preprocess"].named_steps["select"].score_func is f_regression


@pytest.mark.parametrize("mode", ["basic", "pipeline", "safe_categorical"])
def test_export_predicts_raw_missing_and_unseen_categories(tmp_path, mode):
    frame = pd.DataFrame({"age": np.arange(60, dtype=float),
                          "city": ["A", "B", "C"] * 20, "target": np.arange(60) * 2.0})
    frame.loc[::7, "age"] = np.nan
    model, _ = train(tmp_path, frame, mode)
    unseen = pd.DataFrame({"age": [np.nan, 25.0], "city": ["NEW", None]})
    assert np.isfinite(model.predict(unseen)).all()


def test_holdout_evaluation_excludes_features_dropped_during_training(tmp_path):
    """Catches raw dropped columns reaching a validated export at holdout time."""
    frame = pd.DataFrame({
        "age": np.arange(60, dtype=float),
        "city": ["A", "B", "C"] * 20,
        "opaque_id": [f"id_{index}" for index in range(60)],
        "target": np.arange(60) * 2.0,
    })
    _, result = train(tmp_path, frame, "basic", dropped=["opaque_id"])

    from tools.validation import evaluate_holdout
    holdout = evaluate_holdout(
        result["model_path"], str(tmp_path / "data.csv"), {
            "target": "target",
            "task_type": "regression",
            "metric": "r2",
            "dropped_high_cardinality_columns": ["opaque_id"],
            "leakage_dropped_columns": [],
        },
    )

    assert holdout["samples"] == 12
    assert np.isfinite(holdout["score"])


@pytest.mark.parametrize("mode", ["basic", "pipeline", "safe_categorical"])
def test_holdout_values_do_not_change_training_or_search_score(tmp_path, mode):
    frame = pd.DataFrame({"age": np.arange(60, dtype=float),
                          "city": ["A", "B", "C"] * 20, "target": np.arange(60) * 2.0})
    _, holdout = train_test_split(np.arange(60), test_size=0.2, random_state=42)
    first, result1 = train(tmp_path, frame, mode)
    probe = frame.drop(columns="target").iloc[:5].copy()
    frame.loc[holdout, "age"] = 1e9
    frame.loc[holdout, "city"] = "HOLDOUT_ONLY"
    frame.loc[holdout, "target"] = -1e9
    second, result2 = train(tmp_path, frame, mode)
    assert result1["score"] == pytest.approx(result2["score"])
    np.testing.assert_allclose(first.predict(probe), second.predict(probe))
    assert result1["validation_method"] == "out_of_fold"
    assert result1["cv_folds"] >= 2


def test_cross_validation_fits_imputation_inside_each_fold(tmp_path):
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score

    frame = pd.DataFrame({"age": np.arange(60, dtype=float), "target": np.arange(60) ** 2.0})
    frame.loc[::3, "age"] = np.nan
    _, result = train(tmp_path, frame, "basic")
    development, _ = train_test_split(frame, test_size=0.2, random_state=42)
    expected = cross_val_predict(
        make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                      RandomForestRegressor(n_estimators=5, random_state=42)),
        development[["age"]], development.target,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
    )
    assert result["score"] == pytest.approx(r2_score(development.target, expected))


@pytest.mark.parametrize(("task", "metric"), [("classification", "f1"), ("regression", "rmse")])
def test_generated_training_reports_fold_level_uncertainty(tmp_path, task, metric):
    """Catches missing fold validation data or population-deviation summaries."""
    if task == "classification":
        frame = pd.DataFrame({"x": np.arange(60), "target": np.tile([0, 1], 30)})
    else:
        frame = pd.DataFrame({"x": np.arange(60), "target": np.arange(60, dtype=float) ** 2})

    _, result = train(tmp_path, frame, "basic", task=task, metric=metric)

    assert len(result["cv_scores"]) == 3
    assert all(np.isfinite(result["cv_scores"]))
    assert result["cv_mean"] == pytest.approx(np.mean(result["cv_scores"]))
    assert result["cv_std"] == pytest.approx(np.std(result["cv_scores"], ddof=1))
    assert np.isfinite(result["cv_interval_95"]).all()
    assert result["cv_interval_95"][0] <= result["cv_mean"] <= result["cv_interval_95"][1]
    if metric == "rmse":
        assert all(score > 0 for score in result["cv_scores"])


def test_agent_reports_holdout_separately_from_selection(tmp_path):
    from app.agent import run_agent
    frame = pd.DataFrame({"x": np.arange(80, dtype=float), "target": np.arange(80) * 2.0})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    result = run_agent(str(path), "Predict target", target_column="target",
                       max_iterations=1, workspace_dir=str(tmp_path / "workspace"))
    assert result["holdout_evaluation"]["metric"] == "r2"
    assert result["holdout_evaluation"]["samples"] == 16
    assert result["holdout_evaluation"]["score"] == pytest.approx(1.0)
    assert "holdout_evaluation" not in result["best_experiment"]
    for key in ("cv_scores", "cv_mean", "cv_std", "cv_interval_95"):
        assert key in result["best_experiment"]
    assert "Final holdout" in open(result["report_path"], encoding="utf-8").read()


def test_holdout_does_not_influence_profile_or_automatic_target(tmp_path):
    from app.agent import run_agent
    frame = pd.DataFrame({"x": np.arange(80, dtype=float), "city": ["A", "B"] * 40,
                          "target": np.arange(80) * 2.0})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    first = run_agent(str(path), "Predict target", metric="rmse", max_iterations=1,
                      workspace_dir=str(tmp_path / "first"))
    _, holdout = train_test_split(np.arange(80), test_size=0.2, random_state=42)
    frame.loc[holdout, "city"] = [f"unseen_{i}" for i in holdout]
    frame.loc[holdout, "target"] = 1e9
    frame.to_csv(path, index=False)
    second = run_agent(str(path), "Predict target", metric="rmse", max_iterations=1,
                       workspace_dir=str(tmp_path / "second"))
    assert first["plan"]["target_score"] == second["plan"]["target_score"]
    assert first["plan"]["dropped_high_cardinality_columns"] == second["plan"]["dropped_high_cardinality_columns"]
    assert first["best_experiment"]["score"] == pytest.approx(second["best_experiment"]["score"])


def test_legacy_scores_cannot_be_mixed_with_cross_validation(tmp_path):
    from app.agent import run_agent
    frame = pd.DataFrame({"x": np.arange(80, dtype=float), "target": np.arange(80) * 2.0})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    history = tmp_path / "experiments.json"
    original = json.dumps([{"score": 1.0, "metric": "r2", "model": "linear_regression"}])
    history.write_text(original)
    with pytest.raises(ValueError, match="validation.*reset"):
        run_agent(str(path), "Predict target", max_iterations=0,
                  workspace_dir=str(tmp_path / "workspace"), experiments_path=str(history))
    assert history.read_text() == original


def test_encoding_cap_is_checked_before_feature_selection(monkeypatch):
    import tools.ml as ml
    monkeypatch.setattr(ml, "MAX_ONEHOT_COLUMNS", 10)
    frame = pd.DataFrame(np.ones((8, 11)), columns=[f"x{i}" for i in range(11)])
    preprocessor = ml.build_preprocessing_pipeline(list(frame.columns), [], feature_selection_k=3)
    with pytest.raises(RuntimeError, match="exceeding the safety cap"):
        preprocessor.fit_transform(frame, [0, 1] * 4)


def test_frequency_encoder_keeps_training_mapping_at_prediction_time():
    from tools.ml import FrequencyEncoder
    encoder = FrequencyEncoder().fit(pd.DataFrame({"city": ["A", "A", "A", "B"]}))
    actual = encoder.transform(pd.DataFrame({"city": ["B", "A", "NEW", None]}))
    np.testing.assert_allclose(actual[:, 0], [0.25, 0.75, 0.0, 0.0])


def test_frequency_pipeline_can_be_loaded_in_a_new_python_process(tmp_path):
    from pathlib import Path
    from tools.python_runner import run_script
    frame = pd.DataFrame({"age": np.arange(60, dtype=float),
                          "city": ["A", "B", "C"] * 20, "target": np.arange(60) * 2.0})
    _, result = train(tmp_path, frame, "safe_categorical")
    code = f"""
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
import joblib
import pandas as pd
import numpy as np
model = joblib.load({result['model_path']!r})
predictions = model.predict(pd.DataFrame({{'age': [None, 25.0], 'city': ['NEW', None]}}))
assert len(predictions) == 2 and np.isfinite(predictions).all()
"""
    execution = run_script(code, str(tmp_path / "inference"))
    assert execution.success, execution.stderr


def test_run_without_a_model_does_not_leave_a_stale_holdout_score(tmp_path):
    from app.agent import run_agent
    frame = pd.DataFrame({"x": np.arange(80, dtype=float), "target": np.arange(80) * 2.0})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    reports = tmp_path / "workspace" / "reports"
    reports.mkdir(parents=True)
    artifact = reports / "holdout.json"
    artifact.write_text('{"score": 1.0}')
    result = run_agent(str(path), "Predict target", max_iterations=0,
                       workspace_dir=str(tmp_path / "workspace"))
    new_artifact = Path(result["report_path"]).with_name("holdout.json")
    assert new_artifact != artifact
    assert "score" not in json.loads(new_artifact.read_text())
