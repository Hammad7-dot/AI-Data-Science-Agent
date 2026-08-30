import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from app.coder import generate_code
from tools.python_runner import run_script

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = str(PROJECT_ROOT / "workspace" / "datasets" / "sample_churn.csv")


def test_generated_code_drops_leakage_columns():
    """Regression test: the profiler flagging a column as likely leakage
    must actually change the generated code, not just print a warning
    that the training script ignores. Covers all three feature-
    engineering modes since leakage columns must never survive any of
    them.
    """
    plan = {
        "task_type": "regression",
        "target": "opening_ply",
        "candidate_models": ["random_forest_regressor"],
        "metric": "r2",
        "dropped_high_cardinality_columns": ["id"],
        "leakage_dropped_columns": ["winner", "victory_status", "last_move_at"],
        "low_cardinality_columns": ["rated"],
        "medium_cardinality_columns": [],
    }
    for fe in ("basic", "pipeline", "safe_categorical"):
        code = generate_code(
            "dummy.csv", plan, iteration=1, model_name="random_forest_regressor",
            feature_engineering=fe, hyperparams={},
        )
        assert "DROPPED_LEAKAGE_COLUMNS = ['winner', 'victory_status', 'last_move_at']" in code
        assert "DROPPED_HIGH_CARDINALITY_COLUMNS + DROPPED_LEAKAGE_COLUMNS" in code


def _classification_plan(metric=None):
    return {
        "task_type": "classification",
        "target": "churn",
        "metric": metric or "f1",
        "candidate_models": ["random_forest"],
        "steps": [],
    }


def _regression_plan(metric=None):
    return {
        "task_type": "regression",
        "target": "monthly_charges",
        "metric": metric or "rmse",
        "candidate_models": ["random_forest_regressor"],
        "steps": [],
    }


def test_none_hyperparam_does_not_produce_literal_null(tmp_path):
    plan = _classification_plan()
    hyperparams = {"max_depth": None, "n_estimators": 50}
    code = generate_code(
        DATASET_PATH,
        plan,
        iteration=0,
        model_name="random_forest",
        feature_engineering="basic",
        hyperparams=hyperparams,
    )

    hp_line = [line for line in code.splitlines() if line.startswith("HYPERPARAMS")][0]
    assert "null" not in hp_line
    assert "None" in hp_line

    # Must be valid Python.
    ast.parse(code)


def test_none_hyperparam_script_executes_successfully(tmp_path):
    plan = _classification_plan()
    hyperparams = {"max_depth": None, "n_estimators": 50}
    code = generate_code(
        DATASET_PATH,
        plan,
        iteration=0,
        model_name="random_forest",
        feature_engineering="basic",
        hyperparams=hyperparams,
    )

    result = run_script(code, str(tmp_path), filename="null_hp_iteration.py")

    # The real regression test: this used to crash with
    # NameError: name 'null' is not defined because HYPERPARAMS was
    # embedded via json.dumps() instead of repr().
    assert result.success is True, result.stderr
    assert "NameError" not in (result.stderr or "")
    last_line = result.stdout.strip().splitlines()[-1]
    parsed = json.loads(last_line)
    assert parsed["hyperparams"]["max_depth"] is None


def test_regression_metric_r2_is_honored(tmp_path):
    plan = _regression_plan(metric="r2")
    code = generate_code(
        DATASET_PATH,
        plan,
        iteration=0,
        model_name="linear_regression",
        feature_engineering="basic",
        hyperparams={},
    )
    result = run_script(code, str(tmp_path), filename="r2_iteration.py")
    assert result.success is True, result.stderr

    import json

    last_line = result.stdout.strip().splitlines()[-1]
    parsed = json.loads(last_line)
    assert parsed["metric"] == "r2"
    # r2 is typically in (-inf, 1], not rmse-scale.
    assert parsed["score"] == parsed["r2"]
    assert parsed["score"] != parsed["rmse"]


def test_classification_metric_accuracy_is_honored(tmp_path):
    plan = _classification_plan(metric="accuracy")
    code = generate_code(
        DATASET_PATH,
        plan,
        iteration=0,
        model_name="logistic_regression",
        feature_engineering="basic",
        hyperparams={},
    )
    result = run_script(code, str(tmp_path), filename="accuracy_iteration.py")
    assert result.success is True, result.stderr

    import json

    last_line = result.stdout.strip().splitlines()[-1]
    parsed = json.loads(last_line)
    assert parsed["metric"] == "accuracy"
    assert parsed["score"] == parsed["accuracy"]


def test_generate_code_backward_compatible_without_overrides():
    plan = _classification_plan()
    # No explicit model_name/feature_engineering/hyperparams: should still
    # derive from iteration like before.
    code = generate_code(DATASET_PATH, plan, iteration=1)
    ast.parse(code)


def test_generate_code_without_models_dir_has_no_model_path(tmp_path):
    """Backward compatible: omitting models_dir should not crash, and
    model_path should just be None in the printed result."""
    plan = _classification_plan()
    code = generate_code(
        DATASET_PATH,
        plan,
        iteration=0,
        model_name="random_forest",
        feature_engineering="basic",
        hyperparams={"n_estimators": 20},
    )
    result = run_script(code, str(tmp_path), filename="no_models_dir_iteration.py")
    assert result.success is True, result.stderr
    parsed = json.loads(result.stdout.strip().splitlines()[-1])
    assert parsed["model_path"] is None


def _run_and_check_saved_model(tmp_path, feature_engineering, model_name, plan):
    models_dir = str(tmp_path / "models")
    code = generate_code(
        DATASET_PATH,
        plan,
        iteration=3,
        model_name=model_name,
        feature_engineering=feature_engineering,
        hyperparams={},
        models_dir=models_dir,
    )
    result = run_script(code, str(tmp_path), filename=f"model_save_{feature_engineering}.py")
    assert result.success is True, result.stderr

    last_line = result.stdout.strip().splitlines()[-1]
    parsed = json.loads(last_line)
    assert "model_path" in parsed
    model_path = parsed["model_path"]
    assert model_path is not None
    assert Path(model_path).exists()

    loaded = joblib.load(model_path)
    assert hasattr(loaded, "predict")
    return loaded


def test_classification_basic_saves_model_to_disk(tmp_path):
    plan = _classification_plan()
    loaded = _run_and_check_saved_model(tmp_path, "basic", "random_forest", plan)
    # Basic path fits the bare estimator directly -- it should NOT be a
    # full Pipeline (no named_steps).
    assert not hasattr(loaded, "named_steps")


def test_classification_pipeline_saves_full_pipeline_to_disk(tmp_path):
    plan = _classification_plan()
    loaded = _run_and_check_saved_model(tmp_path, "pipeline", "random_forest", plan)
    # Pipeline path must save the WHOLE fitted Pipeline (preprocessing +
    # model), not just the inner estimator -- required for inference on
    # new raw data later.
    assert hasattr(loaded, "named_steps")
    assert "preprocess" in loaded.named_steps
    assert "model" in loaded.named_steps


def test_regression_pipeline_saves_full_pipeline_to_disk(tmp_path):
    plan = _regression_plan()
    loaded = _run_and_check_saved_model(tmp_path, "pipeline", "random_forest_regressor", plan)
    assert hasattr(loaded, "named_steps")


def test_global_iteration_number_in_header_comment():
    plan = _classification_plan()
    code = generate_code(
        DATASET_PATH,
        plan,
        iteration=17,
        model_name="random_forest",
        feature_engineering="basic",
        hyperparams={},
    )
    assert code.splitlines()[0] == "# Iteration 17 - generated by AI Data Science Agent"
    assert "ITERATION = 17" in code


def test_basic_feature_engineering_has_no_pipeline_imports():
    plan = _classification_plan()
    code = generate_code(
        DATASET_PATH,
        plan,
        iteration=1,
        model_name="random_forest",
        feature_engineering="basic",
        hyperparams={},
    )
    assert "from sklearn.pipeline import Pipeline" not in code
    assert "build_preprocessing_pipeline" not in code
    assert "from tools.ml import get_model" in code


def test_pipeline_feature_engineering_has_pipeline_imports():
    plan = _classification_plan()
    code = generate_code(
        DATASET_PATH,
        plan,
        iteration=1,
        model_name="random_forest",
        feature_engineering="pipeline",
        hyperparams={},
    )
    assert "from sklearn.pipeline import Pipeline" in code
    assert "from tools.ml import get_model, build_preprocessing_pipeline" in code


def test_invalid_metric_raises_value_error_at_runtime(tmp_path):
    plan = _classification_plan(metric="not_a_real_metric")
    code = generate_code(
        DATASET_PATH,
        plan,
        iteration=1,
        model_name="random_forest",
        feature_engineering="basic",
        hyperparams={},
    )
    result = run_script(code, str(tmp_path))
    assert not result.success
    assert "not_a_real_metric" in (result.stderr or "")
