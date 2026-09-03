import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.dataset import profile_dataset
from app.coder import generate_code, MAX_ONEHOT_COLUMNS
from app.evaluator import evaluate, _classify_error
from app.experiment_memory import ExperimentMemory
from app.experiment_strategist import choose_next_experiment
from app.ralph import _is_better
from tools.python_runner import ExecutionResult


def test_profile_dataset_cardinality_buckets(tmp_path):
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            # nunique/n_rows must be <= 0.01 (LOW_CARDINALITY_MAX_RATIO) for
            # "low": with n=100 rows that means a single repeated value.
            "low_card": ["a"] * n,  # 1 unique / 100 = 0.01 -> low
            "med_card": rng.choice([f"g{i}" for i in range(15)], size=n),  # 15/100=0.15 -> medium
            "id_col": [f"row_{i}" for i in range(n)],  # 100 unique -> high/id_like
            "target": rng.integers(0, 2, size=n),
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    profile = profile_dataset(str(csv_path), target_hint="target")

    cc = profile["categorical_cardinality"]
    assert cc["low_card"]["bucket"] == "low"
    assert cc["med_card"]["bucket"] == "medium"
    assert cc["id_col"]["bucket"] == "high"
    assert cc["id_col"]["id_like"] is True
    assert "id_col" in profile["high_cardinality_columns"]
    assert "low_card" not in profile["high_cardinality_columns"]


def test_generated_code_has_onehot_safety_cap():
    plan = {
        "task_type": "regression",
        "target": "y",
        "candidate_models": ["linear_regression"],
        "metric": "r2",
        "dropped_high_cardinality_columns": ["Game"],
    }
    code = generate_code("dummy.csv", plan, iteration=1, model_name="linear_regression",
                          feature_engineering="basic", hyperparams={})
    assert f"MAX_ONEHOT_COLUMNS = {MAX_ONEHOT_COLUMNS}" in code
    assert "build_preprocessing_pipeline" in code
    assert "DROPPED_HIGH_CARDINALITY_COLUMNS = ['Game']" in code


def test_classify_error_memory_error():
    result = ExecutionResult(
        success=False, stdout="", stderr="Traceback...\nMemoryError: unable to allocate",
        exit_code=1, duration_seconds=1.0, artifacts=[], timed_out=False,
    )
    plan = {"metric": "r2"}
    out = evaluate(result, plan, target_score=0.8)
    assert out["status"] == "fail"
    assert out["error_type"] == "MemoryError"


def test_classify_error_timeout():
    result = ExecutionResult(
        success=False, stdout="", stderr="", exit_code=-1,
        duration_seconds=60.0, artifacts=[], timed_out=True,
    )
    assert _classify_error(result, "") == "TimeoutError"


def test_strategist_switches_to_pipeline_after_memory_error():
    history = [
        {
            "iteration": 1,
            "model": "random_forest",
            "score": None,
            "status": "fail",
            "error_type": "MemoryError",
            "hyperparams": {},
            "feature_engineering": "basic",
        }
    ]
    memory = ExperimentMemory(history)
    plan = {
        "task_type": "classification",
        "metric": "f1",
        "direction": "maximize",
        "candidate_models": ["random_forest"],
    }
    choice = choose_next_experiment(plan, memory, iteration=2)
    assert choice is not None
    assert choice["model_name"] == "random_forest"
    assert choice["feature_engineering"] == "pipeline"
    assert "MemoryError" in choice["rationale"]


def test_is_better_rejects_none_score_as_first_candidate():
    failed_candidate = {"score": None, "status": "fail"}
    assert _is_better(failed_candidate, None, "f1") is False


def test_create_plan_recommends_safe_categorical_for_high_estimate():
    from app.planner import create_plan

    profile = {
        "target": {"name": "y", "suggested_task_type": "regression"},
        "categorical_cardinality": {
            "pub": {"nunique": 1200, "unique_ratio": 0.06, "bucket": "medium", "id_like": False},
            "dev": {"nunique": 1500, "unique_ratio": 0.075, "bucket": "medium", "id_like": False},
        },
        "high_cardinality_columns": [],
    }
    plan = create_plan(profile, "predict y")
    assert plan["recommended_feature_engineering"] == "safe_categorical"
    assert plan["estimated_onehot_columns"] == 2700


def test_create_plan_recommends_basic_for_low_estimate():
    from app.planner import create_plan

    profile = {
        "target": {"name": "y", "suggested_task_type": "regression"},
        "categorical_cardinality": {
            "small": {"nunique": 5, "unique_ratio": 0.005, "bucket": "low", "id_like": False},
        },
        "high_cardinality_columns": [],
    }
    plan = create_plan(profile, "predict y")
    assert plan["recommended_feature_engineering"] == "basic"
    assert plan["estimated_onehot_columns"] == 5


def test_generated_code_safe_categorical_has_frequency_encoding():
    plan = {
        "task_type": "regression",
        "target": "y",
        "candidate_models": ["linear_regression"],
        "metric": "r2",
        "dropped_high_cardinality_columns": ["id_col"],
        "low_cardinality_columns": ["low_card"],
        "medium_cardinality_columns": ["med_card"],
    }
    code = generate_code("dummy.csv", plan, iteration=1, model_name="linear_regression",
                          feature_engineering="safe_categorical", hyperparams={})
    assert 'frequency_cols=frequency_cols' in code
    assert "MEDIUM_CARDINALITY_COLUMNS = ['med_card']" in code

    assert f"MAX_ONEHOT_COLUMNS = {MAX_ONEHOT_COLUMNS}" in code


def test_classify_error_high_cardinality_encoding():
    result = ExecutionResult(
        success=False, stdout="",
        stderr="RuntimeError: one-hot encoding produced 2250 columns from 20058 rows, "
               "exceeding the safety cap of 2000; likely a high-cardinality column was not dropped",
        exit_code=1, duration_seconds=1.0, artifacts=[], timed_out=False,
    )
    plan = {"metric": "r2"}
    out = evaluate(result, plan, target_score=0.8)
    assert out["status"] == "fail"
    assert out["error_type"] == "HIGH_CARDINALITY_ENCODING"


def test_choose_next_experiment_uses_recommended_safe_categorical():
    memory = ExperimentMemory([])
    plan = {
        "task_type": "regression",
        "metric": "r2",
        "direction": "maximize",
        "candidate_models": ["linear_regression"],
        "recommended_feature_engineering": "safe_categorical",
        "estimated_onehot_columns": 2250,
        "dropped_high_cardinality_columns": ["id_col"],
    }
    choice = choose_next_experiment(plan, memory, iteration=1)
    assert choice is not None
    assert choice["feature_engineering"] == "safe_categorical"
    assert "2250" in choice["rationale"]


def test_second_exploration_trial_uses_gap_context_not_profiling_repeat():
    """Regression test: once a real scored result exists, later
    exploration trials for other candidate models must explain the
    choice via the previous result's gap to target, not repeat the
    'dataset profiling estimated ~N one-hot columns' explanation every
    time -- that explanation is a one-time, first-trial-only rationale.
    """
    history = [
        {
            "iteration": 1,
            "model": "linear_regression",
            "score": 0.55,
            "status": "fail",
            "hyperparams": {},
            "feature_engineering": "safe_categorical",
        }
    ]
    memory = ExperimentMemory(history)
    plan = {
        "task_type": "regression",
        "metric": "r2",
        "target_score": 0.9,
        "direction": "maximize",
        "candidate_models": ["linear_regression", "random_forest_regressor"],
        "recommended_feature_engineering": "safe_categorical",
        "estimated_onehot_columns": 2250,
        "dropped_high_cardinality_columns": [],
    }
    choice = choose_next_experiment(plan, memory, iteration=2)
    assert choice is not None
    assert choice["model_name"] == "random_forest_regressor"
    assert "one-hot columns" not in choice["rationale"], "should not repeat the profiling explanation on a later trial"
    assert "0.55" in choice["rationale"] or "0.5500" in choice["rationale"]
    assert "short of the 0.9 target" in choice["rationale"]


def test_strategist_switches_to_safe_categorical_after_high_cardinality_error():
    history = [
        {
            "iteration": 1,
            "model": "random_forest",
            "score": None,
            "status": "fail",
            "error_type": "HIGH_CARDINALITY_ENCODING",
            "hyperparams": {},
            "feature_engineering": "basic",
        }
    ]
    memory = ExperimentMemory(history)
    plan = {
        "task_type": "classification",
        "metric": "f1",
        "direction": "maximize",
        "candidate_models": ["random_forest"],
    }
    choice = choose_next_experiment(plan, memory, iteration=2)
    assert choice is not None
    assert choice["model_name"] == "random_forest"
    assert choice["feature_engineering"] == "safe_categorical"

