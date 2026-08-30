import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.planner import create_plan


def test_create_plan_drops_leakage_columns_and_excludes_from_encoding():
    """Regression test for the games.csv report: the profiler flagging
    'winner'/'victory_status' as leakage must actually remove them from
    leakage_dropped_columns AND keep them out of low/medium
    cardinality_columns (which drive one-hot/frequency encoding) -- a
    column can't be simultaneously "dropped as leakage" and "encoded as
    a feature".
    """
    profile = {
        "target": {"name": "opening_ply", "suggested_task_type": "regression"},
        "categorical_cardinality": {
            "rated": {"nunique": 2, "unique_ratio": 0.02, "bucket": "low", "id_like": False},
            "winner": {"nunique": 3, "unique_ratio": 0.03, "bucket": "low", "id_like": False},
            "victory_status": {"nunique": 4, "unique_ratio": 0.04, "bucket": "low", "id_like": False},
        },
        "high_cardinality_columns": [],
        "leakage_warnings": [
            {"column": "winner", "reason": "outcome info", "risk": "high", "recommended_action": "drop"},
            {"column": "victory_status", "reason": "outcome info", "risk": "high", "recommended_action": "drop"},
        ],
    }
    plan = create_plan(profile, "predict opening_ply")
    assert set(plan["leakage_dropped_columns"]) == {"winner", "victory_status"}
    assert "winner" not in plan["low_cardinality_columns"]
    assert "victory_status" not in plan["low_cardinality_columns"]
    assert "rated" in plan["low_cardinality_columns"]


def test_create_plan_classification():
    profile = {
        "target": {
            "name": "churn",
            "suggested_task_type": "classification",
        }
    }
    plan = create_plan(profile, "Predict churn")
    assert plan["task_type"] == "classification"
    assert plan["candidate_models"]
    assert "metric" in plan


def test_create_plan_regression():
    profile = {
        "target": {
            "name": "price",
            "suggested_task_type": "regression",
        }
    }
    plan = create_plan(profile, "Predict price")
    assert plan["task_type"] == "regression"
    assert plan["candidate_models"]
    assert "metric" in plan


def test_create_plan_eda_fallback():
    profile = {"target": None}
    plan = create_plan(profile, "Explore this data")
    assert plan["task_type"] == "eda"
    assert plan["candidate_models"] == []
