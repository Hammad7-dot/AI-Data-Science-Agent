import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import honest_summary_sentence, _build_report


def _fake_result(status="max_iterations_reached"):
    return {
        "status": status,
        "iterations_run": 15,
        "plan": {
            "metric": "f1",
            "direction": "maximize",
            "target_score": 0.999,
            "task_type": "classification",
            "target": "churn",
            "candidate_models": ["random_forest", "knn"],
            "steps": [],
        },
        "profile": {"rows": 100, "columns": 5, "missing_values": 0, "duplicate_rows": 0, "target": {}},
        "best_experiment": {
            "iteration": 11,
            "model": "random_forest",
            "metric": "f1",
            "score": 0.8667,
            "status": "fail",
            "hyperparams": {"max_depth": 10, "n_estimators": 200},
            "feature_engineering": "pipeline",
        },
        "all_experiments": [],
        "objective_changed": False,
    }


def test_honest_summary_sentence_format_maximize():
    result = _fake_result()
    sentence = honest_summary_sentence(result)
    assert sentence == (
        "Target f1 >= 0.999 was not achieved. Best observed f1 was 0.8667 using "
        "random_forest (max_depth=10, n_estimators=200, feature_engineering=pipeline)."
    )


def test_honest_summary_sentence_format_minimize():
    result = _fake_result()
    result["plan"]["direction"] = "minimize"
    result["plan"]["metric"] = "rmse"
    result["plan"]["target_score"] = 1.0
    result["best_experiment"]["metric"] = "rmse"
    result["best_experiment"]["score"] = 2.35
    sentence = honest_summary_sentence(result)
    assert sentence.startswith("Target rmse <= 1.0 was not achieved. Best observed rmse was 2.3500")


def test_honest_summary_sentence_appears_in_report_when_not_success():
    result = _fake_result(status="search_space_exhausted")
    report = _build_report("Predict churn", 0.999, result)
    assert honest_summary_sentence(result) in report
    assert "## Honest result summary" in report


def test_honest_summary_absent_when_success():
    result = _fake_result(status="success")
    report = _build_report("Predict churn", 0.999, result)
    assert "## Honest result summary" not in report
