import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.reporting import target_gap_text, diminishing_return_note


def test_target_gap_text_maximize():
    plan = {"target_score": 0.90, "direction": "maximize"}
    best = {"score": 0.8680868300748117}
    assert target_gap_text(plan, best) == "0.0319"


def test_target_gap_text_none_when_no_best():
    assert target_gap_text({"target_score": 0.9, "direction": "maximize"}, None) is None
    assert target_gap_text({"target_score": 0.9, "direction": "maximize"}, {"score": None}) is None


def test_diminishing_return_note_flags_flattened_n_estimators():
    all_experiments = [
        {"iteration": 1, "model": "random_forest_regressor", "score": 0.86630, "hyperparams": {"n_estimators": 150}},
        {"iteration": 2, "model": "random_forest_regressor", "score": 0.86754, "hyperparams": {"n_estimators": 225}},
        {"iteration": 3, "model": "random_forest_regressor", "score": 0.86781, "hyperparams": {"n_estimators": 338}},
        {"iteration": 4, "model": "random_forest_regressor", "score": 0.8680224360853683, "hyperparams": {"n_estimators": 507}},
        {"iteration": 5, "model": "random_forest_regressor", "score": 0.8680868300748117, "hyperparams": {"n_estimators": 760}},
    ]
    best = all_experiments[-1]
    note = diminishing_return_note(best, all_experiments, direction="maximize")
    assert note is not None
    assert "n_estimators" in note
    assert "max_depth" in note  # the suggested alternative hyperparameter


def test_diminishing_return_note_none_without_signal():
    all_experiments = [
        {"iteration": 1, "model": "random_forest_regressor", "score": 0.70, "hyperparams": {"n_estimators": 100}},
        {"iteration": 2, "model": "random_forest_regressor", "score": 0.85, "hyperparams": {"n_estimators": 200}},
    ]
    best = all_experiments[-1]
    assert diminishing_return_note(best, all_experiments, direction="maximize") is None
