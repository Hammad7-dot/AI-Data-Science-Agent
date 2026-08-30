import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.experiment_memory import ExperimentMemory
from app.experiment_strategist import choose_next_experiment
from tools.ml import hyperparameter_grid, iter_hyperparam_combos


def _base_plan():
    return {
        "task_type": "classification",
        "metric": "f1",
        "direction": "maximize",
        "candidate_models": ["random_forest", "gradient_boosting"],
    }


def test_diminishing_returns_stops_expanding_the_same_key():
    """Regression test for the 'agent keeps blindly increasing
    n_estimators forever' bug report: once the marginal gain from
    scaling n_estimators has flattened (507->760 gained only 0.00006),
    the next choice must not be a further n_estimators expansion on
    the same model/feature_engineering -- it should fall to the static
    grid (trying a different hyperparameter, e.g. max_depth) instead.
    """
    history = [
        {"iteration": 1, "model": "random_forest_regressor", "score": 0.80, "hyperparams": {}, "feature_engineering": "basic"},
        {"iteration": 2, "model": "gradient_boosting_regressor", "score": 0.70, "hyperparams": {}, "feature_engineering": "basic"},
        {"iteration": 3, "model": "random_forest_regressor", "score": 0.86630, "hyperparams": {"n_estimators": 150}, "feature_engineering": "basic"},
        {"iteration": 4, "model": "random_forest_regressor", "score": 0.86754, "hyperparams": {"n_estimators": 225}, "feature_engineering": "basic"},
        {"iteration": 5, "model": "random_forest_regressor", "score": 0.86781, "hyperparams": {"n_estimators": 338}, "feature_engineering": "basic"},
        {"iteration": 6, "model": "random_forest_regressor", "score": 0.8680224360853683, "hyperparams": {"n_estimators": 507}, "feature_engineering": "basic"},
        {"iteration": 7, "model": "random_forest_regressor", "score": 0.8680868300748117, "hyperparams": {"n_estimators": 760}, "feature_engineering": "basic"},
    ]
    memory = ExperimentMemory(history)
    plan = {
        "task_type": "regression",
        "metric": "r2",
        "target_score": 0.90,
        "direction": "maximize",
        "candidate_models": ["random_forest_regressor", "gradient_boosting_regressor"],
    }

    choice = choose_next_experiment(plan, memory, iteration=8)
    assert choice is not None
    if choice["model_name"] == "random_forest_regressor":
        assert choice["hyperparams"].get("n_estimators", 760) == 760, (
            "should not push n_estimators further once diminishing -- it should switch to "
            "a different structural hyperparameter (e.g. min_samples_split/max_features) "
            "instead, keeping n_estimators at its best-known value"
        )
        rationale = choice["rationale"]
        assert "Diminishing returns detected on 'n_estimators'" in rationale
        assert "short of the 0.9 target" in rationale


def test_exploitation_uses_directed_expansion_not_static_grid():
    """After a best config is established, the next chosen experiment's
    hyperparams should not duplicate any previously-tried combo, and
    should come from expand_hyperparams (a numeric value strictly
    outside the static grid's range) rather than recycling a grid cell.
    """
    history = [
        {"iteration": 1, "model": "gradient_boosting", "score": 0.70, "hyperparams": {}, "feature_engineering": "basic"},
        {"iteration": 2, "model": "random_forest", "score": 0.80, "hyperparams": {}, "feature_engineering": "basic"},
        {"iteration": 3, "model": "random_forest", "score": 0.8667, "hyperparams": {"max_depth": 10, "n_estimators": 200}, "feature_engineering": "basic"},
    ]
    memory = ExperimentMemory(history)
    plan = _base_plan()

    choice = choose_next_experiment(plan, memory, iteration=4)
    assert choice is not None
    assert choice["model_name"] == "random_forest"

    grid_max_n_estimators = max(hyperparameter_grid("random_forest")["n_estimators"])
    assert choice["hyperparams"].get("n_estimators", 0) > grid_max_n_estimators or (
        choice["hyperparams"].get("max_depth") not in (None, 10)
    )

    static_combos = iter_hyperparam_combos("random_forest")
    assert choice["hyperparams"] not in static_combos

    tried = {(r["model"], tuple(sorted((r.get("hyperparams") or {}).items())), r["feature_engineering"]) for r in history}
    assert (
        choice["model_name"],
        tuple(sorted(choice["hyperparams"].items())),
        choice["feature_engineering"],
    ) not in tried


def test_stagnation_escalates_to_feature_engineering_then_model_switch(monkeypatch):
    """When directed expansion also stops improving, the strategist
    should escalate: first alternate feature_engineering, then (if that
    is exhausted too) switch models -- never loop the same model/FE
    combo forever.
    """
    import tools.ml as ml_module

    # Shrink random_forest's grid to a single key so the static-grid
    # fallback space is small and fully coverable by the seeded history
    # below (isolates the escalation behavior from grid size noise).
    def tiny_grid(model_name):
        return {"n_estimators": [50, 100, 200]} if model_name == "random_forest" else {}

    monkeypatch.setattr(ml_module, "hyperparameter_grid", tiny_grid)

    plan = _base_plan()

    # Build a plateaued random_forest history where hyperparams have
    # already walked through both the static grid AND a directed
    # expansion neighborhood without meaningful improvement.
    history = [
        {"iteration": 1, "model": "gradient_boosting", "score": 0.75, "hyperparams": {}, "feature_engineering": "basic"},
        {"iteration": 2, "model": "random_forest", "score": 0.80, "hyperparams": {}, "feature_engineering": "basic"},
        {"iteration": 3, "model": "random_forest", "score": 0.827, "hyperparams": {"n_estimators": 50}, "feature_engineering": "basic"},
        {"iteration": 4, "model": "random_forest", "score": 0.853, "hyperparams": {"n_estimators": 100}, "feature_engineering": "basic"},
        {"iteration": 5, "model": "random_forest", "score": 0.8400, "hyperparams": {"n_estimators": 200}, "feature_engineering": "basic"},
        {"iteration": 6, "model": "random_forest", "score": 0.8401, "hyperparams": {"n_estimators": 300}, "feature_engineering": "basic"},
        {"iteration": 7, "model": "random_forest", "score": 0.8400, "hyperparams": {"n_estimators": 450}, "feature_engineering": "basic"},
    ]
    memory = ExperimentMemory(history)
    assert memory.is_stagnant(window=3, min_improvement=0.01, direction="maximize") is True
    assert memory.is_stagnating() is True

    choice = choose_next_experiment(plan, memory, iteration=8)
    assert choice is not None
    # rung 1: alternate feature_engineering on random_forest
    assert choice["model_name"] == "random_forest"
    assert choice["feature_engineering"] == "pipeline"

    # Now simulate rung 1 (pipeline FE) also exhausted/plateaued -- add
    # a pipeline record with the same plateaued score and force the
    # strategist to escalate to rung 2 (model switch) by monkeypatching
    # nothing: instead, seed enough tried identities that expand+grid
    # for random_forest/pipeline is exhausted too.
    # Cover the whole plausible expansion neighborhood (both up- and
    # down-scaled values from every static/expanded n_estimators we
    # might have seeded from) so that regardless of which record the
    # strategist treats as "best", expand_hyperparams() has nothing
    # untried left to offer for feature_engineering='pipeline'.
    saturating_values = [12, 25, 38, 50, 75, 100, 150, 200, 300, 450]
    history2 = list(history) + [
        {
            "iteration": 8 + i,
            "model": "random_forest",
            "score": 0.842,
            "hyperparams": {"n_estimators": v},
            "feature_engineering": "pipeline",
        }
        for i, v in enumerate(saturating_values)
    ]
    memory2 = ExperimentMemory(history2)
    choice2 = choose_next_experiment(plan, memory2, iteration=15)
    assert choice2 is not None
    assert choice2["model_name"] == "gradient_boosting"


def test_never_repeats_a_tried_identity_even_when_expansion_collides():
    """If a computed expansion candidate happens to collide with
    something already tried, the strategist must fall through to the
    next candidate/rung rather than returning a duplicate.
    """
    plan = _base_plan()
    history = [
        {"iteration": 1, "model": "gradient_boosting", "score": 0.70, "hyperparams": {}, "feature_engineering": "basic"},
        {"iteration": 2, "model": "random_forest", "score": 0.80, "hyperparams": {}, "feature_engineering": "basic"},
        # Pre-seed the exact expansion candidates that would be
        # generated from this best_hyperparams so the strategist is
        # forced past them.
        {"iteration": 3, "model": "random_forest", "score": 0.83, "hyperparams": {"n_estimators": 150}, "feature_engineering": "basic"},
        {"iteration": 4, "model": "random_forest", "score": 0.81, "hyperparams": {"n_estimators": 50}, "feature_engineering": "basic"},
    ]
    memory = ExperimentMemory(history)
    choice = choose_next_experiment(plan, memory, iteration=5)
    assert choice is not None
    tried = memory.tried_identities()
    identity = (choice["model_name"], choice["feature_engineering"])
    import json as _json

    key = (choice["model_name"], _json.dumps(choice["hyperparams"] or {}, sort_keys=True), choice["feature_engineering"])
    assert key not in tried
