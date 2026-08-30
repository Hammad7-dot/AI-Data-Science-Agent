import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.ml import expand_hyperparams, hyperparameter_grid, iter_hyperparam_combos, simple_trend


def test_expand_hyperparams_goes_beyond_static_grid_for_random_forest():
    best_hyperparams = {"max_depth": 10, "n_estimators": 200}
    candidates = expand_hyperparams("random_forest", best_hyperparams)

    assert candidates, "expected at least one directed candidate"
    assert any(c.get("n_estimators", 0) > 200 for c in candidates), candidates

    static_combos = iter_hyperparam_combos("random_forest")
    for c in candidates:
        assert c not in static_combos, f"{c} duplicates a static grid cell"


def test_expand_hyperparams_repeated_calls_never_reproduce_grid_cells():
    best_hyperparams = {"max_depth": 10, "n_estimators": 200}
    static_combos = iter_hyperparam_combos("random_forest")

    for _ in range(5):
        candidates = expand_hyperparams("random_forest", best_hyperparams)
        for c in candidates:
            assert c not in static_combos


def test_expand_hyperparams_none_max_depth_is_not_expanded():
    # max_depth=None (unbounded) should not be "expanded"; only the
    # other numeric key (n_estimators) should produce candidates, and
    # max_depth should stay None (or absent) across all of them.
    candidates = expand_hyperparams("random_forest", {"max_depth": None, "n_estimators": 100})
    assert candidates
    for c in candidates:
        assert c.get("max_depth") in (None, 10) or "max_depth" not in c
        # specifically: we never invent a *new* non-None value for max_depth
        # when it started as None and wasn't the trending key.


def test_expand_hyperparams_skips_non_numeric_params():
    # svm's grid only has "C" (numeric); "kernel" isn't in the grid, so
    # nothing should choke on it even if passed in best_hyperparams.
    candidates = expand_hyperparams("svm", {"C": 1.0, "kernel": "rbf"})
    for c in candidates:
        assert "kernel" not in c or c["kernel"] == "rbf"


def test_expand_hyperparams_respects_bounds():
    candidates = expand_hyperparams("random_forest", {"n_estimators": 900, "max_depth": 40})
    for c in candidates:
        if "n_estimators" in c:
            assert c["n_estimators"] <= 1000
        if "max_depth" in c and c["max_depth"] is not None:
            assert c["max_depth"] <= 50


def test_expand_hyperparams_empty_grid_returns_empty():
    assert expand_hyperparams("linear_regression", {}) == []


def test_simple_trend_detects_increasing_helps():
    history = [
        {"model": "random_forest", "score": 0.80, "hyperparams": {"n_estimators": 50}},
        {"model": "random_forest", "score": 0.85, "hyperparams": {"n_estimators": 100}},
        {"model": "random_forest", "score": 0.87, "hyperparams": {"n_estimators": 200}},
    ]
    trend = simple_trend(history, "random_forest", direction="maximize")
    assert trend.get("n_estimators") == "increasing_helps"


def test_simple_trend_not_enough_history():
    history = [{"model": "random_forest", "score": 0.8, "hyperparams": {"n_estimators": 100}}]
    assert simple_trend(history, "random_forest") == {}


def test_simple_trend_detects_diminishing_returns_on_last_step():
    """Regression test: n_estimators 338->507->760 where the earlier jump
    (338->507) was a real gain but the most recent step (507->760) barely
    moved the score -- the marginal (last-step) signal must win over the
    window's overall low/high span, or the strategist keeps scaling a
    hyperparameter that has already flattened out.
    """
    history = [
        {"model": "random_forest_regressor", "score": 0.86781, "hyperparams": {"n_estimators": 338}},
        {"model": "random_forest_regressor", "score": 0.8680224360853683, "hyperparams": {"n_estimators": 507}},
        {"model": "random_forest_regressor", "score": 0.8680868300748117, "hyperparams": {"n_estimators": 760}},
    ]
    trend = simple_trend(history, "random_forest_regressor", direction="maximize")
    assert trend.get("n_estimators") == "diminishing"


def test_expand_hyperparams_skips_diminishing_key():
    """Once n_estimators is flagged diminishing, expand_hyperparams must
    not offer a further push on it -- but should still offer candidates
    on the model's OTHER structural keys (min_samples_split etc.), so
    the strategist has somewhere to switch to instead of giving up.
    """
    candidates = expand_hyperparams(
        "random_forest_regressor",
        {"n_estimators": 760},
        trend={"n_estimators": "diminishing"},
    )
    assert candidates, "should offer candidates on other structural keys, not give up entirely"
    for c in candidates:
        assert c.get("n_estimators", 760) == 760, "n_estimators must not be pushed further once diminishing"
