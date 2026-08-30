"""
tools/ml.py

Phase 10 - ML experimentation expansion. Phase 13 - directed
hyperparameter expansion beyond the static grid (Ralph Decision Engine).

Reusable, testable helpers used by app/coder.py (and generated scripts)
to pick models, build preprocessing pipelines, and search a small
hyperparameter grid. Single source of truth for "which models exist for
which task type" -- app/planner.py imports list_models_for_task instead
of hardcoding its own lists.

Caller: app/experiment_strategist.py (choose_next_experiment) calls
expand_hyperparams() during the exploitation phase to propose directed,
novel candidates beyond hyperparameter_grid()/iter_hyperparam_combos().
No duplicate-purpose file exists for this -- hyperparameter_grid() and
iter_hyperparam_combos() remain the static/exploration-phase source;
expand_hyperparams() is new and additive. All data used anywhere in this
project is synthetic sample data (workspace/datasets/*.csv) -- no real
or sensitive data is read or produced by this module.
"""

from __future__ import annotations

import itertools

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

CLASSIFICATION_MODELS = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "knn",
    "svm",
]

REGRESSION_MODELS = [
    "linear_regression",
    "random_forest_regressor",
    "gradient_boosting_regressor",
    "ridge",
    "knn_regressor",
]


def list_models_for_task(task_type: str) -> list[str]:
    """Return the candidate model name list for a task type."""
    if task_type == "classification":
        return list(CLASSIFICATION_MODELS)
    if task_type == "regression":
        return list(REGRESSION_MODELS)
    return []


def get_model(name: str, task_type: str, hyperparams: dict | None = None):
    """Factory returning an sklearn estimator instance for `name`.

    `hyperparams` (may be None or partial) are applied as constructor
    kwargs on top of sane defaults.
    """
    hyperparams = dict(hyperparams or {})

    if name == "logistic_regression":
        params = {"max_iter": 1000, "random_state": 42}
        params.update(hyperparams)
        return LogisticRegression(**params)
    if name == "random_forest":
        params = {"n_estimators": 100, "max_depth": None, "random_state": 42}
        params.update(hyperparams)
        return RandomForestClassifier(**params)
    if name == "gradient_boosting":
        params = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
        params.update(hyperparams)
        return GradientBoostingClassifier(**params)
    if name == "knn":
        params = {"n_neighbors": 5}
        params.update(hyperparams)
        return KNeighborsClassifier(**params)
    if name == "svm":
        params = {"C": 1.0, "kernel": "rbf", "probability": True, "random_state": 42}
        params.update(hyperparams)
        return SVC(**params)

    if name == "linear_regression":
        params = {}
        params.update(hyperparams)
        return LinearRegression(**params)
    if name == "random_forest_regressor":
        params = {"n_estimators": 100, "max_depth": None, "random_state": 42}
        params.update(hyperparams)
        return RandomForestRegressor(**params)
    if name == "gradient_boosting_regressor":
        params = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
        params.update(hyperparams)
        return GradientBoostingRegressor(**params)
    if name == "ridge":
        params = {"alpha": 1.0, "random_state": 42}
        params.update(hyperparams)
        return Ridge(**params)
    if name == "knn_regressor":
        params = {"n_neighbors": 5}
        params.update(hyperparams)
        return KNeighborsRegressor(**params)

    raise ValueError(f"Unknown model name: {name!r} (task_type={task_type!r})")


def build_preprocessing_pipeline(
    numeric_cols,
    categorical_cols,
    scale_numeric: bool = True,
    feature_selection_k: int | None = None,
):
    """Return a ColumnTransformer usable as the first step of a full
    sklearn Pipeline whose final step is a model.
    """
    numeric_steps = [("impute", _median_imputer())]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            ("impute", _most_frequent_imputer()),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, list(numeric_cols)))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, list(categorical_cols)))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if feature_selection_k:
        return Pipeline(
            [
                ("preprocess", preprocessor),
                ("select", SelectKBest(k=feature_selection_k)),
            ]
        )
    return Pipeline([("preprocess", preprocessor)])


def _median_imputer():
    from sklearn.impute import SimpleImputer

    return SimpleImputer(strategy="median")


def _most_frequent_imputer():
    from sklearn.impute import SimpleImputer

    return SimpleImputer(strategy="most_frequent")


def hyperparameter_grid(model_name: str) -> dict:
    """Small, cheap-to-search hyperparameter options per model (2-3
    values each) suitable for a quick manual/randomized search.
    """
    # random_forest[_regressor] carry structural params beyond n_estimators
    # (max_depth, min_samples_split, min_samples_leaf, max_features) so the
    # strategist's directed expansion (expand_hyperparams) and diminishing-
    # returns detection have somewhere to go once n_estimators flattens out
    # -- otherwise "switch optimization dimension" has nothing to switch to.
    grids = {
        "logistic_regression": {"C": [0.1, 1.0, 10.0]},
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": [0.5, 1.0],
        },
        "gradient_boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2]},
        "knn": {"n_neighbors": [3, 5, 9]},
        "svm": {"C": [0.5, 1.0, 5.0]},
        "linear_regression": {},
        "random_forest_regressor": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": [0.5, 1.0],
        },
        "gradient_boosting_regressor": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2]},
        "ridge": {"alpha": [0.1, 1.0, 10.0]},
        "knn_regressor": {"n_neighbors": [3, 5, 9]},
    }
    return grids.get(model_name, {})


def iter_hyperparam_combos(model_name: str) -> list[dict]:
    """Return the full cartesian product of hyperparameter_grid(model_name)
    as a list of dicts. If the grid is empty, returns [{}] so there's
    always at least one (default) combo to try.
    """
    grid = hyperparameter_grid(model_name)
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    value_lists = [grid[key] for key in keys]
    combos = []
    for values in itertools.product(*value_lists):
        combos.append(dict(zip(keys, values)))
    return combos


def pick_hyperparams(model_name: str, iteration: int) -> dict:
    """Deterministically pick one hyperparameter combo from
    hyperparameter_grid(model_name) based on iteration index, so
    successive iterations vary hyperparameters instead of repeating the
    same combo.
    """
    grid = hyperparameter_grid(model_name)
    if not grid:
        return {}
    keys = sorted(grid.keys())
    combo = {}
    for i, key in enumerate(keys):
        values = grid[key]
        combo[key] = values[(iteration + i) % len(values)]
    return combo


# --- Directed hyperparameter expansion (Ralph Decision Engine) ---------

# Model-specific default values used when a numeric grid key is missing
# or None in `best_hyperparams`, and bounds used to clamp expanded
# values so the directed search can't run away to absurd values.
_MODEL_DEFAULTS = {
    "logistic_regression": {"C": 1.0},
    "random_forest": {
        "n_estimators": 100, "max_depth": None,
        "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 1.0,
    },
    "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1},
    "knn": {"n_neighbors": 5},
    "svm": {"C": 1.0},
    "linear_regression": {},
    "random_forest_regressor": {
        "n_estimators": 100, "max_depth": None,
        "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 1.0,
    },
    "gradient_boosting_regressor": {"n_estimators": 100, "learning_rate": 0.1},
    "ridge": {"alpha": 1.0},
    "knn_regressor": {"n_neighbors": 5},
}

# (min, max) clamps per hyperparam key, applied regardless of model.
_BOUNDS = {
    "n_estimators": (1, 1000),
    "max_depth": (1, 50),
    "n_neighbors": (1, 200),
    "C": (1e-4, 100.0),
    "learning_rate": (1e-4, 100.0),
    "alpha": (1e-4, 100.0),
    "min_samples_split": (2, 100),
    "min_samples_leaf": (1, 100),
    "max_features": (0.1, 1.0),
}

# Which keys are integer-valued vs float-valued, for sensible rounding.
_INT_KEYS = {"n_estimators", "max_depth", "n_neighbors", "min_samples_split", "min_samples_leaf"}

# A step whose marginal gain (vs the immediately preceding same-key value)
# is smaller than this is "diminishing returns" -- not worth another push
# in the same direction on the same key. Same order of magnitude as
# app.ralph.MIN_IMPROVEMENT ("not a meaningful improvement"), kept as a
# separate constant here to avoid a circular import.
DIMINISHING_RETURN_THRESHOLD = 1e-4


def _is_expandable_numeric(key: str, value) -> bool:
    """True if `value` is a numeric type we know how to expand (skips
    strings like svm's 'kernel', bools, and None -- None is handled
    separately as the "nothing further to try" special case).
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and key in _BOUNDS


def _clamp(key: str, value):
    lo, hi = _BOUNDS.get(key, (None, None))
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    if key in _INT_KEYS:
        value = int(round(value))
    return value


def expand_hyperparams(
    model_name: str,
    best_hyperparams: dict | None,
    trend: dict | None = None,
) -> list[dict]:
    """Return a small list (2-4) of NEW candidate hyperparameter dicts
    for `model_name`, seeded from `best_hyperparams`, that push beyond
    the static hyperparameter_grid() values -- i.e. genuinely novel
    values, not a re-pick from the static grid.

    `trend` (optional), e.g. {"n_estimators": "increasing_helps"},
    biases candidates toward extending that key further in the
    improving direction. See _simple_trend() below for how it's
    derived -- a lightweight heuristic, not real Bayesian optimization.

    # TODO: replace this heuristic (scale-by-1.5x + trend bias) with a
    # real directed search (e.g. Bayesian optimization / a proper
    # grid-refinement search) once this needs to scale beyond a rule
    # based Ralph Decision Engine.
    """
    grid = hyperparameter_grid(model_name)
    if not grid:
        return []

    best_hyperparams = dict(best_hyperparams or {})
    trend = trend or {}
    defaults = _MODEL_DEFAULTS.get(model_name, {})

    numeric_keys = [k for k in grid.keys() if k in _BOUNDS]
    if not numeric_keys:
        return []

    candidates: list[dict] = []

    for key in numeric_keys:
        current = best_hyperparams.get(key, defaults.get(key))
        if not _is_expandable_numeric(key, current):
            # None (e.g. max_depth=None/unbounded) or a non-numeric
            # value -- nothing further to expand in that direction for
            # this key; skip it and look at other keys instead.
            continue

        direction_bias = trend.get(key)
        if direction_bias == "diminishing":
            # The last push on this key barely moved the score -- stop
            # expanding it further and let directed_choice_for() fall
            # through to the static grid / a different hyperparameter
            # instead of blindly continuing to scale this one up.
            continue

        if key in _INT_KEYS:
            up = _clamp(key, current * 1.5 if current > 0 else current + 1)
            down = _clamp(key, current * 0.5 if current > 1 else current)
        else:
            up = _clamp(key, current * 1.5 if current > 0 else current + 0.1)
            down = _clamp(key, current * 0.5 if current > 0 else current)

        new_values = []
        if direction_bias == "increasing_helps":
            new_values = [up, _clamp(key, up * 1.5 if key in _INT_KEYS else up * 1.5)]
        elif direction_bias == "decreasing_helps":
            new_values = [down, _clamp(key, down * 0.5 if down > 0 else down)]
        else:
            new_values = [up, down]

        for val in new_values:
            if val == current:
                continue
            candidate = dict(best_hyperparams)
            candidate[key] = val
            if candidate not in candidates:
                candidates.append(candidate)

    # Never return something that's actually just a static grid cell in
    # disguise (e.g. scaling down landed back on an existing grid
    # value) -- the whole point is genuinely novel candidates.
    static_combos = iter_hyperparam_combos(model_name)
    candidates = [c for c in candidates if c not in static_combos]

    return candidates[:4]


def simple_trend(history: list[dict], model_name: str, direction: str = "maximize") -> dict:
    """Lightweight heuristic: look at the 2-3 most recent same-model
    records and, for each numeric hyperparam key, check whether the
    record with the higher value for that key also scored better. If
    so, mark that key "increasing_helps"; if the record with the lower
    value scored better, mark it "decreasing_helps". Ambiguous/no
    signal -> key omitted from the returned dict.

    # TODO: replace with real trend/importance estimation (e.g. a small
    # regression over hyperparam deltas vs score deltas) -- this is a
    # deliberately simple heuristic, not statistics.
    """
    better = (lambda a, b: a > b) if direction == "maximize" else (lambda a, b: a < b)

    same_model = [
        r for r in history
        if r.get("model") == model_name and r.get("score") is not None
    ]
    recent = same_model[-3:]
    if len(recent) < 2:
        return {}

    grid = hyperparameter_grid(model_name)
    numeric_keys = [k for k in grid.keys() if k in _BOUNDS]

    trend: dict = {}
    for key in numeric_keys:
        pairs = []
        for r in recent:
            hp = r.get("hyperparams") or {}
            val = hp.get(key)
            if _is_expandable_numeric(key, val):
                pairs.append((val, r["score"]))
        if len(pairs) < 2:
            continue
        # Use the two most recent (chronologically last) values for this
        # key, not the window's low/high extremes -- a window can span an
        # earlier big jump and a since-flattened last step (e.g.
        # n_estimators 338->507->760, where 338 vs 760 still looks like
        # +0.00028 even though the actual last step, 507->760, only
        # gained +0.00006). The marginal step is what tells us whether
        # pushing this key further is still worth it.
        prev_val, prev_score = pairs[-2]
        last_val, last_score = pairs[-1]
        if last_val == prev_val:
            continue
        if better(last_score, prev_score):
            gain = abs(last_score - prev_score)
            trend[key] = "diminishing" if gain < DIMINISHING_RETURN_THRESHOLD else "increasing_helps"
        elif better(prev_score, last_score):
            gain = abs(last_score - prev_score)
            trend[key] = "diminishing" if gain < DIMINISHING_RETURN_THRESHOLD else "decreasing_helps"

    return trend
