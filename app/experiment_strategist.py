"""
app/experiment_strategist.py

Adaptive "what to try next" decision -- the piece that turns the Ralph
Loop from a blind grid-search walk into an agentic improvement loop
("Ralph Decision Engine").

STAND-IN FOR AN LLM CALL (rule-based for now, same TODO pattern as the
rest of app/).

# TODO: replace the rule-based logic below with an LLM call using this
# prompt:
#   "Given this experiment memory: {memory.summary_for_planner()}, and
#    this plan: {plan}, decide the single best next experiment to run.
#    Respond with JSON with keys 'model_name', 'feature_engineering',
#    'hyperparams', 'rationale'."

Strategy (see app/experiment_memory.py and tools/ml.py for the
primitives used):

1. Exploration -- try every candidate model once (default hyperparams,
   'basic' feature engineering) to get a cheap baseline per model.

2. Exploitation -- once every model has a baseline, DIRECTED expansion
   takes priority: compute a simple trend for the current best model
   (tools.ml.simple_trend) and call tools.ml.expand_hyperparams() to
   propose new hyperparameter values genuinely outside the static grid
   (e.g. n_estimators=300 when the grid only offers up to 200), trying
   the same feature_engineering as the best experiment so only one
   thing changes at a time. Only once expand_hyperparams() has nothing
   untried left to offer (capped at a few attempts) does the strategist
   fall back to the remaining untried static
   tools.ml.iter_hyperparam_combos(best_model) entries.

3. Stagnation handling (memory.is_stagnant(...)) -- an escalating
   ladder, logged in `rationale`:
     a. First: try the alternate feature_engineering on the current
        best model's best hyperparams.
     b. Then: move to the next-best baseline model and resume directed
        expansion (step 2) for it.
     c. If every candidate model has been tried, expanded, and both
        feature_engineering modes attempted with nothing untried left
        -- return None (search_space_exhausted).

4. Rule 2 (never repeat) is enforced centrally: every returned
   candidate is checked against `memory.tried_identities()` before
   being returned (see `_first_untried` below); a computed candidate
   that happens to collide with something already tried is skipped in
   favor of the next one, never returned as-is.
"""

from __future__ import annotations

import json

from tools.ml import DIMINISHING_RETURN_THRESHOLD, expand_hyperparams, iter_hyperparam_combos, simple_trend
from tools.dataset import MAX_ONEHOT_COLUMNS

# Cap on how many expand_hyperparams() candidates we'll try before
# giving up on directed expansion for this round and falling back to
# the static grid / stagnation ladder.
_MAX_EXPANSION_ATTEMPTS = 3


def _identity(model_name, hyperparams, feature_engineering) -> tuple:
    return (model_name, json.dumps(hyperparams or {}, sort_keys=True), feature_engineering)


def _previous_result_gap_context(plan: dict, memory, direction: str) -> str:
    """One sentence naming the previous best result's gap to target, for
    exploration-phase rationale -- so "trying a different model" reads as
    a decision informed by the last result, not a fixed script that
    would say the same thing whether the last score was 0.01 or 0.74.
    Empty string if there's no scored history yet (nothing to compare).
    """
    best_record = memory.best(direction)
    if best_record is None or best_record.get("score") is None:
        return ""
    metric = plan.get("metric")
    target = plan.get("target_score")
    score = best_record["score"]
    gap = (target - score) if direction != "minimize" else (score - target)
    gap_str = f"{gap:.4f}" if isinstance(gap, (int, float)) else str(gap)
    return (
        f"Previous best ('{best_record.get('model')}') scored {metric}={score:.4f}, "
        f"{gap_str} short of the {target} target. "
    )


def _last_marginal_gain(model_hist: list[dict], key: str) -> float | None:
    """Magnitude of the most recent step's score change for `key`, for
    naming the actual number in a diminishing-returns rationale (e.g.
    "last gain 0.000064") instead of just a flat label. Mirrors the pair
    selection tools.ml.simple_trend() uses internally.
    """
    from tools.ml import _is_expandable_numeric  # same numeric-eligibility check as simple_trend

    scored = [r for r in model_hist if r.get("score") is not None]
    pairs = []
    for r in scored[-3:]:
        hp = r.get("hyperparams") or {}
        val = hp.get(key)
        if _is_expandable_numeric(key, val):
            pairs.append((val, r["score"]))
    if len(pairs) < 2:
        return None
    return abs(pairs[-1][1] - pairs[-2][1])


def _exploitation_rationale(plan: dict, memory, best_model: str, best_score, direction: str) -> str:
    """Rationale for continuing to refine the current best model --
    target-aware (names the gap to target_score) and diminishing-returns-
    aware (names the specific hyperparameter that's flattened out and
    explains why the strategy is switching away from it), instead of the
    generic "refining its hyperparameters directed by recent trend".
    """
    metric = plan.get("metric")
    target = plan.get("target_score")
    gap_str = ""
    if target is not None and isinstance(best_score, (int, float)):
        gap = (target - best_score) if direction != "minimize" else (best_score - target)
        gap_str = f", {gap:.4f} short of the {target} target"

    model_hist = memory.history_for_model(best_model)
    trend = simple_trend(model_hist, best_model, direction=direction)
    diminishing = [k for k, v in trend.items() if v == "diminishing"]

    if diminishing:
        gains = [
            (k, g) for k in diminishing for g in [_last_marginal_gain(model_hist, k)] if g is not None
        ]
        gains_str = ", ".join(f"'{k}' last gain {g:.6f}" for k, g in gains)
        keys_str = ", ".join(f"'{k}'" for k in diminishing)
        return (
            f"'{best_model}' is currently the best performer ({metric}={best_score}){gap_str}. "
            f"Diminishing returns detected on {keys_str} ({gains_str}, below the "
            f"{DIMINISHING_RETURN_THRESHOLD} improvement threshold) -- further increasing "
            f"{keys_str} is unlikely to close the gap. Switching optimization dimension "
            "(a different hyperparameter or feature engineering) instead of continuing to "
            "push the same one."
        )

    return (
        f"'{best_model}' is currently the best performer ({metric}={best_score}){gap_str}; "
        "refining its hyperparameters directed by recent trend next instead of cycling to "
        "another model."
    )


def choose_next_experiment(plan: dict, memory, iteration: int) -> dict | None:
    candidate_models = plan.get("candidate_models") or []
    if not candidate_models:
        return None

    direction = plan.get("direction", "maximize")
    tried = memory.tried_identities()

    def is_new(model_name, hyperparams, feature_engineering) -> bool:
        return _identity(model_name, hyperparams, feature_engineering) not in tried

    # --- Phase 1: exploration -- baseline every candidate model once ---
    recommended_fe = plan.get("recommended_feature_engineering", "basic")
    for model_name in candidate_models:
        model_hist = memory.history_for_model(model_name)
        if not model_hist:
            if is_new(model_name, {}, recommended_fe):
                # A real scored result already exists -- explain the choice in
                # terms of that result (history-aware), not by re-explaining
                # WHY safe_categorical was picked in the first place. The
                # profiling rationale below is a one-time, first-trial-only
                # explanation; every subsequent exploration trial should read
                # like a decision informed by what's happened so far.
                gap_context = _previous_result_gap_context(plan, memory, direction)
                if gap_context:
                    fe_note = (
                        f"retaining '{recommended_fe}' feature engineering (already established "
                        "as working)" if recommended_fe != "basic" else "preprocessing is working"
                    )
                    rationale = (
                        f"{gap_context}The {fe_note}, so the shortfall is coming from the model "
                        f"itself, not feature engineering. Trying '{model_name}' next -- a "
                        "different model family that may capture relationships the previous "
                        "model couldn't -- before spending budget refining any single model's "
                        "hyperparameters."
                    )
                elif recommended_fe == "safe_categorical":
                    rationale = (
                        f"Dataset profiling estimated ~{plan.get('estimated_onehot_columns')} "
                        "one-hot columns if 'basic' feature engineering were used -- exceeding "
                        f"the {MAX_ONEHOT_COLUMNS}-column safety limit. Using 'safe_categorical' "
                        "feature engineering (one-hot for low-cardinality columns, "
                        "frequency-encoding for medium-cardinality columns, dropping "
                        f"{len(plan.get('dropped_high_cardinality_columns') or [])} "
                        "high-cardinality/ID-like columns) instead of 'basic' for "
                        f"'{model_name}'s baseline, before comparing models."
                    )
                else:
                    rationale = (
                        f"Exploration phase: '{model_name}' has no baseline yet. "
                        "Trying it once with default hyperparams and basic feature "
                        "engineering before spending budget refining any one model."
                    )
                return {
                    "model_name": model_name,
                    "feature_engineering": recommended_fe,
                    "hyperparams": {},
                    "rationale": rationale,
                }
        else:
            # Only one attempt exists (the just-failed baseline) and it
            # crashed with a preprocessing-related error on 'basic' --
            # retry the SAME model with a safer feature engineering mode
            # instead of moving to a different model that would hit the
            # same broken preprocessing.
            last = model_hist[-1]
            error_type = last.get("error_type")
            if (
                error_type in ("MemoryError", "HIGH_CARDINALITY_ENCODING")
                and last.get("feature_engineering") == "basic"
            ):
                retry_fe = "safe_categorical" if error_type == "HIGH_CARDINALITY_ENCODING" else "pipeline"
                if is_new(model_name, {}, retry_fe):
                    if error_type == "HIGH_CARDINALITY_ENCODING":
                        rationale = (
                            "Previous attempt with 'basic' feature engineering hit the "
                            f"one-hot safety cap ({MAX_ONEHOT_COLUMNS} columns) from "
                            "high/medium-cardinality categorical columns; retrying "
                            f"'{model_name}' with 'safe_categorical' feature engineering "
                            "(frequency-encoding medium-cardinality columns) instead of "
                            "moving to a different model with the same broken preprocessing."
                        )
                    else:
                        rationale = (
                            f"Previous attempt with 'basic' feature engineering failed with "
                            "MemoryError from a high-cardinality one-hot explosion; retrying "
                            f"'{model_name}' with 'pipeline' feature engineering (sparse/capped "
                            "encoding) instead of moving to a different model with the same "
                            "broken preprocessing."
                        )
                    return {
                        "model_name": model_name,
                        "feature_engineering": retry_fe,
                        "hyperparams": {},
                        "rationale": rationale,
                    }

    # All models have a baseline -- rank them by their own best score so
    # far (exclude models with no scored runs).
    per_model_best = {}
    for model_name in candidate_models:
        hist = [r for r in memory.history_for_model(model_name) if r.get("score") is not None]
        if hist:
            best = max(hist, key=lambda r: r["score"]) if direction == "maximize" else min(hist, key=lambda r: r["score"])
            per_model_best[model_name] = best["score"]

    ranked_models = sorted(
        per_model_best.keys(),
        key=lambda m: per_model_best[m],
        reverse=(direction == "maximize"),
    )
    best_model = ranked_models[0] if ranked_models else candidate_models[0]

    def untried_expansion(model_name, fe) -> dict | None:
        """First expand_hyperparams() candidate (seeded from that
        model's best-so-far hyperparams) not already tried, trying up
        to _MAX_EXPANSION_ATTEMPTS candidates before giving up.
        """
        best_record = memory.best(direction)
        model_hist = memory.history_for_model(model_name)
        model_best_hp = {}
        for r in sorted(
            [r for r in model_hist if r.get("score") is not None],
            key=lambda r: r["score"],
            reverse=(direction == "maximize"),
        )[:1]:
            model_best_hp = r.get("hyperparams") or {}

        trend = simple_trend(model_hist, model_name, direction=direction)
        candidates = expand_hyperparams(model_name, model_best_hp, trend=trend)
        for hp in candidates[:_MAX_EXPANSION_ATTEMPTS]:
            if is_new(model_name, hp, fe):
                return hp
        return None

    def untried_hp_combos(model_name, fe):
        combos = iter_hyperparam_combos(model_name)
        return [hp for hp in combos if is_new(model_name, hp, fe)]

    def directed_choice_for(model_name, fe, rationale) -> dict | None:
        """Directed expansion first, static grid fallback second, for
        (model_name, fe). Returns a full choice dict or None if nothing
        untried is available for this (model, fe) pair.
        """
        hp = untried_expansion(model_name, fe)
        if hp is not None:
            return {
                "model_name": model_name,
                "feature_engineering": fe,
                "hyperparams": hp,
                "rationale": rationale + " (directed expansion beyond the static grid)",
            }
        left = untried_hp_combos(model_name, fe)
        if left:
            return {
                "model_name": model_name,
                "feature_engineering": fe,
                "hyperparams": left[0],
                "rationale": rationale + " (static grid, directed expansion exhausted)",
            }
        return None

    # User-specified stricter/simpler stagnation detector (patience=3,
    # min_improvement=1e-4 vs is_stagnant's window=3/0.01) -- direction-
    # agnostic by design: a flat cluster of raw scores is stagnation
    # regardless of maximize/minimize.
    stagnant = memory.is_stagnating()

    if stagnant:
        # --- Phase 3a: stagnation rung 1 -- alternate feature_engineering ---
        current_fe = None
        best_record = memory.best(direction)
        if best_record and best_record.get("model") == best_model:
            current_fe = best_record.get("feature_engineering")
        alt_fe = "pipeline" if current_fe != "pipeline" else "basic"

        choice = directed_choice_for(
            best_model,
            alt_fe,
            (
                f"No meaningful {plan.get('metric')} improvement in the recent "
                f"iterations on '{best_model}'; stagnation rung 1 -- switching to "
                f"'{alt_fe}' feature engineering to see if that unlocks further "
                "improvement instead of continuing to tweak hyperparameters the "
                "same way."
            ),
        )
        if choice:
            return choice

        # --- Phase 3b: stagnation rung 2 -- move to the next-best model ---
        for model_name in ranked_models[1:]:
            for fe in ("basic", "pipeline"):
                choice = directed_choice_for(
                    model_name,
                    fe,
                    (
                        f"'{best_model}'s hyperparameter space (directed expansion + "
                        f"static grid, both feature-engineering modes) is exhausted "
                        f"with no recent improvement; stagnation rung 2 -- trying "
                        f"'{model_name}' next, which had the next-best baseline score "
                        f"({per_model_best.get(model_name)})."
                    ),
                )
                if choice:
                    return choice

        # --- Phase 3c: everything exhausted ---
        return None

    # --- Phase 2: exploitation -- directed expansion of the current best ---
    best_record = memory.best(direction)
    best_fe = best_record.get("feature_engineering") if best_record and best_record.get("model") == best_model else "basic"

    exploitation_rationale = _exploitation_rationale(plan, memory, best_model, best_score=per_model_best.get(best_model), direction=direction)

    choice = directed_choice_for(best_model, best_fe, exploitation_rationale)
    if choice:
        return choice

    # try the other feature_engineering mode on the best model
    other_fe = "pipeline" if best_fe != "pipeline" else "basic"
    choice = directed_choice_for(
        best_model,
        other_fe,
        (
            f"'{best_model}'s hyperparameter space for feature_engineering="
            f"'{best_fe}' is exhausted; trying the '{other_fe}' feature "
            "engineering variant on the same (best-so-far) model."
        ),
    )
    if choice:
        return choice

    # best model fully exhausted -- move down the ranking.
    for model_name in ranked_models[1:]:
        for fe in ("basic", "pipeline"):
            choice = directed_choice_for(
                model_name,
                fe,
                (
                    f"'{best_model}' is fully explored (directed expansion + static "
                    f"grid, both feature-engineering modes); trying '{model_name}' "
                    f"next (next-best baseline score {per_model_best.get(model_name)})."
                ),
            )
            if choice:
                return choice

    return None
