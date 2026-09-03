"""
app/planner.py

Phase 5 - Planner.

STAND-IN FOR AN LLM CALL.

In a full implementation, this module would send the dataset profile and
the user's natural-language objective to an LLM with a prompt like:

    "Given this dataset profile: {profile}, and this objective: {objective},
     produce a JSON analysis plan with keys task_type, target, metric,
     steps, candidate_models. Respond with JSON only."

and parse the LLM's JSON response into the plan dict below.

For Phase 0 MVP (no API key available), we instead use deterministic,
rule-based logic driven entirely by the dataset profile produced by
tools.dataset.profile_dataset. This keeps the pipeline runnable end to
end today. Swap the body of create_plan() for an LLM call + JSON parse
when an API key is available -- the function signature and return
schema should not need to change.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.ml import list_models_for_task  # noqa: E402
from tools.dataset import MAX_ONEHOT_COLUMNS  # noqa: E402
from app.evaluator import LOWER_IS_BETTER  # noqa: E402

DEFAULT_METRIC = {
    "classification": "f1",
    "regression": "r2",
    "eda": "completeness",
}

# Generic, bounded-metric default targets. rmse/mae have no fixed scale
# so they're handled separately (see _derive_target_score).
DEFAULT_TARGET_SCORE = {
    "f1": 0.80,
    "accuracy": 0.80,
    "r2": 0.75,
    "completeness": 1.0,
}


def _derive_target_score(metric: str, target_info: dict) -> tuple[float, str]:
    """Auto-derive a target_score when the caller didn't specify one.

    Returns (target_score, explanation) so the plan's `steps` and the
    final report can state clearly whether the number came from the
    user or was a heuristic.
    """
    if metric in LOWER_IS_BETTER:
        std = target_info.get("std")
        if std:
            value = round(0.5 * std, 4)
            return value, (
                f"Target score not specified; defaulting to {metric} <= {value} "
                f"(heuristic: 0.5 * target std={std:.4f}) because {metric} has no "
                "fixed scale and a flat constant like 0.80 would be meaningless "
                "for a raw-unit regression error."
            )
        # No std available (e.g. degenerate/constant target) -- fall back
        # to a conservative literal value rather than crashing.
        return 1.0, (
            f"Target score not specified and target std unavailable; defaulting "
            f"to {metric} <= 1.0 as a last-resort heuristic."
        )
    value = DEFAULT_TARGET_SCORE.get(metric, 0.80)
    return value, f"Target score not specified; defaulting to {metric} >= {value} (generic bounded-metric default)."


def create_plan(
    profile: dict,
    objective: str,
    metric: str | None = None,
    target_score: float | None = None,
) -> dict:
    """Build a structured analysis plan from a dataset profile.

    # TODO: replace with an LLM call using this prompt:
    #   "Given this dataset profile: {profile}, and this objective:
    #    '{objective}', produce a JSON analysis plan with keys task_type,
    #    target, metric, steps, candidate_models. Respond with JSON only."

    Returns a fixed-schema dict, never prose:
        {
            "task_type": "classification" | "regression" | "eda",
            "target": str | None,
            "metric": str,
            "steps": [str, ...],
            "candidate_models": [str, ...],
        }

    Raises ValueError if the profile has no usable target and the task
    cannot be inferred as anything other than "eda" -- but "eda" itself
    is always a valid fallback and never raises.
    """
    target_info = profile.get("target") or {}
    task_type = target_info.get("suggested_task_type")
    target_name = target_info.get("name")

    if task_type not in ("classification", "regression") or not target_name:
        # No usable target -> fall back to exploratory data analysis.
        eda_metric = metric or DEFAULT_METRIC["eda"]
        return {
            "task_type": "eda",
            "target": None,
            "metric": eda_metric,
            "direction": "minimize" if eda_metric in LOWER_IS_BETTER else "maximize",
            "steps": [
                "Load dataset and compute shape",
                "Compute per-column summary statistics (describe)",
                "Compute missing-value counts per column",
                "Report data completeness as the evaluation metric",
            ],
            "candidate_models": [],
        }

    if task_type == "classification":
        candidate_models = list_models_for_task("classification")
        steps = [
            f"Load dataset from CSV and identify target column '{target_name}'",
            "Drop target from feature matrix X",
            "Reserve a final 20% holdout (random_state=42; stratified for classification)",
            "Fit encoding, imputation and feature selection inside each cross-validation fold",
            "Select models using out-of-fold predictions from up to three development folds",
            f"Train a {candidate_models[0]} classifier as the first candidate",
            "Evaluate with accuracy, precision, recall, and f1 (weighted average)",
        ]
    else:  # regression
        candidate_models = list_models_for_task("regression")
        steps = [
            f"Load dataset from CSV and identify target column '{target_name}'",
            "Drop target from feature matrix X",
            "Reserve a final 20% holdout (random_state=42; stratified for classification)",
            "Fit encoding, imputation and feature selection inside each cross-validation fold",
            "Select models using out-of-fold predictions from up to three development folds",
            f"Train a {candidate_models[0]} regressor as the first candidate",
            "Evaluate with rmse, mae, and r2",
        ]

    # High-cardinality/id_like feature columns (per tools.dataset's
    # profiler) get dropped before one-hot encoding in generated code --
    # never drop the target itself even if it happens to look id-like.
    dropped_high_cardinality_columns = [
        c for c in (profile.get("high_cardinality_columns") or []) if c != target_name
    ]
    if dropped_high_cardinality_columns:
        steps.append(
            "Drop high-cardinality/ID-like columns before encoding: "
            f"{', '.join(dropped_high_cardinality_columns)} (would explode one-hot "
            "encoding column count)"
        )

    leakage_warnings = profile.get("leakage_warnings") or []
    # Actually act on the warning, not just display it: every column whose
    # recommended_action is "drop" is dropped from the feature matrix in
    # generated code (app/coder.py's drop_block), the same way
    # dropped_high_cardinality_columns already is -- a profiler that says
    # "possible leakage" but a generated script that still trains on the
    # flagged column is a contradiction, not a warning.
    leakage_dropped_columns = [
        w["column"] for w in leakage_warnings
        if w.get("recommended_action") == "drop" and w["column"] != target_name
    ]
    if leakage_dropped_columns:
        cols = ", ".join(leakage_dropped_columns)
        steps.append(
            f"Drop likely-leakage column(s) before training: {cols} -- name-heuristic "
            "flagged these as probably only known after the outcome being predicted "
            "(see plan['leakage_warnings'] for the reason per column)"
        )

    # Column names by cardinality bucket (target and leakage-dropped columns
    # excluded) -- threaded through to app/coder.py the same way
    # dropped_high_cardinality_columns already is, so 'safe_categorical'
    # feature engineering knows which columns to one-hot vs frequency-encode.
    cardinality_info = profile.get("categorical_cardinality") or {}
    low_cardinality_columns = [
        c for c, info in cardinality_info.items()
        if info.get("bucket") == "low" and c != target_name and c not in leakage_dropped_columns
    ]
    medium_cardinality_columns = [
        c for c, info in cardinality_info.items()
        if info.get("bucket") == "medium" and c != target_name and c not in leakage_dropped_columns
    ]

    # Estimate the one-hot column count 'basic' feature engineering would
    # produce (sum of nunique across non-dropped low/medium categorical
    # columns) so the strategist can proactively avoid 'basic' on datasets
    # that would blow the MAX_ONEHOT_COLUMNS safety cap, instead of
    # waiting for every candidate model to crash with the same
    # RuntimeError first.
    estimated_onehot_columns = sum(
        cardinality_info[c]["nunique"] for c in low_cardinality_columns + medium_cardinality_columns
    )
    recommended_feature_engineering = (
        "safe_categorical" if estimated_onehot_columns > MAX_ONEHOT_COLUMNS else "basic"
    )
    if recommended_feature_engineering == "safe_categorical":
        steps.append(
            f"Dataset profiling estimated ~{estimated_onehot_columns} one-hot columns from "
            f"basic encoding, exceeding the {MAX_ONEHOT_COLUMNS}-column safety cap; "
            "recommending 'safe_categorical' feature engineering instead"
        )

    resolved_metric = metric or DEFAULT_METRIC[task_type]
    direction = "minimize" if resolved_metric in LOWER_IS_BETTER else "maximize"

    if target_score is not None:
        resolved_target_score = target_score
        steps.append(f"Target score explicitly set by caller: {resolved_metric} >= {resolved_target_score}"
                     if direction == "maximize"
                     else f"Target score explicitly set by caller: {resolved_metric} <= {resolved_target_score}")
    else:
        resolved_target_score, explanation = _derive_target_score(resolved_metric, target_info)
        steps.append(explanation)

    return {
        "task_type": task_type,
        "target": target_name,
        "metric": resolved_metric,
        "direction": direction,
        "target_score": resolved_target_score,
        "steps": steps,
        "candidate_models": candidate_models,
        "dropped_high_cardinality_columns": dropped_high_cardinality_columns,
        "low_cardinality_columns": low_cardinality_columns,
        "medium_cardinality_columns": medium_cardinality_columns,
        "estimated_onehot_columns": estimated_onehot_columns,
        "recommended_feature_engineering": recommended_feature_engineering,
        "leakage_warnings": leakage_warnings,
        "leakage_dropped_columns": leakage_dropped_columns,
    }
