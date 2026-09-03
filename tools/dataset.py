"""
tools/dataset.py

Phase 3 - Dataset profiler.

Pure pandas/numpy. No AI involved. This gives the rest of the agent
ground truth about a CSV instead of asking an LLM to guess it.
"""

from __future__ import annotations

import pandas as pd

LOW_CARDINALITY_MAX_RATIO = 0.01   # -> "low" -> safe for one-hot
MEDIUM_CARDINALITY_MAX_RATIO = 0.20  # -> "medium" -> one-hot still OK but capped
# above 0.20 -> "high" -> likely an ID column, drop by default
ID_LIKE_UNIQUE_RATIO = 0.95  # near-1:1 with row count -> flag explicitly as "id_like"

# Defensive cap: if one-hot encoding would still produce more columns than
# this (after dropping known high-cardinality/id_like columns), generated
# scripts raise a clear RuntimeError instead of attempting a dense
# conversion that can OOM the process. Lives here (not app/coder.py) so
# app/planner.py can import it too without a circular import.
MAX_ONEHOT_COLUMNS = 2000

# Column-name substrings (lowercased) that commonly indicate a feature is
# only known AFTER the outcome being predicted -- i.e. a target-leakage
# risk. Deliberately a name heuristic, not a statistical leakage detector
# (e.g. train/test correlation analysis) -- cheap, no false sense of
# certainty implied, just a flag for a human to check before trusting the
# score. Never auto-dropped: an outcome-ish name can still be legitimate
# for a differently-scoped objective, so this only warns.
LEAKAGE_SUSPECT_KEYWORDS = [
    "winner", "victory", "outcome", "result", "final_score", "score_diff",
]
# Timestamp-ish columns whose name suggests "when the event/record ended",
# which is often unavailable at prediction time even though it's a normal
# datetime column, not a categorical one.
POST_EVENT_TIME_KEYWORDS = [
    "last_move", "ended_at", "closed_at", "finished_at", "resolved_at", "end_time",
]


def _cardinality_bucket(unique_ratio: float) -> str:
    if unique_ratio <= LOW_CARDINALITY_MAX_RATIO:
        return "low"
    if unique_ratio <= MEDIUM_CARDINALITY_MAX_RATIO:
        return "medium"
    return "high"


def _guess_target(df: pd.DataFrame, hint: str | None = None) -> str | None:
    """Best-effort target column guess.

    If `hint` names a real column, use it. Otherwise fall back to common
    target-ish names, then to the last column as a last resort.
    """
    if hint and hint in df.columns:
        return hint

    common_names = [
        "target", "label", "class", "churn", "outcome", "y",
        "survived", "price", "sales", "default", "fraud",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for name in common_names:
        if name in lower_map:
            return lower_map[name]

    if hint and hint.lower() in lower_map:
        return lower_map[hint.lower()]

    return df.columns[-1] if len(df.columns) > 0 else None


def _detect_leakage_risks(column_names: list[str], target: str | None) -> list[dict]:
    """Name-heuristic scan for columns that look like they'd only be
    known after the outcome/event being predicted -- e.g. `winner_white`,
    `victory_status_resign`, `last_move_at` for a "predict the game
    result" objective. Returns a list of {"column", "reason"} dicts,
    excluding the target column itself. Empty list -> nothing suspicious
    found (not proof of no leakage -- just no name-based red flag).
    """
    warnings = []
    for col in column_names:
        if col == target:
            continue
        lower = col.lower()
        for kw in LEAKAGE_SUSPECT_KEYWORDS:
            if kw in lower:
                warnings.append({
                    "column": col,
                    "reason": (
                        f"name contains '{kw}', which usually describes the outcome of the "
                        "event being predicted -- verify this is actually available at "
                        "prediction time before trusting scores that use it as a feature."
                    ),
                    "risk": "high",
                    # Both keyword lists are narrow/high-confidence on purpose (game
                    # results, post-event timestamps) -- everything the name-heuristic
                    # catches at all is treated as "drop by default", not a softer
                    # "flag but keep" tier. See app.planner.create_plan, which is the
                    # single place that reads recommended_action to decide what
                    # actually gets dropped from the generated code.
                    "recommended_action": "drop",
                })
                break
        else:
            for kw in POST_EVENT_TIME_KEYWORDS:
                if kw in lower:
                    warnings.append({
                        "column": col,
                        "reason": (
                            f"name contains '{kw}', suggesting a timestamp recorded at/after "
                            "the event ended -- likely unavailable at prediction time."
                        ),
                        "risk": "high",
                        "recommended_action": "drop",
                    })
                    break
    return warnings


def profile_dataset(path: str, target_hint: str | None = None) -> dict:
    """Load a CSV and compute a structured profile of it.

    Returns a JSON-serializable dict. Never raises for "normal" messy
    data (missing values, mixed types) - it should always describe what
    it finds.
    """
    return profile_frame(pd.read_csv(path), path, target_hint)


def profile_frame(df: pd.DataFrame, path: str, target_hint: str | None = None,
                  task_type_hint: str | None = None) -> dict:
    """Profile an already selected partition without rereading the full CSV."""

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    target = _guess_target(df, target_hint)

    missing_by_col = {c: int(n) for c, n in df.isnull().sum().items() if n > 0}

    target_info = {}
    if target is not None and target in df.columns:
        nunique = int(df[target].nunique(dropna=True))
        is_numeric = target in numeric_cols
        if is_numeric and nunique > 20:
            task_type = "regression"
        else:
            task_type = "classification"
        if task_type_hint is not None:
            task_type = task_type_hint
        target_info = {
            "name": target,
            "dtype": str(df[target].dtype),
            "unique_values": nunique,
            "suggested_task_type": task_type,
            "class_balance": (
                df[target].value_counts(normalize=True).round(4).to_dict()
                if task_type == "classification"
                else None
            ),
        }
        if task_type == "regression":
            # Scale-awareness: the regression target's own distribution is
            # needed downstream (app/planner.py) to derive a sensible
            # target_score instead of comparing a raw-dollar-scale error
            # against a fixed constant like 0.80.
            non_null = df[target].dropna()
            desc = non_null.describe() if len(non_null) else None
            target_info.update(
                {
                    "mean": float(desc["mean"]) if desc is not None else None,
                    "std": float(desc["std"]) if desc is not None and pd.notna(desc.get("std")) else None,
                    "min": float(desc["min"]) if desc is not None else None,
                    "max": float(desc["max"]) if desc is not None else None,
                }
            )

    n_rows = len(df)
    categorical_cardinality = {}
    high_cardinality_columns = []
    # Non-numeric columns, plus any numeric column that's an ID-like
    # sequence (all-unique integers with a very high unique_ratio) -- a
    # cheap check, not a general numeric-cardinality analysis.
    cols_to_check = list(categorical_cols)
    for c in numeric_cols:
        # Only integer-typed columns -- a continuous float feature (e.g.
        # "monthly_charges") is naturally near-all-unique without being
        # an ID column, so restrict this cheap check to integer dtypes.
        if not pd.api.types.is_integer_dtype(df[c]):
            continue
        if n_rows and df[c].nunique(dropna=True) / n_rows >= ID_LIKE_UNIQUE_RATIO:
            cols_to_check.append(c)

    for c in cols_to_check:
        nunique = int(df[c].nunique(dropna=True))
        unique_ratio = (nunique / n_rows) if n_rows else 0.0
        bucket = _cardinality_bucket(unique_ratio)
        id_like = unique_ratio >= ID_LIKE_UNIQUE_RATIO
        categorical_cardinality[c] = {
            "nunique": nunique,
            "unique_ratio": round(unique_ratio, 4),
            "bucket": bucket,
            "id_like": id_like,
        }
        if bucket == "high" or id_like:
            high_cardinality_columns.append(c)

    profile = {
        "path": path,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": int(df.isnull().sum().sum()),
        "missing_by_column": missing_by_col,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "target": target_info or None,
        "categorical_cardinality": categorical_cardinality,
        "high_cardinality_columns": high_cardinality_columns,
        "leakage_warnings": _detect_leakage_risks(df.columns.tolist(), target),
    }
    return profile


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("usage: python dataset.py <path-to-csv> [target_column]")
        raise SystemExit(1)

    target_arg = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(profile_dataset(sys.argv[1], target_arg), indent=2, default=str))
