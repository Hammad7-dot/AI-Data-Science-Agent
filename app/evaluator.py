"""
app/evaluator.py

Phase 8 - Evaluator.

Parses the single JSON metrics line printed to stdout by generated
scripts (see app.coder) and decides pass/fail against a target score,
respecting metric direction (higher-is-better vs lower-is-better).
"""

from __future__ import annotations

import json

HIGHER_IS_BETTER = {"f1", "accuracy", "r2", "completeness", "precision", "recall"}
LOWER_IS_BETTER = {"rmse", "mae"}

# Known Python exception class names to look for in stderr, in priority
# order. Not an exhaustive taxonomy -- just the handful of failure modes
# worth distinguishing for strategist/UI purposes.
_KNOWN_ERROR_TYPES = ["MemoryError", "ValueError", "KeyError", "RuntimeError"]

# Distinctive substring from app/coder.py's MAX_ONEHOT_COLUMNS safety-cap
# RuntimeError message -- checked before the generic "RuntimeError" match
# so this specific, targeted failure gets its own error_type instead of
# being lumped in as a generic RuntimeError.
_HIGH_CARDINALITY_ENCODING_MARKER = "exceeding the safety cap"


def _classify_error(execution_result, stderr: str) -> str | None:
    """Best-effort classification of a failed execution's error type.

    Checks `execution_result.timed_out` directly for the timeout case
    (no reliable "TimeoutError" string to grep for), then looks for a
    handful of known exception class names in stderr text.
    """
    if getattr(execution_result, "timed_out", False):
        return "TimeoutError"
    stderr = stderr or ""
    if _HIGH_CARDINALITY_ENCODING_MARKER in stderr:
        return "HIGH_CARDINALITY_ENCODING"
    for name in _KNOWN_ERROR_TYPES:
        if name in stderr:
            return name
    return None


def evaluate(execution_result, plan: dict, target_score: float) -> dict:
    """Evaluate an ExecutionResult against `target_score` for the plan's metric."""
    if not getattr(execution_result, "success", False):
        return {
            "status": "fail",
            "score": None,
            "metric": plan.get("metric"),
            "model": None,
            "raw": None,
            "reason": execution_result.stderr or "execution did not succeed",
            "error_type": _classify_error(execution_result, execution_result.stderr),
        }

    stdout = execution_result.stdout or ""
    try:
        last_line = stdout.strip().splitlines()[-1]
        parsed = json.loads(last_line)
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "status": "fail",
            "score": None,
            "metric": plan.get("metric"),
            "model": None,
            "raw": None,
            "reason": execution_result.stderr or f"could not parse output: {exc}",
            "error_type": _classify_error(execution_result, execution_result.stderr),
        }

    score = parsed.get("score")
    metric = parsed.get("metric", plan.get("metric"))
    model = parsed.get("model")

    if score is None:
        return {
            "status": "fail",
            "score": None,
            "metric": metric,
            "model": model,
            "raw": parsed,
            "reason": "parsed output had no 'score' field",
            "error_type": None,
        }

    if metric in LOWER_IS_BETTER:
        passed = score <= target_score
    else:
        passed = score >= target_score

    return {
        "status": "success" if passed else "fail",
        "score": score,
        "metric": metric,
        "model": model,
        "raw": parsed,
        "reason": None if passed else f"score {score} did not meet target {target_score} for metric '{metric}'",
        "error_type": None,
    }
