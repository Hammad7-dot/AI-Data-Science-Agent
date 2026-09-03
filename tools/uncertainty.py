"""Summaries for validation scores produced by cross-validation."""

from __future__ import annotations

import math
from collections.abc import Iterable


def summarize_cv_scores(scores: Iterable[float]) -> dict:
    """Return finite Python-float summaries for cross-validation scores."""
    try:
        cv_scores = [float(score) for score in scores]
    except (TypeError, ValueError) as exc:
        raise ValueError("Cross-validation scores must be finite numbers.") from exc

    if not cv_scores or not all(math.isfinite(score) for score in cv_scores):
        raise ValueError("Cross-validation scores must be finite numbers.")

    cv_mean = math.fsum(cv_scores) / len(cv_scores)
    if len(cv_scores) == 1:
        return {
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": 0.0,
            "cv_interval_95": None,
        }

    cv_std = math.sqrt(math.fsum((score - cv_mean) ** 2 for score in cv_scores) / (len(cv_scores) - 1))
    margin = 1.96 * cv_std / math.sqrt(len(cv_scores))
    return {
        "cv_scores": cv_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_interval_95": [cv_mean - margin, cv_mean + margin],
    }
