"""
app/experiment_memory.py

Experiment Memory -- a small structured view over the list of experiment
records (loaded from disk + accumulated this run) that the adaptive
Ralph Loop and Planner/Strategist reason over instead of blindly walking
a precomputed grid.

# TODO: `summary_for_planner()` is shaped as compact structured data
# rather than prose specifically so that, when an LLM Planner replaces
# app/experiment_strategist.py's rule-based logic, it can be dropped
# straight into a prompt like:
#   "Given this experiment memory: {summary_for_planner()}, and this
#    plan: {plan}, decide what to try next. Respond with JSON with keys
#    'model_name', 'feature_engineering', 'hyperparams', 'rationale'."
"""

from __future__ import annotations

import json


def _identity(record: dict) -> tuple:
    hyperparams = record.get("hyperparams") or {}
    return (
        record.get("model"),
        json.dumps(hyperparams, sort_keys=True),
        record.get("feature_engineering"),
    )


class ExperimentMemory:
    def __init__(self, records: list[dict] | None = None):
        self._records: list[dict] = list(records or [])

    def add(self, record: dict) -> None:
        self._records.append(record)

    def all(self) -> list[dict]:
        return list(self._records)

    def best(self, direction: str) -> dict | None:
        scored = [r for r in self._records if r.get("score") is not None]
        if not scored:
            return None
        if direction == "minimize":
            return min(scored, key=lambda r: r["score"])
        return max(scored, key=lambda r: r["score"])

    def history_for_model(self, model_name: str) -> list[dict]:
        return [r for r in self._records if r.get("model") == model_name]

    def tried_identities(self) -> set:
        return {_identity(r) for r in self._records}

    def recent_scores(self, n: int) -> list[float]:
        scores = [r["score"] for r in self._records if r.get("score") is not None]
        return scores[-n:] if n > 0 else []

    def is_stagnant(
        self,
        window: int = 3,
        min_improvement: float = 0.01,
        direction: str = "maximize",
    ) -> bool:
        """True when the best score across the last `window` scored
        records improved by less than `min_improvement` (in the
        metric's favorable direction) over the best score before that
        window. Not enough history yet -> False (never call it
        stagnant prematurely).
        """
        scored = [r["score"] for r in self._records if r.get("score") is not None]
        if len(scored) < window + 1:
            return False

        recent = scored[-window:]
        earlier = scored[:-window]
        if not earlier:
            return False

        if direction == "minimize":
            best_recent = min(recent)
            best_earlier = min(earlier)
            improvement = best_earlier - best_recent  # positive = got smaller/better
        else:
            best_recent = max(recent)
            best_earlier = max(earlier)
            improvement = best_recent - best_earlier  # positive = got bigger/better

        return improvement < min_improvement

    def is_stagnating(self, patience: int = 3, min_improvement: float = 1e-4) -> bool:
        """Simpler, stricter, direction-agnostic stagnation check (user
        spec): the last `patience` scores span (max-min) less than
        `min_improvement`. Direction-agnostic on purpose -- a flat
        cluster of recent scores means stagnation whether the metric is
        maximized or minimized. Kept alongside `is_stagnant` (still used
        by summary_for_planner) rather than replacing it.
        """
        history = [r["score"] for r in self._records if r.get("score") is not None]
        if len(history) < patience + 1:
            return False
        recent = history[-patience:]
        improvement = max(recent) - min(recent)
        return improvement < min_improvement

    def best_summary_text(self, direction: str = "maximize") -> str:
        """One-line human-readable summary of the single best experiment
        so far, e.g.:
          "random_forest (max_depth=10, n_estimators=200,
           feature_engineering=pipeline): f1=0.8667"
        or "No successful experiments yet" if memory is empty.
        """
        best = self.best(direction)
        if not best:
            return "No successful experiments yet"

        hyperparams = best.get("hyperparams") or {}
        hp_str = ", ".join(f"{k}={v}" for k, v in sorted(hyperparams.items()))
        fe = best.get("feature_engineering")
        parts = [p for p in [hp_str, f"feature_engineering={fe}" if fe else None] if p]
        params_str = ", ".join(parts)

        model = best.get("model")
        metric = best.get("metric")
        score = best.get("score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)

        if params_str:
            return f"{model} ({params_str}): {metric}={score_str}"
        return f"{model}: {metric}={score_str}"

    def summary_for_planner(self, direction: str = "maximize", window: int = 3) -> dict:
        better = (lambda a, b: a < b) if direction == "minimize" else (lambda a, b: a > b)

        per_model_best: dict[str, float] = {}
        for r in self._records:
            if r.get("score") is None or r.get("model") is None:
                continue
            model, score = r["model"], r["score"]
            if model not in per_model_best or better(score, per_model_best[model]):
                per_model_best[model] = score

        best_overall = self.best(direction)

        return {
            "iterations_run": len(self._records),
            "best": (
                {
                    "model": best_overall.get("model"),
                    "score": best_overall.get("score"),
                    "hyperparams": best_overall.get("hyperparams"),
                    "feature_engineering": best_overall.get("feature_engineering"),
                }
                if best_overall
                else None
            ),
            "per_model_best": per_model_best,
            "stagnant": self.is_stagnant(window=window, direction=direction),
        }
