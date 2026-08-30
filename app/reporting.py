"""
app/reporting.py

Small shared helper for rendering the "Objective" panel (task type,
metric, target score with direction-aware comparison symbol, direction)
from a `plan` dict. Used identically by app/agent.py (report.md),
app/main.py (CLI output), and app/streamlit_app.py (UI panel) so the
objective is formatted the same way everywhere instead of three
separate ad hoc implementations.
"""

from __future__ import annotations

from tools.ml import DIMINISHING_RETURN_THRESHOLD, hyperparameter_grid, simple_trend


def _direction_symbol(direction: str) -> str:
    return "<=" if direction == "minimize" else ">="


def objective_lines(plan: dict) -> list[str]:
    """Plain-text lines describing the run's objective, derived from
    `plan` (the single frozen source of truth for task_type, metric,
    target_score, direction -- see app/ralph.py run_ralph_loop).
    """
    direction = plan.get("direction", "maximize")
    symbol = _direction_symbol(direction)
    return [
        "Objective",
        f"Task: {plan.get('task_type')}",
        f"Metric: {plan.get('metric')}",
        f"Target: {symbol} {plan.get('target_score')}",
        f"Direction: {'Minimize' if direction == 'minimize' else 'Maximize'}",
    ]


def display_status(record: dict) -> str:
    """Three-way display status for one experiment record, distinct from
    the underlying `status` field used for loop control (which stays
    "success"/"fail" everywhere else -- evaluator.py, ralph.py's stop
    condition, memory.is_stagnating, etc. -- unchanged).

    "fail" collapses two very different situations that read the same
    way in a plain success/fail badge: an experiment that trained fine
    and simply scored below the target, vs. one that never produced a
    score at all (crashed, timed out, threw during preprocessing). Split
    them for display:
      - "success"      -- status == "success"
      - "below_target" -- status == "fail" but a real score was produced
      - "failed"        -- status == "fail" and score is None (didn't execute)
    """
    if record.get("status") == "success":
        return "success"
    if record.get("score") is not None:
        return "below_target"
    return "failed"


DISPLAY_STATUS_EMOJI = {"success": "🟢", "below_target": "🟡", "failed": "🔴"}


def target_gap_text(plan: dict, best_experiment: dict | None) -> str | None:
    """Signed distance from the winning experiment's score to the target,
    in the metric's favorable direction (positive = still short of the
    target). None when there's no scored best experiment or no target
    to compare against.
    """
    if not best_experiment or best_experiment.get("score") is None:
        return None
    target = plan.get("target_score")
    if target is None:
        return None
    direction = plan.get("direction", "maximize")
    score = best_experiment["score"]
    gap = (target - score) if direction != "minimize" else (score - target)
    return f"{gap:.4f}"


def diminishing_return_note(best_experiment: dict | None, all_experiments: list, direction: str) -> str | None:
    """One-line, target-agnostic note naming any hyperparameter on the
    winning model whose most recent step gained less than
    tools.ml.DIMINISHING_RETURN_THRESHOLD -- i.e. the same signal
    app.experiment_strategist already uses to stop expanding that key --
    surfaced here for the final report/summary instead of only shaping
    the next iteration's choice silently. None if there's no winning
    model or nothing is flattening out yet.
    """
    if not best_experiment:
        return None
    model = best_experiment.get("model")
    if not model:
        return None
    model_hist = [e for e in all_experiments if e.get("model") == model]
    trend = simple_trend(model_hist, model, direction=direction)
    diminishing = [k for k, v in trend.items() if v == "diminishing"]
    if not diminishing:
        return None
    other_keys = [k for k in hyperparameter_grid(model).keys() if k not in diminishing]
    suggestion = ", ".join(other_keys) if other_keys else "an alternative feature engineering strategy"
    keys_str = ", ".join(diminishing)
    return (
        f"Diminishing returns detected for {keys_str} (most recent improvement below "
        f"{DIMINISHING_RETURN_THRESHOLD}). Further increasing {keys_str} is unlikely to "
        f"close the gap -- recommend exploring {suggestion} instead."
    )


def objective_dict(plan: dict) -> dict:
    """Structured form of the same objective info, for callers (e.g.
    Streamlit) that want to render fields individually rather than as
    preformatted lines.
    """
    direction = plan.get("direction", "maximize")
    return {
        "task_type": plan.get("task_type"),
        "metric": plan.get("metric"),
        "target_score": plan.get("target_score"),
        "direction": direction,
        "symbol": _direction_symbol(direction),
    }
