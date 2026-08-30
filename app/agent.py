"""
app/agent.py

Phase 1-9 - Top level agent orchestration.

Wires profile -> plan -> Ralph loop -> Markdown report together for a
single dataset + objective.
"""

from __future__ import annotations

import os

from app.ralph import run_ralph_loop
from app.experiment_memory import ExperimentMemory
from app.reporting import objective_lines, display_status, target_gap_text, diminishing_return_note
from app.planner import create_plan
from tools.dataset import profile_dataset
from app.run_scope import default_experiments_path


def _direction_symbol(direction: str) -> str:
    return "<=" if direction == "minimize" else ">="


def honest_summary_sentence(result: dict) -> str:
    """Plain-language, non-hidden summary sentence used whenever the
    final status is anything other than 'success'. Reused by the CLI
    (app/main.py) and the Streamlit banner (app/streamlit_app.py) so
    the message is identical everywhere -- honest reporting instead of
    a generic "max_iterations_reached"/"search_space_exhausted" string.
    """
    plan = result.get("plan") or {}
    metric = plan.get("metric")
    direction = plan.get("direction", "maximize")
    target_score = plan.get("target_score")
    symbol = _direction_symbol(direction)

    best = result.get("best_experiment")
    if not best:
        return f"Target {metric} {symbol} {target_score} was not achieved. No successful or scored experiment was produced."

    hyperparams = best.get("hyperparams") or {}
    hp_str = ", ".join(f"{k}={v}" for k, v in sorted(hyperparams.items()))
    fe = best.get("feature_engineering")

    achieved_prefix = (
        f"Target {metric} {symbol} {target_score} was achieved during the run "
        f"(stop_mode=optimize kept searching for a better result). "
        if result.get("target_ever_achieved")
        else f"Target {metric} {symbol} {target_score} was not achieved. "
    )

    return (
        f"{achieved_prefix}"
        f"Best observed {metric} was {_fmt_score(best.get('score'))} using "
        f"{best.get('model')} ({hp_str}, feature_engineering={fe})."
    )


def _fmt_score(score) -> str:
    if score is None:
        return "n/a"
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def _build_report(objective: str, target_score, result: dict) -> str:
    profile = result["profile"]
    plan = result["plan"]
    experiments = result["all_experiments"]
    best = result["best_experiment"]

    lines = []
    lines.append("# AI Data Science Agent - Report")
    lines.append("")

    obj_lines = objective_lines(plan)
    lines.append(f"## {obj_lines[0]}")
    lines.append("")
    for line in obj_lines[1:]:
        lines.append(f"- {line}")
    lines.append("")

    lines.append(f"**Objective (as given):** {objective}")
    lines.append("")
    source = "explicitly set by the user" if target_score is not None else "auto-derived (heuristic, see Plan steps below)"
    lines.append(f"**Target score:** {plan.get('target_score')} for metric `{plan.get('metric')}` ({source})")
    lines.append("")
    lines.append(f"**Final status:** {result['status']}")
    lines.append("")

    leakage_warnings = plan.get("leakage_warnings") or []
    if leakage_warnings:
        lines.append(
            "**⚠️ Possible target leakage:** the following column(s) look, by name, like "
            "they may only be known after the outcome being predicted -- verify they're "
            "genuinely available at prediction time before trusting scores that use them:"
        )
        for w in leakage_warnings:
            lines.append(f"- `{w['column']}` -- {w['reason']}")
        lines.append("")
    if result.get("objective_changed"):
        lines.append(
            "**Warning:** this experiments log already contained runs against a "
            f"different objective ({result.get('previous_objective')}). Comparing "
            "scores across the two objectives is not meaningful."
        )
        lines.append("")

    lines.append("## Dataset profile")
    lines.append("")
    lines.append(f"- Rows: {profile.get('rows')}")
    lines.append(f"- Columns: {profile.get('columns')}")
    lines.append(f"- Missing values (total): {profile.get('missing_values')}")
    lines.append(f"- Duplicate rows: {profile.get('duplicate_rows')}")
    target_info = profile.get("target") or {}
    if target_info:
        lines.append(f"- Target column: {target_info.get('name')} ({target_info.get('suggested_task_type')})")
    lines.append("")

    lines.append("## Plan")
    lines.append("")
    lines.append(f"- Task type: {plan.get('task_type')}")
    lines.append(f"- Target: {plan.get('target')}")
    lines.append(f"- Metric: {plan.get('metric')}")
    lines.append(f"- Candidate models: {', '.join(plan.get('candidate_models') or []) or 'n/a'}")
    lines.append("")
    lines.append("Steps:")
    for step in plan.get("steps", []):
        lines.append(f"1. {step}")
    lines.append("")

    # Split into rows produced by THIS run vs. rows already on disk from
    # earlier runs against the same objective -- otherwise stale history
    # (e.g. from a run made before a bug fix) reads as if it happened
    # just now, which is exactly the confusion reported: "target achieved
    # at iteration 1" alongside a table that appears to run to iteration 17.
    n_new = result.get("iterations_run") or 0
    new_experiments = experiments[-n_new:] if n_new else []
    prior_experiments = experiments[:-n_new] if n_new else experiments

    def _iterations_table(rows):
        out = ["| Iteration | Model | Hyperparams | Feature engineering | Metric | Score | Status | Model path |",
               "|---|---|---|---|---|---|---|---|"]
        for exp in rows:
            out.append(
                f"| {exp.get('iteration')} | {exp.get('model')} | {exp.get('hyperparams')} "
                f"| {exp.get('feature_engineering')} | {exp.get('metric')} "
                f"| {_fmt_score(exp.get('score'))} | {display_status(exp)} | {exp.get('model_path') or 'n/a'} |"
            )
        return out

    lines.append("## Iterations (this run)")
    lines.append("")
    lines.extend(_iterations_table(new_experiments) if new_experiments else ["_No new iterations were run._"])
    lines.append("")

    if prior_experiments:
        lines.append("## Prior iterations (earlier runs against this objective)")
        lines.append("")
        lines.extend(_iterations_table(prior_experiments))
        lines.append("")

    lines.append("## Agent reasoning (this run)")
    lines.append("")
    lines.append(
        "Why the strategist chose each experiment -- exploration, directed "
        "hyperparameter expansion, or a stagnation-triggered strategy switch:"
    )
    lines.append("")
    for exp in new_experiments:
        if exp.get("rationale"):
            lines.append(f"- Iteration {exp.get('iteration')} ({exp.get('model')}): {exp.get('rationale')}")
    lines.append("")

    lines.append("## Analyst notes")
    lines.append("")
    for exp in experiments:
        if exp.get("analysis"):
            lines.append(f"- Iteration {exp.get('iteration')}: {exp.get('analysis')}")
    lines.append("")

    lines.append("## Best experiment")
    lines.append("")
    if best:
        lines.append(f"- Model: {best.get('model')}")
        lines.append(f"- Metric: {best.get('metric')}")
        lines.append(f"- Score: {_fmt_score(best.get('score'))}")
        lines.append(f"- Status: {best.get('status')}")
        if best.get("analysis"):
            lines.append(f"- Analyst summary: {best.get('analysis')}")
        if result.get("best_model_path"):
            lines.append(f"- Best model saved to: {result.get('best_model_path')}")
    else:
        lines.append("No successful or scored experiment was produced.")
    lines.append("")

    if result.get("status") not in ("success", "optimized"):
        lines.append("## Honest result summary")
        lines.append("")
        if result.get("status") == "search_space_exhausted":
            lines.append(
                f"Stopped: search space exhausted after {len(experiments)} distinct "
                "configurations -- every candidate model/hyperparameter/"
                "feature-engineering combination (including directed expansion "
                "beyond the static grid) was already tried without reaching the "
                "target."
            )
        else:
            lines.append(f"Stopped after {result.get('iterations_run')} iterations (max iterations reached).")
        lines.append("")
        lines.append(honest_summary_sentence(result))
        lines.append("")

        gap = target_gap_text(plan, best)
        note = diminishing_return_note(best, experiments, plan.get("direction", "maximize"))
        if best or gap or note:
            lines.append("**Target not reached**")
            lines.append("")
            lines.append(f"- Target: {plan.get('metric')} {_direction_symbol(plan.get('direction', 'maximize'))} {plan.get('target_score')}")
            lines.append(f"- Best: {plan.get('metric')} = {_fmt_score(best.get('score')) if best else 'n/a'}")
            if gap:
                lines.append(f"- Gap: {gap}")
            if best:
                hp_str = ", ".join(f"{k}={v}" for k, v in sorted((best.get('hyperparams') or {}).items()))
                lines.append(
                    f"- Best model: {best.get('model')} (feature_engineering={best.get('feature_engineering')}"
                    f"{', ' + hp_str if hp_str else ''})"
                )
            lines.append(f"- Optimization status: {note or 'still improving -- no diminishing-returns signal yet.'}")
            lines.append("")
    elif result.get("status") == "optimized":
        lines.append("## Honest result summary")
        lines.append("")
        lines.append(
            f"Target was achieved during the run (stop_mode=optimize kept searching "
            f"through all {result.get('iterations_run')} iterations for the best "
            "possible result instead of stopping at the first success)."
        )
        lines.append("")

    last_exp = experiments[-1] if experiments else None
    if last_exp and result.get("status") not in ("success",):
        lines.append("## Recommendation")
        lines.append("")
        lines.append(
            "The Analyst's recommendation from the final iteration "
            "(what to try next if this run were continued):"
        )
        lines.append("")
        lines.append(f"> {last_exp.get('analysis') or 'n/a'}")
        lines.append("")

    return "\n".join(lines)


def run_agent(
    dataset_path: str,
    objective: str,
    target_score: float | None = None,
    metric: str | None = None,
    max_iterations: int = 10,
    workspace_dir: str = "workspace",
    use_docker: bool = False,
    experiments_path: str | None = None,
    stop_mode: str = "target",
    reset: bool = False,
    target_column: str | None = None,
) -> dict:
    workdir = os.path.join(workspace_dir, "generated")
    reports_dir = os.path.join(workspace_dir, "reports")
    report_path = os.path.join(reports_dir, "report.md")

    os.makedirs(workdir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Profile once here so we can compute the objective-scoped
    # experiments_path (dataset + task_type + target_column + metric)
    # before entering the Ralph loop; pass the profile through so
    # run_ralph_loop doesn't have to re-profile the same dataset.
    #
    # `target_column`, when given, is an explicit user choice -- passed
    # as profile_dataset's target_hint so the agent predicts what the
    # user actually asked for instead of silently guessing "whichever
    # numeric column looks target-shaped" (profile_dataset's fallback,
    # which is a reasonable last resort but not a substitute for the
    # user telling it directly).
    profile = profile_dataset(dataset_path, target_hint=target_column)
    plan_preview = create_plan(profile, objective, metric=metric, target_score=target_score)

    if experiments_path is None:
        experiments_path = default_experiments_path(
            workspace_dir=workspace_dir,
            dataset_path=dataset_path,
            task_type=plan_preview.get("task_type"),
            target_column=plan_preview.get("target"),
            metric=plan_preview.get("metric"),
        )

    result = run_ralph_loop(
        dataset_path=dataset_path,
        objective=objective,
        target_score=target_score,
        metric=metric,
        max_iterations=max_iterations,
        workdir=workdir,
        experiments_path=experiments_path,
        use_docker=use_docker,
        profile=profile,
        stop_mode=stop_mode,
        reset=reset,
    )

    report_md = _build_report(objective, target_score, result)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    result["report_path"] = report_path
    return result
