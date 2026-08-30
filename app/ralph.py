"""
app/ralph.py

Phase 9 - Ralph Loop. Extended in Phase 10/11/12 to track the expanded
experiment identity (model, hyperparams, feature_engineering), call
app.analyst after each evaluation, and -- Phase 12 -- replace the
precomputed itertools.product grid walk with an adaptive
exploration -> exploitation -> stagnation-handling strategy driven by
app.experiment_memory / app.experiment_strategist.

Iteratively generate -> execute -> evaluate -> analyze code until the
evaluator's target score is met, the adaptive search space is
exhausted, or max_iterations is reached, persisting a log of every
attempt.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone

from tools.dataset import profile_dataset
from app.planner import create_plan
from app.coder import generate_code
from app.executor import execute
from app.evaluator import evaluate, HIGHER_IS_BETTER, LOWER_IS_BETTER
from app.analyst import interpret_results
from app.experiment_memory import ExperimentMemory
from app.experiment_strategist import choose_next_experiment
from app.run_scope import default_objective_path, default_models_dir


def _load_experiments(experiments_path: str) -> list:
    if os.path.exists(experiments_path):
        try:
            with open(experiments_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_experiments(experiments_path: str, experiments: list) -> None:
    os.makedirs(os.path.dirname(experiments_path) or ".", exist_ok=True)
    with open(experiments_path, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2, default=str)


def _experiment_identity(record: dict) -> tuple:
    """Identity used for "don't repeat identical experiments": model +
    hyperparams + feature_engineering, not just model name, so the
    expanded search space actually gets explored.
    """
    hyperparams = record.get("hyperparams") or {}
    return (
        record.get("model"),
        json.dumps(hyperparams, sort_keys=True),
        record.get("feature_engineering"),
    )


def _objective_path(experiments_path: str) -> str:
    return default_objective_path(experiments_path)


def _load_objective(experiments_path: str) -> dict | None:
    path = _objective_path(experiments_path)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_objective(experiments_path: str, objective: dict) -> None:
    path = _objective_path(experiments_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(objective, f, indent=2, default=str)


# Used by experiment_memory.is_stagnating() / the strategist's diminishing-
# return detection to decide "is this worth continuing to chase" -- NOT by
# _is_better() below. Best-result tracking must record the genuine highest
# score seen, however small the margin, or the reported "best" silently
# lags behind an iteration that actually scored higher (e.g. n_estimators
# 760 beating 507 by only 0.00006 is still the real best model).
MIN_IMPROVEMENT = 1e-4


def _is_better(candidate: dict, current_best: dict | None, metric: str) -> bool:
    # A failed/crashed candidate (score=None) must NEVER become "best",
    # not even as the very first-ever candidate -- check this before the
    # `current_best is None` short-circuit below, otherwise the first
    # crashed iteration of a run wins the best slot unconditionally.
    if candidate.get("score") is None:
        return False
    if current_best is None:
        return True
    if current_best.get("score") is None:
        return True
    if metric in LOWER_IS_BETTER:
        return candidate["score"] < current_best["score"]
    return candidate["score"] > current_best["score"]


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _write_best_model_artifacts(models_dir: str, best_experiment: dict, plan: dict) -> None:
    """Write best_model.json, feature_info.json, training_history.json is
    handled separately -- this writes the two per-winning-experiment
    files. Consolidates what would otherwise be a fourth near-duplicate
    "metrics.json" file: accuracy/precision/recall/f1 (or
    rmse/mae/r2) -- whatever's on the winning record -- lives inside
    best_model.json instead of a separate file (see README.md).
    """
    best_model_json = {
        "model": best_experiment.get("model"),
        "task_type": plan.get("task_type"),
        "metric": best_experiment.get("metric"),
        "score": best_experiment.get("score"),
        "target_score": plan.get("target_score"),
        "direction": plan.get("direction"),
        "feature_engineering": best_experiment.get("feature_engineering"),
        "hyperparameters": best_experiment.get("hyperparams"),
        "status": best_experiment.get("status"),
    }
    # Fold in any extra metric fields present on the raw evaluator output
    # (accuracy/precision/recall/f1 or rmse/mae/r2) instead of a
    # separate metrics.json.
    for key in ("accuracy", "precision", "recall", "f1", "rmse", "mae", "r2"):
        if key in best_experiment:
            best_model_json[key] = best_experiment[key]
    _write_json(os.path.join(models_dir, "best_model.json"), best_model_json)

    feature_info_json = {
        "features": best_experiment.get("feature_names"),
        "target": plan.get("target"),
        "n_features": best_experiment.get("n_features"),
        "train_samples": best_experiment.get("train_samples"),
        "test_samples": best_experiment.get("test_samples"),
    }
    _write_json(os.path.join(models_dir, "feature_info.json"), feature_info_json)


def _write_training_history(models_dir: str, experiments: list) -> None:
    history = [
        {
            "iteration": r.get("iteration"),
            "model": r.get("model"),
            "metric": r.get("metric"),
            "score": r.get("score"),
            "status": r.get("status"),
            "feature_engineering": r.get("feature_engineering"),
            "hyperparams": r.get("hyperparams"),
        }
        for r in experiments
    ]
    _write_json(os.path.join(models_dir, "training_history.json"), history)


def run_ralph_loop(
    dataset_path: str,
    objective: str,
    target_score: float | None,
    metric: str | None,
    max_iterations: int,
    workdir: str,
    experiments_path: str,
    use_docker: bool = False,
    profile: dict | None = None,
    stop_mode: str = "target",
    reset: bool = False,
) -> dict:
    if profile is None:
        profile = profile_dataset(dataset_path)

    # The objective (metric, target_score, direction) is frozen here, at
    # the single entry point into the plan, and is never reassigned
    # anywhere below this line. `target_score=None` means "not specified
    # by the caller" -- create_plan auto-derives a scale-aware value.
    plan = create_plan(profile, objective, metric=metric, target_score=target_score)

    # Record/compare the frozen objective against any prior run that
    # used this same experiments_path, so mixing incompatible histories
    # is surfaced honestly instead of silently happening.
    current_objective = {
        "task_type": plan.get("task_type"),
        "target_column": plan.get("target"),
        "metric": plan.get("metric"),
        "target_score": plan.get("target_score"),
        "direction": plan.get("direction"),
        "candidate_models": plan.get("candidate_models"),
    }
    # experiments_path = <workspace_dir>/experiments/<scope_key>/experiments.json
    # (see app/run_scope.py::default_experiments_path) -- walk back up
    # structurally to recover workspace_dir rather than string-replacing
    # "experiments" -> "models", which is fragile.
    workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(experiments_path))))
    models_dir = default_models_dir(
        workspace_dir=workspace_dir,
        dataset_path=dataset_path,
        task_type=plan.get("task_type"),
        target_column=plan.get("target"),
        metric=plan.get("metric"),
    )

    # `reset=True` starts this objective's history over from scratch --
    # old records (e.g. from a run made under a since-fixed bug) are
    # otherwise loaded and displayed alongside this run's own results,
    # which reads as "the loop kept going" even when it correctly
    # stopped after one iteration this time.
    previous_objective = None if reset else _load_objective(experiments_path)
    objective_changed = bool(
        previous_objective
        and (
            previous_objective.get("metric") != current_objective.get("metric")
            or previous_objective.get("target_score") != current_objective.get("target_score")
        )
    )
    _save_objective(experiments_path, current_objective)

    experiments = [] if reset else _load_experiments(experiments_path)
    status = "max_iterations_reached"
    iterations_run = 0
    target_ever_achieved = False

    # Seed cross-run memory from every record already on disk.
    memory = ExperimentMemory(experiments)
    previous_feedback = None

    direction = plan.get("direction", "maximize")
    candidate_models = plan.get("candidate_models") or []

    # Seed best_experiment from the FULL history already on disk (via
    # `memory`, which was just constructed from it), not None -- a run
    # against an existing experiments_path must start its best-tracking
    # from the true best-so-far, otherwise a later run's own single
    # local result can trivially "win" against an empty/None seed even
    # when an earlier run already found something better. If the
    # objective changed since the last run (objective_changed, above),
    # comparing scores across incompatible metrics would be meaningless,
    # so start from scratch in that case instead.
    best_experiment = None if (objective_changed or reset) else memory.best(direction)

    # Continuous iteration numbering across runs against the same
    # experiments_path: `iteration_offset` is the count of records
    # already on disk from prior runs, so this run's records report
    # iteration numbers that continue where the last run left off
    # instead of every run restarting the display count at 1.
    iteration_offset = len(experiments)

    for iteration in range(1, max_iterations + 1):
        if candidate_models:
            choice = choose_next_experiment(plan, memory, iteration)
            if choice is None:
                status = "search_space_exhausted"
                break
            model_name = choice["model_name"]
            feature_engineering = choice["feature_engineering"]
            hyperparams = choice["hyperparams"]
            rationale = choice.get("rationale")
        else:
            # eda plan: no adaptive search space, single-script behavior.
            model_name = feature_engineering = hyperparams = None
            rationale = None

        iterations_run = iteration
        global_iteration = iteration_offset + iteration

        code = generate_code(
            dataset_path,
            plan,
            global_iteration,
            previous_feedback=previous_feedback,
            model_name=model_name,
            feature_engineering=feature_engineering,
            hyperparams=hyperparams,
            models_dir=models_dir,
        )
        execution_result = execute(code, workdir, iteration, use_docker=use_docker)
        eval_result = evaluate(execution_result, plan, plan.get("target_score"))
        analysis = interpret_results(execution_result, eval_result, plan, memory=memory)

        raw = eval_result.get("raw") or {}
        record = {
            "iteration": global_iteration,
            "model": eval_result.get("model") or model_name,
            "metric": eval_result.get("metric"),
            "score": eval_result.get("score"),
            "status": eval_result.get("status"),
            "reason": eval_result.get("reason"),
            "error_type": eval_result.get("error_type"),
            "hyperparams": raw.get("hyperparams") if raw.get("hyperparams") is not None else hyperparams,
            "feature_engineering": raw.get("feature_engineering") or feature_engineering,
            "model_path": raw.get("model_path"),
            "n_features": raw.get("n_features"),
            "train_samples": raw.get("train_samples"),
            "test_samples": raw.get("test_samples"),
            "feature_names": raw.get("feature_names"),
            "rationale": rationale,
            "analysis": analysis.get("summary"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        experiments.append(record)
        _save_experiments(experiments_path, experiments)
        memory.add(record)

        if _is_better(eval_result, best_experiment, plan.get("metric")):
            best_experiment = dict(record)

        if eval_result.get("status") == "success":
            target_ever_achieved = True
            if stop_mode != "optimize":
                status = "success"
                break

        previous_feedback = analysis.get("recommendation") or eval_result.get("reason")

    if stop_mode == "optimize" and target_ever_achieved and status != "success":
        status = "optimized"

    best_model_path = None
    if best_experiment is not None and best_experiment.get("model_path"):
        try:
            os.makedirs(models_dir, exist_ok=True)
            best_model_path = os.path.join(models_dir, "best_model.joblib")
            shutil.copy2(best_experiment["model_path"], best_model_path)
        except OSError:
            best_model_path = None

    if best_experiment is not None:
        try:
            os.makedirs(models_dir, exist_ok=True)
            _write_best_model_artifacts(models_dir, best_experiment, plan)
        except OSError:
            pass
    try:
        os.makedirs(models_dir, exist_ok=True)
        _write_training_history(models_dir, experiments)
    except OSError:
        pass

    stagnation_detected = memory.is_stagnating()

    result = {
        "status": status,
        "iterations_run": iterations_run,
        "best_experiment": best_experiment,
        "best_model_path": best_model_path,
        "all_experiments": experiments,
        "plan": plan,
        "profile": profile,
        "objective_changed": objective_changed,
        "target_ever_achieved": target_ever_achieved,
        "stop_mode": stop_mode,
        "experiments_path": experiments_path,
        "stagnation_detected": stagnation_detected,
    }
    if objective_changed:
        result["previous_objective"] = previous_objective
    return result
