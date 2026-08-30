"""
app/main.py

CLI entrypoint for the AI Data Science Agent (Phase 0 MVP).

Usage:
    python app/main.py --dataset workspace/datasets/data.csv \\
        --objective "Predict churn" --target-score 0.80 --metric f1 \\
        --max-iterations 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import run_agent, honest_summary_sentence  # noqa: E402
from app.reporting import (  # noqa: E402
    objective_lines,
    display_status,
    DISPLAY_STATUS_EMOJI,
    target_gap_text,
    diminishing_return_note,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AI Data Science Agent - Phase 0 MVP")
    parser.add_argument("--dataset", required=True, help="Path to input CSV dataset")
    parser.add_argument("--objective", required=True, help="Natural language objective")
    parser.add_argument(
        "--target-score",
        type=float,
        default=None,
        help="Target metric score. Omit to auto-derive a scale-aware default from the task/metric.",
    )
    parser.add_argument("--metric", default=None, help="Metric override (default inferred from task type)")
    parser.add_argument(
        "--target-column",
        default=None,
        help=(
            "Explicit target column name to predict. Omit to auto-detect (common target-ish "
            "names, else the last column) -- auto-detect is a reasonable default but not a "
            "substitute for telling the agent what you actually want predicted."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=10, help="Max Ralph loop iterations")
    parser.add_argument("--workspace-dir", default="workspace", help="Workspace directory for outputs")
    parser.add_argument(
        "--experiments-path",
        default=None,
        help=(
            "Explicit override for the experiments.json path. Omit to auto-scope "
            "per objective (dataset + task type + target column + metric) under "
            "<workspace-dir>/experiments/."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["target", "optimize"],
        default="target",
        help=(
            "target = stop as soon as any model meets the goal (default). "
            "optimize = keep searching for the best possible result up to max iterations."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help=(
            "Start this objective's experiment history over from scratch, ignoring "
            "any experiments.json already on disk for it (e.g. from a run made "
            "before a bug fix). Without this, prior history is loaded and factored "
            "into best-tracking/stagnation as usual."
        ),
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        default=False,
        help=(
            "Run generated code inside the Docker sandbox (tools/python_runner.run_script_in_docker) "
            "instead of a plain local subprocess. Off by default since Docker may not be available "
            "in dev/test environments; falls back to plain subprocess automatically if Docker is missing."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    result = run_agent(
        dataset_path=str(Path(args.dataset).resolve()),
        objective=args.objective,
        target_score=args.target_score,
        metric=args.metric,
        max_iterations=args.max_iterations,
        workspace_dir=args.workspace_dir,
        use_docker=args.sandbox,
        experiments_path=args.experiments_path,
        stop_mode=args.mode,
        reset=args.reset,
        target_column=args.target_column,
    )

    best = result.get("best_experiment") or {}
    plan = result.get("plan") or {}

    print("=" * 60)
    print("AI Data Science Agent - Run Trail")
    print("=" * 60)
    n_new = result.get("iterations_run") or 0
    all_experiments = result.get("all_experiments") or []
    new_experiments = all_experiments[-n_new:] if n_new else []
    for exp in new_experiments:
        ds = display_status(exp)
        print(f"Iteration {exp.get('iteration')}")
        print(f"{exp.get('model')}")
        print(f"{exp.get('metric')} = {exp.get('score')}")
        if ds == "success":
            print(f"{DISPLAY_STATUS_EMOJI[ds]} success -- target met")
        elif ds == "below_target":
            print(f"{DISPLAY_STATUS_EMOJI[ds]} below_target -- executed successfully, target not met")
        else:
            print(f"{DISPLAY_STATUS_EMOJI[ds]} failed -- did not execute ({exp.get('error_type') or exp.get('reason') or 'unknown error'})")
        if exp.get("rationale"):
            print("Agent reasoning:")
            print(f'  "{exp.get("rationale")}"')
        print()

    leakage_warnings = plan.get("leakage_warnings") or []
    if leakage_warnings:
        print("=" * 60)
        print("⚠️  POSSIBLE TARGET LEAKAGE")
        print("=" * 60)
        for w in leakage_warnings:
            print(f"  {w['column']}: {w['reason']}")

    print("=" * 60)
    print("AI Data Science Agent - Run Summary")
    print("=" * 60)
    obj_lines = objective_lines(plan)
    print(obj_lines[0])
    print("-" * 60)
    for line in obj_lines[1:]:
        print(line)
    print("=" * 60)
    stop_mode = result.get("stop_mode")
    print(
        f"Stop mode:       {stop_mode} "
        f"({'stop at first success' if stop_mode != 'optimize' else 'search to max iterations'})"
    )
    print(f"Status:          {result['status']}")
    print(f"Iterations run:  {result['iterations_run']}")
    print(f"Best model:      {best.get('model', 'n/a')}")
    print(f"Best metric:     {best.get('metric', 'n/a')}")
    print(f"Best score:      {best.get('score', 'n/a')}")
    print(f"Report path:     {result['report_path']}")
    if result.get("best_model_path"):
        print(f"Best model saved to: {result['best_model_path']}")
    if result["status"] not in ("success",):
        print("-" * 60)
        print(honest_summary_sentence(result))
        gap = target_gap_text(plan, best)
        note = diminishing_return_note(best, all_experiments, plan.get("direction", "maximize"))
        if best or gap or note:
            print("-" * 60)
            print("TARGET NOT REACHED")
            print(f"  Target: {plan.get('metric')} {'<=' if plan.get('direction') == 'minimize' else '>='} {plan.get('target_score')}")
            print(f"  Best:   {plan.get('metric')} = {best.get('score', 'n/a')}")
            if gap:
                print(f"  Gap:    {gap}")
            print(f"  Optimization status: {note or 'still improving -- no diminishing-returns signal yet.'}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
