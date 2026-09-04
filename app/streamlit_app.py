"""
app/streamlit_app.py

Phase 12a - Streamlit UI.

Run with:
    streamlit run app/streamlit_app.py

from the project root.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.agent import run_agent, honest_summary_sentence  # noqa: E402
from app.reporting import (  # noqa: E402
    objective_lines,
    display_status,
    DISPLAY_STATUS_EMOJI,
    target_gap_text,
    diminishing_return_note,
)
from app.evaluator import LOWER_IS_BETTER  # noqa: E402
from app.run_scope import store_upload

COEF_MODELS = {"linear_regression", "ridge", "logistic_regression"}
IMPORTANCE_MODELS = {
    "random_forest",
    "gradient_boosting",
    "random_forest_regressor",
    "gradient_boosting_regressor",
}

st.set_page_config(page_title="AI Data Science Agent", layout="wide")
st.title("AI Data Science Agent")

METRIC_OPTIONS = ["auto", "f1", "accuracy", "rmse", "r2"]

with st.sidebar:
    st.header("Run configuration")
    uploaded_file = st.file_uploader("Dataset (CSV)", type=["csv"])
    # Explicit target-column choice instead of only silently auto-detecting
    # "whichever numeric column looks target-shaped" -- with several
    # plausible numeric columns (white_rating, turns, opening_ply, ...)
    # auto-detect can guess something the user didn't mean to predict.
    target_column_choice = "(auto-detect)"
    if uploaded_file is not None:
        try:
            _cols = pd.read_csv(uploaded_file, nrows=0).columns.tolist()
            uploaded_file.seek(0)
            target_column_choice = st.selectbox(
                "Target column (what to predict)",
                options=["(auto-detect)"] + _cols,
                index=0,
                help=(
                    "Auto-detect guesses from common target-ish names or falls back to "
                    "the last column -- pick explicitly to be sure the agent predicts "
                    "what you actually want, not just whichever column looks numeric."
                ),
            )
        except Exception:
            st.caption("Could not read column names from this file yet -- target will auto-detect.")
    objective = st.text_input("Objective", value="Predict the target column")
    metric_choice = st.selectbox("Target metric", METRIC_OPTIONS, index=0)
    auto_target = st.checkbox("Auto-select target from dataset scale", value=False)
    min_score = st.number_input(
        "Minimum score", min_value=0.0, max_value=1000.0, value=0.80, step=0.05, disabled=auto_target
    )
    if auto_target:
        st.caption(
            "Auto-select is on: the Minimum score field above is ignored and a "
            "heuristic target is derived from the dataset instead. Uncheck this "
            "to set your own target score."
        )
    max_iterations = st.number_input("Max iterations", min_value=1, max_value=10, value=10, step=1)
    use_docker = st.checkbox("Run code in Docker sandbox", value=False)
    reset_history = st.checkbox(
        "Start fresh (clear prior experiment history for this objective)",
        value=False,
        help=(
            "Off by default: prior runs against the same dataset/objective are kept "
            "and shown for reference. Check this to ignore/replace that history -- "
            "useful after a bug fix, or to avoid old runs cluttering the results."
        ),
    )
    stop_mode = st.radio(
        "Stop mode",
        options=["target", "optimize"],
        index=0,
        help=(
            "target = stop as soon as any model meets the goal. "
            "optimize = keep searching for the best possible result up to max iterations."
        ),
    )
    run_clicked = st.button("Run Agent", type="primary")


def _status_badge(exp: dict) -> str:
    ds = display_status(exp)
    label = {"success": "success", "below_target": "below_target", "failed": "failed"}[ds]
    return f"{DISPLAY_STATUS_EMOJI[ds]} `{label}`"


def _fmt_score_val(score) -> str:
    if score is None:
        return "n/a"
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def _load_json_artifact(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as artifact:
            payload = json.load(artifact)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _format_interval(interval) -> str:
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        return f"[{_fmt_score_val(interval[0])}, {_fmt_score_val(interval[1])}]"
    return "n/a"


def _render_model_parameters(best: dict, model_path: str, feature_info_json: dict | None) -> None:
    """Model-specific interpretability view: coefficients for linear-ish
    models, feature importances for tree-based models, and a graceful
    skip message otherwise. Pipeline-variant experiments (feature
    engineering != 'basic') are skipped with an explanatory caption
    instead of showing misleading unaligned coefficients/importances,
    since a ColumnTransformer/OneHotEncoder step means the fitted
    estimator's coef_/feature_importances_ no longer line up 1:1 with
    feature_info.json's original feature names.
    """
    model_name = best.get("model")
    feature_engineering = best.get("feature_engineering")
    feature_names = (feature_info_json or {}).get("features") or best.get("feature_names")

    if model_name not in COEF_MODELS and model_name not in IMPORTANCE_MODELS:
        st.caption("This model type doesn't expose interpretable coefficients or feature importances.")
        return

    if feature_engineering == "pipeline":
        st.caption(
            "Coefficients/feature importances not shown for pipeline-preprocessed "
            "models -- the ColumnTransformer/OneHotEncoder step means the fitted "
            "estimator's parameters no longer line up 1:1 with the original feature names."
        )
        return

    try:
        loaded = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Could not load saved model to inspect parameters: {exc}")
        return

    estimator = loaded
    if hasattr(loaded, "named_steps"):
        estimator = loaded.named_steps.get("model", loaded)

    if not feature_names:
        st.caption("Feature names unavailable -- cannot align coefficients/importances.")
        return

    if model_name in COEF_MODELS and hasattr(estimator, "coef_"):
        coef = estimator.coef_
        coef = coef[0] if getattr(coef, "ndim", 1) > 1 else coef
        if len(coef) == len(feature_names):
            st.write("Intercept:", estimator.intercept_)
            coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coef}).sort_values(
                "coefficient", key=abs, ascending=False
            )
            st.dataframe(coef_df, use_container_width=True)
        else:
            st.caption("Coefficient count doesn't match feature count -- skipping to avoid a misleading pairing.")
    elif model_name in IMPORTANCE_MODELS and hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        if len(importances) == len(feature_names):
            imp_df = (
                pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .set_index("feature")
            )
            st.bar_chart(imp_df)
        else:
            st.caption("Feature importance count doesn't match feature count -- skipping to avoid a misleading pairing.")
    else:
        st.caption("This model type doesn't expose interpretable coefficients or feature importances.")


def _save_uploaded_file(uploaded) -> str:
    return store_upload(PROJECT_ROOT / "workspace", uploaded.name, uploaded.getbuffer())


if run_clicked:
    if uploaded_file is None:
        st.error("Please upload a CSV dataset before running the agent.")
    elif not objective.strip():
        st.error("Please provide an objective.")
    else:
        try:
            dataset_path = _save_uploaded_file(uploaded_file)
            metric = None if metric_choice == "auto" else metric_choice

            with st.status("Running AI Data Science Agent...", expanded=True) as status_box:
                st.write(f"Dataset saved to `{dataset_path}`")
                st.write("Running plan -> code -> execute -> evaluate -> analyze loop...")
                result = run_agent(
                    dataset_path=dataset_path,
                    objective=objective,
                    target_score=None if auto_target else float(min_score),
                    metric=metric,
                    max_iterations=int(max_iterations),
                    workspace_dir=str(PROJECT_ROOT / "workspace"),
                    stop_mode=stop_mode,
                    use_docker=use_docker,
                    reset=reset_history,
                    target_column=None if target_column_choice == "(auto-detect)" else target_column_choice,
                )
                status_box.update(label="Run complete", state="complete")

            plan = result.get("plan") or {}
            if result.get("status") == "sandbox_unavailable":
                st.error(result["all_experiments"][-1]["reason"])
            if plan.get("task_type") in ("classification", "regression"):
                st.caption("Search scores and target status use out-of-fold cross-validation on development data.")
                holdout = result.get("holdout_evaluation")
                if holdout:
                    st.metric(f"Final holdout {holdout['metric']}", f"{holdout['score']:.4f}")
                    st.caption(f"Evaluated on {holdout['samples']} held-out rows after model selection; not used to choose the winner.")
                elif result.get("holdout_error"):
                    st.warning(f"Final holdout evaluation unavailable: {result['holdout_error']}")
            obj_lines = objective_lines(plan)
            with st.container(border=True):
                st.markdown("  \n".join([f"**{obj_lines[0]}**"] + obj_lines[1:]))

            leakage_warnings = plan.get("leakage_warnings") or []
            if leakage_warnings:
                with st.container(border=True):
                    st.markdown("**⚠️ Possible target leakage**")
                    st.caption(
                        "These column names suggest they may only be known after the "
                        "outcome being predicted -- verify they're genuinely available at "
                        "prediction time before trusting scores that use them as features."
                    )
                    for w in leakage_warnings:
                        st.markdown(f"- `{w['column']}` -- {w['reason']}")

            result_stop_mode = result.get("stop_mode")
            if result_stop_mode == "optimize":
                st.markdown("**Stop Mode:** 🔬 optimize (search to max iterations)")
            else:
                st.markdown("**Stop Mode:** 🎯 target (stop at first success)")

            experiments = result.get("all_experiments") or []
            best = result.get("best_experiment") or {}
            models_dir = os.path.dirname(result.get("best_model_path") or "") or None
            direction = plan.get("direction", "maximize")
            metric = plan.get("metric")
            target_score = plan.get("target_score")

            # -- Load the new purpose-built artifact files when present,
            # falling back to fields already on `result` if a file is
            # missing (e.g. no scored experiment at all this run).
            best_model_json = None
            feature_info_json = None
            model_schema = result.get("model_schema")
            explanation = result.get("explainability")
            if models_dir:
                bm_path = os.path.join(models_dir, "best_model.json")
                fi_path = os.path.join(models_dir, "feature_info.json")
                if os.path.exists(bm_path):
                    best_model_json = _load_json_artifact(bm_path)
                if os.path.exists(fi_path):
                    feature_info_json = _load_json_artifact(fi_path)
                if not isinstance(model_schema, dict):
                    model_schema = _load_json_artifact(os.path.join(models_dir, "schema.json"))
                if not isinstance(explanation, dict):
                    explanation = _load_json_artifact(os.path.join(models_dir, "explainability.json"))

            # === 1. Best Model card =========================================
            # `best` (= result["best_experiment"]) is the single canonical
            # best-result object -- used here, in Agent Analysis, and in
            # the report, so they can never disagree. A below-target
            # result (score present, target not met) is still a real best
            # result and must be shown, not hidden -- only a genuinely
            # unscored best_experiment (score is None) has nothing to show.
            best_is_scored = bool(best) and best.get("score") is not None
            if best and not best_is_scored:
                best = {}
            if best:
                with st.container(border=True):
                    st.subheader("🏆 Best Model")
                    target_met = result.get("status") in ("success", "optimized")
                    cols = st.columns(3)
                    cols[0].metric("Model", best.get("model"))
                    cols[1].metric(f"Score ({best.get('metric')})", _fmt_score_val(best.get("score")))
                    cols[2].metric("Target", _fmt_score_val(target_score))
                    folds = best.get("cv_folds")
                    if folds is None:
                        folds = len(best.get("cv_scores") or [])
                    trust_cols = st.columns(4)
                    trust_cols[0].metric("CV folds", folds)
                    trust_cols[1].metric("CV mean", _fmt_score_val(best.get("cv_mean")))
                    trust_cols[2].metric("CV std", _fmt_score_val(best.get("cv_std")))
                    trust_cols[3].metric("95% descriptive interval", _format_interval(best.get("cv_interval_95")))
                    st.markdown(
                        f"**Task type:** `{plan.get('task_type')}` &nbsp;&nbsp; "
                        f"**Target met:** {'✅ Yes' if target_met else '⚠️ No'} &nbsp;&nbsp; "
                        f"**Feature engineering:** `{best.get('feature_engineering')}`"
                    )
                    successful_scores = [
                        e.get("score") for e in experiments if e.get("status") == "success" and e.get("score") is not None
                    ]
                    if len(successful_scores) > 1 and successful_scores[0]:
                        improvement_pct = (best.get("score") - successful_scores[0]) / abs(successful_scores[0]) * 100
                        st.markdown(f"**Improvement over first success:** {improvement_pct:+.2f}%")
                    if best.get("hyperparams"):
                        st.markdown("**Hyperparameters:**")
                        st.json(best.get("hyperparams"))
                    if feature_info_json:
                        scols = st.columns(3)
                        scols[0].metric("Train samples", feature_info_json.get("train_samples"))
                        scols[1].metric("Test samples", feature_info_json.get("test_samples"))
                        scols[2].metric("Features", feature_info_json.get("n_features"))
            else:
                failed = [e for e in experiments if e.get("status") != "success"]
                error_types = Counter(e.get("error_type") for e in failed if e.get("error_type"))
                if failed and error_types:
                    with st.container(border=True):
                        st.subheader("🏆 Best Model")
                        st.write("No successful model yet.")
                        st.write(f"{len(failed)} experiments failed.")
                        primary_type, _ = error_types.most_common(1)[0]
                        st.write(f"Primary failure: `{primary_type}`")
                else:
                    st.info(honest_summary_sentence(result))

            st.subheader("Raw input schema")
            if isinstance(model_schema, dict) and isinstance(model_schema.get("raw_columns"), list):
                target_name = model_schema.get("target_name")
                if target_name is not None:
                    st.caption(f"Prediction input target: {target_name}")
                dtype_families = model_schema.get("dtype_families") or {}
                schema_rows = [
                    {"column": column, "dtype family": dtype_families.get(column, "unknown")}
                    for column in model_schema["raw_columns"]
                ]
                st.dataframe(pd.DataFrame(schema_rows), width="stretch", hide_index=True)
            else:
                st.caption("Raw input schema metadata is unavailable for this run.")

            st.subheader("Model explanation")
            if isinstance(explanation, dict) and explanation.get("available"):
                st.caption(f"Method: {explanation.get('method', 'n/a')}")
                explanation_rows = [
                    row for row in explanation.get("features", []) if isinstance(row, dict)
                ]
                if explanation_rows:
                    explanation_df = pd.DataFrame(explanation_rows).sort_values(
                        "importance", ascending=False
                    ).head(10)
                    st.dataframe(explanation_df, width="stretch", hide_index=True)
                    st.bar_chart(explanation_df.set_index("feature")["importance"])
                else:
                    st.caption("Explanation metadata did not include feature rows.")
            else:
                reason = explanation.get("reason") if isinstance(explanation, dict) else None
                st.caption(f"Unavailable: {reason or 'model explanation metadata was not produced.'}")

            # === 2. Training Progress chart =================================
            if experiments:
                st.subheader("📈 Training Progress")
                progress_df = pd.DataFrame(
                    {
                        "iteration": [e.get("iteration") for e in experiments],
                        "score": [e.get("score") for e in experiments],
                    }
                ).dropna()
                if not progress_df.empty:
                    st.line_chart(progress_df.set_index("iteration"))

            # === 3. Per-iteration list (with running "best so far" tag) ====
            # Split into rows produced by THIS run vs. rows already on disk
            # from earlier runs against the same objective. Without this
            # split, stale history (e.g. from a run made before a bug fix)
            # renders as if it happened just now -- exactly the reported
            # confusion of "target achieved at iteration 1" next to a list
            # that appears to run all the way to iteration 17.
            _n_new = result.get("iterations_run") or 0
            _new_experiments = experiments[-_n_new:] if _n_new else []
            _prior_experiments = experiments[:-_n_new] if _n_new else experiments

            def _render_iteration_rows(rows, running_best_score):
                better = (lambda a, b: a < b) if direction == "minimize" else (lambda a, b: a > b)
                for exp in rows:
                    score = exp.get("score")
                    is_new_best = score is not None and (running_best_score is None or better(score, running_best_score))
                    if is_new_best:
                        running_best_score = score
                    with st.container(border=True):
                        st.markdown(f"**Iteration {exp.get('iteration')}** {_status_badge(exp)}")
                        cols = st.columns(4)
                        cols[0].markdown(f"Model: `{exp.get('model')}`")
                        cols[1].markdown(f"Feature engineering: `{exp.get('feature_engineering')}`")
                        cols[2].markdown(f"Hyperparameters: `{exp.get('hyperparams')}`")
                        cols[3].markdown(f"Score ({exp.get('metric')}): `{score}`")
                        if is_new_best:
                            st.markdown("⭐ **Best score so far**")
                        if exp.get("rationale"):
                            st.markdown(f"🧠 **Agent reasoning:** {exp.get('rationale')}")
                        if exp.get("analysis"):
                            st.caption(exp.get("analysis"))
                return running_best_score

            st.subheader("Iterations (this run)")
            if _new_experiments:
                _render_iteration_rows(_new_experiments, None)
            else:
                st.caption("No new iterations were run.")

            if _prior_experiments:
                with st.expander(f"Prior iterations from earlier runs ({len(_prior_experiments)})"):
                    st.caption(
                        "These rows are existing history for this objective, not part of the run above. "
                        "Check \"Start fresh\" in the sidebar to clear them before the next run."
                    )
                    _render_iteration_rows(_prior_experiments, None)

            # === 5. Model Parameters (model-specific) =======================
            if best and result.get("best_model_path") and os.path.exists(result["best_model_path"]):
                st.subheader("🔬 Model Parameters")
                if result.get("sandbox_requested"):
                    st.caption("Sandbox model files are not deserialized by the UI. Use the downloaded metadata or inspect the model inside Docker.")
                else:
                    _render_model_parameters(best, result["best_model_path"], feature_info_json)

            # === 6. Generated code viewer ====================================
            iterations_run = result.get("iterations_run") or 0
            if iterations_run and experiments:
                st.subheader("🧠 Generated Code")
                new_records = experiments[-iterations_run:]
                for local_idx, exp in enumerate(new_records, start=1):
                    code_path = Path(exp.get("code_path") or os.path.join(result["generated_dir"], f"iteration_{local_idx}.py"))
                    with st.expander(f"View Generated Code -- Iteration {exp.get('iteration')} ({exp.get('model')})"):
                        if code_path.exists():
                            st.code(code_path.read_text(encoding="utf-8"), language="python")
                        else:
                            st.caption(f"Generated script not found on disk: `{code_path}`")

            st.subheader("🧠 AGENT ANALYSIS")
            status_val = result.get("status")
            n_iters = result.get("iterations_run")
            best_model_name = best.get("model") if best else None
            if status_val == "success":
                best_iteration = best.get("iteration") if best else None
                st.write(f"Target achieved at iteration {best_iteration}. Best model: {best_model_name}. No further optimization required.")
            elif status_val == "optimized":
                st.write(f"Target was achieved and the run kept searching through all {n_iters} iterations. Best model: {best_model_name}.")
            elif status_val == "search_space_exhausted":
                st.write(f"Every candidate configuration was tried (over {n_iters} iterations) without reaching the target. Best model so far: {best_model_name}.")
            else:
                st.write(f"Max iterations ({n_iters}) reached without hitting the target. Best model so far: {best_model_name}.")

            if result_stop_mode == "optimize" and result.get("stagnation_detected"):
                st.warning(
                    "⚠️ STAGNATION DETECTED -- recent iterations show little to no score "
                    "improvement. Recommendation: try a different model family."
                )

            if status_val not in ("success", "optimized"):
                gap = target_gap_text(plan, best)
                note = diminishing_return_note(best, experiments, direction)
                if best or gap or note:
                    with st.container(border=True):
                        st.markdown("**⚠️ TARGET NOT REACHED**")
                        st.markdown(f"- Target: `{metric} {'<=' if direction == 'minimize' else '>='} {target_score}`")
                        st.markdown(f"- Best: `{metric} = {_fmt_score_val(best.get('score')) if best else 'n/a'}`")
                        if gap:
                            st.markdown(f"- Gap: `{gap}`")
                        if best:
                            st.markdown(
                                f"- Best model: `{best.get('model')}` "
                                f"(feature_engineering=`{best.get('feature_engineering')}`, "
                                f"hyperparams=`{best.get('hyperparams')}`)"
                            )
                        st.markdown(f"- Optimization status: {note or 'still improving -- no diminishing-returns signal yet.'}")

            if result.get("status") == "success":
                st.success(
                    f"✅ TARGET ACHIEVED -- best model **{best.get('model')}** "
                    f"scored **{best.get('score')}** ({best.get('metric')})"
                )
            elif result.get("status") == "optimized":
                st.success(
                    f"✅ TARGET ACHIEVED, OPTIMIZED FURTHER -- best model **{best.get('model')}** "
                    f"scored **{best.get('score')}** ({best.get('metric')}) across "
                    f"{result.get('iterations_run')} iterations"
                )
            elif result.get("status") == "search_space_exhausted":
                st.warning(f"⚠️ SEARCH SPACE EXHAUSTED -- {honest_summary_sentence(result)}")
            else:
                st.warning(f"⚠️ MAX ITERATIONS REACHED -- {honest_summary_sentence(result)}")

            report_path = result.get("report_path")
            experiments_path = result.get("run_experiments_path")
            best_model_path = result.get("best_model_path")

            dl_cols = st.columns(3)
            if report_path and os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    dl_cols[0].download_button(
                        "Download report.md", data=f.read(), file_name="report.md", mime="text/markdown"
                    )
            if experiments_path and os.path.exists(experiments_path):
                with open(experiments_path, "rb") as f:
                    dl_cols[1].download_button(
                        "Download experiments.json",
                        data=f.read(),
                        file_name="experiments.json",
                        mime="application/json",
                    )
            if best_model_path and os.path.exists(best_model_path):
                with open(best_model_path, "rb") as f:
                    dl_cols[2].download_button(
                        "Download best_model.joblib",
                        data=f.read(),
                        file_name="best_model.joblib",
                        mime="application/octet-stream",
                    )

            if models_dir:
                dl_cols2 = st.columns(3)
                for col, fname in zip(dl_cols2, ["best_model.json", "feature_info.json", "training_history.json"]):
                    fpath = os.path.join(models_dir, fname)
                    if os.path.exists(fpath):
                        with open(fpath, "rb") as f:
                            col.download_button(
                                f"Download {fname}", data=f.read(), file_name=fname, mime="application/json"
                            )
        except Exception as exc:  # noqa: BLE001 - surface any agent failure in the UI, not a crash
            st.error(str(exc))
else:
    st.write("Upload a CSV dataset, set an objective, and click **Run Agent** to start.")
