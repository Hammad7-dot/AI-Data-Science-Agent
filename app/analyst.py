"""
app/analyst.py

Phase 11 - Analyst.

STAND-IN FOR AN LLM CALL.

In a full implementation, this module would send the execution result,
evaluation dict, and plan to an LLM with a prompt like:

    "Given this evaluation result: {evaluation}, and this plan:
     {plan}, write a 1-3 sentence plain-language summary of what
     happened, and a specific recommendation for what to try next if
     the result failed to meet the target. Respond with JSON with keys
     'summary' and 'recommendation'."

and parse the LLM's JSON response into the dict below.

For Phase 0-9 MVP-derived logic (no API key available), we instead use
deterministic, rule-based text generation driven by the evaluation dict
and the raw parsed script output (which now carries "hyperparams" and
"feature_engineering", see tools/ml.py + app/coder.py). Swap the body
of interpret_results() for an LLM call when an API key is available --
the function signature and return schema should not need to change.
"""

from __future__ import annotations


def _fmt_score(score) -> str:
    if score is None:
        return "n/a"
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


def interpret_results(execution_result, evaluation: dict, plan: dict, memory=None) -> dict:
    """Produce a human-readable interpretation of one Ralph loop iteration.

    # TODO: replace with an LLM call using this prompt:
    #   "Given this evaluation result: {{evaluation}}, and this plan:
    #    {{plan}}, write a 1-3 sentence plain-language summary and a
    #    specific recommendation for what to try next. Respond with
    #    JSON with keys 'summary' and 'recommendation'."

    Returns:
        {
            "summary": str,
            "recommendation": str,
            "score": float | None,
            "metric": str | None,
            "model": str | None,
        }
    """
    status = evaluation.get("status")
    score = evaluation.get("score")
    metric = evaluation.get("metric")
    model = evaluation.get("model")
    raw = evaluation.get("raw") or {}
    hyperparams = raw.get("hyperparams")
    feature_engineering = raw.get("feature_engineering")
    target_score = plan.get("target_score")

    if status == "success":
        summary = f"{model} achieved {metric}={_fmt_score(score)}, meeting the target."
        if feature_engineering:
            summary += f" Feature engineering used: {feature_engineering}."
        recommendation = "Target met -- no further iterations needed."
        return {
            "summary": summary,
            "recommendation": recommendation,
            "score": score,
            "metric": metric,
            "model": model,
        }

    if score is None:
        reason = evaluation.get("reason") or "the script did not produce a usable result"
        summary = f"Iteration failed: {reason}."
        recommendation = "Fix the generated script's execution error before trying new hyperparameters."
        return {
            "summary": summary,
            "recommendation": recommendation,
            "score": score,
            "metric": metric,
            "model": model,
        }

    summary = f"{model} scored {metric}={_fmt_score(score)}"
    if target_score is not None:
        summary += f", short of the {target_score} target."
    else:
        summary += "."
    if feature_engineering:
        summary += f" (feature engineering: {feature_engineering}, hyperparams: {hyperparams})."

    if memory is not None and model:
        prior = [r for r in memory.history_for_model(model) if r.get("score") is not None]
        if len(prior) >= 2:
            trend_scores = [r["score"] for r in prior]
            summary += f" Trend for {model}: {_fmt_score(trend_scores[0])} -> {_fmt_score(trend_scores[-1])} over {len(trend_scores)} runs."

    # Simple rule-based "what to try next" logic.
    suggestions = []
    if feature_engineering == "basic":
        suggestions.append("switch to the scaled/feature-selected preprocessing pipeline")
    elif feature_engineering == "pipeline":
        suggestions.append("try a different feature-selection width or disable scaling")

    if model in ("logistic_regression", "linear_regression", "ridge"):
        suggestions.append("try a tree-based model (random_forest or gradient_boosting) for more capacity")
    elif model in ("random_forest", "random_forest_regressor"):
        suggestions.append("try gradient boosting with a lower learning rate")
    elif model in ("gradient_boosting", "gradient_boosting_regressor"):
        suggestions.append("try more estimators or a lower learning rate")
    else:
        suggestions.append("try a different candidate model")

    recommendation = "Next: " + "; ".join(suggestions) + "."

    return {
        "summary": summary,
        "recommendation": recommendation,
        "score": score,
        "metric": metric,
        "model": model,
    }
