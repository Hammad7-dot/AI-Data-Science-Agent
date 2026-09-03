"""Native feature-importance summaries for exported sklearn pipelines."""

import numpy as np

from tools.model_export import ValidatedModel


_UNSUPPORTED_REASON = "Model does not expose coefficients or feature importances."


def _unavailable(reason):
    return {
        "available": False,
        "method": None,
        "features": [],
        "reason": reason,
    }


def _features(names, importances, method, coefficients=None):
    if len(names) != len(importances):
        return _unavailable("Transformed feature names and importances differ in length.")

    rows = []
    for index, (name, importance) in enumerate(zip(names, importances)):
        row = {"feature": str(name), "importance": float(importance)}
        if coefficients is not None:
            row["coefficient"] = float(coefficients[index])
        rows.append(row)
    return {
        "available": True,
        "method": method,
        "features": sorted(rows, key=lambda row: row["importance"], reverse=True),
        "reason": None,
    }


def explain_model(exported_model) -> dict:
    """Return JSON-safe native importances for a fitted exported pipeline."""
    pipeline = exported_model.model if isinstance(exported_model, ValidatedModel) else exported_model
    try:
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        estimator = pipeline.named_steps["model"]
    except (AttributeError, KeyError):
        return _unavailable(_UNSUPPORTED_REASON)

    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_)
        if coefficients.ndim == 1:
            return _features(names, np.abs(coefficients), "coefficient", coefficients=coefficients)
        if coefficients.ndim == 2 and coefficients.shape[0] == 1:
            vector = coefficients[0]
            return _features(names, np.abs(vector), "coefficient", coefficients=vector)
        if coefficients.ndim == 2:
            return _features(names, np.mean(np.abs(coefficients), axis=0), "coefficient")
        return _unavailable(_UNSUPPORTED_REASON)

    if hasattr(estimator, "feature_importances_"):
        importances = np.maximum(0, np.asarray(estimator.feature_importances_))
        return _features(names, importances, "feature_importance")

    return _unavailable(_UNSUPPORTED_REASON)
