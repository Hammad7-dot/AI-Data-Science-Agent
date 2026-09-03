import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tools.explainability import explain_model
from tools.model_export import ValidatedModel


def _fitted_export(estimator):
    frame = pd.DataFrame({
        "age": [20, 30, 40, 50, 60, 70],
        "city": ["a", "b", "a", "b", "a", "b"],
    })
    pipeline = Pipeline([
        ("preprocess", ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), ["age"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["city"]),
        ])),
        ("model", estimator),
    ]).fit(frame, [0, 0, 0, 1, 1, 1])
    return ValidatedModel(pipeline, frame, "target")


def test_linear_explanation_aligns_transformed_features():
    """Catches coefficients being paired with raw rather than transformed names."""
    exported_model = _fitted_export(LogisticRegression(max_iter=500))
    transformed_names = exported_model.named_steps["preprocess"].get_feature_names_out()

    explanation = explain_model(exported_model)

    assert explanation["available"] is True
    assert explanation["method"] == "coefficient"
    assert {row["feature"] for row in explanation["features"]} == set(transformed_names)
    assert explanation["features"] == sorted(
        explanation["features"], key=lambda row: row["importance"], reverse=True
    )


def test_unsupported_model_explains_unavailability():
    """Catches unsupported estimators returning a fabricated explanation."""
    exported_model = _fitted_export(KNeighborsClassifier(n_neighbors=1))

    explanation = explain_model(exported_model)

    assert explanation == {
        "available": False,
        "method": None,
        "features": [],
        "reason": "Model does not expose coefficients or feature importances.",
    }
