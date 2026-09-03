import json
import contextlib
import io

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tools.model_export import ValidatedModel
from app.coder import generate_code


def fitted_model(frame):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), ["age"]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["city"]),
        ])),
        ("model", LogisticRegression(max_iter=500)),
    ]).fit(frame, [0, 1, 0])


def test_reloaded_validated_model_accepts_reordered_columns(tmp_path):
    """Catches exports that pass caller column order directly to sklearn."""
    frame = pd.DataFrame({"age": [20, 30, 40], "city": ["a", "b", "a"]})
    export = ValidatedModel(fitted_model(frame), frame, "target")
    path = tmp_path / "model.joblib"
    joblib.dump(export, path)

    reloaded = joblib.load(path)
    predictions = reloaded.predict(frame[["city", "age"]])

    assert len(predictions) == 3
    assert isinstance(reloaded.named_steps["model"], LogisticRegression)


@pytest.mark.parametrize("frame, phrases", [
    (pd.DataFrame({"age": [20]}), ["missing", "city"]),
    (pd.DataFrame({"age": [20], "city": ["a"], "noise": [1]}), ["unexpected", "noise"]),
    (pd.DataFrame({"age": ["old"], "city": ["a"]}), ["incompatible", "age", "numeric"]),
])
def test_schema_errors_are_actionable(frame, phrases):
    """Catches missing, extra, or wrong-family raw inputs reaching sklearn."""
    training_frame = pd.DataFrame({"age": [20, 30, 40], "city": ["a", "b", "a"]})
    export = ValidatedModel(fitted_model(training_frame), training_frame, "target")

    with pytest.raises(ValueError) as error:
        export.predict(frame)

    assert all(phrase in str(error.value).lower() for phrase in phrases)


def test_schema_errors_reject_duplicate_feature_columns():
    """Catches inputs containing a required feature more than once."""
    training_frame = pd.DataFrame({"age": [20, 30, 40], "city": ["a", "b", "a"]})
    export = ValidatedModel(fitted_model(training_frame), training_frame, "target")
    duplicate_city = pd.DataFrame([[20, "a", "a"]], columns=["age", "city", "city"])

    with pytest.raises(ValueError) as error:
        export.predict(duplicate_city)

    assert "unexpected" in str(error.value).lower()
    assert "city" in str(error.value).lower()


@pytest.mark.parametrize("training, supplied, expected_family", [
    (
        pd.DataFrame({"enabled": pd.Series([True, False], dtype="boolean")}),
        pd.DataFrame({"enabled": [1, 0]}),
        "boolean",
    ),
    (
        pd.DataFrame({"observed_at": pd.to_datetime(["2026-01-01", "2026-01-02"])}),
        pd.DataFrame({"observed_at": ["2026-01-01"]}),
        "datetime",
    ),
])
def test_schema_errors_name_boolean_and_datetime_families(training, supplied, expected_family):
    """Catches boolean or datetime inputs being classified as generic numeric/text."""
    export = ValidatedModel(DummyRegressor(), training, "target")

    with pytest.raises(ValueError) as error:
        export.predict(supplied)

    assert "incompatible" in str(error.value).lower()
    assert expected_family in str(error.value).lower()


def test_predictions_require_a_pandas_dataframe():
    """Catches array inputs that bypass named raw-column validation."""
    frame = pd.DataFrame({"age": [20, 30, 40], "city": ["a", "b", "a"]})
    export = ValidatedModel(fitted_model(frame), frame, "target")

    with pytest.raises(ValueError, match="pandas DataFrame"):
        export.predict([[20, "a"]])


def test_schema_metadata_is_json_safe_and_describes_raw_inputs():
    """Catches exports that lose their raw schema or return non-JSON pandas values."""
    frame = pd.DataFrame({"age": [20, 30, 40], "city": ["a", "b", "a"]})
    export = ValidatedModel(fitted_model(frame), frame, "target")

    metadata = export.schema_metadata()

    assert json.loads(json.dumps(metadata)) == {
        "schema_version": 1,
        "raw_columns": ["age", "city"],
        "dtype_families": {"age": "numeric", "city": "categorical"},
        "target_name": "target",
    }


def test_probability_and_decision_methods_validate_then_delegate():
    """Catches validation being applied only to predict instead of classifier APIs."""
    frame = pd.DataFrame({"age": [20, 30, 40], "city": ["a", "b", "a"]})
    export = ValidatedModel(fitted_model(frame), frame, "target")
    reordered = frame[["city", "age"]]

    assert export.predict_proba(reordered).shape == (3, 2)
    assert export.decision_function(reordered).shape == (3,)


def test_unknown_attributes_delegate_to_the_wrapped_model():
    """Catches compatibility breaks for fitted sklearn attributes such as classes_."""
    frame = pd.DataFrame({"age": [20, 30, 40], "city": ["a", "b", "a"]})
    export = ValidatedModel(fitted_model(frame), frame, "target")

    assert export.classes_.tolist() == [0, 1]


def test_generated_training_saves_a_validated_model(tmp_path):
    """Catches generated training scripts that serialize the bare pipeline."""
    frame = pd.DataFrame({
        "age": range(60),
        "city": ["a", "b", "c"] * 20,
        "target": [0, 1] * 30,
    })
    dataset_path = tmp_path / "data.csv"
    frame.to_csv(dataset_path, index=False)
    plan = {
        "task_type": "classification",
        "target": "target",
        "metric": "f1",
        "candidate_models": ["random_forest"],
    }
    code = generate_code(
        str(dataset_path), plan, 1, model_name="random_forest",
        hyperparams={"n_estimators": 5}, models_dir=str(tmp_path / "models"),
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "generated_export.py", "exec"), {})
    result = json.loads(output.getvalue().strip())

    assert isinstance(joblib.load(result["model_path"]), ValidatedModel)
