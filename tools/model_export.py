"""Serializable model exports that retain their raw prediction schema."""

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


def dtype_family(dtype):
    if is_bool_dtype(dtype):
        return "boolean"
    if is_numeric_dtype(dtype):
        return "numeric"
    if is_datetime64_any_dtype(dtype):
        return "datetime"
    return "categorical"


def format_schema_error(missing, unexpected, incompatible, dtype_families):
    parts = []
    if missing:
        parts.append("missing columns: " + ", ".join(missing))
    if unexpected:
        parts.append("unexpected columns: " + ", ".join(unexpected))
    if incompatible:
        details = ", ".join(
            f"{name} (expected {dtype_families[name]})" for name in incompatible
        )
        parts.append("incompatible columns: " + details)
    return "Prediction input schema mismatch: " + "; ".join(parts)


class ValidatedModel:
    def __init__(self, model, feature_frame, target_name=None):
        self.model = model
        self.raw_columns = list(feature_frame.columns)
        self.dtype_families = {
            name: dtype_family(feature_frame[name].dtype) for name in self.raw_columns
        }
        self.target_name = target_name

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "model"), name)

    @property
    def named_steps(self):
        return self.model.named_steps

    def schema_metadata(self):
        return {
            "schema_version": 1,
            "raw_columns": self.raw_columns,
            "dtype_families": self.dtype_families,
            "target_name": self.target_name,
        }

    def _validated(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Prediction input must be a pandas DataFrame.")

        column_counts = X.columns.value_counts()
        missing = [name for name in self.raw_columns if name not in X.columns]
        unexpected = [
            name for name in X.columns
            if name not in self.raw_columns or column_counts[name] > 1
        ]
        incompatible = [
            name for name in self.raw_columns
            if name in X and column_counts[name] == 1
            and dtype_family(X[name].dtype) != self.dtype_families[name]
        ]
        if missing or unexpected or incompatible:
            raise ValueError(
                format_schema_error(missing, unexpected, incompatible, self.dtype_families)
            )
        return X[self.raw_columns]

    def predict(self, X):
        return self.model.predict(self._validated(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._validated(X))

    def decision_function(self, X):
        return self.model.decision_function(self._validated(X))
