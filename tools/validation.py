"""One reproducible development/holdout split, shared by search and final evaluation."""
import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

VALIDATION_VERSION = "oof-v1"


def profile_for_training(path, target_hint=None):
    """Use development-only statistics for encoding choices and target thresholds."""
    from tools.dataset import profile_frame
    frame = pd.read_csv(path)
    schema = profile_frame(frame, path, target_hint)
    target_info = schema.get("target")
    if not target_info:
        return schema
    target, task = target_info["name"], target_info["suggested_task_type"]
    labeled = frame.dropna(subset=[target])
    X_dev, _, y_dev, _ = split_development_holdout(labeled.drop(columns=target), labeled[target], task)
    development = X_dev.assign(**{target: y_dev})
    profile = profile_frame(development, path, target, task_type_hint=task)
    profile["profile_rows"] = len(development)
    profile["rows"] = len(frame)
    profile["statistics_partition"] = "development"
    return profile


def split_development_holdout(X, y, task_type):
    if len(y) < 10:
        raise ValueError("At least 10 labeled rows are required for validation and holdout evaluation.")
    return train_test_split(X, y, test_size=0.2, random_state=42,
                            stratify=y if task_type == "classification" else None)


def validation_folds(y, task_type):
    if task_type == "classification":
        folds = min(3, int(y.value_counts().min()))
        if y.nunique() < 2 or folds < 2:
            raise ValueError("Cross-validation requires at least two examples per class in development data.")
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return KFold(n_splits=3, shuffle=True, random_state=42)


def evaluate_holdout(model_path, dataset_path, plan):
    """Evaluate only the selected model, after search; never feed this score back."""
    frame = pd.read_csv(dataset_path).dropna(subset=[plan["target"]])
    X = frame.drop(columns=plan["target"])
    excluded_columns = (
        (plan.get("dropped_high_cardinality_columns") or [])
        + (plan.get("leakage_dropped_columns") or [])
    )
    X = X.drop(columns=[column for column in excluded_columns if column in X.columns])
    _, X_holdout, _, y_holdout = split_development_holdout(
        X, frame[plan["target"]], plan["task_type"])
    predictions = joblib.load(model_path).predict(X_holdout)
    if plan["task_type"] == "classification":
        metrics = {"accuracy": accuracy_score(y_holdout, predictions),
                   "f1": f1_score(y_holdout, predictions, average="weighted", zero_division=0),
                   "precision": precision_score(y_holdout, predictions, average="weighted", zero_division=0),
                   "recall": recall_score(y_holdout, predictions, average="weighted", zero_division=0)}
    else:
        metrics = {"r2": r2_score(y_holdout, predictions),
                   "rmse": mean_squared_error(y_holdout, predictions) ** 0.5,
                   "mae": mean_absolute_error(y_holdout, predictions)}
    return {"metric": plan["metric"], "score": float(metrics[plan["metric"]]),
            "samples": len(y_holdout), "metrics": {k: float(v) for k, v in metrics.items()}}
