"""
app/run_scope.py

Caller: app/agent.py (run_agent), consumed by app/ralph.py callers.
No duplicate-purpose file exists for this; app/ralph.py's _objective_path
is now re-exported from here as the single source of truth for the
"objective.json sits beside experiments.json" convention.

Synthetic-only sample datasets are used throughout this project (no real
or sensitive data).

Instruction being implemented: "scope experiment storage per objective so
different datasets/tasks never share one history file, add optional
target vs optimize stop mode."

Derives a stable, filesystem-safe scope key from (dataset_path, task_type,
target_column, metric) so unrelated objectives (different dataset, task
type, target column, or metric) never collide in the same
experiments.json/objective.json pair, while reruns of the literal same
objective keep accumulating into the same scoped file.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return value or "x"


def scope_key(
    dataset_path: str,
    task_type: str,
    target_column: str | None,
    metric: str,
) -> str:
    """Short, stable, filesystem-safe slug identifying one objective.

    Resolves dataset_path to an absolute path first so the same dataset
    referenced by relative vs absolute path still scopes identically.
    """
    abs_dataset = str(Path(dataset_path).resolve())
    identity = "|".join(
        [abs_dataset, str(task_type), str(target_column), str(metric)]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]

    dataset_stem = _slug(Path(abs_dataset).stem)
    task_slug = _slug(str(task_type))
    return f"{dataset_stem}_{task_slug}_{digest}"


def default_experiments_path(
    workspace_dir: str,
    dataset_path: str,
    task_type: str,
    target_column: str | None,
    metric: str,
) -> str:
    # Each scope gets its own subdirectory (not just its own filename)
    # so that objective.json -- which lives alongside experiments.json
    # in the same directory (see default_objective_path) -- is also
    # correctly scoped per objective and never shared across objectives.
    key = scope_key(dataset_path, task_type, target_column, metric)
    return os.path.join(workspace_dir, "experiments", key, "experiments.json")


def default_models_dir(
    workspace_dir: str,
    dataset_path: str,
    task_type: str,
    target_column: str | None,
    metric: str,
) -> str:
    """Sibling of default_experiments_path for persisted trained models:
    <workspace_dir>/models/<scope_key>/ -- same scope-key convention so
    models from different objectives never collide, mirroring how
    workspace/experiments/<scope>/ is already scoped.
    """
    key = scope_key(dataset_path, task_type, target_column, metric)
    return os.path.join(workspace_dir, "models", key)


def default_objective_path(experiments_path: str) -> str:
    """Same convention app/ralph.py's _objective_path already uses:
    objective.json alongside experiments_path, in the same directory.
    """
    return os.path.join(os.path.dirname(experiments_path) or ".", "objective.json")
