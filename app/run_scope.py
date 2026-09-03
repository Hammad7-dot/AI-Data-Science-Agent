"""Content-based experiment identities, isolated uploads, and cross-process history locks."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from tools.validation import VALIDATION_VERSION

EXPERIMENT_CONFIG_VERSION = "pipeline-search-v2"


def store_upload(workspace_dir, filename, data):
    name = Path(filename.replace("\\", "/")).name
    if name in ("", ".", ".."):
        raise ValueError("Upload must have a filename")
    destination = Path(workspace_dir).resolve() / "uploads" / uuid.uuid4().hex / name
    destination.parent.mkdir(parents=True)
    destination.write_bytes(data)
    return str(destination)


def dataset_digest(dataset_path: str) -> str:
    digest = hashlib.sha256()
    with open(dataset_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def history_lock(experiments_path):
    """Fail promptly on a concurrent writer; the OS releases locks on process exit."""
    lock_path = Path(str(Path(experiments_path).resolve()) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("An active run is using this experiment history; retry when it finishes.") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def lock_experiment_history(function):
    signature = inspect.signature(function)
    @wraps(function)
    def locked(*args, **kwargs):
        path = signature.bind(*args, **kwargs).arguments["experiments_path"]
        with history_lock(path):
            return function(*args, **kwargs)
    return locked


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return value or "x"


def scope_key(
    dataset_path: str,
    task_type: str,
    target_column: str | None,
    metric: str,
    use_docker: bool = False,
) -> str:
    """Short, stable, filesystem-safe slug identifying one objective.

    Dataset bytes, task, target, metric, protocol versions and execution mode
    identify compatible experiments; file names and locations do not.
    """
    identity = json.dumps([dataset_digest(dataset_path), task_type, target_column,
                           metric, VALIDATION_VERSION, EXPERIMENT_CONFIG_VERSION, use_docker])
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    task_slug = _slug(str(task_type))
    return f"{task_slug}_{digest}"


def default_experiments_path(
    workspace_dir: str,
    dataset_path: str,
    task_type: str,
    target_column: str | None,
    metric: str,
    use_docker: bool = False,
) -> str:
    # Each scope gets its own subdirectory (not just its own filename)
    # so that objective.json -- which lives alongside experiments.json
    # in the same directory (see default_objective_path) -- is also
    # correctly scoped per objective and never shared across objectives.
    key = scope_key(dataset_path, task_type, target_column, metric, use_docker=use_docker)
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
    path = Path(experiments_path)
    name = "objective.json" if path.name == "experiments.json" else path.name + ".objective.json"
    return str(path.with_name(name))
