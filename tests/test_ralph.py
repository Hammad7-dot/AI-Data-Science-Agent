import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from app.ralph import run_ralph_loop, _experiment_identity
from app.experiment_memory import ExperimentMemory
from app.experiment_strategist import choose_next_experiment
from app.agent import run_agent
from app.run_scope import default_experiments_path, scope_key
from tools.validation import VALIDATION_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = str(PROJECT_ROOT / "workspace" / "datasets" / "sample_churn.csv")
REGRESSION_DATASET_PATH = str(PROJECT_ROOT / "workspace" / "datasets" / "sample_regression.csv")


def test_duplicate_prevention_persists_across_runs(tmp_path):
    experiments_path = str(tmp_path / "experiments.json")
    workdir = str(tmp_path / "generated")

    # Seed experiments.json with fake prior "failed" records.
    seeded = [
        {
            "iteration": 1,
            "model": "random_forest",
            "metric": "f1",
            "score": 0.1,
            "status": "fail",
            "hyperparams": {"n_estimators": 50, "max_depth": None},
            "feature_engineering": "basic",
        },
        {
            "iteration": 2,
            "model": "random_forest",
            "metric": "f1",
            "score": 0.1,
            "status": "fail",
            "hyperparams": {"n_estimators": 50, "max_depth": None},
            "feature_engineering": "pipeline",
        },
    ]
    with open(experiments_path, "w", encoding="utf-8") as f:
        json.dump([dict(record, validation_version=VALIDATION_VERSION, scope_key=scope_key(DATASET_PATH, "classification", "churn", "f1")) for record in seeded], f)

    seeded_identities = {_experiment_identity(r) for r in seeded}

    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.999,  # unreachable, forces many iterations
        metric="f1",
        max_iterations=3,
        workdir=workdir,
        experiments_path=experiments_path,
    )

    new_records = result["all_experiments"][len(seeded):]
    for rec in new_records:
        assert _experiment_identity(rec) not in seeded_identities

    assert result["status"] in ("fail", "max_iterations_reached", "search_space_exhausted", "success")


def test_search_space_exhausted(tmp_path, monkeypatch):
    experiments_path = str(tmp_path / "experiments.json")
    workdir = str(tmp_path / "generated")

    # Shrink both the candidate model list and the hyperparameter grid so
    # the search space is tiny and gets exhausted within a couple of
    # iterations.
    import tools.ml as ml_module
    import app.planner as planner_module

    def tiny_grid(model_name):
        return {"n_estimators": [50]} if model_name == "random_forest" else {}

    def tiny_models(task_type):
        return ["random_forest"] if task_type == "classification" else []

    monkeypatch.setattr(ml_module, "hyperparameter_grid", tiny_grid)
    monkeypatch.setattr(planner_module, "list_models_for_task", tiny_models)

    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.999,  # unreachable
        metric="f1",
        max_iterations=50,
        workdir=workdir,
        experiments_path=experiments_path,
    )

    # search space: 1 model x 2 feature-engineering modes x (1 static hp combo
    # + a small directed-expansion neighborhood around it) -- larger than the
    # old pure-static-grid count now that expand_hyperparams() contributes
    # extra novel candidates before the space is considered exhausted. A bit
    # larger still now that simple_trend() flags a flattened-out key as
    # "diminishing" (stopping expansion on that key) rather than continuing
    # to offer ever-larger candidates on it -- the strategist correctly
    # falls through to the static grid for a couple more genuinely-untried
    # combos before the space is truly exhausted.
    assert result["status"] == "search_space_exhausted"
    assert result["iterations_run"] <= 15


def test_exploration_phase_visits_each_model_once(tmp_path, monkeypatch):
    """First N iterations (N = number of candidate models) should each
    exercise a *different* model exactly once -- the baseline
    exploration phase -- not a static grid order.
    """
    experiments_path = str(tmp_path / "experiments.json")
    workdir = str(tmp_path / "generated")

    import tools.ml as ml_module
    import app.planner as planner_module

    def tiny_grid(model_name):
        return {"n_estimators": [50, 100]}

    def tiny_models(task_type):
        return ["random_forest", "gradient_boosting", "knn"] if task_type == "classification" else []

    monkeypatch.setattr(ml_module, "hyperparameter_grid", tiny_grid)
    monkeypatch.setattr(planner_module, "list_models_for_task", tiny_models)

    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.999,  # unreachable
        metric="f1",
        max_iterations=3,
        workdir=workdir,
        experiments_path=experiments_path,
    )

    models_visited = [r["model"] for r in result["all_experiments"][:3]]
    assert len(set(models_visited)) == 3, models_visited


def test_exploitation_phase_refines_best_model(tmp_path, monkeypatch):
    """After every model has a baseline, subsequent iterations should
    stay on the best-performing model, varying hyperparams/feature
    engineering, rather than cycling to other models.
    """
    experiments_path = str(tmp_path / "experiments.json")
    workdir = str(tmp_path / "generated")

    import tools.ml as ml_module
    import app.planner as planner_module

    def tiny_grid(model_name):
        return {"n_estimators": [50, 100, 200]}

    def tiny_models(task_type):
        return ["random_forest", "knn"] if task_type == "classification" else []

    monkeypatch.setattr(ml_module, "hyperparameter_grid", tiny_grid)
    monkeypatch.setattr(planner_module, "list_models_for_task", tiny_models)

    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.999,  # unreachable, forces full exploration+exploitation
        metric="f1",
        max_iterations=5,
        workdir=workdir,
        experiments_path=experiments_path,
    )

    experiments = result["all_experiments"]
    # first 2 iterations = exploration (one per model)
    assert len({e["model"] for e in experiments[:2]}) == 2

    # determine which model actually won the baseline round
    best_model = max(
        (e["model"] for e in experiments[:2]),
        key=lambda m: next(e["score"] for e in experiments[:2] if e["model"] == m and e["score"] is not None),
    )

    later = experiments[2:]
    assert later, "expected at least one exploitation iteration"
    for exp in later:
        assert exp["model"] == best_model
    # hyperparams should differ across the exploitation iterations
    identities_seen = {_experiment_identity(e) for e in later}
    assert len(identities_seen) == len(later)


def test_no_duplicate_identity_within_run(tmp_path, monkeypatch):
    experiments_path = str(tmp_path / "experiments.json")
    workdir = str(tmp_path / "generated")

    import tools.ml as ml_module
    import app.planner as planner_module

    def tiny_grid(model_name):
        return {"n_estimators": [50, 100]}

    def tiny_models(task_type):
        return ["random_forest", "knn"] if task_type == "classification" else []

    monkeypatch.setattr(ml_module, "hyperparameter_grid", tiny_grid)
    monkeypatch.setattr(planner_module, "list_models_for_task", tiny_models)

    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.999,
        metric="f1",
        max_iterations=20,
        workdir=workdir,
        experiments_path=experiments_path,
    )

    identities = [_experiment_identity(r) for r in result["all_experiments"]]
    assert len(identities) == len(set(identities))


def test_best_experiment_is_best_across_whole_run_not_last_iteration(tmp_path, monkeypatch):
    """best_experiment in run_ralph_loop's return value must be the
    single best-scoring record across the WHOLE run, even when the
    best occurred mid-run and later iterations scored worse.
    """
    experiments_path = str(tmp_path / "experiments.json")
    workdir = str(tmp_path / "generated")

    import app.ralph as ralph_module

    # Fake out generate_code/execute/evaluate so we control scores
    # directly and deterministically -- iteration 3 is the best,
    # iterations 4-6 are worse.
    scores_by_iteration = {1: 0.70, 2: 0.75, 3: 0.90, 4: 0.60, 5: 0.65, 6: 0.55}

    def fake_generate_code(dataset_path, plan, idx, previous_feedback=None, model_name=None, feature_engineering=None, hyperparams=None, models_dir=None):
        return "# fake code"

    def fake_execute(code, workdir, iteration, use_docker=False):
        class FakeResult:
            success = True
            stdout = ""
            stderr = ""

        return FakeResult()

    def fake_evaluate(execution_result, plan, target_score):
        iteration = fake_evaluate.counter
        fake_evaluate.counter += 1
        score = scores_by_iteration[iteration]
        return {
            "status": "fail",  # never "success" -- force max_iterations_reached
            "score": score,
            "metric": "f1",
            "model": "random_forest",
            "raw": {"hyperparams": {"n_estimators": iteration * 10}, "feature_engineering": "basic"},
            "reason": None,
        }

    fake_evaluate.counter = 1

    def fake_interpret(execution_result, eval_result, plan, memory=None):
        return {"summary": "n/a", "recommendation": "n/a"}

    monkeypatch.setattr(ralph_module, "generate_code", fake_generate_code)
    monkeypatch.setattr(ralph_module, "execute", fake_execute)
    monkeypatch.setattr(ralph_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(ralph_module, "interpret_results", fake_interpret)

    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.999,  # unreachable -- forces all 6 iterations
        metric="f1",
        max_iterations=6,
        workdir=workdir,
        experiments_path=experiments_path,
    )

    assert result["status"] == "max_iterations_reached"
    assert result["best_experiment"]["iteration"] == 3
    assert result["best_experiment"]["score"] == 0.90


def test_best_experiment_seeded_from_prior_run_history(tmp_path, monkeypatch):
    """A second run against the same experiments_path must compare its
    own result against the TRUE best-so-far from prior runs, not start
    best-tracking from None. Simulates: run A produces a high score,
    run B (fresh call to run_ralph_loop, same experiments_path) produces
    a lower score that still meets its own target -- the returned
    best_experiment must still be run A's superior record.
    """
    experiments_path = str(tmp_path / "experiments.json")
    workdir = str(tmp_path / "generated")

    import app.ralph as ralph_module

    def fake_generate_code(dataset_path, plan, idx, previous_feedback=None, model_name=None, feature_engineering=None, hyperparams=None, models_dir=None):
        return "# fake code"

    def fake_execute(code, workdir, iteration, use_docker=False):
        class FakeResult:
            success = True
            stdout = ""
            stderr = ""

        return FakeResult()

    def make_fake_evaluate(score, model, status="success"):
        def fake_evaluate(execution_result, plan, target_score):
            return {
                "status": status,
                "score": score,
                "metric": "f1",
                "model": model,
                "raw": {"hyperparams": {}, "feature_engineering": "basic"},
                "reason": None,
            }

        return fake_evaluate

    def fake_interpret(execution_result, eval_result, plan, memory=None):
        return {"summary": "n/a", "recommendation": "n/a"}

    monkeypatch.setattr(ralph_module, "generate_code", fake_generate_code)
    monkeypatch.setattr(ralph_module, "execute", fake_execute)
    monkeypatch.setattr(ralph_module, "interpret_results", fake_interpret)

    # Run A: linear_regression-equivalent scores high (0.90) and succeeds.
    monkeypatch.setattr(ralph_module, "evaluate", make_fake_evaluate(0.90, "model_a"))
    result_a = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.80,
        metric="f1",
        max_iterations=1,
        workdir=workdir,
        experiments_path=experiments_path,
    )
    assert result_a["status"] == "success"
    assert result_a["best_experiment"]["score"] == 0.90

    # Run B: fresh call, same experiments_path. New model scores lower
    # (0.85) but still clears its own target on the first iteration.
    monkeypatch.setattr(ralph_module, "evaluate", make_fake_evaluate(0.85, "model_b"))
    result_b = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.80,
        metric="f1",
        max_iterations=1,
        workdir=workdir,
        experiments_path=experiments_path,
    )
    assert result_b["status"] == "success"
    # The genuinely-better run A record must still win, not be
    # discarded because run B's best-tracking started at None.
    assert result_b["best_experiment"]["model"] == "model_a"
    assert result_b["best_experiment"]["score"] == 0.90


def test_iteration_numbers_continuous_across_runs(tmp_path, monkeypatch):
    """Iteration numbers recorded in experiments.json must continue
    across separate run_ralph_loop calls against the same
    experiments_path, not restart at 1 each time.
    """
    experiments_path = str(tmp_path / "experiments.json")
    workdir = str(tmp_path / "generated")

    import app.ralph as ralph_module

    def fake_generate_code(dataset_path, plan, idx, previous_feedback=None, model_name=None, feature_engineering=None, hyperparams=None, models_dir=None):
        return "# fake code"

    def fake_execute(code, workdir, iteration, use_docker=False):
        class FakeResult:
            success = True
            stdout = ""
            stderr = ""

        return FakeResult()

    def make_fake_evaluate(score, model):
        def fake_evaluate(execution_result, plan, target_score):
            return {
                "status": "success",
                "score": score,
                "metric": "f1",
                "model": model,
                "raw": {"hyperparams": {}, "feature_engineering": "basic"},
                "reason": None,
            }

        return fake_evaluate

    def fake_interpret(execution_result, eval_result, plan, memory=None):
        return {"summary": "n/a", "recommendation": "n/a"}

    monkeypatch.setattr(ralph_module, "generate_code", fake_generate_code)
    monkeypatch.setattr(ralph_module, "execute", fake_execute)
    monkeypatch.setattr(ralph_module, "interpret_results", fake_interpret)

    monkeypatch.setattr(ralph_module, "evaluate", make_fake_evaluate(0.90, "model_a"))
    result_a = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.80,
        metric="f1",
        max_iterations=1,
        workdir=workdir,
        experiments_path=experiments_path,
    )
    assert [r["iteration"] for r in result_a["all_experiments"]] == [1]

    monkeypatch.setattr(ralph_module, "evaluate", make_fake_evaluate(0.85, "model_b"))
    result_b = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.80,
        metric="f1",
        max_iterations=1,
        workdir=workdir,
        experiments_path=experiments_path,
    )
    new_records = result_b["all_experiments"][len(result_a["all_experiments"]):]
    assert [r["iteration"] for r in new_records] == [2]


def test_stagnation_changes_next_choice():
    """Hand-built fake history: random_forest's score plateaus for
    several iterations. Stagnation detection should switch the next
    choice's feature_engineering (or model) away from the naive
    "just try the next hyperparameter combo on the same model, same
    feature_engineering" sequence.
    """
    plan = {
        "task_type": "classification",
        "metric": "f1",
        "direction": "maximize",
        "candidate_models": ["random_forest", "gradient_boosting"],
    }

    history = [
        {"iteration": 1, "model": "gradient_boosting", "score": 0.80, "hyperparams": {}, "feature_engineering": "basic"},
        {"iteration": 2, "model": "random_forest", "score": 0.827, "hyperparams": {"n_estimators": 50}, "feature_engineering": "basic"},
        {"iteration": 3, "model": "random_forest", "score": 0.853, "hyperparams": {"n_estimators": 100}, "feature_engineering": "basic"},
        {"iteration": 4, "model": "random_forest", "score": 0.8400, "hyperparams": {"n_estimators": 200}, "feature_engineering": "basic"},
        {"iteration": 5, "model": "random_forest", "score": 0.8401, "hyperparams": {"n_estimators": 250}, "feature_engineering": "basic"},
        {"iteration": 6, "model": "random_forest", "score": 0.8400, "hyperparams": {"n_estimators": 300}, "feature_engineering": "basic"},
    ]
    memory = ExperimentMemory(history)
    assert memory.is_stagnant(window=3, min_improvement=0.01, direction="maximize") is True
    assert memory.is_stagnating() is True

    choice = choose_next_experiment(plan, memory, iteration=5)
    assert choice is not None
    # Naive continuation would keep model=random_forest, feature_engineering=basic.
    # Stagnation handling must change at least one of those.
    assert not (choice["model_name"] == "random_forest" and choice["feature_engineering"] == "basic")


def test_different_objectives_get_scoped_separate_experiment_files(tmp_path):
    """Classification against sample_churn.csv and regression against
    sample_regression.csv must never share an experiments.json -- each
    objective (dataset + task_type + target + metric) is scoped to its
    own file, so records never cross-contaminate.
    """
    workspace_dir = str(tmp_path / "workspace")

    result_cls = run_agent(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        max_iterations=2,
        workspace_dir=workspace_dir,
    )
    result_reg = run_agent(
        dataset_path=REGRESSION_DATASET_PATH,
        objective="Predict the target",
        max_iterations=2,
        workspace_dir=workspace_dir,
    )

    cls_path = result_cls["experiments_path"]
    reg_path = result_reg["experiments_path"]
    assert cls_path != reg_path

    for exp in result_cls["all_experiments"]:
        assert exp.get("metric") == "f1"
    for exp in result_reg["all_experiments"]:
        assert exp.get("metric") == "r2"

    experiments_dir = os.path.join(workspace_dir, "experiments")
    scope_dirs = [
        d for d in os.listdir(experiments_dir)
        if os.path.isdir(os.path.join(experiments_dir, d))
    ]
    assert len(scope_dirs) == 2


def test_rerun_same_objective_reuses_same_scoped_file(tmp_path):
    """Two runs against the exact same dataset+objective must land in
    the same scoped experiments.json and keep accumulating/comparing
    against each other (continuous iteration numbering, best-tracking),
    even though the path is now computed via auto-scoping.
    """
    workspace_dir = str(tmp_path / "workspace")

    result_1 = run_agent(
        dataset_path=REGRESSION_DATASET_PATH,
        objective="Predict the target",
        max_iterations=1,
        workspace_dir=workspace_dir,
    )
    result_2 = run_agent(
        dataset_path=REGRESSION_DATASET_PATH,
        objective="Predict the target",
        max_iterations=1,
        workspace_dir=workspace_dir,
    )

    assert result_1["experiments_path"] == result_2["experiments_path"]
    assert len(result_2["all_experiments"]) > len(result_1["all_experiments"])
    iterations = [r["iteration"] for r in result_2["all_experiments"]]
    assert iterations == sorted(iterations)


def test_scope_key_stable_for_relative_vs_absolute_dataset_path():
    abs_path = str(Path(DATASET_PATH).resolve())
    rel_path = os.path.relpath(DATASET_PATH)
    key_abs = scope_key(abs_path, "classification", "churn", "f1")
    key_rel = scope_key(rel_path, "classification", "churn", "f1")
    assert key_abs == key_rel


def test_stop_mode_optimize_runs_past_first_success(tmp_path, monkeypatch):
    """With stop_mode='optimize' the loop must not break on the first
    success -- it should keep going until max_iterations (or search
    space exhaustion), and best_experiment must reflect the true best
    across all iterations run, while status still reflects that the
    target was achieved.
    """
    experiments_path = str(tmp_path / "experiments.json")
    workdir = str(tmp_path / "generated")

    import app.ralph as ralph_module

    scores_by_iteration = {1: 0.85, 2: 0.80, 3: 0.95}

    def fake_generate_code(dataset_path, plan, idx, previous_feedback=None, model_name=None, feature_engineering=None, hyperparams=None, models_dir=None):
        return "# fake code"

    def fake_execute(code, workdir, iteration, use_docker=False):
        class FakeResult:
            success = True
            stdout = ""
            stderr = ""

        return FakeResult()

    def fake_evaluate(execution_result, plan, target_score):
        iteration = fake_evaluate.counter
        fake_evaluate.counter += 1
        score = scores_by_iteration[iteration]
        return {
            "status": "success" if score >= 0.80 else "fail",
            "score": score,
            "metric": "f1",
            "model": f"model_{iteration}",
            "raw": {"hyperparams": {}, "feature_engineering": "basic"},
            "reason": None,
        }

    fake_evaluate.counter = 1

    def fake_interpret(execution_result, eval_result, plan, memory=None):
        return {"summary": "n/a", "recommendation": "n/a"}

    monkeypatch.setattr(ralph_module, "generate_code", fake_generate_code)
    monkeypatch.setattr(ralph_module, "execute", fake_execute)
    monkeypatch.setattr(ralph_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(ralph_module, "interpret_results", fake_interpret)

    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.80,
        metric="f1",
        max_iterations=3,
        workdir=workdir,
        experiments_path=experiments_path,
        stop_mode="optimize",
    )

    # Should have run all 3 iterations rather than stopping at iteration 1.
    assert result["iterations_run"] == 3
    assert len(result["all_experiments"]) == 3
    assert result["target_ever_achieved"] is True
    assert result["status"] in ("success", "optimized")
    assert result["best_experiment"]["score"] == 0.95


def test_run_ralph_loop_saves_best_model_to_disk(tmp_path):
    """End-to-end: run_ralph_loop against a small synthetic dataset
    should persist a best_model.joblib that's a real, loadable, fitted
    estimator (or pipeline)."""
    workspace_dir = str(tmp_path / "workspace")
    experiments_path = os.path.join(workspace_dir, "experiments", "scope", "experiments.json")
    workdir = str(tmp_path / "generated")

    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.999,  # unreachable -- forces several iterations
        metric="f1",
        max_iterations=3,
        workdir=workdir,
        experiments_path=experiments_path,
    )

    assert result.get("best_model_path")
    assert os.path.exists(result["best_model_path"])

    loaded = joblib.load(result["best_model_path"])
    assert hasattr(loaded, "predict")

    # Multiple per-iteration model files should also exist (one per
    # experiment that trained something).
    model_paths = [e.get("model_path") for e in result["all_experiments"] if e.get("model_path")]
    assert len(model_paths) >= 1
    for p in model_paths:
        assert os.path.exists(p)


def test_current_validation_records_without_model_path_do_not_crash(tmp_path):
    """Current-protocol history with a missing model path remains readable."""
    workspace_dir = str(tmp_path / "workspace")
    experiments_path = os.path.join(workspace_dir, "experiments", "scope", "experiments.json")
    workdir = str(tmp_path / "generated")

    seeded = [
        {
            "iteration": 1,
            "model": "random_forest",
            "metric": "f1",
            "score": 0.95,
            "status": "fail",
            "hyperparams": {"n_estimators": 50, "max_depth": None},
            "feature_engineering": "basic",
            # no "model_path" key -- old-format record
        },
    ]
    os.makedirs(os.path.dirname(experiments_path), exist_ok=True)
    with open(experiments_path, "w", encoding="utf-8") as f:
        json.dump([dict(record, validation_version=VALIDATION_VERSION, scope_key=scope_key(DATASET_PATH, "classification", "churn", "f1")) for record in seeded], f)

    # Should not raise even though the seeded best_experiment (score 0.95,
    # very likely to remain best) has no model_path.
    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.999,
        metric="f1",
        max_iterations=1,
        workdir=workdir,
        experiments_path=experiments_path,
    )

    assert result["best_experiment"].get("model_path") is None
    # best_model_path should be None (nothing to copy) rather than a crash.
    assert result.get("best_model_path") is None


def test_run_ralph_loop_writes_new_artifact_files(tmp_path):
    """best_model.json, feature_info.json, and training_history.json
    must be written alongside best_model.joblib in the scoped models
    dir, with valid JSON and the expected keys/values."""
    workspace_dir = str(tmp_path / "workspace")
    experiments_path = os.path.join(workspace_dir, "experiments", "scope", "experiments.json")
    workdir = str(tmp_path / "generated")

    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.999,  # unreachable -- forces several iterations
        metric="f1",
        max_iterations=3,
        workdir=workdir,
        experiments_path=experiments_path,
    )

    models_dir = os.path.dirname(result["best_model_path"])
    best_model_json_path = os.path.join(models_dir, "best_model.json")
    feature_info_path = os.path.join(models_dir, "feature_info.json")
    training_history_path = os.path.join(models_dir, "training_history.json")

    for p in (best_model_json_path, feature_info_path, training_history_path):
        assert os.path.exists(p), p

    with open(best_model_json_path) as f:
        best_model_json = json.load(f)
    for key in ("model", "task_type", "metric", "score", "target_score", "direction", "feature_engineering", "hyperparameters", "status"):
        assert key in best_model_json

    best = result["best_experiment"]
    assert best_model_json["score"] == best["score"]
    assert best_model_json["model"] == best["model"]
    assert best_model_json["metric"] == best["metric"]

    with open(feature_info_path) as f:
        feature_info = json.load(f)
    for key in ("features", "target", "n_features", "train_samples", "test_samples"):
        assert key in feature_info
    assert feature_info["n_features"] == len(feature_info["features"])
    total = feature_info["train_samples"] + feature_info["test_samples"]
    # dataset has a small, known row count -- allow slack for dropped-NaN-target rows.
    import pandas as pd
    df_rows = len(pd.read_csv(DATASET_PATH))
    assert abs(total - df_rows) <= df_rows  # sanity: non-zero and plausible
    assert total > 0

    with open(training_history_path) as f:
        history = json.load(f)
    assert isinstance(history, list)
    assert len(history) == len(result["all_experiments"])
    for rec in history:
        for key in ("iteration", "model", "metric", "score", "status", "feature_engineering", "hyperparams"):
            assert key in rec


def test_run_ralph_loop_promotes_schema_and_explanation_artifacts(tmp_path):
    """Catches winner metadata being lost after the sandbox training result."""
    workspace_dir = str(tmp_path / "workspace")
    experiments_path = os.path.join(workspace_dir, "experiments", "scope", "experiments.json")
    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.999,
        metric="f1",
        max_iterations=1,
        workdir=str(tmp_path / "generated"),
        experiments_path=experiments_path,
    )

    models_dir = os.path.dirname(result["best_model_path"])
    with open(os.path.join(models_dir, "schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    with open(os.path.join(models_dir, "explainability.json"), encoding="utf-8") as f:
        explanation = json.load(f)

    assert schema == result["model_schema"] == result["best_experiment"]["model_schema"]
    assert explanation == result["explainability"] == result["best_experiment"]["explainability"]
    assert schema["raw_columns"]
    assert explanation["available"] is True


def test_failed_model_promotion_preserves_existing_trust_artifacts(tmp_path, monkeypatch):
    """Catches schema metadata describing a model that failed to promote."""
    import app.ralph as ralph_module

    run_dir = tmp_path / "run"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True)
    old_schema = {"schema_version": 1, "raw_columns": ["old"]}
    old_explanation = {"available": False, "method": None, "features": [], "reason": "old"}
    (models_dir / "schema.json").write_text(json.dumps(old_schema), encoding="utf-8")
    (models_dir / "explainability.json").write_text(json.dumps(old_explanation), encoding="utf-8")

    def fail_copy(source, destination):
        raise OSError("promotion copy failed")

    monkeypatch.setattr(ralph_module.shutil, "copy2", fail_copy)
    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.999,
        metric="f1",
        max_iterations=1,
        workdir=str(tmp_path / "generated"),
        experiments_path=str(tmp_path / "experiments.json"),
        run_dir=str(run_dir),
    )

    assert result["best_model_path"] is None
    assert json.loads((models_dir / "schema.json").read_text(encoding="utf-8")) == old_schema
    assert json.loads((models_dir / "explainability.json").read_text(encoding="utf-8")) == old_explanation


def test_generated_code_reports_feature_info_for_basic_and_pipeline(tmp_path):
    """Generated code for both 'basic' and 'pipeline' feature_engineering
    variants must execute successfully and report n_features/train_samples/
    test_samples/feature_names in its printed JSON result."""
    from app.coder import generate_code
    from app.executor import execute
    from app.planner import create_plan
    from tools.dataset import profile_dataset
    import json as _json

    profile = profile_dataset(DATASET_PATH)
    plan = create_plan(profile, "Predict churn", metric="f1", target_score=0.5)
    workdir = str(tmp_path / "generated")
    models_dir = str(tmp_path / "models")

    for i, fe in enumerate(["basic", "pipeline"], start=1):
        code = generate_code(
            DATASET_PATH,
            plan,
            i,
            model_name="random_forest",
            feature_engineering=fe,
            hyperparams={},
            models_dir=models_dir,
        )
        exec_result = execute(code, workdir, i)
        assert exec_result.success, exec_result.stderr
        parsed = _json.loads(exec_result.stdout.strip().splitlines()[-1])
        assert parsed["n_features"] > 0
        assert parsed["train_samples"] > 0
        assert parsed["test_samples"] > 0
        assert isinstance(parsed["feature_names"], list)
        assert len(parsed["feature_names"]) == parsed["n_features"]


def test_reset_clears_prior_history_and_iteration_numbering(tmp_path):
    """`reset=True` must clear the loaded experiments list itself, not
    just the objective/best-tracking seed -- otherwise iteration_offset
    (len(experiments)) still counts old records, so a "fresh" run's own
    iterations report numbers continuing from the old history instead of
    starting at 1, and old records leak into what should be an empty
    "prior" bucket. Regression test for a reported bug where a reset run
    against a scope with 2 stale records still numbered its own first
    iteration as 3.
    """
    experiments_path = str(tmp_path / "experiments.json")
    workdir = str(tmp_path / "generated")

    seeded = [
        {"iteration": 1, "model": "random_forest", "metric": "f1", "score": 0.5,
         "status": "fail", "hyperparams": {}, "feature_engineering": "basic"},
        {"iteration": 2, "model": "ridge", "metric": "f1", "score": 0.55,
         "status": "fail", "hyperparams": {}, "feature_engineering": "basic"},
    ]
    with open(experiments_path, "w", encoding="utf-8") as f:
        json.dump(seeded, f)

    result = run_ralph_loop(
        dataset_path=DATASET_PATH,
        objective="Predict churn",
        target_score=0.01,  # trivially achievable -- stops at iteration 1
        metric="f1",
        max_iterations=3,
        workdir=workdir,
        experiments_path=experiments_path,
        reset=True,
    )

    assert result["iterations_run"] == 1
    new_records = result["all_experiments"][-result["iterations_run"]:]
    assert new_records[0]["iteration"] == 1, "reset run's first iteration must be numbered 1, not continue from stale history"
    # With reset=True, the on-disk experiments.json is fully replaced --
    # the two seeded records must not still be present.
    assert len(result["all_experiments"]) == 1


def test_is_better_accepts_any_genuine_improvement():
    """Best-result tracking must record the true highest score seen, even
    a microscopic one -- MIN_IMPROVEMENT is for stagnation detection
    (is it worth continuing?), not for deciding what the actual best
    result is. A real run hit n_estimators=760 scoring 0.00006 higher
    than n_estimators=507; that tiny margin is still the genuine best
    and must not be silently dropped.
    """
    from app.ralph import _is_better

    current_best = {"score": 0.75}
    candidate = {"score": 0.75 + 1e-5}
    assert _is_better(candidate, current_best, "f1") is True

    candidate_real = {"score": 0.75 + 1e-3}
    assert _is_better(candidate_real, current_best, "f1") is True


def test_is_better_rejects_non_improvement():
    from app.ralph import _is_better

    current_best = {"score": 0.75}
    assert _is_better({"score": 0.75}, current_best, "f1") is False
    assert _is_better({"score": 0.74}, current_best, "f1") is False


def test_is_stagnating_flat_scores_true():
    history = [
        {"score": 0.80}, {"score": 0.81}, {"score": 0.82},
        {"score": 0.8200}, {"score": 0.82001}, {"score": 0.82000},
    ]
    memory = ExperimentMemory(history)
    assert memory.is_stagnating() is True


def test_is_stagnating_improving_run_false():
    history = [
        {"score": 0.60}, {"score": 0.70}, {"score": 0.75},
        {"score": 0.80}, {"score": 0.85}, {"score": 0.90},
    ]
    memory = ExperimentMemory(history)
    assert memory.is_stagnating() is False
