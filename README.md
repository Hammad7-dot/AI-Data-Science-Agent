# AI Data Science Agent

An agent that takes a CSV dataset and a natural-language objective, and
produces an analysis plan, generates Python code, executes it,
evaluates the results, and loops ("Ralph Loop") improving the code
until a target metric is reached or a max iteration count is hit --
then writes a Markdown report.

## Pipeline

```
CSV -> Planner -> Coder -> Executor -> Evaluator -> Ralph Loop -> Report
```

1. **Profiler** (`tools/dataset.py`) reads the CSV and computes a
   structured profile (shape, dtypes, missing values, target column
   guess, suggested task type).
2. **Planner** (`app/planner.py`) turns the profile + objective into a
   structured plan (task type, target, metric, steps, candidate models).
3. **Coder** (`app/coder.py`) generates a runnable Python script for the
   current iteration's candidate model.
4. **Executor** (`app/executor.py`) runs that script in a subprocess via
   `tools/python_runner.py` and captures stdout/stderr/artifacts.
5. **Evaluator** (`app/evaluator.py`) parses the script's JSON metrics
   line and compares the score against the target.
6. **Analyst** (`app/analyst.py`) turns the evaluator's raw dict into a
   plain-language summary and a "what to try next" recommendation, fed
   back into the next iteration's `generate_code()` call.
7. **Ralph Loop** (`app/ralph.py`) repeats steps 3-6, varying model,
   hyperparameters, and feature engineering across iterations (see
   `tools/ml.py`), logging every experiment, until success or
   `max_iterations` is reached.
8. **Agent** (`app/agent.py`) orchestrates the whole run and writes a
   Markdown report (including Analyst notes) to
   `workspace/reports/report.md`.

## `tools/ml.py` -- ML experimentation expansion

Single source of truth for model construction and search space, used by
both `app/planner.py` (`list_models_for_task`) and `app/coder.py`
(`get_model`, `build_preprocessing_pipeline`, `hyperparameter_grid`,
`pick_hyperparams`):

- `get_model(name, task_type, hyperparams=None)` -- factory for an
  sklearn estimator (logistic_regression, random_forest,
  gradient_boosting, knn, svm for classification; linear_regression,
  random_forest_regressor, gradient_boosting_regressor, ridge,
  knn_regressor for regression).
- `build_preprocessing_pipeline(numeric_cols, categorical_cols,
  scale_numeric=True, feature_selection_k=None)` -- a ColumnTransformer
  pipeline (median/most-frequent imputation, `StandardScaler`,
  `OneHotEncoder(handle_unknown="ignore")`, optional `SelectKBest`).
- `hyperparameter_grid(model_name)` / `pick_hyperparams(model_name,
  iteration)` -- a small, cheap grid per model and a deterministic pick
  per Ralph iteration.

Each Ralph iteration's generated script uses either "basic" feature
engineering (one-hot + median-fill, the original behavior) or
"pipeline" (`build_preprocessing_pipeline` with scaling/feature
selection). The script's single JSON stdout line includes
`"hyperparams"` and `"feature_engineering"` keys.

## Adaptive Ralph Loop -- Experiment Memory + Strategist

`app/ralph.py` no longer walks a precomputed `itertools.product` grid.
Each iteration it asks `app/experiment_strategist.py`'s
`choose_next_experiment(plan, memory, iteration)` what to try next,
where `memory` is an `app/experiment_memory.py::ExperimentMemory`
seeded from every record already in `experiments.json` (cross-run
memory) plus this run's results so far. The (rule-based, TODO-commented
for a future LLM swap) strategy:

1. **Exploration** -- try every `candidate_models` entry once, with
   default hyperparams and `"basic"` feature engineering, to get a
   cheap baseline per model before spending budget on any one of them.
2. **Exploitation** -- once every model has a baseline, refine the
   *currently best-performing* model's hyperparameters (via
   `tools.ml.iter_hyperparam_combos`), instead of mechanically cycling
   through every model regardless of how it's doing.
3. **Stagnation handling** -- `ExperimentMemory.is_stagnant(window=3,
   min_improvement=0.01, direction=...)` checks whether the best score
   in the last 3 iterations improved meaningfully over the best score
   before that window. If it's stagnant, the strategist pivots: try
   `"pipeline"` feature engineering on the best model if not yet tried,
   or move to the next-best under-explored model, rather than blindly
   continuing the same combo sequence. Each choice carries a
   `"rationale"` string (stored per-record in `experiments.json`,
   distinct from the Analyst's post-hoc `"analysis"`) explaining why it
   was picked.
4. If literally everything (every model x every hyperparam combo x both
   feature-engineering modes) is already in
   `memory.tried_identities()`, `choose_next_experiment` returns `None`
   and the run stops with `status == "search_space_exhausted"`.

Duplicate prevention is unchanged in guarantee, just driven by
`ExperimentMemory.tried_identities()` instead of a static queue: the
(model, hyperparams, feature_engineering) identity of every record ever
written to `experiments.json` -- this run and prior runs -- is never
repeated.

## Frozen objective per run

`app/planner.py::create_plan()` resolves `metric`, `target_score`, and
`direction` ("maximize"/"minimize", derived from
`app.evaluator.LOWER_IS_BETTER`) exactly once, at the start of a run,
and nothing in `app/ralph.py` reassigns those plan keys afterward.
`run_ralph_loop` also writes the frozen objective to
`<experiments_dir>/objective.json` next to `experiments.json` (chosen
over restructuring `experiments.json` itself since it's less invasive
to existing callers/tests, which expect a bare list there). If a run's
objective (metric/target_score) differs from what's recorded in
`objective.json` for that same `experiments_path`, `run_ralph_loop`
returns `"objective_changed": True` and `"previous_objective": {...}`
instead of silently mixing incompatible histories -- surfaced by the
CLI/Streamlit/`report.md` as a warning banner, not a raise.

## Per-objective experiment storage scoping

`app/agent.py::run_agent()` no longer writes every run into one fixed
`workspace/experiments.json`. Before entering the Ralph Loop it profiles
the dataset, derives the plan (task type, target column, metric), and
computes a scope key from `(dataset_path, task_type, target_column,
metric)` via `app/run_scope.py::scope_key()` (dataset path resolved to
absolute first, so relative vs absolute references to the same file
scope identically). Each distinct objective gets its own directory:

```
workspace/experiments/<dataset-stem>_<task_type>_<hash>/experiments.json
workspace/experiments/<dataset-stem>_<task_type>_<hash>/objective.json
```

e.g. a classification run against `sample_churn.csv` and a regression
run against `sample_regression.csv` land in two entirely separate
files/directories and never see each other's records -- this is what
fixes the earlier bug where an unrelated dataset/task's iterations
showed up inside a different run's history and iteration numbering.
Reruns of the *literal same* objective (same dataset + task type +
target + metric) still resolve to the same scoped file, so cross-run
duplicate prevention and best-tracking (`app/experiment_memory.py`,
`app/ralph.py`) continue to work exactly as before within that scope.

Pass an explicit `experiments_path=` to `run_agent()` (or
`--experiments-path` on the CLI) to override auto-scoping.

## `stop_mode`: target vs optimize

`run_agent()` / `run_ralph_loop()` accept `stop_mode` (`--mode` on the
CLI, a radio button in Streamlit):

- **`"target"`** (default) -- stop as soon as any model meets the goal.
  Unchanged from prior behavior.
- **`"optimize"`** -- keep searching for the best possible result up to
  `max_iterations` (or until the search space is exhausted), even after
  the target has already been met. `best_experiment` keeps being
  updated via the normal comparison logic. The result's final `status`
  is `"success"` if the loop stopped because it ran out of exploration
  ("optimize" applied to a run that never happens to overshoot),
  `"optimized"` if the target was met at some point but the loop kept
  running to the end anyway, or the usual `"search_space_exhausted"` /
  `"max_iterations_reached"` if the target was never met at all. Check
  `result["target_ever_achieved"]` for an unambiguous boolean.

## Scale-aware regression target/metric

`profile_dataset()` (`tools/dataset.py`) now includes `mean`/`std`/
`min`/`max` of the target column in `target_info` for regression tasks.
`app/planner.py`'s default regression metric changed from `"rmse"` to
`"r2"` (bounded, scale-free, matches how a human would sanity-check a
result) *unless* the caller explicitly requests `"rmse"`/`"mae"`. When
`target_score` isn't specified by the caller (`None`, distinguishable
from an explicit `0.80`): classification defaults to `0.80` for f1
(unchanged -- f1/accuracy are already bounded `[0,1]`); r2 defaults to
`0.75`; rmse/mae -- which have no fixed scale -- derive a heuristic
target of `0.5 * target_std` from the profile, documented inline as a
heuristic in both a code comment and the plan's `steps` list. This is
why a `$50k`-scale salary dataset no longer gets compared against a
flat, meaningless `0.80`.

## `app/analyst.py` -- interpretation stand-in

Rule-based (not LLM) interpretation of each iteration's result: a 1-3
sentence `"summary"` and a `"recommendation"` for what to try next,
stored per-iteration in `experiments.json` (`"analysis"` key) and in
`report.md`. Marked with the same `# TODO: replace with an LLM call`
style as `planner.py`/`coder.py`.

## Rule-based Planner/Coder/Analyst (no API key needed)

The Planner, Coder, and Analyst are **deterministic, rule-based Python
functions** -- not real LLM calls -- so the whole pipeline runs
end-to-end with zero API keys. Each module has a docstring and inline
`# TODO` comments marking exactly where a real LLM call would replace
the logic:

- `app/planner.py` -> `create_plan()`
- `app/coder.py` -> `generate_code()`
- `app/analyst.py` -> `interpret_results()`

Swapping in a real LLM later should not require changing the function
signatures or the calling code in `app/ralph.py`.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python app/main.py \
  --dataset workspace/datasets/sample_churn.csv \
  --objective "Predict whether a customer will churn" \
  --target-score 0.80 \
  --metric f1 \
  --max-iterations 5
```

Outputs are written under `workspace/`:

- `workspace/generated/iteration_N.py` -- generated scripts per iteration
- `workspace/experiments/<scope>/experiments.json` -- log of every
  experiment tried for this objective (see "Per-objective experiment
  storage scoping" above)
- `workspace/reports/report.md` -- final Markdown report
- `workspace/models/<scope>/` -- the fitted model object from each
  experiment (`iteration_<N>_<model_name>.joblib`, saved via
  `joblib.dump` right after training -- the full sklearn `Pipeline`
  object when feature_engineering=`"pipeline"`, or the bare estimator
  for `"basic"`), plus `best_model.joblib`, a copy of the best-scoring
  experiment's model file for this scope. Load one back with:

  ```python
  import joblib
  model = joblib.load("workspace/models/<scope>/best_model.joblib")
  predictions = model.predict(new_data)
  ```

Add `--sandbox` to run generated code inside a Docker container instead
of a plain local subprocess (see "Docker sandbox" below). Off by
default. Add `--mode optimize` to keep searching past the first
success instead of stopping immediately (default `--mode target`).

## Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

from the project root. Upload a CSV, set an objective/metric/target
score/max iterations, and click **Run Agent**. Shows a per-iteration
breakdown (model, hyperparams, feature engineering, score, status, and
the Analyst's summary), a final "TARGET ACHIEVED" / "MAX ITERATIONS
REACHED" banner, and download buttons for `report.md` and
`experiments.json`. Uploaded CSVs are saved to
`workspace/datasets/<uploaded_name>`. Agent errors (bad CSV, no usable
target, etc.) are shown with `st.error(...)` instead of crashing the
page.

## Docker sandbox (Phase 12b)

`tools/python_runner.run_script_in_docker(...)` runs a generated script
inside a throwaway container built from `docker/sandbox.Dockerfile`
(python:3.11-slim + pandas/numpy/scikit-learn only), with `--network
none` and bounded `--memory`/`--cpus`. Build the image once:

```bash
docker build -t ai-ds-agent-sandbox:latest -f docker/sandbox.Dockerfile .
```

Enable it end-to-end via the CLI's `--sandbox` flag, or `use_docker=True`
on `app.executor.execute` / `app.agent.run_agent` / `app.ralph.run_ralph_loop`.

If the `docker` binary isn't on `PATH`, or a daemon isn't reachable
(`docker info` fails), `run_script_in_docker` automatically falls back
to the plain subprocess path (`ExecutionResult.sandboxed = False`) so
the pipeline keeps working without Docker installed. **This is the
state actually verified in this repo/CI** -- Docker was not available
in the dev environment used to build this, so only the graceful-fallback
path has been exercised (`tests/test_python_runner_docker.py`, which
skips instead of testing real isolation if a working daemon *is*
present). The isolated-container path is implemented per spec but not
exercised end-to-end here.

## Docker Compose (Phase 13)

```bash
docker compose up --build
```

Builds and runs the Streamlit UI (`docker/app.Dockerfile`) on
`localhost:8501`, with `./workspace` mounted into the container for
persistence. `docker-compose.yml` also documents (commented out) a
`sandbox` service stub built from `docker/sandbox.Dockerfile` -- using
it from inside the `agent` container would require mounting the host's
Docker socket (Docker-in-Docker), which is left as an opt-in, not
enabled by default.

## Tests

```bash
pytest tests/
```
