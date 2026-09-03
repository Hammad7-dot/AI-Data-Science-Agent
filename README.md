# AI Data Science Agent

An agent that takes a CSV dataset and a natural-language objective, and
produces an analysis plan, generates Python code, executes it,
evaluates cross-validation results, and loops ("Ralph Loop") improving the code
until a target metric is reached or a max iteration count is hit --
then evaluates the selected model on a held-out partition and writes a Markdown report.

## Validation and exported models

Supervised runs reserve 20% of labeled rows for a final holdout before training.
Classification splits are stratified. The remaining 80% is development data:
its profile drives encoding choices and automatic regression targets, and up to
three shuffled folds produce out-of-fold predictions for model selection.
Every fold fits its own imputation, encoding, scaling, and feature selection.
The target status refers to the cross-validation score, not the final holdout.
At least 10 labeled rows are required; classification also needs enough examples
per class for the stratified holdout and at least two development folds.

After search, `run_agent()` evaluates the selected model once on the holdout.
The CLI, UI, and report show this score separately; `reports/holdout.json` saves
it as an artifact. It never enters experiment memory or influences the search.
The saved model remains fitted on development data, preserving this evaluation.

All encoding modes export a fitted sklearn Pipeline that accepts raw feature
columns, handles missing values and unseen categories, and includes any feature
selection. `basic` uses imputation and one-hot encoding, `pipeline` additionally
scales numeric features and selects up to ten features when appropriate, and
`safe_categorical` learns frequency mappings for medium-cardinality columns.
Regression feature selection uses `f_regression`; classification uses `f_classif`.
Feature names in model metadata describe the actual transformed/selected inputs.
Load models from the project environment so the custom transformers in `tools.ml`
are importable, and supply a DataFrame with the original feature column names.

Validation has a new storage version, so default runs leave previous histories
untouched and start a separate scope. If you explicitly override `experiments_path`
with a legacy log, choose a new path or use `--reset` to discard its old scores.
Previously exported bare estimators need to be retrained to gain preprocessing.

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
   `workspace/runs/<run-id>/reports/report.md`.

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
engineering (training-only one-hot encoding and median imputation) or
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

## Dataset identity and isolated runs

Experiment history is keyed by a SHA-256 hash of the dataset bytes, task type,
target column, metric, validation version, experiment configuration version,
and local versus Docker execution mode. Renaming or moving identical bytes
preserves the scope; replacing a CSV's contents creates a new scope. Explicit
history overrides are checked too, including records copied without their
objective file. Bump `EXPERIMENT_CONFIG_VERSION` in `app/run_scope.py` when
changing training/search behavior incompatibly.

Every `run_agent()` call snapshots its CSV before profiling and creates:

```
workspace/runs/<run-id>/inputs/<dataset.csv>
workspace/runs/<run-id>/generated/iteration_N.py
workspace/runs/<run-id>/models/iteration_N_<model>.joblib
workspace/runs/<run-id>/models/best_model.joblib
workspace/runs/<run-id>/reports/report.md
workspace/runs/<run-id>/reports/holdout.json
workspace/runs/<run-id>/experiments.json
```

Shared resumable history remains in `workspace/experiments/<scope>/experiments.json`.
A run's `experiments.json` is a frozen snapshot used for UI downloads. Reports,
models, input files and generated code from earlier runs remain unchanged.
Uploads also get unique directories, even when their filenames match.

Cross-process locks prevent two writers using the same history simultaneously;
a second writer gets a clear active-run error and can retry after completion.
Different histories can run concurrently. History writes use atomic replacement.
No filesystem location is inferred from an explicit history override for model
output; models always stay inside their run directory.

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
target of `0.5 * target_std` from the development profile, documented inline as a
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

Outputs use the run layout above. The CLI prints the exact report and model
paths; `run_agent()` also returns `run_dir`, `generated_dir`, `dataset_snapshot`,
`report_path`, `best_model_path`, and `run_experiments_path`.

For a local run, load a model from the project environment using its returned path:

```python
import joblib
import pandas as pd

model = joblib.load(result["best_model_path"])
new_data = pd.DataFrame({
    "age": [34, 52],
    "city": ["Lahore", "Karachi"],
})
predictions = model.predict(new_data)  # raw feature DataFrame
```

`best_model.joblib` accepts the original, unprocessed feature columns. It
performs the saved imputation, encoding, scaling, and feature selection itself.
The DataFrame must include every raw feature exactly once and use the same dtype
family (`numeric`, `boolean`, `datetime`, or categorical/text) seen at training.
If it does not, prediction raises an actionable `ValueError` such as
`Prediction input schema mismatch: missing columns: city` or
`incompatible columns: age (expected numeric)`. Use the exported schema to see
the required input contract:

```python
import json
from pathlib import Path

models_dir = Path(result["best_model_path"]).parent
schema = json.loads((models_dir / "schema.json").read_text())
print(schema["raw_columns"])
print(schema["dtype_families"])
```

Each successful run publishes three model-trust outputs:

- Validation uncertainty in `result["best_experiment"]`: `cv_scores`,
  `cv_mean`, `cv_std`, and `cv_interval_95` summarize the development folds.
- `models/schema.json` (also `result["model_schema"]`) describes the raw
  columns and dtype families accepted by `best_model.joblib`.
- `models/explainability.json` (also `result["explainability"]`) ranks native
  coefficients or feature importances when the selected estimator exposes them;
  otherwise it records why an explanation is unavailable.

The 95% fold interval is descriptive: it shows variation among the validation
fold scores. It is neither a confidence guarantee for the selected model nor a
prediction interval for individual rows.

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
`workspace/uploads/<upload-id>/<uploaded_name>`. Agent errors (bad CSV, no usable
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

If Docker or its daemon/image is unavailable, the requested sandbox run fails
clearly and the CLI exits nonzero. Code is never retried on the host. The runner
mounts the dataset and `tools/` helpers read-only, maps writable generated-code
and model directories explicitly, disables networking, uses a read-only root
filesystem and bounds memory, CPU and process count. Timeouts forcibly remove
the named container. The executed script is mounted read-only, and model promotion
uses atomic replacement to avoid following pre-existing output links. Generated
scripts resolve their dataset, helper and output
paths from environment variables supplied by the runner.

Model paths returned by container code must resolve inside the model output
mount. Holdout prediction and model deserialization also run inside Docker;
the Streamlit UI does not deserialize sandbox-created model files on the host.
Enable the UI's **Run code in Docker sandbox** checkbox for this execution mode.

`tests/test_sandbox_contract.py` includes command-contract tests and a real
container training/holdout test. The latter skips locally if Docker or the image
is missing. The Linux CI job builds the sandbox image before running the suite.

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
