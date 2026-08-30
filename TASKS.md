# TASKS

Roadmap checklist for the AI Data Science Agent.

## Phase 0 MVP (complete)

- [x] Phase 1 - Project scaffolding
- [x] Phase 2 - Requirements / env config
- [x] Phase 3 - Dataset profiler (`tools/dataset.py`)
- [x] Phase 4 - Python execution sandbox (`tools/python_runner.py`)
- [x] Phase 5 - Planner (rule-based stand-in, `app/planner.py`)
- [x] Phase 6 - Coder (rule-based stand-in, `app/coder.py`)
- [x] Phase 7 - Executor (`app/executor.py`)
- [x] Phase 8 - Evaluator (`app/evaluator.py`)
- [x] Phase 9 - Ralph Loop + report + CLI (`app/ralph.py`, `app/agent.py`, `app/main.py`)

## Phase 10-13 (complete)

- [x] Phase 10 - ML experimentation expansion (`tools/ml.py`): model
      factory (`get_model`) covering 5 classification + 5 regression
      models, `build_preprocessing_pipeline` (scaling, one-hot,
      optional `SelectKBest`), `hyperparameter_grid`/`pick_hyperparams`,
      `list_models_for_task` (now the single source of truth, used by
      `app/planner.py`). `app/coder.py` alternates "basic" vs "pipeline"
      feature engineering and varies hyperparameters per iteration;
      generated scripts' JSON output carries `hyperparams` and
      `feature_engineering`. `app/ralph.py` records both and uses
      (model, hyperparams, feature_engineering) as the experiment
      identity.
- [x] Phase 11 - Analyst split (`app/analyst.py`): rule-based
      `interpret_results()` stand-in for an LLM call, producing a
      `summary` + `recommendation` per iteration. Wired into
      `app/ralph.py` (stored as `"analysis"`, recommendation feeds the
      next iteration's `previous_feedback`) and into `app/agent.py`'s
      `report.md` (Analyst notes section + final recommendation).
- [x] Phase 12a - Streamlit UI (`app/streamlit_app.py`): upload CSV,
      set objective/metric/target score/max iterations, run the agent,
      view per-iteration results + Analyst summaries, final banner,
      download `report.md`/`experiments.json`. Errors from `run_agent`
      are caught and shown via `st.error` instead of crashing.
- [x] Phase 12b - Docker sandbox (`docker/sandbox.Dockerfile`,
      `tools/python_runner.run_script_in_docker`,
      `app/executor.execute(..., use_docker=...)`, CLI `--sandbox`
      flag). **Honest status**: Docker was not available (no daemon)
      in the environment this was built/tested in, so only the
      graceful-fallback path (`sandboxed=False`, plain subprocess) is
      actually exercised by `tests/test_python_runner_docker.py`. The
      real containerized-isolation path is implemented per spec
      (`--network none`, memory/CPU limits) but not run end-to-end
      here; the test skips itself if a working docker daemon is
      present rather than silently passing.
- [x] Phase 13 - Deployment scaffolding: `docker/app.Dockerfile`
      (python:3.11-slim, installs `requirements.txt`, runs
      `streamlit run app/streamlit_app.py`), `docker-compose.yml`
      (`agent` service on 8501, `./workspace` volume; commented-out
      `sandbox` service stub). Not run end-to-end in this environment
      either (no docker daemon) -- Dockerfiles/compose file are
      reviewed for correctness but not build-tested here.
