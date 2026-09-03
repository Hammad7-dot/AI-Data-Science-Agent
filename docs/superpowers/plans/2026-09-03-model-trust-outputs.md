# Model Trust Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export models that validate raw input schemas and publish model explanations and cross-validation uncertainty throughout the agent's artifacts and interfaces.

**Architecture:** A serializable wrapper owns schema validation around the fitted sklearn pipeline. Focused helpers extract native model importance and calculate uncertainty, while generated training scripts emit fold statistics and Ralph promotes all trust metadata for the selected model.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn, joblib, pytest, Streamlit

**Spec:** `docs/superpowers/specs/2026-09-03-model-trust-outputs-design.md`

## Global Constraints

- Keep the final holdout untouched until model selection finishes.
- Add no SHAP or other explainability dependency.
- Preserve loading of existing bare sklearn exports.
- Validate pandas DataFrames using numeric, boolean, datetime, and categorical/text dtype families.
- Treat the 95% interval as a descriptive interval over validation folds.

---

### Task 1: Serializable raw-input schema validation

**Files:**
- Create: `tools/model_export.py`
- Create: `tests/test_model_export.py`
- Modify: `app/coder.py`

**Interfaces:**
- Produces: `ValidatedModel(model, feature_frame, target_name)`, `ValidatedModel.predict(X)`, `predict_proba(X)`, `decision_function(X)`, and `schema_metadata() -> dict`.
- Consumes: a fitted sklearn estimator/pipeline and the development pandas DataFrame used to fit it.

- [ ] **Step 1: Write failing wrapper tests**

```python
def test_reloaded_validated_model_accepts_reordered_columns(tmp_path):
    X = pd.DataFrame({"age": [20, 30, 40], "city": ["a", "b", "a"]})
    fitted = Pipeline([("prep", ColumnTransformer([...])) , ("model", LogisticRegression())]).fit(X, [0, 1, 0])
    export = ValidatedModel(fitted, X, "target")
    path = tmp_path / "model.joblib"
    joblib.dump(export, path)
    predictions = joblib.load(path).predict(X[["city", "age"]])
    assert len(predictions) == 3

@pytest.mark.parametrize("frame, phrases", [
    (pd.DataFrame({"age": [20]}), ["missing", "city"]),
    (pd.DataFrame({"age": [20], "city": ["a"], "noise": [1]}), ["unexpected", "noise"]),
    (pd.DataFrame({"age": ["old"], "city": ["a"]}), ["incompatible", "age", "numeric"]),
])
def test_schema_errors_are_actionable(frame, phrases):
    with pytest.raises(ValueError) as error:
        export.predict(frame)
    assert all(phrase in str(error.value).lower() for phrase in phrases)
```

- [ ] **Step 2: Verify the tests fail because `tools.model_export` is absent**

Run: `python -m pytest tests/test_model_export.py -q`

Expected: FAIL during import with `ModuleNotFoundError: tools.model_export`.

- [ ] **Step 3: Implement the minimal wrapper**

```python
def dtype_family(dtype):
    if is_bool_dtype(dtype): return "boolean"
    if is_numeric_dtype(dtype): return "numeric"
    if is_datetime64_any_dtype(dtype): return "datetime"
    return "categorical"

class ValidatedModel:
    def __init__(self, model, feature_frame, target_name=None):
        self.model = model
        self.raw_columns = list(feature_frame.columns)
        self.dtype_families = {name: dtype_family(feature_frame[name].dtype) for name in self.raw_columns}
        self.target_name = target_name

    def _validated(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Prediction input must be a pandas DataFrame.")
        missing = [c for c in self.raw_columns if c not in X.columns]
        unexpected = [c for c in X.columns if c not in self.raw_columns]
        incompatible = [c for c in self.raw_columns if c in X and dtype_family(X[c].dtype) != self.dtype_families[c]]
        if missing or unexpected or incompatible:
            raise ValueError(format_schema_error(missing, unexpected, incompatible, self.dtype_families))
        return X[self.raw_columns]
```

Delegate `predict`, `predict_proba`, and `decision_function` after `_validated`, expose `named_steps` for compatibility, and return JSON-safe schema metadata with schema version `1`.

- [ ] **Step 4: Save the wrapper from generated training code**

Import `ValidatedModel` in the generated script and replace `joblib.dump(model, MODEL_PATH)` with:

```python
exported_model = ValidatedModel(model, X_train, TARGET)
joblib.dump(exported_model, MODEL_PATH)
```

- [ ] **Step 5: Verify wrapper and existing training contracts**

Run: `python -m pytest tests/test_model_export.py tests/test_training_contract.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/model_export.py app/coder.py tests/test_model_export.py
git commit -m "feat: validate exported model inputs"
```

### Task 2: Native explainability artifacts

**Files:**
- Create: `tools/explainability.py`
- Create: `tests/test_explainability.py`
- Modify: `app/ralph.py`
- Modify: `tests/test_ralph.py`

**Interfaces:**
- Consumes: `explain_model(exported_model) -> dict` where the export may be `ValidatedModel` or a bare sklearn pipeline.
- Produces: `{"available": bool, "method": str | None, "features": list[dict], "reason": str | None}` and `models/explainability.json`.

- [ ] **Step 1: Write failing explanation tests**

```python
def test_linear_explanation_aligns_transformed_features():
    explanation = explain_model(fitted_validated_logistic_model)
    assert explanation["available"] is True
    assert explanation["method"] == "coefficient"
    assert {row["feature"] for row in explanation["features"]} == set(transformed_names)
    assert explanation["features"] == sorted(explanation["features"], key=lambda row: row["importance"], reverse=True)

def test_unsupported_model_explains_unavailability():
    explanation = explain_model(fitted_knn_export)
    assert explanation == {"available": False, "method": None, "features": [], "reason": "Model does not expose coefficients or feature importances."}
```

- [ ] **Step 2: Verify the tests fail because the helper is absent**

Run: `python -m pytest tests/test_explainability.py -q`

Expected: FAIL during import with `ModuleNotFoundError: tools.explainability`.

- [ ] **Step 3: Implement native importance extraction**

Unwrap `ValidatedModel.model`, read `pipeline.named_steps["preprocess"].get_feature_names_out()`, and inspect `pipeline.named_steps["model"]`. For `coef_`, compute importance as absolute value for one vector or the mean absolute value across class vectors; include `coefficient` only for one vector. For `feature_importances_`, emit nonnegative importance. Return unavailable when names and values differ in length.

- [ ] **Step 4: Promote the winner's artifacts**

After `best_model.joblib` is atomically promoted in `app/ralph.py`, load it, write `schema.json` from `schema_metadata()` when present, call `explain_model`, write `explainability.json`, and add both dictionaries to the Ralph result as `model_schema` and `explainability`.

- [ ] **Step 5: Verify explanation and artifact tests**

Run: `python -m pytest tests/test_explainability.py tests/test_ralph.py -q`

Expected: PASS and the best-model artifact test finds both new JSON files.

- [ ] **Step 6: Commit**

```bash
git add tools/explainability.py app/ralph.py tests/test_explainability.py tests/test_ralph.py
git commit -m "feat: publish model explanations"
```

### Task 3: Fold-level uncertainty

**Files:**
- Create: `tools/uncertainty.py`
- Create: `tests/test_uncertainty.py`
- Modify: `app/coder.py`
- Modify: `app/ralph.py`
- Modify: `tests/test_training_contract.py`

**Interfaces:**
- Produces: `summarize_cv_scores(scores) -> {"cv_scores": list[float], "cv_mean": float, "cv_std": float, "cv_interval_95": [float, float] | None}`.
- Generated experiment JSON includes these four fields.

- [ ] **Step 1: Write failing numerical tests**

```python
def test_cv_summary_uses_sample_standard_deviation():
    result = summarize_cv_scores([0.7, 0.8, 0.9])
    assert result["cv_mean"] == pytest.approx(0.8)
    assert result["cv_std"] == pytest.approx(0.1)
    margin = 1.96 * 0.1 / math.sqrt(3)
    assert result["cv_interval_95"] == pytest.approx([0.8 - margin, 0.8 + margin])
```

- [ ] **Step 2: Verify the helper test fails**

Run: `python -m pytest tests/test_uncertainty.py -q`

Expected: FAIL during import with `ModuleNotFoundError: tools.uncertainty`.

- [ ] **Step 3: Implement the summary helper**

Convert NumPy values to finite Python floats, calculate the arithmetic mean, sample standard deviation (`ddof=1`), and the interval formula from the spec. Reject empty or nonfinite inputs with `ValueError`; return `None` for the interval and `0.0` standard deviation for one score.

- [ ] **Step 4: Calculate fold scores in generated code**

Use the already created splitter for both `cross_val_predict` and `cross_validate`. Select scorers as follows: `accuracy`, `precision_weighted`, `recall_weighted`, `f1_weighted`, `r2`, `neg_root_mean_squared_error`, and `neg_mean_absolute_error`. Negate loss scores before summarizing and merge the returned summary into the emitted iteration result. Continue using the out-of-fold prediction aggregate as `score`.

- [ ] **Step 5: Preserve uncertainty fields in experiment memory**

Add `cv_scores`, `cv_mean`, `cv_std`, and `cv_interval_95` to the successful record constructed in `app/ralph.py`.

- [ ] **Step 6: Verify numerical and generated-training tests**

Run: `python -m pytest tests/test_uncertainty.py tests/test_training_contract.py tests/test_ralph.py -q`

Expected: PASS; classification and regression records contain three finite fold scores and consistent summaries.

- [ ] **Step 7: Commit**

```bash
git add tools/uncertainty.py app/coder.py app/ralph.py tests/test_uncertainty.py tests/test_training_contract.py
git commit -m "feat: report validation uncertainty"
```

### Task 4: Reports, CLI, and Streamlit presentation

**Files:**
- Modify: `app/agent.py`
- Modify: `app/main.py`
- Modify: `app/streamlit_app.py`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_streamlit_smoke.py`

**Interfaces:**
- Consumes: best experiment uncertainty fields, `result["model_schema"]`, and `result["explainability"]`.
- Produces: Markdown sections `Validation uncertainty`, `Raw input schema`, and `Model explanation`; equivalent concise CLI and Streamlit views.

- [ ] **Step 1: Write failing report assertions**

Extend an end-to-end agent test to assert:

```python
report = Path(result["report_path"]).read_text()
assert "## Validation uncertainty" in report
assert "95% descriptive interval" in report
assert "## Raw input schema" in report
assert "## Model explanation" in report
assert result["model_schema"]["schema_version"] == 1
assert "available" in result["explainability"]
```

- [ ] **Step 2: Verify report tests fail on missing sections**

Run: `python -m pytest tests/test_agent.py -q`

Expected: FAIL because the trust sections are absent.

- [ ] **Step 3: Add Markdown and CLI output**

In `_build_report`, display fold values to four decimals, mean ± standard deviation, and the descriptive interval. List raw schema columns with dtype families. Show the ten highest-ranked explanation rows or the unavailability reason. Add the same uncertainty summary and artifact paths to CLI output.

- [ ] **Step 4: Add Streamlit cards**

In the Best Model area, show `CV mean`, `CV std`, and `95% interval`; then add `Raw input schema` and `Model explanation` sections. Render explanation rows in a dataframe and bar chart, and render the explicit reason when unavailable. Load JSON artifacts only as a fallback when the result lacks the dictionaries.

- [ ] **Step 5: Verify interface tests**

Run: `python -m pytest tests/test_agent.py tests/test_streamlit_smoke.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/agent.py app/main.py app/streamlit_app.py tests/test_agent.py tests/test_streamlit_smoke.py
git commit -m "feat: surface model trust details"
```

### Task 5: Documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_sandbox_contract.py`

**Interfaces:**
- Consumes: final exported model and trust artifacts.
- Produces: documented prediction contract and Docker regression coverage.

- [ ] **Step 1: Extend the Docker assertion**

After the real Docker run, assert `schema.json` and `explainability.json` exist beside `best_model.joblib`, and assert the result contains finite `cv_std` and a two-value `cv_interval_95`.

- [ ] **Step 2: Verify the Docker test locally when available**

Run: `python -m pytest tests/test_sandbox_contract.py::test_real_docker_training_and_holdout -q`

Expected: PASS with Docker or SKIP when Docker is unavailable; GitHub's Ubuntu job must execute it.

- [ ] **Step 3: Document prediction and interpretation behavior**

Add README examples that load `best_model.joblib`, pass a raw DataFrame, explain actionable schema failures, identify the three new fields/artifacts, and state the descriptive limitation of the fold interval.

- [ ] **Step 4: Run the complete suite and static diff checks**

Run: `python -m pytest tests/ -q`

Expected: all tests pass, with only the existing local Docker skip when Docker is unavailable.

Run: `git diff --check`

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_sandbox_contract.py
git commit -m "docs: explain model trust artifacts"
```

- [ ] **Step 6: Push and verify CI**

Run: `git push origin main`

Then watch the new `Tests` workflow until Ubuntu and Windows pass. If either job fails, inspect the failing log, add a regression test, fix it, and repeat verification before pushing the correction.
