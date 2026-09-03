# Model Trust Outputs Design

## Goal

Make trained results safer to consume and easier to assess by validating raw prediction inputs, explaining the selected model, and reporting variation across validation folds.

## Export schema validation

Add a small, joblib-serializable `ValidatedModel` wrapper in `tools/model_export.py`. It owns the fitted sklearn pipeline and a raw input schema captured from the development feature frame. `predict`, `predict_proba`, and `decision_function` validate a pandas DataFrame before delegating.

Validation requires every training feature exactly once, rejects unexpected columns by default, accepts any column order by reordering internally, and checks broad dtype families: numeric, boolean, datetime, and categorical/text. Errors list missing, unexpected, and incompatible columns in one actionable message. The wrapper exposes the underlying pipeline through `model` and delegates other attributes for compatibility. Generated training scripts save this wrapper rather than a bare pipeline. Holdout evaluation continues to call `.predict()` and therefore exercises the exported contract after reload.

The exported metadata records schema version, ordered raw columns, dtype families, and target name. Existing bare joblib files remain loadable but do not gain validation until retrained.

## Explainability

Add model-agnostic artifact extraction in `tools/explainability.py` for models that expose `coef_` or `feature_importances_`. It reads transformed feature names from the fitted preprocessor, verifies exact dimensional alignment, aggregates multiclass coefficients by mean absolute magnitude, and returns a ranked list containing feature, importance, and signed coefficient when one unambiguous coefficient vector exists.

Unsupported models return an explicit availability status and reason. The result is written to `models/explainability.json`, included in the result object, summarized in the Markdown report, and rendered as a table/chart in Streamlit. No SHAP dependency is introduced.

## Validation uncertainty

Generated training code will use `cross_validate` with the selected metric so each existing validation split yields a score. Classification uses sklearn scorers compatible with the current weighted metrics. Loss metrics are converted back to positive RMSE or MAE values. The experiment result keeps the current aggregate selection score and adds `cv_scores`, `cv_mean`, `cv_std`, and a descriptive 95% interval calculated as `mean ± 1.96 * sample_std / sqrt(n)`.

Model selection continues to use the existing aggregate out-of-fold prediction score to preserve behavior. The interval describes fold variation and is never treated as a formal population guarantee. Reports, CLI, and Streamlit show fold count, mean, standard deviation, interval, and the separate untouched holdout score.

## Data flow and artifacts

Each iteration trains and serializes a validated model, then emits uncertainty fields in its JSON record. When Ralph promotes the winner, it reloads the promoted artifact, extracts explainability, and writes `schema.json` and `explainability.json` next to existing model metadata. `run_agent` exposes both artifacts and includes them in the report. Docker runs use the same modules already mounted under `/opt/agent/tools`.

## Error handling

- Invalid prediction inputs raise `ValueError` before sklearn preprocessing.
- Explainability alignment errors produce an unavailable artifact rather than misleading rankings.
- Missing native model importance produces a clear unsupported reason.
- Fewer than two fold scores omit the interval, though current supervised validation requires at least two folds.

## Tests and acceptance criteria

Tests are written before implementation and must demonstrate:

1. A reloaded export predicts raw data with reordered columns.
2. Missing, extra, and incompatible columns produce actionable validation errors.
3. Coefficient and tree importance rankings align with transformed feature names; unsupported models report why.
4. Fold scores, mean, sample standard deviation, and interval are numerically correct for classification and regression.
5. Best-model schema and explainability artifacts are written and surfaced in Markdown and Streamlit-facing result data.
6. Holdout evaluation remains separate and all existing tests pass locally and in GitHub Actions, including the Linux Docker integration.
