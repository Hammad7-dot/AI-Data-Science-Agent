# AI Data Science Agent

## Goal

The agent accepts a CSV dataset and a natural-language
data science objective.

## Input

- CSV dataset
- User objective

Example:

"Predict whether a customer will churn."

## Agent responsibilities

1. Inspect dataset
2. Identify target column
3. Analyze missing values
4. Perform exploratory data analysis
5. Prepare data
6. Select models
7. Train models
8. Evaluate models
9. Improve poor results
10. Generate final report

## Ralph Loop

The agent may perform multiple iterations.

Each iteration must:

1. Inspect current state
2. Generate/improve code
3. Execute code
4. Evaluate result
5. Record experiment
6. Decide whether to continue

## Stop condition

Stop when:

- required metric is reached
- maximum iterations reached
- or the task is successfully completed