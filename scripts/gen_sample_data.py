"""One-off generator for workspace/datasets/sample_churn.csv (synthetic data)."""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 150

customer_id = np.arange(1, n + 1)
age = np.random.randint(18, 75, size=n)
tenure_months = np.random.randint(0, 72, size=n)
monthly_charges = np.round(np.random.uniform(20, 120, size=n), 2)
contract_type = np.random.choice(
    ["month-to-month", "one-year", "two-year"], size=n, p=[0.5, 0.3, 0.2]
)

# churn probability: higher charges + low tenure + month-to-month -> more churn
base = (
    0.35
    + 0.004 * (monthly_charges - 70)
    - 0.006 * tenure_months
    + np.where(contract_type == "month-to-month", 0.20, 0.0)
    + np.where(contract_type == "two-year", -0.15, 0.0)
)
prob = np.clip(base, 0.02, 0.95)
churn = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    "customer_id": customer_id,
    "age": age,
    "monthly_charges": monthly_charges,
    "tenure_months": tenure_months,
    "contract_type": contract_type,
    "churn": churn,
})

df.to_csv("workspace/datasets/sample_churn.csv", index=False)
print("wrote", len(df), "rows; churn rate =", df["churn"].mean())
