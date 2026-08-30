import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.dataset import profile_dataset


def test_profile_dataset_flags_target_leakage_risk_columns(tmp_path):
    """Regression test for the games.csv report: columns like
    `winner_white`/`victory_status_resign`/`last_move_at` look, by name,
    like they'd only be known after the game ended -- the profiler
    should flag them (never silently drop -- that's a human call).
    """
    df = pd.DataFrame({
        "opening_eco": ["A00", "B01", "C02"],
        "turns": [10, 20, 30],
        "winner_white": [1, 0, 1],
        "victory_status_resign": [0, 1, 0],
        "last_move_at": [1000, 2000, 3000],
        "rated": [True, False, True],
    })
    csv_path = tmp_path / "games.csv"
    df.to_csv(csv_path, index=False)

    profile = profile_dataset(str(csv_path), target_hint="rated")

    flagged = {w["column"] for w in profile["leakage_warnings"]}
    assert "winner_white" in flagged
    assert "victory_status_resign" in flagged
    assert "last_move_at" in flagged
    assert "opening_eco" not in flagged
    assert "turns" not in flagged


def test_profile_dataset_no_leakage_warnings_when_nothing_suspicious(tmp_path):
    df = pd.DataFrame({"age": [20, 30, 40], "income": [1.0, 2.0, 3.0], "churn": [0, 1, 0]})
    csv_path = tmp_path / "clean.csv"
    df.to_csv(csv_path, index=False)

    profile = profile_dataset(str(csv_path), target_hint="churn")
    assert profile["leakage_warnings"] == []


def test_profile_dataset(tmp_path):
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame({
        "age": rng.integers(18, 70, size=n),
        "income": rng.normal(50000, 15000, size=n),
        "churn": rng.integers(0, 2, size=n).astype(int),
    })
    # introduce a couple of NaNs in income
    df.loc[0, "income"] = np.nan
    df.loc[1, "income"] = np.nan

    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    profile = profile_dataset(str(csv_path), target_hint="churn")

    assert profile["rows"] == n
    assert profile["columns"] == 3
    assert "income" in profile["missing_by_column"]
    assert profile["missing_by_column"]["income"] == 2

    assert profile["target"] is not None
    assert profile["target"]["name"] == "churn"
    assert profile["target"]["suggested_task_type"] == "classification"
