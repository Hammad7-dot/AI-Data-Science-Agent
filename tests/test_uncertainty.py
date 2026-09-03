import math

import pytest

from tools.uncertainty import summarize_cv_scores


def test_cv_summary_uses_sample_standard_deviation():
    """Catches population deviation or a confidence interval with the wrong margin."""
    result = summarize_cv_scores([0.7, 0.8, 0.9])

    assert result["cv_mean"] == pytest.approx(0.8)
    assert result["cv_std"] == pytest.approx(0.1)
    margin = 1.96 * 0.1 / math.sqrt(3)
    assert result["cv_interval_95"] == pytest.approx([0.8 - margin, 0.8 + margin])


@pytest.mark.parametrize("scores", [[], [0.8, float("nan")], [float("inf")]])
def test_cv_summary_rejects_empty_or_nonfinite_scores(scores):
    """Catches invalid fold scores being emitted as misleading uncertainty data."""
    with pytest.raises(ValueError, match="finite"):
        summarize_cv_scores(scores)


def test_cv_summary_for_one_score_has_zero_deviation_and_no_interval():
    """Catches a fabricated fold interval when only one fold score is available."""
    assert summarize_cv_scores([0.75]) == {
        "cv_scores": [0.75],
        "cv_mean": 0.75,
        "cv_std": 0.0,
        "cv_interval_95": None,
    }
