import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.python_runner import ExecutionResult
from app.evaluator import evaluate


def _make_result(stdout: str) -> ExecutionResult:
    return ExecutionResult(
        success=True,
        stdout=stdout,
        stderr="",
        exit_code=0,
        duration_seconds=0.1,
        artifacts=[],
        timed_out=False,
    )


def test_evaluate_f1_success():
    result = _make_result('{"model":"random_forest","metric":"f1","score":0.85}\n')
    plan = {"metric": "f1"}
    out = evaluate(result, plan, target_score=0.80)
    assert out["status"] == "success"


def test_evaluate_f1_fail():
    result = _make_result('{"model":"random_forest","metric":"f1","score":0.85}\n')
    plan = {"metric": "f1"}
    out = evaluate(result, plan, target_score=0.90)
    assert out["status"] == "fail"


def test_evaluate_rmse_success():
    result = _make_result('{"model":"linear_regression","metric":"rmse","score":5.0}\n')
    plan = {"metric": "rmse"}
    out = evaluate(result, plan, target_score=10)
    assert out["status"] == "success"


def test_evaluate_rmse_fail():
    result = _make_result('{"model":"linear_regression","metric":"rmse","score":5.0}\n')
    plan = {"metric": "rmse"}
    out = evaluate(result, plan, target_score=2)
    assert out["status"] == "fail"
