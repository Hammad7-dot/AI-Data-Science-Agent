import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.python_runner import run_script


def test_run_script_success(tmp_path):
    code = '''
with open("out.txt", "w") as f:
    f.write("done")
print("hello")
'''
    result = run_script(code, str(tmp_path), filename="ok_script.py")

    assert result.success is True
    assert "hello" in result.stdout
    assert result.exit_code == 0
    artifact_names = [Path(a).name for a in result.artifacts]
    assert "out.txt" in artifact_names


def test_run_script_failure(tmp_path):
    code = '''
raise RuntimeError("boom")
'''
    result = run_script(code, str(tmp_path), filename="bad_script.py")

    assert result.success is False
    assert result.exit_code != 0
    assert result.stderr.strip() != ""
