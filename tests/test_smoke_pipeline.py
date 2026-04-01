import os
import subprocess
import sys
from pathlib import Path


def test_smoke_pipeline_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    fixture = repo_root / "tests" / "fixtures" / "mini_apf.txt"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)

    cmd = [
        sys.executable,
        "-m",
        "darkhunter_rv.pipeline",
        str(fixture),
        "--instrument",
        "APF",
        "--teff",
        "5500",
        "--no-bias",
        "--log-level",
        "ERROR",
    ]
    p = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert p.returncode == 0
