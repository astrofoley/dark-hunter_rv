"""Mask-only pipeline mode (stellar mask chunk RVs, no PHOENIX bank)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_mask_only_pipeline_runs_on_fixture(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    fixture = repo_root / "tests" / "fixtures" / "mini_apf.txt"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env["DARKHUNTER_OUTPUT_DIR"] = str(tmp_path)

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
        "--mask-only",
        "--log-level",
        "ERROR",
    ]
    p = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    stem = fixture.stem
    assert (tmp_path / f"{stem}_orders.txt").is_file()
