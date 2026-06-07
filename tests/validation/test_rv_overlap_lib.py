"""Tests for Phase A overlap inventory and calibration gates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation.rv_overlap_lib import (
    BJD_TO_MJD_OFFSET,
    bjd_to_mjd,
    build_overlap_stars,
    enrich_pairs_with_deltas,
    find_pair_candidates,
    load_literature_epochs,
    summarize_absolute_gate,
    summarize_relative_gate,
)
from validation.rv_phase_a_baseline import run_phase_a
from validation.rv_overlap_lib import PhaseAGoals

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phase_a"


def test_bjd_mjd_roundtrip() -> None:
    bjd = 2459900.5
    assert bjd_to_mjd(bjd) == pytest.approx(bjd - BJD_TO_MJD_OFFSET)


def test_load_literature_mini() -> None:
    df = load_literature_epochs(_FIXTURES / "literature_mini.csv")
    assert len(df) == 3
    assert "mjd" in df.columns
    assert df["gaia_dr3_id"].nunique() == 2


def _synthetic_apf() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1000000000000000001",
                "name": "STAR_A",
                "basename": "Gaia_DR3_1000000000000000001_epoch_1.txt",
                "mjd": bjd_to_mjd(2459900.5),
                "bjd": 2459900.5,
                "rv_kms": 10.2,
                "rv_err_kms": 0.05,
                "epoch_id": "apf:e1",
                "reference_key": "",
                "instrument": "APF",
                "bias_correction_applied": False,
            },
            {
                "gaia_dr3_id": "1000000000000000001",
                "name": "STAR_A",
                "basename": "Gaia_DR3_1000000000000000001_epoch_2.txt",
                "mjd": bjd_to_mjd(2459902.0),
                "bjd": 2459902.0,
                "rv_kms": 10.25,
                "rv_err_kms": 0.05,
                "epoch_id": "apf:e2",
                "reference_key": "",
                "instrument": "APF",
                "bias_correction_applied": False,
            },
            {
                "gaia_dr3_id": "1000000000000000002",
                "name": "STAR_B",
                "basename": "Gaia_DR3_1000000000000000002_epoch_1.txt",
                "mjd": bjd_to_mjd(2459800.0),
                "bjd": 2459800.0,
                "rv_kms": 20.5,
                "rv_err_kms": 0.1,
                "epoch_id": "apf:e3",
                "reference_key": "",
                "instrument": "APF",
                "bias_correction_applied": False,
            },
        ]
    )


def test_pair_candidates_types_and_window() -> None:
    lit = load_literature_epochs(_FIXTURES / "literature_mini.csv")
    apf = _synthetic_apf()
    overlap = build_overlap_stars(lit, apf)
    assert len(overlap) == 2
    pairs = find_pair_candidates(lit, apf, overlap, window_days=7.0)
    pairs = enrich_pairs_with_deltas(pairs)
    types = set(pairs["pair_type"])
    assert "apf_literature" in types
    assert "apf_apf" in types
    assert "literature_literature" in types
    assert (pairs["pair_type"] == "apf_literature").sum() >= 2


def test_absolute_gate_pass_fail() -> None:
    pairs = pd.DataFrame(
        [
            {"pair_type": "apf_literature", "gaia_dr3_id": "1", "delta_rv_kms": 0.5, "abs_delta_rv_kms": 0.5},
            {"pair_type": "apf_literature", "gaia_dr3_id": "1", "delta_rv_kms": 2.0, "abs_delta_rv_kms": 2.0},
        ]
    )
    s = summarize_absolute_gate(pairs, threshold_kms=1.0)
    assert s["n_pairs"] == 2
    assert s["n_pass"] == 1
    assert s["pass_rate"] == pytest.approx(0.5)


def test_relative_gate_summary() -> None:
    pairs = pd.DataFrame(
        [
            {"pair_type": "apf_apf", "gaia_dr3_id": "1", "delta_rv_kms": 0.05, "abs_delta_rv_kms": 0.05},
            {"pair_type": "apf_apf", "gaia_dr3_id": "1", "delta_rv_kms": 0.2, "abs_delta_rv_kms": 0.2},
        ]
    )
    s = summarize_relative_gate(pairs, goal_kms=0.1)
    assert s["n_pairs"] == 2
    assert s["frac_below_goal"] == pytest.approx(0.5)


def test_run_phase_a_synthetic(tmp_path: Path) -> None:
    summary_dir = tmp_path / "output"
    summary_dir.mkdir()
    gid = "1000000000000000001"
    text = (
        f"# Gaia DR3 {gid}\n\n[PIPELINE RESULTS]\n"
        f"# File | MJD | RV | Err | RMS\n"
        f"Gaia_DR3_{gid}_epoch_1.txt {bjd_to_mjd(2459900.5):.4f} 10.2 0.05 0.1\n"
        f"Gaia_DR3_{gid}_epoch_2.txt {bjd_to_mjd(2459902.0):.4f} 10.25 0.05 0.1\n"
    )
    (summary_dir / f"Gaia_DR3_{gid}_summary.txt").write_text(text, encoding="utf-8")

    out = tmp_path / "phase_a_out"
    manifest = run_phase_a(
        master_path=_FIXTURES / "literature_mini.csv",
        summary_dir=summary_dir,
        diagnostics_glob=None,
        out_dir=out,
        goals=PhaseAGoals(pair_window_days=7.0, absolute_gate_kms=1.0, relative_goal_kms=0.1),
        bias_correction_applied=False,
        prefer_diagnostics_rv=False,
        run_id="test",
    )
    assert (out / "overlap_stars.csv").is_file()
    assert (out / "pair_candidates.csv").is_file()
    assert (out / "plots" / "inventory_star_counts.png").is_file()
    assert manifest.inventory["n_overlap_stars"] >= 1
    assert manifest.absolute_gate["n_pairs"] >= 1
