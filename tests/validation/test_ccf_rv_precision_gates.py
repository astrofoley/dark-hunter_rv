"""Tests for CCF RV precision phase gates."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.ccf_rv_precision_gates import check_phase_gate, load_baseline, save_baseline


def test_gate_passes_when_metrics_improve(tmp_path: Path) -> None:
    baseline_path = tmp_path / "ref.json"
    save_baseline(
        baseline_path,
        phase="A",
        metrics={
            "median_sigma_rv_kms": 0.35,
            "p90_sigma_rv_kms": 0.55,
            "median_chunk_scatter_kms": 0.28,
            "bias_curve_rms_kms": 0.12,
            "stellar_bias_cv_rmse_kms": 0.15,
            "low_snr_finite_rate": 0.75,
        },
    )
    prior = load_baseline(baseline_path)
    new_metrics = {
        "median_sigma_rv_kms": 0.33,
        "p90_sigma_rv_kms": 0.52,
        "median_chunk_scatter_kms": 0.26,
        "bias_curve_rms_kms": 0.11,
        "stellar_bias_cv_rmse_kms": 0.14,
        "low_snr_finite_rate": 0.78,
    }
    result = check_phase_gate("C", new_metrics, prior, strict=True)
    assert result["passed"] is True
    assert not result["failures"]


def test_gate_fails_on_sigma_regression(tmp_path: Path) -> None:
    baseline_path = tmp_path / "ref.json"
    save_baseline(
        baseline_path,
        phase="A",
        metrics={"median_sigma_rv_kms": 0.30, "low_snr_finite_rate": 0.80},
    )
    prior = load_baseline(baseline_path)
    result = check_phase_gate(
        "C",
        {"median_sigma_rv_kms": 0.32, "low_snr_finite_rate": 0.80},
        prior,
        strict=True,
    )
    assert result["passed"] is False
    assert any("median_sigma_rv_kms" in f for f in result["failures"])


def test_gate_fails_on_coverage_drop(tmp_path: Path) -> None:
    baseline_path = tmp_path / "ref.json"
    save_baseline(baseline_path, phase="A", metrics={"low_snr_finite_rate": 0.80})
    prior = load_baseline(baseline_path)
    result = check_phase_gate("C", {"low_snr_finite_rate": 0.70}, prior, strict=True)
    assert result["passed"] is False
