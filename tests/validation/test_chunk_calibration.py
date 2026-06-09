"""Tests for chunk bias application and relative reassessment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.chunk_calibration import (
    build_intrinsic_scatter_model,
    find_apf_apf_pairs,
    load_chunk_bias_tables,
    select_chunks_cdf_weight,
    stack_calibrated_exposure,
    summarize_relative_gate,
    summarize_sigma_rv_metrics,
    _sigma_total_kms,
)


def test_sigma_total_quadrature() -> None:
    assert _sigma_total_kms(0.1, 0.05) == pytest.approx(np.hypot(0.1, 0.05))


def test_stack_calibrated_exposure_uses_spectrum_stat_error() -> None:
    per_object = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1",
                "chunk_key": "10",
                "weighted_mean_residual_kms": 1.0,
                "statistical_err_kms": 0.5,
                "intrinsic_scatter_kms": 0.05,
                "teff": 5500.0,
            },
            {
                "gaia_dr3_id": "1",
                "chunk_key": "11",
                "weighted_mean_residual_kms": 0.5,
                "statistical_err_kms": 0.5,
                "intrinsic_scatter_kms": 0.05,
                "teff": 5500.0,
            },
            {
                "gaia_dr3_id": "1",
                "chunk_key": "12",
                "weighted_mean_residual_kms": 0.0,
                "statistical_err_kms": 0.5,
                "intrinsic_scatter_kms": 0.05,
                "teff": 5500.0,
            },
        ]
    )
    fallback = pd.DataFrame(
        columns=["chunk_key", "bias_kms", "statistical_err_kms", "intrinsic_scatter_kms"]
    )
    chunk_df = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1",
                "chunk_key": "10",
                "rv_kms": 11.0,
                "rv_err_kms": 0.1,
                "chunk_kept": True,
                "teff": 5500.0,
            },
            {
                "gaia_dr3_id": "1",
                "chunk_key": "11",
                "rv_kms": 10.5,
                "rv_err_kms": 0.1,
                "chunk_kept": True,
                "teff": 5500.0,
            },
            {
                "gaia_dr3_id": "1",
                "chunk_key": "12",
                "rv_kms": 10.0,
                "rv_err_kms": 0.1,
                "chunk_kept": True,
                "teff": 5500.0,
            },
        ]
    )
    intrinsic_model = build_intrinsic_scatter_model(per_object)
    out = stack_calibrated_exposure(
        chunk_df,
        per_object=per_object,
        fallback=fallback,
        intrinsic_model=intrinsic_model,
        min_chunks=3,
    )
    assert out["n_chunks_used"] == 3
    assert out["rv_calibrated_kms"] == pytest.approx(10.0, abs=0.05)
    sig = float(np.hypot(0.1, 0.05))
    expected_err = 1.0 / np.sqrt(3 * (1.0 / sig**2))
    assert out["rv_err_calibrated_kms"] == pytest.approx(expected_err, rel=0.01)


def test_cdf_weight_selection_prefers_low_sigma_chunks() -> None:
    sigmas = np.array([0.2, 0.4, 1.0, 2.0])
    weights = 1.0 / sigmas**2
    idx, err = select_chunks_cdf_weight(sigmas, weights, min_chunks=2, weight_fraction=0.9)
    assert len(idx) >= 2
    assert int(idx[0]) == 0  # lowest-σ chunk first
    err_all = 1.0 / np.sqrt(np.sum(weights))
    assert err >= err_all  # 90% weight subset is fewer chunks → larger σ_RV


def test_summarize_sigma_rv_metrics() -> None:
    epochs = pd.DataFrame(
        {
            "rv_err_calibrated_kms": [0.05, 0.08, 0.12],
            "sigma_rv_core90_kms": [0.06, 0.09, 0.15],
        }
    )
    m = summarize_sigma_rv_metrics(epochs)
    assert m["min_sigma_rv_kms"] == pytest.approx(0.05)
    assert m["median_sigma_rv_kms"] == pytest.approx(0.08)
    assert m["p90_sigma_rv_kms"] == pytest.approx(0.112)


def test_find_apf_apf_pairs() -> None:
    apf = pd.DataFrame(
        [
            {"gaia_dr3_id": "1", "name": "A", "file": "e1", "mjd": 60000.0, "rv_kms": 10.0, "rv_err_kms": 0.1},
            {"gaia_dr3_id": "1", "name": "A", "file": "e2", "mjd": 60002.0, "rv_kms": 10.05, "rv_err_kms": 0.1},
        ]
    )
    pairs = find_apf_apf_pairs(apf, pair_window_days=7.0)
    assert len(pairs) == 1
    assert float(pairs["abs_delta_rv_kms"].iloc[0]) == pytest.approx(0.05)


def test_summarize_relative_gate() -> None:
    pairs = pd.DataFrame({"delta_rv_kms": [0.05, 0.2], "abs_delta_rv_kms": [0.05, 0.2], "gaia_dr3_id": ["1", "1"]})
    s = summarize_relative_gate(pairs, goal_kms=0.1)
    assert s["frac_below_goal"] == pytest.approx(0.5)


def test_load_chunk_bias_tables() -> None:
    df = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1",
                "chunk_key": "10",
                "weighted_mean_residual_kms": 0.1,
                "statistical_err_kms": 0.2,
                "intrinsic_scatter_kms": 0.05,
                "sample_kept": True,
            }
        ]
    )
    per_obj, fallback = load_chunk_bias_tables(df)
    assert len(per_obj) == 1
    assert float(fallback.iloc[0]["bias_kms"]) == pytest.approx(0.1)
