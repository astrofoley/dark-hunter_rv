"""Tests for stellar-parameter bias regression per estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation.ccf_estimator_bias import _wide_to_bias_csv
from validation.chunk_bias_regression import choose_best_model, compare_models


def test_compare_models_prefers_stellar_when_teff_trend_present():
    rng = np.random.default_rng(1)
    n = 100
    teff = np.repeat([4200, 4800, 5400, 6000], n // 4)
    order = np.tile(np.arange(10, 20), n // 10)
    bias = 0.001 * (teff - 5000) + 0.05 * (order - order.mean()) / order.std() + rng.normal(0, 0.02, n)
    df = pd.DataFrame(
        {
            "gaia_dr3_id": np.repeat(np.arange(n // 10), 10).astype(str),
            "chunk_key": order.astype(str),
            "chunk_order": order,
            "chunk_order_norm": (order - order.min()) / (order.max() - order.min()),
            "weighted_mean_residual_kms": bias,
            "statistical_err_kms": np.full(n, 0.08),
            "intrinsic_scatter_kms": np.full(n, 0.03),
            "teff": teff,
            "logg": 4.5 + rng.normal(0, 0.1, n),
            "mh": rng.normal(0, 0.1, n),
            "log10_median_mask_ccf_peak_snr": np.full(n, 1.0),
        }
    )
    cmp = compare_models(df)
    chosen = choose_best_model(cmp)
    assert chosen in ("stellar", "curve_stellar", "curve_stellar_interaction")


def test_wide_to_bias_csv_builds_object_table():
    wide = pd.DataFrame(
        [
            {"file": "a.txt", "gaia_dr3_id": "1", "chunk_key": "10_0", "chunk_order": 10,
             "rv_kms__gauss_offset": 10.0, "rv_err_kms__gauss_offset": 0.1, "peak_snr": 12.0},
            {"file": "a.txt", "gaia_dr3_id": "1", "chunk_key": "10_1", "chunk_order": 10,
             "rv_kms__gauss_offset": 12.0, "rv_err_kms__gauss_offset": 0.1, "peak_snr": 11.0},
            {"file": "b.txt", "gaia_dr3_id": "1", "chunk_key": "10_0", "chunk_order": 10,
             "rv_kms__gauss_offset": 10.5, "rv_err_kms__gauss_offset": 0.1, "peak_snr": 10.0},
            {"file": "b.txt", "gaia_dr3_id": "1", "chunk_key": "10_1", "chunk_order": 10,
             "rv_kms__gauss_offset": 12.5, "rv_err_kms__gauss_offset": 0.1, "peak_snr": 9.0},
        ]
    )
    bias = _wide_to_bias_csv(wide, "gauss_offset")
    assert len(bias) >= 1
    assert "weighted_mean_residual_kms" in bias.columns
