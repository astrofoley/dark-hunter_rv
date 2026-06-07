"""Tests for chunk residual plotting helpers."""
from __future__ import annotations

import numpy as np
import pytest

from validation.plot_chunk_residuals import (
    _chunk_sort_key,
    _ordered_chunks,
    _summarize_chunks_per_object,
    _weighted_mean_and_errors,
    apply_spectrum_chunk_outlier_clip,
    iterative_loo_sigma_clip_mask,
)


def test_chunk_sort_key() -> None:
    assert _chunk_sort_key("42") < _chunk_sort_key("43")
    assert _chunk_sort_key("42_1") < _chunk_sort_key("42_2")


def test_ordered_chunks() -> None:
    assert _ordered_chunks(["10", "2", "2_1"]) == ["2", "2_1", "10"]


def test_weighted_mean_and_errors() -> None:
    mu, stat, intrinsic = _weighted_mean_and_errors(
        np.array([0.1, 0.2, 0.0]),
        np.array([0.05, 0.05, 0.05]),
    )
    assert mu == pytest.approx(0.1, abs=0.05)
    assert stat > 0
    assert intrinsic >= 0


def test_iterative_loo_sigma_clip_removes_outlier() -> None:
    rv = np.array([10.0, 10.1, 10.05, 50.0])
    keep = iterative_loo_sigma_clip_mask(rv, nsigma=10.0)
    assert not keep[3]
    assert keep[:3].all()


def test_apply_spectrum_chunk_outlier_clip() -> None:
    import pandas as pd

    df = pd.DataFrame(
        [
            {"file": "a", "rv_kms": 10.0, "rv_err_kms": 0.05, "exposure_rv_kms": 10.0, "residual_kms": 0.0},
            {"file": "a", "rv_kms": 10.1, "rv_err_kms": 0.05, "exposure_rv_kms": 10.0, "residual_kms": 0.1},
            {"file": "a", "rv_kms": 50.0, "rv_err_kms": 0.05, "exposure_rv_kms": 10.0, "residual_kms": 40.0},
        ]
    )
    out = apply_spectrum_chunk_outlier_clip(df, nsigma=10.0)
    assert not bool(out.loc[2, "chunk_kept"])
    kept = out[out["chunk_kept"]]
    assert len(kept) == 2
    assert kept["residual_kms"].abs().max() < 0.2


def test_summarize_chunks_per_object() -> None:
    import pandas as pd

    df = pd.DataFrame(
        [
            {"chunk_key": "1", "file": "a", "residual_kms": 0.1, "rv_err_kms": 0.05},
            {"chunk_key": "1", "file": "b", "residual_kms": 0.3, "rv_err_kms": 0.05},
            {"chunk_key": "2", "file": "a", "residual_kms": -0.2, "rv_err_kms": 0.05},
        ]
    )
    s = _summarize_chunks_per_object(df)
    assert len(s) == 2
    assert "weighted_mean_residual_kms" in s.columns
