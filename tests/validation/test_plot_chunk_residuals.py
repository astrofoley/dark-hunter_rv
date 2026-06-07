"""Tests for chunk residual plotting helpers."""
from __future__ import annotations

import numpy as np
import pytest

from validation.plot_chunk_residuals import (
    _chunk_sort_key,
    _ordered_chunks,
    _summarize_chunks_per_object,
    _weighted_mean_and_errors,
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
