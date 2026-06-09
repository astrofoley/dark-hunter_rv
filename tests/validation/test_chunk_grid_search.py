"""Tests for chunk layout grid builders."""
from __future__ import annotations

import pytest

from validation.chunk_grid_search import composite_score
from validation.chunk_layout import (
    build_equal_subchunk_layout,
    build_merge_orders_layout,
    default_parametric_grid,
)


def test_build_merge_orders_layout_width_2() -> None:
    lay = build_merge_orders_layout(merge_width=2, valid_orders=[3, 4, 5, 6])
    assert lay.merge_orders == [[3, 4], [5, 6]]


def test_default_parametric_grid_includes_merge_and_subchunk() -> None:
    layouts = default_parametric_grid(subchunk_counts=[1, 2], merge_widths=[1, 2])
    names = {lay.name for lay in layouts}
    assert "subchunks_2" in names
    assert "merge_w2" in names


def test_composite_score_offline_valid() -> None:
    import pandas as pd

    row = pd.Series(
        {
            "offline_eval_valid": True,
            "median_sigma_rv_kms": 0.05,
            "p90_sigma_rv_kms": 0.08,
            "relative_median_abs_delta_kms": 0.1,
        }
    )
    assert composite_score(row) == pytest.approx(0.35 * 0.05 + 0.35 * 0.08 + 0.30 * 0.1)


def test_composite_score_requires_both_metrics() -> None:
    import math
    import pandas as pd

    row = pd.Series({"offline_eval_valid": True, "relative_median_abs_delta_kms": 0.1})
    assert math.isinf(composite_score(row))
