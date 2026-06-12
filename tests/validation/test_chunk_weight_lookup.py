"""Tests for chunk weight lookup merge semantics."""
from __future__ import annotations

import pandas as pd

from validation.chunk_weight_lookup import (
    LOOKUP_COLUMNS,
    build_lookup_from_fallback,
    merge_lookup,
)


def _fallback(layout: str, ck: str, bias: float = 0.1, stat: float = 0.2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layout_name": layout,
                "chunk_key": ck,
                "bias_kms": bias,
                "statistical_err_kms": stat,
                "intrinsic_scatter_kms": 0.05,
            }
        ]
    )


def test_merge_preserves_other_layouts() -> None:
    a = build_lookup_from_fallback(_fallback("subchunks_4", "15_0"), layout_name="subchunks_4")
    b = build_lookup_from_fallback(_fallback("merge_w4", "merge_0"), layout_name="merge_w4")
    merged = merge_lookup(a, b)
    assert len(merged) == 2
    assert set(merged["layout_name"]) == {"subchunks_4", "merge_w4"}


def test_re_run_updates_only_touched_layout() -> None:
    v1 = build_lookup_from_fallback(_fallback("subchunks_4", "15_0", bias=0.1), layout_name="subchunks_4")
    v1 = merge_lookup(
        v1,
        build_lookup_from_fallback(_fallback("merge_w4", "merge_0"), layout_name="merge_w4"),
    )
    v2_piece = build_lookup_from_fallback(_fallback("subchunks_4", "15_0", bias=0.2), layout_name="subchunks_4")
    v2 = merge_lookup(v1, v2_piece)
    sub = v2[v2["layout_name"] == "subchunks_4"]
    merge = v2[v2["layout_name"] == "merge_w4"]
    assert float(sub.iloc[0]["bias_kms"]) == 0.2
    assert len(merge) == 1
