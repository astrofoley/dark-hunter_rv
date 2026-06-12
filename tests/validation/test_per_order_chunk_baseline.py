"""Tests for per-order chunk baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation.chunk_adaptive_stack import ChunkMeas
from validation.chunk_layout import build_equal_subchunk_layout, build_merge_orders_layout
from validation.per_order_chunk_baseline import (
    greedy_assign_file,
    enumerate_order_candidates,
    needs_subchunks_8_from_map,
    stack_norm_factor,
    sigma_rv_normalized_kms,
)


def _meas(
    layout: str,
    ck: str,
    rv: float,
    err: float,
    orders: frozenset[int],
    *,
    file: str = "f1",
    gid: str = "1",
) -> ChunkMeas:
    return ChunkMeas(
        layout_name=layout,
        chunk_key=ck,
        rv_kms=rv,
        rv_err_kms=err,
        orders=orders,
        qc_pass=True,
        file=file,
        gaia_dr3_id=gid,
        mjd=60000.0,
        teff=5500.0,
    )


def _fallback_for(rows: list[ChunkMeas]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layout_name": r.layout_name,
                "chunk_key": r.chunk_key,
                "bias_kms": 0.0,
                "statistical_err_kms": r.rv_err_kms,
                "intrinsic_scatter_kms": 0.0,
            }
            for r in rows
        ]
    )


def test_merge_norm_multiplies_by_sqrt_n() -> None:
    m = _meas("merge_w2", "merge_0", 10.0, 0.02, frozenset({10, 11}))
    assert stack_norm_factor([m]) == np.sqrt(2)
    assert sigma_rv_normalized_kms(0.02, [m]) == 0.02 * np.sqrt(2)


def test_subchunks_norm_divides_by_sqrt_n() -> None:
    chunks = [
        _meas("subchunks_4", "10_0", 10.0, 0.1, frozenset({10})),
        _meas("subchunks_4", "10_1", 10.0, 0.1, frozenset({10})),
        _meas("subchunks_4", "10_2", 10.0, 0.1, frozenset({10})),
        _meas("subchunks_4", "10_3", 10.0, 0.1, frozenset({10})),
    ]
    assert stack_norm_factor(chunks) == 0.5
    sig_stack = 0.05
    assert sigma_rv_normalized_kms(sig_stack, chunks) == 0.025


def test_merge_covers_second_order() -> None:
    orders = list(range(10, 18))
    merge = build_merge_orders_layout(merge_width=2, valid_orders=orders)
    s4 = build_equal_subchunk_layout(4)
    layouts = {"merge_w2": merge, "subchunks_4": s4}
    file = "f1"
    rows: list[ChunkMeas] = []
    for o in orders:
        for si in range(4):
            rows.append(_meas("subchunks_4", f"{o}_{si}", 10.0, 0.05, frozenset({o})))
    for gi, (lo, hi) in enumerate(merge.merge_orders or []):
        rows.append(_meas("merge_w2", f"merge_{gi}", 10.0, 0.02, frozenset(range(lo, hi + 1))))
    idx = {(r.file, r.layout_name, r.chunk_key): r for r in rows}
    empty_bias = pd.DataFrame(columns=["gaia_dr3_id", "layout_name", "chunk_key", "weighted_mean_residual_kms"])
    intrinsic = __import__(
        "validation.chunk_calibration", fromlist=["build_intrinsic_scatter_model"]
    ).build_intrinsic_scatter_model(empty_bias)
    fallback = _fallback_for(rows)
    steps, stack = greedy_assign_file(
        file,
        idx,
        layouts,
        per_object=empty_bias,
        fallback=fallback,
        intrinsic_model=intrinsic,
        star_meta={},
        split_ns=(1, 2, 4),
    )
    assert steps
    assert np.isfinite(stack["rv_err_calibrated_kms"])


def test_split_candidates_include_merge_and_subchunks() -> None:
    orders = [10, 11]
    merge = build_merge_orders_layout(merge_width=2, valid_orders=orders)
    s2 = build_equal_subchunk_layout(2)
    layouts = {"merge_w2": merge, "subchunks_2": s2, "subchunks_4": build_equal_subchunk_layout(4)}
    rows = [
        _meas("subchunks_4", "10_0", 1.0, 0.1, frozenset({10})),
        _meas("subchunks_4", "10_1", 1.0, 0.1, frozenset({10})),
        _meas("subchunks_4", "10_2", 1.0, 0.1, frozenset({10})),
        _meas("subchunks_4", "10_3", 1.0, 0.1, frozenset({10})),
        _meas("subchunks_2", "10_0", 1.0, 0.1, frozenset({10})),
        _meas("subchunks_2", "10_1", 1.0, 0.1, frozenset({10})),
        _meas("merge_w2", "merge_0", 1.0, 0.05, frozenset({10, 11})),
    ]
    idx = {(r.file, r.layout_name, r.chunk_key): r for r in rows}
    cands = enumerate_order_candidates("f1", 10, idx, layouts, split_ns=(1, 2))
    names = {c.name for c in cands}
    assert "subchunks_1" in names
    assert "subchunks_2" in names
    assert "merge_w2" in names


def test_needs_subchunks_8_flag() -> None:
    chunk_map = pd.DataFrame([{"order": 15, "choice": "subchunks_4"}, {"order": 16, "choice": "subchunks_2"}])
    assert needs_subchunks_8_from_map(chunk_map, has_subchunks_8=False) is True
    assert needs_subchunks_8_from_map(chunk_map, has_subchunks_8=True) is False
    chunk_map2 = pd.DataFrame([{"order": 15, "choice": "subchunks_2"}])
    assert needs_subchunks_8_from_map(chunk_map2, has_subchunks_8=False) is False
