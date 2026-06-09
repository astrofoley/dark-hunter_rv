"""Tests for adaptive multi-layout chunk stacking."""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation.chunk_adaptive_stack import (
    ChunkMeas,
    adaptive_stack_for_file,
    enumerate_stack_candidates,
    select_best_candidate,
)
from validation.chunk_layout import build_equal_subchunk_layout, build_merge_orders_layout


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


def test_whole_layout_candidate_present() -> None:
    coarse = build_merge_orders_layout(merge_width=2, valid_orders=[10, 11, 12, 13, 14, 15, 16, 17])
    s3 = build_equal_subchunk_layout(3)
    layouts = {"merge_w2": coarse, "subchunks_3": s3}
    file = "f1"
    rows = [
        _meas("merge_w2", "merge_0", 10.0, 0.25, frozenset({10, 11})),
        _meas("merge_w2", "merge_1", 10.0, 0.25, frozenset({12, 13})),
        _meas("merge_w2", "merge_2", 10.0, 0.25, frozenset({14, 15})),
        _meas("merge_w2", "merge_3", 10.0, 0.25, frozenset({16, 17})),
    ]
    for o in (10, 11, 12, 13, 14, 15, 16, 17):
        for si in range(3):
            rows.append(_meas("subchunks_3", f"{o}_{si}", 10.0, 0.05, frozenset({o})))
    df = pd.DataFrame([r.__dict__ | {"orders": r.orders} for r in rows])
    idx = {(r.file, r.layout_name, r.chunk_key): r for r in rows}
    empty_bias = pd.DataFrame(columns=["gaia_dr3_id", "layout_name", "chunk_key", "weighted_mean_residual_kms"])
    intrinsic = __import__(
        "validation.chunk_calibration", fromlist=["build_intrinsic_scatter_model"]
    ).build_intrinsic_scatter_model(empty_bias)
    fallback = _fallback_for(rows)
    cands = enumerate_stack_candidates(
        file,
        idx,
        layouts=layouts,
        per_object=empty_bias,
        fallback=fallback,
        intrinsic_model=intrinsic,
        star_meta={},
    )
    names = {c.name for c in cands}
    assert "whole:subchunks_3" in names
    assert "whole:merge_w2" in names


def test_adaptive_matches_best_whole_layout() -> None:
    s3 = build_equal_subchunk_layout(3)
    s4 = build_equal_subchunk_layout(4)
    layouts = {"subchunks_3": s3, "subchunks_4": s4}
    file = "f1"
    rows = []
    for o in (10, 11, 12):
        for si in range(3):
            rows.append(_meas("subchunks_3", f"{o}_{si}", 10.0, 0.04, frozenset({o})))
        for si in range(4):
            rows.append(_meas("subchunks_4", f"{o}_{si}", 10.0, 0.06, frozenset({o})))
    df = pd.DataFrame([r.__dict__ | {"orders": r.orders} for r in rows])
    empty_bias = pd.DataFrame(columns=["gaia_dr3_id", "layout_name", "chunk_key", "weighted_mean_residual_kms"])
    fallback = _fallback_for(rows)
    intrinsic = __import__(
        "validation.chunk_calibration", fromlist=["build_intrinsic_scatter_model"]
    ).build_intrinsic_scatter_model(empty_bias)
    final, _err, info = adaptive_stack_for_file(
        file,
        df,
        layouts=layouts,
        per_object=empty_bias,
        fallback=fallback,
        intrinsic_model=intrinsic,
        star_meta={},
    )
    assert info["candidate_name"] == "whole:subchunks_3"
    assert len(final) == 9
    assert all(m.layout_name == "subchunks_3" for m in final)


def test_select_best_never_worse_than_whole_candidates() -> None:
    s3 = build_equal_subchunk_layout(3)
    layouts = {"subchunks_3": s3}
    file = "f1"
    rows = []
    for o in (10, 11, 12):
        for si in range(3):
            rows.append(_meas("subchunks_3", f"{o}_{si}", 10.0, 0.05, frozenset({o})))
    idx = {(r.file, r.layout_name, r.chunk_key): r for r in rows}
    empty_bias = pd.DataFrame(columns=["gaia_dr3_id", "layout_name", "chunk_key", "weighted_mean_residual_kms"])
    fallback = _fallback_for(rows)
    intrinsic = __import__(
        "validation.chunk_calibration", fromlist=["build_intrinsic_scatter_model"]
    ).build_intrinsic_scatter_model(empty_bias)
    cands = enumerate_stack_candidates(
        file, idx, layouts=layouts, per_object=empty_bias, fallback=fallback,
        intrinsic_model=intrinsic, star_meta={},
    )
    best, best_out = select_best_candidate(
        cands, per_object=empty_bias, fallback=fallback, intrinsic_model=intrinsic, star_meta={},
    )
    assert best is not None
    for cand in cands:
        if not cand.name.startswith("whole:"):
            continue
        out = __import__(
            "validation.chunk_adaptive_stack", fromlist=["_stack_result_for_meas"]
        )._stack_result_for_meas(
            cand.chunks, per_object=empty_bias, fallback=fallback, intrinsic_model=intrinsic, star_meta={},
        )
        assert float(best_out["rv_err_calibrated_kms"]) <= float(out["rv_err_calibrated_kms"]) + 1e-9
