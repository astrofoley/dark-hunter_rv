"""Tests for exhaustive spectrum tiling search."""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation.chunk_adaptive_stack import ChunkMeas
from validation.chunk_calibration import lookup_chunk_bias, stack_calibrated_exposure
from validation.chunk_layout import build_equal_subchunk_layout, build_merge_orders_layout
from validation.spectrum_tiling import (
    TileRegistry,
    count_complete_tilings,
    file_meas_index,
    find_best_tiling_for_file,
    full_order_region,
    register_custom_tile_layout,
    stack_tiling_pipeline,
    TileCandidate,
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


def _registry(
    rows: list[ChunkMeas],
    layouts: dict,
    *,
    file: str = "f1",
    mix_layout_names: list[str] | None = None,
) -> TileRegistry:
    global_idx = {(r.file, r.layout_name, r.chunk_key): r for r in rows}
    fidx = file_meas_index(global_idx, file)
    return TileRegistry.for_file(
        file,
        layouts,
        fidx,
        mix_layout_names=mix_layout_names,
    )


def _bias_tables(rows: list[ChunkMeas]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_object = pd.DataFrame(
        [
            {
                "gaia_dr3_id": r.gaia_dr3_id,
                "layout_name": r.layout_name,
                "chunk_key": r.chunk_key,
                "weighted_mean_residual_kms": 0.0,
                "statistical_err_kms": r.rv_err_kms,
                "intrinsic_scatter_kms": 0.0,
                "teff": r.teff,
            }
            for r in rows
        ]
    )
    fallback = pd.DataFrame(
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
    intrinsic = __import__(
        "validation.chunk_calibration", fromlist=["build_intrinsic_scatter_model"]
    ).build_intrinsic_scatter_model(per_object)
    return per_object, fallback, intrinsic


def test_layout_aware_bias_lookup() -> None:
    per_object = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1",
                "layout_name": "subchunks_4",
                "chunk_key": "10_0",
                "weighted_mean_residual_kms": 1.0,
            },
            {
                "gaia_dr3_id": "1",
                "layout_name": "subchunks_2",
                "chunk_key": "10_0",
                "weighted_mean_residual_kms": 2.0,
            },
        ]
    )
    fallback = pd.DataFrame(columns=["layout_name", "chunk_key", "bias_kms"])
    b4, _ = lookup_chunk_bias("1", "10_0", per_object, fallback, layout_name="subchunks_4")
    b2, _ = lookup_chunk_bias("1", "10_0", per_object, fallback, layout_name="subchunks_2")
    assert b4 == 1.0
    assert b2 == 2.0


def test_count_tilings_two_orders() -> None:
    orders = [10, 11]
    merge = build_merge_orders_layout(merge_width=2, valid_orders=orders)
    s2 = build_equal_subchunk_layout(2)
    s4 = build_equal_subchunk_layout(4)
    layouts = {"merge_w2": merge, "subchunks_2": s2, "subchunks_4": s4}
    rows: list[ChunkMeas] = []
    for o in orders:
        for si in range(4):
            rows.append(_meas("subchunks_4", f"{o}_{si}", 10.0, 0.04, frozenset({o})))
        for si in range(2):
            rows.append(_meas("subchunks_2", f"{o}_{si}", 10.0, 0.06, frozenset({o})))
    rows.append(_meas("merge_w2", "merge_0", 10.0, 0.02, frozenset({10, 11})))
    registry = _registry(rows, layouts)
    assert count_complete_tilings(registry) == 5


def test_find_best_prefers_merge_when_tighter() -> None:
    orders = list(range(10, 16))
    merge = build_merge_orders_layout(merge_width=2, valid_orders=orders)
    s4 = build_equal_subchunk_layout(4)
    layouts = {"merge_w2": merge, "subchunks_4": s4}
    rows: list[ChunkMeas] = []
    for o in orders:
        for si in range(4):
            rows.append(_meas("subchunks_4", f"{o}_{si}", 10.0, 0.08, frozenset({o})))
    for gi, grp in enumerate(merge.merge_orders or []):
        lo, hi = int(grp[0]), int(grp[1])
        covered = frozenset(o for o in range(lo, hi + 1) if o in orders)
        if covered:
            rows.append(_meas("merge_w2", f"merge_{gi}", 10.0, 0.02, covered))
    registry = _registry(rows, layouts)
    per_object, fallback, intrinsic = _bias_tables(rows)
    result = find_best_tiling_for_file(
        registry,
        per_object=per_object,
        fallback=fallback,
        intrinsic_model=intrinsic,
        star_meta={},
        include_whole_layouts=False,
    )
    assert result is not None
    assert result.tiling_name.startswith("mix:") and "merge_w2" in result.tiling_name
    assert result.stack["rv_err_calibrated_kms"] < 0.04
    assert int(result.stack["n_chunks_used"]) >= 3


def test_custom_tile_provider_registers_partial_order_tiles() -> None:
    from validation.spectrum_tiling import SpectrumRegion

    class _HybridProvider:
        def __init__(self, _layout: object, ctx: dict) -> None:
            self.file_idx = ctx["file_idx"]

        def tiles_for_anchor(
            self,
            anchor,
            *,
            uncovered: tuple,
        ) -> list[TileCandidate]:
            out: list[TileCandidate] = []
            if anchor.order == 10:
                m = self.file_idx.get(("hybrid", "10+11a"))
                if m is not None:
                    out.append(
                        TileCandidate(
                            name="hybrid:10+11a",
                            layout_name="hybrid",
                            measurements=(m,),
                            regions=frozenset(
                                {full_order_region(10), SpectrumRegion(11, 0.0, 0.5)}
                            ),
                        )
                    )
            if anchor.order == 11 and anchor.pixel_lo >= 0.5:
                parts = [
                    self.file_idx.get(("hybrid", "11b_0")),
                    self.file_idx.get(("hybrid", "11b_1")),
                ]
                if all(p is not None for p in parts):
                    out.append(
                        TileCandidate(
                            name="hybrid:11b_split",
                            layout_name="hybrid",
                            measurements=tuple(parts),  # type: ignore[arg-type]
                            regions=frozenset({SpectrumRegion(11, 0.5, 1.0)}),
                        )
                    )
            return out

    register_custom_tile_layout("hybrid", lambda lay, ctx: _HybridProvider(lay, ctx))

    class ChunkLayoutStub:
        name = "hybrid"
        merge_orders = None

        def n_chunks_per_order(self) -> int:
            return 1

    rows = [
        _meas("hybrid", "10+11a", 10.0, 0.02, frozenset({10, 11})),
        _meas("hybrid", "11b_0", 10.0, 0.04, frozenset({11})),
        _meas("hybrid", "11b_1", 10.0, 0.04, frozenset({11})),
    ]
    registry = _registry(rows, {"hybrid": ChunkLayoutStub()}, mix_layout_names=["hybrid"])
    assert count_complete_tilings(registry) == 1
    tiles = registry._tiles_for_anchor_uncovered(
        full_order_region(10),
        [full_order_region(10), SpectrumRegion(11, 0.0, 1.0)],
    )
    assert any(t.name == "hybrid:10+11a" for t in tiles)


def test_stack_tiling_pipeline_matches_stack_calibrated_exposure() -> None:
    rows = [
        _meas("subchunks_4", "10_0", 10.0, 0.1, frozenset({10})),
        _meas("subchunks_4", "10_1", 10.1, 0.1, frozenset({10})),
        _meas("subchunks_4", "11_0", 10.0, 0.1, frozenset({11})),
    ]
    per_object, fallback, intrinsic = _bias_tables(rows)
    out = stack_tiling_pipeline(
        rows,
        per_object=per_object,
        fallback=fallback,
        intrinsic_model=intrinsic,
        star_meta={},
        min_chunks=3,
    )
    chunk_df = pd.DataFrame(
        [
            {
                "gaia_dr3_id": "1",
                "layout_name": r.layout_name,
                "chunk_key": r.chunk_key,
                "rv_kms": r.rv_kms,
                "rv_err_kms": r.rv_err_kms,
                "chunk_kept": True,
                "teff": 5500.0,
            }
            for r in rows
        ]
    )
    direct = stack_calibrated_exposure(
        chunk_df,
        per_object=per_object,
        fallback=fallback,
        intrinsic_model=intrinsic,
        min_chunks=3,
    )
    assert out["rv_err_calibrated_kms"] == direct["rv_err_calibrated_kms"]
    assert out["sigma_rv_core90_kms"] == direct["sigma_rv_core90_kms"]
