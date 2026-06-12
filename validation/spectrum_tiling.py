"""
Exhaustive search over valid full-spectrum chunk tilings from campaign cache.

Memory model: one exposure at a time, single DFS pass, backtracking mutable state
(no duplicate count pass, no recursive tuple chains). Uses file-scoped measurement
indexes instead of scanning the full campaign index.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python3 -m validation.spectrum_tiling_search \\
    --campaign-dir validation_output/chunk_campaign
"""
from __future__ import annotations

import gc
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from validation.chunk_adaptive_stack import (
    ChunkMeas,
    _index_by_file_layout,
    build_multi_layout_bias_tables,
    load_campaign_measurements,
    load_layouts,
)
from validation.chunk_calibration import (
    stack_calibrated_exposure,
    summarize_sigma_rv_metrics,
)
from validation.chunk_layout import ChunkLayout, apf_valid_orders, merge_order_groups

logger = logging.getLogger(__name__)

MIN_CHUNKS = 3
DEFAULT_METRIC = "rv_err_calibrated_kms"

# Type alias: per-file measurement index (layout_name, chunk_key) -> ChunkMeas
FileMeasIndex = dict[tuple[str, str], ChunkMeas]


@dataclass(frozen=True)
class SpectrumRegion:
    """
    Fraction of one echelle order covered by a tile.

    Full-order tiles use the default ``pixel_lo=0``, ``pixel_hi=1``. Future layouts
    (e.g. merge order 10 + first half of order 11) register tiles with narrower spans.
    """

    order: int
    pixel_lo: float = 0.0
    pixel_hi: float = 1.0

    def is_full_order(self) -> bool:
        return self.pixel_lo == 0.0 and self.pixel_hi == 1.0

    def overlaps(self, other: SpectrumRegion) -> bool:
        if self.order != other.order:
            return False
        return self.pixel_lo < other.pixel_hi and other.pixel_lo < self.pixel_hi


def full_order_region(order: int) -> SpectrumRegion:
    return SpectrumRegion(order=int(order))


def _region_contained(inner: SpectrumRegion, outer: SpectrumRegion) -> bool:
    return (
        inner.order == outer.order
        and inner.pixel_lo >= outer.pixel_lo
        and inner.pixel_hi <= outer.pixel_hi
    )


def _tile_fits_uncovered(tile: TileCandidate, uncovered: list[SpectrumRegion]) -> bool:
    return all(
        any(_region_contained(tr, ur) for ur in uncovered) for tr in tile.regions
    )


def _subtract_regions_list(
    uncovered: list[SpectrumRegion],
    removed: frozenset[SpectrumRegion],
) -> list[SpectrumRegion]:
    """Return a new uncovered list with ``removed`` spans subtracted."""
    out: list[SpectrumRegion] = []
    for u in uncovered:
        remaining: list[SpectrumRegion] = [u]
        for r in removed:
            if u.order != r.order:
                continue
            new_remaining: list[SpectrumRegion] = []
            for part in remaining:
                if not part.overlaps(r):
                    new_remaining.append(part)
                    continue
                if r.pixel_lo <= part.pixel_lo and r.pixel_hi >= part.pixel_hi:
                    continue
                if r.pixel_lo > part.pixel_lo:
                    new_remaining.append(
                        SpectrumRegion(part.order, part.pixel_lo, r.pixel_lo)
                    )
                if r.pixel_hi < part.pixel_hi:
                    new_remaining.append(
                        SpectrumRegion(part.order, r.pixel_hi, part.pixel_hi)
                    )
            remaining = new_remaining
        out.extend(remaining)
    return out


def _anchor_index(uncovered: list[SpectrumRegion]) -> int:
    return min(range(len(uncovered)), key=lambda i: (uncovered[i].order, uncovered[i].pixel_lo))


def _sig_key(meas_keys: set[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(meas_keys))


@dataclass(frozen=True)
class TileCandidate:
    """One installable tile: cached measurements covering a set of spectrum regions."""

    name: str
    layout_name: str
    measurements: tuple[ChunkMeas, ...]
    regions: frozenset[SpectrumRegion]

    @property
    def meas_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset((m.layout_name, m.chunk_key) for m in self.measurements)


class CustomTileProvider(Protocol):
    """Hook for layouts that do not map to per-order or merge-group tiles."""

    def tiles_for_anchor(
        self,
        anchor: SpectrumRegion,
        *,
        uncovered: tuple[SpectrumRegion, ...],
    ) -> list[TileCandidate]:
        ...


CustomTileProviderFactory = Callable[[ChunkLayout, dict], CustomTileProvider]

_CUSTOM_TILE_FACTORIES: dict[str, CustomTileProviderFactory] = {}


def register_custom_tile_layout(
    layout_name: str,
    factory: CustomTileProviderFactory,
) -> None:
    """Register a provider for exotic layouts (partial-order merges, etc.)."""
    _CUSTOM_TILE_FACTORIES[str(layout_name)] = factory


def file_meas_index(
    global_idx: dict[tuple[str, str, str], ChunkMeas],
    file: str,
) -> FileMeasIndex:
    """Slice campaign index to one exposure (no file key in stored tuples)."""
    return {
        (lay, ck): m
        for (f, lay, ck), m in global_idx.items()
        if f == file
    }


def _layout_has_cache_data(layout_name: str, file_idx: FileMeasIndex) -> bool:
    return any(lay == layout_name for (lay, _ck) in file_idx)


def _layout_kind(layout: ChunkLayout) -> str:
    if layout.name in _CUSTOM_TILE_FACTORIES:
        return "custom"
    if layout.merge_orders:
        return "merge_groups"
    return "per_order_partition"


def _orders_in_merge_group(gi: int, layout: ChunkLayout) -> frozenset[int]:
    if not layout.merge_orders or gi < 0 or gi >= len(layout.merge_orders):
        return frozenset()
    lo, hi = layout.merge_orders[gi]
    valid = set(apf_valid_orders())
    return frozenset(o for o in range(int(lo), int(hi) + 1) if o in valid)


def _order_partition_chunks(
    order: int,
    layout_name: str,
    layout: ChunkLayout,
    file_idx: FileMeasIndex,
) -> list[ChunkMeas] | None:
    n = layout.n_chunks_per_order()
    chunks: list[ChunkMeas] = []
    for si in range(n):
        ck = f"{order}_{si}" if n > 1 else str(order)
        m = file_idx.get((layout_name, ck))
        if m is None:
            return None
        chunks.append(m)
    return chunks


def _merge_group_tile(
    order: int,
    layout_name: str,
    layout: ChunkLayout,
    file_idx: FileMeasIndex,
) -> TileCandidate | None:
    gi = merge_order_groups(order, layout.merge_orders)
    if gi is None:
        return None
    m = file_idx.get((layout_name, f"merge_{gi}"))
    if m is None:
        return None
    covered_orders = _orders_in_merge_group(int(gi), layout)
    regions = frozenset(full_order_region(o) for o in covered_orders)
    return TileCandidate(
        name=f"{layout_name}:merge_{gi}",
        layout_name=layout_name,
        measurements=(m,),
        regions=regions,
    )


def _per_order_partition_tile(
    order: int,
    layout_name: str,
    layout: ChunkLayout,
    file_idx: FileMeasIndex,
) -> TileCandidate | None:
    chunks = _order_partition_chunks(order, layout_name, layout, file_idx)
    if chunks is None:
        return None
    return TileCandidate(
        name=f"{layout_name}:order_{order}",
        layout_name=layout_name,
        measurements=tuple(chunks),
        regions=frozenset({full_order_region(order)}),
    )


@dataclass
class TileRegistry:
    """
    Tile discovery for one exposure.

    Uses a file-scoped ``file_idx`` so searches never scan the full campaign cache.
    """

    file: str
    layouts: dict[str, ChunkLayout]
    file_idx: FileMeasIndex
    mix_layout_names: tuple[str, ...] = field(default_factory=tuple)
    whole_layout_names: tuple[str, ...] = field(default_factory=tuple)
    custom_providers: dict[str, CustomTileProvider] = field(default_factory=dict)
    _order_tile_cache: dict[int, tuple[TileCandidate, ...]] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def for_file(
        cls,
        file: str,
        layouts: dict[str, ChunkLayout],
        file_idx: FileMeasIndex,
        *,
        mix_layout_names: list[str] | None = None,
        whole_layout_names: list[str] | None = None,
    ) -> TileRegistry:
        if mix_layout_names is None:
            mix_layout_names = sorted(
                name
                for name, lay in layouts.items()
                if _layout_kind(lay) in ("per_order_partition", "merge_groups", "custom")
                and _layout_has_cache_data(name, file_idx)
            )
        if whole_layout_names is None:
            whole_layout_names = sorted(
                name for name in layouts if _layout_has_cache_data(name, file_idx)
            )

        custom: dict[str, CustomTileProvider] = {}
        for name in mix_layout_names:
            if name in _CUSTOM_TILE_FACTORIES and name in layouts:
                custom[name] = _CUSTOM_TILE_FACTORIES[name](layouts[name], {"file_idx": file_idx})

        return cls(
            file=file,
            layouts=layouts,
            file_idx=file_idx,
            mix_layout_names=tuple(mix_layout_names),
            whole_layout_names=tuple(whole_layout_names),
            custom_providers=custom,
        )

    def spectrum_regions(self) -> list[SpectrumRegion]:
        from validation.per_order_chunk_baseline import _orders_with_split_data

        global_idx = {
            (self.file, lay, ck): m for (lay, ck), m in self.file_idx.items()
        }
        orders = _orders_with_split_data(self.file, global_idx, self.layouts)
        if not orders and self.custom_providers:
            extra: set[int] = set()
            for (_lay, _ck), m in self.file_idx.items():
                if m.layout_name in self.custom_providers:
                    extra |= {int(o) for o in m.orders}
            orders = sorted(extra)
        return [full_order_region(o) for o in orders]

    def _tiles_for_anchor_uncovered(
        self,
        anchor: SpectrumRegion,
        uncovered: list[SpectrumRegion],
    ) -> list[TileCandidate]:
        uncovered_t = tuple(uncovered)
        out: list[TileCandidate] = []
        seen: set[frozenset[tuple[str, str]]] = set()
        order = anchor.order

        for layout_name in self.mix_layout_names:
            layout = self.layouts.get(layout_name)
            if layout is None:
                continue
            kind = _layout_kind(layout)
            if kind == "custom" and layout_name in self.custom_providers:
                for tile in self.custom_providers[layout_name].tiles_for_anchor(
                    anchor, uncovered=uncovered_t
                ):
                    if not _tile_fits_uncovered(tile, uncovered):
                        continue
                    if tile.meas_keys in seen:
                        continue
                    seen.add(tile.meas_keys)
                    out.append(tile)
                continue
            cand = (
                _merge_group_tile(order, layout_name, layout, self.file_idx)
                if kind == "merge_groups"
                else _per_order_partition_tile(order, layout_name, layout, self.file_idx)
            )
            if cand is None or not _tile_fits_uncovered(cand, uncovered):
                continue
            if cand.meas_keys in seen:
                continue
            seen.add(cand.meas_keys)
            out.append(cand)
        return out

    def candidate_tiles_at_anchor(self, anchor: SpectrumRegion) -> tuple[TileCandidate, ...]:
        """Tiles for ``anchor`` when all splittable orders are still uncovered (cached)."""
        order = anchor.order
        cached = self._order_tile_cache.get(order)
        if cached is not None:
            return cached
        regions = self.spectrum_regions()
        if not regions:
            cached = ()
        else:
            cached = tuple(
                self._tiles_for_anchor_uncovered(anchor, regions)
            )
        self._order_tile_cache[order] = cached
        return cached

    def whole_layout_tile(self, layout_name: str) -> TileCandidate | None:
        chunks = [m for (lay, _ck), m in self.file_idx.items() if lay == layout_name]
        if len(chunks) < MIN_CHUNKS:
            return None
        regions: set[SpectrumRegion] = set()
        for m in chunks:
            for o in m.orders:
                regions.add(full_order_region(int(o)))
        return TileCandidate(
            name=f"whole:{layout_name}",
            layout_name=layout_name,
            measurements=tuple(sorted(chunks, key=lambda m: m.chunk_key)),
            regions=frozenset(regions),
        )


def count_complete_tilings(registry: TileRegistry, *, max_count: int | None = None) -> int:
    """Count distinct complete mixed tilings (single DFS pass, compact dedup keys)."""
    mixed_count, _evaluated, _best, _trunc = _search_mixed_tilings(
        registry,
        per_object=pd.DataFrame(),
        fallback=pd.DataFrame(),
        intrinsic_model=None,
        star_meta={},
        metric=DEFAULT_METRIC,
        max_tilings=max_count,
        include_whole_layouts=False,
        min_chunks=MIN_CHUNKS,
        count_only=True,
    )
    return mixed_count


def iterate_complete_tilings(
    registry: TileRegistry,
) -> Iterator[tuple[tuple[tuple[str, str], ...], tuple[TileCandidate, ...]]]:
    """Yield complete tilings (testing helper — prefer ``find_best_tiling_for_file``)."""
    regions = registry.spectrum_regions()
    if not regions:
        return
    seen: set[tuple[tuple[str, str], ...]] = set()
    tiles_acc: list[TileCandidate] = []
    keys_acc: set[tuple[str, str]] = set()

    def _dfs(uncovered: list[SpectrumRegion]) -> Iterator[tuple[tuple[tuple[str, str], ...], tuple[TileCandidate, ...]]]:
        if not uncovered:
            key = _sig_key(keys_acc)
            if key not in seen:
                seen.add(key)
                yield key, tuple(tiles_acc)
            return
        anchor = uncovered[_anchor_index(uncovered)]
        for tile in registry.candidate_tiles_at_anchor(anchor):
            if not _tile_fits_uncovered(tile, uncovered):
                continue
            if tile.meas_keys & keys_acc:
                continue
            tiles_acc.append(tile)
            keys_acc.update(tile.meas_keys)
            yield from _dfs(_subtract_regions_list(uncovered, tile.regions))
            keys_acc.difference_update(tile.meas_keys)
            tiles_acc.pop()

    yield from _dfs(list(regions))


def measurements_from_tiles(tiles: tuple[TileCandidate, ...]) -> list[ChunkMeas]:
    out: list[ChunkMeas] = []
    seen: set[tuple[str, str]] = set()
    for tile in tiles:
        for m in tile.measurements:
            key = (m.layout_name, m.chunk_key)
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
    return out


def tiling_name(tiles: tuple[TileCandidate, ...]) -> str:
    if len(tiles) == 1 and tiles[0].name.startswith("whole:"):
        return tiles[0].name
    parts = sorted(tile.name for tile in tiles)
    return "mix:" + "+".join(parts)


def stack_tiling_pipeline(
    measurements: list[ChunkMeas],
    *,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model,
    star_meta: dict,
    min_chunks: int = MIN_CHUNKS,
) -> dict:
    """Pipeline metric: debias + IVW + CDF core90 via stack_calibrated_exposure."""
    if len(measurements) < min_chunks:
        return {
            "rv_calibrated_kms": np.nan,
            "rv_err_calibrated_kms": np.nan,
            "sigma_rv_core90_kms": np.nan,
            "n_chunks_used": len(measurements),
            "n_chunks_core90": 0,
            "bias_source_mix": "",
        }
    gid = str(measurements[0].gaia_dr3_id)
    per_obj = per_object
    if len(per_object) and "layout_name" in per_object.columns:
        per_obj = per_object[per_object["gaia_dr3_id"] == gid]

    rv = np.array([m.rv_kms for m in measurements], dtype=float)
    err = np.array([m.rv_err_kms for m in measurements], dtype=float)
    chunk_df = pd.DataFrame(
        {
            "gaia_dr3_id": gid,
            "layout_name": [m.layout_name for m in measurements],
            "chunk_key": [m.chunk_key for m in measurements],
            "rv_kms": rv,
            "rv_err_kms": err,
            "chunk_kept": True,
            "teff": measurements[0].teff,
            "logg": star_meta.get("logg", np.nan),
            "mh": star_meta.get("mh", np.nan),
        }
    )
    return stack_calibrated_exposure(
        chunk_df,
        per_object=per_obj,
        fallback=fallback,
        intrinsic_model=intrinsic_model,
        min_chunks=min_chunks,
    )


@dataclass
class TilingSearchResult:
    file: str
    tiling_name: str
    tiles: tuple[TileCandidate, ...]
    stack: dict
    n_tilings_evaluated: int
    n_mixed_tilings: int
    n_whole_layouts: int
    search_truncated: bool

    @property
    def metric_value(self) -> float:
        return float(self.stack.get(DEFAULT_METRIC, np.nan))

    @property
    def n_tilings_available(self) -> int:
        return self.n_mixed_tilings + self.n_whole_layouts


def _search_mixed_tilings(
    registry: TileRegistry,
    *,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model,
    star_meta: dict,
    metric: str,
    max_tilings: int | None,
    include_whole_layouts: bool,
    min_chunks: int,
    count_only: bool = False,
) -> tuple[int, int, TilingSearchResult | None, bool]:
    """
    Single backtracking DFS.

    Returns ``(n_mixed_tilings, n_evaluated, best_result_or_none, truncated)``.
    When ``count_only=True``, skips stack evaluation (bias/intrinsic may be None).
    """
    regions = registry.spectrum_regions()
    n_whole = 0
    if include_whole_layouts:
        n_whole = sum(
            1 for lay in registry.whole_layout_names if registry.whole_layout_tile(lay) is not None
        )

    best: TilingSearchResult | None = None
    best_val = float("inf")
    evaluated = 0
    mixed_count = 0
    seen: set[tuple[tuple[str, str], ...]] = set()
    truncated = False

    tiles_acc: list[TileCandidate] = []
    keys_acc: set[tuple[str, str]] = set()

    def _at_limit() -> bool:
        return max_tilings is not None and evaluated >= max_tilings

    def _evaluate_leaf() -> None:
        nonlocal best, best_val, evaluated, mixed_count
        key = _sig_key(keys_acc)
        if key in seen:
            return
        seen.add(key)
        mixed_count += 1
        if count_only:
            if _at_limit():
                return
            evaluated += 1
            return
        if _at_limit():
            return
        meas = measurements_from_tiles(tuple(tiles_acc))
        stack = stack_tiling_pipeline(
            meas,
            per_object=per_object,
            fallback=fallback,
            intrinsic_model=intrinsic_model,
            star_meta=star_meta,
            min_chunks=min_chunks,
        )
        evaluated += 1
        val = float(stack.get(metric, np.nan))
        if not np.isfinite(val):
            return
        if val < best_val - 1e-12:
            best_val = val
            best = TilingSearchResult(
                file=registry.file,
                tiling_name=tiling_name(tuple(tiles_acc)),
                tiles=tuple(tiles_acc),
                stack=stack,
                n_tilings_evaluated=evaluated,
                n_mixed_tilings=mixed_count,
                n_whole_layouts=n_whole,
                search_truncated=False,
            )

    def _dfs(uncovered: list[SpectrumRegion]) -> None:
        nonlocal truncated
        if _at_limit():
            truncated = max_tilings is not None
            return
        if not uncovered:
            _evaluate_leaf()
            return
        anchor = uncovered[_anchor_index(uncovered)]
        for tile in registry.candidate_tiles_at_anchor(anchor):
            if _at_limit():
                truncated = max_tilings is not None
                return
            if not _tile_fits_uncovered(tile, uncovered):
                continue
            if tile.meas_keys & keys_acc:
                continue
            tiles_acc.append(tile)
            keys_acc.update(tile.meas_keys)
            _dfs(_subtract_regions_list(uncovered, tile.regions))
            keys_acc.difference_update(tile.meas_keys)
            tiles_acc.pop()

    if regions:
        _dfs(list(regions))

    if include_whole_layouts and not count_only:
        for layout_name in registry.whole_layout_names:
            if _at_limit():
                truncated = True
                break
            tile = registry.whole_layout_tile(layout_name)
            if tile is None:
                continue
            key = _sig_key(set(tile.meas_keys))
            if key in seen:
                continue
            seen.add(key)
            stack = stack_tiling_pipeline(
                list(tile.measurements),
                per_object=per_object,
                fallback=fallback,
                intrinsic_model=intrinsic_model,
                star_meta=star_meta,
                min_chunks=min_chunks,
            )
            evaluated += 1
            val = float(stack.get(metric, np.nan))
            if not np.isfinite(val):
                continue
            if val < best_val - 1e-12:
                best_val = val
                best = TilingSearchResult(
                    file=registry.file,
                    tiling_name=tiling_name((tile,)),
                    tiles=(tile,),
                    stack=stack,
                    n_tilings_evaluated=evaluated,
                    n_mixed_tilings=mixed_count,
                    n_whole_layouts=n_whole,
                    search_truncated=truncated,
                )

    if best is not None:
        best.n_tilings_evaluated = evaluated
        best.search_truncated = truncated
    return mixed_count, evaluated, best, truncated


def find_best_tiling_for_file(
    registry: TileRegistry,
    *,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model,
    star_meta: dict,
    metric: str = DEFAULT_METRIC,
    max_tilings: int | None = None,
    include_whole_layouts: bool = True,
    min_chunks: int = MIN_CHUNKS,
) -> TilingSearchResult | None:
    """One DFS pass: dedupe, count mixed tilings, and track best pipeline σ."""
    if max_tilings is not None:
        logger.info(
            "%s: capping evaluations at %d tilings",
            registry.file,
            max_tilings,
        )
    mixed_count, _evaluated, best, truncated = _search_mixed_tilings(
        registry,
        per_object=per_object,
        fallback=fallback,
        intrinsic_model=intrinsic_model,
        star_meta=star_meta,
        metric=metric,
        max_tilings=max_tilings,
        include_whole_layouts=include_whole_layouts,
        min_chunks=min_chunks,
        count_only=False,
    )
    if truncated and best is not None:
        logger.warning(
            "%s: search truncated at %d evaluations (%d mixed tilings seen)",
            registry.file,
            best.n_tilings_evaluated,
            _mixed_count,
        )
    return best


def run_campaign_tiling_search(
    campaign_dir,
    *,
    max_tilings_per_file: int | None = None,
    metric: str = DEFAULT_METRIC,
    mix_layout_names: list[str] | None = None,
    whole_layout_names: list[str] | None = None,
    count_tilings: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Search best tiling per exposure; one file in memory at a time."""
    from pathlib import Path

    from validation.chunk_bias_lib import load_stellar_metadata
    from validation.chunk_adaptive_stack import REPO_ROOT

    campaign_dir = Path(campaign_dir)
    layouts = load_layouts(campaign_dir)
    meas_df = load_campaign_measurements(campaign_dir, layouts)
    if meas_df.empty:
        raise ValueError(f"No measurements in {campaign_dir}")

    per_object, fallback, intrinsic_model = build_multi_layout_bias_tables(campaign_dir, layouts)
    meta_tbl = load_stellar_metadata(REPO_ROOT / "output")
    global_idx = _index_by_file_layout(meas_df)

    file_labels = sorted(meas_df["file"].astype(str).unique())
    meta_by_gid: dict[str, dict] = {}
    if not meta_tbl.empty:
        for gid, row in meta_tbl.set_index("gaia_dr3_id").iterrows():
            meta_by_gid[str(gid)] = {
                "logg": float(row.get("logg", np.nan)),
                "mh": float(row.get("mh", np.nan)),
            }

    epoch_rows: list[dict] = []
    count_rows: list[dict] = []

    for fi, file_label in enumerate(file_labels):
        if fi % 10 == 0:
            logger.info("Tiling search %d/%d at %s", fi + 1, len(file_labels), file_label)

        gid = str(meas_df.loc[meas_df["file"].astype(str) == file_label, "gaia_dr3_id"].iloc[0])
        star_meta = meta_by_gid.get(gid, {"logg": np.nan, "mh": np.nan})

        fidx = file_meas_index(global_idx, file_label)
        registry = TileRegistry.for_file(
            file_label,
            layouts,
            fidx,
            mix_layout_names=mix_layout_names,
            whole_layout_names=whole_layout_names,
        )

        if count_tilings:
            n_mixed = count_complete_tilings(registry)
            count_rows.append({"file": file_label, "n_mixed_tilings": n_mixed})
            registry._order_tile_cache.clear()
            del registry, fidx
            gc.collect()
            continue

        result = find_best_tiling_for_file(
            registry,
            per_object=per_object,
            fallback=fallback,
            intrinsic_model=intrinsic_model,
            star_meta=star_meta,
            metric=metric,
            max_tilings=max_tilings_per_file,
        )

        if result is not None:
            count_rows.append({"file": file_label, "n_mixed_tilings": result.n_mixed_tilings})

        if result is not None:
            teff = float(meas_df.loc[meas_df["file"].astype(str) == file_label, "teff"].iloc[0])
            mjd = float(meas_df.loc[meas_df["file"].astype(str) == file_label, "mjd"].iloc[0])
            epoch_rows.append(
                {
                    "gaia_dr3_id": gid,
                    "file": file_label,
                    "mjd": mjd,
                    "teff": teff,
                    "tiling_name": result.tiling_name,
                    "rv_calibrated_kms": result.stack["rv_calibrated_kms"],
                    "rv_err_calibrated_kms": result.stack["rv_err_calibrated_kms"],
                    "sigma_rv_core90_kms": result.stack.get("sigma_rv_core90_kms", np.nan),
                    "n_chunks_used": result.stack.get("n_chunks_used", np.nan),
                    "n_chunks_core90": result.stack.get("n_chunks_core90", np.nan),
                    "n_mixed_tilings": result.n_mixed_tilings,
                    "n_tilings_evaluated": result.n_tilings_evaluated,
                    "search_truncated": result.search_truncated,
                    "tile_plan_json": json.dumps([t.name for t in result.tiles]),
                }
            )

        registry._order_tile_cache.clear()
        del registry, fidx, result
        gc.collect()

    epochs = pd.DataFrame(epoch_rows)
    counts = pd.DataFrame(count_rows)
    summary_metrics = summarize_sigma_rv_metrics(epochs) if not epochs.empty else {}
    summary = pd.DataFrame(
        [
            {
                "n_exposures": len(epochs),
                "metric": metric,
                **summary_metrics,
                "n_files_truncated": int(epochs["search_truncated"].sum()) if not epochs.empty else 0,
                "median_n_mixed_tilings": float(counts["n_mixed_tilings"].median())
                if not counts.empty
                else np.nan,
            }
        ]
    )
    return epochs, summary, counts
