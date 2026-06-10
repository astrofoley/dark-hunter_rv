"""Chunk layout specification: N chunks from N+1 pixel or wavelength edges."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class ChunkLayout:
    """
    Layout for splitting each echelle order into chunks.

    - ``subchunks``: equal pixel splits (overrides pixel_edges if >1 and pixel_edges is default)
    - ``pixel_edges``: N+1 fractions in [0, 1] along the pixel array (default 0..1)
    - ``merge_orders``: optional list of [order_start, order_end] inclusive groups → single chunk key
    """

    name: str = "whole_order"
    subchunks: int = 1
    pixel_edges: list[float] = field(default_factory=lambda: [0.0, 1.0])
    merge_orders: list[list[int]] | None = None
    min_pixels: int = 5
    edge_preset: str | None = None  # equal | blue_heavy | red_heavy | telluric
    telluric_split: bool = False

    def n_chunks_per_order(self) -> int:
        if self.subchunks > 1:
            return int(self.subchunks)
        return max(len(self.pixel_edges) - 1, 1)

    def normalized_pixel_edges(self) -> np.ndarray:
        if self.subchunks > 1:
            return np.linspace(0.0, 1.0, self.subchunks + 1)
        edges = np.asarray(self.pixel_edges, float)
        if len(edges) < 2:
            return np.array([0.0, 1.0])
        edges = np.clip(edges, 0.0, 1.0)
        edges[0] = 0.0
        edges[-1] = 1.0
        return edges


def preset_pixel_edges(n_chunks: int, preset: str) -> list[float]:
    """Return N+1 fractional edges for a named preset."""
    n = max(int(n_chunks), 1)
    if preset == "equal" or n == 1:
        return np.linspace(0.0, 1.0, n + 1).tolist()
    if preset == "blue_heavy":
        if n == 2:
            return [0.0, 0.35, 1.0]
        if n == 3:
            return [0.0, 0.35, 0.6, 1.0]
        if n == 4:
            return [0.0, 0.3, 0.5, 0.7, 1.0]
    if preset == "red_heavy":
        if n == 2:
            return [0.0, 0.65, 1.0]
        if n == 3:
            return [0.0, 0.2, 0.45, 1.0]
        if n == 4:
            return [0.0, 0.15, 0.35, 0.6, 1.0]
    return np.linspace(0.0, 1.0, n + 1).tolist()


def build_edge_preset_layout(n_chunks: int, preset: str) -> ChunkLayout:
    name = f"n{n_chunks}_{preset}"
    if preset == "telluric":
        return ChunkLayout(name=name, subchunks=1, telluric_split=True, edge_preset="telluric")
    edges = preset_pixel_edges(n_chunks, preset)
    return ChunkLayout(name=name, subchunks=1, pixel_edges=edges, edge_preset=preset)


def build_campaign_broad_grid() -> list[ChunkLayout]:
    """Broad grid: split N=2,3,4 and merge W=2,3,4."""
    layouts: list[ChunkLayout] = []
    for n in (2, 3, 4):
        layouts.append(build_equal_subchunk_layout(n))
    for w in (2, 3, 4):
        layouts.append(build_merge_orders_layout(merge_width=w))
    return layouts


def build_campaign_edge_grid(n_chunks: int = 3) -> list[ChunkLayout]:
    """
    Edge preset variants after broad grid.

    Omits ``equal`` — same pixel edges as ``subchunks_{n}`` from :func:`build_campaign_broad_grid`.
    """
    presets = ("blue_heavy", "red_heavy", "telluric")
    return [build_edge_preset_layout(n_chunks, p) for p in presets]


def measurement_id_for_chunk(chunk_key: str, pixel_edges: list[float] | None, merge_orders: bool) -> str:
    """Canonical cache key for a chunk measurement (layout-independent when edges match)."""
    if merge_orders:
        return f"merge:{chunk_key}"
    edges = pixel_edges or [0.0, 1.0]
    edge_tag = "-".join(f"{e:.4f}" for e in edges)
    return f"chunk:{chunk_key}:e:{edge_tag}"


def save_chunk_layout(layout: ChunkLayout, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": layout.name,
        "subchunks": layout.subchunks,
        "pixel_edges": [float(x) for x in layout.normalized_pixel_edges()],
        "merge_orders": layout.merge_orders,
        "min_pixels": layout.min_pixels,
        "edge_preset": layout.edge_preset,
        "telluric_split": layout.telluric_split,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_chunk_layout(path: Path | str) -> ChunkLayout:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    return ChunkLayout(
        name=str(raw.get("name", path.stem)),
        subchunks=int(raw.get("subchunks", 1)),
        pixel_edges=[float(x) for x in raw.get("pixel_edges", [0.0, 1.0])],
        merge_orders=raw.get("merge_orders"),
        min_pixels=int(raw.get("min_pixels", 5)),
        edge_preset=raw.get("edge_preset"),
        telluric_split=bool(raw.get("telluric_split", False)),
    )


def apf_valid_orders(*, num_orders: int = 70, bad_orders: list[int] | None = None) -> list[int]:
    bad = bad_orders if bad_orders is not None else [0, 1, 2, 53, 57, 58, 59, 60, 63, 64, 65]
    return [o for o in range(num_orders) if o not in bad]


def build_equal_subchunk_layout(n: int) -> ChunkLayout:
    """N equal pixel sub-chunks per order (N+1 edges at 0 … 1)."""
    n = max(int(n), 1)
    return ChunkLayout(
        name=f"subchunks_{n}",
        subchunks=n,
        pixel_edges=np.linspace(0.0, 1.0, n + 1).tolist(),
    )


def build_custom_edge_layout(name: str, pixel_edges: list[float]) -> ChunkLayout:
    """N chunks from explicit N+1 fractional pixel edges."""
    edges = [float(x) for x in pixel_edges]
    if len(edges) < 2:
        raise ValueError("pixel_edges requires at least two values")
    return ChunkLayout(name=name, subchunks=1, pixel_edges=edges)


def build_merge_orders_layout(
    *,
    merge_width: int,
    valid_orders: list[int] | None = None,
) -> ChunkLayout:
    """
    Coarsen whole-order chunks by merging ``merge_width`` adjacent echelle orders.

    ``merge_width=1`` is equivalent to whole-order (no merge groups).
    """
    merge_width = max(int(merge_width), 1)
    orders = list(valid_orders or apf_valid_orders())
    if merge_width == 1:
        return ChunkLayout(name="whole_order", subchunks=1)
    groups: list[list[int]] = []
    for i in range(0, len(orders), merge_width):
        block = orders[i : i + merge_width]
        if block:
            groups.append([block[0], block[-1]])
    return ChunkLayout(
        name=f"merge_w{merge_width}",
        subchunks=1,
        merge_orders=groups,
    )


def default_parametric_grid(
    *,
    subchunk_counts: list[int] | None = None,
    merge_widths: list[int] | None = None,
) -> list[ChunkLayout]:
    """Rough parametric grid: equal sub-chunks N and order-merge widths."""
    subchunk_counts = subchunk_counts or [1, 2, 4]
    merge_widths = merge_widths or [1, 2, 4]
    layouts: list[ChunkLayout] = []
    seen: set[str] = set()
    for n in subchunk_counts:
        lay = build_equal_subchunk_layout(n)
        if lay.name not in seen:
            layouts.append(lay)
            seen.add(lay.name)
    for w in merge_widths:
        lay = build_merge_orders_layout(merge_width=w)
        if lay.name not in seen:
            layouts.append(lay)
            seen.add(lay.name)
    return layouts


def layout_to_dict(layout: ChunkLayout) -> dict[str, Any]:
    return {
        "name": layout.name,
        "subchunks": layout.subchunks,
        "pixel_edges": layout.normalized_pixel_edges().tolist(),
        "merge_orders": layout.merge_orders,
        "min_pixels": layout.min_pixels,
        "n_chunks_per_order": layout.n_chunks_per_order(),
    }


def merge_order_groups(order: int, merge_orders: list[list[int]] | None) -> int | None:
    """Return merge-group index if order belongs to a merge group, else None."""
    if not merge_orders:
        return None
    for gi, grp in enumerate(merge_orders):
        lo, hi = int(grp[0]), int(grp[1])
        if lo <= order <= hi:
            return gi
    return None


def map_chunk_key(old_key: str, layout: ChunkLayout) -> str:
    """
    Map an existing diagnostics chunk_key to a layout chunk_key.

    Whole-order keys ``"{order}"`` map to sub-chunks only when layout.subchunks > 1
    (cannot split offline — returns same key with a warning tag).

    Supports ``merge_orders`` by combining ``"{o}"`` → ``"merge_{gi}"``.
    """
    parts = str(old_key).split("_")
    try:
        order = int(parts[0])
    except ValueError:
        return old_key
    sub = int(parts[1]) if len(parts) > 1 else 0

    mg = merge_order_groups(order, layout.merge_orders)
    if mg is not None:
        if len(parts) > 1:
            return f"merge_{mg}_{sub}"
        return f"merge_{mg}"

    if layout.subchunks > 1 and len(parts) == 1:
        # Cannot split whole-order diagnostics offline
        return old_key

    if layout.subchunks > 1:
        n = layout.subchunks
        new_sub = min(sub, n - 1)
        return f"{order}_{new_sub}"
    return str(order)


def rebinned_chunk_rows(df, layout: ChunkLayout):
    """
    Rebin long chunk-level rows to a coarser layout (merge orders or regroup keys).

    Returns DataFrame with one row per (file, new_chunk_key) using IVW of rv_kms.
    """
    import pandas as pd

    if df.empty:
        return df
    tab = df.copy()
    tab["new_chunk_key"] = tab["chunk_key"].astype(str).map(lambda ck: map_chunk_key(ck, layout))
    rows = []
    for (file_label, nck), g in tab.groupby(["file", "new_chunk_key"], sort=False):
        rv = g["rv_kms"].astype(float).values
        err = g["rv_err_kms"].astype(float).values
        ok = np.isfinite(rv) & np.isfinite(err) & (err > 0)
        if not np.any(ok):
            continue
        w = 1.0 / err[ok] ** 2
        mu = float(np.sum(w * rv[ok]) / np.sum(w))
        sig = float(1.0 / np.sqrt(np.sum(w)))
        r0 = g.iloc[0]
        rows.append(
            {
                **{c: r0[c] for c in g.columns if c not in ("chunk_key", "rv_kms", "rv_err_kms", "new_chunk_key")},
                "chunk_key": str(nck),
                "rv_kms": mu,
                "rv_err_kms": sig,
                "n_merged_chunks": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def iter_order_chunks_from_layout(
    spec_data: dict,
    bad_orders: list[int],
    layout: ChunkLayout,
):
    """Yield (chunk_key, wave, flux, eflux) for a layout (including merged orders)."""
    from darkhunter_rv.chunking import iter_order_chunks, iter_order_chunks_with_edges

    if layout.merge_orders:
        groups: dict[int, dict[str, list]] = {}
        for order in sorted(spec_data.keys()):
            if order in bad_orders:
                continue
            gi = merge_order_groups(order, layout.merge_orders)
            if gi is None:
                continue
            data = spec_data[order]
            bucket = groups.setdefault(gi, {"wavelength": [], "flux": [], "eflux": []})
            for k in ("wavelength", "flux", "eflux"):
                bucket[k].extend(data[k])
        for gi, data in sorted(groups.items()):
            w = np.array(data["wavelength"], float)
            f = np.array(data["flux"], float)
            e = np.array(data["eflux"], float)
            if len(w) < layout.min_pixels:
                continue
            yield f"merge_{gi}", w, f, e
        return

    from darkhunter_rv.chunking import telluric_pixel_edges

    for order in sorted(spec_data.keys()):
        if order in bad_orders:
            continue
        data = spec_data[order]
        w = np.array(data["wavelength"], float)
        f = np.array(data["flux"], float)
        e = np.array(data["eflux"], float)
        n = len(w)
        if n < layout.min_pixels:
            continue
        if layout.telluric_split:
            edges_frac = telluric_pixel_edges(w, min_pixels=layout.min_pixels)
        elif layout.subchunks > 1:
            edges_frac = np.linspace(0.0, 1.0, layout.subchunks + 1)
        else:
            edges_frac = layout.normalized_pixel_edges()
        idx_edges = np.round(edges_frac * n).astype(int)
        idx_edges[0] = 0
        idx_edges[-1] = n
        for si in range(len(idx_edges) - 1):
            i0, i1 = int(idx_edges[si]), int(idx_edges[si + 1])
            if i1 - i0 < layout.min_pixels:
                continue
            key = str(order) if len(idx_edges) == 2 else f"{order}_{si}"
            yield key, w[i0:i1], f[i0:i1], e[i0:i1]
