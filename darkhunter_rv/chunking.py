"""Split orders into wavelength/pixel sub-chunks for robust RV."""
from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import numpy as np


def iter_order_chunks(
    spec_data: Dict[int, Dict[str, List]],
    bad_orders: List[int],
    subchunks: int,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (chunk_key, wave, flux, eflux). chunk_key is \"{order}\" or \"{order}_{si}\"."""
    for order in sorted(spec_data.keys()):
        if order in bad_orders:
            continue
        data = spec_data[order]
        w = np.array(data["wavelength"], float)
        f = np.array(data["flux"], float)
        e = np.array(data["eflux"], float)
        if len(w) < 5:
            continue
        if subchunks <= 1:
            yield str(order), w, f, e
            continue
        edges = np.linspace(0, len(w), subchunks + 1, dtype=int)
        for si in range(subchunks):
            i0, i1 = int(edges[si]), int(edges[si + 1])
            if i1 - i0 < 5:
                continue
            yield f"{order}_{si}", w[i0:i1], f[i0:i1], e[i0:i1]


def iter_order_chunks_with_edges(
    spec_data: Dict[int, Dict[str, List]],
    bad_orders: List[int],
    pixel_edges: np.ndarray | List[float],
    *,
    min_pixels: int = 5,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Yield chunks using N+1 pixel fractional edges in [0, 1].

    ``pixel_edges`` has length N+1; chunk i spans [edge[i], edge[i+1]).
    """
    edges_frac = np.asarray(pixel_edges, float)
    if len(edges_frac) < 2:
        raise ValueError("pixel_edges requires at least two values (N+1 edges for N chunks)")
    for order in sorted(spec_data.keys()):
        if order in bad_orders:
            continue
        data = spec_data[order]
        w = np.array(data["wavelength"], float)
        f = np.array(data["flux"], float)
        e = np.array(data["eflux"], float)
        n = len(w)
        if n < min_pixels:
            continue
        idx_edges = np.round(edges_frac * n).astype(int)
        idx_edges[0] = 0
        idx_edges[-1] = n
        for si in range(len(idx_edges) - 1):
            i0, i1 = int(idx_edges[si]), int(idx_edges[si + 1])
            if i1 - i0 < min_pixels:
                continue
            key = str(order) if len(idx_edges) == 2 else f"{order}_{si}"
            yield key, w[i0:i1], f[i0:i1], e[i0:i1]


def parse_chunk_key(chunk_key: str) -> tuple[int | None, int, str]:
    """
    Parse ``chunk_key`` into (echelle_order, sort_index, kind).

    - ``"15"`` → (15, 15, "order")
    - ``"15_2"`` → (15, 15, "subchunk")  (sub index in sort tie-break)
    - ``"merge_3"`` → (None, 3, "merge")
    """
    parts = str(chunk_key).split("_")
    if parts and parts[0] == "merge":
        try:
            gi = int(parts[1])
        except (IndexError, ValueError):
            gi = 0
        return None, gi, "merge"
    try:
        order = int(parts[0])
    except ValueError:
        return None, 0, "unknown"
    sub = int(parts[1]) if len(parts) > 1 else 0
    kind = "subchunk" if sub > 0 or len(parts) > 1 else "order"
    return order, order * 100 + sub, kind


def bias_order_from_chunk_key(chunk_key: str) -> int | None:
    """Echelle order for per-order bias lookup, or None for merged chunks."""
    order, _, _ = parse_chunk_key(chunk_key)
    return order


def chunk_sort_key(chunk_key: str) -> tuple[int, int, int]:
    """Sort key for chunk identifiers (orders, subchunks, merge groups)."""
    order, sort_idx, kind = parse_chunk_key(chunk_key)
    kind_rank = {"order": 0, "subchunk": 0, "merge": 1}.get(kind, 2)
    primary = sort_idx if order is None else order
    return (kind_rank, primary, sort_idx)


def telluric_pixel_edges(
    wave: np.ndarray,
    *,
    min_pixels: int = 5,
    max_interior_edges: int = 3,
) -> np.ndarray:
    """
    N+1 fractional pixel edges splitting an order at telluric/clean boundaries.

    Uses contamination bands from ``qc.rv_contamination_bands()``.
    Falls back to [0, 1] when the order is too short or fully telluric.
    """
    from darkhunter_rv import qc

    w = np.asarray(wave, float)
    n = len(w)
    if n < 2 * min_pixels:
        return np.array([0.0, 1.0])
    contam = qc.wavelength_band_mask(w, qc.rv_contamination_bands())
    if not np.any(contam):
        return np.array([0.0, 1.0])
    if np.all(contam):
        return np.array([0.0, 1.0])

    # Transition indices where contamination flag changes.
    transitions = [0]
    for i in range(1, n):
        if bool(contam[i]) != bool(contam[i - 1]):
            transitions.append(i)
    transitions.append(n)
    if len(transitions) <= 2:
        return np.array([0.0, 1.0])

    # Fractional edges at transitions; merge tiny segments into neighbours.
    frac_edges = [t / n for t in transitions]
    merged: list[float] = [0.0]
    for i in range(1, len(frac_edges) - 1):
        e = frac_edges[i]
        seg_len = frac_edges[i + 1] - frac_edges[i - 1] if i + 1 < len(frac_edges) else 1.0
        if (frac_edges[i] - frac_edges[i - 1]) * n >= min_pixels and (frac_edges[i + 1] - frac_edges[i]) * n >= min_pixels:
            if e not in merged:
                merged.append(e)
    merged.append(1.0)
    merged = sorted(set(merged))
    if len(merged) > max_interior_edges + 2:
        # Keep endpoints + evenly subsample interior transition edges.
        interior = merged[1:-1]
        idx = np.linspace(0, len(interior) - 1, max_interior_edges).astype(int)
        merged = [0.0] + [interior[i] for i in idx] + [1.0]
    if len(merged) < 2:
        return np.array([0.0, 1.0])
    return np.asarray(merged, float)
