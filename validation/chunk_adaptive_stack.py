#!/usr/bin/env python3
"""
Adaptive chunk stacking: combine measurements from multiple campaign layouts.

Each exposure is scored against a **complete candidate set**:

1. **Whole layouts** — all chunks from one layout (merge_w4, subchunks_3, n3_red_heavy, …).
   Guarantees the adaptive pick is at least as good as the best single layout on
   exposures where that layout is valid.

2. **Per-order greedy mix** — each echelle order uses one layout's **full** subchunk
   partition for that order (never splices ``15_0`` from subchunks_4 with ``15_1``
   from subchunks_3; edge presets are not assumed commensurate).

Selection uses the same calibrated-stack σ_RV as evaluation.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.chunk_adaptive_stack --campaign-dir validation_output/chunk_campaign
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.chunk_bias_lib import load_stellar_metadata  # noqa: E402
from validation.chunk_calibration import (  # noqa: E402
    build_intrinsic_scatter_model,
    relative_pair_table,
    summarize_relative_gate,
    summarize_sigma_rv_metrics,
)
from validation.chunk_grid_search import composite_score  # noqa: E402
from validation.chunk_layout import (  # noqa: E402
    ChunkLayout,
    apf_valid_orders,
    load_chunk_layout,
)
from validation.chunk_measurement_cache import (  # noqa: E402
    diagnostics_glob_for_layout,
    ingest_layout_diagnostics_dir,
    load_cache,
)
from validation.plot_chunk_residuals import (  # noqa: E402
    DEFAULT_CHUNK_MAX_DELTA_KMS,
    DEFAULT_CHUNK_OUTLIER_SIGMA,
    _load_chunk_rows,
    _summarize_chunks_per_object,
    apply_spectrum_chunk_outlier_clip,
)

logger = logging.getLogger(__name__)

# Fine per-order layouts (distinct pixel edges — not nested refinements of each other).
PER_ORDER_LAYOUTS = (
    "subchunks_2",
    "subchunks_3",
    "subchunks_4",
    "n3_red_heavy",
    "n3_blue_heavy",
)
COARSE_LAYOUTS = ("merge_w4", "merge_w3", "merge_w2")
ALL_WHOLE_LAYOUTS = COARSE_LAYOUTS + PER_ORDER_LAYOUTS
MIN_CHUNKS = 3


@dataclass(frozen=True)
class StackCandidate:
    """Named chunk selection for one exposure."""

    name: str
    chunks: tuple[ChunkMeas, ...]


@dataclass(frozen=True)
class ChunkMeas:
    layout_name: str
    chunk_key: str
    rv_kms: float
    rv_err_kms: float
    orders: frozenset[int]
    qc_pass: bool
    file: str
    gaia_dr3_id: str
    mjd: float
    teff: float

    @property
    def bias_key(self) -> tuple[str, str]:
        return (self.layout_name, self.chunk_key)


def _valid_row(rv: float, err: float, qc: bool) -> bool:
    return bool(qc) and np.isfinite(rv) and np.isfinite(err) and err > 0


def _orders_for_chunk(chunk_key: str, layout: ChunkLayout) -> frozenset[int]:
    from darkhunter_rv.chunking import parse_chunk_key

    _order, sort_idx, kind = parse_chunk_key(chunk_key)
    if kind == "merge" and layout.merge_orders:
        gi = int(sort_idx)
        if 0 <= gi < len(layout.merge_orders):
            lo, hi = layout.merge_orders[gi]
            valid = set(apf_valid_orders())
            return frozenset(o for o in range(int(lo), int(hi) + 1) if o in valid)
        return frozenset()
    if _order is not None:
        return frozenset([int(_order)])
    return frozenset()


def load_layouts(campaign_dir: Path) -> dict[str, ChunkLayout]:
    layouts: dict[str, ChunkLayout] = {}
    layout_dir = campaign_dir / "layouts"
    if layout_dir.is_dir():
        for p in sorted(layout_dir.glob("*.yaml")):
            lay = load_chunk_layout(p)
            layouts[lay.name] = lay
    return layouts


def load_campaign_measurements(
    campaign_dir: Path,
    layouts: dict[str, ChunkLayout],
) -> pd.DataFrame:
    """Cache + diagnostics for layouts missing from cache."""
    cache_path = campaign_dir / "measurement_cache.csv"
    cache = load_cache(cache_path)
    have_layouts = set(cache["layout_name"].astype(str).unique()) if len(cache) else set()
    for name, layout in layouts.items():
        if name in have_layouts:
            continue
        diag_dir = campaign_dir / "diagnostics" / name
        if diag_dir.is_dir():
            cache = ingest_layout_diagnostics_dir(diag_dir, layout=layout, cache=cache)
    rows: list[ChunkMeas] = []
    for _, r in cache.iterrows():
        layout_name = str(r["layout_name"])
        layout = layouts.get(layout_name)
        if layout is None:
            continue
        ck = str(r["chunk_key"])
        rv = float(r["rv_kms"])
        err = float(r["rv_err_kms"])
        qc = bool(r.get("qc_pass", True))
        if not _valid_row(rv, err, qc):
            continue
        rows.append(
            ChunkMeas(
                layout_name=layout_name,
                chunk_key=ck,
                rv_kms=rv,
                rv_err_kms=err,
                orders=_orders_for_chunk(ck, layout),
                qc_pass=qc,
                file=str(r["file"]),
                gaia_dr3_id=str(r["gaia_dr3_id"]),
                mjd=float(r["mjd"]),
                teff=float(r["teff"]),
            )
        )
    return pd.DataFrame([m.__dict__ | {"orders": m.orders} for m in rows])


def _meas_list_to_chunk_df(meas_list: list[ChunkMeas], *, meta: dict) -> pd.DataFrame:
    rows = []
    for m in meas_list:
        rows.append(
            {
                "gaia_dr3_id": m.gaia_dr3_id,
                "file": m.file,
                "mjd": m.mjd,
                "teff": m.teff,
                "logg": meta.get("logg", np.nan),
                "mh": meta.get("mh", np.nan),
                "layout_name": m.layout_name,
                "chunk_key": m.chunk_key,
                "rv_kms": m.rv_kms,
                "rv_err_kms": m.rv_err_kms,
                "chunk_kept": True,
            }
        )
    return pd.DataFrame(rows)


def _lookup_layout_bias(
    gaia_id: str,
    layout_name: str,
    chunk_key: str,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
) -> float:
    po = pd.DataFrame()
    if len(per_object) and "layout_name" in per_object.columns:
        po = per_object[
            (per_object["gaia_dr3_id"] == str(gaia_id))
            & (per_object["layout_name"].astype(str) == str(layout_name))
            & (per_object["chunk_key"].astype(str) == str(chunk_key))
        ]
    if len(po):
        return float(po.iloc[0]["weighted_mean_residual_kms"])
    fb = fallback[
        (fallback["layout_name"].astype(str) == str(layout_name))
        & (fallback["chunk_key"].astype(str) == str(chunk_key))
    ]
    if len(fb) and np.isfinite(fb.iloc[0]["bias_kms"]):
        return float(fb.iloc[0]["bias_kms"])
    return float("nan")


def build_multi_layout_bias_tables(
    campaign_dir: Path,
    layouts: dict[str, ChunkLayout],
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Per-(layout, chunk_key) bias and intrinsic tables."""
    parts = []
    for name in sorted(layouts):
        glob_pat = diagnostics_glob_for_layout(campaign_dir, name)
        tab = _load_chunk_rows(glob_pat)
        if tab.empty:
            continue
        tab = apply_spectrum_chunk_outlier_clip(
            tab, nsigma=DEFAULT_CHUNK_OUTLIER_SIGMA, max_delta_kms=DEFAULT_CHUNK_MAX_DELTA_KMS
        )
        tab["layout_name"] = name
        parts.append(tab)
    if not parts:
        empty = pd.DataFrame(columns=["gaia_dr3_id", "layout_name", "chunk_key", "weighted_mean_residual_kms"])
        return empty, empty, build_intrinsic_scatter_model(empty)
    tab = pd.concat(parts, ignore_index=True)
    meta = load_stellar_metadata(REPO_ROOT / "output")
    if not meta.empty:
        tab = tab.merge(meta, on="gaia_dr3_id", how="left")
        if "teff_gaia" in tab.columns:
            miss = ~np.isfinite(tab["teff"].astype(float))
            tab.loc[miss, "teff"] = tab.loc[miss, "teff_gaia"]
    bias_rows = []
    for (gid, layout_name), obj in tab.groupby(["gaia_dr3_id", "layout_name"], sort=False):
        summ = _summarize_chunks_per_object(obj[obj["chunk_kept"].astype(bool)])
        if summ.empty:
            continue
        summ["gaia_dr3_id"] = str(gid)
        summ["layout_name"] = str(layout_name)
        if "teff" not in summ.columns:
            summ["teff"] = float(obj["teff"].iloc[0]) if "teff" in obj.columns else np.nan
        bias_rows.append(summ)
    bias = pd.concat(bias_rows, ignore_index=True) if bias_rows else pd.DataFrame()
    if not bias.empty and not meta.empty:
        bias = bias.merge(meta[["gaia_dr3_id", "logg", "mh"]], on="gaia_dr3_id", how="left")

    fallback_rows = []
    if not bias.empty:
        for (layout_name, ck), g in bias.groupby(["layout_name", "chunk_key"]):
            intrinsic = g["intrinsic_scatter_kms"].astype(float).values
            b = g["weighted_mean_residual_kms"].astype(float).values
            stat = g["statistical_err_kms"].astype(float).values
            fallback_rows.append(
                {
                    "layout_name": str(layout_name),
                    "chunk_key": str(ck),
                    "bias_kms": float(np.nanmedian(b)),
                    "statistical_err_kms": float(np.nanmedian(stat[np.isfinite(stat)]))
                    if np.any(np.isfinite(stat))
                    else np.nan,
                    "intrinsic_scatter_kms": float(np.nanmedian(intrinsic[np.isfinite(intrinsic) & (intrinsic > 0)]))
                    if np.any(np.isfinite(intrinsic) & (intrinsic > 0))
                    else 0.0,
                }
            )
    fallback = pd.DataFrame(fallback_rows)
    intrinsic_model = build_intrinsic_scatter_model(bias) if not bias.empty else build_intrinsic_scatter_model(
        pd.DataFrame()
    )
    return bias, fallback, intrinsic_model


def _stack_layout_aware(
    chunk_df: pd.DataFrame,
    *,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model,
    min_chunks: int,
) -> dict:
    """IVW stack with per-layout bias lookup and per-spectrum stat errors."""
    from validation.chunk_calibration import _sigma_total_kms, _sigma_rv_from_weights

    gid = str(chunk_df["gaia_dr3_id"].iloc[0])
    kept = chunk_df[chunk_df["chunk_kept"].astype(bool)] if "chunk_kept" in chunk_df.columns else chunk_df
    rv_d: list[float] = []
    wts: list[float] = []
    for _, r in kept.iterrows():
        layout_name = str(r["layout_name"])
        ck = str(r["chunk_key"])
        bias = _lookup_layout_bias(gid, layout_name, ck, per_object, fallback)
        if not np.isfinite(bias):
            continue
        stat = float(r.get("rv_err_kms", np.nan))
        if not np.isfinite(stat) or stat <= 0:
            continue
        intrinsic = intrinsic_model.predict(
            ck,
            teff=float(r.get("teff", np.nan)),
            logg=float(r.get("logg", np.nan)),
            mh=float(r.get("mh", np.nan)),
        )
        sigma = _sigma_total_kms(stat, intrinsic)
        rv_d.append(float(r["rv_kms"]) - bias)
        wts.append(1.0 / sigma**2)
    if len(rv_d) < min_chunks:
        return {"rv_calibrated_kms": np.nan, "rv_err_calibrated_kms": np.nan, "n_chunks_used": len(rv_d)}
    rv_arr = np.asarray(rv_d, float)
    w_arr = np.asarray(wts, float)
    mu = float(np.sum(w_arr * rv_arr) / np.sum(w_arr))
    return {
        "rv_calibrated_kms": mu,
        "rv_err_calibrated_kms": _sigma_rv_from_weights(w_arr),
        "n_chunks_used": int(len(rv_d)),
    }


def _stack_result_for_meas(
    meas_list: list[ChunkMeas] | tuple[ChunkMeas, ...],
    *,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model,
    star_meta: dict,
    min_chunks: int = MIN_CHUNKS,
) -> dict:
    if len(meas_list) < min_chunks:
        return {"rv_calibrated_kms": np.nan, "rv_err_calibrated_kms": np.nan, "n_chunks_used": len(meas_list)}
    chunk_df = _meas_list_to_chunk_df(list(meas_list), meta=star_meta)
    return _stack_layout_aware(
        chunk_df,
        per_object=per_object,
        fallback=fallback,
        intrinsic_model=intrinsic_model,
        min_chunks=min_chunks,
    )


def _index_by_file_layout(df: pd.DataFrame) -> dict[tuple[str, str, str], ChunkMeas]:
    out: dict[tuple[str, str, str], ChunkMeas] = {}
    for _, r in df.iterrows():
        m = ChunkMeas(
            layout_name=str(r["layout_name"]),
            chunk_key=str(r["chunk_key"]),
            rv_kms=float(r["rv_kms"]),
            rv_err_kms=float(r["rv_err_kms"]),
            orders=r["orders"],
            qc_pass=bool(r["qc_pass"]),
            file=str(r["file"]),
            gaia_dr3_id=str(r["gaia_dr3_id"]),
            mjd=float(r["mjd"]),
            teff=float(r["teff"]),
        )
        out[(m.file, m.layout_name, m.chunk_key)] = m
    return out


def _whole_layout_chunks(
    file: str,
    layout_name: str,
    idx: dict[tuple[str, str, str], ChunkMeas],
) -> list[ChunkMeas] | None:
    """All valid chunks for one file under a single layout."""
    chunks = [m for (f, lay, _ck), m in idx.items() if f == file and lay == layout_name]
    if len(chunks) < MIN_CHUNKS:
        return None
    return sorted(chunks, key=lambda m: (m.chunk_key, m.layout_name))


def _orders_with_fine_data(
    file: str,
    idx: dict[tuple[str, str, str], ChunkMeas],
    layouts: dict[str, ChunkLayout],
) -> list[int]:
    orders: set[int] = set()
    for (f, layout_name, _ck), m in idx.items():
        if f != file or layout_name not in PER_ORDER_LAYOUTS:
            continue
        orders |= set(m.orders)
    return sorted(orders)


def _order_subchunks(
    file: str,
    order: int,
    layout_name: str,
    layout: ChunkLayout,
    idx: dict[tuple[str, str, str], ChunkMeas],
) -> list[ChunkMeas] | None:
    n = layout.n_chunks_per_order()
    chunks: list[ChunkMeas] = []
    for si in range(n):
        ck = f"{order}_{si}" if n > 1 else str(order)
        m = idx.get((file, layout_name, ck))
        if m is None:
            return None
        chunks.append(m)
    return chunks


def _per_order_greedy_candidate(
    file: str,
    *,
    layouts: dict[str, ChunkLayout],
    idx: dict[tuple[str, str, str], ChunkMeas],
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model,
    star_meta: dict,
) -> StackCandidate | None:
    """
    Greedily assign each echelle order an entire layout partition (all subchunks).

    Layouts with incommensurate edges (subchunks_3 vs subchunks_4 vs n3_red_heavy)
    are never mixed within the same order.
    """
    orders = _orders_with_fine_data(file, idx, layouts)
    if not orders:
        return None
    selected: list[ChunkMeas] = []
    for order in orders:
        best_chunks: list[ChunkMeas] | None = None
        best_err = float("inf")
        for layout_name in PER_ORDER_LAYOUTS:
            layout = layouts.get(layout_name)
            if layout is None:
                continue
            order_chunks = _order_subchunks(file, order, layout_name, layout, idx)
            if order_chunks is None:
                continue
            trial = selected + order_chunks
            out = _stack_result_for_meas(
                trial,
                per_object=per_object,
                fallback=fallback,
                intrinsic_model=intrinsic_model,
                star_meta=star_meta,
                min_chunks=MIN_CHUNKS,
            )
            err = float(out["rv_err_calibrated_kms"])
            if np.isfinite(err) and err < best_err:
                best_err = err
                best_chunks = order_chunks
        if best_chunks is None:
            return None
        selected.extend(best_chunks)
    if len(selected) < MIN_CHUNKS:
        return None
    return StackCandidate(name="per_order_greedy", chunks=tuple(selected))


def enumerate_stack_candidates(
    file: str,
    idx: dict[tuple[str, str, str], ChunkMeas],
    *,
    layouts: dict[str, ChunkLayout],
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model,
    star_meta: dict,
) -> list[StackCandidate]:
    """All complete stacking candidates for one exposure."""
    candidates: list[StackCandidate] = []
    seen: set[frozenset[tuple[str, str]]] = set()

    def _add(name: str, chunks: list[ChunkMeas] | tuple[ChunkMeas, ...]) -> None:
        sig = frozenset(m.bias_key for m in chunks)
        if sig in seen:
            return
        seen.add(sig)
        candidates.append(StackCandidate(name=name, chunks=tuple(chunks)))

    for layout_name in ALL_WHOLE_LAYOUTS:
        if layout_name not in layouts:
            continue
        whole = _whole_layout_chunks(file, layout_name, idx)
        if whole is not None:
            _add(f"whole:{layout_name}", whole)

    mix = _per_order_greedy_candidate(
        file,
        layouts=layouts,
        idx=idx,
        per_object=per_object,
        fallback=fallback,
        intrinsic_model=intrinsic_model,
        star_meta=star_meta,
    )
    if mix is not None:
        _add(mix.name, mix.chunks)
    return candidates


def select_best_candidate(
    candidates: list[StackCandidate],
    *,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model,
    star_meta: dict,
) -> tuple[StackCandidate | None, dict]:
    """Pick candidate with lowest calibrated σ_RV (same metric as evaluation)."""
    best: StackCandidate | None = None
    best_out: dict = {}
    best_err = float("inf")
    for cand in candidates:
        out = _stack_result_for_meas(
            cand.chunks,
            per_object=per_object,
            fallback=fallback,
            intrinsic_model=intrinsic_model,
            star_meta=star_meta,
        )
        err = float(out["rv_err_calibrated_kms"])
        if not np.isfinite(err):
            continue
        if err < best_err - 1e-12:
            best_err = err
            best = cand
            best_out = out
    return best, best_out


def adaptive_stack_for_file(
    file: str,
    df: pd.DataFrame,
    *,
    layouts: dict[str, ChunkLayout],
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model,
    star_meta: dict,
) -> tuple[list[ChunkMeas], float, dict]:
    """Select best complete candidate; never worse than best whole layout on same file."""
    sub = df[df["file"].astype(str) == str(file)]
    if sub.empty:
        return [], float("nan"), {}
    idx = _index_by_file_layout(sub)
    candidates = enumerate_stack_candidates(
        file,
        idx,
        layouts=layouts,
        per_object=per_object,
        fallback=fallback,
        intrinsic_model=intrinsic_model,
        star_meta=star_meta,
    )
    best, stack = select_best_candidate(
        candidates,
        per_object=per_object,
        fallback=fallback,
        intrinsic_model=intrinsic_model,
        star_meta=star_meta,
    )
    if best is None:
        return [], float("nan"), {"candidate_name": "", "n_candidates": len(candidates)}

    final = list(best.chunks)
    layout_counts: dict[str, int] = {}
    for m in final:
        layout_counts[m.layout_name] = layout_counts.get(m.layout_name, 0) + 1
    err = float(stack["rv_err_calibrated_kms"])
    return final, err, {
        "rv_calibrated_kms": stack["rv_calibrated_kms"],
        "rv_err_calibrated_kms": err,
        "n_chunks_used": stack["n_chunks_used"],
        "candidate_name": best.name,
        "n_candidates": len(candidates),
        "layout_mix_json": json.dumps(layout_counts),
    }


def _star_meta_for_file(gid: str, meta_tbl: pd.DataFrame) -> dict:
    star_meta = {"logg": np.nan, "mh": np.nan}
    if meta_tbl.empty:
        return star_meta
    sm = meta_tbl[meta_tbl["gaia_dr3_id"] == str(gid)]
    if len(sm):
        star_meta["logg"] = float(sm.iloc[0].get("logg", np.nan))
        star_meta["mh"] = float(sm.iloc[0].get("mh", np.nan))
    return star_meta


def epochs_whole_layout_from_cache(
    meas_df: pd.DataFrame,
    layout_name: str,
    *,
    idx: dict[tuple[str, str, str], ChunkMeas],
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model,
    meta_tbl: pd.DataFrame,
) -> pd.DataFrame:
    """Stack whole-layout epochs from the same measurement cache as adaptive."""
    rows = []
    for file_label in sorted(meas_df["file"].astype(str).unique()):
        whole = _whole_layout_chunks(file_label, layout_name, idx)
        if whole is None:
            continue
        gid = str(whole[0].gaia_dr3_id)
        stack = _stack_result_for_meas(
            whole,
            per_object=per_object,
            fallback=fallback,
            intrinsic_model=intrinsic_model,
            star_meta=_star_meta_for_file(gid, meta_tbl),
        )
        if not np.isfinite(stack.get("rv_calibrated_kms", np.nan)):
            continue
        rows.append(
            {
                "gaia_dr3_id": gid,
                "file": str(file_label),
                "mjd": float(whole[0].mjd),
                "layout": layout_name,
                "rv_calibrated_kms": stack["rv_calibrated_kms"],
                "rv_err_calibrated_kms": stack["rv_err_calibrated_kms"],
                "n_chunks_used": stack["n_chunks_used"],
            }
        )
    return pd.DataFrame(rows)


def _metrics_row(label: str, epochs: pd.DataFrame, *, cohort_files: set[str] | None = None) -> dict:
    sub = epochs
    if cohort_files is not None:
        sub = epochs[epochs["file"].astype(str).isin(cohort_files)]
    if sub.empty:
        return {"layout": label, "n_exposures": 0}
    sigma = summarize_sigma_rv_metrics(sub)
    pairs = relative_pair_table(sub, rv_col="rv_calibrated_kms", err_col="rv_err_calibrated_kms")
    gate = summarize_relative_gate(pairs, goal_kms=0.1)
    row = {
        "layout": label,
        "n_exposures": int(sub["rv_calibrated_kms"].notna().sum()),
        **sigma,
        "relative_median_abs_delta_kms": gate.get("median_abs_delta_rv_kms", np.nan),
        "relative_p90_abs_delta_kms": gate.get("p90_abs_delta_rv_kms", np.nan),
        "n_relative_pairs": gate.get("n_pairs", 0),
        "offline_eval_valid": True,
    }
    row["composite_score"] = composite_score(pd.Series(row))
    return row


def run_adaptive_evaluation(campaign_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    layouts = load_layouts(campaign_dir)
    if "merge_w4" not in layouts:
        raise ValueError("merge_w4 layout YAML required in campaign layouts/")
    meas_df = load_campaign_measurements(campaign_dir, layouts)
    per_object, fallback, intrinsic_model = build_multi_layout_bias_tables(campaign_dir, layouts)
    meta_tbl = load_stellar_metadata(REPO_ROOT / "output")

    epoch_rows = []
    files = sorted(meas_df["file"].astype(str).unique())
    for fi, file_label in enumerate(files):
        if fi % 20 == 0:
            logger.info("Adaptive stack %d/%d", fi, len(files))
        gid = str(meas_df.loc[meas_df["file"].astype(str) == file_label, "gaia_dr3_id"].iloc[0])
        teff = float(meas_df.loc[meas_df["file"].astype(str) == file_label, "teff"].iloc[0])
        mjd = float(meas_df.loc[meas_df["file"].astype(str) == file_label, "mjd"].iloc[0])
        star_meta = _star_meta_for_file(gid, meta_tbl)
        _, sig, info = adaptive_stack_for_file(
            file_label,
            meas_df,
            layouts=layouts,
            per_object=per_object,
            fallback=fallback,
            intrinsic_model=intrinsic_model,
            star_meta=star_meta,
        )
        if not np.isfinite(info.get("rv_calibrated_kms", np.nan)):
            continue
        epoch_rows.append(
            {
                "gaia_dr3_id": gid,
                "file": file_label,
                "mjd": mjd,
                "teff": teff,
                "layout": "adaptive_mix",
                "rv_calibrated_kms": info["rv_calibrated_kms"],
                "rv_err_calibrated_kms": info["rv_err_calibrated_kms"],
                "n_chunks_used": info["n_chunks_used"],
                "candidate_name": info.get("candidate_name", ""),
                "layout_mix_json": info["layout_mix_json"],
            }
        )
    adaptive_epochs = pd.DataFrame(epoch_rows)
    idx = _index_by_file_layout(meas_df)

    compare_layouts = [lay for lay in ALL_WHOLE_LAYOUTS if lay in layouts]
    layout_epochs: dict[str, pd.DataFrame] = {}
    for lay in compare_layouts:
        ep = epochs_whole_layout_from_cache(
            meas_df,
            lay,
            idx=idx,
            per_object=per_object,
            fallback=fallback,
            intrinsic_model=intrinsic_model,
            meta_tbl=meta_tbl,
        )
        if not ep.empty:
            layout_epochs[lay] = ep

    adaptive_files = set(adaptive_epochs["file"].astype(str))
    paired_summary_rows = []
    paired_detail_rows = []

    for lay in compare_layouts:
        if lay not in layout_epochs:
            continue
        base = layout_epochs[lay]
        shared_files = adaptive_files & set(base["file"].astype(str))
        if not shared_files:
            continue
        sub_a = adaptive_epochs[adaptive_epochs["file"].astype(str).isin(shared_files)]
        sub_b = base[base["file"].astype(str).isin(shared_files)]
        row_a = _metrics_row("adaptive_mix", sub_a)
        row_b = _metrics_row(lay, sub_b)
        paired_detail_rows.append(
            {
                "baseline_layout": lay,
                "n_shared_files": len(shared_files),
                "adaptive_median_sigma_rv_kms": row_a.get("median_sigma_rv_kms", np.nan),
                "baseline_median_sigma_rv_kms": row_b.get("median_sigma_rv_kms", np.nan),
                "adaptive_p90_sigma_rv_kms": row_a.get("p90_sigma_rv_kms", np.nan),
                "baseline_p90_sigma_rv_kms": row_b.get("p90_sigma_rv_kms", np.nan),
                "adaptive_rel_median_kms": row_a.get("relative_median_abs_delta_kms", np.nan),
                "baseline_rel_median_kms": row_b.get("relative_median_abs_delta_kms", np.nan),
                "adaptive_composite_score": row_a.get("composite_score", np.nan),
                "baseline_composite_score": row_b.get("composite_score", np.nan),
            }
        )
        row_b["cohort"] = f"shared_with_adaptive"
        row_b["n_shared_with_adaptive"] = len(shared_files)
        paired_summary_rows.append(row_b)

    if paired_detail_rows:
        pd.DataFrame(paired_detail_rows).to_csv(campaign_dir / "adaptive_stack_paired.csv", index=False)

    # Strict common cohort: identical files where adaptive and every whole layout are valid.
    common_files = set(adaptive_files)
    for ep in layout_epochs.values():
        common_files &= set(ep["file"].astype(str))
    common_rows = []
    if common_files:
        for label, epochs in [("adaptive_mix", adaptive_epochs)] + list(layout_epochs.items()):
            common_rows.append(
                _metrics_row(label, epochs, cohort_files=common_files)
                | {"cohort": "common_all_layouts", "n_common_files": len(common_files)}
            )
        common_summary = pd.DataFrame(common_rows).sort_values("composite_score")
        common_summary.to_csv(campaign_dir / "adaptive_stack_common_cohort.csv", index=False)
        summary = common_summary
    else:
        summary = pd.DataFrame()

    # Per-baseline paired cohort (adaptive vs each layout on that layout's shared files).
    paired_cohort_rows = []
    for lay in compare_layouts:
        if lay not in layout_epochs:
            continue
        shared_files = adaptive_files & set(layout_epochs[lay]["file"].astype(str))
        if not shared_files:
            continue
        paired_cohort_rows.append(
            _metrics_row("adaptive_mix", adaptive_epochs, cohort_files=shared_files)
            | {
                "cohort": f"shared_with_{lay}",
                "n_shared_files": len(shared_files),
                "paired_baseline": lay,
            }
        )
        paired_cohort_rows.append(
            _metrics_row(lay, layout_epochs[lay], cohort_files=shared_files)
            | {
                "cohort": f"shared_with_{lay}",
                "n_shared_files": len(shared_files),
                "paired_baseline": lay,
            }
        )
    if paired_cohort_rows:
        pd.DataFrame(paired_cohort_rows).sort_values(["paired_baseline", "layout"]).to_csv(
            campaign_dir / "adaptive_stack_paired_cohorts.csv", index=False
        )

    if layout_epochs:
        pd.concat(layout_epochs.values(), ignore_index=True).to_csv(
            campaign_dir / "whole_layout_epochs_cache.csv", index=False
        )

    return adaptive_epochs, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-dir", type=Path, default=REPO_ROOT / "validation_output" / "chunk_campaign")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    adaptive_epochs, summary = run_adaptive_evaluation(args.campaign_dir)
    out_dir = args.campaign_dir
    adaptive_epochs.to_csv(out_dir / "adaptive_stack_epochs.csv", index=False)
    summary.to_csv(out_dir / "adaptive_stack_comparison.csv", index=False)
    logger.info("Adaptive stack: %d exposures", len(adaptive_epochs))
    if not summary.empty:
        best = summary.iloc[0]
        logger.info(
            "Best (paired/shared cohort): %s composite=%.4f median σ_RV=%.4f p90 σ_RV=%.4f rel=%.4f",
            best["layout"],
            best["composite_score"],
            best["median_sigma_rv_kms"],
            best["p90_sigma_rv_kms"],
            best["relative_median_abs_delta_kms"],
        )
        cols = [
            "layout",
            "cohort",
            "n_exposures",
            "n_shared_with_adaptive",
            "median_sigma_rv_kms",
            "p90_sigma_rv_kms",
            "relative_median_abs_delta_kms",
            "composite_score",
        ]
        show = [c for c in cols if c in summary.columns]
        print(summary[show].to_string(index=False))
    common_path = out_dir / "adaptive_stack_common_cohort.csv"
    if common_path.is_file():
        common = pd.read_csv(common_path)
        logger.info("Common cohort (%d files) — see %s", int(common.iloc[0].get("n_common_files", 0)), common_path)


if __name__ == "__main__":
    main()
