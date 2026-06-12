#!/usr/bin/env python3
"""
Per-order chunk baseline: best equal split N (1–4, optionally 8) and merge widths per echelle order.

Uses campaign measurement cache (no pipeline). Emits marginal scores, greedy per-exposure
assignment, and campaign median σ_RV vs global layouts.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python3 -m validation.per_order_chunk_baseline \\
    --campaign-dir validation_output/chunk_campaign
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.chunk_adaptive_stack import (  # noqa: E402
    MIN_CHUNKS,
    ChunkMeas,
    _index_by_file_layout,
    _lookup_layout_bias,
    _stack_result_for_meas,
    build_multi_layout_bias_tables,
    load_campaign_measurements,
    load_layouts,
)
from validation.chunk_calibration import summarize_sigma_rv_metrics  # noqa: E402
from validation.chunk_layout import (  # noqa: E402
    ChunkLayout,
    apf_valid_orders,
    merge_order_groups,
)
from validation.chunk_bias_lib import load_stellar_metadata  # noqa: E402
from darkhunter_rv.chunking import parse_chunk_key  # noqa: E402

logger = logging.getLogger(__name__)

SPLIT_NS = (1, 2, 3, 4, 8)
MERGE_LAYOUTS = ("merge_w2", "merge_w3", "merge_w4")
WHOLE_ORDER_LAYOUT = "whole_order"
FINE_SPLIT_LAYOUT = "subchunks_4"


def is_merge_meas(meas: ChunkMeas) -> bool:
    return meas.layout_name in MERGE_LAYOUTS or str(meas.chunk_key).startswith("merge")


def stack_norm_factor(chunks: list[ChunkMeas] | tuple[ChunkMeas, ...]) -> float:
    """
    Normalize stacked σ to n=1 (single sub-chunk / single-order equivalent).

    - Sub-chunk measurements: divide by √n_sub (n_sub = count of non-merge chunks).
    - Merge super-chunks: multiply by √(orders merged) per merge measurement.

    Approximate — ignores per-chunk IVW weight differences.
    """
    n_sub = sum(1 for m in chunks if not is_merge_meas(m))
    factor = 1.0 / np.sqrt(max(n_sub, 1))
    for m in chunks:
        if is_merge_meas(m):
            factor *= np.sqrt(max(len(m.orders), 1))
    return float(factor)


def sigma_rv_normalized_kms(
    sigma_rv_kms: float,
    chunks: list[ChunkMeas] | tuple[ChunkMeas, ...],
) -> float:
    if not np.isfinite(sigma_rv_kms) or sigma_rv_kms <= 0:
        return float("nan")
    return float(sigma_rv_kms * stack_norm_factor(chunks))


@dataclass(frozen=True)
class OrderCandidate:
    """One chunking option for a single echelle order on one exposure."""

    name: str
    layout_name: str
    chunks: tuple[ChunkMeas, ...]
    covered_orders: frozenset[int]

    @property
    def subchunk_n(self) -> int | None:
        if self.name.startswith("subchunks_"):
            return int(self.name.split("_", 1)[1])
        return None

    @property
    def n_chunks(self) -> int:
        """RV measurements contributed (1 for merge super-chunks)."""
        return len(self.chunks)


def _candidate_name_subchunks(n: int) -> str:
    return f"subchunks_{n}"


def _candidate_name_merge(layout_name: str) -> str:
    return layout_name


def _ivw_rv_err(rv: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(rv) & np.isfinite(err) & (err > 0)
    if not np.any(ok):
        return float("nan"), float("nan")
    w = 1.0 / err[ok] ** 2
    mu = float(np.sum(w * rv[ok]) / np.sum(w))
    sig = float(1.0 / np.sqrt(np.sum(w)))
    return mu, sig


def _synthetic_whole_order_chunk(
    file: str,
    order: int,
    idx: dict[tuple[str, str, str], ChunkMeas],
) -> ChunkMeas | None:
    """Coarsen finest available equal split to N=1 via IVW (offline proxy for whole_order)."""
    for n_sub in (8, 4, 3, 2):
        layout_name = f"subchunks_{n_sub}"
        parts: list[ChunkMeas] = []
        for si in range(n_sub):
            m = idx.get((file, layout_name, f"{order}_{si}"))
            if m is None:
                parts = []
                break
            parts.append(m)
        if len(parts) != n_sub:
            continue
        rv = np.array([p.rv_kms for p in parts], float)
        err = np.array([p.rv_err_kms for p in parts], float)
        mu, sig = _ivw_rv_err(rv, err)
        if not np.isfinite(mu) or not np.isfinite(sig):
            continue
        ref = parts[0]
        return ChunkMeas(
            layout_name=layout_name,
            chunk_key=str(order),
            rv_kms=mu,
            rv_err_kms=sig,
            orders=frozenset({order}),
            qc_pass=True,
            file=file,
            gaia_dr3_id=ref.gaia_dr3_id,
            mjd=ref.mjd,
            teff=ref.teff,
        )
    return None


def _chunks_for_split(
    file: str,
    order: int,
    n: int,
    idx: dict[tuple[str, str, str], ChunkMeas],
    layouts: dict[str, ChunkLayout],
) -> list[ChunkMeas] | None:
    if n == 1:
        # Prefer pipelined whole_order (one chunk = one echelle order).
        m = idx.get((file, WHOLE_ORDER_LAYOUT, str(order)))
        if m is not None:
            return [m]
        m = _synthetic_whole_order_chunk(file, order, idx)
        return [m] if m is not None else None
    layout_name = f"subchunks_{n}"
    if layout_name not in layouts:
        return None
    layout = layouts[layout_name]
    n_chunks = layout.n_chunks_per_order()
    out: list[ChunkMeas] = []
    for si in range(n_chunks):
        ck = f"{order}_{si}" if n_chunks > 1 else str(order)
        m = idx.get((file, layout_name, ck))
        if m is None:
            return None
        out.append(m)
    return out


def _orders_with_split_data(
    file: str,
    idx: dict[tuple[str, str, str], ChunkMeas],
    layouts: dict[str, ChunkLayout],
) -> list[int]:
    """Echelle orders with at least one fine-split layout in cache for this file."""
    orders: set[int] = set()
    for n in (4, 2, 3, 8):
        layout_name = f"subchunks_{n}"
        if layout_name not in layouts:
            continue
        for (_f, lay, ck), _m in idx.items():
            if _f != file or lay != layout_name:
                continue
            order, _, kind = parse_chunk_key(ck)
            if order is None or kind == "merge":
                continue
            if _chunks_for_split(file, int(order), n, idx, layouts) is not None:
                orders.add(int(order))
    return sorted(orders)


def _orders_in_merge_group(gi: int, layout: ChunkLayout) -> frozenset[int]:
    if not layout.merge_orders or gi < 0 or gi >= len(layout.merge_orders):
        return frozenset()
    lo, hi = layout.merge_orders[gi]
    valid = set(apf_valid_orders())
    return frozenset(o for o in range(int(lo), int(hi) + 1) if o in valid)


def _chunks_for_merge(
    file: str,
    order: int,
    merge_layout_name: str,
    layout: ChunkLayout,
    idx: dict[tuple[str, str, str], ChunkMeas],
) -> list[ChunkMeas] | None:
    gi = merge_order_groups(order, layout.merge_orders)
    if gi is None:
        return None
    m = idx.get((file, merge_layout_name, f"merge_{gi}"))
    if m is None:
        return None
    return [m]


def enumerate_order_candidates(
    file: str,
    order: int,
    idx: dict[tuple[str, str, str], ChunkMeas],
    layouts: dict[str, ChunkLayout],
    *,
    split_ns: tuple[int, ...] = SPLIT_NS,
) -> list[OrderCandidate]:
    cands: list[OrderCandidate] = []
    for n in split_ns:
        chunks = _chunks_for_split(file, order, n, idx, layouts)
        if chunks is None:
            continue
        cands.append(
            OrderCandidate(
                name=_candidate_name_subchunks(n),
                layout_name=chunks[0].layout_name,
                chunks=tuple(chunks),
                covered_orders=frozenset({order}),
            )
        )
    for merge_name in MERGE_LAYOUTS:
        layout = layouts.get(merge_name)
        if layout is None:
            continue
        chunks = _chunks_for_merge(file, order, merge_name, layout, idx)
        if chunks is None:
            continue
        gi = merge_order_groups(order, layout.merge_orders)
        covered = _orders_in_merge_group(int(gi), layout) if gi is not None else frozenset({order})
        cands.append(
            OrderCandidate(
                name=_candidate_name_merge(merge_name),
                layout_name=merge_name,
                chunks=tuple(chunks),
                covered_orders=covered,
            )
        )
    return cands


@dataclass
class GreedyStep:
    order: int
    candidate: OrderCandidate
    sigma_rv_kms: float
    sigma_rv_normalized_kms: float
    norm_factor: float


def greedy_assign_file(
    file: str,
    idx: dict[tuple[str, str, str], ChunkMeas],
    layouts: dict[str, ChunkLayout],
    *,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model,
    star_meta: dict,
    split_ns: tuple[int, ...] = SPLIT_NS,
) -> tuple[list[GreedyStep], dict]:
    """Greedy per-order assignment for one exposure; returns steps and final stack metrics."""
    orders = _orders_with_split_data(file, idx, layouts)
    if not orders:
        return [], {}
    order_set = set(orders)

    covered: set[int] = set()
    steps: list[GreedyStep] = []
    selected_chunks: list[ChunkMeas] = []

    for order in sorted(orders):
        if order in covered:
            continue
        best_cand: OrderCandidate | None = None
        best_norm = float("inf")
        best_err = float("nan")
        best_factor = float("nan")
        for cand in enumerate_order_candidates(file, order, idx, layouts, split_ns=split_ns):
            if not cand.covered_orders <= order_set:
                continue
            trial = selected_chunks + list(cand.chunks)
            out = _stack_result_for_meas(
                trial,
                per_object=per_object,
                fallback=fallback,
                intrinsic_model=intrinsic_model,
                star_meta=star_meta,
                min_chunks=1,
            )
            err = float(out["rv_err_calibrated_kms"])
            norm_factor = stack_norm_factor(trial)
            norm = sigma_rv_normalized_kms(err, trial)
            if np.isfinite(norm) and norm < best_norm:
                best_norm = norm
                best_err = err
                best_factor = norm_factor
                best_cand = cand
        if best_cand is None:
            continue
        selected_chunks.extend(best_cand.chunks)
        covered |= set(best_cand.covered_orders)
        steps.append(
            GreedyStep(
                order=order,
                candidate=best_cand,
                sigma_rv_kms=best_err,
                sigma_rv_normalized_kms=best_norm,
                norm_factor=best_factor,
            )
        )

    if len(selected_chunks) < MIN_CHUNKS:
        return [], {}
    stack = _stack_result_for_meas(
        selected_chunks,
        per_object=per_object,
        fallback=fallback,
        intrinsic_model=intrinsic_model,
        star_meta=star_meta,
        min_chunks=MIN_CHUNKS,
    )
    stack["sigma_rv_normalized_kms"] = sigma_rv_normalized_kms(
        float(stack["rv_err_calibrated_kms"]), selected_chunks
    )
    stack["norm_factor"] = stack_norm_factor(selected_chunks)
    return steps, stack


def augment_fallback_for_coarse_keys(fallback: pd.DataFrame) -> pd.DataFrame:
    """Add whole-order chunk_key rows (median bias/stat) for split_1 proxy lookup."""
    if fallback.empty:
        return fallback
    extra: list[dict] = []
    existing = set(
        zip(fallback["layout_name"].astype(str), fallback["chunk_key"].astype(str))
    )
    for layout_name, grp in fallback.groupby("layout_name", sort=False):
        if not str(layout_name).startswith("subchunks_"):
            continue
        by_order: dict[int, list[pd.Series]] = {}
        for _, r in grp.iterrows():
            order, _, kind = parse_chunk_key(str(r["chunk_key"]))
            if order is None or kind == "merge":
                continue
            by_order.setdefault(int(order), []).append(r)
        for order, rs in by_order.items():
            ck = str(order)
            if (str(layout_name), ck) in existing:
                continue
            biases = [float(x["bias_kms"]) for x in rs if np.isfinite(x.get("bias_kms", np.nan))]
            stats = [float(x["statistical_err_kms"]) for x in rs if np.isfinite(x.get("statistical_err_kms", np.nan))]
            intr = [float(x["intrinsic_scatter_kms"]) for x in rs if np.isfinite(x.get("intrinsic_scatter_kms", np.nan))]
            if not biases:
                continue
            extra.append(
                {
                    "layout_name": str(layout_name),
                    "chunk_key": ck,
                    "bias_kms": float(np.median(biases)),
                    "statistical_err_kms": float(np.median(stats)) if stats else float("nan"),
                    "intrinsic_scatter_kms": float(np.median(intr)) if intr else 0.0,
                }
            )
    if not extra:
        return fallback
    return pd.concat([fallback, pd.DataFrame(extra)], ignore_index=True)


def marginal_scores(
    meas_df: pd.DataFrame,
    *,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    layouts: dict[str, ChunkLayout],
    intrinsic_model,
    split_ns: tuple[int, ...] = SPLIT_NS,
) -> pd.DataFrame:
    """
    Campaign-wide scores per (order, candidate).

    Primary ``marginal_score``: median σ_norm per (order, candidate). Sub-chunks:
    σ_norm = σ_stack / √n_sub. Merges: σ_norm = σ_stack × √(orders merged). Target n=1.
    Approximate (ignores IVW weights). Secondary: |residual|.
    """
    idx = _index_by_file_layout(meas_df)
    rows: list[dict] = []
    idx_all = _index_by_file_layout(meas_df)
    orders = sorted(
        {
            o
            for f in meas_df["file"].astype(str).unique()
            for o in _orders_with_split_data(f, idx_all, layouts)
        }
    )
    meta_by_file: dict[str, dict] = {}
    for file_label in meas_df["file"].astype(str).unique():
        sub = meas_df[meas_df["file"].astype(str) == file_label]
        meta_by_file[file_label] = {"teff": float(sub.iloc[0]["teff"])}

    for order in orders:
        abs_res: dict[str, list[float]] = {}
        norm_sig: dict[str, list[float]] = {}
        raw_sig: dict[str, list[float]] = {}
        cand_norm_factor: dict[str, float] = {}
        subchunks_1_proxy: dict[str, bool] = {}
        for file_label in meas_df["file"].astype(str).unique():
            for cand in enumerate_order_candidates(file_label, order, idx, layouts, split_ns=split_ns):
                order_chunks = [m for m in cand.chunks if order in m.orders]
                if not order_chunks:
                    continue
                cand_norm_factor[cand.name] = stack_norm_factor(order_chunks)
                out = _stack_result_for_meas(
                    order_chunks,
                    per_object=per_object,
                    fallback=fallback,
                    intrinsic_model=intrinsic_model,
                    star_meta=meta_by_file.get(file_label, {}),
                    min_chunks=1,
                )
                err = float(out["rv_err_calibrated_kms"])
                norm = sigma_rv_normalized_kms(err, order_chunks)
                if np.isfinite(norm):
                    norm_sig.setdefault(cand.name, []).append(norm)
                if np.isfinite(err):
                    raw_sig.setdefault(cand.name, []).append(err)
                if cand.name == "subchunks_1" and cand.chunks:
                    m0 = cand.chunks[0]
                    subchunks_1_proxy[cand.name] = m0.layout_name != WHOLE_ORDER_LAYOUT
                for m in order_chunks:
                    bias = _lookup_layout_bias(
                        m.gaia_dr3_id, m.layout_name, m.chunk_key, per_object, fallback
                    )
                    if not np.isfinite(bias):
                        continue
                    abs_res.setdefault(cand.name, []).append(abs(float(m.rv_kms) - bias))
        for cand_name in sorted(set(norm_sig) | set(abs_res)):
            norm_list = norm_sig.get(cand_name, [])
            res_list = abs_res.get(cand_name, [])
            rows.append(
                {
                    "order": order,
                    "candidate": cand_name,
                    "median_sigma_normalized_kms": float(np.median(norm_list)) if norm_list else float("nan"),
                    "median_sigma_stack_kms": float(np.median(raw_sig.get(cand_name, [])))
                    if raw_sig.get(cand_name)
                    else float("nan"),
                    "median_abs_residual_kms": float(np.median(res_list)) if res_list else float("nan"),
                    "norm_factor": float(cand_norm_factor.get(cand_name, np.nan)),
                    "n_exposures": len(norm_list),
                    "marginal_score": float(np.median(norm_list)) if norm_list else float("nan"),
                    "subchunks_1_is_proxy": bool(subchunks_1_proxy.get(cand_name, False)),
                }
            )
    return pd.DataFrame(rows)


def needs_subchunks_8_from_map(chunk_map: pd.DataFrame, *, has_subchunks_8: bool) -> bool:
    """True when any order prefers subchunks_4 and subchunks_8 is not yet available."""
    if has_subchunks_8 or chunk_map.empty:
        return False
    return bool((chunk_map["choice"].astype(str) == "subchunks_4").any())


def _plot_heatmap(scores: pd.DataFrame, out_path: Path) -> None:
    if scores.empty:
        return
    pivot = scores.pivot_table(
        index="order", columns="candidate", values="marginal_score", aggfunc="first"
    )
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.15 * len(pivot))))
    data = pivot.astype(float).values
    im = ax.imshow(data, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int), fontsize=7)
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Echelle order")
    ax.set_title("Per-order marginal score (median σ_norm, lower better)")
    fig.colorbar(im, ax=ax, label="km/s")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_choice_counts(chunk_map: pd.DataFrame, out_path: Path) -> None:
    if chunk_map.empty:
        return
    counts = chunk_map["choice"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color="C0", alpha=0.85)
    ax.set_ylabel("Number of orders")
    ax.set_xlabel("Greedy winner (modal across exposures)")
    ax.set_title("Per-order chunk choice distribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_baseline(
    campaign_dir: Path,
    *,
    split_ns: tuple[int, ...] = SPLIT_NS,
) -> dict[str, pd.DataFrame | bool]:
    campaign_dir = Path(campaign_dir)
    layouts = load_layouts(campaign_dir)
    from validation.chunk_layout import build_equal_subchunk_layout

    for n in split_ns:
        if n > 1:
            name = f"subchunks_{n}"
            if name not in layouts:
                layouts[name] = build_equal_subchunk_layout(n)
    if 1 in split_ns and WHOLE_ORDER_LAYOUT not in layouts:
        layouts[WHOLE_ORDER_LAYOUT] = ChunkLayout(name=WHOLE_ORDER_LAYOUT, subchunks=1)

    meas_df = load_campaign_measurements(campaign_dir, layouts)
    has_whole_order = bool((meas_df["layout_name"].astype(str) == WHOLE_ORDER_LAYOUT).any())
    if meas_df.empty:
        raise ValueError(f"No measurements in {campaign_dir}")

    per_object, fallback, intrinsic_model = build_multi_layout_bias_tables(campaign_dir, layouts)
    fallback = augment_fallback_for_coarse_keys(fallback)
    meta_tbl = load_stellar_metadata(REPO_ROOT / "output")
    idx = _index_by_file_layout(meas_df)

    scores = marginal_scores(
        meas_df,
        per_object=per_object,
        fallback=fallback,
        layouts=layouts,
        intrinsic_model=intrinsic_model,
        split_ns=split_ns,
    )

    greedy_rows: list[dict] = []
    epoch_rows: list[dict] = []
    per_file_choices: list[list[tuple[int, str]]] = []

    for file_label in sorted(meas_df["file"].astype(str).unique()):
        gid = str(meas_df[meas_df["file"].astype(str) == file_label].iloc[0]["gaia_dr3_id"])
        star_meta = {"logg": np.nan, "mh": np.nan}
        if not meta_tbl.empty:
            sm = meta_tbl[meta_tbl["gaia_dr3_id"] == gid]
            if len(sm):
                star_meta["logg"] = float(sm.iloc[0].get("logg", np.nan))
                star_meta["mh"] = float(sm.iloc[0].get("mh", np.nan))

        steps, stack = greedy_assign_file(
            file_label,
            idx,
            layouts,
            per_object=per_object,
            fallback=fallback,
            intrinsic_model=intrinsic_model,
            star_meta=star_meta,
            split_ns=split_ns,
        )
        if not steps or not np.isfinite(stack.get("rv_err_calibrated_kms", np.nan)):
            continue
        file_choice_list: list[tuple[int, str]] = []
        for step in steps:
            pick = step.candidate
            for o in pick.covered_orders:
                greedy_rows.append(
                    {
                        "file": file_label,
                        "order": int(o),
                        "choice": pick.name,
                        "layout_name": pick.layout_name,
                        "chunk_keys": ",".join(m.chunk_key for m in pick.chunks),
                        "greedy_step_sigma_rv_kms": step.sigma_rv_kms,
                        "greedy_step_sigma_normalized_kms": step.sigma_rv_normalized_kms,
                        "norm_factor_in_stack": step.norm_factor,
                    }
                )
                file_choice_list.append((int(o), pick.name))

        per_file_choices.append(file_choice_list)
        ref = steps[0].candidate.chunks[0]
        n_used = int(stack["n_chunks_used"])
        err_final = float(stack["rv_err_calibrated_kms"])
        epoch_rows.append(
            {
                "gaia_dr3_id": gid,
                "file": file_label,
                "mjd": ref.mjd,
                "teff": ref.teff,
                "rv_calibrated_kms": stack["rv_calibrated_kms"],
                "rv_err_calibrated_kms": err_final,
                "sigma_rv_normalized_kms": float(stack.get("sigma_rv_normalized_kms", np.nan)),
                "n_chunks_used": n_used,
                "norm_factor": float(stack.get("norm_factor", np.nan)),
                "layout_mix_json": json.dumps(
                    dict(
                        Counter(
                            m.layout_name
                            for step in steps
                            for m in step.candidate.chunks
                        )
                    )
                ),
            }
        )

    epochs = pd.DataFrame(epoch_rows)
    summary_metrics = summarize_sigma_rv_metrics(epochs) if not epochs.empty else {}
    if not epochs.empty and "sigma_rv_normalized_kms" in epochs.columns:
        norm = epochs["sigma_rv_normalized_kms"].astype(float)
        norm_ok = norm[np.isfinite(norm) & (norm > 0)]
        if len(norm_ok):
            summary_metrics["median_sigma_rv_normalized_kms"] = float(np.median(norm_ok))
            summary_metrics["p90_sigma_rv_normalized_kms"] = float(np.percentile(norm_ok, 90))

    # Modal choice per order across exposures
    order_choice_counts: dict[int, Counter[str]] = {}
    for file_choices in per_file_choices:
        for order, choice in file_choices:
            order_choice_counts.setdefault(order, Counter())[choice] += 1
    map_rows = []
    for order in sorted(order_choice_counts):
        ctr = order_choice_counts[order]
        choice, count = ctr.most_common(1)[0]
        map_rows.append(
            {
                "order": order,
                "choice": choice,
                "n_files": count,
                "frac_files": count / max(len(per_file_choices), 1),
                "n_candidates_tried": len(ctr),
            }
        )
    chunk_map = pd.DataFrame(map_rows)

    grid_path = campaign_dir / "campaign_grid_summary.csv"
    comparisons: dict[str, float] = {}
    if grid_path.is_file():
        grid = pd.read_csv(grid_path)
        for layout in ("subchunks_4", "n3_red_heavy", "adaptive_mix"):
            sub = grid[grid["layout"].astype(str) == layout]
            if len(sub):
                comparisons[f"ref_{layout}_median_sigma_rv_kms"] = float(
                    sub.iloc[0]["median_sigma_rv_kms"]
                )

    has_subchunks_8 = "subchunks_8" in layouts and any(
        meas_df["layout_name"].astype(str) == "subchunks_8"
    )
    needs_subchunks_8 = needs_subchunks_8_from_map(chunk_map, has_subchunks_8=has_subchunks_8)

    subchunks_1_proxy = (
        bool(scores.get("subchunks_1_is_proxy", pd.Series(dtype=bool)).any())
        if not scores.empty
        else True
    )
    summary_row = {
        "n_exposures": len(epochs),
        "needs_subchunks_8": bool(needs_subchunks_8),
        "has_whole_order_in_cache": has_whole_order,
        "subchunks_1_uses_ivw_proxy": subchunks_1_proxy and not has_whole_order,
        **summary_metrics,
        **comparisons,
    }
    summary = pd.DataFrame([summary_row])

    out_scores = campaign_dir / "per_order_candidate_scores.csv"
    out_map = campaign_dir / "per_order_chunk_map.csv"
    out_greedy = campaign_dir / "per_order_greedy_detail.csv"
    out_epochs = campaign_dir / "per_order_greedy_epochs.csv"
    out_summary = campaign_dir / "per_order_chunk_summary.csv"
    scores.to_csv(out_scores, index=False)
    chunk_map.to_csv(out_map, index=False)
    pd.DataFrame(greedy_rows).to_csv(out_greedy, index=False)
    epochs.to_csv(out_epochs, index=False)
    summary.to_csv(out_summary, index=False)

    plot_dir = campaign_dir / "plots"
    _plot_heatmap(scores, plot_dir / "per_order_candidate_heatmap.png")
    _plot_choice_counts(chunk_map, plot_dir / "per_order_choice_counts.png")

    logger.info(
        "Per-order baseline: %d exposures, median σ_RV=%.4f km/s, needs_subchunks_8=%s",
        len(epochs),
        summary_metrics.get("median_sigma_rv_kms", float("nan")),
        needs_subchunks_8,
    )
    return {
        "scores": scores,
        "chunk_map": chunk_map,
        "epochs": epochs,
        "summary": summary,
        "needs_subchunks_8": needs_subchunks_8,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-order N + merge chunk baseline from campaign cache.")
    p.add_argument(
        "--campaign-dir",
        type=Path,
        default=REPO_ROOT / "validation_output" / "chunk_campaign",
    )
    p.add_argument(
        "--split-ns",
        type=str,
        default="1,2,3,4",
        help="Comma-separated equal split counts (add 8 after subchunks_8 campaign).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    split_ns = tuple(int(x.strip()) for x in args.split_ns.split(",") if x.strip())
    run_baseline(args.campaign_dir, split_ns=split_ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
