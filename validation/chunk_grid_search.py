#!/usr/bin/env python3
"""
Parametric chunk layout grid search (rough N choices) on existing diagnostics.

Evaluates offline-order-merge coarsening immediately; equal sub-chunk layouts
(N=2,4,…) are written as YAML specs but require a pipeline rerun for valid scores.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.chunk_grid_search \\
    --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \\
    --out-dir validation_output/chunk_grid_search
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.chunk_layout import (  # noqa: E402
    ChunkLayout,
    build_custom_edge_layout,
    build_equal_subchunk_layout,
    build_merge_orders_layout,
    default_parametric_grid,
    save_chunk_layout,
)
from validation.evaluate_chunk_layout import evaluate_layouts  # noqa: E402
from validation.plot_chunk_residuals import (  # noqa: E402
    DEFAULT_CHUNK_MAX_DELTA_KMS,
    DEFAULT_CHUNK_OUTLIER_SIGMA,
)

logger = logging.getLogger(__name__)

# Rough grid defaults — small enough to run without pipeline reruns for merge arm.
DEFAULT_SUBCHUNK_COUNTS = [1, 2, 4]
DEFAULT_MERGE_WIDTHS = [1, 2, 4]


def composite_score(row: pd.Series) -> float:
    """
    Lower is better. Median + p90 σ_RV (typical and worst-case precision) plus relative gate.

    NaN metrics → inf.
    """
    if not bool(row.get("offline_eval_valid", False)):
        return float("inf")
    sigma_med = float(row.get("median_sigma_rv_kms", np.nan))
    sigma_p90 = float(row.get("p90_sigma_rv_kms", np.nan))
    rel = float(row.get("relative_median_abs_delta_kms", np.nan))
    if not all(np.isfinite(x) for x in (sigma_med, sigma_p90, rel)):
        return float("inf")
    return 0.35 * sigma_med + 0.35 * sigma_p90 + 0.30 * rel


def build_grid_layouts(
    *,
    subchunk_counts: list[int],
    merge_widths: list[int],
    include_telluric_split_example: bool,
) -> list[ChunkLayout]:
    layouts = default_parametric_grid(subchunk_counts=subchunk_counts, merge_widths=merge_widths)
    if include_telluric_split_example:
        # Example asymmetric edges (N=2 unequal split) — pipeline-only, for edge-search demos.
        layouts.append(
            build_custom_edge_layout("edges_blue_heavy_2", [0.0, 0.35, 1.0]),
        )
    return layouts


def _plot_grid_summary(summary: pd.DataFrame, out_path: Path) -> None:
    offline = summary[summary["offline_eval_valid"].astype(bool)].copy()
    if offline.empty:
        return
    offline = offline.sort_values("composite_score")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(offline))
    labels = offline["layout"].astype(str).tolist()

    for ax, col, title in zip(
        axes,
        ["median_sigma_rv_kms", "p90_sigma_rv_kms", "relative_median_abs_delta_kms"],
        ["Median σ_RV", "p90 σ_RV (worst cases)", "Relative median |ΔRV|"],
    ):
        y = offline[col].astype(float).values
        ax.bar(x, y, color="C0", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("km/s")
    fig.suptitle("Offline-valid chunk grid (lower is better)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_edge_search_advice(path: Path, *, best_offline: str | None) -> None:
    text = f"""# Searching for optimal chunk edges

## What the rough grid search tested

| Arm | Parameter | Offline? | Meaning |
|-----|-----------|----------|---------|
| **Merge width** | W = 1, 2, 4 adjacent orders | Yes | Coarsen chunks by IVW-merging whole orders |
| **Sub-chunks** | N = 1, 2, 4 equal pixel splits | **No** (needs pipeline) | N+1 edges at 0, 1/N, …, 1 per order |

Best offline layout in latest run: **{best_offline or "see grid_summary.csv"}**

Sub-chunk layouts (`subchunks_2`, `subchunks_4`) are exported under `layouts/` but
scores are duplicated from whole-order until you rerun:

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
python -m darkhunter_rv.pipeline ... --subchunks 4 --instrument APF
python -m validation.plot_chunk_residuals ...
python -m validation.chunk_grid_search --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv'
```

---

## Recommended edge search strategy (after rough N)

Once a rough **N** is chosen (equal splits or merge width), refine **edge locations**
in three stages:

### Stage A — Coarse grid on edge fractions (cheap)

For each order (or a representative subset of orders), try a small set of **N+1**
pixel-fraction edge vectors, e.g. for N=4:

| Layout | Edges |
|--------|-------|
| equal | `[0, 0.25, 0.5, 0.75, 1]` |
| blue-heavy | `[0, 0.35, 0.6, 0.8, 1]` |
| red-heavy | `[0, 0.2, 0.45, 0.7, 1]` |
| telluric-aware | split at cumulative telluric-fraction jumps |

Use `build_custom_edge_layout(name, pixel_edges)` → YAML → pipeline rerun →
`chunk_grid_search` metrics. Keep 3–5 candidates per N, not a full factorial.

### Stage B — Coordinate descent on one edge at a time

1. Fix N and all edges except interior edge `e_k`.
2. Scan `e_k` ∈ (e_{{k-1}}+0.05, e_{{k+1}}−0.05) on a 1-D grid (≈10 points).
3. Score with the same composite metric (relative gate + chunk scatter + bias RMS).
4. Move to next edge; repeat until convergence.

This is efficient when N is small (4–8 chunks per order) and avoids exploding combinatorics.

### Stage C — Telluric-anchored breakpoints

Use `darkhunter_rv.qc.TELLURIC_BANDS` and per-pixel λ:

1. Compute telluric pixel mask per order.
2. Candidate edges = boundaries where telluric fraction in a window crosses 10% / 25%.
3. Snap to nearest pixel index; enforce `min_pixels` per chunk.
4. Evaluate with the validation loop.

This often beats arbitrary equal splits for APF red orders.

### Stage D — ML-assisted ranking (optional)

Train a gradient-boosted regressor on chunk-level rows:

**Features:** order, λ_center, telluric_fraction, mask_line_count, CCF S/N, Teff  
**Target:** |chunk residual| or contribution to exposure scatter

Use predicted mean |residual| to **score** candidate edge sets before expensive pipeline
reruns. BDT does not optimize edges directly — it prioritizes which candidates to run.

---

## Composite score (grid ranking)

```
score = 0.5 × relative_median|ΔRV| + 0.3 × median_chunk_scatter + 0.2 × bias_curve_RMS
```

Lower is better. Tune weights once relative gate pairs are stable.

---

## Files produced by `chunk_grid_search`

| File | Content |
|------|---------|
| `grid_summary.csv` | All layouts, metrics, composite score, rank |
| `layouts/*.yaml` | Layout specs for pipeline |
| `plots/grid_metrics.png` | Bar charts for offline-valid layouts |
| `EDGE_SEARCH_ADVICE.md` | This document |
| `REPORT.md` | Short summary |
"""
    path.write_text(text, encoding="utf-8")


def write_report(path: Path, summary: pd.DataFrame) -> None:
    offline = summary[summary["offline_eval_valid"].astype(bool)].copy()
    lines = [
        "# Chunk parametric grid search",
        "",
        f"- Layouts tested: {len(summary)}",
        f"- Offline-valid: {len(offline)}",
        "",
        "## Rankings (offline-valid only)",
        "",
        "| rank | layout | score | rel median |ΔRV| | chunk scatter | bias RMS |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if len(offline):
        offline = offline.sort_values("composite_score")
        for i, (_, r) in enumerate(offline.iterrows(), start=1):
            lines.append(
                f"| {i} | {r['layout']} | {r['composite_score']:.4f} | "
                f"{r['relative_median_abs_delta_kms']:.3f} | {r['median_chunk_scatter_kms']:.3f} | "
                f"{r['bias_curve_rms_kms']:.3f} |"
            )
    else:
        lines.append("| — | no offline-valid layouts | — | — | — | — |")

    pipeline = summary[~summary["offline_eval_valid"].astype(bool)]
    if len(pipeline):
        lines.extend(
            [
                "",
                "## Pipeline rerun required",
                "",
            ]
        )
        for _, r in pipeline.iterrows():
            lines.append(
                f"- `{r['layout']}` (subchunks={r.get('cfg_subchunks', '?')}) — "
                f"run pipeline with matching `--subchunks`, then re-run grid search"
            )
    lines.extend(["", "See `EDGE_SEARCH_ADVICE.md` for refining edge locations.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_grid_search(
    *,
    diagnostics_glob: str,
    out_dir: Path,
    subchunk_counts: list[int],
    merge_widths: list[int],
    clip_sigma: float,
    clip_max_delta_kms: float,
    min_chunks: int,
    include_telluric_split_example: bool,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    layouts_dir = out_dir / "layouts"
    layouts_dir.mkdir(exist_ok=True)

    layouts = build_grid_layouts(
        subchunk_counts=subchunk_counts,
        merge_widths=merge_widths,
        include_telluric_split_example=include_telluric_split_example,
    )
    for lay in layouts:
        save_chunk_layout(lay, layouts_dir / f"{lay.name}.yaml")

    summary = evaluate_layouts(
        layouts,
        diagnostics_glob,
        clip_sigma=clip_sigma,
        clip_max_delta_kms=clip_max_delta_kms,
        min_chunks=min_chunks,
    )
    summary["composite_score"] = summary.apply(composite_score, axis=1)
    summary["rank"] = summary["composite_score"].rank(method="min").astype(int)
    summary = summary.sort_values(["composite_score", "layout"])

    summary.to_csv(out_dir / "grid_summary.csv", index=False)
    (out_dir / "grid_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    _plot_grid_summary(summary, out_dir / "plots" / "grid_metrics.png")
    best = None
    offline = summary[summary["offline_eval_valid"].astype(bool)]
    if len(offline):
        best = str(offline.sort_values("composite_score").iloc[0]["layout"])
    write_edge_search_advice(out_dir / "EDGE_SEARCH_ADVICE.md", best_offline=best)
    write_report(out_dir / "REPORT.md", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostics-glob", default="output/Gaia_DR3_*_diagnostics.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "validation_output" / "chunk_grid_search")
    ap.add_argument(
        "--subchunk-counts",
        default=",".join(str(x) for x in DEFAULT_SUBCHUNK_COUNTS),
        help="Comma-separated equal sub-chunk counts N (default: 1,2,4)",
    )
    ap.add_argument(
        "--merge-widths",
        default=",".join(str(x) for x in DEFAULT_MERGE_WIDTHS),
        help="Comma-separated adjacent-order merge widths (default: 1,2,4)",
    )
    ap.add_argument("--chunk-outlier-sigma", type=float, default=DEFAULT_CHUNK_OUTLIER_SIGMA)
    ap.add_argument("--chunk-max-delta-kms", type=float, default=DEFAULT_CHUNK_MAX_DELTA_KMS)
    ap.add_argument("--min-chunks", type=int, default=3)
    ap.add_argument("--include-telluric-split-example", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    subchunk_counts = [int(x.strip()) for x in args.subchunk_counts.split(",") if x.strip()]
    merge_widths = [int(x.strip()) for x in args.merge_widths.split(",") if x.strip()]

    summary = run_grid_search(
        diagnostics_glob=args.diagnostics_glob,
        out_dir=args.out_dir,
        subchunk_counts=subchunk_counts,
        merge_widths=merge_widths,
        clip_sigma=args.chunk_outlier_sigma,
        clip_max_delta_kms=args.chunk_max_delta_kms,
        min_chunks=args.min_chunks,
        include_telluric_split_example=args.include_telluric_split_example,
    )
    offline = summary[summary["offline_eval_valid"].astype(bool)]
    if len(offline):
        best = offline.sort_values("composite_score").iloc[0]
        logger.info(
            "Best offline layout: %s (score=%.4f, rel=%.3f km/s)",
            best["layout"],
            best["composite_score"],
            best["relative_median_abs_delta_kms"],
        )
    logger.info("Grid search written to %s", args.out_dir)


if __name__ == "__main__":
    main()
