#!/usr/bin/env python3
"""
Evaluate candidate chunk layouts on existing diagnostics (offline rebinning).

Supports order-merging layouts immediately. Sub-chunk splits require a pipeline
rerun with ``--subchunks N`` or a custom layout YAML.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.evaluate_chunk_layout \\
    --layouts calibration/chunk_layouts/*.yaml \\
    --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \\
    --out-dir validation_output/chunk_layout_eval
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from glob import glob as glob_paths
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.chunk_bias_lib import load_stellar_metadata, sample_mean_bias_curve  # noqa: E402
from validation.chunk_calibration import (  # noqa: E402
    build_intrinsic_scatter_model,
    build_layout_fallback_tables,
    stack_calibrated_exposure,
    summarize_relative_gate,
    summarize_sigma_rv_metrics,
    relative_pair_table,
)
from validation.chunk_layout import ChunkLayout, load_chunk_layout, layout_to_dict, rebinned_chunk_rows  # noqa: E402
from validation.plot_chunk_residuals import (  # noqa: E402
    DEFAULT_CHUNK_MAX_DELTA_KMS,
    DEFAULT_CHUNK_OUTLIER_SIGMA,
    _load_chunk_rows,
    apply_spectrum_chunk_outlier_clip,
    _summarize_chunks_per_object,
)

logger = logging.getLogger(__name__)


def _layout_metrics(
    tab: pd.DataFrame,
    layout_name: str,
    *,
    clip_sigma: float,
    clip_max_delta_kms: float,
    min_chunks: int,
) -> dict:
    if tab.empty:
        return {"layout": layout_name, "n_exposures": 0}

    tab = apply_spectrum_chunk_outlier_clip(tab, nsigma=clip_sigma, max_delta_kms=clip_max_delta_kms)
    meta = load_stellar_metadata(REPO_ROOT / "output")
    if not meta.empty:
        tab = tab.merge(meta, on="gaia_dr3_id", how="left")
    bias_rows = []
    for gid, obj in tab.groupby("gaia_dr3_id"):
        summ = _summarize_chunks_per_object(obj[obj["chunk_kept"].astype(bool)] if "chunk_kept" in obj.columns else obj)
        if summ.empty:
            continue
        summ["gaia_dr3_id"] = str(gid)
        bias_rows.append(summ)
    if not bias_rows:
        return {"layout": layout_name, "n_exposures": 0}
    bias = pd.concat(bias_rows, ignore_index=True)
    teff_by_star = tab.groupby("gaia_dr3_id", as_index=False)["teff"].median()
    bias = bias.merge(teff_by_star, on="gaia_dr3_id", how="left")
    if not meta.empty:
        extra = meta[["gaia_dr3_id", "logg", "mh"]].copy()
        if "teff_gaia" in meta.columns:
            extra["teff_gaia"] = meta["teff_gaia"]
        bias = bias.merge(extra, on="gaia_dr3_id", how="left")
        if "teff_gaia" in bias.columns:
            miss = ~np.isfinite(bias["teff"].astype(float))
            bias.loc[miss, "teff"] = bias.loc[miss, "teff_gaia"]
    curve = sample_mean_bias_curve(bias)
    curve_rms = float(np.sqrt(np.mean(curve["sample_mean_bias_kms"].astype(float) ** 2)))

    _, fallback = build_layout_fallback_tables(bias)
    intrinsic_model = build_intrinsic_scatter_model(bias)

    epoch_rows = []
    for file_label, g in tab.groupby("file", sort=False):
        gid = str(g.iloc[0]["gaia_dr3_id"])
        per_obj = bias[bias["gaia_dr3_id"] == gid]
        stack = stack_calibrated_exposure(
            g,
            per_object=per_obj,
            fallback=fallback,
            intrinsic_model=intrinsic_model,
            min_chunks=min_chunks,
        )
        r0 = g.iloc[0]
        epoch_rows.append(
            {
                "gaia_dr3_id": str(r0["gaia_dr3_id"]),
                "file": str(file_label),
                "mjd": float(r0["mjd"]),
                "rv_calibrated_kms": stack["rv_calibrated_kms"],
                "rv_err_calibrated_kms": stack["rv_err_calibrated_kms"],
                "sigma_rv_core90_kms": stack["sigma_rv_core90_kms"],
                "chunk_scatter_calibrated_kms": stack["chunk_scatter_calibrated_kms"],
                "n_chunks_used": stack["n_chunks_used"],
                "n_chunks_core90": stack["n_chunks_core90"],
            }
        )
    epochs = pd.DataFrame(epoch_rows)
    sc = epochs["chunk_scatter_calibrated_kms"].astype(float)
    sc_ok = sc[np.isfinite(sc)]
    sigma_metrics = summarize_sigma_rv_metrics(epochs)

    pairs = relative_pair_table(epochs, rv_col="rv_calibrated_kms", err_col="rv_err_calibrated_kms")
    gate = summarize_relative_gate(pairs, goal_kms=0.1)

    return {
        "layout": layout_name,
        "n_exposures": int(epochs["rv_calibrated_kms"].notna().sum()),
        "n_chunk_bias_rows": int(len(bias)),
        "bias_curve_rms_kms": curve_rms,
        "min_sigma_rv_kms": sigma_metrics["min_sigma_rv_kms"],
        "p10_sigma_rv_kms": sigma_metrics["p10_sigma_rv_kms"],
        "median_sigma_rv_kms": sigma_metrics["median_sigma_rv_kms"],
        "p90_sigma_rv_kms": sigma_metrics["p90_sigma_rv_kms"],
        "min_sigma_rv_core90_kms": sigma_metrics["min_sigma_rv_core90_kms"],
        "median_chunk_scatter_kms": float(np.median(sc_ok)) if len(sc_ok) else float("nan"),
        "p90_chunk_scatter_kms": float(np.percentile(sc_ok, 90)) if len(sc_ok) else float("nan"),
        "relative_median_abs_delta_kms": gate.get("median_abs_delta_rv_kms", float("nan")),
        "relative_p90_abs_delta_kms": gate.get("p90_abs_delta_rv_kms", float("nan")),
        "relative_frac_below_0p1_kms": gate.get("frac_below_goal", float("nan")),
        "n_relative_pairs": gate.get("n_pairs", 0),
    }


def evaluate_layout(
    layout: ChunkLayout,
    base: pd.DataFrame,
    *,
    clip_sigma: float,
    clip_max_delta_kms: float,
    min_chunks: int,
    rebinned: pd.DataFrame | None = None,
) -> dict:
    if rebinned is None:
        rebinned = rebinned_chunk_rows(base, layout)
    offline_ok = layout.merge_orders is not None or layout.subchunks <= 1
    if rebinned is base or rebinned.equals(base):
        offline_ok = True
    if layout.subchunks > 1 and layout.merge_orders is None and rebinned is base:
        offline_ok = True
    elif layout.subchunks > 1 and layout.merge_orders is None and rebinned is not base:
        logger.warning(
            "Layout %s uses subchunks=%d; offline eval cannot split whole-order diagnostics — rerun pipeline",
            layout.name,
            layout.subchunks,
        )
    metrics = _layout_metrics(
        rebinned,
        layout.name,
        clip_sigma=clip_sigma,
        clip_max_delta_kms=clip_max_delta_kms,
        min_chunks=min_chunks,
    )
    metrics.update({f"cfg_{k}": v for k, v in layout_to_dict(layout).items() if k != "name"})
    metrics["eval_mode"] = "pipeline" if offline_ok and layout.subchunks > 1 else (
        "offline_merge" if layout.merge_orders else (
            "offline_baseline" if layout.subchunks <= 1 else "needs_pipeline_rerun"
        )
    )
    metrics["offline_eval_valid"] = bool(offline_ok or (rebinned is not base))
    return metrics


def evaluate_layouts_from_glob(
    layouts: list[ChunkLayout],
    diagnostics_glob: str,
    *,
    clip_sigma: float,
    clip_max_delta_kms: float,
    min_chunks: int,
) -> pd.DataFrame:
    """Evaluate layouts using diagnostics already produced for each layout (no rebin)."""
    rows = []
    for layout in layouts:
        tab = _load_chunk_rows(diagnostics_glob)
        if tab.empty:
            continue
        metrics = evaluate_layout(
            layout,
            tab,
            clip_sigma=clip_sigma,
            clip_max_delta_kms=clip_max_delta_kms,
            min_chunks=min_chunks,
            rebinned=tab,
        )
        metrics["offline_eval_valid"] = True
        metrics["eval_mode"] = "pipeline"
        rows.append(metrics)
    return pd.DataFrame(rows)


def evaluate_layouts(
    layouts: list[Path | ChunkLayout],
    diagnostics_glob: str,
    *,
    clip_sigma: float,
    clip_max_delta_kms: float,
    min_chunks: int,
) -> pd.DataFrame:
    base = _load_chunk_rows(diagnostics_glob)
    rows = []
    for item in layouts:
        layout = load_chunk_layout(item) if isinstance(item, Path) else item
        metrics = evaluate_layout(
            layout,
            base,
            clip_sigma=clip_sigma,
            clip_max_delta_kms=clip_max_delta_kms,
            min_chunks=min_chunks,
        )
        if isinstance(item, Path):
            metrics["layout_file"] = str(item)
        rows.append(metrics)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layouts", nargs="+", required=True, help="YAML layout files or globs")
    ap.add_argument("--diagnostics-glob", default="output/Gaia_DR3_*_diagnostics.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "validation_output" / "chunk_layout_eval")
    ap.add_argument("--chunk-outlier-sigma", type=float, default=DEFAULT_CHUNK_OUTLIER_SIGMA)
    ap.add_argument("--chunk-max-delta-kms", type=float, default=DEFAULT_CHUNK_MAX_DELTA_KMS)
    ap.add_argument("--min-chunks", type=int, default=3)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    layout_paths: list[Path] = []
    for pat in args.layouts:
        if "*" in pat:
            layout_paths.extend(Path(p) for p in glob_paths(pat))
        else:
            layout_paths.append(Path(pat))
    layout_paths = sorted({p.resolve() for p in layout_paths if p.is_file()})
    if not layout_paths:
        logger.error("No layout YAML files found")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_layouts(
        layout_paths,
        args.diagnostics_glob,
        clip_sigma=args.chunk_outlier_sigma,
        clip_max_delta_kms=args.chunk_max_delta_kms,
        min_chunks=args.min_chunks,
    )
    summary.to_csv(args.out_dir / "layout_comparison.csv", index=False)
    (args.out_dir / "layout_comparison.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote layout comparison (%d layouts) -> %s", len(summary), args.out_dir)


if __name__ == "__main__":
    main()
