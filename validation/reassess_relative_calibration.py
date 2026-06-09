#!/usr/bin/env python3
"""
Re-assess APF–APF relative calibration after chunk bias debias and stat+intrinsic weighting.

Uses per-object chunk biases from ``validation_output/chunk_residuals/per_object_chunk_bias.csv``
(produced by ``validation.plot_chunk_residuals``).

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.reassess_relative_calibration \\
    --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \\
    --bias-csv validation_output/chunk_residuals/per_object_chunk_bias.csv \\
    --out-dir validation_output/chunk_calibration_assessment
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

from validation.chunk_calibration import (  # noqa: E402
    build_calibrated_exposure_table,
    relative_pair_table,
    summarize_relative_gate,
)
from validation.plot_chunk_residuals import (  # noqa: E402
    DEFAULT_CHUNK_MAX_DELTA_KMS,
    DEFAULT_CHUNK_OUTLIER_SIGMA,
    DEFAULT_MIN_CHUNK_MEASUREMENTS,
    _load_name_lookup,
    _object_name,
)
from validation.rv_overlap_lib import PhaseAGoals, per_star_gate_table  # noqa: E402

logger = logging.getLogger(__name__)
PRECISION_GOAL_KMS = 0.1


def _plot_relative_histogram(
    pairs_before: pd.DataFrame,
    pairs_after: pd.DataFrame,
    out_path: Path,
    *,
    goal_kms: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if len(pairs_before):
        ax.hist(
            pairs_before["abs_delta_rv_kms"].astype(float),
            bins=min(30, max(10, len(pairs_before))),
            alpha=0.55,
            label="pipeline",
            color="#4c72b0",
        )
    if len(pairs_after):
        ax.hist(
            pairs_after["abs_delta_rv_kms"].astype(float),
            bins=min(30, max(10, len(pairs_after))),
            alpha=0.55,
            label="chunk-calibrated",
            color="#c44e52",
        )
    ax.axvline(goal_kms, color="gray", ls="--", label=f"goal {goal_kms} km/s")
    ax.set_xlabel("|ΔRV| APF–APF (km/s)")
    ax.set_ylabel("Count")
    ax.set_title("Relative gate: pipeline vs chunk-calibrated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_chunk_scatter_compare(epochs: pd.DataFrame, out_path: Path) -> None:
    ok = np.isfinite(epochs["chunk_scatter_calibrated_kms"].astype(float))
    if not ok.any():
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        epochs.loc[ok, "rv_pipeline_kms"],
        epochs.loc[ok, "rv_calibrated_kms"],
        s=20,
        alpha=0.7,
    )
    lims = [
        np.nanmin([epochs["rv_pipeline_kms"].min(), epochs["rv_calibrated_kms"].min()]),
        np.nanmax([epochs["rv_pipeline_kms"].max(), epochs["rv_calibrated_kms"].max()]),
    ]
    pad = 5.0
    ax.plot([lims[0] - pad, lims[1] + pad], [lims[0] - pad, lims[1] + pad], "k--", lw=1)
    ax.set_xlabel("pipeline exposure RV (km/s)")
    ax.set_ylabel("chunk-calibrated RV (km/s)")
    ax.set_title("Exposure RV: pipeline vs calibrated")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    summary_before: dict,
    summary_after: dict,
    epochs: pd.DataFrame,
    goals: PhaseAGoals,
) -> None:
    sc = epochs["chunk_scatter_calibrated_kms"].astype(float)
    sc_ok = sc[np.isfinite(sc)]
    lines = [
        "# Chunk calibration relative-gate reassessment",
        "",
        "## Method",
        "- Per-spectrum outlier clip (7σ LOO, ±20 km/s) as in chunk residual study",
        "- Per chunk: subtract object bias (`weighted_mean_residual_kms`); fallback to sample mean",
        "- Weight ∝ 1 / (σ_stat² + σ_intrinsic²) from bias table",
        f"- Minimum {DEFAULT_MIN_CHUNK_MEASUREMENTS} chunks with valid bias per exposure",
        "",
        f"## Relative gate (APF–APF, ≤{goals.pair_window_days} d)",
        "",
        "### Pipeline (pre-chunk-calibration)",
        f"- Pairs: {summary_before.get('n_pairs', 0)}",
        f"- Median |ΔRV|: {summary_before.get('median_abs_delta_rv_kms', float('nan')):.3f} km/s",
        f"- p90 |ΔRV|: {summary_before.get('p90_abs_delta_rv_kms', float('nan')):.3f} km/s",
        f"- RMS ΔRV: {summary_before.get('rms_delta_rv_kms', float('nan')):.3f} km/s",
        f"- Fraction < {goals.relative_goal_kms} km/s: {100*summary_before.get('frac_below_goal', 0):.1f}%",
        "",
        "### Chunk-calibrated",
        f"- Pairs: {summary_after.get('n_pairs', 0)}",
        f"- Median |ΔRV|: {summary_after.get('median_abs_delta_rv_kms', float('nan')):.3f} km/s",
        f"- p90 |ΔRV|: {summary_after.get('p90_abs_delta_rv_kms', float('nan')):.3f} km/s",
        f"- RMS ΔRV: {summary_after.get('rms_delta_rv_kms', float('nan')):.3f} km/s",
        f"- Fraction < {goals.relative_goal_kms} km/s: {100*summary_after.get('frac_below_goal', 0):.1f}%",
        "",
        "## Per-exposure chunk scatter (calibrated stack)",
    ]
    if len(sc_ok):
        lines.extend(
            [
                f"- Median: {float(np.median(sc_ok)):.3f} km/s",
                f"- p90: {float(np.percentile(sc_ok, 90)):.3f} km/s",
                f"- Fraction < {PRECISION_GOAL_KMS} km/s: {100*float(np.mean(sc_ok < PRECISION_GOAL_KMS)):.1f}%",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_assessment(
    *,
    diagnostics_glob: str,
    bias_csv: Path,
    out_dir: Path,
    summary_dir: Path,
    pair_window_days: float,
    goal_kms: float,
    clip_sigma: float,
    clip_max_delta_kms: float,
    min_chunks: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = build_calibrated_exposure_table(
        diagnostics_glob,
        bias_csv,
        clip_sigma=clip_sigma,
        clip_max_delta_kms=clip_max_delta_kms,
        min_chunks=min_chunks,
    )
    if epochs.empty:
        return {"n_epochs": 0}

    name_lookup = _load_name_lookup(REPO_ROOT)
    epochs["name"] = [
        _object_name(str(g), summary_dir, name_lookup) for g in epochs["gaia_dr3_id"].astype(str)
    ]
    epochs.to_csv(out_dir / "calibrated_exposure_epochs.csv", index=False)

    pairs_before = relative_pair_table(
        epochs,
        rv_col="rv_pipeline_kms",
        err_col=None,
        pair_window_days=pair_window_days,
    )
    pairs_after = relative_pair_table(
        epochs,
        rv_col="rv_calibrated_kms",
        err_col="rv_err_calibrated_kms",
        pair_window_days=pair_window_days,
    )
    pairs_before.to_csv(out_dir / "relative_pairs_pipeline.csv", index=False)
    pairs_after.to_csv(out_dir / "relative_pairs_calibrated.csv", index=False)

    summary_before = summarize_relative_gate(pairs_before, goal_kms=goal_kms)
    summary_after = summarize_relative_gate(pairs_after, goal_kms=goal_kms)
    pd.DataFrame([{"stage": "pipeline", **summary_before}, {"stage": "chunk_calibrated", **summary_after}]).to_csv(
        out_dir / "relative_gate_summary.csv", index=False
    )

    per_star = per_star_gate_table(pairs_after, absolute_threshold_kms=1e9)
    if len(per_star):
        per_star.to_csv(out_dir / "per_star_relative_calibrated.csv", index=False)

    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    _plot_relative_histogram(
        pairs_before, pairs_after, out_dir / "plots" / "relative_delta_rv_histogram.png", goal_kms=goal_kms
    )
    _plot_chunk_scatter_compare(epochs, out_dir / "plots" / "exposure_rv_pipeline_vs_calibrated.png")

    goals = PhaseAGoals(pair_window_days=pair_window_days, relative_goal_kms=goal_kms)
    _write_report(
        out_dir / "REPORT.md",
        summary_before=summary_before,
        summary_after=summary_after,
        epochs=epochs,
        goals=goals,
    )
    manifest = {
        "n_epochs": int(len(epochs)),
        "n_epochs_calibrated": int(epochs["rv_calibrated_kms"].notna().sum()),
        "relative_gate_pipeline": summary_before,
        "relative_gate_calibrated": summary_after,
        "clip_sigma": clip_sigma,
        "clip_max_delta_kms": clip_max_delta_kms,
        "min_chunks": min_chunks,
    }
    (out_dir / "assessment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostics-glob", default="output/Gaia_DR3_*_diagnostics.csv")
    ap.add_argument(
        "--bias-csv",
        type=Path,
        default=REPO_ROOT / "validation_output" / "chunk_residuals" / "per_object_chunk_bias.csv",
    )
    ap.add_argument("--summary-dir", type=Path, default=REPO_ROOT / "output")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "validation_output" / "chunk_calibration_assessment",
    )
    ap.add_argument("--pair-window-days", type=float, default=7.0)
    ap.add_argument("--relative-goal-kms", type=float, default=PRECISION_GOAL_KMS)
    ap.add_argument("--chunk-outlier-sigma", type=float, default=DEFAULT_CHUNK_OUTLIER_SIGMA)
    ap.add_argument("--chunk-max-delta-kms", type=float, default=DEFAULT_CHUNK_MAX_DELTA_KMS)
    ap.add_argument("--min-chunks", type=int, default=DEFAULT_MIN_CHUNK_MEASUREMENTS)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.bias_csv.is_file():
        logger.error("Bias CSV not found: %s (run plot_chunk_residuals first)", args.bias_csv)
        sys.exit(1)

    manifest = run_assessment(
        diagnostics_glob=args.diagnostics_glob,
        bias_csv=args.bias_csv,
        out_dir=args.out_dir,
        summary_dir=args.summary_dir,
        pair_window_days=args.pair_window_days,
        goal_kms=args.relative_goal_kms,
        clip_sigma=args.chunk_outlier_sigma,
        clip_max_delta_kms=args.chunk_max_delta_kms,
        min_chunks=args.min_chunks,
    )
    logger.info("Assessment written to %s", args.out_dir)
    logger.info(
        "pipeline median|ΔRV|=%.3f km/s → calibrated %.3f km/s",
        manifest.get("relative_gate_pipeline", {}).get("median_abs_delta_rv_kms", float("nan")),
        manifest.get("relative_gate_calibrated", {}).get("median_abs_delta_rv_kms", float("nan")),
    )


if __name__ == "__main__":
    main()
