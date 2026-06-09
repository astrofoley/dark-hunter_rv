#!/usr/bin/env python3
"""
Benchmark cool-star, high-S/N mask CCF precision (RV pipeline step 01).

Reads ``*_diagnostics.csv``, filters exposures in the mask applicability region with
``log10(median_mask_ccf_peak_snr) >= --min-log10-snr`` (default 1.0 ≈ S/N 10), and reports
per-exposure chunk scatter and summary statistics vs the 0.1 km/s precision goal.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.benchmark_cool_precision \\
    --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \\
    --out-dir validation_output/benchmark_cool_precision
"""
from __future__ import annotations

import argparse
import logging
import sys
from glob import glob as glob_paths
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv.method_evaluation import exposure_method_flags  # noqa: E402
from darkhunter_rv.method_regions import region_mask_applicable  # noqa: E402

logger = logging.getLogger(__name__)

PRECISION_GOAL_KMS = 0.1


def _load_exposure_rows(paths: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        stem = Path(p).name.replace("_diagnostics.csv", "")
        file_col = df["file"].iloc[0] if "file" in df.columns and len(df) else stem
        chunk_rows = [r for _, r in df.iterrows() if str(r.get("chunk_key", "")) != "all"]
        if not chunk_rows:
            continue
        flags = exposure_method_flags(chunk_rows)
        teff = float(chunk_rows[0].get("teff", np.nan))
        snr_med = float(flags.get("median_mask_ccf_peak_snr", np.nan))
        log_snr = float(np.log10(snr_med)) if np.isfinite(snr_med) and snr_med > 0 else np.nan
        if not bool(region_mask_applicable(np.array([teff]), np.array([log_snr]))[0]):
            continue
        scatter = float(chunk_rows[0].get("chunk_scatter_kms", np.nan))
        if not np.isfinite(scatter):
            mask_rows = [r for r in chunk_rows if r.get("method") == "mask_ccf" and r.get("used_in_exposure_stack")]
            if len(mask_rows) >= 2:
                rvs = np.array([float(r["rv_kms"]) for r in mask_rows if np.isfinite(r.get("rv_kms", np.nan))])
                scatter = float(np.std(rvs, ddof=1)) if len(rvs) >= 2 else float("nan")
        rows.append(
            {
                "file": file_col,
                "teff": teff,
                "median_mask_ccf_peak_snr": snr_med,
                "log10_median_mask_ccf_peak_snr": log_snr,
                "chunk_scatter_kms": scatter,
                "mask_rv_kms": flags.get("mask_rv_kms"),
                "mask_err_kms": flags.get("mask_err_kms"),
                "mask_valid": flags.get("mask_valid"),
            }
        )
    return pd.DataFrame(rows)


def _summarize(tab: pd.DataFrame, min_log10_snr: float) -> dict[str, float]:
    good = tab[np.isfinite(tab["chunk_scatter_kms"].astype(float))]
    good = good[good["log10_median_mask_ccf_peak_snr"].astype(float) >= min_log10_snr]
    sc = good["chunk_scatter_kms"].astype(float).values
    if len(sc) == 0:
        return {"n": 0.0}
    return {
        "n": float(len(sc)),
        "median_chunk_scatter_kms": float(np.median(sc)),
        "p90_chunk_scatter_kms": float(np.percentile(sc, 90)),
        "frac_below_0p1_kms": float(np.mean(sc < PRECISION_GOAL_KMS)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostics-glob", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("validation_output/benchmark_cool_precision"))
    ap.add_argument("--min-log10-snr", type=float, default=1.0, help="High-S/N cut (default log10=1 → S/N≥10)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    paths = sorted(glob_paths(args.diagnostics_glob))
    if not paths:
        logger.error("No diagnostics matched %s", args.diagnostics_glob)
        sys.exit(1)

    tab = _load_exposure_rows(paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(args.out_dir / "per_exposure_cool_high_snr.csv", index=False)

    summary = _summarize(tab, args.min_log10_snr)
    summary["precision_goal_kms"] = PRECISION_GOAL_KMS
    summary["min_log10_snr"] = args.min_log10_snr
    summary["meets_goal_median"] = bool(
        summary.get("median_chunk_scatter_kms", np.inf) < PRECISION_GOAL_KMS
    )
    pd.DataFrame([summary]).to_csv(args.out_dir / "summary.csv", index=False)

    good = tab[
        np.isfinite(tab["chunk_scatter_kms"].astype(float))
        & (tab["log10_median_mask_ccf_peak_snr"].astype(float) >= args.min_log10_snr)
    ]
    if len(good) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            good["log10_median_mask_ccf_peak_snr"].astype(float),
            good["chunk_scatter_kms"].astype(float),
            s=12,
            alpha=0.7,
        )
        ax.axhline(PRECISION_GOAL_KMS, color="crimson", ls="--", label=f"goal {PRECISION_GOAL_KMS} km/s")
        ax.set_xlabel("log10(median mask CCF peak S/N)")
        ax.set_ylabel("chunk scatter (km/s)")
        ax.set_title("Cool high-S/N mask precision benchmark (step 01)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / "chunk_scatter_vs_log10_snr.png", dpi=120)
        plt.close(fig)

    logger.info("Wrote %s (%d exposures, n_high_snr=%s)", args.out_dir, len(tab), summary.get("n"))
    if summary.get("n", 0) > 0:
        logger.info(
            "median scatter=%.3f km/s p90=%.3f frac<0.1=%.1f%% goal_met_median=%s",
            summary["median_chunk_scatter_kms"],
            summary["p90_chunk_scatter_kms"],
            100.0 * summary["frac_below_0p1_kms"],
            summary["meets_goal_median"],
        )


if __name__ == "__main__":
    main()
