#!/usr/bin/env python3
"""Generate RV data-only plots from summary files (no pipeline rerun)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from astropy.time import Time

from darkhunter_rv.rv_keplerian_plots import our_telescope_points, plot_rv_data_only
from darkhunter_rv.summary_paths import discover_summary_files, parse_object_id_from_summary
from fit_apf_rv_keplerian import parse_summary, resolve_observability_window


def minimal_report(points, *, summary_path: Path, obs_cache: Path | None) -> dict:
    t = np.array([p.mjd for p in points], dtype=float)
    t_ref = float(np.median(t)) if t.size else float(Time.now().mjd)
    sid = parse_object_id_from_summary(summary_path)
    obs = resolve_observability_window(summary_path, sid, obs_cache)
    return {
        "t_ref_mjd": t_ref,
        "now_mjd": float(Time.now().mjd),
        "next_rv_max_mjd": t_ref,
        "next_rv_min_mjd": t_ref,
        "observability_window": obs,
    }


def build_plot(summary_path: Path, out_png: Path, *, obs_cache: Path | None = None) -> bool:
    points = our_telescope_points(parse_summary(summary_path))
    if len(points) < 1:
        return False
    plot_rv_data_only(
        summary_path,
        points,
        minimal_report(points, summary_path=summary_path, obs_cache=obs_cache),
        out_png,
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Build RV data plots from Gaia summary files only.")
    ap.add_argument("--summary-dir", required=True, help="Pipeline output root (flat or nested summaries)")
    ap.add_argument("--plots-root", required=True, help="Output root for per-star plot subdirs")
    ap.add_argument("--star-id", default=None, help="Optional single Gaia source id")
    ap.add_argument(
        "--observability-cache",
        default=None,
        help="Optional observability_windows_cache.json (for Lick twilight sibling path)",
    )
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    plots_root = Path(args.plots_root)
    obs_cache = Path(args.observability_cache) if args.observability_cache else None

    files = discover_summary_files(summary_dir)
    if args.star_id:
        files = [p for p in files if parse_object_id_from_summary(p) == str(args.star_id)]
    if not files:
        print("No summary files found.")
        return 2

    built = 0
    skipped = 0
    for summ in files:
        sid = parse_object_id_from_summary(summ)
        if not sid:
            skipped += 1
            continue
        out_png = plots_root / f"Gaia_DR3_{sid}" / f"Gaia_DR3_{sid}_rv_plot.png"
        if build_plot(summ, out_png, obs_cache=obs_cache):
            built += 1
        else:
            skipped += 1
    print(f"Built {built} summary-based RV data plots (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
