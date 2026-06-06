#!/usr/bin/env python3
"""Generate RV data-only plots from summary files (no pipeline rerun)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from astropy.time import Time

from darkhunter_rv.rv_keplerian_plots import our_telescope_points, plot_rv_data_only
from darkhunter_rv.rv_point_filters import rv_value_is_valid
from fit_apf_rv_keplerian import RVPoint, _pipeline_telescope_from_filename, parse_object_id_from_summary


def parse_gaia_id(path: Path) -> str | None:
    return parse_object_id_from_summary(path)


def parse_points(summary_path: Path) -> list[RVPoint]:
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("[PIPELINE RESULTS]", 1)[-1].splitlines() if "[PIPELINE RESULTS]" in text else text.splitlines()
    points: list[RVPoint] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#") or (s.startswith("[") and s.endswith("]")):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        if len(parts) >= 6 and parts[-1] in ("True", "False"):
            parts = parts[:-1]
        if len(parts) < 5:
            continue
        try:
            rv = float(parts[2])
            if not rv_value_is_valid(rv):
                continue
            points.append(
                RVPoint(
                    file=parts[0],
                    mjd=float(parts[1]),
                    rv=rv,
                    rv_err=max(float(parts[3]), 1e-4),
                    rms=max(abs(float(parts[4])), 1e-4),
                    telescope=_pipeline_telescope_from_filename(parts[0]),
                    is_literature=False,
                )
            )
        except ValueError:
            continue
    points.sort(key=lambda p: p.mjd)
    return points


def minimal_report(points: list[RVPoint], *, summary_path: Path | None = None) -> dict:
    t = np.array([p.mjd for p in points], dtype=float)
    t_ref = float(np.median(t)) if t.size else float(Time.now().mjd)
    obs = None
    if summary_path is not None:
        from darkhunter_rv.apf_observability import observability_for_summary

        obs_row = observability_for_summary(summary_path)
        if obs_row is not None:
            obs = {k: v for k, v in obs_row.items() if k != "gaia_source_id"}
    return {
        "t_ref_mjd": t_ref,
        "now_mjd": float(Time.now().mjd),
        "next_rv_max_mjd": t_ref,
        "next_rv_min_mjd": t_ref,
        "observability_window": obs,
    }


def build_plot(summary_path: Path, out_png: Path) -> bool:
    points = our_telescope_points(parse_points(summary_path))
    if len(points) < 2:
        return False
    plot_rv_data_only(summary_path, points, minimal_report(points, summary_path=summary_path), out_png)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Build RV data plots from Gaia summary files only.")
    ap.add_argument("--summary-dir", required=True, help="Directory containing Gaia_DR3_*_summary.txt")
    ap.add_argument("--plots-root", required=True, help="Output root for per-star plot subdirs")
    ap.add_argument("--star-id", default=None, help="Optional single Gaia source id")
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    plots_root = Path(args.plots_root)

    pattern = f"Gaia_DR3_{args.star_id}_summary.txt" if args.star_id else "Gaia_DR3_*_summary.txt"
    files = sorted(summary_dir.glob(pattern))
    if not files:
        print("No summary files found.")
        return 2

    built = 0
    skipped = 0
    for summ in files:
        sid = parse_gaia_id(summ)
        if not sid:
            skipped += 1
            continue
        out_png = plots_root / f"Gaia_DR3_{sid}" / f"Gaia_DR3_{sid}_rv_plot.png"
        if build_plot(summ, out_png):
            built += 1
        else:
            skipped += 1
    print(f"Built {built} summary-based RV data plots (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
