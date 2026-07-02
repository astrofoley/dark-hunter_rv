#!/usr/bin/env python3
"""Generate RV data-only plots from summary files (no pipeline rerun)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import numpy as np
from astropy.time import Time

from darkhunter_rv.apf_observability import _target_coord_from_summary, normalize_observability_window
from darkhunter_rv.rv_keplerian_plots import plot_rv_data_only, points_for_rv_data_plot
from darkhunter_rv.summary_paths import discover_summary_files, discover_summary_path, parse_object_id_from_summary
from darkhunter_rv.website_plot_sync import maybe_stage_gaia_plots, resolve_web_root
from fit_apf_rv_keplerian import parse_summary, resolve_observability_window


def observability_for_plot(
    sid: Optional[str],
    summary_path: Path,
    obs_cache: Optional[Path],
    reports_dir: Optional[Path],
    *,
    lick_cache: Optional[Path] = None,
) -> Optional[Dict]:
    """Live window from summary coords; fit JSON fallback only if Lick cache unavailable."""
    obs = resolve_observability_window(
        summary_path, sid, obs_cache, lick_cache_path=lick_cache
    )
    if obs is not None:
        return obs
    if not sid or reports_dir is None:
        return None
    fit_json = reports_dir / f"{sid}_keplerian_fit.json"
    if not fit_json.is_file():
        return None
    try:
        raw = json.loads(fit_json.read_text(encoding="utf-8")).get("observability_window")
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return normalize_observability_window(raw)


def minimal_report(
    points,
    *,
    summary_path: Path,
    obs_cache: Optional[Path],
    reports_dir: Optional[Path],
    lick_cache: Optional[Path] = None,
) -> dict:
    t = np.array([p.mjd for p in points], dtype=float)
    t_ref = float(np.median(t)) if t.size else float(Time.now().mjd)
    sid = parse_object_id_from_summary(summary_path)
    obs = observability_for_plot(
        sid, summary_path, obs_cache, reports_dir, lick_cache=lick_cache
    )
    return {
        "t_ref_mjd": t_ref,
        "now_mjd": float(Time.now().mjd),
        "next_rv_max_mjd": t_ref,
        "next_rv_min_mjd": t_ref,
        "observability_window": obs,
    }


def build_plot(
    summary_path: Path,
    out_png: Path,
    *,
    obs_cache: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    lick_cache: Optional[Path] = None,
) -> bool:
    sid = parse_object_id_from_summary(summary_path)
    if _target_coord_from_summary(summary_path) is None:
        print(
            f"[WARN] {sid or summary_path.name}: no [GAIA METADATA] RA/Dec; APF window may be missing",
            flush=True,
        )
    all_points = parse_summary(summary_path)
    points, _literature_only = points_for_rv_data_plot(all_points)
    if not points:
        print(f"[SKIP] {sid or summary_path.name}: no RV epochs in summary", flush=True)
        return False
    report = minimal_report(
        points,
        summary_path=summary_path,
        obs_cache=obs_cache,
        reports_dir=reports_dir,
        lick_cache=lick_cache,
    )
    if report.get("observability_window") is None:
        print(f"[WARN] {sid or summary_path.name}: no APF observability window (check Lick twilight cache)", flush=True)
    plot_rv_data_only(summary_path, all_points, report, out_png)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Build RV data plots from Gaia summary files only.")
    ap.add_argument("--summary-dir", required=True, help="Pipeline output root (flat or nested summaries)")
    ap.add_argument("--plots-root", required=True, help="Output root for per-star plot subdirs")
    ap.add_argument("--reports-dir", default=None, help="rv_fit_reports dir (fit JSON fallback for APF window)")
    ap.add_argument("--star-id", default=None, help="Optional single Gaia source id")
    ap.add_argument(
        "--observability-cache",
        default=None,
        help="Optional observability_windows_cache.json (for Lick twilight sibling path)",
    )
    ap.add_argument(
        "--lick-cache",
        default=None,
        help="Lick twilight JSON (default: sibling of observability cache or repo default)",
    )
    ap.add_argument(
        "--web-root",
        default=None,
        help="Website root for auto-staging plots (default: WEB_ROOT env, e.g. /var/www/html/darkhunter/rv)",
    )
    ap.add_argument(
        "--no-sync-website",
        action="store_true",
        help="Do not copy plots into WEB_ROOT/stars/.../Gaia/Plots/",
    )
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    plots_root = Path(args.plots_root)
    obs_cache = Path(args.observability_cache) if args.observability_cache else None
    reports_dir = Path(args.reports_dir) if args.reports_dir else None
    lick_cache = Path(args.lick_cache) if args.lick_cache else None
    web_root = resolve_web_root(args.web_root, sync_enabled=not args.no_sync_website)

    if args.star_id:
        summ = discover_summary_path(summary_dir, str(args.star_id))
        files = [summ] if summ is not None else []
    else:
        files = discover_summary_files(summary_dir)
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
        if build_plot(
            summ,
            out_png,
            obs_cache=obs_cache,
            reports_dir=reports_dir,
            lick_cache=lick_cache,
        ):
            built += 1
            if sid and web_root is not None:
                maybe_stage_gaia_plots(
                    sid,
                    plots_root / f"Gaia_DR3_{sid}",
                    web_root=web_root,
                    reports_dir=reports_dir,
                )
        else:
            skipped += 1
    print(f"Built {built} summary-based RV data plots (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
