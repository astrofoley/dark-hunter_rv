#!/usr/bin/env python3
"""Replot RV figures from stored Keplerian fits (no refit) with live APF windows."""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from astropy.time import Time

from darkhunter_rv.rv_keplerian_plots import (
    our_telescope_points,
    plot_fit_residuals,
    plot_multi_fit,
    plot_rv_data_only,
)
from darkhunter_rv.summary_paths import discover_summary_files, parse_object_id_from_summary
from fit_apf_rv_keplerian import parse_summary, resolve_observability_window


def _finite_points(summary_path: Path):
    points = parse_summary(summary_path)
    return [p for p in points if np.isfinite(p.mjd) and np.isfinite(p.rv)]


def replot_from_fit_json(
    fit_json: Path,
    *,
    out_dir: Path,
    reports_dir: Path,
    obs_cache: Path | None,
    lick_cache: Path | None = None,
) -> bool:
    gid = fit_json.stem.replace("_keplerian_fit", "")
    summ = out_dir / f"Gaia_DR3_{gid}_summary.txt"
    if not summ.is_file():
        print(f"[skip] {gid}: no summary")
        return False

    report = json.loads(fit_json.read_text())
    pbv = report.get("params_by_variant") or {}
    fv = report.get("fit_variants") or {}
    if not pbv or not fv or "free" not in pbv:
        print(f"[skip] {gid}: no stored fit_variants")
        return False

    report["observability_window"] = resolve_observability_window(
        summ, gid, obs_cache, lick_cache_path=lick_cache
    )
    report["now_mjd"] = float(Time.now().mjd)

    points_fit = _finite_points(summ)
    fit_variants = {k: (np.asarray(v, dtype=float), fv[k]) for k, v in pbv.items() if k in fv}
    m1 = report.get("used_m1_msun")
    star_dir = out_dir / f"Gaia_DR3_{gid}"
    star_dir.mkdir(parents=True, exist_ok=True)

    plot_multi_fit(summ, points_fit, fit_variants, report, reports_dir / f"{gid}_keplerian_fit.png", m1_msun=m1)
    plot_fit_residuals(
        summ, points_fit, fit_variants, report, reports_dir / f"{gid}_keplerian_residuals.png", m1_msun=m1
    )
    ours = our_telescope_points(points_fit)
    plot_rv_data_only(summ, ours, report, star_dir / f"Gaia_DR3_{gid}_rv_plot.png")

    for name in (f"{gid}_keplerian_fit.png", f"{gid}_keplerian_residuals.png"):
        src = reports_dir / name
        if src.is_file():
            shutil.copy2(src, star_dir / f"Gaia_DR3_{name}")

    fit_json.write_text(json.dumps(report, indent=2))
    return True


def main() -> int:
    try:
        from erfa import ErfaWarning  # type: ignore

        warnings.filterwarnings("ignore", category=ErfaWarning)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Replot RV figures from fit JSON without refitting.")
    ap.add_argument("--output-dir", default=None, help="Pipeline output (default: REPO/output)")
    ap.add_argument("--reports-dir", default=None, help="Fit reports (default: REPO/rv_fit_reports)")
    ap.add_argument("--observability-cache", default=None, help="observability_windows_cache.json path")
    ap.add_argument("--lick-cache", default=None, help="Lick twilight JSON path")
    ap.add_argument("--star-id", default=None, help="Optional single Gaia DR3 id")
    ap.add_argument(
        "--also-summaries-without-fits",
        action="store_true",
        help="Build RV data plots for summaries that lack fit JSON",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else repo / "output"
    reports_dir = Path(args.reports_dir) if args.reports_dir else repo / "rv_fit_reports"
    obs_cache = Path(args.observability_cache) if args.observability_cache else reports_dir / "observability_windows_cache.json"
    lick_cache = Path(args.lick_cache) if args.lick_cache else obs_cache.parent / "lick_twilight_cache.json"

    fit_jsons = sorted(reports_dir.glob("*_keplerian_fit.json"))
    if args.star_id:
        fit_jsons = [p for p in fit_jsons if p.stem.replace("_keplerian_fit", "") == str(args.star_id)]

    ok = skip = 0
    for fit_json in fit_jsons:
        if replot_from_fit_json(
            fit_json, out_dir=out_dir, reports_dir=reports_dir, obs_cache=obs_cache, lick_cache=lick_cache
        ):
            ok += 1
            if ok % 25 == 0:
                print(f"... replotted {ok}", flush=True)
        else:
            skip += 1

    if args.also_summaries_without_fits:
        import importlib.util

        mod_path = Path(__file__).resolve().parent / "plot_rv_from_summaries.py"
        spec = importlib.util.spec_from_file_location("plot_rv_from_summaries", mod_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        build_plot = mod.build_plot

        fit_ids = {p.stem.replace("_keplerian_fit", "") for p in reports_dir.glob("*_keplerian_fit.json")}
        extra = 0
        for summ in discover_summary_files(out_dir):
            sid = parse_object_id_from_summary(summ)
            if not sid or sid in fit_ids:
                continue
            if args.star_id and sid != str(args.star_id):
                continue
            out_png = out_dir / f"Gaia_DR3_{sid}" / f"Gaia_DR3_{sid}_rv_plot.png"
            if build_plot(
                summ, out_png, obs_cache=obs_cache, reports_dir=reports_dir, lick_cache=lick_cache
            ):
                extra += 1
        print(f"RV data plots for summaries without fits: {extra}")

    print(f"Done: replotted={ok} skipped={skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
