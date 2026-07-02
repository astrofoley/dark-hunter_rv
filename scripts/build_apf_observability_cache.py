#!/usr/bin/env python3
"""Build rv_fit_reports/observability_windows_cache.json for APF visibility shading.

Entries are a snapshot at build time (logging / offline review). Plots and fits
call resolve_observability_window(), which recomputes windows relative to today.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import List, Optional

from astropy.time import Time

from darkhunter_rv.apf_observability import (
    SCAN_HORIZON_DAYS,
    observability_for_summary,
    normalize_observability_window,
    reference_now,
)
from darkhunter_rv.lick_twilight_cache import (
    cache_mjd_bounds,
    default_cache_path as default_lick_cache_path,
    load_cache,
    years_covering_mjd_range,
)
from darkhunter_rv.website_table_csv import gaia_id_from_row
from darkhunter_rv.summary_paths import discover_summary_path


def _ensure_lick_cache(
    cache_path: Path,
    years: Optional[str],
    *,
    scan_horizon_days: int = SCAN_HORIZON_DAYS,
) -> None:
    from darkhunter_rv.lick_twilight_cache import build_cache_years

    now_mjd, _, today_start_mjd = reference_now()
    need_end_mjd = now_mjd + float(scan_horizon_days)
    if years:
        year_list = [int(y.strip()) for y in years.split(",") if y.strip()]
    else:
        year_list = years_covering_mjd_range(today_start_mjd - 30.0, need_end_mjd + 30.0)

    cached = load_cache(cache_path) if cache_path.is_file() else {}
    cached_years = {int(y) for y in (cached.get("years") or []) if str(y).isdigit()}
    _, cache_end_mjd = cache_mjd_bounds(cache_path)
    if (
        cached.get("nights")
        and cache_end_mjd is not None
        and cache_end_mjd >= need_end_mjd + 1.0
        and cached_years.issuperset(year_list)
    ):
        return

    all_years = sorted(set(year_list) | cached_years)
    print(
        f"Fetching Lick twilight tables for {all_years} "
        f"(need coverage through MJD {need_end_mjd:.0f}) …",
        flush=True,
    )
    build_cache_years(all_years, cache_path=cache_path)


def main() -> int:
    try:
        from erfa import ErfaWarning  # type: ignore

        warnings.filterwarnings("ignore", category=ErfaWarning)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Build APF observability windows cache from summaries.")
    ap.add_argument(
        "--data-csv",
        default="/var/www/html/darkhunter/rv/tables/data.csv",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Pipeline summaries directory (default: REPO/output)",
    )
    ap.add_argument(
        "--cache",
        default=None,
        help="Output JSON path (default: REPO/rv_fit_reports/observability_windows_cache.json)",
    )
    ap.add_argument(
        "--lick-cache",
        default=None,
        help="Lick twilight JSON (default: REPO/rv_fit_reports/lick_twilight_cache.json)",
    )
    ap.add_argument(
        "--lick-years",
        default=None,
        help="Years for Lick twilight download if cache missing (default: y-1,y,y+1)",
    )
    ap.add_argument("--gaia-id", default=None, help="Single Gaia DR3 source id (default: all table rows).")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else repo / "output"
    cache_path = Path(args.cache) if args.cache else repo / "rv_fit_reports" / "observability_windows_cache.json"
    lick_cache = Path(args.lick_cache) if args.lick_cache else default_lick_cache_path(repo)
    data_csv = Path(args.data_csv)

    _ensure_lick_cache(lick_cache, args.lick_years)

    gaia_ids: List[str] = []
    if args.gaia_id:
        gaia_ids = [str(args.gaia_id).strip()]
    elif data_csv.is_file():
        with data_csv.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if rows:
            hdr = rows[0]
            if "GAIA NAME" in hdr:
                gaia_i = hdr.index("GAIA NAME")
                for r in rows[1:]:
                    sid = gaia_id_from_row(r[gaia_i] if gaia_i < len(r) else "")
                    if sid:
                        gaia_ids.append(sid)

    cache: dict = {}
    if cache_path.is_file():
        try:
            loaded = json.loads(cache_path.read_text())
            if isinstance(loaded, dict):
                cache = loaded
        except Exception:
            cache = {}

    built = 0
    skipped = 0
    n_ids = len(gaia_ids)
    for idx, sid in enumerate(gaia_ids, start=1):
        summ = discover_summary_path(out_dir, sid)
        if summ is None:
            skipped += 1
            print(f"[{idx}/{n_ids}] {sid}: no summary", flush=True)
            continue
        row = observability_for_summary(summ, lick_cache_path=lick_cache)
        if row is None:
            skipped += 1
            print(f"[{idx}/{n_ids}] {sid}: not observable in scan horizon", flush=True)
            continue
        entry = {k: v for k, v in row.items() if k != "gaia_source_id"}
        entry = normalize_observability_window(entry) or entry
        cache[sid] = entry
        built += 1
        win = entry.get("next_window_start_date", "")
        end = entry.get("next_window_end_date", "")
        circ = entry.get("circumpolar", False)
        if circ:
            print(f"[{idx}/{n_ids}] {sid}: circumpolar (year-round at APF limits)", flush=True)
        elif not win and not end:
            print(f"[{idx}/{n_ids}] {sid}: no observable window in scan horizon", flush=True)
        elif win and end and win == end:
            print(f"[{idx}/{n_ids}] {sid}: WARNING same-day window {win}", flush=True)
            print(f"[{idx}/{n_ids}] {sid}: {win} to {end}", flush=True)
        else:
            if win and end and not circ:
                try:
                    span = float(Time(end, format="iso", scale="utc").mjd) - float(
                        Time(win, format="iso", scale="utc").mjd
                    )
                    if span > 250.0:
                        print(f"[{idx}/{n_ids}] {sid}: WARNING long window {win} to {end}", flush=True)
                except Exception:
                    pass
            print(f"[{idx}/{n_ids}] {sid}: {win} to {end}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    print(f"Wrote {built} observability entries to {cache_path} (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
