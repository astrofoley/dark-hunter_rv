#!/usr/bin/env python3
"""Repair tables/data.csv layout and refresh column values from existing summaries/fits."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from fit_apf_rv_keplerian import (
    _load_json_cache,
    discover_summary_path,
    enrich_gaia_cache_from_summaries,
    inclination_deg_for_website_table,
    lookup_fit_report_by_gaia_id,
    website_table_m2_at_i_pe_fixed_from_report,
    website_table_masses_from_report,
)
from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary
from darkhunter_rv.website_table_csv import (
    INCLINATION_COLUMN,
    M2SINI_ERR_COLUMN,
    M2_AT_I_COLUMN,
    M2_AT_I_PE_COLUMN,
    M2SINI_COLUMN,
    G_MAG_COLUMN,
    N_OBS_COLUMN,
    days_since_last_apf_from_summary,
    format_next_rv_event_cell,
    format_optional_error_cell,
    format_optional_g_mag_cell,
    format_optional_inclination_cell,
    format_optional_mass_cell,
    gaia_g_mag_from_metadata,
    gaia_id_from_row,
    n_apf_obs_from_summary,
    next_rv_event_from_fit_report,
    normalize_data_csv,
    parse_next_rv_cell_to_mjd,
    parse_table_m1_msun,
)


def load_gaia_nss_cache(reports_dir: Path) -> dict:
    for path in (
        reports_dir / "gaia_nss_cache.json",
        reports_dir.parent / "rv_fit_reports" / "gaia_nss_cache.json",
    ):
        if path.is_file():
            return _load_json_cache(path)
    return {}


def update_table_columns(
    data_csv: Path,
    *,
    out_dir: Path,
    reports_dir: Path,
    gaia_id: Optional[str] = None,
    gaia_cache: Optional[dict] = None,
) -> Dict[str, int]:
    rows: List[List[str]] = []
    with data_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit("tables/data.csv is empty")

    hdr = rows[0]
    data_rows = rows[1:]
    _, n_stray = normalize_data_csv(hdr, data_rows)

    gaia_i = hdr.index("GAIA NAME")
    m2_i = hdr.index("M2 (Msun)")
    m2sini_i = hdr.index(M2SINI_COLUMN)
    m2sini_err_i = hdr.index(M2SINI_ERR_COLUMN)
    m2over_i = hdr.index(M2_AT_I_COLUMN)
    m2_pe_i = hdr.index(M2_AT_I_PE_COLUMN)
    incl_i = hdr.index(INCLINATION_COLUMN)
    g_mag_i = hdr.index(G_MAG_COLUMN) if G_MAG_COLUMN in hdr else -1
    n_obs_i = hdr.index(N_OBS_COLUMN)
    apf_days_i = hdr.index("DAYS SINCE LAST APF")
    next_rv_i = hdr.index("NEXT RV EVENT (DATE)")

    if gaia_cache is None:
        gaia_cache = load_gaia_nss_cache(reports_dir)

    gaia_ids = [
        gaia_id_from_row((r[hdr.index("GAIA NAME")] if hdr.index("GAIA NAME") < len(r) else ""))
        for r in data_rows
        if r
    ]
    gaia_ids = [g for g in gaia_ids if g]
    n_cache_summ = enrich_gaia_cache_from_summaries(gaia_cache, out_dir, gaia_ids=gaia_ids)

    reports: Dict[str, dict] = {}
    if reports_dir.is_dir():
        for p in sorted(reports_dir.glob("*_keplerian_fit.json")):
            sid = p.stem.replace("_keplerian_fit", "")
            try:
                reports[sid] = json.loads(p.read_text())
            except Exception:
                continue

    n_apf_days = 0
    n_n_obs = 0
    n_m2 = 0
    n_m2sini = 0
    n_m2sini_err = 0
    n_m2_at_i = 0
    n_m2_at_i_eq_sin = 0
    n_m2_pe = 0
    n_incl = 0
    n_g_mag = 0
    n_next = 0
    target = (gaia_id or "").strip()
    for r in data_rows:
        if not r:
            continue
        gaia = (r[gaia_i] if gaia_i < len(r) else "").strip()
        sid = gaia_id_from_row(gaia)
        if not sid:
            continue
        if target and sid != target:
            continue

        table_m1 = parse_table_m1_msun(r, hdr)
        summ = discover_summary_path(out_dir, sid)
        if summ is not None and summ.is_file():
            n_obs = n_apf_obs_from_summary(summ)
            while len(r) <= n_obs_i:
                r.append("")
            r[n_obs_i] = str(n_obs)
            n_n_obs += 1
            age = days_since_last_apf_from_summary(summ)
            if age is not None:
                while len(r) <= apf_days_i:
                    r.append("")
                r[apf_days_i] = f"{age:.2f}"
                n_apf_days += 1
            if g_mag_i >= 0:
                meta = parse_gaia_metadata_from_star_summary(summ) or {}
                g_mag = gaia_g_mag_from_metadata(meta)
                if g_mag is not None:
                    while len(r) <= g_mag_i:
                        r.append("")
                    r[g_mag_i] = format_optional_g_mag_cell(g_mag)
                    n_g_mag += 1

        rep = lookup_fit_report_by_gaia_id(reports, sid)
        if rep is None:
            continue
        incl_deg = inclination_deg_for_website_table(
            rep,
            summary_path=summ if summ is not None else None,
            gaia_cache=gaia_cache,
        )
        while len(r) <= incl_i:
            r.append("")
        r[incl_i] = format_optional_inclination_cell(incl_deg)
        if incl_deg is not None:
            n_incl += 1

        masses = website_table_masses_from_report(
            rep,
            summary_path=summ if summ is not None else None,
            gaia_cache=gaia_cache,
            table_m1_msun=table_m1,
        )
        if masses["m2_msun"] is not None:
            while len(r) <= m2_i:
                r.append("")
            r[m2_i] = f"{masses['m2_msun']:.5f}"
            n_m2 += 1
        while len(r) <= m2sini_i:
            r.append("")
        r[m2sini_i] = format_optional_mass_cell(masses["m2sin_i_msun"])
        if masses["m2sin_i_msun"] is not None:
            n_m2sini += 1
        while len(r) <= m2sini_err_i:
            r.append("")
        r[m2sini_err_i] = format_optional_error_cell(masses.get("m2sin_i_msun_stderr"))
        if masses.get("m2sin_i_msun_stderr") is not None:
            n_m2sini_err += 1
        while len(r) <= m2over_i:
            r.append("")
        r[m2over_i] = format_optional_mass_cell(masses["m2_at_i_msun"])
        if masses["m2_at_i_msun"] is not None:
            n_m2_at_i += 1
            if masses["m2sin_i_msun"] is not None:
                if abs(masses["m2_at_i_msun"] - masses["m2sin_i_msun"]) <= max(
                    1e-6, 1e-4 * masses["m2sin_i_msun"]
                ):
                    n_m2_at_i_eq_sin += 1
        m2_pe = website_table_m2_at_i_pe_fixed_from_report(
            rep,
            summary_path=summ if summ is not None else None,
            gaia_cache=gaia_cache,
            table_m1_msun=table_m1,
        )
        while len(r) <= m2_pe_i:
            r.append("")
        r[m2_pe_i] = format_optional_mass_cell(m2_pe)
        if m2_pe is not None:
            n_m2_pe += 1
        nxt = next_rv_event_from_fit_report(rep)
        if nxt is not None:
            while len(r) <= next_rv_i:
                r.append("")
            r[next_rv_i] = format_next_rv_event_cell(nxt)
            n_next += 1

    for r in data_rows:
        if not r or next_rv_i >= len(r) or not r[next_rv_i]:
            continue
        mjd = parse_next_rv_cell_to_mjd(r[next_rv_i])
        if mjd is not None and "/" not in r[next_rv_i]:
            r[next_rv_i] = format_next_rv_event_cell(mjd)

    with data_csv.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    return {
        "data_rows": len(data_rows),
        "columns": len(hdr),
        "stray_img_cleared": n_stray,
        "n_obs_filled": n_n_obs,
        "apf_days_filled": n_apf_days,
        "m2_filled": n_m2,
        "m2sin_i_filled": n_m2sini,
        "m2sin_i_err_filled": n_m2sini_err,
        "m2_at_i_filled": n_m2_at_i,
        "m2_at_i_equals_m2sin_i": n_m2_at_i_eq_sin,
        "m2_at_i_pe_filled": n_m2_pe,
        "inclination_filled": n_incl,
        "g_mag_filled": n_g_mag,
        "next_rv_filled": n_next,
        "reports_loaded": len(reports),
        "cache_enriched_from_summaries": n_cache_summ,
    }


def main() -> int:
    try:
        from erfa import ErfaWarning  # type: ignore

        warnings.filterwarnings("ignore", category=ErfaWarning)
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="Normalize data.csv and fill schedule/mass columns from existing assets (no refit)."
    )
    ap.add_argument(
        "--data-csv",
        default="/var/www/html/darkhunter/rv/tables/data.csv",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Pipeline summaries (default: REPO/output)",
    )
    ap.add_argument(
        "--reports-dir",
        default=None,
        help="Keplerian JSON reports (default: REPO/rv_fit_reports)",
    )
    ap.add_argument(
        "--gaia-id",
        default=None,
        help="Update only this Gaia DR3 source id row (default: all rows).",
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else repo / "output"
    reports_dir = Path(args.reports_dir) if args.reports_dir else repo / "rv_fit_reports"
    gaia_cache = load_gaia_nss_cache(reports_dir)
    stats = update_table_columns(
        Path(args.data_csv),
        out_dir=out_dir,
        reports_dir=reports_dir,
        gaia_id=args.gaia_id,
        gaia_cache=gaia_cache,
    )
    print(
        f"updated {args.data_csv}: {stats['data_rows']} rows, {stats['columns']} columns, "
        f"cleared {stats['stray_img_cleared']} stray <img>, "
        f"n_obs={stats['n_obs_filled']}, apf_days={stats['apf_days_filled']}, m2={stats['m2_filled']}, "
        f"incl={stats['inclination_filled']}, g_mag={stats['g_mag_filled']}, "
        f"m2sin_i={stats['m2sin_i_filled']}, m2sin_i_err={stats['m2sin_i_err_filled']}, "
        f"m2_at_i={stats['m2_at_i_filled']} "
        f"(m2_at_i=m2sin_i for {stats['m2_at_i_equals_m2sin_i']}, often i≈90°), "
        f"m2_at_i_pe={stats['m2_at_i_pe_filled']}, "
        f"next_rv={stats['next_rv_filled']} (from {stats['reports_loaded']} reports, "
        f"cache+summary incl patches={stats['cache_enriched_from_summaries']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
