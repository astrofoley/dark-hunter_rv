#!/usr/bin/env python3
"""Repair tables/data.csv layout and refresh column values from existing summaries/fits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fit_apf_rv_keplerian import website_table_masses_from_report
from darkhunter_rv.website_table_csv import (
    days_since_last_apf_from_summary,
    gaia_id_from_row,
    next_rv_event_from_fit_report,
    normalize_data_csv,
)


def update_table_columns(
    data_csv: Path,
    *,
    out_dir: Path,
    reports_dir: Path,
) -> dict[str, int]:
    rows: list[list[str]] = []
    with data_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit("tables/data.csv is empty")

    hdr = rows[0]
    data_rows = rows[1:]
    _, n_stray = normalize_data_csv(hdr, data_rows)

    gaia_i = hdr.index("GAIA NAME")
    m2_i = hdr.index("M2 (Msun)")
    m2sini_i = hdr.index("M2sin i (Msun)")
    m2over_i = hdr.index("(M2sin i)/(sin i) (Msun)")
    apf_days_i = hdr.index("DAYS SINCE LAST APF")
    next_rv_i = hdr.index("NEXT RV EVENT (MJD)")

    reports: dict[str, dict] = {}
    if reports_dir.is_dir():
        for p in sorted(reports_dir.glob("*_keplerian_fit.json")):
            sid = p.stem.replace("_keplerian_fit", "")
            try:
                reports[sid] = json.loads(p.read_text())
            except Exception:
                continue

    n_apf_days = 0
    n_m2 = 0
    n_next = 0
    for r in data_rows:
        if not r:
            continue
        gaia = (r[gaia_i] if gaia_i < len(r) else "").strip()
        sid = gaia_id_from_row(gaia)
        if not sid:
            continue

        summ = out_dir / f"Gaia_DR3_{sid}_summary.txt"
        if summ.is_file():
            age = days_since_last_apf_from_summary(summ)
            if age is not None:
                while len(r) <= apf_days_i:
                    r.append("")
                r[apf_days_i] = f"{age:.2f}"
                n_apf_days += 1

        if sid not in reports:
            continue
        rep = reports[sid]
        masses = website_table_masses_from_report(rep)
        if masses["m2_msun"] is not None:
            while len(r) <= m2_i:
                r.append("")
            r[m2_i] = f"{masses['m2_msun']:.5f}"
            n_m2 += 1
        if masses["m2sin_i_msun"] is not None:
            while len(r) <= m2sini_i:
                r.append("")
            r[m2sini_i] = f"{masses['m2sin_i_msun']:.5f}"
        if masses["m2_at_i_msun"] is not None:
            while len(r) <= m2over_i:
                r.append("")
            r[m2over_i] = f"{masses['m2_at_i_msun']:.5f}"
        nxt = next_rv_event_from_fit_report(rep)
        if nxt is not None:
            while len(r) <= next_rv_i:
                r.append("")
            r[next_rv_i] = f"{nxt:.3f}"
            n_next += 1

    with data_csv.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    return {
        "data_rows": len(data_rows),
        "columns": len(hdr),
        "stray_img_cleared": n_stray,
        "apf_days_filled": n_apf_days,
        "m2_filled": n_m2,
        "next_rv_filled": n_next,
        "reports_loaded": len(reports),
    }


def main() -> int:
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
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else repo / "output"
    reports_dir = Path(args.reports_dir) if args.reports_dir else repo / "rv_fit_reports"
    stats = update_table_columns(
        Path(args.data_csv),
        out_dir=out_dir,
        reports_dir=reports_dir,
    )
    print(
        f"updated {args.data_csv}: {stats['data_rows']} rows, {stats['columns']} columns, "
        f"cleared {stats['stray_img_cleared']} stray <img>, "
        f"apf_days={stats['apf_days_filled']}, m2={stats['m2_filled']}, "
        f"next_rv={stats['next_rv_filled']} (from {stats['reports_loaded']} reports)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
