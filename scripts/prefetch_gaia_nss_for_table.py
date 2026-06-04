#!/usr/bin/env python3
"""Query Gaia NSS (period, e, inclination, binary masses) for stars in tables/data.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from darkhunter_rv.website_table_csv import gaia_id_from_row
from fit_apf_rv_keplerian import prefetch_gaia_nss_bulk


def main() -> int:
    ap = argparse.ArgumentParser(description="Prefetch Gaia NSS into gaia_nss_cache.json for table stars.")
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--reports-dir", required=True)
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_path = reports_dir / "gaia_nss_cache.json"

    ids: list[str] = []
    with Path(args.data_csv).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return 1
    hdr = rows[0]
    gaia_i = hdr.index("GAIA NAME")
    for r in rows[1:]:
        if not r:
            continue
        sid = gaia_id_from_row(r[gaia_i] if gaia_i < len(r) else "")
        if sid:
            ids.append(sid)

    prefetch_gaia_nss_bulk(sorted(set(ids)), cache_path)
    print(f"prefetched Gaia NSS for {len(set(ids))} sources → {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
