#!/usr/bin/env python3
"""Repair tables/data.csv column alignment and clear stale plot HTML."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from darkhunter_rv.website_table_csv import normalize_data_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Fix data.csv column order and clear stale plot HTML.")
    ap.add_argument(
        "--data-csv",
        default="/var/www/html/darkhunter/rv/tables/data.csv",
        help="Path to tables/data.csv",
    )
    args = ap.parse_args()
    path = Path(args.data_csv)
    if not path.is_file():
        raise SystemExit(f"not found: {path}")

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit("empty csv")

    hdr = rows[0]
    data_rows = rows[1:]
    _, n_stray = normalize_data_csv(hdr, data_rows)

    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(
        f"fixed: {path} ({len(rows) - 1} data rows, {len(hdr)} columns, "
        f"cleared {n_stray} stray <img> cells)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
