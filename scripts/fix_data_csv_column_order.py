#!/usr/bin/env python3
"""Repair tables/data.csv after header-only column reorder (restores row alignment)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def move_columns_after(hdr: list[str], rows: list[list[str]], names: list[str], after: str) -> None:
    """Move columns in hdr and every data row so names appear right after `after`."""
    if after not in hdr:
        return
    insert_at = hdr.index(after) + 1
    moving: list[tuple[int, str]] = []
    for name in names:
        if name not in hdr:
            continue
        old_i = hdr.index(name)
        moving.append((old_i, name))
    if not moving:
        return
    # Preserve cell values while removing columns from right to left.
    extracted: list[tuple[str, list[str]]] = []
    for old_i, name in sorted(moving, key=lambda x: x[0], reverse=True):
        hdr.pop(old_i)
        col_vals = []
        for r in rows:
            col_vals.append(r.pop(old_i) if old_i < len(r) else "")
        extracted.append((name, list(reversed(col_vals))))
    extracted.reverse()
    for offset, (name, col_vals) in enumerate(extracted):
        hdr.insert(insert_at + offset, name)
        for r, val in zip(rows, col_vals):
            r.insert(insert_at + offset, val)


def clear_media_cells(hdr: list[str], rows: list[list[str]]) -> None:
    """Drop legacy embedded <img> tags; the website builds plot URLs in script.js."""
    for name in ("RV PLOT", "RV FIT", "FLUX PLOT", "SOURCE IMAGE"):
        if name not in hdr:
            continue
        ci = hdr.index(name)
        for r in rows:
            while len(r) <= ci:
                r.append("")
            r[ci] = ""


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
    for col in ("M2sin i (Msun)", "(M2sin i)/(sin i) (Msun)"):
        if col not in hdr:
            hdr.append(col)
    for r in data_rows:
        while len(r) < len(hdr):
            r.append("")

    move_columns_after(
        hdr,
        data_rows,
        ["M2sin i (Msun)", "(M2sin i)/(sin i) (Msun)"],
        "M2 (Msun)",
    )
    clear_media_cells(hdr, data_rows)

    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"fixed: {path} ({len(rows) - 1} data rows, {len(hdr)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
