#!/usr/bin/env python3
"""Build rv_fit_reports/observability_windows_cache.json for APF visibility shading."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

from darkhunter_rv.apf_observability import observability_for_summary
from darkhunter_rv.website_table_csv import gaia_id_from_row
from darkhunter_rv.summary_paths import discover_summary_path


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
    ap.add_argument("--gaia-id", default=None, help="Single Gaia DR3 source id (default: all table rows).")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N table rows (0 = only final summary).",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else repo / "output"
    cache_path = Path(args.cache) if args.cache else repo / "rv_fit_reports" / "observability_windows_cache.json"
    data_csv = Path(args.data_csv)

    gaia_ids: list[str] = []
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
    total = len(gaia_ids)
    if total:
        print(f"Building APF observability for {total} table stars...", flush=True)
    for i, sid in enumerate(gaia_ids, start=1):
        summ = discover_summary_path(out_dir, sid)
        if summ is None:
            skipped += 1
            if args.progress_every and (i == 1 or i % args.progress_every == 0 or i == total):
                print(f"  [{i}/{total}] {sid}: skipped (no summary)", flush=True)
            continue
        row = observability_for_summary(summ)
        if row is None:
            skipped += 1
            if args.progress_every and (i == 1 or i % args.progress_every == 0 or i == total):
                print(f"  [{i}/{total}] {sid}: skipped (no windows)", flush=True)
            continue
        entry = {k: v for k, v in row.items() if k != "gaia_source_id"}
        cache[sid] = entry
        built += 1
        if args.progress_every and (i == 1 or i % args.progress_every == 0 or i == total):
            print(f"  [{i}/{total}] {sid}: ok", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    print(f"Wrote {built} observability entries to {cache_path} (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
