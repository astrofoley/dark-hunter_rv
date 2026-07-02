#!/usr/bin/env python3
"""Backfill G/BP/RP photometry in star summaries from Gaia DR3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Set

from darkhunter_rv.gaia_utils import (
    parse_gaia_metadata_from_star_summary,
    query_gaia_data,
    star_summary_metadata_needs_photometry,
)
from darkhunter_rv.io_utils import write_star_summary
from darkhunter_rv.summary_paths import discover_summary_files, parse_object_id_from_summary


def load_sample_gaia_ids(tags_path: Path) -> Set[str]:
    data = json.loads(tags_path.read_text(encoding="utf-8"))
    out: Set[str] = set()
    for key in ("ATF22", "E24_NS", "E24_FULL"):
        for raw in data.get(key, []):
            sid = str(raw).strip()
            if sid:
                out.add(sid)
    return out


def patch_summaries(
    output_dir: Path,
    *,
    gaia_ids: Optional[Set[str]] = None,
    dry_run: bool = False,
) -> dict:
    patched = 0
    skipped = 0
    failed = 0
    for summ in discover_summary_files(output_dir):
        sid = parse_object_id_from_summary(summ)
        if not sid:
            skipped += 1
            continue
        if gaia_ids is not None and sid not in gaia_ids:
            continue
        meta = parse_gaia_metadata_from_star_summary(summ) or {}
        if not star_summary_metadata_needs_photometry(meta):
            skipped += 1
            continue
        if dry_run:
            print(f"  [dry-run] patch photometry Gaia_DR3_{sid}")
            patched += 1
            continue
        gaia_data = query_gaia_data(int(sid))
        if not gaia_data:
            print(f"[WARN] Gaia_DR3_{sid}: query failed", file=sys.stderr)
            failed += 1
            continue
        write_star_summary(sid, gaia_data, [])
        patched += 1
    return {"patched": patched, "skipped": skipped, "failed": failed}


def main() -> int:
    ap = argparse.ArgumentParser(description="Add Gaia G/BP/RP to summary [GAIA METADATA] blocks.")
    ap.add_argument("--output-dir", default=None, help="Pipeline output root (default: REPO/output)")
    ap.add_argument("--tags-json", default=None, help="Limit to sample_tags.json ids")
    ap.add_argument("--gaia-id", default=None, help="Single Gaia DR3 source id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else repo / "output"
    gaia_ids: Optional[Set[str]] = None
    if args.gaia_id:
        gaia_ids = {str(args.gaia_id).strip()}
    elif args.tags_json or not args.gaia_id:
        tags_path = (
            Path(args.tags_json)
            if args.tags_json
            else repo / "website/rv/tables/sample_tags.json"
        )
        if tags_path.is_file():
            gaia_ids = load_sample_gaia_ids(tags_path)

    stats = patch_summaries(out_dir, gaia_ids=gaia_ids, dry_run=args.dry_run)
    print(
        f"patch_summary_gaia_photometry: patched={stats['patched']} "
        f"skipped={stats['skipped']} failed={stats['failed']}"
    )
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
