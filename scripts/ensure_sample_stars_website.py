#!/usr/bin/env python3
"""Ensure ATF22 / E24 sample stars exist in data.csv and output summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Set

from darkhunter_rv.gaia_utils import (
    parse_gaia_metadata_from_star_summary,
    query_gaia_data,
    resolve_gaia_data,
    star_summary_metadata_needs_photometry,
)
from darkhunter_rv.io_utils import write_star_summary
from darkhunter_rv.summary_paths import discover_summary_path
from darkhunter_rv.website_table_csv import (
    build_table_row_from_metadata,
    gaia_id_from_row,
    normalize_data_csv,
)


def load_sample_gaia_ids(tags_path: Path) -> Set[str]:
    data = json.loads(tags_path.read_text(encoding="utf-8"))
    out: Set[str] = set()
    for key in ("ATF22", "E24_NS", "E24_FULL"):
        for raw in data.get(key, []):
            sid = str(raw).strip()
            if sid:
                out.add(sid)
    return out


def ensure_summary(
    output_dir: Path,
    gaia_id: str,
    *,
    force_gaia: bool,
) -> Optional[Path]:
    flat = output_dir / f"Gaia_DR3_{gaia_id}_summary.txt"
    existing = discover_summary_path(output_dir, gaia_id)
    target = flat
    if existing is not None and existing.is_file():
        target = existing

    gaia_data = None
    if force_gaia or not target.is_file():
        gaia_data = query_gaia_data(int(gaia_id))
    else:
        meta = parse_gaia_metadata_from_star_summary(target) or {}
        if star_summary_metadata_needs_photometry(meta):
            gaia_data = query_gaia_data(int(gaia_id))
        else:
            gaia_data = resolve_gaia_data(int(gaia_id), target, force_query=False)

    if not gaia_data:
        return None

    write_star_summary(gaia_id, gaia_data, [])
    written = output_dir / f"Gaia_DR3_{gaia_id}_summary.txt"
    return written if written.is_file() else target


def ensure_table_rows(
    data_csv: Path,
    sample_ids: Set[str],
    output_dir: Path,
) -> Dict[str, int]:
    rows = list(csv.reader(data_csv.open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{data_csv} is empty")
    hdr = rows[0]
    data_rows = rows[1:]
    normalize_data_csv(hdr, data_rows)

    gaia_i = hdr.index("GAIA NAME")
    existing = {
        gaia_id_from_row(r[gaia_i] if gaia_i < len(r) else "")
        for r in data_rows
        if r
    }
    existing.discard("")

    added = 0
    for gid in sorted(sample_ids):
        if gid in existing:
            continue
        summ = discover_summary_path(output_dir, gid)
        if summ is None or not summ.is_file():
            continue
        meta = parse_gaia_metadata_from_star_summary(summ) or {}
        if not meta:
            continue
        data_rows.append(build_table_row_from_metadata(hdr, gid, meta))
        added += 1

    with data_csv.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([hdr] + data_rows)
    return {"added_rows": added, "total_rows": len(data_rows), "columns": len(hdr)}


def run_rv_plots_for_ids(
    sample_ids: Set[str],
    *,
    repo: Path,
    output_dir: Path,
    plots_root: Path,
    reports_dir: Path,
    obs_cache: Optional[Path],
    web_root: Optional[Path],
) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    rc = 0
    plot_script = repo / "scripts/plot_rv_from_summaries.py"
    for gid in sorted(sample_ids):
        cmd = [
            sys.executable,
            str(plot_script),
            "--summary-dir",
            str(output_dir),
            "--plots-root",
            str(plots_root),
            "--reports-dir",
            str(reports_dir),
            "--star-id",
            gid,
        ]
        if obs_cache is not None:
            cmd.extend(["--observability-cache", str(obs_cache)])
        if web_root is not None:
            cmd.extend(["--web-root", str(web_root)])
        proc = subprocess.run(cmd, cwd=repo, env=env)
        if proc.returncode != 0:
            rc = proc.returncode
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ensure sample-tag stars have summaries, table rows, and optional RV plots."
    )
    ap.add_argument(
        "--tags-json",
        default=None,
        help="sample_tags.json path (default: REPO/website/rv/tables/sample_tags.json)",
    )
    ap.add_argument(
        "--data-csv",
        default="/var/www/html/darkhunter/rv/tables/data.csv",
    )
    ap.add_argument("--output-dir", default=None, help="Pipeline output (default: REPO/output)")
    ap.add_argument("--plots-root", default=None, help="Per-star plot tree (default: output)")
    ap.add_argument("--reports-dir", default=None, help="rv_fit_reports (APF window fallback)")
    ap.add_argument("--observability-cache", default=None)
    ap.add_argument("--web-root", default=None)
    ap.add_argument("--force-gaia", action="store_true", help="Re-query Gaia for every sample star")
    ap.add_argument("--with-plots", action="store_true", help="Build RV data plots when epochs exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    tags_path = Path(args.tags_json) if args.tags_json else repo / "website/rv/tables/sample_tags.json"
    out_dir = Path(args.output_dir) if args.output_dir else repo / "output"
    plots_root = Path(args.plots_root) if args.plots_root else out_dir
    reports_dir = Path(args.reports_dir) if args.reports_dir else repo / "rv_fit_reports"
    data_csv = Path(args.data_csv)
    obs_cache = Path(args.observability_cache) if args.observability_cache else None
    web_root = Path(args.web_root) if args.web_root else None

    if not tags_path.is_file():
        print(f"[ERROR] missing {tags_path}", file=sys.stderr)
        return 2
    if not data_csv.is_file():
        print(f"[ERROR] missing {data_csv}", file=sys.stderr)
        return 2

    sample_ids = load_sample_gaia_ids(tags_path)
    print(f"sample tags: {len(sample_ids)} unique Gaia id(s) from {tags_path.name}")

    n_summ = 0
    for gid in sorted(sample_ids):
        if args.dry_run:
            print(f"  [dry-run] summary Gaia_DR3_{gid}")
            continue
        path = ensure_summary(out_dir, gid, force_gaia=args.force_gaia)
        if path is not None:
            n_summ += 1
        else:
            print(f"[WARN] Gaia_DR3_{gid}: could not write summary", file=sys.stderr)

    if args.dry_run:
        return 0

    table_stats = ensure_table_rows(data_csv, sample_ids, out_dir)
    print(
        f"data.csv: +{table_stats['added_rows']} row(s), "
        f"{table_stats['total_rows']} total, {table_stats['columns']} columns"
    )

    if args.with_plots:
        rc = run_rv_plots_for_ids(
            sample_ids,
            repo=repo,
            output_dir=out_dir,
            plots_root=plots_root,
            reports_dir=reports_dir,
            obs_cache=obs_cache,
            web_root=web_root,
        )
        if rc != 0:
            print(f"[WARN] plot_rv_from_summaries exit {rc}", file=sys.stderr)
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
