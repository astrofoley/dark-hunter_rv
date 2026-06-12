#!/usr/bin/env python3
"""Query external spectroscopic RV catalogs and patch star summary [EXTERNAL RV DATA]."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from darkhunter_rv.apogee_rv_query import APOGEE_TELESCOPE_PREFIX, query_apogee_rvs_for_gaia_ids
from darkhunter_rv.desi_rv_query import DESI_TELESCOPE_PREFIX, query_desi_rvs_for_gaia_ids
from darkhunter_rv.galah_rv_query import GALAH_TELESCOPE_PREFIX, query_galah_rvs_for_gaia_ids
from darkhunter_rv.ges_rv_query import GES_TELESCOPE_PREFIX, query_ges_rvs_for_gaia_ids
from darkhunter_rv.gaia_utils import (
    merge_external_rv_lists,
    normalize_parsed_star_metadata,
    parse_external_rvs_from_star_summary,
    parse_gaia_metadata_from_star_summary,
    query_external_rvs_for_source,
    replace_external_rv_section_in_summary,
)
from darkhunter_rv.summary_paths import discover_summary_files, parse_object_id_from_summary

ALL_SOURCES = ("lamost", "rave", "desi", "galah", "apogee", "ges")
# VizieR batch + DESI; LAMOST/RAVE hit Gaia TAP per star (often 500s during DR4 prep).
DEFAULT_SOURCES = ("desi", "galah", "apogee")
SOURCE_PREFIXES: dict[str, tuple[str, ...]] = {
    "lamost": ("LAMOST_LRS", "LAMOST_MRS"),
    "rave": ("RAVE_DR6",),
    "desi": (DESI_TELESCOPE_PREFIX,),
    "galah": (GALAH_TELESCOPE_PREFIX,),
    "apogee": (APOGEE_TELESCOPE_PREFIX,),
    "ges": (GES_TELESCOPE_PREFIX,),
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _discover_summaries(out_dir: Path) -> list[Path]:
    flat = sorted(out_dir.glob("Gaia_DR3_*_summary.txt"))
    if flat:
        return flat
    return discover_summary_files(out_dir)


def _parse_sources(raw: str) -> tuple[str, ...]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    unknown = [p for p in parts if p not in ALL_SOURCES]
    if unknown:
        raise SystemExit(f"Unknown --sources entries: {unknown} (choose from {ALL_SOURCES})")
    return tuple(parts or DEFAULT_SOURCES)


def _batch_query(
    source: str,
    all_ids: list[int],
    positions_by_id: dict[int, tuple[float, float]],
    *,
    max_rv_err: float | None,
    progress: bool,
) -> dict[int, list[dict]]:
    if source == "desi":
        return query_desi_rvs_for_gaia_ids(
            all_ids, positions_by_id=positions_by_id, progress=progress
        )
    if source == "galah":
        return query_galah_rvs_for_gaia_ids(
            all_ids,
            positions_by_id=positions_by_id,
            max_rv_err=max_rv_err,
            progress=progress,
        )
    if source == "apogee":
        return query_apogee_rvs_for_gaia_ids(
            all_ids,
            positions_by_id=positions_by_id,
            max_rv_err=max_rv_err,
            progress=progress,
        )
    if source == "ges":
        return query_ges_rvs_for_gaia_ids(
            all_ids,
            positions_by_id=positions_by_id,
            max_rv_err=max_rv_err,
            progress=progress,
        )
    return {}


def _merge_source_rows(
    existing: list,
    new_rows: list,
    source: str,
) -> list:
    prefixes = SOURCE_PREFIXES.get(source, ())
    if not prefixes:
        return existing + new_rows
    return merge_external_rv_lists(existing, new_rows, replace_prefixes=prefixes)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch external survey RVs and update summary [EXTERNAL RV DATA] blocks."
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Pipeline output root (default: repo output/ or DARKHUNTER_OUTPUT_DIR)",
    )
    ap.add_argument("--star-id", default=None, help="Optional single Gaia DR3 source id")
    ap.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=(
            f"Comma-separated catalogs (default: {','.join(DEFAULT_SOURCES)}). "
            f"All choices: {','.join(ALL_SOURCES)}. "
            "Add lamost,rave for Gaia-archive cross-match (slow; archive may 500)."
        ),
    )
    ap.add_argument(
        "--max-rv-err",
        type=float,
        default=None,
        metavar="KM_S",
        help="Optional: drop VizieR rows with rv_err above this (km/s). Default: keep all.",
    )
    ap.add_argument("--dry-run", action="store_true", help="List summaries only; do not query or write")
    args = ap.parse_args()

    sources = _parse_sources(args.sources)
    _log(f"update_summary_external_rvs: sources={sources}")

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        os.environ.get("DARKHUNTER_OUTPUT_DIR", repo / "output")
    )
    _log(f"Output dir: {out_dir}")

    files = _discover_summaries(out_dir)
    if args.star_id:
        files = [p for p in files if parse_object_id_from_summary(p) == str(args.star_id)]
    if not files:
        _log("No summary files found.")
        return 2

    targets: list[tuple[Path, int, dict]] = []
    positions_by_id: dict[int, tuple[float, float]] = {}

    for summ in files:
        sid_s = parse_object_id_from_summary(summ)
        if not sid_s:
            continue
        sid = int(sid_s)
        meta = parse_gaia_metadata_from_star_summary(summ)
        if meta is None:
            continue
        meta = normalize_parsed_star_metadata(meta)
        try:
            ra = float(meta.get("RA"))
            dec = float(meta.get("Dec"))
            if np.isfinite(ra) and np.isfinite(dec):
                positions_by_id[sid] = (ra, dec)
        except (TypeError, ValueError):
            pass
        targets.append((summ, sid, meta))

    _log(f"Found {len(targets)} star(s) with metadata.")

    if args.dry_run:
        for summ, sid, _ in targets:
            _log(f"would_update Gaia_DR3_{sid} {summ}")
        return 0

    if not targets:
        return 2

    batch_sources = [s for s in sources if s in ("desi", "galah", "apogee", "ges")]
    per_source: dict[str, dict[int, list[dict]]] = {}
    all_ids = [sid for _, sid, _ in targets]

    for src in batch_sources:
        _log(f"Batch query: {src}...")
        per_source[src] = _batch_query(
            src,
            all_ids,
            positions_by_id,
            max_rv_err=args.max_rv_err,
            progress=True,
        )

    per_star_sources = [s for s in sources if s in ("lamost", "rave")]

    updated = 0
    total_rows = 0
    for i, (summ, sid, meta) in enumerate(targets, start=1):
        existing = parse_external_rvs_from_star_summary(summ)
        merged = list(existing)

        for src in batch_sources:
            rows = per_source.get(src, {}).get(sid, [])
            merged = _merge_source_rows(merged, rows, src)

        if per_star_sources:
            tap_rows = query_external_rvs_for_source(
                sid,
                meta,
                sources=tuple(per_star_sources),
            )
            for src in per_star_sources:
                prefixes = SOURCE_PREFIXES[src]
                src_rows = [
                    r
                    for r in tap_rows
                    if any(str(r.get("telescope", "")).startswith(p) for p in prefixes)
                ]
                merged = _merge_source_rows(merged, src_rows, src)

        replace_external_rv_section_in_summary(summ, merged)
        n_new = sum(
            len(per_source.get(src, {}).get(sid, []))
            for src in batch_sources
        )
        if n_new:
            updated += 1
            total_rows += n_new
            _log(f"[{i}/{len(targets)}] Gaia_DR3_{sid}: wrote external RVs ({len(merged)} total rows)")
        else:
            _log(f"[{i}/{len(targets)}] Gaia_DR3_{sid}: no new batch rows ({len(merged)} total)")

    _log(f"Done: {updated} summaries updated ({total_rows} new batch rows).")
    return 0


if __name__ == "__main__":
    print("update_summary_external_rvs", flush=True)
    raise SystemExit(main())
