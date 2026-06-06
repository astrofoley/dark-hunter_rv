#!/usr/bin/env python3
"""Derive Inclination from Thiele-Innes elements in star summary [GAIA METADATA] blocks."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from darkhunter_rv.thiele_innes_inclination import (
    fill_inclination_in_metadata,
    metadata_inclination_is_missing,
    thiele_innes_from_metadata,
)
from fit_apf_rv_keplerian import discover_summary_files, parse_object_id_from_summary
from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary


def _format_meta_value(key: str, val: float) -> str:
    if key.endswith("_Error") or key.endswith("Error"):
        return f"{val:.8f}"
    return f"{val:.8f}"


def _upsert_gaia_metadata_fields(text: str, updates: dict[str, float]) -> str:
    """Insert or replace Key: value lines inside [GAIA METADATA]."""
    lines = text.splitlines(keepends=True)
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.strip() == "[GAIA METADATA]":
            start = i + 1
            continue
        if start is not None and line.strip().startswith("[") and line.strip().endswith("]"):
            end = i
            break
    if start is None:
        return text

    block_end = end if end is not None else len(lines)
    key_re = {k: re.compile(rf"^{re.escape(k)}\s*:") for k in updates}

    present: set[str] = set()
    for i in range(start, block_end):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for key, rx in key_re.items():
            if rx.match(stripped):
                lines[i] = f"{key}: {_format_meta_value(key, updates[key])}\n"
                present.add(key)
                break

    missing = [k for k in updates if k not in present]
    if missing:
        insert_at = block_end
        new_lines = [f"{k}: {_format_meta_value(k, updates[k])}\n" for k in missing]
        lines[insert_at:block_end] = new_lines + lines[insert_at:block_end]

    return "".join(lines)


def patch_summary_path(path: Path, *, dry_run: bool) -> str:
    meta = parse_gaia_metadata_from_star_summary(path)
    if not meta:
        return "no_metadata"
    if thiele_innes_from_metadata(meta) is None:
        return "no_thiele_innes"
    if not metadata_inclination_is_missing(meta):
        return "has_inclination"

    working = dict(meta)
    if not fill_inclination_in_metadata(working):
        return "derive_failed"

    updates: dict[str, float] = {"Inclination": float(working["Inclination"])}
    if "Inclination_Error" in working:
        updates["Inclination_Error"] = float(working["Inclination_Error"])

    if dry_run:
        return "would_patch"

    text = path.read_text(encoding="utf-8", errors="replace")
    path.write_text(_upsert_gaia_metadata_fields(text, updates), encoding="utf-8")
    return "patched"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fill Inclination in summary.txt from Gaia Thiele-Innes A,B,F,G."
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing *_summary.txt files (searched recursively)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Report only; do not rewrite files")
    args = ap.parse_args()

    paths = discover_summary_files(args.output_dir)
    if not paths:
        print(f"No summaries under {args.output_dir.resolve()}")
        return 1

    counts: dict[str, int] = {}
    for path in paths:
        sid = parse_object_id_from_summary(path) or path.stem
        status = patch_summary_path(path, dry_run=args.dry_run)
        counts[status] = counts.get(status, 0) + 1
        if status in ("patched", "would_patch"):
            print(f"{status}: {sid} ({path})")

    print(
        f"done: {len(paths)} summaries — "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
