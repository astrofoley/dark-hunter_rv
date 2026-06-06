"""Discover star summary files under output/ (no scipy / fit dependencies)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary


def parse_object_id_from_summary(path: Path) -> Optional[str]:
    m = re.search(r"Gaia_DR3_(\d{18,19})", f"{path.parent.name}/{path.stem}")
    if m:
        return m.group(1)
    meta = parse_gaia_metadata_from_star_summary(path)
    if meta is not None:
        sid = meta.get("Source_ID", meta.get("source_id"))
        if sid is not None:
            try:
                return str(int(sid))
            except (TypeError, ValueError):
                pass
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# OBJECT ID"):
            m = re.search(r"=\s*([0-9]+)", line)
            if m:
                return m.group(1)
    m = re.match(r"([0-9]+)_summary$", path.stem)
    return m.group(1) if m else None


def count_pipeline_rows(path: Path) -> int:
    """Rows in [PIPELINE RESULTS] (or legacy table), including NaN RV epochs."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if "[PIPELINE RESULTS]" in text:
        lines = text.split("[PIPELINE RESULTS]", 1)[-1].splitlines()
    else:
        lines = text.splitlines()
    n = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or (line.startswith("[") and line.endswith("]")):
            continue
        if len(line.split()) >= 5:
            n += 1
    return n


def discover_summary_files(output_dir: Path) -> list[Path]:
    """One summary per Gaia source_id; prefer flat output/Gaia_DR3_<id>_summary.txt over nested stubs."""
    if not output_dir.is_dir():
        return []
    out_root = output_dir.resolve()
    by_sid: dict[str, Path] = {}

    def _rank(path: Path) -> tuple:
        flat = int(path.parent.resolve() == out_root)
        gaia_named = int(path.name.startswith("Gaia_DR3_"))
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (flat, gaia_named, count_pipeline_rows(path), mtime)

    for p in output_dir.rglob("*_summary.txt"):
        if not p.is_file():
            continue
        sid = parse_object_id_from_summary(p)
        if not sid:
            continue
        prev = by_sid.get(sid)
        if prev is None or _rank(p) > _rank(prev):
            by_sid[sid] = p.resolve()
    return sorted(by_sid.values())


def discover_summary_path(output_dir: Path, gaia_source_id: str) -> Optional[Path]:
    """Best summary path for one Gaia source (flat or nested under output/)."""
    sid = str(gaia_source_id).strip()
    if not sid:
        return None
    for p in discover_summary_files(output_dir):
        if parse_object_id_from_summary(p) == sid:
            return p
    return None
