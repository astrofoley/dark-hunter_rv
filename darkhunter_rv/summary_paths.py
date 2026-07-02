"""Discover star summary files under output/ (no scipy / fit dependencies)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Set

from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary
from darkhunter_rv.rv_point_filters import rv_epoch_is_valid

_PRIMARY_EPOCH_NAME = re.compile(r"^Gaia_DR3_(\d+)_epoch_\d+\.txt$")
_GAIA_DIR_NAME = re.compile(r"^Gaia_DR3_(\d+)$")


def is_primary_epoch_spectrum_name(filename: str) -> bool:
    """True for full-epoch Gaia_DR3_*_epoch_<N>.txt files (not *_order_* extracts)."""
    return bool(_PRIMARY_EPOCH_NAME.match(Path(filename).name))


def parse_object_id_from_summary(path: Path) -> Optional[str]:
    m = re.search(r"Gaia_DR3_(\d+)", f"{path.parent.name}/{path.stem}")
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
    m = re.match(r"Gaia_DR3_(\d+)_summary$", path.stem)
    if m:
        return m.group(1)
    m = re.match(r"(\d+)_summary$", path.stem)
    return m.group(1) if m else None


def discover_spec_gaia_ids(spec_root: Path) -> Set[str]:
    """Unique Gaia source ids under a reduction tree (any subdirectory depth)."""
    ids: Set[str] = set()
    if not spec_root.is_dir():
        return ids
    for path in spec_root.rglob("Gaia_DR3_*_epoch_*.txt"):
        if not path.is_file() or not is_primary_epoch_spectrum_name(path.name):
            continue
        match = _PRIMARY_EPOCH_NAME.match(path.name)
        if match:
            ids.add(match.group(1))
    for path in spec_root.rglob("Gaia_DR3_*"):
        if not path.is_dir():
            continue
        match = _GAIA_DIR_NAME.match(path.name)
        if match:
            ids.add(match.group(1))
    return ids


def discover_primary_epoch_files(spec_root: Path, gaia_source_id: str) -> list[Path]:
    """Primary per-epoch reduced spectra (exclude per-order splits)."""
    sid = str(gaia_source_id).strip()
    if not sid or not spec_root.is_dir():
        return []
    out: list[Path] = []
    for path in spec_root.rglob(f"Gaia_DR3_{sid}_epoch_*.txt"):
        if path.is_file() and is_primary_epoch_spectrum_name(path.name):
            out.append(path)
    return sorted(out, key=lambda p: p.name)


def count_valid_pipeline_rv_epochs(path: Path) -> int:
    """Finite pipeline RV epochs in [PIPELINE RESULTS] (no matplotlib / fit deps)."""
    if not path.is_file():
        return 0
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
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            mjd = float(parts[1])
            rv = float(parts[2])
        except ValueError:
            continue
        if rv_epoch_is_valid(mjd, rv):
            n += 1
    return n


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
    """One summary per Gaia source_id; prefer the file with the most pipeline epochs."""
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
        n_valid = count_valid_pipeline_rv_epochs(path)
        n_rows = count_pipeline_rows(path)
        return (n_valid, n_rows, flat, gaia_named, mtime)

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
