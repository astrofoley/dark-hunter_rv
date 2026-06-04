"""Normalize tables/data.csv layout and helpers for website column updates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.time import Time

MEDIA_COLUMNS = frozenset({"RV PLOT", "RV FIT", "FLUX PLOT", "SOURCE IMAGE"})

MASS_COLUMNS = ("M2sin i (Msun)", "(M2sin i)/(sin i) (Msun)")

GAIA_DATA_COLUMN = "GAIA DATA"
SCHEDULE_COLUMNS = ("DAYS SINCE LAST APF", "NEXT RV EVENT (DATE)")
LEGACY_COLUMN_RENAMES = {"NEXT RV EVENT (MJD)": "NEXT RV EVENT (DATE)"}

# Inserted immediately after these anchors when rebuilding header order.
_INSERT_AFTER: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("M2 (Msun)", MASS_COLUMNS),
    ("DATA PRODUCTS", (GAIA_DATA_COLUMN,)),
    (GAIA_DATA_COLUMN, SCHEDULE_COLUMNS),
)


def mjd_to_yyyy_mm_dd(mjd: float) -> str:
    t = Time(float(mjd), format="mjd", scale="utc")
    return t.datetime.strftime("%Y/%m/%d")


def format_next_rv_event_cell(mjd: Optional[float]) -> str:
    if mjd is None or not np.isfinite(mjd):
        return ""
    return mjd_to_yyyy_mm_dd(float(mjd))


def row_dict(hdr: List[str], row: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, name in enumerate(hdr):
        key = (name or "").strip()
        if not key:
            continue
        key = LEGACY_COLUMN_RENAMES.get(key, key)
        out[key] = row[i] if i < len(row) else ""
    return out


def align_rows_to_header(hdr: List[str], rows: List[List[str]]) -> None:
    n = len(hdr)
    for r in rows:
        if len(r) < n:
            r.extend([""] * (n - len(r)))
        elif len(r) > n:
            del r[n:]


def clear_media_cells(hdr: List[str], rows: List[List[str]]) -> None:
    for name in MEDIA_COLUMNS:
        if name not in hdr:
            continue
        ci = hdr.index(name)
        for r in rows:
            while len(r) <= ci:
                r.append("")
            r[ci] = ""


def clear_stray_plot_html(hdr: List[str], rows: List[List[str]]) -> int:
    cleared = 0
    for ci, name in enumerate(hdr):
        if name in MEDIA_COLUMNS:
            continue
        for r in rows:
            while len(r) <= ci:
                r.append("")
            if r[ci] and "<img" in r[ci].lower():
                r[ci] = ""
                cleared += 1
    return cleared


def _ensure_columns_present(hdr: List[str], names: Tuple[str, ...]) -> None:
    for name in names:
        if name not in hdr:
            hdr.append(name)


def build_canonical_header(hdr: List[str]) -> List[str]:
    """Stable column order: mass cols after M2; schedule cols after DATA PRODUCTS."""
    seen = set()
    base: List[str] = []
    for h in hdr:
        key = (h or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        base.append(key)

    for _, names in _INSERT_AFTER:
        _ensure_columns_present(base, names)
    _ensure_columns_present(base, (GAIA_DATA_COLUMN,))

    out: List[str] = []
    inserted_mass = False
    inserted_sched = False
    for h in base:
        if h in MASS_COLUMNS or h in SCHEDULE_COLUMNS or h == GAIA_DATA_COLUMN:
            continue
        out.append(h)
        if h == "M2 (Msun)" and not inserted_mass:
            for c in MASS_COLUMNS:
                if c in base:
                    out.append(c)
            inserted_mass = True
        if h == "DATA PRODUCTS":
            if GAIA_DATA_COLUMN in base and GAIA_DATA_COLUMN not in out:
                out.append(GAIA_DATA_COLUMN)
            if not inserted_sched:
                for c in SCHEDULE_COLUMNS:
                    if c in base:
                        out.append(c)
                inserted_sched = True

    if not inserted_mass:
        for c in MASS_COLUMNS:
            if c in base and c not in out:
                out.append(c)
    if not inserted_sched:
        for c in SCHEDULE_COLUMNS:
            if c in base and c not in out:
                out.append(c)

    for h in base:
        if h not in out:
            out.append(h)
    return out


def normalize_data_csv(hdr: List[str], data_rows: List[List[str]]) -> Tuple[List[str], int]:
    """
    Rebuild rows keyed by column name (fixes off-by-one from header-only edits).
    Clears media / stray <img> cells. Returns (new_hdr, n_stray_cleared).
    """
    _ensure_columns_present(hdr, MASS_COLUMNS + (GAIA_DATA_COLUMN,) + SCHEDULE_COLUMNS)
    align_rows_to_header(hdr, data_rows)

    old_hdr = list(hdr)
    maps = [row_dict(old_hdr, r) for r in data_rows]
    new_hdr = build_canonical_header(old_hdr)
    new_rows = [[maps[i].get(name, "") for name in new_hdr] for i in range(len(maps))]

    hdr[:] = new_hdr
    for i, r in enumerate(data_rows):
        new = new_rows[i] if i < len(new_rows) else []
        r.clear()
        r.extend(new)
    align_rows_to_header(hdr, data_rows)
    clear_media_cells(hdr, data_rows)
    n_stray = clear_stray_plot_html(hdr, data_rows)
    return hdr, n_stray


def _parse_pipeline_apf_mjds(summary_path: Path) -> List[float]:
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    if "[PIPELINE RESULTS]" not in text:
        return []
    from darkhunter_rv.rv_point_filters import mjd_is_valid

    block = text.split("[PIPELINE RESULTS]", 1)[-1]
    mjds: List[float] = []
    for raw in block.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            mjd = float(parts[1])
        except ValueError:
            continue
        if not mjd_is_valid(mjd):
            continue
        mjds.append(mjd)
    return mjds


def days_since_last_apf_from_summary(summary_path: Path, *, now_mjd: Optional[float] = None) -> Optional[float]:
    mjds = _parse_pipeline_apf_mjds(summary_path)
    if not mjds:
        return None
    now = float(Time.now().mjd) if now_mjd is None else float(now_mjd)
    latest = max(mjds)
    age = now - latest
    return float(age) if age >= 0 else None


def _finite_event_mjd(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def sooner_rv_extremum_mjd(
    variant_report: dict,
    *,
    now_mjd: Optional[float] = None,
) -> Optional[float]:
    """MJD of the nearer upcoming max or min RV (P+e fixed fit report fields)."""
    now = float(Time.now().mjd) if now_mjd is None else float(now_mjd)
    t_max = _finite_event_mjd(variant_report.get("next_rv_max_mjd"))
    t_min = _finite_event_mjd(variant_report.get("next_rv_min_mjd"))
    candidates: List[float] = []
    for t in (t_max, t_min):
        if t is not None and t >= now - 0.01:
            candidates.append(t)
    if not candidates:
        return None
    return float(min(candidates))


def next_rv_event_from_fit_report(report: dict) -> Optional[float]:
    variants = report.get("fit_variants")
    if isinstance(variants, dict) and "fix_period_ecc" in variants:
        rep = variants["fix_period_ecc"]
        if isinstance(rep, dict):
            t = sooner_rv_extremum_mjd(rep, now_mjd=report.get("now_mjd"))
            if t is not None:
                return t
    return sooner_rv_extremum_mjd(report, now_mjd=report.get("now_mjd"))


def parse_next_rv_cell_to_mjd(value: str) -> Optional[float]:
    """Parse CSV cell: YYYY/MM/DD or legacy MJD float."""
    s = (value or "").strip()
    if not s:
        return None
    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", s)
    if m:
        try:
            return float(
                Time(
                    {"year": int(m.group(1)), "month": int(m.group(2)), "day": int(m.group(3))},
                    format="ymd",
                    scale="utc",
                ).mjd
            )
        except Exception:
            return None
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except ValueError:
        return None


def gaia_id_from_row(gaia_cell: str) -> str:
    m = re.search(r"(\d{8,})", gaia_cell or "")
    return m.group(1) if m else ""
