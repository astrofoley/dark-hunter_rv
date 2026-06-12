"""Query DESI Milky Way Survey RVs from NOIRLab Astro Data Lab (TAP)."""

from __future__ import annotations

import csv
import io
import logging
from typing import Any, Sequence

import numpy as np
import requests

from .cone_utils import angular_sep_deg, filter_rows_in_cone
from .rv_frame import NativeFrame, finalize_external_rv_record
from .vizier_tap import parse_gaia_source_id

logger = logging.getLogger(__name__)

DATALAB_TAP_SYNC = "https://datalab.noirlab.edu/tap/sync"
DESI_RELEASE = "dr1"
DESI_TABLE = f"desi_{DESI_RELEASE}.mws"
ZPIX_TABLE = f"desi_{DESI_RELEASE}.zpix"
DESI_TELESCOPE_PREFIX = "DESI_DR1"
# Match DESI Data Lab Gaia crossmatch radius (1.5 arcsec).
DESI_CONE_RADIUS_ARCSEC = 1.5
DESI_CONE_RADIUS_DEG = DESI_CONE_RADIUS_ARCSEC / 3600.0

_DESI_RV_SELECT = """
SELECT m.source_id, m.targetid, m.survey, m.program,
       m.rv_adop, m.rv_err, m.rvs_warn,
       m.target_ra, m.target_dec,
       z.min_mjd, z.max_mjd
"""

_DESI_RV_FROM = """
FROM {mws} AS m
LEFT JOIN {zpix} AS z
  ON m.targetid = z.targetid AND m.survey = z.survey AND m.program = z.program
"""

_DESI_RV_WHERE_BASE = """
WHERE m.rv_adop IS NOT NULL
  AND m.success = 1
"""

_DESI_RV_SQL_BY_ID = (
    _DESI_RV_SELECT
    + _DESI_RV_FROM
    + _DESI_RV_WHERE_BASE
    + """
  AND m.source_id IN ({ids})
"""
)

_DESI_RV_SQL_BY_BOX = (
    _DESI_RV_SELECT
    + _DESI_RV_FROM
    + _DESI_RV_WHERE_BASE
    + """
  AND m.target_ra BETWEEN ({ra}) - {radius_deg} AND ({ra}) + {radius_deg}
  AND m.target_dec BETWEEN ({dec}) - {radius_deg} AND ({dec}) + {radius_deg}
"""
)


def execute_datalab_adql(sql: str, *, timeout: float = 180.0) -> list[dict[str, Any]]:
    """Run ADQL on Astro Data Lab TAP (sync, CSV)."""
    resp = requests.get(
        DATALAB_TAP_SYNC,
        params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": sql},
        timeout=timeout,
    )
    resp.raise_for_status()
    text = resp.text.strip()
    if text.startswith("<?xml") or text.startswith("<"):
        raise RuntimeError(f"Data Lab TAP query failed: {text[:500]}")
    if not text:
        return []
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def _row_dedupe_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("targetid", "")),
        str(row.get("survey", "")),
        str(row.get("program", "")),
    )


def _merge_desi_rows(*groups: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for group in groups:
        for row in group:
            merged[_row_dedupe_key(row)] = row
    return list(merged.values())


def _mjd_from_row(row: dict[str, Any]) -> float:
    try:
        t0 = float(row.get("min_mjd", np.nan))
        t1 = float(row.get("max_mjd", np.nan))
    except (TypeError, ValueError):
        return float("nan")
    if np.isfinite(t0) and np.isfinite(t1):
        return 0.5 * (t0 + t1)
    if np.isfinite(t1):
        return t1
    if np.isfinite(t0):
        return t0
    return float("nan")


def desi_rows_to_external_rvs(
    rows: Sequence[dict[str, Any]],
    *,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
) -> list[dict[str, Any]]:
    """Convert Data Lab DESI MWS rows to summary [EXTERNAL RV DATA] records (barycentric)."""
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            rv = float(row["rv_adop"])
            if not np.isfinite(rv):
                continue
            err_raw = row.get("rv_err")
            rv_err = float(err_raw) if err_raw not in (None, "", "null") and np.isfinite(float(err_raw)) else 0.0
            mjd = _mjd_from_row(row)
            if not np.isfinite(mjd):
                logger.debug("skip DESI row without MJD: targetid=%s", row.get("targetid"))
                continue
            survey = str(row.get("survey", "") or "")
            program = str(row.get("program", "") or "")
            targetid = str(row.get("targetid", "") or "")
            warn = str(row.get("rvs_warn", "") or "")
            flag = f"{survey}/{program}/{targetid}"
            if warn not in ("", "0", "0.0"):
                flag = f"{flag} warn={warn}"
            cross_sid = parse_gaia_source_id(row.get("source_id"))
            if cross_sid is not None:
                flag = f"{flag} gaia={cross_sid}"
            sep = row.get("sep_deg")
            if sep not in (None, "", "null"):
                try:
                    sep_arcsec = float(sep) * 3600.0
                    if np.isfinite(sep_arcsec):
                        flag = f"{flag} sep={sep_arcsec:.2f}arcsec"
                except (TypeError, ValueError):
                    pass
            rec = finalize_external_rv_record(
                {
                    "telescope": DESI_TELESCOPE_PREFIX,
                    "mjd": float(mjd),
                    "rv": rv,
                    "rv_err": max(rv_err, 1e-4),
                    "flag": flag,
                },
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                native_frame=NativeFrame.HELIOCENTRIC,
                site_key="DESI",
            )
            if rec is not None:
                out.append(rec)
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda r: (r["mjd"], r["flag"]))
    return out


def _query_desi_by_source_ids(
    source_ids: Sequence[int],
    *,
    timeout: float,
) -> list[dict[str, Any]]:
    id_list = ", ".join(str(int(i)) for i in source_ids)
    sql = _DESI_RV_SQL_BY_ID.format(mws=DESI_TABLE, zpix=ZPIX_TABLE, ids=id_list)
    return execute_datalab_adql(sql, timeout=timeout)


def _query_desi_by_cone(
    ra_deg: float,
    dec_deg: float,
    *,
    radius_deg: float = DESI_CONE_RADIUS_DEG,
    timeout: float,
) -> list[dict[str, Any]]:
    # Use a simple RA/Dec box in SQL (Data Lab's parser chokes on sqrt/power/cos and
    # on "dec - -45" which becomes an SQL comment). Refine to a true cone in Python.
    sql = _DESI_RV_SQL_BY_BOX.format(
        mws=DESI_TABLE,
        zpix=ZPIX_TABLE,
        ra=ra_deg,
        dec=dec_deg,
        radius_deg=radius_deg,
    )
    rows = execute_datalab_adql(sql, timeout=timeout)
    return filter_rows_in_cone(rows, ra_deg, dec_deg, radius_deg=radius_deg)


def query_desi_rvs_for_gaia_ids(
    source_ids: Sequence[int],
    *,
    positions_by_id: dict[int, tuple[float, float]] | None = None,
    cone_radius_deg: float = DESI_CONE_RADIUS_DEG,
    chunk_size: int = 40,
    timeout: float = 180.0,
    progress: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    """
    Return DESI external-RV rows keyed by the requested Gaia DR3 source_id.

    Lookup order per star:
    1. ``desi_dr1.mws.source_id`` exact match (DESI's Gaia crossmatch id)
    2. If no rows and RA/Dec are known: cone search on ``target_ra``/``target_dec``
       within ``cone_radius_deg`` (default 1.5 arcsec, same as Data Lab xmatch)
    """
    ids = sorted({int(sid) for sid in source_ids if sid})
    if not ids:
        return {}

    positions = positions_by_id or {}
    rows_by_sid: dict[int, list[dict[str, Any]]] = {sid: [] for sid in ids}

    n_chunks = (len(ids) + chunk_size - 1) // chunk_size
    for chunk_i, start in enumerate(range(0, len(ids), chunk_size), start=1):
        chunk = ids[start : start + chunk_size]
        if progress:
            print(
                f"DESI source_id batch {chunk_i}/{n_chunks} ({len(chunk)} stars)...",
                flush=True,
            )
        try:
            id_rows = _query_desi_by_source_ids(chunk, timeout=timeout)
        except Exception as exc:
            logger.warning("DESI source_id query failed for %d Gaia ids: %s", len(chunk), exc)
            id_rows = []

        id_rows_by_sid: dict[int, list[dict[str, Any]]] = {sid: [] for sid in chunk}
        for row in id_rows:
            sid = parse_gaia_source_id(row.get("source_id"))
            if sid is not None and sid in id_rows_by_sid:
                id_rows_by_sid[sid].append(row)

        cone_sids: list[int] = []
        for sid in chunk:
            merged = _merge_desi_rows(id_rows_by_sid.get(sid, []))
            pos = positions.get(sid)
            ra, dec = (pos[0], pos[1]) if pos else (None, None)
            if merged:
                rows_by_sid[sid] = desi_rows_to_external_rvs(merged, ra_deg=ra, dec_deg=dec)
                continue
            pos = positions.get(sid)
            if pos is None:
                continue
            try:
                ra, dec = float(pos[0]), float(pos[1])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(ra) and np.isfinite(dec):
                cone_sids.append(sid)

        for cone_i, sid in enumerate(cone_sids, start=1):
            pos = positions[sid]
            ra, dec = float(pos[0]), float(pos[1])
            if progress:
                print(
                    f"DESI cone {cone_i}/{len(cone_sids)} Gaia_DR3_{sid} "
                    f"(RA={ra:.4f} Dec={dec:.4f})...",
                    flush=True,
                )
            try:
                cone_rows = _query_desi_by_cone(ra, dec, radius_deg=cone_radius_deg, timeout=timeout)
            except Exception as exc:
                logger.warning("DESI cone query failed for Gaia %s: %s", sid, exc)
                continue
            if cone_rows:
                logger.info(
                    "DESI cone match for Gaia %s at RA=%.6f Dec=%.6f (%d row(s); no source_id match)",
                    sid,
                    ra,
                    dec,
                    len(cone_rows),
                )
                rows_by_sid[sid] = desi_rows_to_external_rvs(
                    cone_rows, ra_deg=ra, dec_deg=dec
                )

    return {sid: rows for sid, rows in rows_by_sid.items() if rows}
