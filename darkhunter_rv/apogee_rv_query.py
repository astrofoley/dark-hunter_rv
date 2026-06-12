"""Query APOGEE DR17 visit RVs from VizieR TAP."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .cone_utils import filter_rows_in_cone
from .rv_frame import NativeFrame, finalize_external_rv_record, site_key_for_apogee_telescope
from .vizier_tap import execute_vizier_adql, parse_gaia_source_id, sql_in_ints, sql_max_err_clause

logger = logging.getLogger(__name__)

APOGEE_TELESCOPE_PREFIX = "APOGEE_DR17"
APOGEE_CATALOG_TABLE = '"III/286/catalog"'
APOGEE_ALLVIS_TABLE = '"III/286/allvis"'
APOGEE_CONE_RADIUS_ARCSEC = 1.5
APOGEE_CONE_RADIUS_DEG = APOGEE_CONE_RADIUS_ARCSEC / 3600.0

_APOGEE_SQL_BY_GAIA = """
SELECT c.GaiaEDR3 AS gaia_id, v.Tel AS apo_telescope, v.MJD AS mjd,
       v.VHelio AS rv, v.e_RV AS rv_err, v.APOGEE AS apogee_id
FROM {catalog} AS c
JOIN {allvis} AS v ON c.APOGEE = v.APOGEE
WHERE c.GaiaEDR3 IN ({ids})
  AND v.VHelio IS NOT NULL
  AND v.MJD IS NOT NULL
{err_clause}"""

_APOGEE_SQL_BY_BOX = """
SELECT v.RAJ2000 AS ra, v.DEJ2000 AS dec, v.Tel AS apo_telescope, v.MJD AS mjd,
       v.VHelio AS rv, v.e_RV AS rv_err, v.APOGEE AS apogee_id
FROM {allvis} AS v
WHERE v.RAJ2000 BETWEEN {ra} - {radius_deg} AND {ra} + {radius_deg}
  AND v.DEJ2000 BETWEEN {dec} - {radius_deg} AND {dec} + {radius_deg}
  AND v.VHelio IS NOT NULL
  AND v.MJD IS NOT NULL
{err_clause}"""


def _float_or_nan(val: Any) -> float:
    try:
        if val in (None, "", "null", "nan"):
            return float("nan")
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def apogee_rows_to_external_rvs(
    rows: Sequence[dict[str, Any]],
    *,
    ra_deg: float | None,
    dec_deg: float | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        rv = _float_or_nan(row.get("rv"))
        rv_err = _float_or_nan(row.get("rv_err"))
        mjd = _float_or_nan(row.get("mjd"))
        if not np.isfinite(rv) or not np.isfinite(mjd):
            continue
        apogee_id = str(row.get("apogee_id", "") or "").strip()
        apo_tel = str(row.get("apo_telescope", "") or "").strip()
        site = site_key_for_apogee_telescope(apo_tel)
        flag = f"apogee={apogee_id}"
        if apo_tel:
            flag = f"{flag} tel={apo_tel.strip()}"
        rec = finalize_external_rv_record(
            {
                "telescope": APOGEE_TELESCOPE_PREFIX,
                "mjd": mjd,
                "rv": rv,
                "rv_err": rv_err if np.isfinite(rv_err) else 0.0,
                "flag": flag,
            },
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            native_frame=NativeFrame.BARYCENTRIC,
            site_key=site,
        )
        if rec is not None:
            out.append(rec)
    out.sort(key=lambda r: (r["mjd"], r["flag"]))
    return out


def _query_apogee_by_gaia_ids(
    gaia_ids: Sequence[int],
    *,
    max_rv_err: float | None,
    timeout: float,
) -> list[dict[str, Any]]:
    sql = _APOGEE_SQL_BY_GAIA.format(
        catalog=APOGEE_CATALOG_TABLE,
        allvis=APOGEE_ALLVIS_TABLE,
        ids=sql_in_ints(list(gaia_ids)),
        err_clause=sql_max_err_clause("v.e_RV", max_rv_err),
    )
    return execute_vizier_adql(sql, timeout=timeout)


def _query_apogee_by_cone(
    ra_deg: float,
    dec_deg: float,
    *,
    max_rv_err: float | None,
    radius_deg: float = APOGEE_CONE_RADIUS_DEG,
    timeout: float,
) -> list[dict[str, Any]]:
    sql = _APOGEE_SQL_BY_BOX.format(
        allvis=APOGEE_ALLVIS_TABLE,
        ra=ra_deg,
        dec=dec_deg,
        radius_deg=radius_deg,
        err_clause=sql_max_err_clause("v.e_RV", max_rv_err),
    )
    rows = execute_vizier_adql(sql, timeout=timeout)
    return filter_rows_in_cone(
        rows, ra_deg, dec_deg, ra_key="ra", dec_key="dec", radius_deg=radius_deg
    )


def query_apogee_rvs_for_gaia_ids(
    source_ids: Sequence[int],
    *,
    positions_by_id: dict[int, tuple[float, float]] | None = None,
    max_rv_err: float | None = None,
    cone_radius_deg: float = APOGEE_CONE_RADIUS_DEG,
    chunk_size: int = 150,
    timeout: float = 180.0,
    progress: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    """Return APOGEE DR17 visit RVs keyed by Gaia DR3 source_id."""
    ids = sorted({int(sid) for sid in source_ids if sid})
    if not ids:
        return {}

    positions = positions_by_id or {}
    rows_by_sid: dict[int, list[dict[str, Any]]] = {}

    n_chunks = (len(ids) + chunk_size - 1) // chunk_size
    for chunk_i, start in enumerate(range(0, len(ids), chunk_size), start=1):
        chunk = ids[start : start + chunk_size]
        if progress:
            print(
                f"APOGEE GaiaEDR3 batch {chunk_i}/{n_chunks} ({len(chunk)} stars)...",
                flush=True,
            )
        try:
            id_rows = _query_apogee_by_gaia_ids(chunk, max_rv_err=max_rv_err, timeout=timeout)
        except Exception as exc:
            logger.warning("APOGEE source_id query failed: %s", exc)
            id_rows = []

        by_gaia: dict[int, list[dict[str, Any]]] = {sid: [] for sid in chunk}
        for row in id_rows:
            sid = parse_gaia_source_id(row.get("gaia_id"))
            if sid is not None and sid in by_gaia:
                by_gaia[sid].append(row)

        for sid in chunk:
            merged = by_gaia.get(sid, [])
            if not merged:
                pos = positions.get(sid)
                if pos is not None:
                    ra, dec = float(pos[0]), float(pos[1])
                    if np.isfinite(ra) and np.isfinite(dec):
                        try:
                            merged = _query_apogee_by_cone(
                                ra,
                                dec,
                                max_rv_err=max_rv_err,
                                radius_deg=cone_radius_deg,
                                timeout=timeout,
                            )
                        except Exception as exc:
                            logger.warning("APOGEE cone query failed for Gaia %s: %s", sid, exc)
                            merged = []
            pos = positions.get(sid)
            ra, dec = (pos[0], pos[1]) if pos else (None, None)
            ext = apogee_rows_to_external_rvs(merged, ra_deg=ra, dec_deg=dec)
            if ext:
                rows_by_sid[sid] = ext

    return rows_by_sid
