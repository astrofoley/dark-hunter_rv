"""Query GALAH DR3 VAC radial velocities from VizieR TAP."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .rv_frame import NativeFrame, finalize_external_rv_record
from .vizier_tap import execute_vizier_adql, parse_gaia_source_id, sql_in_ints, sql_max_err_clause

logger = logging.getLogger(__name__)

GALAH_TELESCOPE_PREFIX = "GALAH_DR3"
GALAH_RV_TABLE = '"J/MNRAS/506/150/rv"'
GALAH_STARS_TABLE = '"J/MNRAS/506/150/stars"'

_GALAH_SQL_BY_GAIA = """
SELECT s.GaiaEDR3 AS gaia_id, r.GALAH AS galah_id,
       r.RVgalah AS rv, r.e_RVgalah AS rv_err, r.MJDlocal AS mjd, r.f_RVgalah AS rv_flag
FROM {rv} AS r
JOIN {stars} AS s ON r.GALAH = s.GALAH
WHERE s.GaiaEDR3 IN ({ids})
  AND r.RVgalah IS NOT NULL
{err_clause}"""


def _float_or_nan(val: Any) -> float:
    try:
        if val in (None, "", "null", "nan"):
            return float("nan")
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def galah_rows_to_external_rvs(
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
        galah_id = str(row.get("galah_id", "") or "")
        rv_flag = str(row.get("rv_flag", "") or "")
        flag = f"galah={galah_id}"
        if rv_flag not in ("", "null"):
            flag = f"{flag} use_rv_flag={rv_flag}"
        rec = finalize_external_rv_record(
            {
                "telescope": GALAH_TELESCOPE_PREFIX,
                "mjd": mjd,
                "rv": rv,
                "rv_err": rv_err if np.isfinite(rv_err) else 0.0,
                "flag": flag,
            },
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            native_frame=NativeFrame.HELIOCENTRIC,
            site_key="GALAH_AAT",
        )
        if rec is not None:
            out.append(rec)
    out.sort(key=lambda r: (r["mjd"], r["flag"]))
    return out


def _query_galah_by_gaia_ids(
    gaia_ids: Sequence[int],
    *,
    max_rv_err: float | None,
    timeout: float,
) -> list[dict[str, Any]]:
    id_list = sql_in_ints(list(gaia_ids))
    sql = _GALAH_SQL_BY_GAIA.format(
        rv=GALAH_RV_TABLE,
        stars=GALAH_STARS_TABLE,
        ids=id_list,
        err_clause=sql_max_err_clause("r.e_RVgalah", max_rv_err),
    )
    return execute_vizier_adql(sql, timeout=timeout)


def query_galah_rvs_for_gaia_ids(
    source_ids: Sequence[int],
    *,
    positions_by_id: dict[int, tuple[float, float]] | None = None,
    max_rv_err: float | None = None,
    chunk_size: int = 200,
    timeout: float = 180.0,
    progress: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    """
    Return GALAH external-RV rows keyed by Gaia DR3 source_id (via GaiaEDR3 in stars table).
    """
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
                f"GALAH GaiaEDR3 batch {chunk_i}/{n_chunks} ({len(chunk)} stars)...",
                flush=True,
            )
        try:
            rows = _query_galah_by_gaia_ids(chunk, max_rv_err=max_rv_err, timeout=timeout)
        except Exception as exc:
            logger.warning("GALAH query failed for %d Gaia ids: %s", len(chunk), exc)
            continue

        by_gaia: dict[int, list[dict[str, Any]]] = {sid: [] for sid in chunk}
        for row in rows:
            sid = parse_gaia_source_id(row.get("gaia_id"))
            if sid is not None and sid in by_gaia:
                by_gaia[sid].append(row)

        for sid in chunk:
            pos = positions.get(sid)
            ra, dec = (pos[0], pos[1]) if pos else (None, None)
            ext = galah_rows_to_external_rvs(by_gaia.get(sid, []), ra_deg=ra, dec_deg=dec)
            if ext:
                rows_by_sid[sid] = ext

    return rows_by_sid
