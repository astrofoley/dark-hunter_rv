"""Query Gaia-ESO DR5.1 radial velocities from VizieR TAP."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from astropy.time import Time

from .cone_utils import filter_rows_in_cone
from .rv_frame import NativeFrame, finalize_external_rv_record
from .vizier_tap import execute_vizier_adql, sql_max_err_clause

logger = logging.getLogger(__name__)

GES_TELESCOPE_PREFIX = "GES_DR5"
# Gaia-ESO DR5.1 is not published on VizieR TAP (table III/500/ges_dr5 returns 404).
GES_VIZIER_AVAILABLE = False
GES_TABLE = '"III/500/ges_dr5"'
GES_CONE_RADIUS_ARCSEC = 2.0
GES_CONE_RADIUS_DEG = GES_CONE_RADIUS_ARCSEC / 3600.0

_GES_SQL_BY_BOX = """
SELECT OBJECT AS ges_object, RA AS ra, DECLINATION AS dec,
       VRAD AS rv, E_VRAD AS rv_err, VRAD_FLAG AS vrad_flag, DATE_OBS AS date_obs
FROM {table}
WHERE RA BETWEEN {ra} - {radius_deg} AND {ra} + {radius_deg}
  AND DECLINATION BETWEEN {dec} - {radius_deg} AND {dec} + {radius_deg}
  AND VRAD IS NOT NULL
  AND (VRAD_FLAG IS NULL OR VRAD_FLAG = 0)
{err_clause}"""


def _float_or_nan(val: Any) -> float:
    try:
        if val in (None, "", "null", "nan"):
            return float("nan")
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _mjd_from_date_obs(date_obs: Any) -> float:
    s = str(date_obs or "").strip()
    if not s:
        return float("nan")
    try:
        return float(Time(s, scale="utc").mjd)
    except Exception:
        return float("nan")


def ges_rows_to_external_rvs(
    rows: Sequence[dict[str, Any]],
    *,
    ra_deg: float | None,
    dec_deg: float | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        rv = _float_or_nan(row.get("rv"))
        rv_err = _float_or_nan(row.get("rv_err"))
        mjd = _mjd_from_date_obs(row.get("date_obs"))
        if not np.isfinite(rv) or not np.isfinite(mjd):
            continue
        obj = str(row.get("ges_object", "") or "")
        flag = f"ges={obj}"
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
                "telescope": GES_TELESCOPE_PREFIX,
                "mjd": mjd,
                "rv": rv,
                "rv_err": rv_err if np.isfinite(rv_err) else 0.0,
                "flag": flag,
            },
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            native_frame=NativeFrame.BARYCENTRIC,
            site_key="GES_VLT",
        )
        if rec is not None:
            out.append(rec)
    out.sort(key=lambda r: (r["mjd"], r["flag"]))
    return out


def _query_ges_by_cone(
    ra_deg: float,
    dec_deg: float,
    *,
    max_rv_err: float | None,
    radius_deg: float = GES_CONE_RADIUS_DEG,
    timeout: float,
) -> list[dict[str, Any]]:
    sql = _GES_SQL_BY_BOX.format(
        table=GES_TABLE,
        ra=ra_deg,
        dec=dec_deg,
        radius_deg=radius_deg,
        err_clause=sql_max_err_clause("E_VRAD", max_rv_err),
    )
    rows = execute_vizier_adql(sql, timeout=timeout)
    return filter_rows_in_cone(
        rows, ra_deg, dec_deg, ra_key="ra", dec_key="dec", radius_deg=radius_deg
    )


def query_ges_rvs_for_gaia_ids(
    source_ids: Sequence[int],
    *,
    positions_by_id: dict[int, tuple[float, float]] | None = None,
    max_rv_err: float | None = None,
    cone_radius_deg: float = GES_CONE_RADIUS_DEG,
    timeout: float = 180.0,
    progress: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    """Return GES DR5.1 RVs keyed by Gaia DR3 source_id (RA/Dec cone only)."""
    ids = sorted({int(sid) for sid in source_ids if sid})
    if not ids:
        return {}
    if not GES_VIZIER_AVAILABLE:
        logger.warning(
            "GES skipped: Gaia-ESO DR5 is not available on VizieR TAP "
            "(use the ESO Gaia-ESO archive directly)."
        )
        return {}

    positions = positions_by_id or {}
    rows_by_sid: dict[int, list[dict[str, Any]]] = {}

    for i, sid in enumerate(ids, start=1):
        pos = positions.get(sid)
        if pos is None:
            continue
        try:
            ra, dec = float(pos[0]), float(pos[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (np.isfinite(ra) and np.isfinite(dec)):
            continue
        if progress:
            print(f"GES cone {i}/{len(ids)} Gaia_DR3_{sid}...", flush=True)
        try:
            raw = _query_ges_by_cone(
                ra,
                dec,
                max_rv_err=max_rv_err,
                radius_deg=cone_radius_deg,
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning("GES cone query failed for Gaia %s: %s", sid, exc)
            continue
        ext = ges_rows_to_external_rvs(raw, ra_deg=ra, dec_deg=dec)
        if ext:
            rows_by_sid[sid] = ext

    return rows_by_sid
