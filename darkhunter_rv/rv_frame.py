"""Normalize external survey RVs to solar-system barycentric (km/s)."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from .rv_point_filters import mjd_is_valid, rv_value_is_valid

logger = logging.getLogger(__name__)

EXTERNAL_RV_HEADER_COMMENT = (
    "# Telescope | MJD | RV (km/s) | Err (km/s) | Flag/ID  "
    "(RVs are solar-system barycentric)"
)


class NativeFrame(str, Enum):
    HELIOCENTRIC = "heliocentric"
    BARYCENTRIC = "barycentric"


# Observatory geolocations (IAU EarthLocation conventions).
OBSERVATORY_SITES: dict[str, EarthLocation] = {
    "LICK": EarthLocation(lat=37.3413 * u.deg, lon=-121.6438 * u.deg, height=1280 * u.m),
    "GALAH_AAT": EarthLocation(lat=-31.2733 * u.deg, lon=149.0642 * u.deg, height=1164 * u.m),
    "APOGEE_APO": EarthLocation(lat=32.7801 * u.deg, lon=-105.8208 * u.deg, height=2798 * u.m),
    "APOGEE_LCO": EarthLocation(lat=-29.0146 * u.deg, lon=-70.7380 * u.deg, height=2287 * u.m),
    "LAMOST": EarthLocation(lat=40.3956 * u.deg, lon=117.5783 * u.deg, height=960 * u.m),
    "RAVE": EarthLocation(lat=-31.2733 * u.deg, lon=149.0642 * u.deg, height=1164 * u.m),
    "DESI": EarthLocation(lat=31.9634 * u.deg, lon=-111.5981 * u.deg, height=2120 * u.m),
    "GES_VLT": EarthLocation(lat=-24.6275 * u.deg, lon=-70.4042 * u.deg, height=2635 * u.m),
}

NATIVE_FRAME_BY_PREFIX: dict[str, NativeFrame] = {
    "LAMOST_LRS": NativeFrame.HELIOCENTRIC,
    "LAMOST_MRS": NativeFrame.BARYCENTRIC,
    "RAVE_DR6": NativeFrame.HELIOCENTRIC,
    "DESI_DR1": NativeFrame.HELIOCENTRIC,
    "GALAH_DR3": NativeFrame.HELIOCENTRIC,
    "APOGEE_DR17": NativeFrame.BARYCENTRIC,
    "GES_DR5": NativeFrame.BARYCENTRIC,
}

DEFAULT_SITE_BY_PREFIX: dict[str, str] = {
    "LAMOST_LRS": "LAMOST",
    "LAMOST_MRS": "LAMOST",
    "RAVE_DR6": "RAVE",
    "DESI_DR1": "DESI",
    "GALAH_DR3": "GALAH_AAT",
    "APOGEE_DR17": "APOGEE_APO",
    "GES_DR5": "GES_VLT",
}


def site_key_for_apogee_telescope(telescope_tag: str | None) -> str:
    """Map APOGEE ``TELESCOPE`` column to observatory site key."""
    t = str(telescope_tag or "").strip().lower()
    if "lco" in t:
        return "APOGEE_LCO"
    return "APOGEE_APO"


def native_frame_for_telescope(telescope: str) -> NativeFrame:
    tele = str(telescope or "").strip()
    for prefix, frame in NATIVE_FRAME_BY_PREFIX.items():
        if tele.startswith(prefix) or tele == prefix:
            return frame
    return NativeFrame.HELIOCENTRIC


def default_site_key_for_telescope(telescope: str) -> str:
    tele = str(telescope or "").strip()
    for prefix, site in DEFAULT_SITE_BY_PREFIX.items():
        if tele.startswith(prefix) or tele == prefix:
            return site
    return "LICK"


# km/s per (AU/day): AU_km / seconds_per_day
_AU_DAY_TO_KM_S = 149597870.7 / 86400.0


def _los_unit_icrs(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = np.radians(float(ra_deg))
    dec = np.radians(float(dec_deg))
    return np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)],
        dtype=float,
    )


def _helio_bary_orbit_delta_kms(ra_deg: float, dec_deg: float, mjd: float) -> float:
    """
    Orbit-only (barycentric − heliocentric) correction along the line of sight (km/s).

    Diurnal terms largely cancel in this difference at a fixed site; sufficient for
  external-RV cross-checks at ~10 km/s precision.
    """
    import erfa

    t = Time(float(mjd), format="mjd", scale="utc")
    d1, d2 = t.tdb.jd1, t.tdb.jd2
    star = _los_unit_icrs(ra_deg, dec_deg)
    pvh, pvb = erfa.epv00(d1, d2)
    v_bary = -float(np.dot(np.asarray(pvb[1], dtype=float), star)) * _AU_DAY_TO_KM_S
    v_helio = -float(np.dot(np.asarray(pvh[1], dtype=float), star)) * _AU_DAY_TO_KM_S
    return v_bary - v_helio


def helio_to_bary_kms(
    rv_helio_kms: float,
    *,
    ra_deg: float,
    dec_deg: float,
    mjd: float,
    site_key: str,
) -> float:
    """Convert heliocentric RV (km/s) at ``site_key`` to solar-system barycentric."""
    del site_key  # orbit delta is site-independent at our precision
    return float(rv_helio_kms) + _helio_bary_orbit_delta_kms(ra_deg, dec_deg, mjd)


def _append_flag(flag: str, suffix: str) -> str:
    s = str(flag or "").strip()
    if not suffix:
        return s
    if suffix in s:
        return s
    return f"{s} {suffix}".strip() if s else suffix


def normalize_external_rv_to_barycentric(
    rv_kms: float,
    *,
    ra_deg: float | None,
    dec_deg: float | None,
    mjd: float,
    telescope: str,
    native_frame: NativeFrame | None = None,
    site_key: str | None = None,
    flag: str = "",
) -> dict[str, Any] | None:
    """
    Return summary external-RV fields with RV in barycentric km/s, or None if unusable.

    Skips rows without valid MJD, coordinates, or RV value.
    """
    if not mjd_is_valid(mjd):
        return None
    if not rv_value_is_valid(rv_kms):
        return None
    try:
        ra = float(ra_deg)
        dec = float(dec_deg)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return None

    frame = native_frame or native_frame_for_telescope(telescope)
    site = site_key or default_site_key_for_telescope(telescope)
    rv_out = float(rv_kms)
    flag_out = str(flag or "")

    if frame == NativeFrame.HELIOCENTRIC:
        try:
            rv_out = helio_to_bary_kms(rv_out, ra_deg=ra, dec_deg=dec, mjd=float(mjd), site_key=site)
        except Exception as exc:
            logger.debug("helio→bary failed for %s MJD=%s: %s", telescope, mjd, exc)
            return None
        flag_out = _append_flag(flag_out, "conv=helio→bary")
    else:
        flag_out = _append_flag(flag_out, "frame=bary-native")

    if not rv_value_is_valid(rv_out):
        return None
    return {"rv": rv_out, "flag": flag_out}


def finalize_external_rv_record(
    record: dict[str, Any],
    *,
    ra_deg: float | None,
    dec_deg: float | None,
    native_frame: NativeFrame | None = None,
    site_key: str | None = None,
) -> dict[str, Any] | None:
    """Apply barycentric normalization to a partial external-RV dict (telescope/mjd/rv/rv_err/flag)."""
    tele = str(record.get("telescope", "") or "")
    try:
        mjd = float(record["mjd"])
        rv = float(record["rv"])
        rv_err = float(record.get("rv_err", 0.0) or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    norm = normalize_external_rv_to_barycentric(
        rv,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        mjd=mjd,
        telescope=tele,
        native_frame=native_frame,
        site_key=site_key,
        flag=str(record.get("flag", "") or ""),
    )
    if norm is None:
        return None
    return {
        "telescope": tele,
        "mjd": mjd,
        "rv": norm["rv"],
        "rv_err": max(rv_err, 1e-4) if np.isfinite(rv_err) and rv_err > 0 else 1e-4,
        "flag": norm["flag"],
    }


def mjd_from_rave_obs_id(obs_id: str) -> float:
    """
    Parse observation date from RAVE ``rave_obs_id`` (leading YYMMDD before first '_').
    """
    s = str(obs_id or "").strip()
    if not s:
        return float("nan")
    head = s.split("_", 1)[0]
    if len(head) != 6 or not head.isdigit():
        return float("nan")
    yy, mm, dd = int(head[:2]), int(head[2:4]), int(head[4:6])
    year = 2000 + yy if yy < 80 else 1900 + yy
    try:
        return float(Time(f"{year:04d}-{mm:02d}-{dd:02d}", format="iso", scale="utc").mjd)
    except Exception:
        return float("nan")
