"""APF observability windows from Lick Observatory site constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary
from darkhunter_rv.summary_paths import parse_object_id_from_summary

# APF at Lick Observatory (approximate IERS values).
LICK_LOCATION = EarthLocation(lat=37.3413 * u.deg, lon=-121.6438 * u.deg, height=1280 * u.m)

TWILIGHT_SUN_ALT_DEG = -18.0
MAX_AIRMASS = 2.5
MIN_OBS_MINUTES = 15
STEP_MINUTES = 5
HORIZON_DAYS = 90


def _airmass_from_altitude_deg(alt_deg: float) -> float:
    if not np.isfinite(alt_deg) or alt_deg <= 0:
        return float("inf")
    z_rad = np.deg2rad(90.0 - float(alt_deg))
    return float(1.0 / np.cos(z_rad))


def _target_coord_from_summary(summary_path: Path) -> Optional[SkyCoord]:
    meta = parse_gaia_metadata_from_star_summary(summary_path)
    if not meta:
        return None
    ra = meta.get("RA") or meta.get("ra")
    dec = meta.get("Dec") or meta.get("dec") or meta.get("DEC")
    try:
        ra_f = float(ra)
        dec_f = float(dec)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(ra_f) or not np.isfinite(dec_f):
        return None
    return SkyCoord(ra=ra_f * u.deg, dec=dec_f * u.deg, frame="icrs")


def _sun_alt_deg(time: Time, location: EarthLocation) -> float:
    """Sun altitude (deg) at Lick; ERFA-based to avoid ephemeris download."""
    import erfa

    jd = time.utc.jd
    d1 = np.floor(jd)
    d2 = jd - d1
    earth_pv, earth_heliocentric = erfa.epv00(d1, d2)
    del earth_pv
    if hasattr(earth_heliocentric, "dtype") and earth_heliocentric.dtype.names:
        eh = np.asarray(earth_heliocentric["p"], dtype=float)
    else:
        eh = np.asarray(earth_heliocentric, dtype=float).reshape(-1)[:3]
    sun_vec = -eh
    sun_dist = float(np.sqrt(np.sum(np.square(sun_vec))))
    if sun_dist <= 0:
        return -90.0
    sun_vec /= sun_dist
    ra_deg = float(np.rad2deg(np.arctan2(sun_vec[1], sun_vec[0]))) % 360.0
    dec_deg = float(np.rad2deg(np.arcsin(np.clip(sun_vec[2], -1.0, 1.0))))
    sun = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    sun_aa = sun.transform_to(AltAz(obstime=time, location=location))
    return float(sun_aa.alt.deg)


def _is_circumpolar_at_lick(coord: SkyCoord) -> bool:
    """Northern circumpolar at Lick: dec > 90 - lat."""
    return float(coord.dec.deg) > (90.0 - float(LICK_LOCATION.lat.deg) - 0.05)


@dataclass
class ObsWindow:
    start_mjd: float
    end_mjd: float
    start_date: str
    end_date: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_mjd": float(self.start_mjd),
            "end_mjd": float(self.end_mjd),
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


def _mjd_to_iso_date(mjd: float) -> str:
    return Time(float(mjd), format="mjd", scale="utc").datetime.strftime("%Y-%m-%d")


def _merge_windows(windows: List[ObsWindow], *, gap_days: float = 1.5) -> List[ObsWindow]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: w.start_mjd)
    merged: List[ObsWindow] = [ordered[0]]
    for w in ordered[1:]:
        prev = merged[-1]
        if w.start_mjd - prev.end_mjd <= gap_days:
            merged[-1] = ObsWindow(
                start_mjd=prev.start_mjd,
                end_mjd=max(prev.end_mjd, w.end_mjd),
                start_date=prev.start_date,
                end_date=w.end_date,
            )
        else:
            merged.append(w)
    return merged


def compute_apf_observability(
    coord: SkyCoord,
    *,
    start_mjd: Optional[float] = None,
    horizon_days: int = HORIZON_DAYS,
) -> Dict[str, Any]:
    """
    Find APF-visible intervals within horizon_days of start_mjd.

    Observable when target airmass <= 2.5 for >= 15 minutes after astronomical twilight (-18°).
    """
    now_mjd = float(Time.now().mjd) if start_mjd is None else float(start_mjd)
    end_mjd = now_mjd + float(horizon_days)

    if _is_circumpolar_at_lick(coord):
        start_date = _mjd_to_iso_date(now_mjd)
        end_date = _mjd_to_iso_date(end_mjd)
        win = ObsWindow(now_mjd, end_mjd, start_date, end_date)
        return {
            "circumpolar": True,
            "windows": [win.to_dict()],
            "next_window_start_date": start_date,
            "next_window_end_date": end_date,
        }

    step_days = STEP_MINUTES / (24.0 * 60.0)
    min_steps = max(1, int(round(MIN_OBS_MINUTES / STEP_MINUTES)))
    windows: List[ObsWindow] = []
    t = now_mjd
    run_start: Optional[float] = None
    run_end: Optional[float] = None
    streak = 0
    night_active = False

    while t <= end_mjd + step_days:
        time = Time(t, format="mjd", scale="utc")
        sun_alt = _sun_alt_deg(time, LICK_LOCATION)
        in_night = sun_alt <= TWILIGHT_SUN_ALT_DEG

        observable = False
        if in_night:
            target_aa = coord.transform_to(AltAz(obstime=time, location=LICK_LOCATION))
            am = _airmass_from_altitude_deg(float(target_aa.alt.deg))
            observable = np.isfinite(am) and am <= MAX_AIRMASS

        if in_night and observable:
            if run_start is None:
                run_start = t
            streak += 1
            run_end = t
            night_active = True
        else:
            if night_active and run_start is not None and run_end is not None and streak >= min_steps:
                windows.append(
                    ObsWindow(
                        start_mjd=run_start,
                        end_mjd=run_end + step_days,
                        start_date=_mjd_to_iso_date(run_start),
                        end_date=_mjd_to_iso_date(run_end),
                    )
                )
            run_start = None
            run_end = None
            streak = 0
            night_active = False
        t += step_days

    merged = _merge_windows(windows)
    if not merged:
        return {"circumpolar": False, "windows": []}

    first = merged[0]
    last = merged[-1]
    return {
        "circumpolar": False,
        "windows": [w.to_dict() for w in merged],
        "next_window_start_date": first.start_date,
        "next_window_end_date": last.end_date,
    }


def observability_for_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
    sid = parse_object_id_from_summary(summary_path)
    coord = _target_coord_from_summary(summary_path)
    if sid is None or coord is None:
        return None
    result = compute_apf_observability(coord)
    if not result.get("windows"):
        return None
    result["gaia_source_id"] = sid
    return result
