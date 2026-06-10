"""APF observability windows from Lick Observatory site constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary
from darkhunter_rv.summary_paths import parse_object_id_from_summary

# APF at Lick Observatory.
LICK_LOCATION = EarthLocation(lat=37.3413 * u.deg, lon=-121.6438 * u.deg, height=1280 * u.m)

TWILIGHT_SUN_ALT_DEG = -12.0
TWILIGHT_BUFFER_MINUTES = 30
MAX_AIRMASS = 1.7
MIN_OBS_MINUTES = 30
STEP_MINUTES = 5
HORIZON_DAYS = 183  # ~6 months
PLOT_HORIZON_DAYS = HORIZON_DAYS


def _altitude_deg_for_airmass(airmass: float) -> float:
    if airmass <= 1.0:
        return 90.0
    z_rad = float(np.arccos(np.clip(1.0 / airmass, -1.0, 1.0)))
    return float(90.0 - np.rad2deg(z_rad))


ALTITUDE_MIN_DEG = _altitude_deg_for_airmass(MAX_AIRMASS)


def _airmass_from_altitude_deg(alt_deg: float) -> float:
    if not np.isfinite(alt_deg) or alt_deg <= 0:
        return float("inf")
    z_rad = np.deg2rad(90.0 - float(alt_deg))
    cos_z = float(np.cos(z_rad))
    if cos_z <= 0:
        return float("inf")
    return 1.0 / cos_z


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


@dataclass
class NightRecord:
    """One local night at Lick between 12° twilights."""

    evening_twilight_mjd: float
    morning_twilight_mjd: float
    calendar_date: str
    strict_ok: bool
    up_during_night: bool


def _mjd_to_iso_date(mjd: float) -> str:
    return Time(float(mjd), format="mjd", scale="utc").datetime.strftime("%Y-%m-%d")


def _sun_alts_deg(mjds: np.ndarray) -> np.ndarray:
    """Vectorized sun altitude (deg) at Lick via ERFA."""
    import erfa

    out = np.empty(len(mjds), dtype=float)
    for i, mjd in enumerate(mjds):
        out[i] = _sun_alt_deg(Time(float(mjd), format="mjd", scale="utc"), LICK_LOCATION)
    return out


def _target_alts_deg(coord: SkyCoord, mjds: np.ndarray) -> np.ndarray:
    times = Time(mjds, format="mjd", scale="utc")
    aa = coord.transform_to(AltAz(obstime=times, location=LICK_LOCATION))
    return np.asarray(aa.alt.deg, dtype=float)


def _sample_nights(
    coord: SkyCoord,
    start_mjd: float,
    end_mjd: float,
) -> List[NightRecord]:
    """Walk 5-min steps, segment into nautical-twilight nights, evaluate each."""
    step_days = STEP_MINUTES / (24.0 * 60.0)
    min_steps = max(1, int(round(MIN_OBS_MINUTES / STEP_MINUTES)))
    buffer_days = TWILIGHT_BUFFER_MINUTES / (24.0 * 60.0)

    mjds = np.arange(start_mjd, end_mjd + step_days, step_days, dtype=float)
    if mjds.size == 0:
        return []

    chunk = 4000
    sun_alts = np.concatenate(
        [_sun_alts_deg(mjds[i : i + chunk]) for i in range(0, len(mjds), chunk)]
    )
    target_alts = np.concatenate(
        [_target_alts_deg(coord, mjds[i : i + chunk]) for i in range(0, len(mjds), chunk)]
    )

    nights: List[NightRecord] = []
    in_night = False
    evening_mjd: Optional[float] = None
    strict_streak = 0
    strict_ok = False
    up_during_night = False

    def _close_night(morn_mjd: float) -> None:
        nonlocal strict_ok, up_during_night, strict_streak
        if evening_mjd is None:
            return
        nights.append(
            NightRecord(
                evening_twilight_mjd=evening_mjd,
                morning_twilight_mjd=morn_mjd,
                calendar_date=_mjd_to_iso_date(evening_mjd),
                strict_ok=strict_ok,
                up_during_night=up_during_night,
            )
        )
        strict_ok = False
        up_during_night = False
        strict_streak = 0

    for i in range(len(mjds)):
        t = float(mjds[i])
        sun_alt = float(sun_alts[i])
        target_alt = float(target_alts[i])
        prev_sun = float(sun_alts[i - 1]) if i > 0 else sun_alt + 1.0
        am = _airmass_from_altitude_deg(target_alt)

        if not in_night and prev_sun > TWILIGHT_SUN_ALT_DEG >= sun_alt:
            in_night = True
            evening_mjd = t
            strict_ok = False
            up_during_night = False
            strict_streak = 0

        if in_night:
            if sun_alt < 0.0 and target_alt > 0.0:
                up_during_night = True

            in_extended = evening_mjd is not None and t >= evening_mjd - buffer_days
            if in_extended and np.isfinite(am) and am <= MAX_AIRMASS:
                strict_streak += 1
                if strict_streak >= min_steps:
                    strict_ok = True
            else:
                strict_streak = 0

            if prev_sun <= TWILIGHT_SUN_ALT_DEG < sun_alt:
                _close_night(t)
                in_night = False
                evening_mjd = None

    return nights


def _is_circumpolar_for_apf(
    coord: SkyCoord,
    start_mjd: float,
    end_mjd: float,
) -> bool:
    """
    True only when the target stays above the airmass-1.7 limit whenever the
    sun is below 12° twilight (observable every night at APF elevations).
    """
    step_days = 15.0 / (24.0 * 60.0)  # coarser grid for classification only
    mjds = np.arange(start_mjd, end_mjd + step_days, step_days, dtype=float)
    if mjds.size == 0:
        return False
    sun_alts = _sun_alts_deg(mjds)
    night_mask = sun_alts <= TWILIGHT_SUN_ALT_DEG
    if not np.any(night_mask):
        return False
    target_alts = _target_alts_deg(coord, mjds[night_mask])
    min_alt = float(np.nanmin(target_alts))
    return min_alt >= ALTITUDE_MIN_DEG - 0.5


def _find_season_window(
    nights: List[NightRecord],
    today_mjd: float,
) -> Optional[Tuple[str, str, float, float]]:
    """
    Return (start_date, end_date, start_mjd, end_mjd) for the current or next season.

    A season is a contiguous block of nights when the target is up during the night.
    The first night of a block must pass the strict twilight/airmass test; interior
    nights only need to be above the horizon at some point during nautical night.
    """
    if not nights:
        return None

    runs: List[List[NightRecord]] = []
    current: List[NightRecord] = []

    for night in nights:
        if not current:
            if night.strict_ok:
                current = [night]
        elif night.up_during_night:
            current.append(night)
        else:
            runs.append(current)
            current = []
            if night.strict_ok:
                current = [night]

    if current:
        runs.append(current)

    if not runs:
        return None

    def _run_span(run: List[NightRecord]) -> Tuple[str, str, float, float]:
        start = run[0]
        end = run[-1]
        return (
            start.calendar_date,
            end.calendar_date,
            start.evening_twilight_mjd,
            end.morning_twilight_mjd,
        )

    # Prefer season containing today; else next season; else last season.
    for run in runs:
        start_mjd, end_mjd = run[0].evening_twilight_mjd, run[-1].morning_twilight_mjd
        if start_mjd <= today_mjd <= end_mjd + 1.0:
            return _run_span(run)

    for run in runs:
        if run[0].evening_twilight_mjd >= today_mjd - 1.0:
            return _run_span(run)

    return _run_span(runs[-1])


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


def compute_apf_observability(
    coord: SkyCoord,
    *,
    start_mjd: Optional[float] = None,
    horizon_days: int = HORIZON_DAYS,
) -> Dict[str, Any]:
    """
    APF visibility season at Lick.

    - Nautical twilight (-12°); target airmass ≤ 1.7 for ≥30 min within ±30 min of twilight
      defines season entry (rising near sunrise) and quality on boundary nights.
    - Between season start and end, any night the target is above the horizon counts.
    - One window: next/current season start → when the target sets out of the window.
    - Circumpolar: target stays above airmass 1.7 all nautical nights (very high dec).
    """
    now_mjd = float(Time.now().mjd) if start_mjd is None else float(start_mjd)
    plot_end_mjd = now_mjd + float(min(horizon_days, PLOT_HORIZON_DAYS))
    scan_end_mjd = plot_end_mjd + 1.0

    if _is_circumpolar_for_apf(coord, now_mjd, scan_end_mjd):
        win = ObsWindow(now_mjd, plot_end_mjd, "", "")
        return {
            "circumpolar": True,
            "windows": [win.to_dict()],
            "next_window_start_date": "",
            "next_window_end_date": "",
        }

    nights = _sample_nights(coord, now_mjd - 2.0, scan_end_mjd)
    season = _find_season_window(nights, now_mjd)

    if season is None:
        return {"circumpolar": False, "windows": []}

    start_date, end_date, start_mjd_win, end_mjd_win = season
    win = ObsWindow(start_mjd_win, end_mjd_win, start_date, end_date)
    return {
        "circumpolar": False,
        "windows": [win.to_dict()],
        "next_window_start_date": start_date,
        "next_window_end_date": end_date,
    }


def observability_for_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
    sid = parse_object_id_from_summary(summary_path)
    coord = _target_coord_from_summary(summary_path)
    if sid is None or coord is None:
        return None
    result = compute_apf_observability(coord)
    if not result.get("windows") and not result.get("circumpolar"):
        return None
    result["gaia_source_id"] = sid
    return result
