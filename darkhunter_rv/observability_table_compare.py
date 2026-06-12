"""Compare our Lick visibility math against a planning-table reference (Gaia 4491)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from .apf_observability import LICK_LOCATION, TWILIGHT_SUN_ALT_DEG, _airmass_from_altitude_deg, _sun_alt_deg

# User table for Gaia DR3 449121951305471360 (correct ICRS RA/Dec from Gaia DR3).
GAIA_4491 = SkyCoord(ra=55.52045150805677, dec=57.90254409416177, unit="deg", frame="icrs")

# Reference rows transcribed from the corrected planning-table screenshot (2026).
REFERENCE_TABLE: Dict[str, Dict[str, float]] = {
    "2026-03-18": {"airm_eve": 1.32, "airm_ctr": 3.45, "airm_morn": 9.06, "hrs3": 4.0, "hrs2": 2.5, "hrs15": 0.9},
    "2026-04-01": {"airm_eve": 1.56, "airm_ctr": 4.57, "airm_morn": 8.07, "hrs3": 2.9, "hrs2": 1.3, "hrs15": 0.0},
    "2026-04-16": {"airm_eve": 1.98, "airm_ctr": 6.28, "airm_morn": 6.83, "hrs3": 1.6, "hrs2": 0.0, "hrs15": 0.0},
    "2026-04-30": {"airm_eve": 2.67, "airm_ctr": 8.20, "airm_morn": 5.68, "hrs3": 0.4, "hrs2": 0.0, "hrs15": 0.0},
    "2026-05-16": {"airm_eve": 4.10, "airm_ctr": 9.60, "airm_morn": 4.48, "hrs3": 0.0, "hrs2": 0.0, "hrs15": 0.0},
    "2026-05-30": {"airm_eve": 6.19, "airm_ctr": 8.88, "airm_morn": 3.56, "hrs3": 0.0, "hrs2": 0.0, "hrs15": 0.0},
    "2026-06-14": {"airm_eve": 8.70, "airm_ctr": 6.72, "airm_morn": 2.73, "hrs3": 0.3, "hrs2": 0.0, "hrs15": 0.0},
    "2026-06-29": {"airm_eve": 9.61, "airm_ctr": 4.70, "airm_morn": 2.09, "hrs3": 1.4, "hrs2": 0.0, "hrs15": 0.0},
    "2026-07-13": {"airm_eve": 8.70, "airm_ctr": 3.42, "airm_morn": 1.66, "hrs3": 2.5, "hrs2": 0.9, "hrs15": 0.0},
    "2026-07-28": {"airm_eve": 7.19, "airm_ctr": 2.56, "airm_morn": 1.37, "hrs3": 3.8, "hrs2": 2.2, "hrs15": 0.6},
    "2026-08-11": {"airm_eve": 5.94, "airm_ctr": 2.05, "airm_morn": 1.20, "hrs3": 5.0, "hrs2": 3.4, "hrs15": 1.9},
    "2026-08-27": {"airm_eve": 4.80, "airm_ctr": 1.68, "airm_morn": 1.10, "hrs3": 6.3, "hrs2": 4.8, "hrs15": 3.2},
    "2026-09-10": {"airm_eve": 4.02, "airm_ctr": 1.46, "airm_morn": 1.07, "hrs3": 7.5, "hrs2": 5.9, "hrs15": 4.4},
    "2026-09-25": {"airm_eve": 3.35, "airm_ctr": 1.30, "airm_morn": 1.09, "hrs3": 8.7, "hrs2": 7.2, "hrs15": 5.6},
    "2026-10-09": {"airm_eve": 2.83, "airm_ctr": 1.20, "airm_morn": 1.15, "hrs3": 9.7, "hrs2": 8.3, "hrs15": 6.7},
    "2026-10-25": {"airm_eve": 2.34, "airm_ctr": 1.12, "airm_morn": 1.29, "hrs3": 10.2, "hrs2": 9.6, "hrs15": 8.0},
}


def _hour_angle_deg(time: Time, coord: SkyCoord) -> float:
    lst = time.sidereal_time("apparent", LICK_LOCATION).deg
    return float((lst - coord.ra.deg + 180.0) % 360.0 - 180.0)


def _hour_angle_midpoint_deg(ha_eve_deg: float, ha_morn_deg: float) -> float:
    h_eve = ha_eve_deg / 15.0
    h_morn = ha_morn_deg / 15.0
    delta_h = (h_morn - h_eve + 12.0) % 24.0 - 12.0
    return float((h_eve + delta_h / 2.0) * 15.0)


def _refine_twilight_mjd(
    mjd_guess: float,
    *,
    evening: bool,
    span_days: float = 0.02,
) -> float:
    lo = mjd_guess - span_days
    hi = mjd_guess + span_days
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        sun_alt = _sun_alt_deg(Time(mid, format="mjd", scale="utc"), LICK_LOCATION)
        if evening:
            if sun_alt > TWILIGHT_SUN_ALT_DEG:
                lo = mid
            else:
                hi = mid
        elif sun_alt < TWILIGHT_SUN_ALT_DEG:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _find_evening_twilight_mjd(evening_date: str) -> Optional[float]:
    """Evening nautical twilight on the UTC calendar date *evening_date*."""
    t0 = Time(evening_date, scale="utc").mjd
    prev_sun: Optional[float] = None
    crossing: Optional[float] = None
    for mjd in np.linspace(t0 - 0.5, t0 + 0.5, 2000):
        sun_alt = _sun_alt_deg(Time(float(mjd), format="mjd", scale="utc"), LICK_LOCATION)
        if prev_sun is not None and prev_sun > TWILIGHT_SUN_ALT_DEG >= sun_alt:
            crossing = float(mjd)
        prev_sun = sun_alt
    if crossing is None:
        return None
    return _refine_twilight_mjd(crossing, evening=True)


def _find_morning_twilight_mjd(evening_mjd: float) -> Optional[float]:
    prev_sun: Optional[float] = None
    crossing: Optional[float] = None
    for mjd in np.linspace(evening_mjd + 0.1, evening_mjd + 1.2, 2000):
        sun_alt = _sun_alt_deg(Time(float(mjd), format="mjd", scale="utc"), LICK_LOCATION)
        if prev_sun is not None and prev_sun < TWILIGHT_SUN_ALT_DEG <= sun_alt:
            crossing = float(mjd)
            break
        prev_sun = sun_alt
    if crossing is None:
        return None
    return _refine_twilight_mjd(crossing, evening=False)


@dataclass
class NightSnapshot:
    evening_date: str
    airm_eve: float
    airm_morn: float
    airm_ctr: float
    hrs_below: Dict[float, float]


def night_snapshot_for_evening_date(
    coord: SkyCoord,
    evening_date: str,
    *,
    step_minutes: int = 5,
) -> Optional[NightSnapshot]:
    """
    Nautical-night model aligned with the Lick planning table:

    - Evening / morning anchors: sun at TWILIGHT_SUN_ALT_DEG (-12°).
    - Evening date: UTC calendar date of evening twilight.
    - Center airmass: target at the hour-angle midpoint between eve and morn.
    - Hours: integrate while sun <= -12°, target above horizon, airmass below limit.
    """
    eve_mjd = _find_evening_twilight_mjd(evening_date)
    if eve_mjd is None:
        return None
    morn_mjd = _find_morning_twilight_mjd(eve_mjd)
    if morn_mjd is None or morn_mjd <= eve_mjd:
        return None

    eve_t = Time(eve_mjd, format="mjd", scale="utc")
    morn_t = Time(morn_mjd, format="mjd", scale="utc")
    eve_star = coord.transform_to(AltAz(obstime=eve_t, location=LICK_LOCATION))
    morn_star = coord.transform_to(AltAz(obstime=morn_t, location=LICK_LOCATION))
    airm_eve = _airmass_from_altitude_deg(float(eve_star.alt.deg))
    airm_morn = _airmass_from_altitude_deg(float(morn_star.alt.deg))

    ha_mid = _hour_angle_midpoint_deg(_hour_angle_deg(eve_t, coord), _hour_angle_deg(morn_t, coord))
    ctr_mjd = eve_mjd
    best_err = 1e9
    for mjd in np.linspace(eve_mjd, morn_mjd, 3000):
        t = Time(float(mjd), format="mjd", scale="utc")
        err = abs((_hour_angle_deg(t, coord) - ha_mid + 180.0) % 360.0 - 180.0)
        if err < best_err:
            best_err = err
            ctr_mjd = float(mjd)
    ctr_star = coord.transform_to(AltAz(obstime=Time(ctr_mjd, format="mjd"), location=LICK_LOCATION))
    airm_ctr = _airmass_from_altitude_deg(float(ctr_star.alt.deg))

    step_days = step_minutes / (24.0 * 60.0)
    hrs_below = {3.0: 0.0, 2.0: 0.0, 1.5: 0.0}
    for mjd in np.arange(eve_mjd, morn_mjd, step_days):
        t = Time(float(mjd), format="mjd", scale="utc")
        if _sun_alt_deg(t, LICK_LOCATION) > TWILIGHT_SUN_ALT_DEG:
            continue
        star = coord.transform_to(AltAz(obstime=t, location=LICK_LOCATION))
        alt = float(star.alt.deg)
        if alt <= 0:
            continue
        am = _airmass_from_altitude_deg(alt)
        for lim in hrs_below:
            if am <= lim:
                hrs_below[lim] += step_minutes / 60.0

    return NightSnapshot(
        evening_date=evening_date,
        airm_eve=airm_eve,
        airm_morn=airm_morn,
        airm_ctr=airm_ctr,
        hrs_below=hrs_below,
    )


def compare_reference_table(
    coord: SkyCoord = GAIA_4491,
    *,
    airm_tol: float = 0.5,
    hrs_tol: float = 1.2,
) -> List[str]:
    """Return human-readable diff lines vs REFERENCE_TABLE."""
    lines: List[str] = []
    for date in REFERENCE_TABLE:
        ref = REFERENCE_TABLE[date]
        snap = night_snapshot_for_evening_date(coord, date)
        if snap is None:
            lines.append(f"{date}: could not pair twilights")
            continue
        parts = [f"{date}:"]
        for key, ref_val in ref.items():
            if key == "airm_eve":
                got = snap.airm_eve
            elif key == "airm_morn":
                got = snap.airm_morn
            elif key == "airm_ctr":
                got = snap.airm_ctr
            elif key == "hrs3":
                got = snap.hrs_below[3.0]
            elif key == "hrs2":
                got = snap.hrs_below[2.0]
            elif key == "hrs15":
                got = snap.hrs_below[1.5]
            else:
                continue
            tol = hrs_tol if key.startswith("hrs") else airm_tol
            flag = "OK" if abs(got - ref_val) <= tol else "DIFF"
            parts.append(f"{key} ref={ref_val} got={got:.2f} [{flag}]")
        lines.append("  ".join(parts))
    return lines
