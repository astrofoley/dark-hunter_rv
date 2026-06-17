"""APF observability windows from Lick Observatory site constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary
from darkhunter_rv.lick_twilight_cache import (
    LICK_TZ,
    default_cache_path as default_lick_twilight_cache_path,
    interpolate_lst_deg,
    nights_in_mjd_range,
)
from darkhunter_rv.summary_paths import parse_object_id_from_summary

# APF at Lick Observatory.
LICK_LOCATION = EarthLocation(lat=37.3413 * u.deg, lon=-121.6438 * u.deg, height=1280 * u.m)
LICK_LAT_DEG = float(LICK_LOCATION.lat.deg)

TWILIGHT_SUN_ALT_DEG = -12.0
MAX_AIRMASS = 1.7
MIN_OBS_MINUTES = 30
NIGHT_SAMPLE_MINUTES = 30
SCAN_HORIZON_DAYS = 365
PLOT_HORIZON_DAYS = 90
HORIZON_DAYS = SCAN_HORIZON_DAYS
MERGE_RUN_GAP_DAYS = 45


def reference_now(reference_mjd: Optional[float] = None) -> Tuple[float, str, float]:
    """Return (now_mjd, Lick-local calendar date, MJD at Pacific midnight of that date)."""
    mjd = float(Time.now().mjd) if reference_mjd is None else float(reference_mjd)
    today_iso = Time(mjd, format="mjd").to_datetime(timezone=LICK_TZ).date().isoformat()
    today_start = datetime.strptime(today_iso, "%Y-%m-%d").replace(tzinfo=LICK_TZ)
    today_start_mjd = float(Time(today_start).mjd)
    return mjd, today_iso, today_start_mjd


def _lick_date_from_mjd(mjd: float) -> str:
    return Time(float(mjd), format="mjd").to_datetime(timezone=LICK_TZ).date().isoformat()


def lick_twilight_cache_for_observability(
    observability_cache_path: Optional[Path] = None,
) -> Path:
    """Resolve Lick twilight JSON beside observability cache, else repo default."""
    if observability_cache_path is not None:
        sibling = observability_cache_path.parent / "lick_twilight_cache.json"
        if sibling.is_file():
            return sibling
    return default_lick_twilight_cache_path()


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


def _target_observable(alt_deg: float, airmass: float) -> bool:
    return (
        np.isfinite(alt_deg)
        and alt_deg > 0.0
        and np.isfinite(airmass)
        and airmass <= MAX_AIRMASS
    )


def _target_altitude_deg_ha(ha_deg: float, dec_deg: float, lat_deg: float = LICK_LAT_DEG) -> float:
    ha_r = math.radians(float(ha_deg))
    dec_r = math.radians(float(dec_deg))
    lat_r = math.radians(float(lat_deg))
    sin_alt = math.sin(dec_r) * math.sin(lat_r) + math.cos(dec_r) * math.cos(lat_r) * math.cos(ha_r)
    return float(math.degrees(math.asin(np.clip(sin_alt, -1.0, 1.0))))


def _hour_angle_deg(lst_deg: float, ra_deg: float) -> float:
    return float((float(lst_deg) - float(ra_deg) + 180.0) % 360.0 - 180.0)


def _target_altitude_at_lst(lst_deg: float, ra_deg: float, dec_deg: float) -> float:
    return _target_altitude_deg_ha(_hour_angle_deg(lst_deg, ra_deg), dec_deg)


def _geometric_min_altitude_deg(coord: SkyCoord) -> float:
    return _target_altitude_deg_ha(180.0, float(coord.dec.deg), LICK_LAT_DEG)


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


def _hour_angle_midpoint_deg(ha_eve_deg: float, ha_morn_deg: float) -> float:
    h_eve = ha_eve_deg / 15.0
    h_morn = ha_morn_deg / 15.0
    delta_h = (h_morn - h_eve + 12.0) % 24.0 - 12.0
    return float((h_eve + delta_h / 2.0) * 15.0)


@dataclass
class NightRecord:
    evening_twilight_mjd: float
    morning_twilight_mjd: float
    calendar_date: str
    observable_night: bool


def twilight_nights_between(
    start_mjd: float,
    end_mjd: float,
    *,
    lick_cache_path: Optional[Path] = None,
) -> List[Tuple[float, float, str]]:
    """Nautical-night anchors from Lick calendar cache (eve MJD, morn MJD, evening date)."""
    rows = nights_in_mjd_range(start_mjd, end_mjd, cache_path=lick_cache_path or default_lick_twilight_cache_path())
    return [
        (float(r["eve_mjd"]), float(r["morn_mjd"]), str(r["evening_date"]))
        for r in rows
    ]


def _night_is_observable(
    ra_deg: float,
    dec_deg: float,
    row: Dict[str, Any],
) -> bool:
    """≥ MIN_OBS_MINUTES at airmass ≤ MAX_AIRMASS during nautical night (LST/HA geometry)."""
    eve_mjd = float(row["eve_mjd"])
    morn_mjd = float(row["morn_mjd"])
    if morn_mjd <= eve_mjd:
        return False
    step_days = NIGHT_SAMPLE_MINUTES / (24.0 * 60.0)
    min_steps = max(1, int(round(MIN_OBS_MINUTES / NIGHT_SAMPLE_MINUTES)))
    streak = 0
    mjd = eve_mjd
    while mjd <= morn_mjd + 0.5 * step_days:
        lst = interpolate_lst_deg(row, mjd)
        alt = _target_altitude_at_lst(lst, ra_deg, dec_deg)
        if _target_observable(alt, _airmass_from_altitude_deg(alt)):
            streak += 1
            if streak >= min_steps:
                return True
        else:
            streak = 0
        mjd += step_days
    return False


def _sample_nights(
    coord: SkyCoord,
    start_mjd: float,
    end_mjd: float,
    *,
    lick_cache_path: Optional[Path] = None,
) -> List[NightRecord]:
    ra_deg = float(coord.ra.deg)
    dec_deg = float(coord.dec.deg)
    cache = lick_cache_path or default_lick_twilight_cache_path()
    rows = nights_in_mjd_range(start_mjd, end_mjd + 1.0, cache_path=cache)
    if not rows:
        raise FileNotFoundError(
            f"No Lick twilight nights in cache for MJD {start_mjd:.0f}–{end_mjd:.0f}. "
            f"Run: python scripts/build_lick_twilight_cache.py --cache {cache}"
        )
    nights: List[NightRecord] = []
    for row in rows:
        eve_mjd = float(row["eve_mjd"])
        if eve_mjd < start_mjd - 0.5 or eve_mjd > end_mjd + 1.0:
            continue
        nights.append(
            NightRecord(
                evening_twilight_mjd=eve_mjd,
                morning_twilight_mjd=float(row["morn_mjd"]),
                calendar_date=str(row["evening_date"]),
                observable_night=_night_is_observable(ra_deg, dec_deg, row),
            )
        )
    return nights


def _is_circumpolar_for_apf(
    coord: SkyCoord,
    start_mjd: float,
    end_mjd: float,
) -> bool:
    del start_mjd, end_mjd
    return _geometric_min_altitude_deg(coord) >= ALTITUDE_MIN_DEG - 0.5


def _build_season_runs(nights: List[NightRecord]) -> List[List[NightRecord]]:
    runs: List[List[NightRecord]] = []
    current: List[NightRecord] = []
    for night in nights:
        if night.observable_night:
            current.append(night)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _merge_close_runs(
    runs: List[List[NightRecord]],
    *,
    gap_days: float = MERGE_RUN_GAP_DAYS,
) -> List[List[NightRecord]]:
    if not runs:
        return []
    merged: List[List[NightRecord]] = [list(runs[0])]
    for run in runs[1:]:
        gap = run[0].evening_twilight_mjd - merged[-1][-1].morning_twilight_mjd
        if gap <= gap_days:
            merged[-1].extend(run)
        else:
            merged.append(list(run))
    return merged


def _run_span(run: List[NightRecord]) -> Tuple[str, str, float, float]:
    start = run[0]
    end = run[-1]
    return (
        start.calendar_date,
        _lick_date_from_mjd(end.morning_twilight_mjd),
        start.evening_twilight_mjd,
        end.morning_twilight_mjd,
    )


def _distance_to_run_mjd(today_mjd: float, run: List[NightRecord]) -> float:
    """Days from today to the nearest point in an observable run (0 if today is inside)."""
    start_mjd = run[0].evening_twilight_mjd
    end_mjd = run[-1].morning_twilight_mjd
    if start_mjd <= today_mjd <= end_mjd + 1.0:
        return 0.0
    if today_mjd < start_mjd:
        return start_mjd - today_mjd
    return today_mjd - end_mjd


def _find_best_window(
    nights: List[NightRecord],
    today_mjd: float,
) -> Optional[Tuple[str, str, float, float]]:
    """Merged observable season closest to today; prefer the longest run on ties."""
    runs = _merge_close_runs(_build_season_runs(nights))
    if not runs:
        return None
    best = min(
        runs,
        key=lambda run: (
            _distance_to_run_mjd(today_mjd, run),
            -len(run),
            run[0].evening_twilight_mjd,
        ),
    )
    return _run_span(best)


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


def window_mjd_bounds(w: dict) -> Tuple[float, float]:
    """
    MJD interval for plot shading.

    ISO calendar dates define the season; stored MJDs are used only when dates are
    missing or clearly inconsistent (e.g. stale cache with year-long dates but
    tonight-only twilight MJDs).
    """
    date_start = date_end = None
    if w.get("start_date") and w.get("end_date"):
        try:
            date_start = float(Time(w["start_date"], format="iso", scale="utc").mjd)
            date_end = float(Time(w["end_date"], format="iso", scale="utc").mjd) + 1.0
        except Exception:
            date_start = date_end = None

    mjd_start = mjd_end = None
    try:
        if w.get("start_mjd") is not None and w.get("end_mjd") is not None:
            mjd_start = float(w["start_mjd"])
            mjd_end = float(w["end_mjd"]) + 1.0
    except Exception:
        mjd_start = mjd_end = None

    if date_start is not None and date_end is not None:
        if mjd_start is not None and mjd_end is not None:
            date_span = date_end - date_start
            mjd_span = mjd_end - mjd_start
            if date_span > 2.0 and mjd_span < min(date_span * 0.25, 3.0):
                return date_start, date_end
        return date_start, date_end
    if mjd_start is not None and mjd_end is not None:
        return mjd_start, mjd_end
    raise ValueError("observability window missing date or MJD bounds")


def normalize_observability_window(obs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Align top-level dates and repair MJDs that disagree with ISO season boundaries."""
    if not isinstance(obs, dict):
        return None
    out = dict(obs)
    windows = out.get("windows")
    if not isinstance(windows, list):
        windows = []
    norm_windows: List[dict] = []
    for w in windows:
        if not isinstance(w, dict):
            continue
        w = dict(w)
        if w.get("start_date") and w.get("end_date"):
            try:
                date_start = float(Time(w["start_date"], format="iso", scale="utc").mjd)
                date_end = float(Time(w["end_date"], format="iso", scale="utc").mjd)
                mjd_start = w.get("start_mjd")
                mjd_end = w.get("end_mjd")
                if mjd_start is None or mjd_end is None:
                    w["start_mjd"] = date_start
                    w["end_mjd"] = date_end
                else:
                    ms = float(mjd_start)
                    me = float(mjd_end)
                    date_span = (date_end - date_start) + 1.0
                    mjd_span = (me - ms) + 1.0
                    if date_span > 2.0 and mjd_span < min(date_span * 0.25, 3.0):
                        w["start_mjd"] = date_start
                        w["end_mjd"] = date_end
            except Exception:
                pass
        norm_windows.append(w)
    if norm_windows:
        out["windows"] = norm_windows
        w0 = norm_windows[0]
        if w0.get("start_date") and w0.get("end_date"):
            out["start_date"] = w0["start_date"]
            out["end_date"] = w0["end_date"]
            out["next_window_start_date"] = w0["start_date"]
            out["next_window_end_date"] = norm_windows[-1]["end_date"]
    return out


def compute_apf_observability(
    coord: SkyCoord,
    *,
    start_mjd: Optional[float] = None,
    scan_horizon_days: int = SCAN_HORIZON_DAYS,
    plot_horizon_days: int = PLOT_HORIZON_DAYS,
    lick_cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    APF visibility at Lick from reference-now through +scan_horizon_days.

    Always anchored to Time.now() (Pacific calendar date at Lick) unless start_mjd
    is passed for tests. Returns the observable run closest to reference-now.
    Plot shading still caps at plot_horizon_days in rv_keplerian_plots.
    """
    now_mjd, today_iso, today_start_mjd = reference_now(start_mjd)
    plot_end_mjd = now_mjd + float(plot_horizon_days)
    scan_end_mjd = now_mjd + float(scan_horizon_days)

    if _is_circumpolar_for_apf(coord, now_mjd, scan_end_mjd):
        win = ObsWindow(now_mjd, plot_end_mjd, "", "")
        return {
            "circumpolar": True,
            "windows": [win.to_dict()],
            "next_window_start_date": "",
            "next_window_end_date": "",
        }

    nights = _sample_nights(coord, today_start_mjd, scan_end_mjd, lick_cache_path=lick_cache_path)
    nights = [n for n in nights if n.calendar_date >= today_iso]
    season = _find_best_window(nights, now_mjd)

    if season is None:
        return {"circumpolar": False, "windows": []}

    start_date, end_date, start_mjd_win, end_mjd_win = season
    win = ObsWindow(start_mjd_win, end_mjd_win, start_date, end_date)
    return normalize_observability_window(
        {
            "circumpolar": False,
            "windows": [win.to_dict()],
            "next_window_start_date": start_date,
            "next_window_end_date": end_date,
        }
    )


def observability_for_summary(
    summary_path: Path,
    *,
    lick_cache_path: Optional[Path] = None,
    reference_mjd: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    sid = parse_object_id_from_summary(summary_path)
    coord = _target_coord_from_summary(summary_path)
    if sid is None or coord is None:
        return None
    try:
        result = compute_apf_observability(
            coord,
            start_mjd=reference_mjd,
            lick_cache_path=lick_cache_path,
        )
    except FileNotFoundError:
        return None
    if not result.get("windows") and not result.get("circumpolar"):
        return None
    result["gaia_source_id"] = sid
    return result


# Re-export for observability_table_compare and tests.
def _find_evening_twilight_mjd(evening_date: str, *, lick_cache_path: Optional[Path] = None) -> Optional[float]:
    rows = nights_in_mjd_range(
        Time(evening_date, scale="utc").mjd - 0.5,
        Time(evening_date, scale="utc").mjd + 1.5,
        cache_path=lick_cache_path or default_lick_twilight_cache_path(),
    )
    for row in rows:
        if row.get("evening_date") == evening_date:
            return float(row["eve_mjd"])
    return None


def _find_morning_twilight_mjd(evening_mjd: float, *, lick_cache_path: Optional[Path] = None) -> Optional[float]:
    rows = nights_in_mjd_range(
        float(evening_mjd) - 0.5,
        float(evening_mjd) + 1.5,
        cache_path=lick_cache_path or default_lick_twilight_cache_path(),
    )
    for row in rows:
        if abs(float(row["eve_mjd"]) - float(evening_mjd)) < 0.02:
            return float(row["morn_mjd"])
    return None


def night_row_for_evening_date(evening_date: str, *, lick_cache_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    t0 = float(Time(evening_date, scale="utc").mjd)
    rows = nights_in_mjd_range(t0 - 0.5, t0 + 1.5, cache_path=lick_cache_path or default_lick_twilight_cache_path())
    for row in rows:
        if row.get("evening_date") == evening_date:
            return row
    return None
