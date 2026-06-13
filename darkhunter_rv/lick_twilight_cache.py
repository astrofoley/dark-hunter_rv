"""Lick Observatory nautical (-12°) twilight from UCO calendar tables."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
from astropy.time import Time

LICK_CALENDAR_BASE = "https://mthamilton.ucolick.org/techdocs/calendars/lick"
LICK_TZ = ZoneInfo("America/Los_Angeles")
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MONTH_TO_NUM = {name.upper(): i + 1 for i, name in enumerate(MONTH_NAMES)}

_LINE_RE = re.compile(r"^\s*[A-Z]{3}\s+([A-Z]{3})\s+(\d{2})\s+(.*)$")
_PAIR_RE = re.compile(r"(\d{2})\s+(\d{2})")


def default_cache_path(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[1]
    return root / "rv_fit_reports" / "lick_twilight_cache.json"


def _hm_to_hours(text: str) -> Optional[float]:
    text = (text or "").strip()
    if not text:
        return None
    m = _PAIR_RE.fullmatch(text)
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2)) / 60.0


def _hours_to_deg(hours: float) -> float:
    return float(hours) * 15.0


def _pacific_mjd(year: int, month: int, day: int, hour: int, minute: int) -> float:
    dt = datetime(year, month, day, hour, minute, tzinfo=LICK_TZ)
    return float(Time(dt).mjd)


def _parse_calendar_line(line: str, year: int) -> Optional[Dict[str, Any]]:
    m = _LINE_RE.match(line)
    if not m:
        return None
    month_name = m.group(1)
    day = int(m.group(2))
    month = MONTH_TO_NUM.get(month_name.upper())
    if month is None:
        return None

    rest = re.split(r"\s+\d+\.\d+", m.group(3), maxsplit=1)[0]
    pairs = [(int(a), int(b)) for a, b in _PAIR_RE.findall(rest)]
    if len(pairs) < 10:
        return None

    if len(pairs) >= 11:
        twi12 = pairs[1]
        dawn12 = pairs[6]
        lst_twi = pairs[8]
        lst_dawn = pairs[10]
    else:
        twi12 = pairs[1]
        dawn12 = pairs[5]
        lst_twi = pairs[7]
        lst_dawn = pairs[9]

    eve_h, eve_m = twi12
    dawn_h, dawn_m = dawn12
    eve_mjd = _pacific_mjd(year, month, day, eve_h, eve_m)

    dawn_day = day
    dawn_month = month
    dawn_year = year
    if dawn_h < 12:
        d = datetime(year, month, day, tzinfo=LICK_TZ) + timedelta(days=1)
        dawn_year, dawn_month, dawn_day = d.year, d.month, d.day
    morn_mjd = _pacific_mjd(dawn_year, dawn_month, dawn_day, dawn_h, dawn_m)

    evening_date = f"{year:04d}-{month:02d}-{day:02d}"
    lst_even_h = lst_twi[0] + lst_twi[1] / 60.0
    lst_morn_h = lst_dawn[0] + lst_dawn[1] / 60.0

    return {
        "evening_date": evening_date,
        "eve_mjd": eve_mjd,
        "morn_mjd": morn_mjd,
        "lst_even_deg": _hours_to_deg(lst_even_h),
        "lst_morn_deg": _hours_to_deg(lst_morn_h),
    }


def fetch_month_calendar(year: int, month_name: str) -> str:
    url = f"{LICK_CALENDAR_BASE}/lick{year}.12.{month_name}.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "dark-hunter-rv/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_calendar_text(text: str, year: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        row = _parse_calendar_line(line, year)
        if row is not None:
            out.append(row)
    return out


def build_cache_years(
    years: List[int],
    *,
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    nights: List[Dict[str, Any]] = []
    for year in years:
        for month_name in MONTH_NAMES:
            try:
                text = fetch_month_calendar(year, month_name)
            except (urllib.error.URLError, TimeoutError) as exc:
                raise RuntimeError(f"failed to fetch Lick calendar {year} {month_name}: {exc}") from exc
            nights.extend(parse_calendar_text(text, year))
    nights.sort(key=lambda r: r["eve_mjd"])
    payload = {
        "source": LICK_CALENDAR_BASE,
        "twilight": "nautical_12_deg",
        "timezone": "America/Los_Angeles",
        "years": years,
        "nights": nights,
    }
    path = cache_path or default_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def load_cache(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or default_cache_path()
    if not p.is_file():
        return {"nights": []}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {"nights": []}
    except Exception:
        return {"nights": []}


def nights_in_mjd_range(
    start_mjd: float,
    end_mjd: float,
    *,
    cache_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    data = load_cache(cache_path)
    nights = data.get("nights") or []
    out: List[Dict[str, Any]] = []
    for row in nights:
        if not isinstance(row, dict):
            continue
        eve = row.get("eve_mjd")
        if eve is None:
            continue
        eve_f = float(eve)
        if eve_f < start_mjd - 0.5 or eve_f > end_mjd + 1.0:
            continue
        out.append(row)
    return out


def interpolate_lst_deg(row: Dict[str, Any], mjd: float) -> float:
    eve = float(row["eve_mjd"])
    morn = float(row["morn_mjd"])
    lst0 = float(row["lst_even_deg"])
    lst1 = float(row["lst_morn_deg"])
    if morn <= eve:
        return lst0
    frac = (float(mjd) - eve) / (morn - eve)
    delta = ((lst1 - lst0 + 180.0) % 360.0) - 180.0
    return (lst0 + frac * delta) % 360.0
