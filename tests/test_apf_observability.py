import time
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time

from darkhunter_rv.apf_observability import (
    ALTITUDE_MIN_DEG,
    CIRCUMPOLAR_DEC_MIN_DEG,
    HORIZON_DAYS,
    PLOT_HORIZON_DAYS,
    SCAN_HORIZON_DAYS,
    _geometric_min_altitude_deg,
    _is_circumpolar_for_apf_mjd_range,
    compute_apf_observability,
    normalize_observability_window,
    window_mjd_bounds,
)
from darkhunter_rv.lick_twilight_cache import default_cache_path as default_lick_twilight_cache_path


def test_circumpolar_requires_very_high_dec(tmp_path: Path) -> None:
    from darkhunter_rv.lick_twilight_cache import build_cache_years

    lick = tmp_path / "lick_twilight_cache.json"
    build_cache_years([2023], cache_path=lick)
    ref = float(Time("2023-06-15", format="iso", scale="utc").mjd)
    # Dec 70°: above astronomical floor but not every APF night over a full year.
    coord70 = SkyCoord(ra=180.0 * u.deg, dec=70.0 * u.deg, frame="icrs")
    assert not _is_circumpolar_for_apf_mjd_range(
        coord70, ref, ref + 365, lick_cache_path=lick
    )

    coord89 = SkyCoord(ra=180.0 * u.deg, dec=89.0 * u.deg, frame="icrs")
    assert _is_circumpolar_for_apf_mjd_range(coord89, ref, ref + 30, lick_cache_path=lick)


def test_circumpolar_output_has_no_dates(tmp_path: Path) -> None:
    from darkhunter_rv.lick_twilight_cache import build_cache_years

    lick = tmp_path / "lick_twilight_cache.json"
    build_cache_years([2023], cache_path=lick)
    ref = float(Time("2023-06-15", format="iso", scale="utc").mjd)
    coord = SkyCoord(ra=180.0 * u.deg, dec=89.0 * u.deg, frame="icrs")
    out = compute_apf_observability(
        coord,
        start_mjd=ref,
        scan_horizon_days=30,
        plot_horizon_days=30,
        lick_cache_path=lick,
    )
    assert out["circumpolar"] is True
    assert out["windows"]
    assert out["next_window_start_date"] == ""
    assert out["next_window_end_date"] == ""


def test_dec_below_lick_lat_never_circumpolar() -> None:
    coord = SkyCoord(ra=120.0 * u.deg, dec=35.77359 * u.deg, frame="icrs")
    assert float(coord.dec.deg) < CIRCUMPOLAR_DEC_MIN_DEG
    out = compute_apf_observability(
        coord,
        start_mjd=61192.0,
        scan_horizon_days=365,
        lick_cache_path=default_lick_twilight_cache_path(),
    )
    assert out["circumpolar"] is False
    if out.get("windows"):
        w = out["windows"][0]
        span = float(Time(w["end_date"], format="iso", scale="utc").mjd) - float(
            Time(w["start_date"], format="iso", scale="utc").mjd
        )
        assert span < 250.0


def test_dec_58_not_circumpolar() -> None:
    coord = SkyCoord(ra=55.52045 * u.deg, dec=57.90254 * u.deg, frame="icrs")
    assert _geometric_min_altitude_deg(coord) < ALTITUDE_MIN_DEG
    out = compute_apf_observability(coord, start_mjd=61000.0, scan_horizon_days=120, lick_cache_path=default_lick_twilight_cache_path())
    assert out["circumpolar"] is False


def test_mid_latitude_single_season_window() -> None:
    coord = SkyCoord(ra=120.0 * u.deg, dec=20.0 * u.deg, frame="icrs")
    out = compute_apf_observability(
        coord,
        start_mjd=61192.0,
        scan_horizon_days=90,
        lick_cache_path=default_lick_twilight_cache_path(),
    )
    assert out["circumpolar"] is False
    if out["windows"]:
        assert len(out["windows"]) == 1
        w = out["windows"][0]
        assert w["start_date"] <= w["end_date"]
        assert w["end_mjd"] > w["start_mjd"]


def test_airmass_limit_is_17() -> None:
    from darkhunter_rv.apf_observability import MAX_AIRMASS

    assert MAX_AIRMASS == pytest.approx(1.7)
    assert ALTITUDE_MIN_DEG == pytest.approx(36.3, abs=0.5)


def test_window_mjd_bounds_rejects_stale_year_long_dates() -> None:
    from astropy.time import Time

    w = {
        "start_date": "2026-06-17",
        "end_date": "2027-06-17",
        "start_mjd": float(Time("2026-06-17T20:00:00", scale="utc").mjd),
        "end_mjd": float(Time("2026-06-18T06:00:00", scale="utc").mjd),
    }
    start, end = window_mjd_bounds(w)
    assert end - start < 5.0


def test_normalize_repairs_stale_year_long_iso_dates() -> None:
    from astropy.time import Time

    obs = normalize_observability_window(
        {
            "circumpolar": False,
            "windows": [
                {
                    "start_date": "2026-06-17",
                    "end_date": "2027-06-17",
                    "start_mjd": float(Time("2026-06-17T20:00:00", scale="utc").mjd),
                    "end_mjd": float(Time("2026-06-18T06:00:00", scale="utc").mjd),
                }
            ],
        }
    )
    assert obs is not None
    w = obs["windows"][0]
    assert w["end_date"] != "2027-06-17"
    assert _season_span_days(w["start_date"], w["end_date"]) < 250.0


def _season_span_days(start_date: str, end_date: str) -> float:
    from astropy.time import Time

    return float(Time(end_date, format="iso", scale="utc").mjd) - float(
        Time(start_date, format="iso", scale="utc").mjd
    )


def test_normalize_observability_window_repairs_stale_mjds() -> None:
    from astropy.time import Time

    from darkhunter_rv.apf_observability import normalize_observability_window

    obs = normalize_observability_window(
        {
            "windows": [
                {
                    "start_date": "2026-06-17",
                    "end_date": "2026-07-31",
                    "start_mjd": float(Time("2026-06-17T20:00:00", scale="utc").mjd),
                    "end_mjd": float(Time("2026-06-18T06:00:00", scale="utc").mjd),
                }
            ]
        }
    )
    assert obs is not None
    w = obs["windows"][0]
    assert w["end_mjd"] - w["start_mjd"] > 20.0
    assert obs["start_date"] == "2026-06-17"
    assert obs["end_date"] == "2026-07-31"


def test_plot_horizon_is_three_months() -> None:
    assert PLOT_HORIZON_DAYS == 90
    assert SCAN_HORIZON_DAYS >= 180


def test_season_window_not_full_scan_year() -> None:
    """Merged runs must not bridge separate seasons into a ~365d window."""
    from astropy.time import Time

    ref = float(Time("2026-06-17T14:00:00", scale="utc").mjd)
    lick = default_lick_twilight_cache_path()
    for ra, dec in ((55.52045, 57.90254), (120.0, 20.0), (180.0, 45.0)):
        out = compute_apf_observability(
            SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
            start_mjd=ref,
            scan_horizon_days=365,
            lick_cache_path=lick,
        )
        if not out.get("windows"):
            continue
        w = out["windows"][0]
        date_span = float(Time(w["end_date"], format="iso", scale="utc").mjd) - float(
            Time(w["start_date"], format="iso", scale="utc").mjd
        )
        assert date_span < 250.0, (ra, dec, w["start_date"], w["end_date"])


def test_trim_run_calendar_span_caps_scan_horizon_merge() -> None:
    from astropy.time import Time

    from darkhunter_rv.apf_observability import (
        MAX_SEASON_CALENDAR_DAYS,
        NightRecord,
        _trim_run_calendar_span,
    )

    ref_mjd = float(Time("2026-07-01T12:00:00", scale="utc").mjd)
    today_start = float(Time("2026-07-01", format="iso", scale="utc").mjd)
    year_run: list[NightRecord] = []
    for day in range(365):
        eve = today_start + float(day) + 0.75
        year_run.append(
            NightRecord(
                evening_twilight_mjd=eve,
                morning_twilight_mjd=eve + 0.4,
                calendar_date=Time(eve, format="mjd").to_datetime().date().isoformat(),
                observable_night=True,
            )
        )
    trimmed = _trim_run_calendar_span(
        year_run,
        today_mjd=ref_mjd,
        today_start_mjd=today_start,
    )
    assert trimmed
    start_date = trimmed[0].calendar_date
    end_date = Time(trimmed[-1].morning_twilight_mjd, format="mjd").to_datetime().date().isoformat()
    span = float(Time(end_date, format="iso", scale="utc").mjd) - float(
        Time(start_date, format="iso", scale="utc").mjd
    )
    assert span <= MAX_SEASON_CALENDAR_DAYS
    assert trimmed[0].evening_twilight_mjd >= today_start - 0.5


def test_clamp_season_dates_caps_year_long_output() -> None:
    from darkhunter_rv.apf_observability import _clamp_season_dates

    start_date, end_date, start_mjd, end_mjd = _clamp_season_dates(
        "2026-07-01",
        "2027-07-02",
        61222.0,
        61588.0,
    )
    assert end_date < "2027-07-02"
    assert _season_span_days(start_date, end_date) <= 250.0


def test_compute_window_from_july_reference_not_full_year(tmp_path: Path) -> None:
    from astropy.time import Time

    from darkhunter_rv.lick_twilight_cache import build_cache_years

    lick_cache = tmp_path / "lick_twilight_cache.json"
    build_cache_years([2026, 2027], cache_path=lick_cache)
    ref = float(Time("2026-07-01T12:00:00", scale="utc").mjd)
    out = compute_apf_observability(
        SkyCoord(ra=120.0 * u.deg, dec=20.0 * u.deg),
        start_mjd=ref,
        scan_horizon_days=365,
        lick_cache_path=lick_cache,
    )
    assert out.get("windows")
    w = out["windows"][0]
    span = _season_span_days(w["start_date"], w["end_date"])
    assert span < 250.0, (w["start_date"], w["end_date"])
    assert w["start_date"] >= "2026-07-01"


def test_full_year_scan_is_fast(tmp_path: Path) -> None:
    from darkhunter_rv.lick_twilight_cache import build_cache_years

    lick_cache = tmp_path / "lick_twilight_cache.json"
    build_cache_years([2026], cache_path=lick_cache)

    coord = SkyCoord(ra=120.0 * u.deg, dec=20.0 * u.deg, frame="icrs")
    t0 = time.perf_counter()
    compute_apf_observability(
        coord,
        start_mjd=61192.0,
        scan_horizon_days=365,
        lick_cache_path=lick_cache,
    )
    assert time.perf_counter() - t0 < 2.0


def test_forward_scan_does_not_start_before_reference_today(tmp_path: Path) -> None:
    """Window search is [reference-today, reference-today+1y) in Lick local calendar."""
    from astropy.time import Time

    from darkhunter_rv.apf_observability import reference_now
    from darkhunter_rv.lick_twilight_cache import build_cache_years

    lick_cache = tmp_path / "lick_twilight_cache.json"
    build_cache_years([2026], cache_path=lick_cache)
    ref_mjd, ref_iso, _ref_start = reference_now(float(Time("2026-06-12").mjd))

    for ra in (55.5, 120.0, 180.0, 270.0):
        for dec in (-10.0, 20.0, 45.0, 58.0):
            out = compute_apf_observability(
                SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                start_mjd=ref_mjd,
                scan_horizon_days=365,
                lick_cache_path=lick_cache,
            )
            start = out.get("next_window_start_date", "")
            if start:
                assert start >= ref_iso


def test_resolve_observability_prefers_live_over_stale_cache(tmp_path: Path) -> None:
    from astropy.time import Time

    from darkhunter_rv.apf_observability import reference_now
    from darkhunter_rv.lick_twilight_cache import build_cache_years
    from fit_apf_rv_keplerian import resolve_observability_window

    lick_cache = tmp_path / "lick_twilight_cache.json"
    build_cache_years([2026], cache_path=lick_cache)
    obs_cache = tmp_path / "observability_windows_cache.json"
    obs_cache.write_text(
        '{"999": {"circumpolar": false, "next_window_start_date": "2020-01-01", '
        '"next_window_end_date": "2020-01-31", "windows": [{"start_date": "2020-01-01", "end_date": "2020-01-31"}]}}'
    )
    summ = tmp_path / "Gaia_DR3_999_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\nSource_ID: 999\nRA: 120.0\nDec: 20.0\n\n[PIPELINE RESULTS]\n"
    )
    ref_mjd, ref_iso, _ref_start = reference_now(float(Time("2026-06-12").mjd))
    obs = resolve_observability_window(summ, "999", obs_cache)
    assert obs is not None
    start = obs.get("next_window_start_date") or obs.get("windows", [{}])[0].get("start_date", "")
    if start:
        assert start >= ref_iso
        assert not start.startswith("2020-")
