import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy import units as u

from darkhunter_rv.apf_observability import (
    ALTITUDE_MIN_DEG,
    HORIZON_DAYS,
    _is_circumpolar_for_apf,
    compute_apf_observability,
)


def test_circumpolar_requires_very_high_dec() -> None:
    # Dec 70° is geometrically circumpolar at Lick but not at airmass 1.7 all night.
    coord70 = SkyCoord(ra=180.0 * u.deg, dec=70.0 * u.deg, frame="icrs")
    assert not _is_circumpolar_for_apf(coord70, 60000.0, 60000.0 + HORIZON_DAYS)

    coord89 = SkyCoord(ra=180.0 * u.deg, dec=89.0 * u.deg, frame="icrs")
    assert _is_circumpolar_for_apf(coord89, 60000.0, 60000.0 + 30)


def test_circumpolar_output_has_no_dates() -> None:
    coord = SkyCoord(ra=180.0 * u.deg, dec=89.0 * u.deg, frame="icrs")
    out = compute_apf_observability(coord, start_mjd=60000.0, horizon_days=30)
    assert out["circumpolar"] is True
    assert out["windows"]
    assert out["next_window_start_date"] == ""
    assert out["next_window_end_date"] == ""


def test_mid_latitude_single_season_window() -> None:
    coord = SkyCoord(ra=120.0 * u.deg, dec=20.0 * u.deg, frame="icrs")
    out = compute_apf_observability(coord, start_mjd=60000.0, horizon_days=90)
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
