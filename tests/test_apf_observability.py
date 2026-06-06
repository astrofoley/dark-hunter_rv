import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy import units as u

from darkhunter_rv.apf_observability import compute_apf_observability


def test_circumpolar_target_has_flag_and_window() -> None:
    # Dec well above Lick latitude → circumpolar at APF.
    coord = SkyCoord(ra=180.0 * u.deg, dec=70.0 * u.deg, frame="icrs")
    out = compute_apf_observability(coord, start_mjd=60000.0, horizon_days=30)
    assert out["circumpolar"] is True
    assert out["windows"]
    assert out["next_window_start_date"]
    assert out["next_window_end_date"]


def test_mid_latitude_can_yield_windows_or_empty() -> None:
    coord = SkyCoord(ra=120.0 * u.deg, dec=20.0 * u.deg, frame="icrs")
    out = compute_apf_observability(coord, start_mjd=60000.0, horizon_days=90)
    assert out["circumpolar"] is False
    assert "windows" in out
