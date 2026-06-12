"""Tests for external RV frame normalization."""

import numpy as np

from darkhunter_rv.rv_frame import (
    NativeFrame,
    finalize_external_rv_record,
    helio_to_bary_kms,
    mjd_from_rave_obs_id,
    normalize_external_rv_to_barycentric,
)


def test_mjd_from_rave_obs_id() -> None:
    mjd = mjd_from_rave_obs_id("101021_0941_2")
    assert np.isfinite(mjd)
    assert 55000 < mjd < 56000


def test_helio_to_bary_shift_is_finite() -> None:
    rv_bary = helio_to_bary_kms(
        10.0,
        ra_deg=180.0,
        dec_deg=45.0,
        mjd=59000.0,
        site_key="DESI",
    )
    assert np.isfinite(rv_bary)
    assert abs(rv_bary - 10.0) < 40.0


def test_bary_native_pass_through() -> None:
    out = normalize_external_rv_to_barycentric(
        -20.5,
        ra_deg=10.0,
        dec_deg=20.0,
        mjd=59000.0,
        telescope="APOGEE_DR17",
        native_frame=NativeFrame.BARYCENTRIC,
    )
    assert out is not None
    assert out["rv"] == -20.5
    assert "frame=bary-native" in out["flag"]


def test_helio_conversion_flag() -> None:
    rec = finalize_external_rv_record(
        {
            "telescope": "DESI_DR1",
            "mjd": 59000.0,
            "rv": 5.0,
            "rv_err": 1.0,
            "flag": "main/bright/1",
        },
        ra_deg=180.0,
        dec_deg=30.0,
        native_frame=NativeFrame.HELIOCENTRIC,
        site_key="DESI",
    )
    assert rec is not None
    assert "conv=helio→bary" in rec["flag"]
    assert rec["rv"] != 5.0


def test_skip_invalid_mjd() -> None:
    assert (
        finalize_external_rv_record(
            {"telescope": "RAVE_DR6", "mjd": 0.0, "rv": 1.0, "rv_err": 0.5, "flag": "x"},
            ra_deg=1.0,
            dec_deg=2.0,
        )
        is None
    )
