"""Tests for LAMOST/RAVE external RV normalization in gaia_utils."""

from darkhunter_rv.gaia_utils import _external_rvs_from_unified_rows


def test_rave_row_gets_mjd_from_obs_id() -> None:
    rows = _external_rvs_from_unified_rows(
        [
            {
                "ext_cat": "RAVE_DR6",
                "obs_str": "",
                "rv_z": 12.5,
                "err_z": 1.0,
                "flag_raw": "101021_0941_2",
            }
        ],
        ra_deg=180.0,
        dec_deg=-30.0,
    )
    assert len(rows) == 1
    assert rows[0]["telescope"] == "RAVE_DR6"
    assert rows[0]["mjd"] > 40000
    assert "conv=helio→bary" in rows[0]["flag"]
