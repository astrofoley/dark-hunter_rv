from pathlib import Path

from darkhunter_rv.website_table_csv import (
    days_since_last_apf_from_summary,
    next_rv_event_from_fit_report,
    sooner_rv_extremum_mjd,
)
from darkhunter_rv.rv_point_filters import mjd_is_valid, rv_epoch_is_valid


def test_mjd_zero_invalid() -> None:
    assert not mjd_is_valid(0.0)
    assert not rv_epoch_is_valid(0.0, 10.0)


def test_sooner_extremum_picks_earlier_event() -> None:
    rep = {"next_rv_max_mjd": 60100.0, "next_rv_min_mjd": 60050.0, "now_mjd": 60000.0}
    assert sooner_rv_extremum_mjd(rep, now_mjd=60000.0) == 60050.0


def test_next_rv_event_prefers_fix_period_ecc(tmp_path: Path) -> None:
    rep = {
        "now_mjd": 60000.0,
        "next_rv_max_mjd": 61000.0,
        "fit_variants": {
            "fix_period_ecc": {"next_rv_max_mjd": 60100.0, "next_rv_min_mjd": 60080.0},
        },
    }
    assert next_rv_event_from_fit_report(rep) == 60080.0


def test_days_since_last_apf_from_summary(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_1_summary.txt"
    summ.write_text(
        "[PIPELINE RESULTS]\n"
        "epoch_1.txt 58000.0 -1.0 0.1 0.2 False\n"
        "epoch_2.txt 58100.0 -2.0 0.1 0.2 False\n"
    )
    age = days_since_last_apf_from_summary(summ, now_mjd=58150.0)
    assert age is not None
    assert age == 50.0
