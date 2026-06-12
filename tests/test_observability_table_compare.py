import pytest

from darkhunter_rv.observability_table_compare import (
    GAIA_4491,
    REFERENCE_TABLE,
    night_snapshot_for_evening_date,
)


@pytest.mark.parametrize(
    "evening_date",
    ["2026-03-18", "2026-07-13", "2026-08-27", "2026-10-25"],
)
def test_night_snapshot_matches_planning_table(evening_date: str) -> None:
    ref = REFERENCE_TABLE[evening_date]
    snap = night_snapshot_for_evening_date(GAIA_4491, evening_date)
    assert snap is not None
    assert snap.airm_ctr == pytest.approx(ref["airm_ctr"], abs=0.15)
    assert snap.airm_morn == pytest.approx(ref["airm_morn"], abs=0.35)
    assert snap.hrs_below[3.0] == pytest.approx(ref["hrs3"], abs=1.2)
    if "hrs2" in ref:
        assert snap.hrs_below[2.0] == pytest.approx(ref["hrs2"], abs=1.2)
