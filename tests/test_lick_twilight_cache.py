"""Tests for Lick calendar twilight cache parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_rv.lick_twilight_cache import parse_calendar_text


SAMPLE = """
 SUN MAR 01  18 06  18 56  19 26  16 16  06 21  05 11  05 41  06 31  05 29  10 34  16 16   10.8  0.0   0  0959 1258   26
 SUN MAR 08  18 13  19 03  19 33  23 41         05 01  05 32  06 21  06 04  11 02  16 34   10.5  4.6  44  1532-2439   89
 SUN MAR 18  18 22  19 12  19 43  06 16  18 23  04 46  05 17  06 06  06 52  11 41  16 59   10.1 10.1 100  0005 0214  >90
"""


def test_parse_calendar_extracts_twilight_and_lst() -> None:
    rows = parse_calendar_text(SAMPLE, 2026)
    assert len(rows) == 3
    mar18 = next(r for r in rows if r["evening_date"] == "2026-03-18")
    assert mar18["lst_even_deg"] == pytest.approx((6.0 + 52.0 / 60.0) * 15.0, abs=0.5)
    assert mar18["morn_mjd"] > mar18["eve_mjd"]


def test_parse_calendar_handles_missing_moonset() -> None:
    rows = parse_calendar_text(SAMPLE, 2026)
    mar08 = next(r for r in rows if r["evening_date"] == "2026-03-08")
    assert mar08["morn_mjd"] > mar08["eve_mjd"]
