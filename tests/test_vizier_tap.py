"""Tests for VizieR TAP helpers."""

from darkhunter_rv.vizier_tap import parse_gaia_source_id, sql_max_err_clause


def test_sql_max_err_clause_empty_when_unset() -> None:
    assert sql_max_err_clause("VERR", None) == ""


def test_sql_max_err_clause_when_set() -> None:
    assert "VERR <= 10" in sql_max_err_clause("VERR", 10.0)


def test_parse_gaia_source_id_avoids_float_roundoff() -> None:
    sid = 4667368899326729856
    assert parse_gaia_source_id(str(sid)) == sid
    assert parse_gaia_source_id(sid) == sid
