"""Tests for GALAH external RV query helpers (no network)."""

from darkhunter_rv.galah_rv_query import galah_rows_to_external_rvs, query_galah_rvs_for_gaia_ids


def test_galah_rows_to_external_rvs() -> None:
    rows = galah_rows_to_external_rvs(
        [
            {
                "gaia_id": "123",
                "galah_id": "999",
                "rv": "-10.5",
                "rv_err": "0.2",
                "mjd": "59000.0",
                "rv_flag": "0",
            }
        ],
        ra_deg=180.0,
        dec_deg=-30.0,
    )
    assert len(rows) == 1
    assert rows[0]["telescope"] == "GALAH_DR3"
    assert "conv=helio→bary" in rows[0]["flag"]


def test_query_galah_batch_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "darkhunter_rv.galah_rv_query.execute_vizier_adql",
        lambda *_a, **_k: [],
    )
    assert query_galah_rvs_for_gaia_ids([123], positions_by_id={123: (180.0, -30.0)}) == {}
