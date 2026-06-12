"""Tests for APOGEE external RV query helpers (no network)."""

from darkhunter_rv.apogee_rv_query import apogee_rows_to_external_rvs, query_apogee_rvs_for_gaia_ids


def test_apogee_rows_to_external_rvs() -> None:
    rows = apogee_rows_to_external_rvs(
        [
            {
                "gaia_id": "123",
                "apogee_id": "2M0000+0000",
                "apo_telescope": "apo25m",
                "rv": "10.0",
                "rv_err": "0.05",
                "mjd": "58402.0",
            }
        ],
        ra_deg=180.0,
        dec_deg=-30.0,
    )
    assert len(rows) == 1
    assert rows[0]["telescope"] == "APOGEE_DR17"
    assert "frame=bary-native" in rows[0]["flag"]
    assert "tel=apo25m" in rows[0]["flag"]


def test_query_apogee_batch_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "darkhunter_rv.apogee_rv_query.execute_vizier_adql",
        lambda *_a, **_k: [],
    )
    assert query_apogee_rvs_for_gaia_ids([123], positions_by_id={123: (180.0, -30.0)}) == {}
