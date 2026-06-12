"""Tests for DESI external RV query helpers (no network)."""

from darkhunter_rv import gaia_utils
from darkhunter_rv.cone_utils import filter_rows_in_cone
from darkhunter_rv.desi_rv_query import (
    desi_rows_to_external_rvs,
    query_desi_rvs_for_gaia_ids,
)


def test_desi_rows_to_external_rvs() -> None:
    rows = desi_rows_to_external_rvs(
        [
            {
                "source_id": "381210920555640576",
                "targetid": "28684312379395",
                "survey": "sv1",
                "program": "other",
                "rv_adop": "-153.35",
                "rv_err": "1.12",
                "rvs_warn": "0",
                "min_mjd": "59230.16",
                "max_mjd": "59230.19",
            }
        ],
        ra_deg=180.0,
        dec_deg=45.0,
    )
    assert len(rows) == 1
    assert rows[0]["telescope"] == "DESI_DR1"
    assert rows[0]["rv"] != -153.35
    assert "conv=helio→bary" in rows[0]["flag"]
    assert rows[0]["mjd"] == 0.5 * (59230.16 + 59230.19)
    assert "sv1/other" in rows[0]["flag"]


def test_desi_rows_skip_missing_mjd() -> None:
    assert desi_rows_to_external_rvs([{"rv_adop": "1.0", "rv_err": "0.1"}]) == []


def test_merge_external_rv_lists_replaces_desi() -> None:
    existing = [
        {"telescope": "LAMOST_LRS", "mjd": 59000.0, "rv": -10.0, "rv_err": 1.0, "flag": "z"},
        {"telescope": "DESI_DR1", "mjd": 59200.0, "rv": -20.0, "rv_err": 2.0, "flag": "old"},
    ]
    new_desi = [{"telescope": "DESI_DR1", "mjd": 59230.0, "rv": -15.0, "rv_err": 1.5, "flag": "new"}]
    merged = gaia_utils.merge_external_rv_lists(existing, new_desi)
    assert len(merged) == 2
    assert merged[0]["telescope"] == "LAMOST_LRS"
    assert merged[1]["rv"] == -15.0


def test_replace_external_rv_section(tmp_path) -> None:
    summ = tmp_path / "Gaia_DR3_111_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\nSource_ID: 111\nRA: 1.0\nDec: 2.0\n"
        "\n[EXTERNAL RV DATA]\n# No external data found.\n"
        "\n[PIPELINE RESULTS]\n# hdr\n"
    )
    gaia_utils.replace_external_rv_section_in_summary(
        summ,
        [{"telescope": "DESI_DR1", "mjd": 59230.0, "rv": -12.0, "rv_err": 1.0, "flag": "main/bright/1"}],
    )
    ext = gaia_utils.parse_external_rvs_from_star_summary(summ)
    assert len(ext) == 1
    assert ext[0]["telescope"] == "DESI_DR1"
    text = summ.read_text()
    assert "[PIPELINE RESULTS]" in text
    assert "[GAIA METADATA]" in text


def test_query_desi_batch_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "darkhunter_rv.desi_rv_query.execute_datalab_adql",
        lambda *_a, **_k: [],
    )
    assert query_desi_rvs_for_gaia_ids([123]) == {}


def test_filter_rows_in_cone_negative_dec() -> None:
    rows = [
        {
            "target_ra": "180.5",
            "target_dec": "-45.25",
            "rv_adop": "1.0",
            "targetid": "1",
            "survey": "main",
            "program": "bright",
            "rv_err": "0.1",
            "rvs_warn": "0",
            "min_mjd": "60000",
            "max_mjd": "60001",
        }
    ]
    kept = filter_rows_in_cone(rows, 180.5, -45.25, radius_deg=0.01)
    assert len(kept) == 1
    assert kept[0]["sep_deg"] == 0.0


def test_query_desi_cone_fallback_when_source_id_misses(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_adql(sql: str, *, timeout: float = 180.0):
        calls.append(sql)
        if "m.source_id IN" in sql:
            return []
        if "BETWEEN" in sql:
            assert "(-45.25)" in sql or "(-45.25) -" in sql
            return [
                {
                    "source_id": "999888777",
                    "targetid": "1",
                    "survey": "main",
                    "program": "bright",
                    "rv_adop": "10.5",
                    "rv_err": "0.8",
                    "rvs_warn": "0",
                    "target_ra": "180.5",
                    "target_dec": "-45.25",
                    "min_mjd": "60000.0",
                    "max_mjd": "60001.0",
                }
            ]
        return []

    monkeypatch.setattr("darkhunter_rv.desi_rv_query.execute_datalab_adql", _fake_adql)
    out = query_desi_rvs_for_gaia_ids([123], positions_by_id={123: (180.5, -45.25)})
    assert 123 in out
    assert out[123][0]["rv"] != 10.5
    assert "conv=helio→bary" in out[123][0]["flag"]
    assert "sep=" in out[123][0]["flag"]
    assert len(calls) == 2
