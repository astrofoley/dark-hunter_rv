from pathlib import Path

import pytest

from darkhunter_rv import gaia_utils


def test_parse_gaia_metadata_roundtrip(tmp_path: Path):
    summ = tmp_path / "Gaia_DR3_123_summary.txt"
    summ.write_text(
        "### STAR ###\n\n"
        "[GAIA METADATA]\n"
        "Teff: 5800.12345678\n"
        "Source_ID: 1702370142434513152\n"
        "RA: 180.5\n"
        "Dec: -45.25\n"
        "\n[EXTERNAL RV DATA]\n"
        "# Telescope | MJD | RV\n"
        "# No external data found.\n"
        "\n[PIPELINE RESULTS]\n"
        "# File | MJD\n"
    )
    md = gaia_utils.parse_gaia_metadata_from_star_summary(summ)
    assert md is not None
    assert md["Teff"] == pytest.approx(5800.12345678)
    assert md["Source_ID"] == 1702370142434513152
    assert gaia_utils.star_summary_metadata_complete(md)


def test_parse_rejects_query_failed(tmp_path: Path):
    summ = tmp_path / "x.txt"
    summ.write_text("[GAIA METADATA]\nNot Found or Query Failed.\n")
    assert gaia_utils.parse_gaia_metadata_from_star_summary(summ) is None


def test_load_requires_keys(tmp_path: Path):
    summ = tmp_path / "x.txt"
    summ.write_text("[GAIA METADATA]\nTeff: 5000.0\n")
    assert gaia_utils.load_gaia_data_from_star_summary(summ) is None


def test_load_ok_when_teff_missing_or_nan(tmp_path: Path):
    summ = tmp_path / "hot.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Teff: nan\n"
        "Source_ID: 412195879777348480\n"
        "RA: 12.5\n"
        "Dec: -34.0\n"
        "\n[EXTERNAL RV DATA]\n# No external data found.\n"
    )
    g = gaia_utils.load_gaia_data_from_star_summary(
        summ, expected_source_id=412195879777348480
    )
    assert g is not None
    assert g["metadata"]["RA"] == pytest.approx(12.5)


def test_load_rejects_wrong_source_id(tmp_path: Path):
    summ = tmp_path / "wrong.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Source_ID: 999\n"
        "RA: 1.0\n"
        "Dec: 2.0\n"
        "\n[EXTERNAL RV DATA]\n# No external data found.\n"
    )
    assert gaia_utils.load_gaia_data_from_star_summary(summ, expected_source_id=1) is None


def test_load_requires_ra_dec_even_with_teff_and_source(tmp_path: Path):
    """Incomplete summaries should not satisfy cache (triggers Gaia query to backfill coordinates)."""
    summ = tmp_path / "s.txt"
    summ.write_text(
        "[GAIA METADATA]\nTeff: 5750.0\nSource_ID: 1702370142434513152\n\n[EXTERNAL RV DATA]\n# No external data found.\n"
    )
    assert gaia_utils.load_gaia_data_from_star_summary(summ) is None


def test_star_summary_has_external_rv_section(tmp_path: Path):
    a = tmp_path / "a.txt"
    a.write_text("[GAIA METADATA]\nx: 1\n[EXTERNAL RV DATA]\n# No external\n")
    assert gaia_utils.star_summary_has_external_rv_section(a)
    b = tmp_path / "b.txt"
    b.write_text("[GAIA METADATA]\nx: 1\n")
    assert not gaia_utils.star_summary_has_external_rv_section(b)


def test_resolve_gaia_skips_network_without_external_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Full metadata on disk but no [EXTERNAL RV DATA] block must not hit TAP."""
    summ = tmp_path / "Gaia_DR3_88_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Source_ID: 88\n"
        "RA: 10.0\n"
        "Dec: 20.0\n"
        "Parallax: 0.01\n"
    )

    def _fail(*_a, **_k):
        raise AssertionError("network should not run")

    monkeypatch.setattr(gaia_utils, "query_gaia_data", _fail)
    monkeypatch.setattr(gaia_utils, "query_external_rvs_only_from_disk_metadata", _fail)
    g = gaia_utils.resolve_gaia_data(88, summ, force_query=False)
    assert g is not None
    assert int(g["metadata"]["Source_ID"]) == 88


def test_resolve_gaia_skips_network_when_cache_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    summ = tmp_path / "Gaia_DR3_99_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Source_ID: 99\n"
        "RA: 11.25\n"
        "Dec: -22.5\n"
        "Parallax: 0.02\n"
        "\n[EXTERNAL RV DATA]\n"
        "# No external data found.\n"
    )

    def _fail(_sid):
        raise AssertionError("network should not run")

    monkeypatch.setattr(gaia_utils, "query_gaia_data", _fail)
    g = gaia_utils.resolve_gaia_data(99, summ, force_query=False)
    assert g is not None
    assert int(g["metadata"]["Source_ID"]) == 99


def test_normalize_ra_dec_aliases(tmp_path: Path):
    summ = tmp_path / "aliases.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Teff: 5750.0\n"
        "Source_ID: 1702370142434513152\n"
        "ra_source: 180.0\n"
        "dec_source: -30.5\n"
        "\n[EXTERNAL RV DATA]\n# No external data found.\n"
    )
    g = gaia_utils.load_gaia_data_from_star_summary(summ)
    assert g is not None
    assert g["metadata"]["RA"] == pytest.approx(180.0)
    assert g["metadata"]["Dec"] == pytest.approx(-30.5)
