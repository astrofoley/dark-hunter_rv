"""Tests for literature RV master table builder."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from validation.build_literature_rv_master import (
    MASTER_COLUMNS,
    build_master,
    write_master,
)

_REPO = Path(__file__).resolve().parents[2]
_DOWNLOADS = Path.home() / "Downloads"


@pytest.fixture(scope="module")
def master_rows(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, str]]:
    paths = {
        "elbadry_2024_rvs": _DOWNLOADS / "arXiv-2405.00089v2/all_ns_rvs.txt",
        "elbadry_ns1_tex": _DOWNLOADS / "arXiv-2402.06722v2/manuscript.tex",
        "elbadry_bh1_tex": _DOWNLOADS / "arXiv-2209.06833v3/gaia_bh1.tex",
        "simon_2026_vel": _DOWNLOADS / "arXiv-2603.20371v1/velocity_table_stub.tex",
    }
    for p in paths.values():
        if not p.is_file():
            pytest.skip(f"Missing local arXiv source: {p}")
    return build_master(
        elbadry_2024_rvs=paths["elbadry_2024_rvs"],
        elbadry_2024_sample_tex=None,
        elbadry_ns1_tex=paths["elbadry_ns1_tex"],
        elbadry_bh1_tex=paths["elbadry_bh1_tex"],
        simon_2026_vel=paths["simon_2026_vel"],
    )


def test_master_row_count(master_rows: list[dict[str, str]]) -> None:
    assert len(master_rows) == 417


def test_master_has_four_references(master_rows: list[dict[str, str]]) -> None:
    keys = {r["reference_key"] for r in master_rows}
    assert keys == {
        "ElBadry2024_NS_population",
        "ElBadry2024_GaiaNS1",
        "ElBadry2022_GaiaBH1",
        "Simon2026_DR3_compact_followup",
    }


def test_j1432_in_two_references(master_rows: list[dict[str, str]]) -> None:
    j1432 = [r for r in master_rows if r["gaia_dr3_id"] == "6328149636482597888"]
    refs = {r["reference_key"] for r in j1432}
    assert refs == {"ElBadry2024_NS_population", "ElBadry2024_GaiaNS1"}
    assert len(j1432) == 34 + 59


def test_bh1_has_resolution_and_snr(master_rows: list[dict[str, str]]) -> None:
    bh1 = [r for r in master_rows if r["reference_key"] == "ElBadry2022_GaiaBH1"]
    assert len(bh1) == 41
    assert all(r["spectral_resolution_R"] and r["snr_per_pixel"] for r in bh1)


def test_committed_master_matches_builder(tmp_path: Path, master_rows: list[dict[str, str]]) -> None:
    out = tmp_path / "literature_rv_master.csv"
    write_master(master_rows, out)
    with out.open() as f:
        written = list(csv.DictReader(f))
    assert [r["reference_key"] for r in written] == [r["reference_key"] for r in master_rows]
    assert list(written[0].keys()) == MASTER_COLUMNS
