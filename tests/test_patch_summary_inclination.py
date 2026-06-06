"""Tests for patch_summary_inclination_from_thiele_innes.py."""

from __future__ import annotations

from pathlib import Path

from darkhunter_rv.thiele_innes_inclination import campbell_to_thiele_innes
from scripts.patch_summary_inclination_from_thiele_innes import patch_summary_path


def test_patch_summary_writes_inclination(tmp_path: Path) -> None:
    A, B, F, G = campbell_to_thiele_innes(2.0, 72.0, 30.0, 110.0)
    summ = tmp_path / "Gaia_DR3_1234567890123456789_summary.txt"
    summ.write_text(
        "### STAR SUMMARY: 1234567890123456789 ###\n\n"
        "[GAIA METADATA]\n"
        f"A_Thiele_Innes: {A:.8f}\n"
        f"B_Thiele_Innes: {B:.8f}\n"
        f"F_Thiele_Innes: {F:.8f}\n"
        f"G_Thiele_Innes: {G:.8f}\n"
        "Inclination: NaN\n"
        "\n[PIPELINE RESULTS]\n# File | MJD | RV\n",
        encoding="utf-8",
    )
    assert patch_summary_path(summ, dry_run=False) == "patched"
    text = summ.read_text(encoding="utf-8")
    assert "Inclination: 72." in text or "Inclination: 71." in text
