"""Tests for summary-only RV data plots (no network)."""

from pathlib import Path

from scripts import plot_rv_from_summaries as rvplot


def _write_summary(path: Path, *, n_epochs: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"Gaia_DR3_111_epoch_{i}.txt {60000 + i} {-10.0 + i * 0.1} 0.02 0.4 False"
        for i in range(1, n_epochs + 1)
    )
    path.write_text(
        "[GAIA METADATA]\nSource_ID: 111\nRA: 1.0\nDec: 2.0\n"
        f"\n[PIPELINE RESULTS]\n# hdr\n{rows}\n"
    )


def test_build_plot_nested_summary(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_111" / "Gaia_DR3_111_summary.txt"
    _write_summary(summ, n_epochs=4)
    out_png = tmp_path / "plots" / "Gaia_DR3_111" / "Gaia_DR3_111_rv_plot.png"
    assert rvplot.build_plot(summ, out_png) is True
    assert out_png.is_file()
    assert out_png.stat().st_size > 1000


def test_build_plot_single_epoch(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_111_summary.txt"
    _write_summary(summ, n_epochs=1)
    out_png = tmp_path / "Gaia_DR3_111_rv_plot.png"
    assert rvplot.build_plot(summ, out_png) is True
    assert out_png.is_file()
