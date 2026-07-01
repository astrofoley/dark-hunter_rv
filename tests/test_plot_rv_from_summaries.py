"""Tests for summary-only RV data plots (no network)."""

import json
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


def test_build_plot_uses_fit_json_observability_fallback(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_111_summary.txt"
    summ.write_text(
        "[PIPELINE RESULTS]\n# hdr\n"
        "Gaia_DR3_111_epoch_1.txt 60001 -10.0 0.02 0.4 False\n"
    )
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "111_keplerian_fit.json").write_text(
        json.dumps(
            {
                "observability_window": {
                    "circumpolar": False,
                    "windows": [
                        {
                            "start_date": "2026-08-14",
                            "end_date": "2027-05-24",
                            "start_mjd": 61200.0,
                            "end_mjd": 61400.0,
                        }
                    ],
                }
            }
        )
    )
    obs = rvplot.observability_for_plot("111", summ, None, reports)
    assert obs is not None
    assert obs["windows"][0]["start_date"] == "2026-08-14"


def test_build_plot_zero_epochs_with_coords(tmp_path: Path) -> None:
    from darkhunter_rv.lick_twilight_cache import build_cache_years

    lick = tmp_path / "lick_twilight_cache.json"
    build_cache_years([2026], cache_path=lick)
    summ = tmp_path / "Gaia_DR3_111_summary.txt"
    summ.write_text("[GAIA METADATA]\nSource_ID: 111\nRA: 120.0\nDec: 20.0\n\n[PIPELINE RESULTS]\n# hdr\n")
    out_png = tmp_path / "Gaia_DR3_111_rv_plot.png"
    assert rvplot.build_plot(summ, out_png, lick_cache=lick) is True
    assert out_png.is_file()
    report = rvplot.minimal_report([], summary_path=summ, obs_cache=None, reports_dir=None, lick_cache=lick)
    assert report.get("observability_window") is not None


def test_build_plot_single_epoch(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_111_summary.txt"
    _write_summary(summ, n_epochs=1)
    out_png = tmp_path / "Gaia_DR3_111_rv_plot.png"
    assert rvplot.build_plot(summ, out_png) is True
    assert out_png.is_file()
