"""Tests for staging plot PNGs into the website tree."""

from pathlib import Path

from darkhunter_rv.website_plot_sync import (
    maybe_stage_gaia_plots,
    resolve_web_root,
    stage_gaia_plots,
    website_plots_dir,
)


def test_stage_gaia_plots_copies_rv_and_residuals(tmp_path: Path) -> None:
    gid = "1978377080333206528"
    src = tmp_path / "output" / f"Gaia_DR3_{gid}"
    src.mkdir(parents=True)
    web = tmp_path / "web"
    reports = tmp_path / "reports"
    reports.mkdir()
    (src / f"Gaia_DR3_{gid}_rv_plot.png").write_bytes(b"rv")
    (reports / f"{gid}_keplerian_residuals.png").write_bytes(b"res")

    n = stage_gaia_plots(gid, src, web, reports_dir=reports)
    dest = website_plots_dir(web, gid)

    assert n == 2
    assert (dest / f"Gaia_DR3_{gid}_rv_plot.png").read_bytes() == b"rv"
    assert (dest / f"Gaia_DR3_{gid}_keplerian_residuals.png").read_bytes() == b"res"


def test_resolve_web_root_honors_no_sync(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WEB_ROOT", str(tmp_path))
    assert resolve_web_root(None, sync_enabled=False) is None
    assert resolve_web_root(None, sync_enabled=True) == tmp_path


def test_plot_script_stages_after_build(tmp_path: Path, monkeypatch) -> None:
    from scripts import plot_rv_from_summaries as rvplot

    gid = "111"
    summ = tmp_path / f"Gaia_DR3_{gid}_summary.txt"
    summ.write_text("[GAIA METADATA]\nSource_ID: 111\nRA: 120.0\nDec: 20.0\n\n[PIPELINE RESULTS]\n# hdr\n")
    plots_root = tmp_path / "output"
    web = tmp_path / "web"
    out_png = plots_root / f"Gaia_DR3_{gid}" / f"Gaia_DR3_{gid}_rv_plot.png"

    monkeypatch.setenv("WEB_ROOT", str(web))
    assert rvplot.build_plot(summ, out_png) is True
    n = maybe_stage_gaia_plots(gid, plots_root / f"Gaia_DR3_{gid}", web_root=web)
    assert n == 1
    assert (website_plots_dir(web, gid) / f"Gaia_DR3_{gid}_rv_plot.png").is_file()
