from pathlib import Path

from scripts.ensure_pipeline_summaries import _needs_summary


def test_needs_summary_when_missing(tmp_path: Path) -> None:
    spec = tmp_path / "spec"
    out = tmp_path / "out"
    gid = "77413727493690112"
    star = spec / f"Gaia_DR3_{gid}"
    star.mkdir(parents=True)
    (star / f"Gaia_DR3_{gid}_epoch_1.txt").write_text("spectrum")
    need, _path, files = _needs_summary(gid, spec_root=spec, out_dir=out)
    assert need is True
    assert len(files) == 1
