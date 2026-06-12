from pathlib import Path

from scripts.build_hbeta_website_plots import _find_spectrum


def test_find_spectrum_prefers_gaia_subdir(tmp_path: Path) -> None:
    spec_root = tmp_path / "apf_reductions"
    star_a = spec_root / "Gaia_DR3_111111111111111111"
    star_b = spec_root / "Gaia_DR3_222222222222222222"
    star_a.mkdir(parents=True)
    star_b.mkdir(parents=True)
    name = "Gaia_DR3_111111111111111111_epoch_1.txt"
    (star_a / name).write_text("x")
    (star_b / "Gaia_DR3_222222222222222222_epoch_1.txt").write_text("y")

    hit = _find_spectrum(name, [spec_root], gaia_id="111111111111111111")
    assert hit == star_a / name


def test_find_spectrum_absolute_path(tmp_path: Path) -> None:
    spec = tmp_path / "Gaia_DR3_1_epoch_2.txt"
    spec.write_text("x")
    hit = _find_spectrum(str(spec), [tmp_path], gaia_id="1")
    assert hit == spec.resolve()


def test_find_spectrum_skips_order_extract(tmp_path: Path) -> None:
    spec_root = tmp_path / "apf_reductions"
    star = spec_root / "Gaia_DR3_1"
    star.mkdir(parents=True)
    order_only = star / "Gaia_DR3_1_epoch_1_order_28.txt"
    order_only.write_text("x")
    assert _find_spectrum(order_only.name, [spec_root], gaia_id="1") is None
