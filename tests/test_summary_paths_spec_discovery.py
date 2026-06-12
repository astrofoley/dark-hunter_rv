from pathlib import Path

from darkhunter_rv.summary_paths import (
    count_valid_pipeline_rv_epochs,
    discover_primary_epoch_files,
    discover_spec_gaia_ids,
    is_primary_epoch_spectrum_name,
)


def test_discover_spec_gaia_ids_primary_epochs_only(tmp_path: Path) -> None:
    root = tmp_path / "apf_reductions"
    star = root / "Gaia_DR3_111111111111111111"
    star.mkdir(parents=True)
    (star / "Gaia_DR3_111111111111111111_epoch_1.txt").write_text("x")
    (star / "Gaia_DR3_111111111111111111_epoch_1_order_28.txt").write_text("x")
    nested = root / "batch" / "Gaia_DR3_222222222222222222"
    nested.mkdir(parents=True)
    (nested / "Gaia_DR3_222222222222222222_epoch_2.txt").write_text("x")

    ids = discover_spec_gaia_ids(root)
    assert ids == {"111111111111111111", "222222222222222222"}
    assert len(discover_primary_epoch_files(root, "111111111111111111")) == 1


def test_is_primary_epoch_spectrum_name() -> None:
    assert is_primary_epoch_spectrum_name("Gaia_DR3_77413727493690112_epoch_1.txt")
    assert not is_primary_epoch_spectrum_name("Gaia_DR3_77413727493690112_epoch_1_order_28.txt")
    assert not is_primary_epoch_spectrum_name("Gaia_DR3_77413727493690112_epoch_10_order_28.txt")


def test_count_valid_pipeline_rv_epochs(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_1_summary.txt"
    summ.write_text(
        "### STAR SUMMARY ###\n\n[PIPELINE RESULTS]\n"
        "Gaia_DR3_1_epoch_1.txt 60532.5 -1.6 0.01 0.01 False\n"
        "Gaia_DR3_1_epoch_2.txt 60555.4 -9999.0 0.01 0.01 False\n"
    )
    assert count_valid_pipeline_rv_epochs(summ) == 1
