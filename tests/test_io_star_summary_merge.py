"""Star summary merges pipeline rows across per-spectrum runs."""
from pathlib import Path

import pytest

from darkhunter_rv import config, io_utils


def test_write_star_summary_merges_epochs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    oid = 9999999999999999999
    gaia_data = {"metadata": {"Teff": 5800.0}, "external_rvs": []}

    io_utils.write_star_summary(
        oid,
        gaia_data,
        [
            {
                "file": "/fake/a_epoch_1.txt",
                "mjd": 60000.0,
                "rv": 1.0,
                "rv_err": 0.1,
                "rv_rms": 0.5,
                "fallback": False,
            }
        ],
    )
    io_utils.write_star_summary(
        oid,
        gaia_data,
        [
            {
                "file": "/other/b_epoch_2.txt",
                "mjd": 60001.0,
                "rv": 2.0,
                "rv_err": 0.2,
                "rv_rms": 0.6,
                "fallback": True,
            }
        ],
    )

    text = (tmp_path / f"Gaia_DR3_{oid}_summary.txt").read_text()
    assert "a_epoch_1.txt" in text
    assert "b_epoch_2.txt" in text
    assert text.index("epoch_1") < text.index("epoch_2")
