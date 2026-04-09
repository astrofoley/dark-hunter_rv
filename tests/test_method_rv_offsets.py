"""Method RV offset table (mask truth) and joint calibration math."""
from __future__ import annotations

import numpy as np
import pytest

from darkhunter_rv.io_utils import read_method_rv_offsets, write_method_rv_offsets
from validation.compute_method_rv_offsets import joint_calibration_from_arrays


def test_read_write_method_rv_offsets_roundtrip(tmp_path):
    p = tmp_path / "method_rv_offsets.txt"
    write_method_rv_offsets(
        p,
        [
            {
                "instrument": "APF",
                "offset_template_fft_kms": -1.4,
                "offset_strong_lines_kms": -2.52,
                "n_exposures_joint": 150,
                "estimator": "median",
            }
        ],
        comment_lines=["# test header"],
    )
    d = read_method_rv_offsets(p)
    assert "APF" in d
    assert d["APF"]["offset_template_fft_kms"] == pytest.approx(-1.4)
    assert d["APF"]["offset_strong_lines_kms"] == pytest.approx(-2.52)
    assert d["APF"]["n_exposures_joint"] == 150
    assert d["APF"]["estimator"] == "median"


def test_joint_calibration_median_aligns_template_and_strong_to_mask():
    # Teff / log10_snr in default overlap for mask ∩ template ∩ strong_lines
    n = 5
    teff = np.full(n, 6000.0)
    log10_snr = np.full(n, 1.0)
    rm = np.zeros(n)
    rt = np.full(n, -1.0)
    rsl = np.full(n, -3.0)
    err = np.full(n, 1.0)
    out = joint_calibration_from_arrays(
        rm,
        rt,
        rsl,
        err,
        err,
        err,
        teff,
        log10_snr,
        max_sigma_kms=2.5,
        apply_method_regions=True,
        estimator="median",
    )
    assert out["offset_template_fft_kms"] == pytest.approx(1.0)
    assert out["offset_strong_lines_kms"] == pytest.approx(3.0)
    assert out["n_joint"] == n
    assert out["post_median_mask_minus_template"] == pytest.approx(0.0)
    assert out["post_median_mask_minus_strong"] == pytest.approx(0.0)
    assert out["post_median_template_minus_strong"] == pytest.approx(0.0)


def test_joint_calibration_mean():
    rm = np.array([0.0, 2.0])
    rt = np.array([-1.0, 1.0])
    rsl = np.array([-2.0, 0.0])
    err = np.ones(2)
    teff = np.array([6000.0, 6000.0])
    log10_snr = np.array([1.0, 1.0])
    out = joint_calibration_from_arrays(
        rm,
        rt,
        rsl,
        err,
        err,
        err,
        teff,
        log10_snr,
        max_sigma_kms=2.5,
        apply_method_regions=False,
        estimator="mean",
    )
    assert out["offset_template_fft_kms"] == pytest.approx(1.0)
    assert out["offset_strong_lines_kms"] == pytest.approx(2.0)
