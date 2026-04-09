"""Fixed Teff bins for high-error fraction table (validation.rv_method_diagnostics_report)."""

import numpy as np

from validation.rv_method_diagnostics_report import (
    _binned_high_err_fraction_table,
    _uniform_teff_bin_edges,
)


def test_uniform_teff_edges_width_and_count():
    e = _uniform_teff_bin_edges(4, 4000.0, 8000.0)
    assert len(e) == 5
    assert e[0] == 4000.0 and e[-1] == 8000.0
    assert np.allclose(np.diff(e), 1000.0)


def test_high_err_fraction_table_has_one_row_per_bin():
    teff = np.array([5000.0, 5500.0, 6000.0, float("nan")])
    err = np.array([1.0, 3.0, 1.0, 1.0])
    tab = _binned_high_err_fraction_table(
        teff,
        err,
        err,
        err,
        2.5,
        3,
        teff_bin_lo=4000.0,
        teff_bin_hi=7000.0,
    )
    assert len(tab) == 3
    assert list(tab["bin_lo"]) == [4000.0, 5000.0, 6000.0]
