"""Tests for validation.benchmark_cool_precision."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from validation.benchmark_cool_precision import _load_exposure_rows, _summarize


def _write_diag(path: Path, teff: float, snr: float, scatter: float) -> None:
    rows = [
        {
            "file": "Gaia_DR3_1_epoch_1.txt",
            "chunk_key": "12",
            "method": "mask_ccf",
            "rv_kms": 0.0,
            "rv_err_kms": 0.05,
            "qc_pass": True,
            "teff": teff,
            "ccf_peak_snr": snr,
            "used_in_exposure_stack": True,
            "chunk_scatter_kms": scatter,
        },
        {
            "file": "Gaia_DR3_1_epoch_1.txt",
            "chunk_key": "all",
            "method": "mask_ccf",
            "rv_kms": 0.0,
            "rv_err_kms": 0.08,
            "qc_pass": True,
            "teff": teff,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_summarize_high_snr_subset(tmp_path: Path) -> None:
    p = tmp_path / "a_diagnostics.csv"
    _write_diag(p, teff=5000.0, snr=15.0, scatter=0.08)
    tab = _load_exposure_rows([str(p)])
    assert len(tab) == 1
    s = _summarize(tab, min_log10_snr=1.0)
    assert s["n"] == 1.0
    assert s["median_chunk_scatter_kms"] == 0.08
    assert s["median_chunk_scatter_kms"] < 0.1
