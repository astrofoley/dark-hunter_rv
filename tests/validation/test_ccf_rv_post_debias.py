"""Post-debias σ_RV stack tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation.ccf_rv_post_debias import stack_post_debias_exposures


def test_post_debias_sigma_smaller_than_raw_scatter(tmp_path) -> None:
    """After debias, exposure σ_RV should be modest when chunk biases are structured."""
    rng = np.random.default_rng(2)
    rows = []
    for star_i in range(5):
        gid = str(100 + star_i)
        for epoch_i, fid in enumerate([f"a{star_i}.txt", f"b{star_i}.txt", f"c{star_i}.txt"]):
            for ck, bias in [("10_0", 50.0), ("10_1", 52.0), ("11_0", 48.0), ("11_1", 51.0)]:
                rv = bias + rng.normal(0, 0.05)
                rows.append(
                    {
                        "file": fid,
                        "gaia_dr3_id": gid,
                        "chunk_key": ck,
                        "chunk_order": int(ck.split("_")[0]) * 100 + int(ck.split("_")[1]),
                        "teff": 5000.0 + star_i * 150.0,
                        "logg": 4.5,
                        "mh": 0.0,
                        "log10_peak_snr": 1.0,
                        "rv_kms__gauss_offset": rv,
                        "rv_err_kms__gauss_offset": 0.08,
                        "peak_snr": 10.0,
                    }
                )
    wide = pd.DataFrame(rows)
    summary_dir = tmp_path / "output"
    summary_dir.mkdir()
    for star_i in range(5):
        gid = str(100 + star_i)
        (summary_dir / f"Gaia_DR3_{gid}_summary.txt").write_text(
            f"Teff: {5000 + star_i * 150}\nlogg: 4.5\n[M/H]: 0.0\n",
            encoding="utf-8",
        )

    epochs, summary = stack_post_debias_exposures(
        wide,
        "gauss_offset",
        summary_dir,
        min_chunks=3,
    )
    assert len(epochs) >= 2
    assert summary["median_sigma_rv_kms"] < 0.5
    assert summary["median_chunk_scatter_debiased_kms"] < 5.0
