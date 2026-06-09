"""Tests for chunk bias regression and layout utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darkhunter_rv.chunking import iter_order_chunks_with_edges
from validation.chunk_bias_lib import fit_linear_model, truncated_cubic_basis
from validation.chunk_bias_regression import _fit_model, choose_best_model, compare_models
from validation.chunk_layout import ChunkLayout, load_chunk_layout, map_chunk_key, rebinned_chunk_rows


def test_truncated_cubic_basis_shape() -> None:
    x = np.linspace(0, 1, 10)
    B = truncated_cubic_basis(x, np.array([0.25, 0.5, 0.75]))
    assert B.shape == (10, 5)


def test_iter_order_chunks_with_edges() -> None:
    spec = {5: {"wavelength": list(range(40)), "flux": [1.0] * 40, "eflux": [1.0] * 40}}
    chunks = list(iter_order_chunks_with_edges(spec, bad_orders=[], pixel_edges=[0.0, 0.5, 1.0]))
    assert [k for k, *_ in chunks] == ["5_0", "5_1"]
    assert sum(len(w) for _, w, *_ in chunks) == 40


def test_map_chunk_key_merge() -> None:
    layout = ChunkLayout(name="m", merge_orders=[[3, 4], [5, 6]])
    assert map_chunk_key("3", layout) == "merge_0"
    assert map_chunk_key("6", layout) == "merge_1"


def test_rebinned_chunk_rows_ivw() -> None:
    df = pd.DataFrame(
        [
            {"file": "a", "gaia_dr3_id": "1", "chunk_key": "3", "rv_kms": 10.0, "rv_err_kms": 0.1, "teff": 5000, "mjd": 1},
            {"file": "a", "gaia_dr3_id": "1", "chunk_key": "4", "rv_kms": 12.0, "rv_err_kms": 0.1, "teff": 5000, "mjd": 1},
        ]
    )
    layout = ChunkLayout(name="m", merge_orders=[[3, 4]])
    out = rebinned_chunk_rows(df, layout)
    assert len(out) == 1
    assert out["chunk_key"].iloc[0] == "merge_0"
    assert out["rv_kms"].iloc[0] == pytest.approx(11.0, abs=0.01)


def test_compare_models_runs_on_synthetic() -> None:
    rng = np.random.default_rng(0)
    n = 80
    order = np.tile(np.arange(10, 20), n // 10)
    df = pd.DataFrame(
        {
            "gaia_dr3_id": np.repeat(np.arange(n // 10), 10).astype(str),
            "chunk_key": order.astype(str),
            "chunk_order": order,
            "chunk_order_norm": (order - order.min()) / (order.max() - order.min()),
            "weighted_mean_residual_kms": 0.5 * (order - order.mean()) / order.std() + rng.normal(0, 0.05, n),
            "statistical_err_kms": np.full(n, 0.1),
            "intrinsic_scatter_kms": np.full(n, 0.05),
            "teff": 5000 + rng.normal(0, 100, n),
            "logg": 4.5 + rng.normal(0, 0.1, n),
            "mh": rng.normal(0, 0.1, n),
            "log10_median_mask_ccf_peak_snr": np.full(n, 1.0),
        }
    )
    cmp = compare_models(df)
    assert "curve" in cmp["model"].values
    chosen = choose_best_model(cmp)
    assert chosen in cmp["model"].values


def test_load_chunk_layout_yaml(tmp_path) -> None:
    p = tmp_path / "t.yaml"
    p.write_text("name: test\nsubchunks: 2\n")
    layout = load_chunk_layout(p)
    assert layout.subchunks == 2
    assert layout.n_chunks_per_order() == 2
