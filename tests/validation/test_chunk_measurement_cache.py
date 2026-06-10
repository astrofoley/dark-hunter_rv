"""Tests for chunk campaign measurement cache deduplication."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from validation.chunk_layout import build_edge_preset_layout, build_equal_subchunk_layout
from validation.chunk_measurement_cache import (
    CACHE_COLUMNS,
    find_cache_layout_collisions,
    ingest_diagnostics_csv,
    rebuild_cache_from_diagnostics,
)


def _write_mini_diagnostics(path: Path, *, file_label: str, chunk_key: str, rv: float) -> None:
    df = pd.DataFrame(
        [
            {
                "file": file_label,
                "chunk_key": chunk_key,
                "method": "mask_ccf",
                "rv_kms": rv,
                "rv_err_kms": 0.1,
                "mjd": 60000.0,
                "teff": 5500.0,
                "qc_pass": True,
            }
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_ingest_keeps_subchunks_3_and_n3_equal_separately(tmp_path: Path) -> None:
    """subchunks_3 and n3_equal share equal pixel edges but must not clobber each other."""
    layout_a = build_equal_subchunk_layout(3)
    layout_b = build_edge_preset_layout(3, "equal")
    assert layout_a.name == "subchunks_3"
    assert layout_b.name == "n3_equal"

    diag_a = tmp_path / "diagnostics" / "subchunks_3" / "Gaia_DR3_1_epoch_1_diagnostics.csv"
    diag_b = tmp_path / "diagnostics" / "n3_equal" / "Gaia_DR3_1_epoch_1_diagnostics.csv"
    _write_mini_diagnostics(diag_a, file_label="/data/epoch_1.txt", chunk_key="9_0", rv=10.0)
    _write_mini_diagnostics(diag_b, file_label="/data/epoch_1.txt", chunk_key="9_0", rv=20.0)

    cache = pd.DataFrame(columns=CACHE_COLUMNS)
    cache = ingest_diagnostics_csv(diag_a, layout=layout_a, cache=cache)
    cache = ingest_diagnostics_csv(diag_b, layout=layout_b, cache=cache)

    assert len(cache) == 2
    by_layout = cache.set_index("layout_name")["rv_kms"].to_dict()
    assert by_layout["subchunks_3"] == 10.0
    assert by_layout["n3_equal"] == 20.0
    # Same measurement_id under two layouts is expected (equal edges, distinct layout_name).
    shared = find_cache_layout_collisions(cache)
    assert len(shared) == 1
    assert set(shared.iloc[0]["layout_names"]) == {"n3_equal", "subchunks_3"}


def test_rebuild_cache_from_diagnostics(tmp_path: Path) -> None:
    from validation.chunk_layout import save_chunk_layout

    layout = build_equal_subchunk_layout(2)
    save_chunk_layout(layout, tmp_path / "layouts" / "subchunks_2.yaml")
    _write_mini_diagnostics(
        tmp_path / "diagnostics" / "subchunks_2" / "Gaia_DR3_99_epoch_1_diagnostics.csv",
        file_label="/data/epoch_1.txt",
        chunk_key="12_0",
        rv=5.0,
    )
    cache = rebuild_cache_from_diagnostics(tmp_path)
    assert len(cache) == 1
    assert cache.iloc[0]["layout_name"] == "subchunks_2"
    assert cache.iloc[0]["rv_kms"] == 5.0
