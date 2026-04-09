import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "validation") not in sys.path:
    sys.path.insert(0, str(_REPO / "validation"))


def _load_diagnose():
    path = _REPO / "validation" / "diagnose_legacy_campaign.py"
    spec = importlib.util.spec_from_file_location("diagnose_legacy_campaign", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_summary_file(tmp_path):
    mod = _load_diagnose()
    p = tmp_path / "x_summary.txt"
    p.write_text(
        "# File Summary\n"
        "# Input File | MJD | RV | Err | RMS\n"
        "Gaia_DR3_1_epoch_1.txt 60613.0 -11.95 0.01 0.05\n"
    )
    rows = mod.parse_summary_file(p)
    assert len(rows) == 1
    assert rows[0]["basename"] == "Gaia_DR3_1_epoch_1.txt"
    assert rows[0]["rv"] == pytest.approx(-11.95)


def test_parse_summary_pipeline_results_block(tmp_path):
    mod = _load_diagnose()
    p = tmp_path / "star.txt"
    p.write_text(
        "### HEADER ###\n"
        "[PIPELINE RESULTS]\n"
        "# File | MJD | RV\n"
        "../data/Gaia_DR3_1_epoch_9.txt 60727.55 -10.416 0.020 0.534 False\n"
    )
    rows = mod.parse_summary_file(p)
    assert len(rows) == 1
    assert rows[0]["basename"] == "../data/Gaia_DR3_1_epoch_9.txt"
    by = mod.summary_rows_by_basename(rows)
    assert "Gaia_DR3_1_epoch_9.txt" in by
    assert by["Gaia_DR3_1_epoch_9.txt"]["rv"] == pytest.approx(-10.416)


def test_apply_legacy_filters():
    mod = _load_diagnose()
    rows = [
        {"rv": 10.0, "rv_err": 0.01, "rms": 0.02, "basename": "a"},
        {"rv": 500.0, "rv_err": 0.01, "rms": 0.02, "basename": "b"},
    ]
    out = mod.apply_legacy_filters(rows, max_abs_rv=200.0, max_err=0.5, max_rms=0.5)
    assert out[0]["passes_legacy_filter"] is True
    assert out[1]["passes_legacy_filter"] is False


def test_compute_method_pair_table():
    from method_pair_stats import compute_method_pair_table

    df = pd.DataFrame(
        {
            "file": ["f1", "f1", "f1", "f1"],
            "chunk_key": ["a", "a", "b", "b"],
            "method": ["mask_ccf", "template_fft", "mask_ccf", "template_fft"],
            "rv_kms": [1.0, 1.1, 2.0, 2.1],
        }
    )
    t = compute_method_pair_table(df)
    assert len(t) == 1
    assert t.iloc[0]["method_a"] == "mask_ccf"
    assert t.iloc[0]["median_offset_kms"] == pytest.approx(-0.1)
    assert t.iloc[0]["n"] == 2
