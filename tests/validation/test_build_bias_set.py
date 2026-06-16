"""Tests for validation.build_bias_set (mask debias table)."""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_rv import config, io_utils
from validation import build_bias_set


def _write_orders(path: Path, lines: list[str]) -> None:
    path.write_text("# Order | RV | Err\n" + "\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.validation
def test_build_bias_filters_spectrum_stems(tmp_path: Path) -> None:
    inp = tmp_path / "orders"
    inp.mkdir()
    _write_orders(inp / "star_a_epoch_1_orders.txt", ["10_0 10.0 1.0", "10_1 10.2 1.0"])
    _write_orders(inp / "star_b_epoch_1_orders.txt", ["10_0 50.0 1.0", "10_1 50.4 1.0"])
    out = tmp_path / "out"
    summary = build_bias_set.build_bias(
        inp,
        n_boot=20,
        out_dir=out,
        spectrum_stems={"star_a_epoch_1"},
    )
    assert summary["n_files"] == 1
    text = (out / "bias_statistics.txt").read_text()
    assert "10_0" in text
    for line in text.splitlines():
        if line.startswith("10_0 "):
            b0 = float(line.split()[1])
            assert abs(b0) < 2.0


@pytest.mark.validation
def test_build_bias_sigma_clips_bad_chunks(tmp_path: Path) -> None:
    inp = tmp_path / "orders"
    inp.mkdir()
    lines = [f"20_{i} {v:.1f} 1.0" for i, v in enumerate([-40.0] * 7 + [120.0])]
    _write_orders(inp / "star_a_epoch_1_orders.txt", lines)
    _write_orders(inp / "star_b_epoch_1_orders.txt", ["20_0 -40.4 1.0", "20_1 -40.2 1.0"])
    out = tmp_path / "out"
    build_bias_set.build_bias(inp, n_boot=20, out_dir=out)
    b0 = None
    for line in (out / "bias_statistics.txt").read_text().splitlines():
        if line.startswith("20_0 "):
            b0 = float(line.split()[1])
    assert b0 is not None
    assert abs(b0) < 2.0


@pytest.mark.validation
def test_build_bias_writes_per_chunk_keys(tmp_path: Path) -> None:
    inp = tmp_path / "orders"
    inp.mkdir()
    _write_orders(
        inp / "star_a_epoch_1_orders.txt",
        ["10_0 10.0 1.0", "10_1 10.5 1.0", "11_0 10.1 1.0"],
    )
    _write_orders(
        inp / "star_b_epoch_1_orders.txt",
        ["10_0 10.2 1.0", "10_1 10.7 1.0", "11_0 10.3 1.0"],
    )
    out = tmp_path / "out"
    build_bias_set.build_bias(inp, n_boot=20, out_dir=out)
    keys = {
        line.split()[0]
        for line in (out / "bias_statistics.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert "10_0" in keys
    assert "10_1" in keys
    assert "11_0" in keys
    assert "10" not in keys


def test_lookup_bias_prefers_chunk_over_order() -> None:
    bias = {10: [9.0, 0.0, 0.0], "10_1": [1.0, 0.0, 0.0]}
    assert io_utils.lookup_bias(bias, "10_1")[0] == 1.0
    assert io_utils.lookup_bias(bias, "10_7")[0] == 9.0


def test_committed_bias_statistics_sane() -> None:
    """Repo-root debias table uses chunk keys and reasonable per-chunk scale."""
    path = config.BIAS_STATISTICS_FILE
    assert path.is_file()
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(lines) >= 50
    for ln in lines:
        parts = ln.split()
        assert len(parts) == 4
        assert "_" in parts[0] or parts[0].isdigit()
        b0, b1, b2 = (float(parts[1]), float(parts[2]), float(parts[3]))
        assert abs(b0) < 5.0
        assert b1 < 10.0
        assert b2 < 3.0
