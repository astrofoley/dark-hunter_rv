"""Calibration manifest JSON helpers."""

from pathlib import Path

from darkhunter_rv.calibration_manifest import load_manifest, new_manifest, save_manifest


def test_manifest_roundtrip(tmp_path):
    m = new_manifest(instrument="APF", repo_root=tmp_path)
    m["bias_phase"]["spectrum_paths"] = ["/a/b.txt"]
    p = tmp_path / "m.json"
    save_manifest(p, m)
    m2 = load_manifest(p)
    assert m2["version"] == 1
    assert m2["instrument"] == "APF"
    assert m2["bias_phase"]["spectrum_paths"] == ["/a/b.txt"]
