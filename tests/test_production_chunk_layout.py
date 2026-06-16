"""Production chunk layout and bias path defaults."""

from pathlib import Path

from darkhunter_rv import config


def test_default_chunk_layout_exists() -> None:
    assert config.DEFAULT_CHUNK_LAYOUT is not None
    assert config.DEFAULT_CHUNK_LAYOUT.is_file()
    assert config.DEFAULT_CHUNK_LAYOUT.name == "subchunks_8.yaml"


def test_bias_statistics_committed() -> None:
    assert config.BIAS_STATISTICS_FILE.is_file()
    text = config.BIAS_STATISTICS_FILE.read_text()
    assert text.lstrip().startswith("#")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(lines) >= 10
