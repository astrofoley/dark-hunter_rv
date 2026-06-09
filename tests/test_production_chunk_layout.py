"""Production chunk layout and bias path defaults."""

from pathlib import Path

from darkhunter_rv import config


def test_default_chunk_layout_exists() -> None:
    assert config.DEFAULT_CHUNK_LAYOUT is not None
    assert config.DEFAULT_CHUNK_LAYOUT.is_file()
    assert config.DEFAULT_CHUNK_LAYOUT.name == "subchunks_4.yaml"


def test_bias_statistics_committed() -> None:
    assert config.BIAS_STATISTICS_FILE.is_file()
    text = config.BIAS_STATISTICS_FILE.read_text()
    assert "Bias_Mean" in text.splitlines()[0]
    assert len(text.splitlines()) >= 10
