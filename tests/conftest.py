"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from darkhunter_rv.lick_twilight_cache import build_cache_years, default_cache_path


@pytest.fixture(scope="session", autouse=True)
def ensure_lick_twilight_cache() -> None:
    """Build the Lick twilight JSON once when missing (CI and fresh clones)."""
    cache = default_cache_path()
    if cache.is_file():
        return
    build_cache_years([2025, 2026], cache_path=cache)
