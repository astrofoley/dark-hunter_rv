"""Rotational broadening grid extent for fast rotators (wide PHOENIX banks)."""

from darkhunter_rv.templates import _broadening_velocity_grid


def test_wide_grid_extends_with_high_vsini_proxy():
    g = _broadening_velocity_grid(100.0, wide=True)
    assert max(g) >= 100.0
    assert min(g) == 0.0


def test_wide_grid_cap_allows_fast_rotator_bank():
    g = _broadening_velocity_grid(140.0, wide=True)
    assert max(g) >= 130.0
