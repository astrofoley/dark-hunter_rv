"""Tests for Thiele-Innes → inclination conversion."""

from __future__ import annotations

import numpy as np
import pytest

from darkhunter_rv.thiele_innes_inclination import (
    campbell_to_thiele_innes,
    fill_inclination_in_metadata,
    inclination_deg_from_thiele_innes,
    inclination_from_thiele_innes,
    thiele_innes_from_metadata,
)


@pytest.mark.parametrize(
    "i_deg,omega_deg,Omega_deg",
    [
        (60.0, 45.0, 120.0),
        (120.0, 30.0, 200.0),
        (35.0, 10.0, 50.0),
        (145.0, 210.0, 80.0),
    ],
)
def test_inclination_round_trip(i_deg: float, omega_deg: float, Omega_deg: float) -> None:
    A, B, F, G = campbell_to_thiele_innes(2.5, i_deg, omega_deg, Omega_deg)
    got = inclination_deg_from_thiele_innes(A, B, F, G)
    assert got is not None
    assert got == pytest.approx(i_deg, abs=1e-6)


def test_inclination_error_propagation() -> None:
    A, B, F, G = campbell_to_thiele_innes(1.8, 74.0, 55.0, 130.0)
    ti = thiele_innes_from_metadata(
        {
            "A_Thiele_Innes": A,
            "B_Thiele_Innes": B,
            "F_Thiele_Innes": F,
            "G_Thiele_Innes": G,
            "A_Thiele_Innes_Error": 0.05,
            "B_Thiele_Innes_Error": 0.05,
            "F_Thiele_Innes_Error": 0.05,
            "G_Thiele_Innes_Error": 0.05,
        }
    )
    assert ti is not None
    i_deg, i_err = inclination_from_thiele_innes(ti)
    assert i_deg == pytest.approx(74.0, abs=1e-5)
    assert i_err is not None
    assert i_err > 0
    assert i_err < 30.0


def test_fill_inclination_in_metadata_from_ti() -> None:
    A, B, F, G = campbell_to_thiele_innes(2.0, 60.0, 40.0, 100.0)
    meta = {
        "Inclination": float("nan"),
        "A_Thiele_Innes": A,
        "B_Thiele_Innes": B,
        "F_Thiele_Innes": F,
        "G_Thiele_Innes": G,
        "A_Thiele_Innes_Error": 0.02,
        "B_Thiele_Innes_Error": 0.02,
        "F_Thiele_Innes_Error": 0.02,
        "G_Thiele_Innes_Error": 0.02,
    }
    assert fill_inclination_in_metadata(meta)
    assert meta["Inclination"] == pytest.approx(60.0, abs=1e-5)
    assert "Inclination_Error" in meta


def test_fill_skips_when_database_inclination_present() -> None:
    A, B, F, G = campbell_to_thiele_innes(2.0, 60.0, 40.0, 100.0)
    meta = {
        "Inclination": 55.0,
        "A_Thiele_Innes": A,
        "B_Thiele_Innes": B,
        "F_Thiele_Innes": F,
        "G_Thiele_Innes": G,
    }
    assert not fill_inclination_in_metadata(meta)
    assert meta["Inclination"] == 55.0


def test_lowercase_adql_keys() -> None:
    A, B, F, G = campbell_to_thiele_innes(1.5, 88.0, 20.0, 60.0)
    ti = thiele_innes_from_metadata(
        {
            "a_thiele_innes": A,
            "b_thiele_innes": B,
            "f_thiele_innes": F,
            "g_thiele_innes": G,
        }
    )
    assert ti is not None
    assert inclination_deg_from_thiele_innes(ti.A, ti.B, ti.F, ti.G) == pytest.approx(88.0, abs=1e-5)
