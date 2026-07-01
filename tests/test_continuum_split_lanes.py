"""Split continuum lanes: mask blaze-only vs template blaze+spline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from darkhunter_rv import config, pipeline
from darkhunter_rv.blaze import BlazeCalibration


def _args_with_calib() -> argparse.Namespace:
    cal = BlazeCalibration(instrument="APF", n_spectra_fit=10, min_snr=3.5, orders={})
    return argparse.Namespace(
        continuum_mode="split",
        no_blaze_continuum=False,
        blaze_calibration=cal,
    )


def test_split_lanes_with_calibration():
    args = _args_with_calib()
    assert pipeline._resolve_continuum_mode(args, "mask") == config.MASK_CONTINUUM_MODE
    assert pipeline._resolve_continuum_mode(args, "template") == config.TEMPLATE_CONTINUUM_MODE
    assert pipeline._resolve_continuum_mode(args, "strong") == config.TEMPLATE_CONTINUUM_MODE


def test_split_without_calibration_falls_back_to_spline():
    args = argparse.Namespace(continuum_mode="split", no_blaze_continuum=False, blaze_calibration=None)
    assert pipeline._resolve_continuum_mode(args, "mask") == "spline"


def test_no_blaze_continuum_forces_spline():
    args = _args_with_calib()
    args.no_blaze_continuum = True
    assert pipeline._resolve_continuum_mode(args, "mask") == "spline"
    assert pipeline._resolve_continuum_mode(args, "template") == "spline"


def test_uniform_sinc_blaze_only_mode():
    args = _args_with_calib()
    args.continuum_mode = "sinc_blaze_only"
    assert pipeline._resolve_continuum_mode(args, "template") == "sinc_blaze_only"
