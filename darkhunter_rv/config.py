"""Paths and constants; override with env vars for portability."""

from __future__ import annotations

import os
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PKG_ROOT.parent

MASK_DIRECTORY = Path(os.environ.get("DARKHUNTER_MASK_DIR", REPO_ROOT / "stellar_masks"))
OUTPUT_DIR = Path(os.environ.get("DARKHUNTER_OUTPUT_DIR", REPO_ROOT / "output"))
PLOT_DIR = Path(os.environ.get("DARKHUNTER_PLOT_DIR", REPO_ROOT / "plots"))

_phoenix = os.environ.get("DARKHUNTER_PHOENIX_DIR")
PHOENIX_BASE_DIR = Path(_phoenix) if _phoenix else REPO_ROOT / "phoenix_models"

C_KMS = 299_792.458
DEFAULT_TEFF = 5800.0
HOT_STAR_TEFF_THRESHOLD = 6500.0
