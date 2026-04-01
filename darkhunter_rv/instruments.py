# instruments.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config


@dataclass
class InstrumentProfile:
    name: str
    file_format: str
    num_orders: int
    bias_file: Optional[str]
    bad_orders: List[int]
    mask_directory: str
    header_keywords: Dict[str, str] = field(default_factory=dict)


INSTRUMENTS = {
    "APF": InstrumentProfile(
        name="APF",
        file_format="txt",
        num_orders=70,
        bias_file=str(config.REPO_ROOT / "bias_statistics.txt"),
        bad_orders=[0, 1, 2, 53, 57, 58, 59, 60, 63, 64, 65],
        mask_directory=str(config.MASK_DIRECTORY),
        header_keywords={"mjd": "# THEMIDPT"},
    ),
    "GHOST": InstrumentProfile(
        name="GHOST",
        file_format="fits",
        num_orders=61,
        bias_file=None,
        bad_orders=[],
        mask_directory=str(config.MASK_DIRECTORY),
        header_keywords={"bjd": "BJD"},
    ),
    "MAROON-X": InstrumentProfile(
        name="MAROON-X",
        file_format="hd5",
        num_orders=61,
        bias_file=None,
        bad_orders=[],
        mask_directory=str(config.MASK_DIRECTORY),
        header_keywords={"jd": "JD_UTC_FLUXWEIGHTED_FRD"},
    ),
}


def get_instrument_profile(name: str) -> InstrumentProfile:
    if name not in INSTRUMENTS:
        raise ValueError(
            f"Instrument '{name}' not recognized. Available: {list(INSTRUMENTS.keys())}"
        )
    return INSTRUMENTS[name]
