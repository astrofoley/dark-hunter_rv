"""Small-angle cone filtering for catalog cross-matches."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def angular_sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Small-angle separation in degrees."""
    dra = float(ra1) - float(ra2)
    ddec = float(dec1) - float(dec2)
    cos_dec = np.cos(np.radians(float(dec1)))
    return float(np.hypot(dra, ddec * cos_dec))


def filter_rows_in_cone(
    rows: Sequence[dict[str, Any]],
    ra_deg: float,
    dec_deg: float,
    *,
    ra_key: str = "target_ra",
    dec_key: str = "target_dec",
    radius_deg: float,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        try:
            tra = float(row[ra_key])
            tdec = float(row[dec_key])
        except (KeyError, TypeError, ValueError):
            continue
        sep = angular_sep_deg(ra_deg, dec_deg, tra, tdec)
        if sep <= radius_deg:
            tagged = dict(row)
            tagged["sep_deg"] = sep
            kept.append(tagged)
    kept.sort(key=lambda r: float(r.get("sep_deg", np.inf)))
    return kept
