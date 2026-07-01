#!/usr/bin/env python3
"""
Build per-order sinc² blaze calibration from many high-S/N campaign spectra.

Writes ``calibration/blaze_orders_apf.json`` (or ``--calibration-out``).

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python3 -m validation.build_blaze_calibration \\
    --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \\
    --overlap-csv validation_output/template_fft_baseline/overlap/overlap_enriched_per_exposure.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from darkhunter_rv import blaze, config, instruments

logger = logging.getLogger(__name__)


def _load_overlap_index(overlap_csv: Path | None) -> dict[str, dict]:
    if overlap_csv is None or not overlap_csv.is_file():
        return {}
    df = pd.read_csv(overlap_csv)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        stem = Path(str(row.get("basename", row.get("diagnostics_csv", "")))).stem.replace(
            "_diagnostics", ""
        )
        if stem:
            out[stem] = row.to_dict()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build per-order APF blaze calibration")
    ap.add_argument("--spectrum-list", type=Path, required=True)
    ap.add_argument("--overlap-csv", type=Path, default=None)
    ap.add_argument(
        "--calibration-out",
        type=Path,
        default=config.BLAZE_CALIBRATION_FILE,
    )
    ap.add_argument("--instrument", default="APF")
    ap.add_argument("--min-snr", type=float, default=3.5)
    ap.add_argument("--min-profiles", type=int, default=8)
    ap.add_argument("--line-mask-half-width", type=float, default=22.0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    spec_paths = [
        Path(ln.strip())
        for ln in args.spectrum_list.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    overlap = _load_overlap_index(args.overlap_csv)
    inst = instruments.get_instrument_profile(args.instrument)

    calib = blaze.build_blaze_calibration(
        spec_paths,
        inst,
        min_snr=float(args.min_snr),
        overlap=overlap,
        line_mask_half_width=float(args.line_mask_half_width),
        min_profiles=int(args.min_profiles),
    )
    calib.save(args.calibration_out)
    logger.info(
        "Wrote %s (%d orders fit from %d spectra, min_snr=%.1f)",
        args.calibration_out.resolve(),
        len(calib.orders),
        calib.n_spectra_fit,
        args.min_snr,
    )
    return 0 if calib.orders else 1


if __name__ == "__main__":
    raise SystemExit(main())
