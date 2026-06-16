#!/usr/bin/env python3
"""
Run template-FFT failure-mode diagnostics for every spectral chunk.

Requires a local PHOENIX tree (``DARKHUNTER_PHOENIX_DIR``). Example::

  python validation/diagnose_template_fft_star.py \\
    --spectrum data/Gaia_DR3_468391369318487040_epoch_1.txt \\
    --instrument APF --teff 5117 --logg 2.77 --mh -0.1 \\
    --output-jsonl validation_output/fft_diag_468391369318487040.jsonl

Optional ``--diagnostics-csv`` merges mask_ccf RV per chunk as ``rv_truth_kms`` (same row chunk_key).

This script only writes JSONL (no figures). For a **production-equivalent** run with
``*_diagnostics.csv``, summary files, and PNGs, use ``validation/run_reference_pipeline.py`` or
``python -m darkhunter_rv ... --run-all-methods --plots``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import chunking, config, continuum, fft_diagnostics, instruments, io_utils, rv_core, templates


def _continuum_kw(hot: bool) -> dict:
    return {
        "continuum_mode": "spline",
        "exclude_near_lines_width": float(
            config.HOT_SPLINE_EXCLUDE_NEAR_LINES_WIDTH if hot else config.COOL_SPLINE_EXCLUDE_NEAR_LINES_WIDTH
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-chunk template FFT failure diagnostics")
    ap.add_argument("--spectrum", type=Path, required=True)
    ap.add_argument("--instrument", default="APF")
    ap.add_argument("--teff", type=float, required=True)
    ap.add_argument("--logg", type=float, default=4.5)
    ap.add_argument("--mh", type=float, default=0.0)
    ap.add_argument("--subchunks", type=int, default=1)
    ap.add_argument("--template-grid-wide", action="store_true")
    ap.add_argument("--diagnostics-csv", type=Path, default=None, help="Pipeline diagnostics CSV for mask RV truth")
    ap.add_argument("--no-bias", action="store_true", help="Do not add per-order barycentric b0 to mask truth")
    ap.add_argument("--output-jsonl", type=Path, required=True)
    args = ap.parse_args()

    inst = instruments.get_instrument_profile(args.instrument)
    hot = args.teff > config.HOT_STAR_TEFF_THRESHOLD
    _, spec_data = io_utils.read_spectrum(str(args.spectrum))

    bank_probe = templates.build_template_bank_cached(
        float(args.teff),
        50.0,
        metallicity=float(args.mh),
        logg=float(args.logg),
        hot_spectrum=hot,
        template_grid_wide=bool(args.template_grid_wide),
    )
    vsini = 10.0
    if bank_probe:
        tw, tf = next(iter(bank_probe.values()))
        all_w, all_f = [], []
        valid_orders = sorted(o for o in spec_data if o not in inst.bad_orders)
        for o in valid_orders:
            w = np.array(spec_data[o]["wavelength"], float)
            f = np.array(spec_data[o]["flux"], float)
            qw, qf = continuum.quick_normalize(w, f)
            all_w.append(qw)
            all_f.append(qf)
        if all_w:
            cat_w = np.concatenate(all_w)
            cat_f = np.concatenate(all_f)
            vb, _ = rv_core.estimate_broadening(cat_w, cat_f, tw, tf)
            if vb is not None:
                vsini = float(
                    np.clip(
                        float(vb),
                        config.VSINI_PROXY_MIN_KMS,
                        config.VSINI_PROXY_MAX_KMS,
                    )
                )

    bank = templates.build_template_bank_cached(
        float(args.teff),
        vsini,
        metallicity=float(args.mh),
        logg=float(args.logg),
        hot_spectrum=hot,
        template_grid_wide=bool(args.template_grid_wide),
    )
    if not bank:
        raise SystemExit("Empty PHOENIX bank (check DARKHUNTER_PHOENIX_DIR and parameters)")

    bias: dict = {}
    if not args.no_bias and inst.bias_file:
        bias = io_utils.read_bias(inst.bias_file)

    mask_rv_by_chunk: dict[str, float] = {}
    if args.diagnostics_csv is not None:
        df = pd.read_csv(args.diagnostics_csv)
        m = df.loc[df["method"] == "mask_ccf", ["chunk_key", "rv_kms"]].dropna(subset=["rv_kms"])
        for _, row in m.iterrows():
            k = str(row["chunk_key"])
            try:
                mask_rv_by_chunk[k] = float(row["rv_kms"])
            except (TypeError, ValueError):
                pass

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    ckw = _continuum_kw(hot)
    with open(args.output_jsonl, "w") as fp:
        for chunk_key, w, f, e in chunking.iter_order_chunks(spec_data, inst.bad_orders, args.subchunks):
            if len(w) < 10:
                continue
            try:
                nw, nf, ne = continuum.fit_continuum(w, f, e, **ckw)
                nw, nf, ne = continuum.despike_normalized_pre_ccf(nw, nf, ne)
            except Exception as ex:
                rec = {"chunk_key": chunk_key, "error": f"continuum:{ex!s}"}
                fp.write(json.dumps(rec) + "\n")
                continue
            truth = mask_rv_by_chunk.get(str(chunk_key))
            truth_arg = None
            if truth is not None and np.isfinite(truth):
                bvec = io_utils.lookup_bias(bias, chunk_key)
                b0 = float(bvec[0]) if isinstance(bvec, (list, tuple)) and len(bvec) >= 1 else 0.0
                # Diagnostics mask RV is after -b0; FFT estimator is before pipeline subtracts b0.
                truth_arg = float(truth) + b0
            diag = fft_diagnostics.summarize_fft_chunk_failure_modes(
                nw,
                1.0 - nf,
                bank,
                vsini,
                rv_truth_kms=truth_arg,
            )
            diag["chunk_key"] = str(chunk_key)
            diag["vsini_proxy"] = vsini
            if truth is not None and np.isfinite(truth):
                diag["mask_rv_kms_csv_frame"] = float(truth)
            if truth_arg is not None:
                diag["mask_rv_kms_fft_frame"] = truth_arg
            fp.write(json.dumps(diag) + "\n")


if __name__ == "__main__":
    main()
