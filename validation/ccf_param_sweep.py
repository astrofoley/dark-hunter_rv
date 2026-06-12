#!/usr/bin/env python3
"""
Phase B: sweep mask-CCF fit parameters on a bias-train subset.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.ccf_param_sweep \\
    --bias-train calibration/bias_train.txt \\
    --chunk-layout calibration/chunk_layouts/subchunks_4.yaml \\
    --out-dir validation_output/ccf_estimator_study/param_sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from darkhunter_rv import config  # noqa: E402
from validation.ccf_rv_precision_gates import check_phase_gate, load_baseline  # noqa: E402
from validation.ccf_rv_precision_metrics import build_per_object_bias_table, summarize_estimator_metrics  # noqa: E402
from validation.chunk_layout import load_chunk_layout  # noqa: E402
from darkhunter_rv import instruments  # noqa: E402

logger = logging.getLogger(__name__)


def _read_paths_list(path: Path, max_n: int | None) -> list[Path]:
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    paths = [Path(ln) if Path(ln).is_absolute() else REPO_ROOT / ln for ln in lines]
    if max_n is not None:
        paths = paths[: int(max_n)]
    return paths


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    layout = load_chunk_layout(args.chunk_layout)
    instrument = instruments.get_instrument_profile(args.instrument)
    spec_paths = _read_paths_list(args.bias_train, max_n=args.max_spectra)

    fit_widths = [int(x) for x in args.fit_widths.split(",")]
    min_snrs = [float(x) for x in args.min_peak_snr.split(",")]
    min_sigmas = [float(x) for x in args.min_gauss_sigma_kms.split(",")]

    rows: list[dict] = []
    orig_sigma = float(config.MASK_CCF_MIN_GAUSS_SIGMA_KMS)

    from darkhunter_rv import continuum, io_utils, rv_core  # noqa: E402

    for fw, msnr, msig in itertools.product(fit_widths, min_snrs, min_sigmas):
        config.MASK_CCF_MIN_GAUSS_SIGMA_KMS = msig
        from darkhunter_rv.pipeline import _mask_tournament  # noqa: E402
        from validation.chunk_layout import iter_order_chunks_from_layout  # noqa: E402

        chunk_rows = []
        for sp in spec_paths:
            if not sp.is_file():
                continue
            _, spec_data = io_utils.read_spectrum(str(sp))
            valid_orders = sorted(o for o in spec_data if o not in instrument.bad_orders)
            mid = len(valid_orders) // 2
            test_orders = valid_orders[max(0, mid - 2) : min(len(valid_orders), mid + 2)]
            mask_pack, _, _ = _mask_tournament(
                spec_data, instrument, test_orders, Path(instrument.mask_directory), args.continuum_mode
            )
            if mask_pack is None:
                continue
            mw, ms = mask_pack["w"], mask_pack["s"]
            for chunk_key, w, f, e in iter_order_chunks_from_layout(spec_data, instrument.bad_orders, layout):
                if len(w) < 10:
                    continue
                try:
                    nw, nf, ne = continuum.fit_continuum(w, f, e, continuum_mode=args.continuum_mode)
                    nw, nf, ne = continuum.despike_normalized_pre_ccf(nw, nf, ne)
                except Exception:
                    continue
                if nw[-1] < mw[0] or nw[0] > mw[-1]:
                    continue
                line_obs = rv_core.mask_line_flux_in_excluded_wavelengths(nw, 1.0 - nf)
                rv, err, _, _, peak, _, snr = rv_core.cross_correlate_stellar_mask(
                    nw,
                    line_obs,
                    mw,
                    ms,
                    fit_width=int(fw),
                    min_peak_snr=float(msnr),
                    estimator="gauss_offset",
                )
                chunk_rows.append(
                    {
                        "file": sp.name,
                        "gaia_dr3_id": sp.stem.replace("Gaia_DR3_", "").split("_")[0],
                        "chunk_key": str(chunk_key),
                        "rv_kms__gauss_offset": float(rv),
                        "rv_err_kms__gauss_offset": float(err),
                        "peak_snr": float(snr),
                    }
                )
        chunk_df = pd.DataFrame(chunk_rows)
        bias_df = build_per_object_bias_table(chunk_df, rv_col="rv_kms__gauss_offset", err_col="rv_err_kms__gauss_offset")
        m = summarize_estimator_metrics(chunk_df, estimator="gauss_offset", bias_df=bias_df)
        m.update(
            {
                "fit_width": fw,
                "min_peak_snr": msnr,
                "min_gauss_sigma_kms": msig,
            }
        )
        rows.append(m)
        logger.info(
            "fit_width=%s min_snr=%s min_sigma=%s -> median_sigma=%.4f",
            fw,
            msnr,
            msig,
            m.get("median_sigma_rv_kms", float("nan")),
        )

    config.MASK_CCF_MIN_GAUSS_SIGMA_KMS = orig_sigma
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bias-train", type=Path, default=REPO_ROOT / "calibration/bias_train.txt")
    ap.add_argument("--chunk-layout", type=Path, required=True)
    ap.add_argument("--instrument", default="APF")
    ap.add_argument("--out-dir", type=Path, default=Path("validation_output/ccf_estimator_study/param_sweep"))
    ap.add_argument("--fit-widths", default="25,50,75")
    ap.add_argument("--min-peak-snr", default="2.5,3.5,5")
    ap.add_argument("--min-gauss-sigma-kms", default="2,2.5,3")
    ap.add_argument("--max-spectra", type=int, default=30)
    ap.add_argument("--default-teff", type=float, default=4500.0)
    ap.add_argument("--continuum-mode", default="chebyshev")
    ap.add_argument("--baseline", type=Path, default=REPO_ROOT / "calibration/ccf_estimator_baseline/phase_a_reference.json")
    ap.add_argument("--check-gate", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_df = run_sweep(args)
    sweep_df.to_csv(args.out_dir / "param_sweep.csv", index=False)

    if sweep_df.empty:
        print(json.dumps({"error": "no results"}))
        return 1

    best = sweep_df.sort_values("median_sigma_rv_kms").iloc[0]
    best_metrics = {k: float(best[k]) for k in sweep_df.columns if k in (
        "median_sigma_rv_kms", "p90_sigma_rv_kms", "median_chunk_scatter_kms",
        "bias_curve_rms_kms", "stellar_bias_cv_rmse_kms", "low_snr_finite_rate",
    )}
    (args.out_dir / "phase_B_best.json").write_text(json.dumps(dict(best), indent=2, default=str) + "\n")

    if args.check_gate and args.baseline.is_file():
        prior = load_baseline(args.baseline)
        gate = check_phase_gate("B", best_metrics, prior, strict=True)
        (args.out_dir / "phase_B_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        if not gate["passed"]:
            return 1
    print(json.dumps({"best": dict(best), "n_configs": len(sweep_df)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
