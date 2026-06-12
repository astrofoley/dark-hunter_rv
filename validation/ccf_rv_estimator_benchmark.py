#!/usr/bin/env python3
"""
Multi-estimator mask-CCF benchmark with phase precision gates.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.ccf_rv_estimator_benchmark \\
    --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \\
    --chunk-layout calibration/chunk_layouts/subchunks_4.yaml \\
    --estimators gauss_offset,parabolic_ls,smooth_peak,bi_gauss,grid,auto \\
    --phase C --check-gate \\
    --baseline calibration/ccf_estimator_baseline/phase_c_reference.json \\
    --out-dir validation_output/ccf_estimator_study
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from darkhunter_rv import continuum, instruments, io_utils, rv_core  # noqa: E402
from darkhunter_rv.pipeline import _mask_tournament  # noqa: E402
from validation.ccf_rv_precision_gates import (  # noqa: E402
    check_phase_gate,
    load_baseline,
    save_baseline,
    update_baseline_if_passed,
)
from validation.ccf_rv_post_debias import compare_estimators_post_debias  # noqa: E402
from validation.ccf_rv_precision_metrics import (  # noqa: E402
    build_per_object_bias_table,
    composite_score,
    summarize_estimator_metrics,
)
from validation.chunk_bias_lib import load_stellar_metadata  # noqa: E402
from validation.chunk_layout import iter_order_chunks_from_layout, load_chunk_layout  # noqa: E402

logger = logging.getLogger(__name__)


def _attach_stellar_metadata(chunk_df: pd.DataFrame, summary_dir: Path) -> pd.DataFrame:
    if chunk_df.empty:
        return chunk_df
    meta = load_stellar_metadata(summary_dir)
    if meta.empty:
        return chunk_df
    out = chunk_df.copy()
    out["gaia_dr3_id"] = out["gaia_dr3_id"].astype(str)
    meta["gaia_dr3_id"] = meta["gaia_dr3_id"].astype(str)
    out = out.merge(meta, on="gaia_dr3_id", how="left", suffixes=("", "_meta"))
    if "teff_gaia" in out.columns:
        out["teff"] = out["teff_gaia"].astype(float)
    if "logg" in out.columns:
        out["logg"] = out["logg"].astype(float)
    if "mh" in out.columns:
        out["mh"] = out["mh"].astype(float)
    return out


def _teff_for_star(gid: str, meta: pd.DataFrame, default_teff: float) -> float:
    if meta.empty:
        return default_teff
    row = meta.loc[meta["gaia_dr3_id"].astype(str) == str(gid)]
    if row.empty:
        return default_teff
    t = float(row["teff_gaia"].iloc[0])
    return t if np.isfinite(t) else default_teff


def summarize_chunk_table(
    chunk_df: pd.DataFrame,
    estimators: tuple[str, ...],
    out_dir: Path,
    *,
    summary_dir: Path,
    phase: str,
    baseline: Path | None,
    check_gate: bool,
    update_baseline: bool,
    post_debias: bool = False,
) -> dict:
    """Write comparison CSV + phase metrics from an existing per-chunk table."""
    comparison_rows = []
    metrics_by_est: dict[str, dict] = {}
    for est in estimators:
        bias_df = build_per_object_bias_table(
            chunk_df,
            rv_col=f"rv_kms__{est}",
            err_col=f"rv_err_kms__{est}",
        )
        if not bias_df.empty:
            bias_df = _attach_stellar_metadata(bias_df, summary_dir)
            if "teff_gaia" in bias_df.columns:
                miss = ~np.isfinite(bias_df["teff"].astype(float)) if "teff" in bias_df.columns else np.ones(len(bias_df), bool)
                if "teff" not in bias_df.columns:
                    bias_df["teff"] = bias_df["teff_gaia"].astype(float)
                else:
                    bias_df.loc[miss, "teff"] = bias_df.loc[miss, "teff_gaia"].astype(float)
            bias_df.to_csv(out_dir / f"per_object_bias__{est}.csv", index=False)
        m = summarize_estimator_metrics(chunk_df, estimator=est, bias_df=bias_df)
        m["composite_score"] = composite_score(m)
        metrics_by_est[est] = m
        comparison_rows.append(m)

    cmp_df = pd.DataFrame(comparison_rows).sort_values("composite_score")
    cmp_df.to_csv(out_dir / "estimator_comparison.csv", index=False)

    winner = str(cmp_df.iloc[0]["estimator_name"]) if not cmp_df.empty else estimators[0]
    phase_metrics = dict(metrics_by_est.get(winner, {}))
    phase_metrics["winner_estimator"] = winner

    phase_path = out_dir / f"phase_{phase}_metrics.json"
    phase_path.write_text(json.dumps(phase_metrics, indent=2) + "\n", encoding="utf-8")

    gate_result = None
    if baseline is not None and baseline.is_file():
        prior = load_baseline(baseline)
        gate_result = check_phase_gate(str(phase), phase_metrics, prior, strict=check_gate)
        gate_path = out_dir / f"phase_{phase}_gate.json"
        gate_path.write_text(json.dumps(gate_result, indent=2) + "\n", encoding="utf-8")
        if check_gate and not gate_result["passed"]:
            logger.error("Phase %s gate FAILED: %s", phase, gate_result["failures"])
        elif gate_result["passed"]:
            logger.info("Phase %s gate PASSED", phase)
        if update_baseline and gate_result.get("passed"):
            update_baseline_if_passed(
                baseline,
                phase=str(phase),
                metrics=phase_metrics,
                gate_result=gate_result,
                estimator=winner,
            )

    post_debias_result = None
    if post_debias:
        post_cmp = compare_estimators_post_debias(
            chunk_df,
            estimators,
            summary_dir,
            out_dir=out_dir,
        )
        post_cmp.to_csv(out_dir / "estimator_comparison_post_debias.csv", index=False)
        post_winner = str(post_cmp.iloc[0]["estimator_name"]) if not post_cmp.empty else winner
        post_debias_result = {
            "winner": post_winner,
            "comparison_path": str(out_dir / "estimator_comparison_post_debias.csv"),
        }
        (out_dir / "phase_C_post_debias_winner.json").write_text(
            json.dumps(post_debias_result, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Post-debias winner: %s (median σ_RV=%.4f km/s)", post_winner, post_cmp.iloc[0]["median_sigma_rv_kms"])

    return {
        "n_chunks": len(chunk_df),
        "n_spectra": chunk_df["file"].nunique() if not chunk_df.empty else 0,
        "winner": winner,
        "phase_metrics": phase_metrics,
        "gate": gate_result,
        "post_debias": post_debias_result,
    }


def _read_spectrum_list(path: Path, *, max_n: int | None) -> list[Path]:
    lines = [
        ln.strip()
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    paths = [Path(ln) if Path(ln).is_absolute() else REPO_ROOT / ln for ln in lines]
    if max_n is not None:
        paths = paths[: int(max_n)]
    return paths


def _gaia_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("Gaia_DR3_"):
        return stem.replace("Gaia_DR3_", "").split("_")[0]
    return stem


def measure_spectrum_chunks(
    spectrum_path: Path,
    *,
    instrument,
    layout,
    estimators: tuple[str, ...],
    teff: float,
    continuum_mode: str,
) -> list[dict]:
    if instrument.name == "GHOST":
        _, spec_data = io_utils.read_spectrum_ghost(str(spectrum_path))
    elif instrument.name == "MAROON-X":
        _, spec_data = io_utils.read_spectrum_maroonx(str(spectrum_path))
    else:
        _, spec_data = io_utils.read_spectrum(str(spectrum_path))

    valid_orders = sorted(o for o in spec_data if o not in instrument.bad_orders)
    if not valid_orders:
        return []
    mid = len(valid_orders) // 2
    test_orders = valid_orders[max(0, mid - 2) : min(len(valid_orders), mid + 2)]
    mask_dir = Path(instrument.mask_directory)
    mask_pack, _, _ = _mask_tournament(spec_data, instrument, test_orders, mask_dir, continuum_mode)
    if mask_pack is None:
        logger.warning("No mask for %s", spectrum_path)
        return []

    mw, ms = mask_pack["w"], mask_pack["s"]
    gid = _gaia_id_from_path(spectrum_path)
    rows: list[dict] = []

    for chunk_key, w, f, e in iter_order_chunks_from_layout(spec_data, instrument.bad_orders, layout):
        if len(w) < 10:
            continue
        try:
            nw, nf, _ = continuum.fit_continuum(w, f, e, continuum_mode=continuum_mode)
            nw, nf, _ = continuum.despike_normalized_pre_ccf(nw, nf, _)
        except Exception as ex:
            logger.debug("continuum fail %s %s: %s", spectrum_path.name, chunk_key, ex)
            continue
        if nw[-1] < mw[0] or nw[0] > mw[-1]:
            continue
        line_obs = rv_core.mask_line_flux_in_excluded_wavelengths(nw, 1.0 - nf)
        results, _, _, peak_snr = rv_core.cross_correlate_stellar_mask_all(
            nw, line_obs, mw, ms, estimators=estimators
        )
        if not results:
            continue
        base = {
            "file": spectrum_path.name,
            "spectrum_path": str(spectrum_path),
            "gaia_dr3_id": gid,
            "chunk_key": str(chunk_key),
            "teff": float(teff),
            "peak_snr": float(peak_snr),
            "log10_peak_snr": float(np.log10(peak_snr)) if np.isfinite(peak_snr) and peak_snr > 0 else float("nan"),
        }
        from darkhunter_rv import chunking

        _, ord_num, _ = chunking.parse_chunk_key(str(chunk_key))
        base["chunk_order"] = int(ord_num)
        for est_name, res in results.items():
            base[f"rv_kms__{est_name}"] = float(res.rv_kms)
            base[f"rv_err_kms__{est_name}"] = float(res.rv_err_kms)
            base[f"asymmetry__{est_name}"] = float(res.asymmetry)
            base[f"fit_ok__{est_name}"] = bool(res.fit_ok)
        rows.append(base)
    return rows


def run_benchmark(args: argparse.Namespace) -> dict:
    layout = load_chunk_layout(args.chunk_layout)
    instrument = instruments.get_instrument_profile(args.instrument)
    estimators = tuple(e.strip() for e in args.estimators.split(",") if e.strip())

    spec_paths = _read_spectrum_list(args.spectrum_list, max_n=args.max_spectra)
    stellar_meta = load_stellar_metadata(args.summary_dir)
    all_rows: list[dict] = []
    for i, sp in enumerate(spec_paths):
        if not sp.is_file():
            logger.warning("Missing spectrum: %s", sp)
            continue
        gid = _gaia_id_from_path(sp)
        teff = _teff_for_star(gid, stellar_meta, float(args.default_teff))
        rows = measure_spectrum_chunks(
            sp,
            instrument=instrument,
            layout=layout,
            estimators=estimators,
            teff=teff,
            continuum_mode=args.continuum_mode,
        )
        all_rows.extend(rows)
        if (i + 1) % 10 == 0:
            logger.info("Processed %d / %d spectra (%d chunks)", i + 1, len(spec_paths), len(all_rows))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_df = pd.DataFrame(all_rows)
    chunk_df = _attach_stellar_metadata(chunk_df, args.summary_dir)
    chunk_df.to_csv(out_dir / "per_chunk_rv.csv", index=False)

    return summarize_chunk_table(
        chunk_df,
        estimators,
        out_dir,
        summary_dir=args.summary_dir,
        phase=str(args.phase),
        baseline=args.baseline,
        check_gate=bool(args.check_gate),
        update_baseline=bool(args.update_baseline),
        post_debias=bool(args.post_debias),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spectrum-list", type=Path, default=None)
    ap.add_argument("--chunk-layout", type=Path, default=None)
    ap.add_argument("--resummarize-only", action="store_true", help="Recompute metrics from out-dir/per_chunk_rv.csv")
    ap.add_argument("--summary-dir", type=Path, default=REPO_ROOT / "output")
    ap.add_argument("--instrument", default="APF")
    ap.add_argument("--estimators", default="gauss_offset,parabolic_ls,smooth_peak,bi_gauss,grid,auto")
    ap.add_argument("--out-dir", type=Path, default=Path("validation_output/ccf_estimator_study"))
    ap.add_argument("--phase", default="C")
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--check-gate", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--max-spectra", type=int, default=None)
    ap.add_argument("--default-teff", type=float, default=4500.0)
    ap.add_argument("--continuum-mode", default="chebyshev")
    ap.add_argument("--post-debias", action="store_true", help="Rank estimators by σ_RV after regression debias")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    if args.baseline is None and args.check_gate:
        args.baseline = REPO_ROOT / "calibration/ccf_estimator_baseline" / f"phase_{args.phase.lower()}_reference.json"

    if args.resummarize_only:
        chunk_path = args.out_dir / "per_chunk_rv.csv"
        if not chunk_path.is_file():
            logger.error("Missing %s", chunk_path)
            return 1
        chunk_df = pd.read_csv(chunk_path)
        chunk_df = _attach_stellar_metadata(chunk_df, args.summary_dir)
        estimators = tuple(e.strip() for e in args.estimators.split(",") if e.strip())
        summary = summarize_chunk_table(
            chunk_df,
            estimators,
            args.out_dir,
            summary_dir=args.summary_dir,
            phase=str(args.phase),
            baseline=args.baseline,
            check_gate=bool(args.check_gate),
            update_baseline=bool(args.update_baseline),
            post_debias=bool(args.post_debias),
        )
    else:
        if args.spectrum_list is None or args.chunk_layout is None:
            ap.error("--spectrum-list and --chunk-layout required unless --resummarize-only")
        summary = run_benchmark(args)
    print(json.dumps(summary, indent=2, default=str))
    if args.check_gate and summary.get("gate") and not summary["gate"].get("passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
