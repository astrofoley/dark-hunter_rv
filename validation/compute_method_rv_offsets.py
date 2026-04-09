#!/usr/bin/env python3
"""
Estimate global RV offsets so template_fft and strong_lines match stellar mask (taken as truth).

Uses the **same** exposure-level stacks as the pipeline (``_weighted_method_rv_from_rows``) and
optional applicability regions (``config.METHOD_REGION_*`` + ``rv_method_diagnostics_report`` helpers).

**Joint calibration set:** exposures where mask, template, and strong_lines are all finite with
σ ≤ ``--max-sigma-kms`` and (by default) all three applicability regions hold. On that set:

  offset_template  = aggregate( RV_mask − RV_template )
  offset_strong    = aggregate( RV_mask − RV_strong_lines )

Apply later as::

  rv_template_corr  = rv_template  + offset_template
  rv_strong_corr    = rv_strong      + offset_strong

Using one shared exposure set makes the pairwise differences **internally consistent** (unlike
taking medians on three different pair-only subsets).

Writes a whitespace table readable by :func:`darkhunter_rv.io_utils.read_method_rv_offsets`.

Example::

  python -m validation.compute_method_rv_offsets \\
    --diagnostics-list validation_output/rv_method_teff_report_full/diagnostics_list.txt \\
    --instrument APF \\
    --output method_rv_offsets.txt
"""
from __future__ import annotations

import argparse
import logging
import sys
from glob import glob as glob_paths
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config as dh_config  # noqa: E402
from darkhunter_rv.io_utils import read_method_rv_offsets, write_method_rv_offsets  # noqa: E402
from darkhunter_rv.method_evaluation import exposure_method_flags  # noqa: E402
from darkhunter_rv.pipeline import _weighted_method_rv_from_rows  # noqa: E402

from darkhunter_rv.method_regions import (  # noqa: E402
    region_mask_applicable,
    region_strong_lines_applicable,
    region_template_applicable,
)


def _load_paths(glob_pat: str | None, list_path: Path | None) -> list[Path]:
    if (glob_pat is None or not str(glob_pat).strip()) == (list_path is None):
        raise ValueError("Provide exactly one of diagnostics_glob or diagnostics_list")
    if list_path is not None:
        lines = list_path.read_text().splitlines()
        out: list[Path] = []
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            p = Path(s)
            if p.is_file():
                out.append(p.resolve())
        return sorted(set(out))
    return sorted(Path(p) for p in glob_paths(str(glob_pat)))


def _exposure_row(recs: list[dict], teff: float) -> dict:
    rv_m, er_m = _weighted_method_rv_from_rows(recs, "mask_ccf")
    rv_t, er_t = _weighted_method_rv_from_rows(recs, "template_fft")
    rv_sl, er_sl = _weighted_method_rv_from_rows(recs, "strong_lines")
    fl = exposure_method_flags(recs)
    snr_m = float(fl["median_mask_ccf_peak_snr"])
    if np.isfinite(snr_m) and snr_m > 0:
        log10_snr = float(np.log10(snr_m))
    else:
        log10_snr = float("nan")
    return {
        "teff": float(teff),
        "log10_snr": log10_snr,
        "rv_mask": rv_m,
        "err_mask": er_m,
        "rv_template": rv_t,
        "err_template": er_t,
        "rv_sl": rv_sl,
        "err_sl": er_sl,
    }


def _sigma_ok(
    err_m: np.ndarray,
    err_t: np.ndarray,
    err_sl: np.ndarray,
    thr: float,
) -> np.ndarray:
    thr = float(thr)
    em = np.asarray(err_m, float)
    et = np.asarray(err_t, float)
    es = np.asarray(err_sl, float)
    return (
        np.isfinite(em)
        & np.isfinite(et)
        & np.isfinite(es)
        & (em > 0)
        & (et > 0)
        & (es > 0)
        & (em <= thr)
        & (et <= thr)
        & (es <= thr)
    )


def _finite_rv(rm, rt, rsl) -> np.ndarray:
    return np.isfinite(np.asarray(rm, float)) & np.isfinite(np.asarray(rt, float)) & np.isfinite(
        np.asarray(rsl, float)
    )


def _joint_offsets(
    rm: np.ndarray,
    rt: np.ndarray,
    rsl: np.ndarray,
    et: np.ndarray,
    esl: np.ndarray,
    estimator: str,
) -> tuple[float, float]:
    d_t = np.asarray(rm, float) - np.asarray(rt, float)
    d_s = np.asarray(rm, float) - np.asarray(rsl, float)
    if estimator == "median":
        return float(np.median(d_t)), float(np.median(d_s))
    if estimator == "mean":
        return float(np.mean(d_t)), float(np.mean(d_s))
    if estimator == "weighted":
        wt = 1.0 / (np.asarray(et, float) ** 2 + 1e-18)
        ws = 1.0 / (np.asarray(esl, float) ** 2 + 1e-18)
        return float(np.sum(wt * d_t) / np.sum(wt)), float(np.sum(ws * d_s) / np.sum(ws))
    raise ValueError(f"Unknown estimator {estimator!r}")


def joint_calibration_from_arrays(
    rm: np.ndarray,
    rt: np.ndarray,
    rsl: np.ndarray,
    em: np.ndarray,
    et: np.ndarray,
    esl: np.ndarray,
    teff: np.ndarray,
    log10_snr: np.ndarray,
    *,
    max_sigma_kms: float,
    apply_method_regions: bool,
    estimator: str,
) -> dict:
    """
    Build joint mask/template/strong calibration set and offsets (mask = truth).

    Returns a dict with offsets, counts, pairwise diagnostics, and post-check medians.
    """
    ok_rv = _finite_rv(rm, rt, rsl)
    ok_sig = _sigma_ok(em, et, esl, max_sigma_kms)
    if apply_method_regions:
        ok_reg = (
            region_mask_applicable(teff, log10_snr)
            & region_template_applicable(teff, log10_snr)
            & region_strong_lines_applicable(teff, log10_snr)
        )
    else:
        ok_reg = np.ones(len(np.asarray(rm)), dtype=bool)

    joint = ok_rv & ok_sig & ok_reg
    n_joint = int(np.sum(joint))
    if n_joint < 1:
        raise ValueError("Joint calibration set is empty (check regions and σ cut)")

    off_t, off_s = _joint_offsets(
        rm[joint],
        rt[joint],
        rsl[joint],
        et[joint],
        esl[joint],
        estimator,
    )

    def _pair_sel(sel: np.ndarray, d: np.ndarray) -> tuple[int, float]:
        m = sel & np.isfinite(d)
        if not np.any(m):
            return 0, float("nan")
        return int(np.sum(m)), float(np.median(d[m]))

    d_mt = rm - rt
    d_ms = rm - rsl
    d_ts = rt - rsl
    ok_mt = (
        np.isfinite(rm)
        & np.isfinite(rt)
        & np.isfinite(em)
        & np.isfinite(et)
        & (em > 0)
        & (et > 0)
        & (em <= max_sigma_kms)
        & (et <= max_sigma_kms)
    )
    ok_ms = (
        np.isfinite(rm)
        & np.isfinite(rsl)
        & np.isfinite(em)
        & np.isfinite(esl)
        & (em > 0)
        & (esl > 0)
        & (em <= max_sigma_kms)
        & (esl <= max_sigma_kms)
    )
    ok_ts = (
        np.isfinite(rt)
        & np.isfinite(rsl)
        & np.isfinite(et)
        & np.isfinite(esl)
        & (et > 0)
        & (esl > 0)
        & (et <= max_sigma_kms)
        & (esl <= max_sigma_kms)
    )

    n_mt, med_mt = _pair_sel(ok_mt, d_mt)
    n_ms, med_ms = _pair_sel(ok_ms, d_ms)
    n_ts, med_ts = _pair_sel(ok_ts, d_ts)

    rt_c = rt[joint] + off_t
    rsl_c = rsl[joint] + off_s
    post_mt = float(np.median(rm[joint] - rt_c))
    post_ms = float(np.median(rm[joint] - rsl_c))
    post_ts = float(np.median(rt_c - rsl_c))

    return {
        "offset_template_fft_kms": off_t,
        "offset_strong_lines_kms": off_s,
        "n_joint": n_joint,
        "pairwise": {
            "n_mask_template": n_mt,
            "median_mask_minus_template": med_mt,
            "n_mask_strong": n_ms,
            "median_mask_minus_strong": med_ms,
            "n_template_strong": n_ts,
            "median_template_minus_strong": med_ts,
        },
        "post_median_mask_minus_template": post_mt,
        "post_median_mask_minus_strong": post_ms,
        "post_median_template_minus_strong": post_ts,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Global method RV offsets vs stellar mask (truth)")
    ap.add_argument("--diagnostics-glob", default=None)
    ap.add_argument("--diagnostics-list", type=Path, default=None)
    ap.add_argument("--instrument", required=True, help="Instrument label for this batch (e.g. APF)")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("method_rv_offsets.txt"),
        help="Output path (whitespace table + comments)",
    )
    ap.add_argument(
        "--max-sigma-kms",
        type=float,
        default=float(dh_config.COMPARISON_REPORT_MAX_RV_ERR_KMS),
        help="Require all three σ ≤ this (km/s) for joint set",
    )
    ap.add_argument(
        "--no-method-regions",
        action="store_true",
        help="Do not require METHOD_REGION_* applicability (only σ + finite RVs)",
    )
    ap.add_argument(
        "--estimator",
        choices=("median", "mean", "weighted"),
        default="median",
        help="How to combine (mask−template) and (mask−strong) on the joint set",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Merge into existing output file (other instruments preserved; same instrument overwritten)",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    try:
        paths = _load_paths(args.diagnostics_glob, args.diagnostics_list)
    except ValueError as e:
        logging.error("%s", e)
        return 2
    if not paths:
        logging.error("No diagnostics CSVs found")
        return 2

    rows: list[dict] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as ex:
            logging.warning("skip %s: %s", p, ex)
            continue
        if df.empty:
            continue
        recs = df.to_dict("records")
        teff = float(df["teff"].iloc[0]) if "teff" in df.columns else float("nan")
        rows.append(_exposure_row(recs, teff))

    if not rows:
        logging.error("No valid rows from diagnostics")
        return 2

    tab = pd.DataFrame(rows)
    teff = tab["teff"].astype(float).values
    log10_snr = tab["log10_snr"].astype(float).values
    rm = tab["rv_mask"].astype(float).values
    rt = tab["rv_template"].astype(float).values
    rsl = tab["rv_sl"].astype(float).values
    em = tab["err_mask"].astype(float).values
    et = tab["err_template"].astype(float).values
    esl = tab["err_sl"].astype(float).values

    try:
        out = joint_calibration_from_arrays(
            rm,
            rt,
            rsl,
            em,
            et,
            esl,
            teff,
            log10_snr,
            max_sigma_kms=args.max_sigma_kms,
            apply_method_regions=not args.no_method_regions,
            estimator=args.estimator,
        )
    except ValueError as e:
        logging.error("%s", e)
        return 2

    off_t = out["offset_template_fft_kms"]
    off_s = out["offset_strong_lines_kms"]
    n_joint = out["n_joint"]
    pw = out["pairwise"]
    n_mt = pw["n_mask_template"]
    med_mt = pw["median_mask_minus_template"]
    n_ms = pw["n_mask_strong"]
    med_ms = pw["median_mask_minus_strong"]
    n_ts = pw["n_template_strong"]
    med_ts = pw["median_template_minus_strong"]
    post_mt = out["post_median_mask_minus_template"]
    post_ms = out["post_median_mask_minus_strong"]
    post_ts = out["post_median_template_minus_strong"]

    comment_lines = [
        f"# method_rv_offsets format_version=1",
        f"# Stellar mask RV = truth. Apply:",
        f"#   rv_template_corrected  = rv_template  + offset_template_fft_kms",
        f"#   rv_strong_lines_corrected = rv_strong_lines + offset_strong_lines_kms",
        f"#",
        f"# joint_set: all three methods finite, σ<={args.max_sigma_kms:g} km/s each"
        + ("" if args.no_method_regions else " + METHOD_REGION_* intersection"),
        f"# joint_n={n_joint} estimator={args.estimator}",
        f"#",
        f"# Pairwise median deltas (subset sizes may differ — not a closed triangle):",
        f"#   median(mask-template) n={n_mt} km/s={med_mt:.4f}",
        f"#   median(mask-strong)  n={n_ms} km/s={med_ms:.4f}",
        f"#   median(template-strong) n={n_ts} km/s={med_ts:.4f}",
        f"#   (mask-strong)-(template-strong) vs median(mask-template): "
        f"{med_ms - med_ts:.4f} vs {med_mt:.4f} (diff {med_ms - med_ts - med_mt:.4f})",
        f"#",
        f"# After applying {args.estimator} joint offsets on joint set — median residuals (should ~0 for mask):",
        f"#   median(mask - corrected_template) = {post_mt:.4f}",
        f"#   median(mask - corrected_strong)   = {post_ms:.4f}",
        f"#   median(corrected_template - corrected_strong) = {post_ts:.4f}",
        f"#",
        f"# Columns: instrument offset_template_fft_kms offset_strong_lines_kms n_exposures_joint estimator",
    ]

    merged: dict[str, dict] = {}
    if args.append and args.output.is_file():
        merged = read_method_rv_offsets(args.output)
    merged[str(args.instrument)] = {
        "offset_template_fft_kms": off_t,
        "offset_strong_lines_kms": off_s,
        "n_exposures_joint": n_joint,
        "estimator": str(args.estimator),
    }
    out_rows = [
        {"instrument": inst, **data} for inst, data in sorted(merged.items(), key=lambda x: x[0])
    ]

    write_method_rv_offsets(args.output, out_rows, comment_lines=comment_lines)
    logging.info("Wrote %s (joint_n=%d)", args.output.resolve(), n_joint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
