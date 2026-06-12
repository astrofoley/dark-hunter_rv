#!/usr/bin/env python3
"""
Build stellar-parameter-corrected bias tables per CCF estimator.

Joins per-object chunk residuals with Teff/logg/[M/H], fits nested regression models,
and exports bias_statistics.txt on regression-adjusted residuals.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.ccf_estimator_bias \\
    --wide-diagnostics validation_output/ccf_estimator_study/per_chunk_rv.csv \\
    --summary-dir output \\
    --estimator auto \\
    --out-dir validation_output/ccf_estimator_study/bias_auto
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

from validation.ccf_rv_precision_metrics import build_per_object_bias_table  # noqa: E402
from validation.chunk_bias_lib import load_stellar_metadata  # noqa: E402
from validation.chunk_bias_regression import (  # noqa: E402
    _fit_model,
    choose_best_model,
    compare_models,
    export_regression_bias_table,
    per_chunk_stellar_slopes,
)
logger = logging.getLogger(__name__)


def _wide_to_bias_csv(wide: pd.DataFrame, estimator: str) -> pd.DataFrame:
    rv_col = f"rv_kms__{estimator}"
    err_col = f"rv_err_kms__{estimator}"
    if rv_col not in wide.columns:
        raise ValueError(f"estimator column missing: {rv_col}")
    return build_per_object_bias_table(wide, rv_col=rv_col, err_col=err_col)


def prepare_bias_regression_table(
    wide: pd.DataFrame,
    estimator: str,
    summary_dir: Path,
) -> pd.DataFrame:
    """Per-object chunk residuals joined with stellar metadata for regression."""
    bias_raw = _wide_to_bias_csv(wide, estimator)
    if bias_raw.empty:
        raise RuntimeError(f"No bias rows for estimator={estimator}")
    meta = load_stellar_metadata(summary_dir)
    bias_raw["gaia_dr3_id"] = bias_raw["gaia_dr3_id"].astype(str)
    if not meta.empty:
        meta["gaia_dr3_id"] = meta["gaia_dr3_id"].astype(str)
        bias_raw = bias_raw.merge(meta, on="gaia_dr3_id", how="left")
    if "teff" not in bias_raw.columns or bias_raw["teff"].isna().all():
        bias_raw["teff"] = (
            bias_raw["teff_gaia"].astype(float) if "teff_gaia" in bias_raw.columns else float("nan")
        )
    if "logg" not in bias_raw.columns:
        bias_raw["logg"] = float("nan")
    if "mh" not in bias_raw.columns:
        bias_raw["mh"] = float("nan")
    if "log10_median_mask_ccf_peak_snr" not in bias_raw.columns:
        if "log10_peak_snr" in wide.columns:
            snr = (
                wide.groupby("gaia_dr3_id")["log10_peak_snr"]
                .median()
                .rename("log10_median_mask_ccf_peak_snr")
            )
            bias_raw = bias_raw.merge(snr, on="gaia_dr3_id", how="left")
        else:
            bias_raw["log10_median_mask_ccf_peak_snr"] = float("nan")
    orders = bias_raw["chunk_order"].astype(float)
    o_min, o_max = float(np.nanmin(orders)), float(np.nanmax(orders))
    bias_raw["chunk_order_norm"] = (orders - o_min) / max(o_max - o_min, 1.0)
    return bias_raw


def fit_regression_bias(
    wide: pd.DataFrame,
    estimator: str,
    summary_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str, pd.DataFrame]:
    """
    Nested chunk/stellar bias regression.

    Returns (adjusted, per_object_stack, chosen_model, model_cmp).
    ``per_object_stack.weighted_mean_residual_kms`` holds ``adjusted_bias_kms`` for IVW stack.
    """
    bias_raw = prepare_bias_regression_table(wide, estimator, summary_dir)
    model_cmp = compare_models(bias_raw)
    chosen = choose_best_model(model_cmp)
    fit = _fit_model(bias_raw, chosen)
    adjusted = export_regression_bias_table(bias_raw, fit)

    per_object_stack = adjusted.copy()
    debias = per_object_stack["adjusted_bias_kms"].astype(float)
    raw_bias = bias_raw["weighted_mean_residual_kms"].astype(float).values
    per_object_stack["weighted_mean_residual_kms"] = np.where(np.isfinite(debias), debias, raw_bias)
    if "sample_kept" not in per_object_stack.columns:
        per_object_stack["sample_kept"] = True
    return adjusted, per_object_stack, chosen, model_cmp


def _export_bias_statistics(adjusted: pd.DataFrame, out_path: Path) -> None:
    """Per-order b0 from sample mean of adjusted bias (stellar+curve corrected)."""
    if adjusted.empty or "adjusted_bias_kms" not in adjusted.columns:
        out_path.write_text("# order bias_dv bias_err_stat bias_rms_stat\n", encoding="utf-8")
        return
    tmp = adjusted.copy()
    tmp["order"] = tmp["chunk_key"].astype(str).str.split("_").str[0].astype(int)
    order_rows = []
    for order, g in tmp.groupby("order"):
        vals = g["adjusted_bias_kms"].astype(float).values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        order_rows.append(
            (int(order), float(np.mean(vals)), float(np.std(vals)), float(np.std(vals)))
        )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# order bias_dv bias_err_stat bias_rms_stat\n")
        for o, b, e, r in sorted(order_rows):
            fh.write(f"{o} {b:.8f} {e:.8f} {r:.8f}\n")


def build_estimator_bias(
    wide_path: Path,
    *,
    summary_dir: Path,
    estimator: str,
    out_dir: Path,
) -> dict:
    wide = pd.read_csv(wide_path)
    bias_raw = prepare_bias_regression_table(wide, estimator, summary_dir)
    adjusted, _, chosen, model_cmp = fit_regression_bias(wide, estimator, summary_dir)
    slopes = per_chunk_stellar_slopes(bias_raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    bias_raw.to_csv(out_dir / "per_object_chunk_bias_raw.csv", index=False)
    adjusted.to_csv(out_dir / "bias_by_chunk_corrected.csv", index=False)
    model_cmp.to_csv(out_dir / "model_comparison.csv", index=False)
    slopes.to_csv(out_dir / "stellar_slope_report.csv", index=False)
    _export_bias_statistics(adjusted, out_dir / "bias_statistics.txt")

    summary = {
        "estimator": estimator,
        "chosen_bias_model": chosen,
        "n_objects": int(bias_raw["gaia_dr3_id"].nunique()),
        "n_rows": int(len(bias_raw)),
        "n_significant_teff_slopes": int(slopes["significant"].sum()) if not slopes.empty else 0,
        "cv_rmse_kms": float(model_cmp.loc[model_cmp["model"] == chosen, "cv_rmse_kms"].iloc[0]),
    }
    (out_dir / "chosen_bias_model.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wide-diagnostics", type=Path, required=True)
    ap.add_argument("--summary-dir", type=Path, default=REPO_ROOT / "output")
    ap.add_argument("--estimator", default="auto")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    summary = build_estimator_bias(
        args.wide_diagnostics,
        summary_dir=args.summary_dir,
        estimator=args.estimator,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
