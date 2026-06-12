#!/usr/bin/env python3
"""
Post-debias σ_RV: apply chunk + stellar regression bias, then IVW stack per exposure.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python -m validation.ccf_rv_post_debias \\
    --wide-diagnostics validation_output/ccf_estimator_study/per_chunk_rv.csv \\
    --summary-dir output \\
    --estimators gauss_offset,parabolic_ls,smooth_peak,bi_gauss,grid,auto \\
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

from validation.ccf_estimator_bias import fit_regression_bias  # noqa: E402
from validation.ccf_rv_precision_metrics import _finite_err_kms  # noqa: E402
from validation.chunk_calibration import (  # noqa: E402
    build_intrinsic_scatter_model,
    build_layout_fallback_tables,
    stack_calibrated_exposure,
    summarize_sigma_rv_metrics,
)

logger = logging.getLogger(__name__)


def _wide_to_stack_chunks(wide: pd.DataFrame, estimator: str) -> pd.DataFrame:
    rv_col = f"rv_kms__{estimator}"
    err_col = f"rv_err_kms__{estimator}"
    if rv_col not in wide.columns:
        raise ValueError(f"missing column {rv_col}")
    out = wide.copy()
    out["rv_kms"] = out[rv_col].astype(float)
    raw_err = out[err_col].astype(float).values if err_col in out.columns else np.full(len(out), np.nan)
    out["rv_err_kms"] = _finite_err_kms(raw_err)
    out["gaia_dr3_id"] = out["gaia_dr3_id"].astype(str)
    out["chunk_key"] = out["chunk_key"].astype(str)
    if "teff_gaia" in out.columns:
        if "teff" not in out.columns:
            out["teff"] = out["teff_gaia"].astype(float)
        else:
            miss = ~np.isfinite(out["teff"].astype(float))
            out.loc[miss, "teff"] = out.loc[miss, "teff_gaia"].astype(float)
    return out


def stack_post_debias_exposures(
    wide: pd.DataFrame,
    estimator: str,
    summary_dir: Path,
    *,
    min_chunks: int = 3,
    cdf_weight_fraction: float = 0.9,
) -> tuple[pd.DataFrame, dict]:
    """
    Per-exposure calibrated RV after regression debias (chunk curve + stellar terms).

    σ_RV = formal IVW stack error on debiased chunks (``rv_err_calibrated_kms``).
    """
    _, per_object_stack, chosen_model, model_cmp = fit_regression_bias(wide, estimator, summary_dir)
    sample = per_object_stack[per_object_stack.get("sample_kept", True).astype(bool)]
    _, fallback = build_layout_fallback_tables(sample)
    intrinsic_model = build_intrinsic_scatter_model(per_object_stack)
    chunks = _wide_to_stack_chunks(wide, estimator)

    epoch_rows: list[dict] = []
    for file, g in chunks.groupby("file"):
        g = g[np.isfinite(g["rv_kms"].astype(float))].copy()
        if len(g) < min_chunks:
            continue
        stack = stack_calibrated_exposure(
            g,
            per_object=per_object_stack,
            fallback=fallback,
            intrinsic_model=intrinsic_model,
            min_chunks=min_chunks,
            cdf_weight_fraction=cdf_weight_fraction,
        )
        sig = float(stack.get("rv_err_calibrated_kms", np.nan))
        if not np.isfinite(sig) or sig <= 0:
            continue
        epoch_rows.append(
            {
                "file": str(file),
                "gaia_dr3_id": str(g["gaia_dr3_id"].iloc[0]),
                "estimator": estimator,
                "chosen_bias_model": chosen_model,
                "rv_calibrated_kms": stack["rv_calibrated_kms"],
                "rv_err_calibrated_kms": sig,
                "sigma_rv_core90_kms": stack["sigma_rv_core90_kms"],
                "chunk_scatter_calibrated_kms": stack["chunk_scatter_calibrated_kms"],
                "n_chunks_used": stack["n_chunks_used"],
                "n_chunks_core90": stack["n_chunks_core90"],
                "bias_source_mix": stack["bias_source_mix"],
            }
        )

    epochs = pd.DataFrame(epoch_rows)
    sigma = summarize_sigma_rv_metrics(epochs) if not epochs.empty else {}
    sc = epochs["chunk_scatter_calibrated_kms"].astype(float) if not epochs.empty else pd.Series(dtype=float)
    sc_ok = sc[np.isfinite(sc)]

    summary = {
        "estimator_name": estimator,
        "chosen_bias_model": chosen_model,
        "bias_cv_rmse_kms": float(
            model_cmp.loc[model_cmp["model"] == chosen_model, "cv_rmse_kms"].iloc[0]
        ),
        "n_exposures_stacked": int(len(epochs)),
        "median_sigma_rv_kms": float(sigma.get("median_sigma_rv_kms", np.nan)),
        "p90_sigma_rv_kms": float(sigma.get("p90_sigma_rv_kms", np.nan)),
        "min_sigma_rv_kms": float(sigma.get("min_sigma_rv_kms", np.nan)),
        "median_chunk_scatter_debiased_kms": float(np.median(sc_ok)) if len(sc_ok) else float("nan"),
        "p90_chunk_scatter_debiased_kms": float(np.percentile(sc_ok, 90)) if len(sc_ok) else float("nan"),
    }
    summary["post_debias_composite_score"] = post_debias_composite_score(summary)
    return epochs, summary


def post_debias_composite_score(summary: dict) -> float:
    """Lower is better; debiased σ_RV is primary."""
    sig = float(summary.get("median_sigma_rv_kms", np.nan))
    scatter = float(summary.get("median_chunk_scatter_debiased_kms", np.nan))
    parts, weights = [], []
    if np.isfinite(sig):
        parts.append(sig)
        weights.append(0.7)
    if np.isfinite(scatter):
        parts.append(scatter)
        weights.append(0.3)
    if not parts:
        return float("inf")
    w = np.asarray(weights, float) / np.sum(weights)
    return float(np.dot(w, parts))


def compare_estimators_post_debias(
    wide: pd.DataFrame,
    estimators: tuple[str, ...],
    summary_dir: Path,
    *,
    min_chunks: int = 3,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Rank estimators by post-debias median σ_RV (lower is better)."""
    rows: list[dict] = []
    for est in estimators:
        epochs, summary = stack_post_debias_exposures(
            wide,
            est,
            summary_dir,
            min_chunks=min_chunks,
        )
        if out_dir is not None:
            epochs.to_csv(out_dir / f"epochs_post_debias__{est}.csv", index=False)
        rows.append(summary)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["median_sigma_rv_kms", "p90_sigma_rv_kms", "median_chunk_scatter_debiased_kms"],
        ascending=[True, True, True],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wide-diagnostics", type=Path, required=True)
    ap.add_argument("--summary-dir", type=Path, default=REPO_ROOT / "output")
    ap.add_argument(
        "--estimators",
        default="gauss_offset,parabolic_ls,smooth_peak,bi_gauss,grid,auto",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("validation_output/ccf_estimator_study"))
    ap.add_argument("--min-chunks", type=int, default=3)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wide = pd.read_csv(args.wide_diagnostics)
    estimators = tuple(e.strip() for e in args.estimators.split(",") if e.strip())
    cmp_df = compare_estimators_post_debias(
        wide,
        estimators,
        args.summary_dir,
        min_chunks=args.min_chunks,
        out_dir=args.out_dir,
    )
    cmp_df.to_csv(args.out_dir / "estimator_comparison_post_debias.csv", index=False)

    winner = str(cmp_df.iloc[0]["estimator_name"]) if not cmp_df.empty else ""
    result = {
        "winner": winner,
        "comparison": cmp_df.to_dict(orient="records"),
    }
    (args.out_dir / "phase_C_post_debias_winner.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
