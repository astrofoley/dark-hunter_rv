"""Aggregate precision metrics for CCF RV estimator benchmark and phase gates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation.chunk_bias_lib import sample_mean_bias_curve
from validation.chunk_bias_regression import choose_best_model, compare_models
from validation.chunk_calibration import summarize_sigma_rv_metrics

LOW_SNR_LOG10_CUT = 0.5
_DEFAULT_CHUNK_ERR_KMS = 0.1


def _finite_err_kms(err: np.ndarray, *, floor_kms: float = _DEFAULT_CHUNK_ERR_KMS) -> np.ndarray:
    er = np.asarray(err, float)
    out = np.where(np.isfinite(er) & (er > 0), er, floor_kms)
    return out


def _exposure_chunk_scatter(df: pd.DataFrame, rv_col: str, err_col: str) -> pd.Series:
    """Per-exposure weighted RMS of chunk RVs about exposure mean."""
    rows: list[float] = []
    keys: list[str] = []
    for file, g in df.groupby("file"):
        rv = g[rv_col].astype(float).values
        er = g.get(err_col, pd.Series(np.full(len(g), 0.1))).astype(float).values
        ok = np.isfinite(rv)
        if ok.sum() < 2:
            continue
        rv_ok = rv[ok]
        er_ok = _finite_err_kms(er[ok])
        w = 1.0 / er_ok**2
        center = float(np.average(rv_ok, weights=w))
        resid = rv_ok - center
        sw = w / w.sum()
        scatter = float(np.sqrt(np.sum(sw * resid**2)))
        rows.append(scatter)
        keys.append(str(file))
    return pd.Series(rows, index=keys, dtype=float)


def _exposure_sigma_rv(df: pd.DataFrame, rv_col: str, err_col: str) -> pd.DataFrame:
    """Build per-exposure table with rv_err_calibrated_kms for summarize_sigma_rv_metrics."""
    epochs: list[dict] = []
    for file, g in df.groupby("file"):
        rv = g[rv_col].astype(float).values
        er = g.get(err_col, pd.Series(np.full(len(g), 0.1))).astype(float).values
        ok = np.isfinite(rv)
        if ok.sum() < 2:
            continue
        rv_ok = rv[ok]
        er_ok = _finite_err_kms(er[ok])
        w = 1.0 / er_ok**2
        sig = float(1.0 / np.sqrt(np.sum(w)))
        center = float(np.average(rv_ok, weights=w))
        scatter = float(np.std(rv_ok - center, ddof=1)) if len(rv_ok) >= 2 else float("nan")
        epochs.append(
            {
                "file": str(file),
                "rv_err_calibrated_kms": sig,
                "sigma_rv_core90_kms": sig,
                "chunk_scatter_calibrated_kms": scatter,
            }
        )
    return pd.DataFrame(epochs)


def _bias_metrics_from_object_table(bias_df: pd.DataFrame) -> dict[str, float]:
    if bias_df.empty or "weighted_mean_residual_kms" not in bias_df.columns:
        return {
            "bias_curve_rms_kms": float("nan"),
            "stellar_bias_cv_rmse_kms": float("nan"),
            "chosen_bias_model": "none",
        }
    work_bias = bias_df[np.isfinite(bias_df["weighted_mean_residual_kms"].astype(float))].copy()
    if work_bias.empty:
        return {
            "bias_curve_rms_kms": float("nan"),
            "stellar_bias_cv_rmse_kms": float("nan"),
            "chosen_bias_model": "none",
        }
    curve = sample_mean_bias_curve(work_bias)
    bias_curve_rms = float("nan")
    if not curve.empty:
        vals = curve["sample_mean_bias_kms"].astype(float).values
        vals = vals[np.isfinite(vals)]
        if len(vals):
            bias_curve_rms = float(np.sqrt(np.mean(vals**2)))

    work = work_bias.copy()
    if "teff" not in work.columns and "teff_gaia" in work.columns:
        work["teff"] = work["teff_gaia"]
    for col, default in [("logg", np.nan), ("mh", np.nan), ("log10_median_mask_ccf_peak_snr", 1.0)]:
        if col not in work.columns:
            work[col] = default
    orders = work["chunk_order"].astype(float)
    o_min, o_max = float(np.nanmin(orders)), float(np.nanmax(orders))
    work["chunk_order_norm"] = (orders - o_min) / max(o_max - o_min, 1.0)

    model_cmp = compare_models(work)
    chosen = choose_best_model(model_cmp)
    cv_rmse = float(model_cmp.loc[model_cmp["model"] == chosen, "cv_rmse_kms"].iloc[0])
    return {
        "bias_curve_rms_kms": bias_curve_rms,
        "stellar_bias_cv_rmse_kms": cv_rmse,
        "chosen_bias_model": str(chosen),
    }


def build_per_object_bias_table(
    chunk_df: pd.DataFrame,
    *,
    rv_col: str,
    err_col: str,
    min_measurements: int = 2,
) -> pd.DataFrame:
    """Per-object weighted mean chunk residual vs exposure-centered RV."""
    rows: list[dict] = []
    for (gid, ck), g in chunk_df.groupby(["gaia_dr3_id", "chunk_key"]):
        if len(g) < min_measurements:
            continue
        resid_vals: list[float] = []
        err_vals: list[float] = []
        for _, row in g.iterrows():
            file_rows = chunk_df[chunk_df["file"] == row["file"]]
            rv = file_rows[rv_col].astype(float).values
            er = file_rows.get(err_col, pd.Series(np.full(len(file_rows), 0.1))).astype(float).values
            ok = np.isfinite(rv)
            if ok.sum() < 2:
                continue
            rv_ok = rv[ok]
            er_ok = _finite_err_kms(er[ok])
            w = 1.0 / er_ok**2
            center = float(np.average(rv_ok, weights=w))
            rv_i = float(row[rv_col])
            if not np.isfinite(rv_i):
                continue
            resid_vals.append(rv_i - center)
            err_i = float(row.get(err_col, 0.1))
            err_vals.append(err_i if np.isfinite(err_i) and err_i > 0 else 0.1)
        if len(resid_vals) < min_measurements:
            continue
        rv_a = np.asarray(resid_vals, float)
        er_a = np.asarray(err_vals, float)
        w = 1.0 / er_a**2
        mu = float(np.average(rv_a, weights=w))
        stat = float(1.0 / np.sqrt(np.sum(w)))
        intrinsic = float(np.std(rv_a - mu, ddof=1)) if len(rv_a) >= 3 else 0.05
        rows.append(
            {
                "gaia_dr3_id": str(gid),
                "chunk_key": str(ck),
                "chunk_order": int(g["chunk_order"].iloc[0]) if "chunk_order" in g.columns else 0,
                "n_measurements": len(resid_vals),
                "weighted_mean_residual_kms": mu,
                "statistical_err_kms": stat,
                "intrinsic_scatter_kms": intrinsic,
                "teff": float(g["teff"].iloc[0]) if "teff" in g.columns else float("nan"),
                "logg": float(g["logg"].iloc[0]) if "logg" in g.columns else float("nan"),
                "mh": float(g["mh"].iloc[0]) if "mh" in g.columns else float("nan"),
                "log10_median_mask_ccf_peak_snr": float(
                    g["log10_peak_snr"].iloc[0] if "log10_peak_snr" in g.columns else np.nan
                ),
                "sample_kept": True,
            }
        )
    return pd.DataFrame(rows)


def summarize_estimator_metrics(
    chunk_df: pd.DataFrame,
    *,
    estimator: str,
    bias_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Precision metrics for one estimator column set in wide chunk table."""
    rv_col = f"rv_kms__{estimator}"
    err_col = f"rv_err_kms__{estimator}"
    if rv_col not in chunk_df.columns:
        rv_col = "rv_kms"
        err_col = "rv_err_kms"

    sub = chunk_df[np.isfinite(chunk_df[rv_col].astype(float))].copy()

    out: dict[str, float] = {
        "estimator_name": estimator,
        "n_chunks": float(len(sub)),
        "n_exposures": float(sub["file"].nunique()) if "file" in sub.columns and len(sub) else 0.0,
    }

    if sub.empty:
        out.update(
            {
                "median_sigma_rv_kms": float("nan"),
                "p90_sigma_rv_kms": float("nan"),
                "median_chunk_scatter_kms": float("nan"),
                "low_snr_finite_rate": 0.0,
                "bias_curve_rms_kms": float("nan"),
                "stellar_bias_cv_rmse_kms": float("nan"),
            }
        )
        return out

    epochs = _exposure_sigma_rv(sub, rv_col, err_col)
    sigma = summarize_sigma_rv_metrics(epochs) if not epochs.empty else {}
    out["median_sigma_rv_kms"] = float(sigma.get("median_sigma_rv_kms", np.nan))
    out["p90_sigma_rv_kms"] = float(sigma.get("p90_sigma_rv_kms", np.nan))

    scatters = _exposure_chunk_scatter(sub, rv_col, err_col)
    sc_ok = scatters[np.isfinite(scatters)]
    out["median_chunk_scatter_kms"] = float(np.median(sc_ok)) if len(sc_ok) else float("nan")

    if "peak_snr" in sub.columns:
        snr = sub["peak_snr"].astype(float)
        log_snr = np.log10(np.clip(snr, 1e-9, None))
        low = log_snr < LOW_SNR_LOG10_CUT
        if low.sum() > 0:
            finite_low = np.isfinite(sub.loc[low, rv_col].astype(float))
            out["low_snr_finite_rate"] = float(finite_low.mean())
        else:
            out["low_snr_finite_rate"] = 1.0
    else:
        out["low_snr_finite_rate"] = float("nan")

    if bias_df is None:
        bias_df = build_per_object_bias_table(sub, rv_col=rv_col, err_col=err_col)
    bias_m = _bias_metrics_from_object_table(bias_df)
    out["bias_curve_rms_kms"] = bias_m["bias_curve_rms_kms"]
    out["stellar_bias_cv_rmse_kms"] = bias_m["stellar_bias_cv_rmse_kms"]
    out["chosen_bias_model"] = bias_m["chosen_bias_model"]
    return out


def composite_score(metrics: dict[str, float]) -> float:
    """Lower is better. Weight precision + bias + coverage penalty."""
    sig = float(metrics.get("median_sigma_rv_kms", np.nan))
    scatter = float(metrics.get("median_chunk_scatter_kms", np.nan))
    bias = float(metrics.get("bias_curve_rms_kms", np.nan))
    finite = float(metrics.get("low_snr_finite_rate", np.nan))
    parts = []
    weights = []
    if np.isfinite(sig):
        parts.append(sig)
        weights.append(0.4)
    if np.isfinite(scatter):
        parts.append(scatter)
        weights.append(0.3)
    if np.isfinite(bias):
        parts.append(bias)
        weights.append(0.2)
    if np.isfinite(finite):
        parts.append(1.0 - finite)
        weights.append(0.1)
    if not parts:
        return float("inf")
    w = np.asarray(weights, float)
    w = w / w.sum()
    return float(np.dot(w, parts))
