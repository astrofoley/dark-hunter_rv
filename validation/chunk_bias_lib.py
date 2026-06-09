"""Shared helpers for chunk bias regression and layout evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary
from darkhunter_rv.summary_paths import discover_summary_files, parse_object_id_from_summary
from validation.plot_chunk_residuals import _chunk_sort_key, _load_name_lookup


def meta_float(meta: dict | None, key: str) -> float:
    if not meta:
        return float("nan")
    v = meta.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_stellar_metadata(summary_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for sp in discover_summary_files(summary_dir):
        gid = parse_object_id_from_summary(sp)
        if not gid:
            continue
        meta = parse_gaia_metadata_from_star_summary(sp)
        rows.append(
            {
                "gaia_dr3_id": str(gid),
                "teff_gaia": meta_float(meta, "Teff"),
                "logg": meta_float(meta, "logg"),
                "mh": meta_float(meta, "MH"),
                "ruwe": meta_float(meta, "RUWE"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["gaia_dr3_id", "teff_gaia", "logg", "mh", "ruwe"])
    return pd.DataFrame(rows).drop_duplicates("gaia_dr3_id")


def load_bias_regression_table(
    bias_csv: Path,
    *,
    summary_dir: Path,
    long_csv_glob: str | None = None,
    sample_kept_only: bool = True,
) -> pd.DataFrame:
    """Join per-object chunk biases with stellar parameters and optional S/N."""
    df = pd.read_csv(bias_csv)
    df["gaia_dr3_id"] = df["gaia_dr3_id"].astype(str)
    df["chunk_key"] = df["chunk_key"].astype(str)
    if sample_kept_only and "sample_kept" in df.columns:
        df = df[df["sample_kept"].astype(bool)].copy()

    meta = load_stellar_metadata(summary_dir)
    df = df.merge(meta, on="gaia_dr3_id", how="left")

    if "chunk_order" not in df.columns:
        df["chunk_order"] = df["chunk_key"].map(lambda ck: _chunk_sort_key(str(ck))[0])

    snr_rows: list[dict] = []
    if long_csv_glob:
        from glob import glob as glob_paths

        for path in glob_paths(long_csv_glob):
            try:
                sub = pd.read_csv(path, usecols=["gaia_dr3_id", "log10_median_mask_ccf_peak_snr"])
            except ValueError:
                sub = pd.read_csv(path)
                if "log10_median_mask_ccf_peak_snr" not in sub.columns:
                    continue
                sub = sub[["gaia_dr3_id", "log10_median_mask_ccf_peak_snr"]]
            snr_rows.append(sub.drop_duplicates("gaia_dr3_id"))
    if snr_rows:
        snr = pd.concat(snr_rows, ignore_index=True)
        snr["gaia_dr3_id"] = snr["gaia_dr3_id"].astype(str)
        snr = snr.groupby("gaia_dr3_id", as_index=False)["log10_median_mask_ccf_peak_snr"].median()
        df = df.merge(snr, on="gaia_dr3_id", how="left")

    if "teff" not in df.columns:
        df["teff"] = df["teff_gaia"]
    else:
        miss = ~np.isfinite(df["teff"].astype(float))
        df.loc[miss, "teff"] = df.loc[miss, "teff_gaia"]

    orders = df["chunk_order"].astype(float)
    o_min, o_max = float(np.nanmin(orders)), float(np.nanmax(orders))
    span = max(o_max - o_min, 1.0)
    df["chunk_order_norm"] = (orders - o_min) / span
    return df


def truncated_cubic_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Design matrix: [1, x, (x-k1)^3+, ...] for natural-ish cubic spline."""
    x = np.asarray(x, float)
    cols = [np.ones_like(x), x]
    for k in knots:
        cols.append(np.maximum(0.0, x - float(k)) ** 3)
    return np.column_stack(cols)


def standardize(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = np.asarray(x, float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (x - mu) / sd, mu, sd


def fit_linear_model(X: np.ndarray, y: np.ndarray, *, weights: np.ndarray | None = None) -> dict:
    """Weighted least squares with pseudo-inverse for stability."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if weights is not None:
        w = np.asarray(weights, float)
        ok &= np.isfinite(w) & (w > 0)
    else:
        w = np.ones(len(y))
    if ok.sum() < X.shape[1] + 1:
        return {"coef": None, "rss": float("nan"), "n": int(ok.sum())}
    Xo = X[ok]
    yo = y[ok]
    wo = w[ok]
    sw = np.sqrt(wo)
    beta, *_ = np.linalg.lstsq(Xo * sw[:, None], yo * sw, rcond=None)
    resid = yo - Xo @ beta
    rss = float(np.sum(wo * resid**2))
    return {"coef": beta, "rss": rss, "n": int(ok.sum()), "mask": ok}


def predict_linear(X: np.ndarray, coef: np.ndarray | None) -> np.ndarray:
    if coef is None:
        return np.full(len(X), np.nan)
    return np.asarray(X, float) @ np.asarray(coef, float)


def nested_f_test(rss_small: float, rss_large: float, n: int, p_small: int, p_large: int) -> dict:
    """Compare nested models (large includes small)."""
    if n <= p_large + 1 or not np.isfinite(rss_small) or not np.isfinite(rss_large):
        return {"f_stat": float("nan"), "p_value": float("nan"), "delta_p": p_large - p_small}
    df1 = p_large - p_small
    df2 = n - p_large
    if df1 <= 0 or df2 <= 0 or rss_large <= 0:
        return {"f_stat": float("nan"), "p_value": float("nan"), "delta_p": df1}
    f_stat = ((rss_small - rss_large) / df1) / (rss_large / df2)
    from scipy import stats

    p_value = float(1.0 - stats.f.cdf(f_stat, df1, df2))
    return {"f_stat": float(f_stat), "p_value": p_value, "delta_p": int(df1)}


def leave_one_object_cv_rmse(
    df: pd.DataFrame,
    *,
    build_X,
    y_col: str = "weighted_mean_residual_kms",
    weight_col: str = "statistical_err_kms",
) -> float:
    """Leave-one-star-out CV RMSE for bias predictions."""
    preds: list[float] = []
    obs: list[float] = []
    for gid, hold in df.groupby("gaia_dr3_id"):
        train = df[df["gaia_dr3_id"] != gid]
        test = hold
        if len(train) < 5 or len(test) < 1:
            continue
        fit = build_X(train, fit=True)
        if fit.get("coef") is None:
            continue
        X_test = build_X(test, fit=False, state=fit)
        y_hat = predict_linear(X_test, fit["coef"])
        y_true = test[y_col].astype(float).values
        ok = np.isfinite(y_hat) & np.isfinite(y_true)
        preds.extend(y_hat[ok].tolist())
        obs.extend(y_true[ok].tolist())
    if not preds:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(preds) - np.asarray(obs)) ** 2)))


def sample_mean_bias_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Sample-wide weighted mean bias vs chunk order (bias curve)."""
    rows = []
    for ck, g in df.groupby("chunk_key"):
        bias = g["weighted_mean_residual_kms"].astype(float).values
        stat = g["statistical_err_kms"].astype(float).values
        intrinsic = g.get("intrinsic_scatter_kms", pd.Series(np.zeros(len(g)))).astype(float).values
        sig2 = stat**2 + intrinsic**2
        sig2 = np.where(np.isfinite(sig2) & (sig2 > 0), sig2, np.nan)
        w = 1.0 / sig2
        w = np.where(np.isfinite(w), w, 0.0)
        if w.sum() <= 0:
            mu = float(np.nanmean(bias))
        else:
            mu = float(np.average(bias, weights=w))
        rows.append(
            {
                "chunk_key": str(ck),
                "chunk_order": int(_chunk_sort_key(str(ck))[0]),
                "sample_mean_bias_kms": mu,
                "n_objects": int(len(g)),
            }
        )
    out = pd.DataFrame(rows).sort_values("chunk_order")
    return out.reset_index(drop=True)
