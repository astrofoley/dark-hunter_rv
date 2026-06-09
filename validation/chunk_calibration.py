"""Apply per-chunk bias and stat+intrinsic weights to form calibrated exposure RVs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from validation.chunk_bias_lib import fit_linear_model, predict_linear, standardize
from validation.plot_chunk_residuals import (
    DEFAULT_CHUNK_MAX_DELTA_KMS,
    DEFAULT_CHUNK_OUTLIER_SIGMA,
    _load_chunk_rows,
    apply_spectrum_chunk_outlier_clip,
)


def _sigma_total_kms(stat_kms: float, intrinsic_kms: float, *, floor_kms: float = 1e-6) -> float:
    s2 = 0.0
    if np.isfinite(stat_kms) and stat_kms > 0:
        s2 += float(stat_kms) ** 2
    if np.isfinite(intrinsic_kms) and intrinsic_kms > 0:
        s2 += float(intrinsic_kms) ** 2
    if s2 <= 0:
        return floor_kms
    return float(max(np.sqrt(s2), floor_kms))


def _sigma_rv_from_weights(weights: np.ndarray) -> float:
    w = weights[np.isfinite(weights) & (weights > 0)]
    if w.size == 0:
        return float("nan")
    return float(1.0 / np.sqrt(np.sum(w)))


@dataclass
class IntrinsicScatterModel:
    """Per-chunk sample intrinsic scatter with optional stellar scaling (not edge placement)."""

    chunk_intrinsic_kms: dict[str, float]
    stellar_coef: np.ndarray | None = None
    stellar_state: dict | None = None

    def predict(
        self,
        chunk_key: str,
        *,
        teff: float = np.nan,
        logg: float = np.nan,
        mh: float = np.nan,
    ) -> float:
        base = float(self.chunk_intrinsic_kms.get(str(chunk_key), 0.0))
        if base <= 0 or self.stellar_coef is None or self.stellar_state is None:
            return max(base, 0.0)
        if not np.isfinite(teff):
            return max(base, 0.0)
        X = _stellar_design_row(teff, logg, mh, self.stellar_state)
        log_mult = float(predict_linear(X, self.stellar_coef)[0])
        if not np.isfinite(log_mult):
            return max(base, 0.0)
        return max(base * float(np.exp(log_mult)), 0.0)


def _stellar_design_row(
    teff: float,
    logg: float,
    mh: float,
    state: dict,
) -> np.ndarray:
    cols = []
    for val, key in [(teff, "teff"), (logg, "logg"), (mh, "mh")]:
        mu, sd = state[key]
        cols.append((float(val) - mu) / sd if np.isfinite(val) else 0.0)
    return np.asarray([[1.0, cols[0], cols[1], cols[2]]], dtype=float)


def build_intrinsic_scatter_model(bias: pd.DataFrame) -> IntrinsicScatterModel:
    """Sample per-chunk intrinsic scatter + global stellar log-multiplier."""
    chunk_intrinsic: dict[str, float] = {}
    for ck, g in bias.groupby("chunk_key"):
        intrinsic = g["intrinsic_scatter_kms"].astype(float).values
        intrinsic = intrinsic[np.isfinite(intrinsic) & (intrinsic > 0)]
        if intrinsic.size:
            chunk_intrinsic[str(ck)] = float(np.median(intrinsic))

    rows = []
    for _, r in bias.iterrows():
        intr = float(r.get("intrinsic_scatter_kms", np.nan))
        ck = str(r["chunk_key"])
        base = chunk_intrinsic.get(ck, np.nan)
        if not np.isfinite(intr) or intr <= 0 or not np.isfinite(base) or base <= 0:
            continue
        rows.append(
            {
                "teff": float(r.get("teff", np.nan)),
                "logg": float(r.get("logg", np.nan)),
                "mh": float(r.get("mh", np.nan)),
                "log_ratio": float(np.log(intr / base)),
            }
        )
    stellar_coef = None
    stellar_state: dict | None = None
    if len(rows) >= 20:
        df = pd.DataFrame(rows)
        state: dict = {}
        X_cols = [np.ones(len(df))]
        for col, key in [("teff", "teff"), ("logg", "logg"), ("mh", "mh")]:
            v = df[col].astype(float).values
            vs, mu, sd = standardize(v)
            state[key] = (mu, sd)
            X_cols.append(vs)
        X = np.column_stack(X_cols)
        y = df["log_ratio"].astype(float).values
        fit = fit_linear_model(X, y)
        if fit.get("coef") is not None:
            stellar_coef = fit["coef"]
            stellar_state = state

    return IntrinsicScatterModel(chunk_intrinsic_kms=chunk_intrinsic, stellar_coef=stellar_coef, stellar_state=stellar_state)


def build_layout_fallback_tables(bias: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-object chunk table and sample fallback (bias + historical stat/intrinsic)."""
    fallback_rows = []
    for ck, g in bias.groupby("chunk_key"):
        stat = g["statistical_err_kms"].astype(float).values
        intrinsic = g["intrinsic_scatter_kms"].astype(float).values
        b = g["weighted_mean_residual_kms"].astype(float).values
        sig2 = np.array(
            [_sigma_total_kms(float(s), float(i)) ** 2 for s, i in zip(stat, intrinsic, strict=True)],
            dtype=float,
        )
        w = 1.0 / np.maximum(sig2, 1e-12)
        w = np.where(np.isfinite(w), w, 0.0)
        fallback_rows.append(
            {
                "chunk_key": str(ck),
                "bias_kms": float(np.average(b, weights=w)) if w.sum() > 0 else float(np.nanmean(b)),
                "statistical_err_kms": float(np.nanmedian(stat[np.isfinite(stat) & (stat > 0)]))
                if np.any(np.isfinite(stat) & (stat > 0))
                else np.nan,
                "intrinsic_scatter_kms": float(np.nanmedian(intrinsic[np.isfinite(intrinsic) & (intrinsic > 0)]))
                if np.any(np.isfinite(intrinsic) & (intrinsic > 0))
                else 0.0,
            }
        )
    return bias, pd.DataFrame(fallback_rows)


def load_chunk_bias_tables(bias_csv: pd.DataFrame | str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_object_chunk, sample_fallback_by_chunk)."""
    if isinstance(bias_csv, (str, Path)):
        bias_csv = pd.read_csv(bias_csv)
    df = bias_csv.copy()
    df["gaia_dr3_id"] = df["gaia_dr3_id"].astype(str)
    df["chunk_key"] = df["chunk_key"].astype(str)
    if "sample_kept" in df.columns:
        sample_src = df[df["sample_kept"].astype(bool)]
    else:
        sample_src = df
    _, fallback = build_layout_fallback_tables(sample_src)
    return df, fallback


def lookup_chunk_bias(
    gaia_id: str,
    chunk_key: str,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
) -> tuple[float, str]:
    """Return debias correction (km/s) and source label."""
    po = per_object[
        (per_object["gaia_dr3_id"] == str(gaia_id))
        & (per_object["chunk_key"] == str(chunk_key))
    ]
    if len(po):
        return float(po.iloc[0]["weighted_mean_residual_kms"]), "object"
    fb = fallback[fallback["chunk_key"] == str(chunk_key)]
    if len(fb) and np.isfinite(fb.iloc[0]["bias_kms"]):
        return float(fb.iloc[0]["bias_kms"]), "sample_fallback"
    return float("nan"), "missing"


def lookup_chunk_calibration(
    gaia_id: str,
    chunk_key: str,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
) -> tuple[float, float, float, str]:
    """Return bias_kms, stat_err, intrinsic_scatter, source (legacy aggregate stat)."""
    bias, src = lookup_chunk_bias(gaia_id, chunk_key, per_object, fallback)
    fb = fallback[fallback["chunk_key"] == str(chunk_key)]
    if len(fb):
        r = fb.iloc[0]
        return (
            bias,
            float(r.get("statistical_err_kms", np.nan)),
            float(r.get("intrinsic_scatter_kms", np.nan)),
            src,
        )
    return bias, float("nan"), float("nan"), src


def _chunk_weight(
    row: pd.Series,
    *,
    per_object: pd.DataFrame,
    intrinsic_model: IntrinsicScatterModel,
    fallback: pd.DataFrame,
) -> tuple[float, float, float, str] | None:
    """Per-chunk debiased RV and IVW weight using this spectrum's stat error."""
    bias, src = lookup_chunk_bias(
        str(row["gaia_dr3_id"]),
        str(row["chunk_key"]),
        per_object,
        fallback,
    )
    if not np.isfinite(bias):
        return None
    stat = float(row.get("rv_err_kms", np.nan))
    if not np.isfinite(stat) or stat <= 0:
        return None
    intrinsic = intrinsic_model.predict(
        str(row["chunk_key"]),
        teff=float(row.get("teff", np.nan)),
        logg=float(row.get("logg", np.nan)),
        mh=float(row.get("mh", np.nan)),
    )
    sigma = _sigma_total_kms(stat, intrinsic)
    rv_db = float(row["rv_kms"]) - bias
    return rv_db, 1.0 / sigma**2, sigma, src


def select_chunks_cdf_weight(
    sigmas: np.ndarray,
    weights: np.ndarray,
    *,
    min_chunks: int,
    weight_fraction: float = 0.9,
) -> tuple[np.ndarray, float]:
    """
    CDF selection: add chunks in order of increasing σ until ``weight_fraction`` of
    total IVW weight is captured (best chunks first).
    """
    if len(sigmas) == 0:
        return np.array([], dtype=int), float("nan")
    order = np.argsort(sigmas)
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted)
    total = float(cum[-1])
    if total <= 0:
        return order[: max(min_chunks, 1)], float("nan")
    target = weight_fraction * total
    k = int(np.searchsorted(cum, target, side="left")) + 1
    k = max(k, min_chunks)
    k = min(k, len(order))
    idx = order[:k]
    return idx, _sigma_rv_from_weights(w_sorted[:k])


def stack_calibrated_exposure(
    chunk_df: pd.DataFrame,
    *,
    per_object: pd.DataFrame,
    fallback: pd.DataFrame,
    intrinsic_model: IntrinsicScatterModel | None = None,
    min_chunks: int = 3,
    cdf_weight_fraction: float = 0.9,
) -> dict[str, float | int | str]:
    """One exposure (file) from clipped chunk rows → calibrated stack RV and σ_RV."""
    kept = chunk_df[chunk_df["chunk_kept"].astype(bool)] if "chunk_kept" in chunk_df.columns else chunk_df
    if intrinsic_model is None:
        intrinsic_model = build_intrinsic_scatter_model(per_object if len(per_object) else fallback)

    rv_d: list[float] = []
    wts: list[float] = []
    sigmas: list[float] = []
    sources: list[str] = []
    for _, r in kept.iterrows():
        parsed = _chunk_weight(
            r, per_object=per_object, intrinsic_model=intrinsic_model, fallback=fallback
        )
        if parsed is None:
            continue
        rv_db, wt, sigma, src = parsed
        rv_d.append(rv_db)
        wts.append(wt)
        sigmas.append(sigma)
        sources.append(src)

    if len(rv_d) < min_chunks:
        return {
            "rv_calibrated_kms": np.nan,
            "rv_err_calibrated_kms": np.nan,
            "sigma_rv_core90_kms": np.nan,
            "chunk_scatter_calibrated_kms": np.nan,
            "n_chunks_used": len(rv_d),
            "n_chunks_core90": 0,
            "bias_source_mix": "",
        }

    rv_arr = np.asarray(rv_d, float)
    w_arr = np.asarray(wts, float)
    sig_arr = np.asarray(sigmas, float)
    mu = float(np.sum(w_arr * rv_arr) / np.sum(w_arr))
    err = _sigma_rv_from_weights(w_arr)
    _, err_core = select_chunks_cdf_weight(
        sig_arr, w_arr, min_chunks=min_chunks, weight_fraction=cdf_weight_fraction
    )
    if len(rv_arr) > 1:
        scatter = float(np.std(rv_arr - mu, ddof=1))
    else:
        scatter = 0.0
    core_idx, _ = select_chunks_cdf_weight(
        sig_arr, w_arr, min_chunks=min_chunks, weight_fraction=cdf_weight_fraction
    )
    return {
        "rv_calibrated_kms": mu,
        "rv_err_calibrated_kms": err,
        "sigma_rv_core90_kms": err_core,
        "chunk_scatter_calibrated_kms": scatter,
        "n_chunks_used": int(len(rv_d)),
        "n_chunks_core90": int(len(core_idx)),
        "bias_source_mix": ",".join(sorted(set(sources))),
    }


def summarize_sigma_rv_metrics(epochs: pd.DataFrame) -> dict[str, float]:
    """Layout-level σ_RV summary: best (min) and distribution tails."""
    sig = epochs["rv_err_calibrated_kms"].astype(float)
    sig_ok = sig[np.isfinite(sig) & (sig > 0)]
    if len(sig_ok) == 0:
        return {
            "min_sigma_rv_kms": float("nan"),
            "p10_sigma_rv_kms": float("nan"),
            "median_sigma_rv_kms": float("nan"),
            "p90_sigma_rv_kms": float("nan"),
            "min_sigma_rv_core90_kms": float("nan"),
        }
    core = epochs.get("sigma_rv_core90_kms", pd.Series(np.nan, index=epochs.index)).astype(float)
    core_ok = core[np.isfinite(core) & (core > 0)]
    return {
        "min_sigma_rv_kms": float(np.min(sig_ok)),
        "p10_sigma_rv_kms": float(np.percentile(sig_ok, 10)),
        "median_sigma_rv_kms": float(np.median(sig_ok)),
        "p90_sigma_rv_kms": float(np.percentile(sig_ok, 90)),
        "min_sigma_rv_core90_kms": float(np.min(core_ok)) if len(core_ok) else float("nan"),
    }


def build_calibrated_exposure_table(
    diagnostics_glob: str,
    bias_csv: str | Path | pd.DataFrame,
    *,
    clip_sigma: float = DEFAULT_CHUNK_OUTLIER_SIGMA,
    clip_max_delta_kms: float = DEFAULT_CHUNK_MAX_DELTA_KMS,
    min_chunks: int = 3,
) -> pd.DataFrame:
    tab = _load_chunk_rows(diagnostics_glob)
    if tab.empty:
        return tab
    tab = apply_spectrum_chunk_outlier_clip(
        tab, nsigma=clip_sigma, max_delta_kms=clip_max_delta_kms
    )
    per_object, fallback = load_chunk_bias_tables(bias_csv)
    intrinsic_model = build_intrinsic_scatter_model(per_object)

    rows: list[dict] = []
    for file_label, g in tab.groupby("file", sort=False):
        stack = stack_calibrated_exposure(
            g,
            per_object=per_object,
            fallback=fallback,
            intrinsic_model=intrinsic_model,
            min_chunks=min_chunks,
        )
        r0 = g.iloc[0]
        pipeline_rv = float(r0.get("exposure_rv_kms_pipeline", r0.get("exposure_rv_kms", np.nan)))
        if not np.isfinite(pipeline_rv):
            pipeline_rv = float(r0["exposure_rv_kms"])
        rows.append(
            {
                "gaia_dr3_id": str(r0["gaia_dr3_id"]),
                "file": str(file_label),
                "mjd": float(r0["mjd"]),
                "teff": float(r0["teff"]),
                "log10_median_mask_ccf_peak_snr": float(r0["log10_median_mask_ccf_peak_snr"]),
                "rv_pipeline_kms": pipeline_rv,
                "rv_calibrated_kms": stack["rv_calibrated_kms"],
                "rv_err_calibrated_kms": stack["rv_err_calibrated_kms"],
                "sigma_rv_core90_kms": stack["sigma_rv_core90_kms"],
                "chunk_scatter_calibrated_kms": stack["chunk_scatter_calibrated_kms"],
                "n_chunks_used": stack["n_chunks_used"],
                "n_chunks_core90": stack["n_chunks_core90"],
                "bias_source_mix": stack["bias_source_mix"],
            }
        )
    return pd.DataFrame(rows)


def find_apf_apf_pairs(apf: pd.DataFrame, *, pair_window_days: float = 7.0) -> pd.DataFrame:
    from itertools import combinations

    from validation.rv_overlap_lib import _pair_row, enrich_pairs_with_deltas

    rows: list[dict] = []
    for gid, g in apf.groupby("gaia_dr3_id"):
        gid = str(gid)
        name = str(g["name"].iloc[0]) if "name" in g.columns else gid
        recs = []
        for _, r in g.iterrows():
            recs.append(
                {
                    "epoch_id": f"apf:{r['file']}",
                    "mjd": float(r["mjd"]),
                    "rv_kms": float(r["rv_kms"]),
                    "rv_err_kms": float(r.get("rv_err_kms", np.nan)),
                }
            )
        for a, b in combinations(recs, 2):
            dt = abs(float(a["mjd"]) - float(b["mjd"]))
            if dt <= pair_window_days and a["epoch_id"] != b["epoch_id"]:
                rows.append(_pair_row("apf_apf", gid, name, a, b, dt))
    if not rows:
        return pd.DataFrame()
    return enrich_pairs_with_deltas(pd.DataFrame(rows))


def relative_pair_table(
    epochs: pd.DataFrame,
    *,
    rv_col: str,
    err_col: str | None = None,
    pair_window_days: float = 7.0,
) -> pd.DataFrame:
    apf = epochs.copy()
    apf["rv_kms"] = apf[rv_col].astype(float)
    if err_col and err_col in apf.columns:
        apf["rv_err_kms"] = apf[err_col].astype(float)
    else:
        apf["rv_err_kms"] = np.nan
    if "name" not in apf.columns:
        apf["name"] = apf["gaia_dr3_id"].astype(str)
    return find_apf_apf_pairs(apf, pair_window_days=pair_window_days)


def summarize_relative_gate(pairs: pd.DataFrame, *, goal_kms: float = 0.1) -> dict[str, float | int]:
    if pairs.empty:
        return {"n_pairs": 0}
    abs_dv = pairs["abs_delta_rv_kms"].astype(float)
    return {
        "n_pairs": int(len(pairs)),
        "n_stars": int(pairs["gaia_dr3_id"].nunique()),
        "median_abs_delta_rv_kms": float(np.median(abs_dv)),
        "p90_abs_delta_rv_kms": float(np.percentile(abs_dv, 90)),
        "max_abs_delta_rv_kms": float(np.max(abs_dv)),
        "rms_delta_rv_kms": float(np.sqrt(np.mean(pairs["delta_rv_kms"].astype(float) ** 2))),
        "frac_below_goal": float(np.mean(abs_dv < goal_kms)),
    }
