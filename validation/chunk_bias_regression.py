#!/usr/bin/env python3
"""
Regress chunk biases on stellar parameters and echelle-order position (bias curve).

Fits nested models (intercept, chunk curve, stellar covariates, interactions), tests
significance, exports regression-adjusted bias table, and optionally re-runs the
relative calibration gate.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.chunk_bias_regression \\
    --bias-csv validation_output/chunk_residuals/per_object_chunk_bias.csv \\
    --summary-dir output \\
    --out-dir validation_output/chunk_bias_regression
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.chunk_bias_lib import (  # noqa: E402
    fit_linear_model,
    load_bias_regression_table,
    nested_f_test,
    predict_linear,
    sample_mean_bias_curve,
    standardize,
    truncated_cubic_basis,
)
from validation.plot_chunk_residuals import _load_name_lookup  # noqa: E402

logger = logging.getLogger(__name__)

SIGNIFICANCE_P = 0.05
CV_IMPROVEMENT_KMS = 0.02


@dataclass
class ModelSpec:
    name: str
    description: str
    feature_builder: str


def _weights_from_df(df: pd.DataFrame) -> np.ndarray:
    stat = df["statistical_err_kms"].astype(float).values
    intrinsic = df.get("intrinsic_scatter_kms", pd.Series(np.zeros(len(df)))).astype(float).values
    sig2 = stat**2 + intrinsic**2
    sig2 = np.where(np.isfinite(sig2) & (sig2 > 0), sig2, np.nan)
    w = 1.0 / sig2
    return np.where(np.isfinite(w), w, 1.0)


def _build_features(
    df: pd.DataFrame,
    *,
    mode: str,
    fit: bool,
    state: dict | None = None,
) -> np.ndarray | dict:
    state = dict(state or {})
    x_order = df["chunk_order_norm"].astype(float).values
    if mode == "intercept":
        X = np.ones((len(df), 1))
        if fit:
            return {"X": X, **state}
        return X
    if mode in ("curve", "curve_stellar", "curve_stellar_interaction", "full"):
        knots = state.get("knots")
        if fit or knots is None:
            knots = np.quantile(x_order[np.isfinite(x_order)], [0.25, 0.5, 0.75])
            state["knots"] = knots
        X_parts = [truncated_cubic_basis(x_order, state["knots"])]
    else:
        X_parts = [np.ones((len(df), 1))]

    if mode in ("stellar", "curve_stellar", "curve_stellar_interaction", "full"):
        for col, key in [("teff", "teff"), ("logg", "logg"), ("mh", "mh"), ("log10_median_mask_ccf_peak_snr", "log_snr")]:
            v = df[col].astype(float).values if col in df.columns else np.full(len(df), np.nan)
            if fit or key not in state:
                vs, mu, sd = standardize(v)
                state[key] = (mu, sd)
            else:
                mu, sd = state[key]
                vs = (v - mu) / sd if np.isfinite(sd) and sd > 0 else v * 0.0
            X_parts.append(vs.reshape(-1, 1))

    if mode in ("curve_stellar_interaction", "full"):
        teff = df["teff"].astype(float).values if "teff" in df.columns else np.full(len(df), np.nan)
        if fit or "teff_int" not in state:
            teff_s, mu, sd = standardize(teff)
            state["teff_int"] = (mu, sd)
        else:
            mu, sd = state["teff_int"]
            teff_s = (teff - mu) / sd if np.isfinite(sd) and sd > 0 else teff * 0.0
        X_parts.append((teff_s * x_order).reshape(-1, 1))

    X = np.hstack(X_parts)
    if fit:
        return {"X": X, **state}
    return X


def _fit_model(df: pd.DataFrame, mode: str) -> dict:
    pack = _build_features(df, mode=mode, fit=True)
    X = pack["X"]
    y = df["weighted_mean_residual_kms"].astype(float).values
    w = _weights_from_df(df)
    fit = fit_linear_model(X, y, weights=w)
    fit["state"] = pack
    fit["mode"] = mode
    fit["p"] = int(X.shape[1])
    if fit.get("coef") is not None:
        fit["y_hat"] = predict_linear(X, fit["coef"])
    else:
        fit["y_hat"] = np.full(len(df), np.nan)
    return fit


def leave_one_object_cv_rmse_mode(df: pd.DataFrame, mode: str) -> float:
    preds: list[float] = []
    obs: list[float] = []
    for gid in df["gaia_dr3_id"].unique():
        test = df[df["gaia_dr3_id"] == gid]
        train = df[df["gaia_dr3_id"] != gid]
        if len(train) < 8:
            continue
        fit = _fit_model(train, mode)
        if fit.get("coef") is None:
            continue
        X_test = _build_features(test, mode=mode, fit=False, state=fit["state"])
        y_hat = predict_linear(X_test, fit["coef"])
        y_true = test["weighted_mean_residual_kms"].astype(float).values
        ok = np.isfinite(y_hat) & np.isfinite(y_true)
        preds.extend(y_hat[ok].tolist())
        obs.extend(y_true[ok].tolist())
    if not preds:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(preds) - np.asarray(obs)) ** 2)))


def per_chunk_stellar_slopes(df: pd.DataFrame, *, min_objects: int = 4) -> pd.DataFrame:
    rows = []
    for ck, g in df.groupby("chunk_key"):
        if len(g) < min_objects:
            continue
        teff = g["teff"].astype(float).values
        bias = g["weighted_mean_residual_kms"].astype(float).values
        ok = np.isfinite(teff) & np.isfinite(bias)
        if ok.sum() < min_objects:
            continue
        X = np.column_stack([np.ones(ok.sum()), teff[ok]])
        fit = fit_linear_model(X, bias[ok])
        if fit.get("coef") is None:
            continue
        slope = float(fit["coef"][1])
        from scipy import stats

        # rough slope SE
        resid = bias[ok] - X @ fit["coef"]
        mse = float(np.sum(resid**2) / max(len(resid) - 2, 1))
        cov = mse * np.linalg.pinv(X.T @ X)
        slope_se = float(np.sqrt(cov[1, 1])) if cov.shape == (2, 2) else float("nan")
        t_stat = slope / slope_se if np.isfinite(slope_se) and slope_se > 0 else float("nan")
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(len(resid) - 2, 1)))) if np.isfinite(t_stat) else float("nan")
        rows.append(
            {
                "chunk_key": str(ck),
                "chunk_order": int(g["chunk_order"].iloc[0]),
                "n_objects": int(ok.sum()),
                "teff_intercept_kms": float(fit["coef"][0]),
                "teff_slope_kms_per_k": slope,
                "teff_slope_se": slope_se,
                "p_value": p_value,
                "significant": bool(np.isfinite(p_value) and p_value < SIGNIFICANCE_P),
            }
        )
    return pd.DataFrame(rows)


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    modes = [
        ("intercept", "Intercept only"),
        ("curve", "Cubic spline on chunk order (bias curve)"),
        ("stellar", "Teff, logg, [M/H], log10 S/N"),
        ("curve_stellar", "Bias curve + stellar covariates"),
        ("curve_stellar_interaction", "Bias curve + stellar + Teff×order interaction"),
    ]
    fits = {m: _fit_model(df, m) for m, _ in modes}
    intercept = fits["intercept"]
    rows = []
    for m, desc in modes:
        fit = fits[m]
        cv = leave_one_object_cv_rmse_mode(df, m)
        row = {
            "model": m,
            "description": desc,
            "n_params": fit.get("p", 0),
            "rss": fit.get("rss", float("nan")),
            "cv_rmse_kms": cv,
        }
        if m != "intercept":
            ft = nested_f_test(intercept["rss"], fit["rss"], fit["n"], intercept["p"], fit["p"])
            row["f_vs_intercept"] = ft["f_stat"]
            row["p_vs_intercept"] = ft["p_value"]
        rows.append(row)
    return pd.DataFrame(rows)


def choose_best_model(model_cmp: pd.DataFrame) -> str:
    ok = model_cmp[np.isfinite(model_cmp["cv_rmse_kms"].astype(float))].copy()
    if ok.empty:
        return "curve_stellar"
    best = ok.loc[ok["cv_rmse_kms"].idxmin(), "model"]
    base_cv = float(model_cmp.loc[model_cmp["model"] == "intercept", "cv_rmse_kms"].iloc[0])
    best_cv = float(ok.loc[ok["model"] == best, "cv_rmse_kms"].iloc[0])
    sig = model_cmp[model_cmp["p_vs_intercept"].astype(float) < SIGNIFICANCE_P]["model"].tolist()
    if best == "intercept" or (base_cv - best_cv) < CV_IMPROVEMENT_KMS:
        if "curve" in sig:
            return "curve"
        return "intercept"
    return str(best)


def export_regression_bias_table(df: pd.DataFrame, fit: dict) -> pd.DataFrame:
    X = _build_features(df, mode=fit["mode"], fit=False, state=fit["state"])
    y_hat = predict_linear(X, fit["coef"])
    out = df.copy()
    out["regression_bias_kms"] = y_hat
    out["bias_residual_kms"] = out["weighted_mean_residual_kms"].astype(float) - y_hat
    # hybrid: regression + shrunk object offset
    obj_off = out.groupby("gaia_dr3_id")["bias_residual_kms"].transform("mean")
    out["adjusted_bias_kms"] = y_hat + obj_off
    out["bias_model"] = fit["mode"]
    return out


def _plot_bias_curve(curve: pd.DataFrame, fit: dict, df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(curve["chunk_order"], curve["sample_mean_bias_kms"], s=40, c="C0", label="sample mean bias", zorder=3)
    x_grid = np.linspace(0, 1, 200)
    if fit["mode"] in ("curve", "curve_stellar", "curve_stellar_interaction", "full"):
        knots = fit["state"]["knots"]
        B = truncated_cubic_basis(x_grid, knots)
        n_curve = B.shape[1]
        coef = np.asarray(fit["coef"], float)[:n_curve]
        y_grid = B @ coef
        o_min = float(df["chunk_order"].min())
        o_max = float(df["chunk_order"].max())
        xs = o_min + x_grid * max(o_max - o_min, 1)
        ax.plot(xs, y_grid, "C1-", lw=1.5, label=f"curve component ({fit['mode']})")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Echelle order")
    ax.set_ylabel("Weighted mean chunk bias (km/s)")
    ax.set_title("Sample bias curve vs echelle order")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_stellar_dependence(df: pd.DataFrame, slopes: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, label in zip(axes, ["teff", "logg", "mh"], ["Teff (K)", "log g", "[M/H]"]):
        sub = df[np.isfinite(df[col].astype(float)) & np.isfinite(df["weighted_mean_residual_kms"].astype(float))]
        ax.scatter(sub[col], sub["weighted_mean_residual_kms"], s=12, alpha=0.35, c="C0")
        ax.set_xlabel(label)
        ax.set_ylabel("Chunk bias (km/s)")
    n_sig = int(slopes["significant"].sum()) if len(slopes) and "significant" in slopes.columns else 0
    fig.suptitle(f"Pooled chunk biases vs stellar params ({n_sig} chunks with significant Teff slope)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _median_stellar_row(df: pd.DataFrame) -> pd.Series:
    cols = ["teff", "logg", "mh", "log10_median_mask_ccf_peak_snr", "chunk_order_norm"]
    row = {}
    for c in cols:
        if c in df.columns:
            row[c] = float(np.nanmedian(df[c].astype(float)))
        else:
            row[c] = float("nan")
    return pd.Series(row)


def _model_bias_vs_teff(
    df: pd.DataFrame,
    fit: dict,
    *,
    chunk_order_norm: float,
    teff_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted bias model along Teff at fixed order and median other covariates."""
    med = _median_stellar_row(df)
    n = len(teff_grid)
    synth = pd.DataFrame(
        {
            "teff": teff_grid,
            "logg": np.full(n, med["logg"]),
            "mh": np.full(n, med["mh"]),
            "log10_median_mask_ccf_peak_snr": np.full(n, med["log10_median_mask_ccf_peak_snr"]),
            "chunk_order_norm": np.full(n, chunk_order_norm),
        }
    )
    X = _build_features(synth, mode=fit["mode"], fit=False, state=fit["state"])
    return predict_linear(X, fit["coef"])


def _plot_rv_bias_vs_teff_with_fits(
    df: pd.DataFrame,
    slopes: pd.DataFrame,
    reg_table: pd.DataFrame,
    chosen_fit: dict,
    interaction_fit: dict,
    out_path: Path,
) -> None:
    """Chunk RV bias vs Teff with per-chunk linear fits and applied correction model."""
    sub = df[np.isfinite(df["teff"].astype(float)) & np.isfinite(df["weighted_mean_residual_kms"].astype(float))].copy()
    if sub.empty:
        return

    teff = sub["teff"].astype(float).values
    t_lo, t_hi = float(np.percentile(teff, 2)), float(np.percentile(teff, 98))
    teff_grid = np.linspace(t_lo, t_hi, 120)
    order_norm = sub["chunk_order_norm"].astype(float).values
    o_min, o_max = float(np.nanmin(order_norm)), float(np.nanmax(order_norm))

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]})

    # --- Panel 1: measured bias + per-chunk Teff fits ---
    ax0 = axes[0]
    sc = ax0.scatter(
        teff,
        sub["weighted_mean_residual_kms"].astype(float),
        c=order_norm,
        cmap="viridis",
        s=22,
        alpha=0.55,
        edgecolors="none",
        label="per-object chunk bias",
    )
    cbar = fig.colorbar(sc, ax=ax0, pad=0.01, fraction=0.03)
    cbar.set_label("normalized echelle order")

    if len(slopes):
        slope_lookup = slopes.set_index("chunk_key")
        for _, row in slopes.iterrows():
            if "teff_intercept_kms" not in row:
                continue
            y_line = float(row["teff_intercept_kms"]) + float(row["teff_slope_kms_per_k"]) * teff_grid
            if bool(row.get("significant", False)):
                color = plt.cm.viridis(
                    (float(row["chunk_order"]) - o_min) / max(o_max - o_min, 1e-6)
                )
                ax0.plot(teff_grid, y_line, color=color, lw=1.4, alpha=0.75)
            else:
                ax0.plot(teff_grid, y_line, color="0.75", lw=0.5, alpha=0.35)

    n_sig = int(slopes["significant"].sum()) if len(slopes) and "significant" in slopes.columns else 0
    ax0.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax0.set_ylabel("Chunk RV bias (km/s)\n(RV_chunk − RV_exposure)")
    ax0.set_title(
        f"Chunk bias vs Teff — per-chunk linear fits ({n_sig} significant at p<{SIGNIFICANCE_P})"
    )
    ax0.plot([], [], color="0.75", lw=1, label="non-significant chunk fit")
    ax0.plot([], [], color="C1", lw=1.5, label="significant chunk fit")
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # --- Panel 2: order-detrended residual + full correction model ---
    ax1 = axes[1]
    reg = reg_table.loc[sub.index].copy()
    raw = sub["weighted_mean_residual_kms"].astype(float).values
    order_component = reg["regression_bias_kms"].astype(float).values
    residual = reg["bias_residual_kms"].astype(float).values
    adjusted = reg["adjusted_bias_kms"].astype(float).values

    ax1.scatter(
        teff,
        residual,
        c=order_norm,
        cmap="viridis",
        s=22,
        alpha=0.45,
        edgecolors="none",
        label="bias after order curve removed",
    )
    ax1.scatter(
        teff,
        adjusted,
        facecolors="none",
        edgecolors="C3",
        s=36,
        linewidths=0.9,
        alpha=0.85,
        label="applied correction (adj. bias)",
    )

    # Interaction model: bias vs Teff at low/mid/high order
    if interaction_fit.get("coef") is not None:
        order_ticks = [0.15, 0.5, 0.85]
        line_styles = ["--", "-", "-."]
        for onorm, ls in zip(order_ticks, line_styles, strict=True):
            y_model = _model_bias_vs_teff(sub, interaction_fit, chunk_order_norm=onorm, teff_grid=teff_grid)
            if np.any(np.isfinite(y_model)):
                ax1.plot(
                    teff_grid,
                    y_model,
                    color="C0",
                    ls=ls,
                    lw=1.8,
                    label=f"curve+stellar+Teff×order (order={onorm:.0%})",
                )

    # Pooled stellar-only trend on order-subtracted residuals
    stellar_fit = _fit_model(sub.assign(weighted_mean_residual_kms=residual), "stellar")
    if stellar_fit.get("coef") is not None:
        y_stellar = _model_bias_vs_teff(
            sub.assign(chunk_order_norm=np.nanmedian(order_norm)),
            stellar_fit,
            chunk_order_norm=float(np.nanmedian(order_norm)),
            teff_grid=teff_grid,
        )
        ax1.plot(teff_grid, y_stellar, color="C2", lw=2.0, label="stellar-only fit (on detrended bias)")

    ax1.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel("Teff (K)")
    ax1.set_ylabel("Bias / correction (km/s)")
    ax1.set_title(
        f"Order-detrended bias vs Teff — chosen model `{chosen_fit['mode']}` + interaction overlay"
    )
    ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    model_cmp: pd.DataFrame,
    chosen: str,
    slopes: pd.DataFrame,
    curve: pd.DataFrame,
    n_rows: int,
    n_stars: int,
) -> None:
    best_row = model_cmp[model_cmp["model"] == chosen].iloc[0]
    lines = [
        "# Chunk bias regression report",
        "",
        "## Data",
        f"- Rows: {n_rows} (per-object × chunk, sample_kept=True)",
        f"- Stars: {n_stars}",
        "",
        "## Model comparison (leave-one-star-out CV)",
        "",
        "| model | CV RMSE (km/s) | p vs intercept |",
        "| --- | --- | --- |",
    ]
    for _, r in model_cmp.iterrows():
        pval = r.get("p_vs_intercept", float("nan"))
        pstr = f"{pval:.4g}" if np.isfinite(float(pval)) else "—"
        cv = r.get("cv_rmse_kms", float("nan"))
        cvstr = f"{cv:.4f}" if np.isfinite(float(cv)) else "—"
        lines.append(f"| {r['model']} | {cvstr} | {pstr} |")
    lines.extend(
        [
            "",
            f"**Selected model:** `{chosen}` (CV RMSE = {best_row['cv_rmse_kms']:.4f} km/s)",
            "",
            "## Bias curve (sample mean vs echelle order)",
            f"- Chunks: {len(curve)}",
            f"- Bias range: {curve['sample_mean_bias_kms'].min():.3f} … {curve['sample_mean_bias_kms'].max():.3f} km/s",
            "",
            "## Per-chunk Teff slope test",
        ]
    )
    if len(slopes):
        n_sig = int(slopes["significant"].sum())
        lines.append(f"- {n_sig} / {len(slopes)} chunks with significant Teff dependence (p < {SIGNIFICANCE_P})")
        if n_sig:
            top = slopes[slopes["significant"]].sort_values("p_value").head(8)
            lines.append("")
            lines.append("```")
            lines.append(top[["chunk_key", "chunk_order", "teff_slope_kms_per_k", "p_value"]].to_string(index=False))
            lines.append("```")
    else:
        lines.append("- Insufficient objects per chunk for per-chunk slopes")
    lines.extend(
        [
            "",
            "## Chunk optimization advice (next step)",
            "",
            "See `CHUNK_OPTIMIZATION_ADVICE.md` in this directory.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chunk_optimization_advice(path: Path) -> None:
    text = """# Chunk definition optimization — recommended approach

## Goal

Choose wavelength/pixel chunk boundaries (N chunks from N+1 edges), telluric handling,
and stack weights that minimize post-debias chunk scatter and relative RV disagreement.

## Recommended staged workflow

### Stage 1 — Parametric grid (do first; no ML required)

1. **Equal sub-chunks:** `--subchunks N` for N ∈ {1, 2, 4, 8} (already supported in `darkhunter_rv/chunking.py`).
2. **Order merging:** evaluate coarser layouts by merging adjacent echelle orders offline (`validation/evaluate_chunk_layout.py`) — fast on existing diagnostics.
3. **Custom pixel edges:** YAML layout with N+1 fractional edges per order (`validation/chunk_layout.py`); rerun pipeline, then repeat bias/RMS loop.

Score each layout with the same metrics:

- Sample bias curve smoothness (regression spline RSS)
- Per-exposure chunk scatter (median / p90)
- APF–APF relative gate (≤7 d), median |ΔRV|

### Stage 2 — Telluric-aware weighting (before ML)

Improve weights using quantities already computed in pipeline QC:

- Down-weight or zero chunks with high `telluric_fraction` (persist in diagnostics)
- Down-weight chunks failing mask-line density cuts
- Scale IVW weights by CCF quality (peak S/N, asymmetry)

This is interpretable and should be tried before black-box chunk placement.

### Stage 3 — ML for chunk edges (optional; issue #53)

**Gradient boosted trees (XGBoost/LightGBM)** are reasonable for *predicting chunk quality* (|residual|, scatter contribution) from per-chunk features (order, λ, telluric fraction, S/N, Teff). Use ML to **rank** candidate edge placements or assign trust weights — not as the primary RV estimator.

**Why not BDT for edges directly?**

- Edge placement is a small discrete/continuous search problem; BDTs do not optimize boundaries natively.
- Better: BDT predicts `|residual|` or `chunk_scatter` from features → aggregate score for a candidate layout on a validation set.

**Practical hybrid:**

1. Generate candidate layouts (grid over N, edge fractions, telluric splits).
2. For each layout, compute bias table + relative gate (this repo's validation loop).
3. Optionally train a BDT on chunk-level rows to predict residual magnitude; use mean predicted |residual| as a cheap proxy before full pipeline reruns.

### Stage 4 — Closed loop

```bash
cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
# 1) pick layout
python -m validation.evaluate_chunk_layout --layouts calibration/chunk_layouts/*.yaml ...
# 2) rerun pipeline with winning layout (when subchunks/edges change)
python -m darkhunter_rv.pipeline ... --chunk-layout calibration/chunk_layouts/winner.yaml
# 3) bias + regression + relative gate
python -m validation.plot_chunk_residuals ...
python -m validation.chunk_bias_regression ...
python -m validation.reassess_relative_calibration --bias-csv .../regression_adjusted_chunk_bias.csv
```

## Starting point for N+1 edges

See `calibration/chunk_layouts/` examples:

- `whole_order.yaml` — current production (1 chunk / order)
- `subchunks_4.yaml` — 4 equal pixel splits per order
- `merge_pairs.yaml` — offline merge of adjacent orders (evaluation only)

Adjust N by editing `subchunks` or `pixel_edges: [0.0, 0.25, 0.5, 0.75, 1.0]`.
"""
    path.write_text(text, encoding="utf-8")


def run_regression(
    *,
    bias_csv: Path,
    summary_dir: Path,
    out_dir: Path,
    long_csv_glob: str | None,
    reassess: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_bias_regression_table(
        bias_csv,
        summary_dir=summary_dir,
        long_csv_glob=long_csv_glob,
        sample_kept_only=True,
    )
    if df.empty:
        return {"n_rows": 0}

    name_lookup = _load_name_lookup(REPO_ROOT)
    if "name" not in df.columns:
        df["name"] = df["gaia_dr3_id"].map(lambda g: name_lookup.get(str(g), str(g)))

    curve = sample_mean_bias_curve(df)
    curve.to_csv(out_dir / "sample_bias_curve.csv", index=False)

    slopes = per_chunk_stellar_slopes(df)
    slopes.to_csv(out_dir / "per_chunk_teff_slopes.csv", index=False)

    model_cmp = compare_models(df)
    model_cmp.to_csv(out_dir / "model_comparison.csv", index=False)
    chosen = choose_best_model(model_cmp)
    best_fit = _fit_model(df, chosen)
    interaction_fit = _fit_model(df, "curve_stellar_interaction")

    reg_table = export_regression_bias_table(df, best_fit)
    reg_table.to_csv(out_dir / "regression_adjusted_chunk_bias.csv", index=False)

    # export format compatible with chunk_calibration (uses adjusted_bias_kms)
    cal_export = reg_table.rename(
        columns={
            "adjusted_bias_kms": "weighted_mean_residual_kms",
        }
    )
    cal_export["sample_kept"] = True
    cal_export.to_csv(out_dir / "regression_chunk_bias_for_calibration.csv", index=False)

    _plot_bias_curve(curve, best_fit, df, out_dir / "plots" / "bias_curve_fit.png")
    _plot_stellar_dependence(df, slopes, out_dir / "plots" / "bias_vs_stellar_params.png")
    _plot_rv_bias_vs_teff_with_fits(
        df,
        slopes,
        reg_table,
        best_fit,
        interaction_fit,
        out_dir / "plots" / "rv_bias_vs_teff_corrections.png",
    )

    write_chunk_optimization_advice(out_dir / "CHUNK_OPTIMIZATION_ADVICE.md")
    _write_report(
        out_dir / "REPORT.md",
        model_cmp=model_cmp,
        chosen=chosen,
        slopes=slopes,
        curve=curve,
        n_rows=len(df),
        n_stars=int(df["gaia_dr3_id"].nunique()),
    )

    manifest = {
        "n_rows": int(len(df)),
        "n_stars": int(df["gaia_dr3_id"].nunique()),
        "chosen_model": chosen,
        "model_comparison": model_cmp.to_dict(orient="records"),
        "n_significant_teff_slopes": int(slopes["significant"].sum()) if len(slopes) else 0,
    }
    (out_dir / "regression_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if reassess:
        from validation.reassess_relative_calibration import run_assessment

        assess_dir = out_dir / "relative_gate_after_regression"
        gate = run_assessment(
            diagnostics_glob="output/Gaia_DR3_*_diagnostics.csv",
            bias_csv=out_dir / "regression_chunk_bias_for_calibration.csv",
            out_dir=assess_dir,
            summary_dir=summary_dir,
            pair_window_days=7.0,
            goal_kms=0.1,
            clip_sigma=7.0,
            clip_max_delta_kms=20.0,
            min_chunks=3,
        )
        manifest["relative_gate_after_regression"] = gate

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bias-csv",
        type=Path,
        default=REPO_ROOT / "validation_output" / "chunk_residuals" / "per_object_chunk_bias.csv",
    )
    ap.add_argument("--summary-dir", type=Path, default=REPO_ROOT / "output")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "validation_output" / "chunk_bias_regression")
    ap.add_argument(
        "--long-csv-glob",
        default=str(REPO_ROOT / "validation_output" / "chunk_residuals" / "*" / "chunk_residuals_long.csv"),
    )
    ap.add_argument("--reassess-relative-gate", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.bias_csv.is_file():
        logger.error("Missing %s", args.bias_csv)
        sys.exit(1)

    manifest = run_regression(
        bias_csv=args.bias_csv,
        summary_dir=args.summary_dir,
        out_dir=args.out_dir,
        long_csv_glob=args.long_csv_glob,
        reassess=args.reassess_relative_gate,
    )
    logger.info("Regression report -> %s (model=%s)", args.out_dir, manifest.get("chosen_model"))


if __name__ == "__main__":
    main()
