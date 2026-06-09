#!/usr/bin/env python3
"""
Per-object mask-CCF chunk residual diagnostics (cool / mask-applicable stars only).

For each object with stellar-mask-applicable exposures:

1. ``*_residuals_by_spectrum.png`` — RV_chunk − RV_exposure for every chunk and spectrum.
2. ``*_chunk_weighted_mean.png`` — per-chunk weighted mean residual across spectra with
   statistical and intrinsic-scatter error bars.
3. ``sample_per_object_chunk_bias.png`` — per-chunk points = each object's weighted mean bias;
   overlay = sample-wide weighted mean across objects (stat + intrinsic error bars).

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.plot_chunk_residuals \\
    --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \\
    --out-dir validation_output/chunk_residuals
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from glob import glob as glob_paths
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv.method_evaluation import exposure_method_flags, median_mask_ccf_peak_snr  # noqa: E402
from darkhunter_rv.method_regions import region_mask_applicable  # noqa: E402
from darkhunter_rv.summary_paths import parse_object_id_from_summary, discover_summary_files  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_OUTLIER_SIGMA = 7.0
DEFAULT_CHUNK_MAX_DELTA_KMS = 20.0
DEFAULT_MIN_CHUNK_MEASUREMENTS = 3


def _parse_gaia_id_from_path(path: str | Path) -> str | None:
    m = re.search(r"Gaia_DR3_(\d{18,19})", str(path))
    return m.group(1) if m else None


def _chunk_sort_key(chunk_key: str) -> tuple[int, int]:
    parts = str(chunk_key).split("_")
    try:
        order = int(parts[0])
    except ValueError:
        order = 9999
    sub = int(parts[1]) if len(parts) > 1 else 0
    return (order, sub)


def _ordered_chunks(chunk_keys: list[str]) -> list[str]:
    return sorted(set(chunk_keys), key=_chunk_sort_key)


def _exposure_rv_weighted(rv: np.ndarray, err: np.ndarray) -> float:
    ok = np.isfinite(rv) & np.isfinite(err) & (err > 0) & (err < 1e20)
    if not np.any(ok):
        v = rv[np.isfinite(rv)]
        return float(np.mean(v)) if len(v) else float("nan")
    v = rv[ok].astype(float)
    e = err[ok].astype(float)
    w = 1.0 / (e**2)
    return float(np.sum(w * v) / np.sum(w))


def iterative_spectrum_chunk_clip_mask(
    rv: np.ndarray,
    err: np.ndarray,
    *,
    nsigma: float = 10.0,
    max_delta_kms: float = 30.0,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Per-spectrum iterative clip until convergence:

    1. |RV_i − weighted_mean(kept)| > max_delta_kms
    2. |RV_i − mean(RV_{j≠i})| > nsigma × RMS(RV_{j≠i})  (leave-one-out)
    """
    rv = np.asarray(rv, float)
    err = np.asarray(err, float)
    n = len(rv)
    keep = np.ones(n, dtype=bool)
    if n < 1:
        return keep

    use_delta = max_delta_kms is not None and np.isfinite(max_delta_kms) and max_delta_kms > 0
    use_sigma = nsigma is not None and np.isfinite(nsigma) and nsigma > 0

    for _ in range(max_iter):
        removed_any = False
        active = np.where(keep)[0]
        if len(active) == 0:
            break

        if use_delta and len(active) >= 1:
            wmean = _exposure_rv_weighted(rv[keep], err[keep])
            if np.isfinite(wmean):
                for i in active:
                    if abs(float(rv[i]) - wmean) > float(max_delta_kms):
                        keep[i] = False
                        removed_any = True

        if use_sigma and keep.sum() >= 2:
            active = np.where(keep)[0]
            for i in active:
                others = rv[keep & (np.arange(n) != i)]
                if len(others) < 1:
                    continue
                mu = float(np.mean(others))
                rms = float(np.std(others, ddof=1)) if len(others) > 1 else 0.0
                if not np.isfinite(rms) or rms <= 0:
                    continue
                if abs(float(rv[i]) - mu) > nsigma * rms:
                    keep[i] = False
                    removed_any = True

        if not removed_any:
            break
    return keep


def apply_spectrum_chunk_outlier_clip(
    tab: pd.DataFrame,
    *,
    nsigma: float = 10.0,
    max_delta_kms: float = 30.0,
) -> pd.DataFrame:
    """Mark chunk_kept per spectrum; recompute exposure_rv_kms and residual_kms from kept chunks."""
    if tab.empty:
        return tab
    out = tab.copy()
    out["chunk_kept"] = True
    out["exposure_rv_kms_pipeline"] = out["exposure_rv_kms"]
    out["residual_kms_pipeline"] = out["residual_kms"]

    for file_label, g in out.groupby("file", sort=False):
        idx = g.index.to_numpy()
        rv = g["rv_kms"].astype(float).to_numpy()
        err = g["rv_err_kms"].astype(float).to_numpy()
        keep = iterative_spectrum_chunk_clip_mask(
            rv, err, nsigma=nsigma, max_delta_kms=max_delta_kms
        )
        out.loc[idx, "chunk_kept"] = keep
        if not np.any(keep):
            continue
        exp_rv = _exposure_rv_weighted(rv[keep], err[keep])
        out.loc[idx[keep], "exposure_rv_kms"] = exp_rv
        out.loc[idx[keep], "residual_kms"] = rv[keep] - exp_rv

    return out


def _weighted_mean_and_errors(
    values: np.ndarray,
    errs: np.ndarray,
) -> tuple[float, float, float]:
    """Return (weighted_mean, statistical_err, intrinsic_scatter)."""
    ok = np.isfinite(values) & np.isfinite(errs) & (errs > 0) & (errs < 1e20)
    if not np.any(ok):
        v = values[np.isfinite(values)]
        if len(v) == 0:
            return float("nan"), float("nan"), float("nan")
        return float(np.mean(v)), float("nan"), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    v = values[ok].astype(float)
    e = errs[ok].astype(float)
    w = 1.0 / (e**2)
    mu = float(np.sum(w * v) / np.sum(w))
    stat = float(1.0 / np.sqrt(np.sum(w)))
    if len(v) > 1:
        resid = v - mu
        var_int = max(0.0, float(np.var(resid, ddof=1)) - stat**2)
        intrinsic = float(np.sqrt(var_int))
    else:
        intrinsic = 0.0
    return mu, stat, intrinsic


def _load_chunk_rows(diagnostics_glob: str) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(glob_paths(diagnostics_glob)):
        gaia_id = _parse_gaia_id_from_path(path)
        if not gaia_id:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        chunk_rows = [
            r for _, r in df.iterrows() if str(r.get("chunk_key", "")) != "all"
        ]
        if not chunk_rows:
            continue
        flags = exposure_method_flags(chunk_rows)
        teff = float(chunk_rows[0].get("teff", np.nan))
        snr_med = float(flags.get("median_mask_ccf_peak_snr", np.nan))
        log_snr = float(np.log10(snr_med)) if np.isfinite(snr_med) and snr_med > 0 else np.nan
        if not bool(region_mask_applicable(np.array([teff]), np.array([log_snr]))[0]):
            continue
        exposure_rv = float(chunk_rows[0].get("exposure_rv_kms", np.nan))
        if not np.isfinite(exposure_rv):
            exposure_rv = float(flags.get("mask_rv_kms", np.nan))
        stem = Path(path).stem.replace("_diagnostics", "")
        file_label = str(chunk_rows[0].get("file", stem))
        mjd = float(chunk_rows[0].get("mjd", np.nan))
        for r in chunk_rows:
            if str(r.get("method", "")) != "mask_ccf":
                continue
            ck = str(r.get("chunk_key", ""))
            rv = float(r.get("rv_kms", np.nan))
            if not np.isfinite(rv) or not np.isfinite(exposure_rv):
                continue
            resid = float(r.get("residual_to_exposure_kms", np.nan))
            if not np.isfinite(resid):
                resid = rv - exposure_rv
            rows.append(
                {
                    "gaia_dr3_id": gaia_id,
                    "file": file_label,
                    "mjd": mjd,
                    "teff": teff,
                    "log10_median_mask_ccf_peak_snr": log_snr,
                    "chunk_key": ck,
                    "chunk_order": _chunk_sort_key(ck)[0],
                    "rv_kms": rv,
                    "rv_err_kms": float(r.get("rv_err_kms", np.nan)),
                    "exposure_rv_kms": exposure_rv,
                    "residual_kms": resid,
                    "used_in_exposure_stack": bool(r.get("used_in_exposure_stack", False)),
                    "qc_pass": bool(r.get("qc_pass", True)),
                    "diagnostics_path": str(path),
                }
            )
    return pd.DataFrame(rows)


def _load_name_lookup(repo_root: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    lit = repo_root / "calibration" / "literature_rv_master.csv"
    if lit.is_file():
        df = pd.read_csv(lit)
        for _, r in df.iterrows():
            gid = str(r.get("gaia_dr3_id", ""))
            nm = str(r.get("name", "")).strip()
            if gid and nm and gid not in names:
                names[gid] = nm
    overlap = repo_root / "validation_output" / "rv_phase_a_baseline" / "overlap_stars.csv"
    if overlap.is_file():
        df = pd.read_csv(overlap)
        for _, r in df.iterrows():
            gid = str(r.get("gaia_dr3_id", ""))
            nm = str(r.get("name", "")).strip()
            if gid and nm:
                names[gid] = nm
    return names


def _object_name(gaia_id: str, summary_dir: Path | None, name_lookup: dict[str, str]) -> str:
    if gaia_id in name_lookup:
        return name_lookup[gaia_id]
    if summary_dir is not None:
        for sp in discover_summary_files(summary_dir):
            if parse_object_id_from_summary(sp) == gaia_id:
                stem = sp.stem.replace("_summary", "")
                if stem.startswith("Gaia_DR3_"):
                    tag = stem.replace(f"Gaia_DR3_{gaia_id}", "").strip("_")
                    if tag:
                        return tag
                return stem
    return gaia_id


def _clip_title_note(clip_sigma: float | None, clip_max_delta_kms: float | None) -> str:
    parts = []
    if clip_sigma is not None and clip_sigma > 0:
        parts.append(f"{clip_sigma:g}σ LOO")
    if clip_max_delta_kms is not None and clip_max_delta_kms > 0:
        parts.append(f"±{clip_max_delta_kms:g} km/s")
    return f", {' + '.join(parts)} clip" if parts else ""


def _clip_legend_label(clip_sigma: float | None, clip_max_delta_kms: float | None) -> str:
    return "excluded (" + ", ".join(
        p
        for p in (
            f">{clip_sigma:g}σ LOO" if clip_sigma and clip_sigma > 0 else "",
            f">±{clip_max_delta_kms:g} km/s" if clip_max_delta_kms and clip_max_delta_kms > 0 else "",
        )
        if p
    ) + ")"


def _plot_residuals_by_spectrum(
    obj_df: pd.DataFrame,
    *,
    gaia_id: str,
    name: str,
    out_path: Path,
    clip_sigma: float | None,
    clip_max_delta_kms: float | None,
) -> None:
    kept_df = obj_df[obj_df["chunk_kept"].astype(bool)] if "chunk_kept" in obj_df.columns else obj_df
    excluded_df = (
        obj_df[~obj_df["chunk_kept"].astype(bool)]
        if "chunk_kept" in obj_df.columns
        else obj_df.iloc[0:0]
    )
    chunks = _ordered_chunks(kept_df["chunk_key"].astype(str).tolist())
    if excluded_df is not None and len(excluded_df):
        chunks = _ordered_chunks(
            chunks + excluded_df["chunk_key"].astype(str).tolist()
        )
    chunk_to_x = {ck: i for i, ck in enumerate(chunks)}
    spectra = kept_df.groupby("file", sort=False)
    n_spec = spectra.ngroups
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_spec, 1)))

    fig, ax = plt.subplots(figsize=(max(8, len(chunks) * 0.22), 5))
    for i, (file_label, g) in enumerate(spectra):
        xs = [chunk_to_x[str(ck)] for ck in g["chunk_key"]]
        ys = g["residual_kms"].astype(float).values
        ax.scatter(
            xs,
            ys,
            s=22,
            alpha=0.8,
            c=[cmap[i]],
            edgecolors="none",
            label=Path(str(file_label)).stem[-24:],
        )
    if len(excluded_df):
        ex_x = [chunk_to_x[str(ck)] for ck in excluded_df["chunk_key"]]
        ax.scatter(
            ex_x,
            excluded_df["residual_kms_pipeline"].astype(float).values,
            marker="x",
            s=28,
            c="0.55",
            alpha=0.45,
            linewidths=0.8,
            label=_clip_legend_label(clip_sigma, clip_max_delta_kms),
        )
    ax.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax.set_xticks(range(len(chunks)))
    ax.set_xticklabels(chunks, rotation=90, fontsize=6)
    ax.set_xlabel("chunk_key")
    ax.set_ylabel("RV_chunk − RV_exposure (km/s)")
    ax.set_title(f"{name} — chunk residuals by spectrum (mask-applicable{_clip_title_note(clip_sigma, clip_max_delta_kms)})")
    if n_spec <= 12:
        ax.legend(fontsize=6, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _summarize_chunks_per_object(
    obj_df: pd.DataFrame,
    *,
    min_measurements: int = DEFAULT_MIN_CHUNK_MEASUREMENTS,
) -> pd.DataFrame:
    rows = []
    for ck, g in obj_df.groupby("chunk_key"):
        n_meas = int(len(g))
        if n_meas < min_measurements:
            continue
        mu, stat, intrinsic = _weighted_mean_and_errors(
            g["residual_kms"].astype(float).values,
            g["rv_err_kms"].astype(float).values,
        )
        rows.append(
            {
                "chunk_key": str(ck),
                "chunk_order": _chunk_sort_key(str(ck))[0],
                "n_measurements": n_meas,
                "n_spectra": int(g["file"].nunique()),
                "weighted_mean_residual_kms": mu,
                "statistical_err_kms": stat,
                "intrinsic_scatter_kms": intrinsic,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["chunk_order", "chunk_key"]).reset_index(drop=True)


def _plot_chunk_weighted_mean(
    summary: pd.DataFrame,
    *,
    gaia_id: str,
    name: str,
    out_path: Path,
    title_suffix: str,
) -> None:
    if summary.empty:
        return
    chunks = summary["chunk_key"].astype(str).tolist()
    x = np.arange(len(chunks))
    mu = summary["weighted_mean_residual_kms"].astype(float).values
    stat = summary["statistical_err_kms"].astype(float).values
    intrinsic = summary["intrinsic_scatter_kms"].astype(float).values
    stat = np.where(np.isfinite(stat), stat, 0.0)
    intrinsic = np.where(np.isfinite(intrinsic), intrinsic, 0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(chunks) * 0.22), 5))
    ax.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax.errorbar(
        x,
        mu,
        yerr=stat,
        fmt="o",
        color="#2166ac",
        ecolor="#2166ac",
        capsize=2,
        label="statistical σ",
        zorder=3,
    )
    ax.errorbar(
        x,
        mu,
        yerr=intrinsic,
        fmt="none",
        ecolor="#b2182b",
        capsize=4,
        elinewidth=1.5,
        label="intrinsic scatter σ",
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(chunks, rotation=90, fontsize=6)
    ax.set_xlabel("chunk_key")
    ax.set_ylabel("weighted mean residual (km/s)")
    ax.set_title(f"{name} — {title_suffix}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _combined_object_bias_err(stat: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    s = np.where(np.isfinite(stat), stat, 0.0)
    i = np.where(np.isfinite(intrinsic), intrinsic, 0.0)
    e = np.sqrt(s**2 + i**2)
    e = np.where(e > 0, e, np.nan)
    return e


def apply_sample_object_bias_clip(
    bias_df: pd.DataFrame,
    *,
    nsigma: float = 5.0,
    max_delta_kms: float = 10.0,
) -> pd.DataFrame:
    """Per chunk_key, clip per-object bias values before full-sample aggregation."""
    if bias_df.empty:
        return bias_df
    out = bias_df.copy()
    out["sample_kept"] = True
    stat_col = out["statistical_err_kms"].astype(float).values
    int_col = out["intrinsic_scatter_kms"].astype(float).values
    err = _combined_object_bias_err(stat_col, int_col)

    for _, g in out.groupby("chunk_key"):
        idx = g.index.to_numpy()
        rv = g["weighted_mean_residual_kms"].astype(float).to_numpy()
        e = err[g.index]
        keep = iterative_spectrum_chunk_clip_mask(
            rv, e, nsigma=nsigma, max_delta_kms=max_delta_kms
        )
        out.loc[idx, "sample_kept"] = keep
    return out


def _plot_sample_per_object_bias(
    bias_df: pd.DataFrame,
    names: dict[str, str],
    out_path: Path,
    *,
    sample_sigma: float | None,
    sample_max_delta_kms: float | None,
) -> None:
    if bias_df.empty:
        return
    chunk_set = set(bias_df["chunk_key"].astype(str).tolist())
    chunks = _ordered_chunks(list(chunk_set))
    chunk_to_x = {ck: i for i, ck in enumerate(chunks)}
    gids = sorted(bias_df["gaia_dr3_id"].astype(str).unique())
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(gids), 1)))
    gid_to_j = {gid: j for j, gid in enumerate(gids)}

    fig, ax = plt.subplots(figsize=(max(10, len(chunks) * 0.25), 6))
    kept = bias_df[bias_df["sample_kept"].astype(bool)] if "sample_kept" in bias_df.columns else bias_df
    excluded = bias_df[~bias_df["sample_kept"].astype(bool)] if "sample_kept" in bias_df.columns else bias_df.iloc[0:0]

    for _, r in kept.iterrows():
        gid = str(r["gaia_dr3_id"])
        ck = str(r["chunk_key"])
        if ck not in chunk_to_x:
            continue
        j = gid_to_j.get(gid, 0)
        x = chunk_to_x[ck] + (j - len(gids) / 2) * 0.02
        label = names.get(gid, gid[:12])
        ax.scatter(
            x,
            float(r["weighted_mean_residual_kms"]),
            s=28,
            alpha=0.8,
            c=[cmap[j % len(cmap)]],
            label=label if chunk_to_x[ck] == 0 else None,
        )

    if len(excluded):
        for _, r in excluded.iterrows():
            ck = str(r["chunk_key"])
            if ck not in chunk_to_x:
                continue
            gid = str(r["gaia_dr3_id"])
            j = gid_to_j.get(gid, 0)
            x = chunk_to_x[ck] + (j - len(gids) / 2) * 0.02
            ax.scatter(
                x,
                float(r["weighted_mean_residual_kms"]),
                marker="x",
                s=24,
                c="0.55",
                alpha=0.45,
                linewidths=0.8,
            )
        ax.scatter(
            [],
            [],
            marker="x",
            c="0.55",
            label=_clip_legend_label(sample_sigma, sample_max_delta_kms),
        )

    ox, oy, ostat, oint = [], [], [], []
    for ck in chunks:
        g = kept[kept["chunk_key"].astype(str) == ck]
        if g.empty:
            continue
        mu, stat, intrinsic = _weighted_mean_and_errors(
            g["weighted_mean_residual_kms"].astype(float).values,
            _combined_object_bias_err(
                g["statistical_err_kms"].astype(float).values,
                g["intrinsic_scatter_kms"].astype(float).values,
            ),
        )
        ox.append(chunk_to_x[ck])
        oy.append(mu)
        ostat.append(stat if np.isfinite(stat) else 0.0)
        oint.append(intrinsic if np.isfinite(intrinsic) else 0.0)
    if ox:
        ox = np.asarray(ox, float)
        oy = np.asarray(oy, float)
        ostat = np.asarray(ostat, float)
        oint = np.asarray(oint, float)
        ax.errorbar(ox, oy, yerr=ostat, fmt="D", color="black", ms=6, capsize=2, label="sample mean (stat)", zorder=5)
        ax.errorbar(ox, oy, yerr=oint, fmt="none", ecolor="black", capsize=5, elinewidth=2, label="sample mean (intrinsic)", zorder=4)

    ax.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax.set_xticks(range(len(chunks)))
    ax.set_xticklabels(chunks, rotation=90, fontsize=6)
    ax.set_xlabel("chunk_key")
    ax.set_ylabel("per-object weighted mean chunk bias (km/s)")
    clip_note = _clip_title_note(sample_sigma, sample_max_delta_kms)
    ax.set_title(f"Sample: per-object chunk bias and cross-object mean{clip_note}")
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run_plot_chunk_residuals(
    *,
    diagnostics_glob: str,
    out_dir: Path,
    summary_dir: Path | None = None,
    gaia_ids: set[str] | None = None,
    chunk_outlier_sigma: float | None = DEFAULT_CHUNK_OUTLIER_SIGMA,
    chunk_max_delta_kms: float | None = DEFAULT_CHUNK_MAX_DELTA_KMS,
    sample_outlier_sigma: float | None = 5.0,
    sample_max_delta_kms: float | None = 10.0,
    min_chunk_measurements: int = DEFAULT_MIN_CHUNK_MEASUREMENTS,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tab = _load_chunk_rows(diagnostics_glob)
    if tab.empty:
        logger.warning("No mask-applicable chunk rows matched %s", diagnostics_glob)
        return {"n_objects": 0, "n_rows": 0}

    if gaia_ids:
        tab = tab[tab["gaia_dr3_id"].astype(str).isin(gaia_ids)]

    n_excluded = 0
    clip_enabled = (
        (chunk_outlier_sigma is not None and chunk_outlier_sigma > 0)
        or (chunk_max_delta_kms is not None and chunk_max_delta_kms > 0)
    )
    if clip_enabled:
        tab = apply_spectrum_chunk_outlier_clip(
            tab,
            nsigma=float(chunk_outlier_sigma or 0.0),
            max_delta_kms=float(chunk_max_delta_kms or 0.0),
        )
        n_excluded = int((~tab["chunk_kept"].astype(bool)).sum())
        tab_plot = tab[tab["chunk_kept"].astype(bool)].copy()
    else:
        tab_plot = tab

    name_lookup = _load_name_lookup(_REPO_ROOT)
    names: dict[str, str] = {}
    summaries: dict[str, pd.DataFrame] = {}
    n_objects = 0

    for gid, obj_all in tab.groupby("gaia_dr3_id"):
        gid = str(gid)
        name = _object_name(gid, summary_dir, name_lookup)
        names[gid] = name
        obj_dir = out_dir / gid
        obj_dir.mkdir(parents=True, exist_ok=True)
        obj_all.to_csv(obj_dir / "chunk_residuals_long.csv", index=False)
        obj_df = obj_all[obj_all["chunk_kept"].astype(bool)].copy() if "chunk_kept" in obj_all.columns else obj_all

        _plot_residuals_by_spectrum(
            obj_all,
            gaia_id=gid,
            name=name,
            out_path=obj_dir / f"{name or gid}_residuals_by_spectrum.png",
            clip_sigma=chunk_outlier_sigma,
            clip_max_delta_kms=chunk_max_delta_kms,
        )

        summary = _summarize_chunks_per_object(obj_df, min_measurements=min_chunk_measurements)
        summary.to_csv(obj_dir / "chunk_weighted_summary.csv", index=False)
        summaries[gid] = summary

        _plot_chunk_weighted_mean(
            summary,
            gaia_id=gid,
            name=name,
            out_path=obj_dir / f"{name or gid}_chunk_weighted_mean.png",
            title_suffix=(
                f"per-chunk weighted mean (≥{min_chunk_measurements} surviving measurements)"
                + _clip_title_note(chunk_outlier_sigma, chunk_max_delta_kms)
            ),
        )
        n_objects += 1

    bias_rows = []
    for gid, sdf in summaries.items():
        for _, r in sdf.iterrows():
            bias_rows.append(
                {
                    "gaia_dr3_id": gid,
                    "name": names.get(gid, gid),
                    **r.to_dict(),
                }
            )
    bias_df = pd.DataFrame(bias_rows) if bias_rows else pd.DataFrame()
    n_sample_excluded = 0
    sample_clip_enabled = (
        (sample_outlier_sigma is not None and sample_outlier_sigma > 0)
        or (sample_max_delta_kms is not None and sample_max_delta_kms > 0)
    )
    if not bias_df.empty and sample_clip_enabled:
        bias_df = apply_sample_object_bias_clip(
            bias_df,
            nsigma=float(sample_outlier_sigma or 0.0),
            max_delta_kms=float(sample_max_delta_kms or 0.0),
        )
        n_sample_excluded = int((~bias_df["sample_kept"].astype(bool)).sum())

    if not bias_df.empty:
        bias_df.to_csv(out_dir / "per_object_chunk_bias.csv", index=False)

    _plot_sample_per_object_bias(
        bias_df,
        names,
        out_dir / "sample_per_object_chunk_bias.png",
        sample_sigma=sample_outlier_sigma,
        sample_max_delta_kms=sample_max_delta_kms,
    )

    return {
        "n_objects": n_objects,
        "n_rows": int(len(tab_plot)),
        "n_rows_total": int(len(tab)),
        "n_chunks_excluded": n_excluded,
        "n_sample_bias_excluded": n_sample_excluded,
        "n_chunks_max": int(tab_plot["chunk_key"].nunique()) if len(tab_plot) else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostics-glob", default="output/Gaia_DR3_*_diagnostics.csv")
    ap.add_argument("--summary-dir", type=Path, default=_REPO_ROOT / "output")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "validation_output" / "chunk_residuals")
    ap.add_argument(
        "--overlap-only",
        action="store_true",
        help="Restrict to stars in calibration/literature overlap (phase A overlap_stars.csv if present)",
    )
    ap.add_argument(
        "--chunk-outlier-sigma",
        type=float,
        default=DEFAULT_CHUNK_OUTLIER_SIGMA,
        help="Iterative leave-one-out clip: exclude chunk RVs > N×RMS(other chunks) per spectrum (0=off)",
    )
    ap.add_argument(
        "--chunk-max-delta-kms",
        type=float,
        default=DEFAULT_CHUNK_MAX_DELTA_KMS,
        help="Iterative clip: exclude chunk RVs > N km/s from weighted mean per spectrum (0=off)",
    )
    ap.add_argument(
        "--min-chunk-measurements",
        type=int,
        default=DEFAULT_MIN_CHUNK_MEASUREMENTS,
        help="Per-object weighted mean: require at least N surviving chunk measurements per chunk_key",
    )
    ap.add_argument(
        "--sample-outlier-sigma",
        type=float,
        default=5.0,
        help="Full-sample clip: per chunk, exclude object biases > N×RMS(other objects) (0=off)",
    )
    ap.add_argument(
        "--sample-max-delta-kms",
        type=float,
        default=10.0,
        help="Full-sample clip: per chunk, exclude object biases > N km/s from weighted mean (0=off)",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    gaia_ids: set[str] | None = None
    if args.overlap_only:
        overlap_csv = _REPO_ROOT / "validation_output" / "rv_phase_a_baseline" / "overlap_stars.csv"
        if overlap_csv.is_file():
            gaia_ids = set(pd.read_csv(overlap_csv)["gaia_dr3_id"].astype(str))
        else:
            logger.warning("overlap_stars.csv not found; using all mask-applicable objects")

    clip_sigma = float(args.chunk_outlier_sigma) if args.chunk_outlier_sigma > 0 else None
    clip_delta = float(args.chunk_max_delta_kms) if args.chunk_max_delta_kms > 0 else None
    sample_sigma = float(args.sample_outlier_sigma) if args.sample_outlier_sigma > 0 else None
    sample_delta = float(args.sample_max_delta_kms) if args.sample_max_delta_kms > 0 else None
    stats = run_plot_chunk_residuals(
        diagnostics_glob=args.diagnostics_glob,
        out_dir=args.out_dir,
        summary_dir=args.summary_dir,
        gaia_ids=gaia_ids,
        chunk_outlier_sigma=clip_sigma,
        chunk_max_delta_kms=clip_delta,
        sample_outlier_sigma=sample_sigma,
        sample_max_delta_kms=sample_delta,
        min_chunk_measurements=int(args.min_chunk_measurements),
    )
    logger.info("Wrote chunk residual plots to %s (%s)", args.out_dir, stats)


if __name__ == "__main__":
    main()
