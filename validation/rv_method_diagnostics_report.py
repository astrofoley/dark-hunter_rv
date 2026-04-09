#!/usr/bin/env python3
"""
Aggregate per-exposure ``*_diagnostics.csv`` files and build method-comparison plots.

Expects diagnostics from ``--run-all-methods`` / ``--compare-rv-methods`` (mask_ccf, template_fft,
strong_lines = Voigt+Lorentz Hβ centroid on **all** Teff).

Plots (written to ``--out-dir``):

1. Current mask CCF exposure RV vs legacy pipeline RV (from star summary), filtered by small legacy errors.
2. Residual (mask − template) vs Teff: only exposures in the **applicability overlap**
   (``config.METHOD_REGION_*``): mask = Teff < 5200 K or (Teff < 7000 K and log10 median mask CCF
   S/N > 0.65); template = all Teff. Points still require both σ ≤ ``--max-method-err``. Orange dashed
   line = **median residual** (single bias estimate) in that overlap.
3. Residual (mask − strong lines): overlap = mask region ∩ strong lines (Teff > 5500 K and
   log10 S/N > 0.65), i.e. 5500 < Teff < 7000 K with log10 S/N > 0.65, plus σ filter.
4. Residual (template − strong lines): overlap = strong-lines region (template unrestricted), plus σ filter.
5. Fraction of exposures with exposure-level σ above ``--max-method-err`` vs Teff (one curve per
   method; error bars from Poisson counting). Teff uses **equal-width bins** from
   ``--teff-bin-lo`` to ``--teff-bin-hi`` (defaults in ``config``), not sample quantiles, so the
   same bin edges apply to every run.

Also writes ``method_comparison_per_exposure.csv`` with one row per diagnostics file and
``binned_high_err_fraction_vs_teff.csv`` for the fraction plot.

**Per-exposure RVs** come from ``darkhunter_rv.pipeline._weighted_method_rv_from_rows`` (same rules
as the pipeline: QC, min chunk counts, etc.). **Scatter comparison plots** additionally require each
method’s exposure-level ``rv_err_kms <= --max-method-err`` (default from
``config.COMPARISON_REPORT_MAX_RV_ERR_KMS``, typically 2.5 km/s) for every method shown on that plot.
Residual panels filter by applicability regions and σ cap; ``method_comparison_per_exposure.csv``
includes ``log10_median_mask_ccf_peak_snr`` for downstream use.

**``legacy_vs_current_mask``** further requires finite legacy RV and error and
``legacy_err_kms <= --max-legacy-err`` (default 0.5 km/s), finite current mask RV, and current mask
``rv_err_kms <= --max-method-err``.

Diagnostics must come from pipeline runs with ``--run-all-methods`` / ``--compare-rv-methods``
so each CSV contains mask_ccf, template_fft, and strong_lines rows (campaign scripts set this).

Example (epoch 1 only, paths match a typical clone under ``darkhunter_rvs``)::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.rv_method_diagnostics_report \\
    --diagnostics-glob '/Users/rfoley/darkhunter/rvs/dark-hunter_rv/output/Gaia_DR3_*_epoch_1_diagnostics.csv' \\
    --legacy-summary-dir /Users/rfoley/darkhunter/rvs/dark-hunter_rv/output \\
    --max-legacy-err 0.5 \\
    --out-dir /Users/rfoley/darkhunter/rvs/dark-hunter_rv/validation_output/rv_method_teff_report

Or use ``python -m validation.run_first_epoch_campaign ... --method-teff-report`` to run the
report after the campaign (it picks a diagnostics glob from ``--spectrum-glob`` by default, or
writes ``diagnostics_list.txt`` when ``--max-sources`` is used so the report matches only those runs).

Use ``--diagnostics-list path/to/diagnostics_list.txt`` instead of ``--diagnostics-glob`` when you
need an exact file list (e.g. a 20-star batch in a directory that also holds older CSVs).

**Legacy outliers (|mask − legacy| > threshold):** after this report, run
``python -m validation.plot_legacy_outlier_orders`` with ``--comparison-csv`` pointing at
``method_comparison_per_exposure.csv`` (see that module's docstring).

**Overlap / adopted RV / S/N bins:** run ``python -m validation.rv_method_overlap_report`` with the
same diagnostics glob (see ``docs/rv_methods_evaluation.md``).
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

from darkhunter_rv import config as dh_config  # noqa: PLC2701
from darkhunter_rv.method_evaluation import exposure_method_flags  # noqa: PLC2701
from darkhunter_rv.method_regions import (  # noqa: PLC2701
    region_mask_applicable as _region_mask_applicable,
    region_strong_lines_applicable as _region_strong_lines_applicable,
    region_template_applicable as _region_template_applicable,
)
from darkhunter_rv.pipeline import _weighted_method_rv_from_rows  # noqa: PLC2701

from validation.diagnose_legacy_campaign import parse_summary_file, summary_rows_by_basename  # noqa: E402


def _gaia_from_path(s: str) -> int | None:
    m = re.search(r"Gaia_DR3_(\d{18,19})", str(s))
    return int(m.group(1)) if m else None


def _load_legacy_rv(basename: str, legacy_dir: Path, gaia_id: int) -> tuple[float, float] | None:
    sp = legacy_dir / f"Gaia_DR3_{gaia_id}_summary.txt"
    if not sp.is_file():
        return None
    rows = parse_summary_file(sp)
    byb = summary_rows_by_basename(rows)
    r = byb.get(basename)
    if not r:
        return None
    return float(r["rv"]), float(r["rv_err"])


def _method_err_acceptable(err: np.ndarray, max_err: float) -> np.ndarray:
    """True where stacked method error is positive, finite, and at most ``max_err`` (km/s)."""
    e = np.asarray(err, float)
    return np.isfinite(e) & (e > 0.0) & (e <= float(max_err))


def _teff_bin_edges(teff: np.ndarray, n_bins: int) -> np.ndarray:
    t = teff[np.isfinite(teff)]
    if len(t) < 2:
        return np.array([4000.0, 8000.0])
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(t, q))
    if len(edges) < 2:
        edges = np.array([float(np.min(t)), float(np.max(t)) + 1.0])
    return edges


def _uniform_teff_bin_edges(n_bins: int, lo: float, hi: float) -> np.ndarray:
    """``n_bins`` equal-width bins covering ``[lo, hi]`` K (inclusive outer edges)."""
    nb = int(max(n_bins, 1))
    lo_f, hi_f = float(lo), float(hi)
    if not (np.isfinite(lo_f) and np.isfinite(hi_f)) or hi_f <= lo_f:
        lo_f, hi_f = 4000.0, 8000.0
    return np.linspace(lo_f, hi_f, nb + 1)


def _binned_curve(teff: np.ndarray, y: np.ndarray, edges: np.ndarray):
    centers, meds, mads = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (teff >= lo) & (teff < hi) if i < len(edges) - 2 else (teff >= lo) & (teff <= hi)
        m &= np.isfinite(y) & np.isfinite(teff)
        if int(np.sum(m)) < 1:
            continue
        yy = y[m]
        med = float(np.median(yy))
        mad = float(np.median(np.abs(yy - med)) * 1.4826)
        centers.append(0.5 * (lo + hi))
        meds.append(med)
        mads.append(mad)
    return np.array(centers), np.array(meds), np.array(mads)


def _plot_residual_vs_teff(
    teff: np.ndarray,
    residual: np.ndarray,
    ylabel: str,
    title: str,
    outpath: Path,
    n_bins: int,
    *,
    median_bias_kms: float | None = None,
    subtitle: str | None = None,
) -> None:
    ok = np.isfinite(teff) & np.isfinite(residual)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(teff[ok], residual[ok], s=18, alpha=0.45, color="0.25", label="exposures")
    ax.axhline(0.0, color="k", lw=0.7)
    if median_bias_kms is not None and np.isfinite(float(median_bias_kms)):
        b = float(median_bias_kms)
        ax.axhline(
            b,
            color="C1",
            ls="--",
            lw=1.35,
            label=f"median bias = {b:.2f} km/s",
        )
    edges = _teff_bin_edges(teff[ok], n_bins)
    xc, ym, ye = _binned_curve(teff[ok], residual[ok], edges)
    if len(xc) > 0:
        ax.errorbar(xc, ym, yerr=ye, fmt="s-", color="C0", capsize=3, lw=1.2, ms=5, label="median ± MAD")
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel(ylabel)
    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def _high_err_bin_stats(err: np.ndarray, mbin: np.ndarray, thr: float) -> tuple[int, int, float, float]:
    use = mbin & np.isfinite(err) & (err > 0.0)
    n = int(np.sum(use))
    if n < 1:
        return 0, 0, float("nan"), float("nan")
    bad = use & (err > thr)
    k = int(np.sum(bad))
    f = k / n
    sf = float(np.sqrt(max(k, 1)) / n)
    return n, k, f, sf


def _binned_high_err_fraction_table(
    teff: np.ndarray,
    err_mask: np.ndarray,
    err_tpl: np.ndarray,
    err_sl: np.ndarray,
    max_err: float,
    n_bins: int,
    *,
    teff_bin_lo: float,
    teff_bin_hi: float,
) -> pd.DataFrame:
    """Per Teff bin: N and count with err > max_err for each method; Poisson σ on fraction.

    Bins are **fixed equal width** from ``teff_bin_lo`` to ``teff_bin_hi`` (one row per bin, even
    when N=0) so fraction-vs-Teff stays comparable across different input lists.
    """
    t = np.asarray(teff, float)
    em = np.asarray(err_mask, float)
    et = np.asarray(err_tpl, float)
    es = np.asarray(err_sl, float)
    thr = float(max_err)
    ok_t = np.isfinite(t)
    edges = _uniform_teff_bin_edges(n_bins, teff_bin_lo, teff_bin_hi)
    rows: list[dict] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < len(edges) - 2:
            mbin = (t >= lo) & (t < hi)
        else:
            mbin = (t >= lo) & (t <= hi)
        mbin &= ok_t
        n_m, k_m, f_m, s_m = _high_err_bin_stats(em, mbin, thr)
        n_t, k_t, f_t, s_t = _high_err_bin_stats(et, mbin, thr)
        n_s, k_s, f_s, s_s = _high_err_bin_stats(es, mbin, thr)
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "bin_center": 0.5 * (lo + hi),
                "n_mask": n_m,
                "bad_err_mask": k_m,
                "frac_bad_mask": f_m,
                "sigma_frac_poisson_mask": s_m,
                "n_template": n_t,
                "bad_err_template": k_t,
                "frac_bad_template": f_t,
                "sigma_frac_poisson_template": s_t,
                "n_strong_lines": n_s,
                "bad_err_strong_lines": k_s,
                "frac_bad_strong_lines": f_s,
                "sigma_frac_poisson_strong_lines": s_s,
            }
        )
    return pd.DataFrame(rows)


def _plot_high_err_fraction_vs_teff(
    btab: pd.DataFrame,
    max_err: float,
    outpath: Path,
    *,
    teff_bin_lo: float,
    teff_bin_hi: float,
) -> None:
    if btab.empty or not np.any(
        (btab["n_mask"].values > 0) | (btab["n_template"].values > 0) | (btab["n_strong_lines"].values > 0)
    ):
        return
    x = btab["bin_center"].values
    fig, ax = plt.subplots(figsize=(7.8, 4.6))

    def _plot_one(n_col: str, frac_col: str, sig_col: str, fmt: str, color: str, label: str) -> None:
        n = btab[n_col].values.astype(int)
        m = n > 0
        if not np.any(m):
            return
        ax.errorbar(
            x[m],
            btab[frac_col].values[m].astype(float),
            yerr=btab[sig_col].values[m].astype(float),
            fmt=fmt,
            color=color,
            capsize=3,
            lw=1.1,
            ms=5,
            label=label,
        )

    _plot_one("n_mask", "frac_bad_mask", "sigma_frac_poisson_mask", "o-", "C0", "mask CCF")
    _plot_one("n_template", "frac_bad_template", "sigma_frac_poisson_template", "s-", "C1", "template FFT")
    _plot_one(
        "n_strong_lines",
        "frac_bad_strong_lines",
        "sigma_frac_poisson_strong_lines",
        "^-",
        "C2",
        "strong lines",
    )
    ax.set_xlim(float(teff_bin_lo), float(teff_bin_hi))
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("Fraction with σ > threshold")
    ax.set_title(f"Fraction of exposures with method σ > {max_err:g} km/s (Poisson error bars)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RV method diagnostics report")
    ap.add_argument(
        "--diagnostics-glob",
        default=None,
        help="Quoted glob for *_diagnostics.csv (use exactly one of glob or list)",
    )
    ap.add_argument(
        "--diagnostics-list",
        type=Path,
        default=None,
        help="Text file: one absolute or relative path to a *_diagnostics.csv per line (# comments ok)",
    )
    ap.add_argument("--legacy-summary-dir", type=Path, default=None, help="Directory with Gaia_DR3_*_summary.txt")
    ap.add_argument("--max-legacy-err", type=float, default=0.5)
    ap.add_argument(
        "--max-method-err",
        type=float,
        default=float(dh_config.COMPARISON_REPORT_MAX_RV_ERR_KMS),
        help="Exclude comparison-plot points where any involved method has σ above this (km/s).",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--teff-bins", type=int, default=6)
    ap.add_argument(
        "--teff-bin-lo",
        type=float,
        default=float(dh_config.COMPARISON_REPORT_TEFF_BIN_LO_K),
        help="Low edge (K) of fixed equal-width Teff bins for the high-σ fraction plot/CSV.",
    )
    ap.add_argument(
        "--teff-bin-hi",
        type=float,
        default=float(dh_config.COMPARISON_REPORT_TEFF_BIN_HI_K),
        help="High edge (K) of fixed equal-width Teff bins for the high-σ fraction plot/CSV.",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    g_ok = args.diagnostics_glob is not None and str(args.diagnostics_glob).strip() != ""
    l_ok = args.diagnostics_list is not None
    if g_ok == l_ok:
        logging.error("Provide exactly one of --diagnostics-glob or --diagnostics-list.")
        return 2

    if l_ok:
        lines = args.diagnostics_list.read_text().splitlines()
        paths = []
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            paths.append(Path(s))
        paths = sorted({p.resolve() for p in paths if p.is_file()})
        if not paths:
            logging.error("No existing files listed in %s", args.diagnostics_list)
            return 2
    else:
        paths = sorted(Path(p) for p in glob_paths(args.diagnostics_glob))
        if not paths:
            logging.error("No diagnostics matched %r", args.diagnostics_glob)
            return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as ex:
            logging.warning("skip %s: %s", p, ex)
            continue
        recs = df.to_dict("records")
        fn = str(df["file"].iloc[0]) if "file" in df.columns and len(df) else ""
        basename = Path(fn).name if fn else p.stem.replace("_diagnostics", "") + ".txt"
        teff = float(df["teff"].iloc[0]) if "teff" in df.columns and len(df) else float("nan")
        gid = _gaia_from_path(fn) or _gaia_from_path(p.name)

        rv_m, er_m = _weighted_method_rv_from_rows(recs, "mask_ccf")
        rv_t, er_t = _weighted_method_rv_from_rows(recs, "template_fft")
        rv_sl, er_sl = _weighted_method_rv_from_rows(recs, "strong_lines")

        fl = exposure_method_flags(recs)
        snr_m = float(fl["median_mask_ccf_peak_snr"])
        if np.isfinite(snr_m) and snr_m > 0:
            log10_snr = float(np.log10(snr_m))
        else:
            log10_snr = float("nan")

        leg_rv, leg_e = float("nan"), float("nan")
        if args.legacy_summary_dir and gid is not None:
            lr = _load_legacy_rv(basename, args.legacy_summary_dir, gid)
            if lr is not None:
                leg_rv, leg_e = lr

        rows_out.append(
            {
                "diagnostics_csv": str(p),
                "basename": basename,
                "gaia_source_id": gid,
                "teff": teff,
                "log10_median_mask_ccf_peak_snr": log10_snr,
                "mask_rv_kms": rv_m,
                "mask_err_kms": er_m,
                "template_rv_kms": rv_t,
                "template_err_kms": er_t,
                "strong_lines_rv_kms": rv_sl,
                "strong_lines_err_kms": er_sl,
                "legacy_rv_kms": leg_rv,
                "legacy_err_kms": leg_e,
            }
        )

    tab = pd.DataFrame(rows_out)
    tab.to_csv(args.out_dir / "method_comparison_per_exposure.csv", index=False)

    # --- Plots ---
    teff = tab["teff"].astype(float).values
    log10_snr = tab["log10_median_mask_ccf_peak_snr"].astype(float).values
    reg_m = _region_mask_applicable(teff, log10_snr)
    reg_t = _region_template_applicable(teff, log10_snr)
    reg_sl = _region_strong_lines_applicable(teff, log10_snr)

    if args.legacy_summary_dir is not None and tab["legacy_rv_kms"].notna().any():
        mleg = (
            np.isfinite(tab["legacy_rv_kms"].values)
            & np.isfinite(tab["legacy_err_kms"].values)
            & (tab["legacy_err_kms"].values <= args.max_legacy_err)
            & np.isfinite(tab["mask_rv_kms"].values)
            & _method_err_acceptable(tab["mask_err_kms"].values, args.max_method_err)
        )
        if int(np.sum(mleg)) >= 1:
            fig, ax = plt.subplots(figsize=(6.2, 6.0))
            lx = tab["legacy_rv_kms"].values[mleg]
            mx = tab["mask_rv_kms"].values[mleg]
            ax.scatter(lx, mx, s=22, alpha=0.7, color="0.2")
            lo = float(np.nanmin([np.nanmin(lx), np.nanmin(mx)]))
            hi = float(np.nanmax([np.nanmax(lx), np.nanmax(mx)]))
            if hi > lo:
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.85, alpha=0.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Legacy mask RV (km/s)")
            ax.set_ylabel("Current mask CCF RV (km/s)")
            ax.set_title(
                f"Legacy vs current mask (legacy σ ≤ {args.max_legacy_err:g}, current mask σ ≤ {args.max_method_err:g} km/s)"
            )
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(args.out_dir / "legacy_vs_current_mask.png", dpi=130)
            plt.close(fig)

    ok_m = _method_err_acceptable(tab["mask_err_kms"].values, args.max_method_err)
    ok_t = _method_err_acceptable(tab["template_err_kms"].values, args.max_method_err)
    ok_sl = _method_err_acceptable(tab["strong_lines_err_kms"].values, args.max_method_err)

    smin = float(dh_config.METHOD_REGION_LOG10_SNR_MIN)
    tc = float(dh_config.METHOD_REGION_MASK_COOL_TEFF_K)
    tw = float(dh_config.METHOD_REGION_MASK_WARM_TEFF_K)
    tsl = float(dh_config.METHOD_REGION_STRONG_LINES_MIN_TEFF_K)
    sub_mt = (
        f"Overlap: mask region (Teff<{tc:g} or (Teff<{tw:g} & log10S/N>{smin:g})); template all. "
        f"σ≤{args.max_method_err:g} km/s."
    )
    sub_ms = (
        f"Overlap: {tsl:g}<Teff<{tw:g} & log10S/N>{smin:g}. σ≤{args.max_method_err:g} km/s."
    )
    sub_ts = f"Overlap: Teff>{tsl:g} & log10S/N>{smin:g}; template all. σ≤{args.max_method_err:g} km/s."

    if np.any(ok_m & ok_t & reg_m & reg_t & np.isfinite(tab["mask_rv_kms"]) & np.isfinite(tab["template_rv_kms"])):
        d_mt = tab["mask_rv_kms"].values - tab["template_rv_kms"].values
        mm = ok_m & ok_t & reg_m & reg_t
        bias_mt = float(np.nanmedian(d_mt[mm]))
        _plot_residual_vs_teff(
            teff[mm],
            d_mt[mm],
            "mask − template (km/s)",
            "Mask CCF minus template FFT",
            args.out_dir / "residual_mask_minus_template_vs_teff.png",
            args.teff_bins,
            median_bias_kms=bias_mt,
            subtitle=sub_mt,
        )

    if np.any(ok_m & ok_sl & reg_m & reg_sl & np.isfinite(tab["mask_rv_kms"]) & np.isfinite(tab["strong_lines_rv_kms"])):
        d_ms = tab["mask_rv_kms"].values - tab["strong_lines_rv_kms"].values
        mm = ok_m & ok_sl & reg_m & reg_sl
        bias_ms = float(np.nanmedian(d_ms[mm]))
        _plot_residual_vs_teff(
            teff[mm],
            d_ms[mm],
            "mask − strong lines (km/s)",
            "Mask CCF minus strong lines (Voigt+Lorentz Hβ)",
            args.out_dir / "residual_mask_minus_strong_lines_vs_teff.png",
            args.teff_bins,
            median_bias_kms=bias_ms,
            subtitle=sub_ms,
        )

    if np.any(ok_t & ok_sl & reg_t & reg_sl & np.isfinite(tab["template_rv_kms"]) & np.isfinite(tab["strong_lines_rv_kms"])):
        d_ts = tab["template_rv_kms"].values - tab["strong_lines_rv_kms"].values
        mm = ok_t & ok_sl & reg_t & reg_sl
        bias_ts = float(np.nanmedian(d_ts[mm]))
        _plot_residual_vs_teff(
            teff[mm],
            d_ts[mm],
            "template − strong lines (km/s)",
            "Template FFT minus strong lines (Voigt+Lorentz Hβ)",
            args.out_dir / "residual_template_minus_strong_lines_vs_teff.png",
            args.teff_bins,
            median_bias_kms=bias_ts,
            subtitle=sub_ts,
        )

    frac_tab = _binned_high_err_fraction_table(
        teff,
        tab["mask_err_kms"].values,
        tab["template_err_kms"].values,
        tab["strong_lines_err_kms"].values,
        args.max_method_err,
        args.teff_bins,
        teff_bin_lo=args.teff_bin_lo,
        teff_bin_hi=args.teff_bin_hi,
    )
    frac_tab.to_csv(args.out_dir / "binned_high_err_fraction_vs_teff.csv", index=False)
    _plot_high_err_fraction_vs_teff(
        frac_tab,
        args.max_method_err,
        args.out_dir / "high_err_fraction_vs_teff.png",
        teff_bin_lo=args.teff_bin_lo,
        teff_bin_hi=args.teff_bin_hi,
    )

    logging.info("Wrote report under %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
