#!/usr/bin/env python3
"""
Enrich per-exposure diagnostics with method validity, overlap, adopted RV, Gaia metadata, and S/N proxy.

Reads ``*_diagnostics.csv`` (same glob as :mod:`validation.rv_method_diagnostics_report`), optionally merges
``RUWE``, ``MH``, ``logg`` from ``Gaia_DR3_*_summary.txt`` via
:func:`darkhunter_rv.gaia_utils.parse_gaia_metadata_from_star_summary`.

Writes (under ``--out-dir``):

* ``overlap_enriched_per_exposure.csv`` — validity flags, pairwise residuals where both methods pass,
  ``adopted_*`` from smallest reported error among valid methods in overlap (see
  :mod:`darkhunter_rv.method_evaluation`).
* ``binned_mask_minus_template_vs_teff.csv`` — median/MAD of (mask − template) in Teff bins
  (only bins with ``n >= --min-bin-count``).
* ``binned_mask_minus_template_vs_log10_snr.csv`` — same vs ``log10(median_mask_ccf_peak_snr)``.
* Plots: ``overlap_method_count_hist.png``, ``residual_mask_minus_template_vs_log10_snr.png``,
  ``residual_mask_minus_template_vs_teff_overlap_only.png`` (mask and template both valid, each
  with σ ≤ ``--max-method-err``, default ``config.COMPARISON_REPORT_MAX_RV_ERR_KMS``).
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

from darkhunter_rv import config as dh_config  # noqa: E402
from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary  # noqa: E402
from darkhunter_rv.method_evaluation import (  # noqa: E402
    exposure_method_flags,
    recommend_adopted_rv,
)


def _gaia_from_path(s: str) -> int | None:
    m = re.search(r"Gaia_DR3_(\d{18,19})", str(s))
    return int(m.group(1)) if m else None


def _meta_float(meta: dict | None, key: str) -> float:
    if not meta or key not in meta:
        return float("nan")
    v = meta[key]
    try:
        xf = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return xf if np.isfinite(xf) else float("nan")


def _bin_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    t = x[np.isfinite(x)]
    if len(t) < 2:
        return np.array([0.0, 1.0])
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(t, q))
    if len(edges) < 2:
        edges = np.array([float(np.min(t)), float(np.max(t)) + 1e-6])
    return edges


def _binned_table(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    min_count: int,
    x_label: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < len(edges) - 2:
            m = (x >= lo) & (x < hi)
        else:
            m = (x >= lo) & (x <= hi)
        m &= np.isfinite(x) & np.isfinite(y)
        n = int(np.sum(m))
        if n < min_count:
            continue
        yy = y[m]
        med = float(np.median(yy))
        mad = float(np.median(np.abs(yy - med)) * 1.4826)
        rows.append(
            {
                "x_var": x_label,
                "bin_lo": lo,
                "bin_hi": hi,
                "bin_center": 0.5 * (lo + hi),
                "n": n,
                "median_y": med,
                "mad_y": mad,
            }
        )
    return pd.DataFrame(rows)


def _plot_residual_vs_x(
    x: np.ndarray,
    residual: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    outpath: Path,
    n_bins: int,
    min_bin_count: int,
) -> None:
    ok = np.isfinite(x) & np.isfinite(residual)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(x[ok], residual[ok], s=18, alpha=0.45, color="0.25", label="exposures")
    ax.axhline(0.0, color="k", lw=0.7)
    edges = _bin_edges(x[ok], n_bins)
    tab = _binned_table(x[ok], residual[ok], edges, min_bin_count, xlabel)
    if len(tab) > 0:
        ax.errorbar(
            tab["bin_center"].values,
            tab["median_y"].values,
            yerr=tab["mad_y"].values,
            fmt="s-",
            color="C0",
            capsize=3,
            lw=1.2,
            ms=5,
            label=f"median ± MAD (n≥{min_bin_count})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RV method overlap / adoption enrichment report")
    ap.add_argument(
        "--diagnostics-glob",
        default=None,
        help="Quoted glob for *_diagnostics.csv (exactly one of glob or list)",
    )
    ap.add_argument(
        "--diagnostics-list",
        type=Path,
        default=None,
        help="Text file: one path per line to *_diagnostics.csv (same as rv_method_diagnostics_report)",
    )
    ap.add_argument(
        "--gaia-summary-dir",
        type=Path,
        default=None,
        help="Directory with Gaia_DR3_*_summary.txt for RUWE, MH, logg (optional)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--teff-bins", type=int, default=6)
    ap.add_argument("--snr-bins", type=int, default=6)
    ap.add_argument("--min-bin-count", type=int, default=15)
    ap.add_argument(
        "--max-method-err",
        type=float,
        default=float(dh_config.COMPARISON_REPORT_MAX_RV_ERR_KMS),
        help="Mask/template residual plots: require both σ ≤ this value (km/s).",
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

        meta = None
        if args.gaia_summary_dir is not None and gid is not None:
            sp = args.gaia_summary_dir / f"Gaia_DR3_{gid}_summary.txt"
            meta = parse_gaia_metadata_from_star_summary(sp)

        fl = exposure_method_flags(recs)
        snr = float(fl["median_mask_ccf_peak_snr"])
        log_snr = float(np.log10(snr)) if np.isfinite(snr) and snr > 0 else float("nan")
        ad = recommend_adopted_rv(
            fl,
            teff=teff,
            log10_median_mask_ccf_peak_snr=log_snr,
        )

        rv_m, rv_t = fl["mask_rv_kms"], fl["template_rv_kms"]
        rv_s = fl["strong_lines_rv_kms"]
        d_mt = float(rv_m - rv_t) if fl["mask_valid"] and fl["template_valid"] else float("nan")
        d_ms = float(rv_m - rv_s) if fl["mask_valid"] and fl["strong_lines_valid"] else float("nan")
        d_ts = float(rv_t - rv_s) if fl["template_valid"] and fl["strong_lines_valid"] else float("nan")

        rows_out.append(
            {
                "diagnostics_csv": str(p),
                "basename": basename,
                "gaia_source_id": gid,
                "teff": teff,
                "gaia_ruwe": _meta_float(meta, "RUWE"),
                "gaia_mh": _meta_float(meta, "MH"),
                "gaia_logg": _meta_float(meta, "logg"),
                "median_mask_ccf_peak_snr": snr,
                "log10_median_mask_ccf_peak_snr": log_snr,
                "n_mask_chunks_qc_pass": fl["n_mask_chunks_qc_pass"],
                "n_template_chunks_qc_pass": fl["n_template_chunks_qc_pass"],
                "mask_valid": fl["mask_valid"],
                "template_valid": fl["template_valid"],
                "strong_lines_valid": fl["strong_lines_valid"],
                "n_methods_valid": fl["n_methods_valid"],
                "overlap_2plus": fl["overlap_2plus"],
                "mask_rv_kms": fl["mask_rv_kms"],
                "mask_err_kms": fl["mask_err_kms"],
                "template_rv_kms": fl["template_rv_kms"],
                "template_err_kms": fl["template_err_kms"],
                "strong_lines_rv_kms": fl["strong_lines_rv_kms"],
                "strong_lines_err_kms": fl["strong_lines_err_kms"],
                "delta_mask_minus_template_kms": d_mt,
                "delta_mask_minus_strong_lines_kms": d_ms,
                "delta_template_minus_strong_lines_kms": d_ts,
                "adopted_method": ad["adopted_method"],
                "adopted_rv_kms": ad["adopted_rv_kms"],
                "adopted_err_kms": ad["adopted_err_kms"],
            }
        )

    tab = pd.DataFrame(rows_out)
    tab.to_csv(args.out_dir / "overlap_enriched_per_exposure.csv", index=False)

    # --- Binned cross-method bias (mask − template): Teff ---
    me = tab["mask_err_kms"].astype(float).values
    te = tab["template_err_kms"].astype(float).values
    err_ok = (
        np.isfinite(me)
        & np.isfinite(te)
        & (me > 0.0)
        & (te > 0.0)
        & (me <= float(args.max_method_err))
        & (te <= float(args.max_method_err))
    )
    m_pair = tab["mask_valid"].values & tab["template_valid"].values & err_ok
    teff = tab["teff"].astype(float).values
    d_mt = tab["delta_mask_minus_template_kms"].astype(float).values
    if int(np.sum(m_pair)) >= 1:
        edges_t = _bin_edges(teff[m_pair], args.teff_bins)
        b_teff = _binned_table(teff[m_pair], d_mt[m_pair], edges_t, args.min_bin_count, "teff_K")
        b_teff.to_csv(args.out_dir / "binned_mask_minus_template_vs_teff.csv", index=False)
        _plot_residual_vs_x(
            teff[m_pair],
            d_mt[m_pair],
            "Teff (K)",
            "mask − template (km/s)",
            "Mask CCF minus template FFT (both valid)",
            args.out_dir / "residual_mask_minus_template_vs_teff_overlap_only.png",
            args.teff_bins,
            args.min_bin_count,
        )

    # --- Binned vs log10(S/N proxy) ---
    m_snr = m_pair & np.isfinite(tab["log10_median_mask_ccf_peak_snr"].astype(float).values)
    log_snr_col = tab["log10_median_mask_ccf_peak_snr"].astype(float).values
    if int(np.sum(m_snr)) >= 1:
        edges_s = _bin_edges(log_snr_col[m_snr], args.snr_bins)
        b_snr = _binned_table(
            log_snr_col[m_snr],
            d_mt[m_snr],
            edges_s,
            args.min_bin_count,
            "log10_median_mask_ccf_peak_snr",
        )
        b_snr.to_csv(args.out_dir / "binned_mask_minus_template_vs_log10_snr.csv", index=False)
        _plot_residual_vs_x(
            log_snr_col[m_snr],
            d_mt[m_snr],
            r"$\log_{10}$ median mask CCF peak S/N",
            "mask − template (km/s)",
            "Mask CCF minus template FFT vs S/N proxy (both valid)",
            args.out_dir / "residual_mask_minus_template_vs_log10_snr.png",
            args.snr_bins,
            args.min_bin_count,
        )

    # --- Histogram: number of valid methods per exposure ---
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    counts = tab["n_methods_valid"].value_counts().reindex(range(0, 4), fill_value=0)
    ax.bar(counts.index.astype(int), counts.values.astype(int), color="0.35", edgecolor="k", lw=0.6)
    ax.set_xlabel("Number of methods with valid RV")
    ax.set_ylabel("Exposures")
    ax.set_title("Method overlap (validity) distribution")
    ax.set_xticks(list(range(0, 4)))
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "overlap_method_count_hist.png", dpi=130)
    plt.close(fig)

    logging.info("Wrote overlap report under %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
