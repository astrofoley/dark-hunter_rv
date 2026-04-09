#!/usr/bin/env python3
"""
2D maps: (Teff, log10 median mask CCF peak S/N) vs method success.

Input: overlap_enriched_per_exposure.csv from rv_method_overlap_report.

For each cell, the sample is spectra with **finite Teff** and **finite** ``log10_median_mask_ccf_peak_snr``
in that bin (``n_total``).

Per method, two heatmaps (six PNGs total):

1. **Of all spectra in the cell:** count with a measured RV **and** reported σ strictly below the
   threshold, divided by ``n_total``.  (Matches: N(σ < dv_max) / N(spectra in cell).)

2. **Conditional on a measurement:** same numerator, denominator = spectra in the cell that have
   **any** finite exposure-level RV from that method.  (Matches: N(σ < dv_max) / N(measured).)

Uses strict inequality ``rv_err_kms < --max-sigma-kms`` (not ≤).

Also writes ``method_success_teff_logsnr_grid.csv`` with counts and both fractions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config as dh_config


def _uniform_edges(n_bins: int, lo: float, hi: float) -> np.ndarray:
    nb = int(max(n_bins, 1))
    lo_f, hi_f = float(lo), float(hi)
    if hi_f <= lo_f:
        hi_f = lo_f + 1.0
    return np.linspace(lo_f, hi_f, nb + 1)


def _cell_mask(teff, log_snr, te_edges, sn_edges, i, j):
    n_te = len(te_edges) - 1
    n_sn = len(sn_edges) - 1
    t_lo, t_hi = float(te_edges[i]), float(te_edges[i + 1])
    s_lo, s_hi = float(sn_edges[j]), float(sn_edges[j + 1])
    mt = (teff >= t_lo) & (teff < t_hi) if i < n_te - 1 else (teff >= t_lo) & (teff <= t_hi)
    ms = (log_snr >= s_lo) & (log_snr < s_hi) if j < n_sn - 1 else (log_snr >= s_lo) & (log_snr <= s_hi)
    return np.isfinite(teff) & np.isfinite(log_snr) & mt & ms


def _accumulate(teff, log_snr, measured, sigma_ok, te_edges, sn_edges):
    n_te = len(te_edges) - 1
    n_sn = len(sn_edges) - 1
    n_tot = np.zeros((n_te, n_sn))
    n_m = np.zeros((n_te, n_sn))
    n_g = np.zeros((n_te, n_sn))
    for i in range(n_te):
        for j in range(n_sn):
            m = _cell_mask(teff, log_snr, te_edges, sn_edges, i, j)
            nt = float(np.sum(m))
            n_tot[i, j] = nt
            if nt > 0:
                n_m[i, j] = float(np.sum(m & measured))
                n_g[i, j] = float(np.sum(m & sigma_ok))
    with np.errstate(divide="ignore", invalid="ignore"):
        f_m = np.where(n_tot > 0, n_m / n_tot, np.nan)
        f_g_of_total = np.where(n_tot > 0, n_g / n_tot, np.nan)
        f_g_of_measured = np.where(n_m > 0, n_g / n_m, np.nan)
    return n_tot, n_m, n_g, f_m, f_g_of_total, f_g_of_measured


def _plot_mesh(te_edges, sn_edges, Z, title, cbar_label, outpath: Path):
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    mesh = ax.pcolormesh(te_edges, sn_edges, Z.T, shading="flat", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel(r"$\log_{10}$ median mask CCF peak S/N")
    ax.set_title(title)
    ax.set_xlim(float(te_edges[0]), float(te_edges[-1]))
    ax.set_ylim(float(sn_edges[0]), float(sn_edges[-1]))
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    try:
        cte = 0.5 * (te_edges[:-1] + te_edges[1:])
        csn = 0.5 * (sn_edges[:-1] + sn_edges[1:])
        Zt = np.where(np.isfinite(Z), Z, np.nan)
        ax.contour(cte, csn, Zt.T, levels=[0.25, 0.5, 0.75], colors="white", linewidths=0.55, alpha=0.5)
    except ValueError:
        pass
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--max-sigma-kms",
        type=float,
        default=float(dh_config.COMPARISON_REPORT_MAX_RV_ERR_KMS),
        help="Strict upper bound: count σ < this value (km/s), e.g. 2.5 → keep err < 2.5.",
    )
    ap.add_argument("--teff-bins", type=int, default=6)
    ap.add_argument("--teff-bin-lo", type=float, default=float(dh_config.COMPARISON_REPORT_TEFF_BIN_LO_K))
    ap.add_argument("--teff-bin-hi", type=float, default=float(dh_config.COMPARISON_REPORT_TEFF_BIN_HI_K))
    ap.add_argument("--snr-bins", type=int, default=6)
    ap.add_argument("--log10-snr-lo", type=float, default=0.35)
    ap.add_argument("--log10-snr-hi", type=float, default=1.2)
    args = ap.parse_args(argv)

    tab = pd.read_csv(args.overlap_csv)
    teff = tab["teff"].astype(float).values
    log_snr = tab["log10_median_mask_ccf_peak_snr"].astype(float).values

    te_edges = _uniform_edges(args.teff_bins, args.teff_bin_lo, args.teff_bin_hi)
    sn_edges = _uniform_edges(args.snr_bins, args.log10_snr_lo, args.log10_snr_hi)

    thr = float(args.max_sigma_kms)
    methods = [
        ("mask_ccf", "mask_valid", "mask_err_kms", "mask"),
        ("template_fft", "template_valid", "template_err_kms", "template_fft"),
        ("strong_lines", "strong_lines_valid", "strong_lines_err_kms", "strong_lines"),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for meth, vcol, ecol, slug in methods:
        measured = tab[vcol].astype(bool).values
        err = tab[ecol].astype(float).values
        sigma_ok = measured & np.isfinite(err) & (err > 0) & (err < thr)

        n_tot, n_m, n_g, f_meas_of_total, f_sig_of_total, f_sig_of_measured = _accumulate(
            teff, log_snr, measured, sigma_ok, te_edges, sn_edges
        )
        n_te, n_sn = n_tot.shape
        for i in range(n_te):
            for j in range(n_sn):
                rows.append({
                    "method": meth,
                    "teff_bin_lo": float(te_edges[i]),
                    "teff_bin_hi": float(te_edges[i + 1]),
                    "log10_snr_bin_lo": float(sn_edges[j]),
                    "log10_snr_bin_hi": float(sn_edges[j + 1]),
                    "n_total": int(n_tot[i, j]),
                    "n_measured": int(n_m[i, j]),
                    "n_sigma_lt_threshold": int(n_g[i, j]),
                    "frac_measured_of_total": float(f_meas_of_total[i, j])
                    if np.isfinite(f_meas_of_total[i, j])
                    else float("nan"),
                    "frac_sigma_lt_threshold_of_total": float(f_sig_of_total[i, j])
                    if np.isfinite(f_sig_of_total[i, j])
                    else float("nan"),
                    "frac_sigma_lt_threshold_of_measured": float(f_sig_of_measured[i, j])
                    if np.isfinite(f_sig_of_measured[i, j])
                    else float("nan"),
                })

        sig_name = str(thr).replace(".", "p")
        _plot_mesh(
            te_edges,
            sn_edges,
            f_sig_of_total,
            f"{meth}: N(σ < {thr:g} km/s) / N(spectra in cell)",
            f"Fraction with σ < {thr:g} km/s",
            args.out_dir / f"success_teff_logsnr_{slug}_sigma_lt_{sig_name}kms_over_total.png",
        )
        _plot_mesh(
            te_edges,
            sn_edges,
            f_sig_of_measured,
            f"{meth}: N(σ < {thr:g} km/s) / N(with method RV in cell)",
            f"Fraction with σ < {thr:g} km/s | measured",
            args.out_dir / f"success_teff_logsnr_{slug}_sigma_lt_{sig_name}kms_over_measured.png",
        )

    pd.DataFrame(rows).to_csv(args.out_dir / "method_success_teff_logsnr_grid.csv", index=False)
    print(
        "Wrote 6 PNGs (σ<threshold / total, σ<threshold / measured) + "
        "method_success_teff_logsnr_grid.csv ->",
        args.out_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
