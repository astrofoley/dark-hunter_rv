"""Diagnostic figures for RV pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, rv_core

logger = logging.getLogger(__name__)


def plot_normalized_order(
    wave: np.ndarray,
    flux_norm: np.ndarray,
    continuum: np.ndarray | None,
    outpath: Path,
    title: str = "",
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(wave, flux_norm, "k-", lw=0.6, label="norm flux")
    if continuum is not None and len(continuum) == len(wave):
        ax.plot(wave, continuum / np.nanmedian(continuum), "r--", lw=0.8, alpha=0.7, label="continuum (scaled)")
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("Wavelength (A)")
    ax.set_ylabel("Norm flux")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def plot_ccf(
    vel: np.ndarray,
    ccf: np.ndarray,
    outpath: Path,
    title: str = "",
    peak_vel: float | None = None,
    gauss_params: tuple[float, ...] | None = None,
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(vel, ccf, "b-", lw=0.8, label="CCF")
    gauss_ok = False
    mu_plot: float | None = None
    if gauss_params is not None:
        gp = gauss_params
        if len(gp) == 5:
            y0, c1, a, mu, sig = gp
        elif len(gp) == 4:
            y0, a, mu, sig = gp
            c1 = 0.0
        else:
            y0, c1, a, mu, sig = 0.0, 0.0, gp[0], gp[1], gp[2]
        if np.isfinite(a) and np.isfinite(mu) and np.isfinite(sig) and abs(sig) > 1e-6:
            gauss_ok = True
            mu_plot = float(mu)
            vg = np.linspace(float(np.min(vel)), float(np.max(vel)), max(200, len(vel) * 2))
            yg = y0 + c1 * vg + a * np.exp(-0.5 * ((vg - mu) / sig) ** 2)
            ax.plot(vg, yg, "g-", lw=1.0, alpha=0.85, label="Gaussian fit")
    if gauss_ok and mu_plot is not None:
        ax.axvline(mu_plot, color="r", ls="--", lw=1.0, label=f"Gaussian center μ = {mu_plot:.2f} km/s")
    elif peak_vel is not None and np.isfinite(peak_vel):
        ax.axvline(
            peak_vel,
            color="0.55",
            ls=":",
            lw=1.0,
            alpha=0.75,
            label=f"CCF argmax only = {peak_vel:.2f} km/s (no Gaussian)",
        )
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("CCF")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def plot_ccf_order_grid(
    panels: list[tuple[str, np.ndarray, np.ndarray, float | None, tuple[float, ...] | None]],
    outpath: Path,
    title: str = "",
) -> None:
    """
    panels: (label, vel, ccf, peak_vel, gauss_params or None).

    Green curve = linear pedestal + Gaussian fit to the CCF peak region. Red vertical = Gaussian
    center **μ** (same frame as ``vel``). Gray dotted = adopted RV from grid argmax when no Gaussian
    was accepted.
    """
    if not panels:
        return
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 2.8 * nrows), squeeze=False)
    for idx, ax in enumerate(axes.ravel()):
        if idx >= n:
            ax.set_visible(False)
            continue
        label, vel, ccf, peak_vel, gauss_p = panels[idx]
        ax.plot(vel, ccf, "b-", lw=0.7)
        gauss_ok = False
        mu_mark: float | None = None
        if gauss_p is not None:
            gp = gauss_p
            if len(gp) == 5:
                y0, c1, a, mu, sig = gp
            elif len(gp) == 4:
                y0, a, mu, sig = gp
                c1 = 0.0
            else:
                y0, c1, a, mu, sig = 0.0, 0.0, gp[0], gp[1], gp[2]
            if np.isfinite(a) and np.isfinite(mu) and np.isfinite(sig) and abs(sig) > 1e-6:
                gauss_ok = True
                mu_mark = float(mu)
                vg = np.linspace(float(np.min(vel)), float(np.max(vel)), 200)
                yg = y0 + c1 * vg + a * np.exp(-0.5 * ((vg - mu) / sig) ** 2)
                ax.plot(vg, yg, "g-", lw=0.9, alpha=0.85)
        if gauss_ok and mu_mark is not None:
            ax.axvline(mu_mark, color="r", ls="--", lw=0.95, alpha=0.9)
        elif peak_vel is not None and np.isfinite(peak_vel):
            ax.axvline(peak_vel, color="0.55", ls=":", lw=0.9, alpha=0.7)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("km/s")
        ax.set_ylabel("CCF")
    fig.suptitle(
        title + "\n(green = offset + Gaussian fit; red = Gaussian μ; gray = CCF argmax, no fit)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def plot_rv_vs_order(
    orders: np.ndarray,
    rvs: np.ndarray,
    errs: np.ndarray,
    methods: list[str],
    outpath: Path,
    exposure_rv: float | None = None,
    title: str = "",
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"mask_ccf": "C0", "template_fft": "C1"}
    for m in sorted(set(methods)):
        sel = np.array([x == m for x in methods])
        c = colors.get(m, "0.4")
        ax.errorbar(
            orders[sel],
            rvs[sel],
            yerr=errs[sel],
            fmt="o",
            ms=3,
            capsize=2,
            label=m,
            color=c,
            alpha=0.85,
        )
    if exposure_rv is not None and np.isfinite(exposure_rv):
        o0, o1 = float(np.min(orders)), float(np.max(orders))
        ax.plot([o0, o1], [exposure_rv, exposure_rv], "k--", lw=0.9, label="exposure RV")
    ax.set_xlabel("Echelle order")
    ax.set_ylabel("RV (km/s)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def plot_chunk_rvs(
    chunk_keys: list,
    rvs: np.ndarray,
    errs: np.ndarray,
    outpath: Path,
    title: str = "",
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(chunk_keys))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(x, rvs, yerr=errs, fmt="o", capsize=2, ms=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in chunk_keys], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("RV (km/s)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_tournament_scores(names: list, scores: list, outpath: Path) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.2)))
    y = np.arange(len(names))
    ax.barh(y, scores)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Tournament score (sum CCF peaks)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def plot_methods_rv_vs_mjd(
    msum_with_mjd: pd.DataFrame,
    outpath: Path,
    title: str = "",
) -> None:
    """
    Per-method exposure-mean RV vs MJD (for hot stars / multi-method diagnostics).
    Expects columns: basename, method, rv_kms, mjd; optional rv_err_kms.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df = msum_with_mjd.copy()
    df["mjd"] = pd.to_numeric(df["mjd"], errors="coerce")
    df["rv_kms"] = pd.to_numeric(df["rv_kms"], errors="coerce")
    df = df[np.isfinite(df["mjd"].values) & np.isfinite(df["rv_kms"].values)]
    if df.empty:
        return
    methods = sorted(df["method"].dropna().unique())
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, meth in enumerate(methods):
        sub = df[df["method"] == meth].sort_values("mjd")
        mj = sub["mjd"].values
        rv = sub["rv_kms"].values
        if "rv_err_kms" in sub.columns:
            err = pd.to_numeric(sub["rv_err_kms"], errors="coerce").values
            err = np.where(np.isfinite(err), err, 0.0)
            ax.errorbar(
                mj,
                rv,
                yerr=err,
                fmt=markers[i % len(markers)],
                ms=4,
                capsize=2,
                ls="-",
                lw=0.9,
                alpha=0.85,
                label=str(meth),
            )
        else:
            ax.plot(
                mj,
                rv,
                marker=markers[i % len(markers)],
                ms=5,
                ls="-",
                lw=0.9,
                alpha=0.85,
                label=str(meth),
            )
    ax.set_xlabel("MJD")
    ax.set_ylabel("RV (km/s)")
    ax.set_title(title or "Per-method exposure RV vs MJD")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def plot_balmer_panels(spec_data: dict, mjd: float, rv: float, outpath: Path) -> None:
    lines = {"Ha": 6562.8, "Hb": 4861.3, "Hg": 4340.5, "Hd": 4101.7}
    all_w, all_f = [], []
    for _o, d in spec_data.items():
        all_w.append(np.array(d["wavelength"]))
        all_f.append(np.array(d["flux"]))
    if not all_w:
        return
    flat_w = np.concatenate(all_w)
    flat_f = np.concatenate(all_f)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, (name, rest) in zip(axes, lines.items()):
        m = (flat_w > rest - 50) & (flat_w < rest + 50)
        if np.sum(m) < 10:
            ax.set_visible(False)
            continue
        w, f = flat_w[m], flat_f[m]
        f = f / np.nanpercentile(f, 95)
        vel = config.C_KMS * (w - rest) / rest
        ax.plot(vel, f, lw=0.8)
        ax.axvline(rv, color="k", ls="--")
        ax.set_title(name)
        ax.set_xlabel("km/s")
    fig.suptitle(f"MJD {mjd:.4f}  RV ~ {rv:.1f} km/s")
    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def _match_absorption_display(ref: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    """Affine-adjust template absorption for plotting when variance is tiny (e.g. bad overlap)."""
    r = np.asarray(ref, float)
    t = np.asarray(tpl, float)
    m = np.isfinite(r) & np.isfinite(t)
    if int(np.sum(m)) < 8:
        return t
    ro, tt = r[m], t[m]
    so, st = float(np.nanstd(ro)), float(np.nanstd(tt))
    if st <= 1e-12:
        return (t - float(np.nanmean(tt))) + float(np.nanmean(ro))
    if st < 0.11 * max(so, 1e-12):
        mo, mt = float(np.nanmean(ro)), float(np.nanmean(tt))
        return (t - mt) * (so / st) + mo
    return t


def _match_zscore_display(obs_z: np.ndarray, tpl_z: np.ndarray) -> np.ndarray:
    if int(np.sum(np.isfinite(obs_z) & np.isfinite(tpl_z))) < 8:
        return tpl_z
    m = np.isfinite(obs_z) & np.isfinite(tpl_z)
    so, st = float(np.nanstd(obs_z[m])), float(np.nanstd(tpl_z[m]))
    if st < 0.09 * max(so, 1e-12):
        mo, mt = float(np.nanmean(obs_z[m])), float(np.nanmean(tpl_z[m]))
        return (tpl_z - mt) * (so / max(st, 1e-12)) + mo
    return tpl_z


def plot_fft_template_comparison(
    series: dict,
    chunk_key: str,
    tpl_key: str,
    rv_fft_kms: float,
    rv_order_kms: float | None,
    outpath: Path,
    rejected: bool = False,
    ccf_bundle: dict | None = None,
) -> None:
    """
    Spectrum vs best template at FFT RV (LSF-degraded), z-scored FFT inputs, and optional FFT CCF
    with restricted / double-Gaussian fits (``ccf_bundle`` from ``rv_core.fit_fft_ccf_models``).
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    wo = np.asarray(series["wavelength_obs"], float)
    oa = np.asarray(series["obs_absorption"], float)
    ta = np.asarray(series["tpl_absorption_on_obs"], float)
    wg = np.asarray(series["wavelength_fft_grid"], float)
    oz = np.asarray(series["obs_zscore"], float)
    tz = np.asarray(series["tpl_mean_centered"], float)

    nrows = 3 if ccf_bundle and ccf_bundle.get("ok") else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.15 + 3.05 * nrows), sharex=False)
    ax0, ax1 = axes[0], axes[1]
    ta_plot = _match_absorption_display(oa, ta)
    tlab0 = (
        "template (LSF, shifted, display-scaled)"
        if not (
            ta.shape == ta_plot.shape
            and np.all(np.isfinite(ta))
            and np.all(np.isfinite(ta_plot))
            and float(np.nanmax(np.abs(ta - ta_plot))) < 1e-8
        )
        else "template (LSF, shifted)"
    )
    ax0.plot(wo, oa, "k-", lw=0.7, label="obs (1 − norm)")
    ax0.plot(wo, ta_plot, "r-", lw=0.7, alpha=0.85, label=tlab0)
    ax0.set_xlabel("Wavelength (Å)")
    ax0.set_ylabel("Absorption-style flux")
    ax0.legend(fontsize=8)
    ax0.set_title(f"{chunk_key}  template={tpl_key}")

    tz_plot = _match_zscore_display(oz, tz)
    tlab1 = (
        "template (display-scaled)"
        if not (
            tz.shape == tz_plot.shape
            and np.all(np.isfinite(tz))
            and np.all(np.isfinite(tz_plot))
            and float(np.nanmax(np.abs(tz - tz_plot))) < 1e-8
        )
        else "template z-scored (affine-matched)"
    )
    ax1.plot(wg, oz, "k-", lw=0.6, label="obs z-scored")
    ax1.plot(wg, tz_plot, "r-", lw=0.6, alpha=0.85, label=tlab1)
    ax1.set_xlabel("Wavelength (Å)")
    ax1.set_ylabel("FFT-style (pre-Hanning)")
    ax1.legend(fontsize=8)

    if nrows == 3:
        ax2 = axes[2]
        vf = np.asarray(ccf_bundle["vel_full"], float)
        yf = np.asarray(ccf_bundle["ccf_full"], float)
        ax2.plot(vf, yf, "k-", lw=0.65, alpha=0.85, label="FFT CCF")
        vw = ccf_bundle.get("vel_wide")
        yw = ccf_bundle.get("ccf_wide")
        if vw is not None and yw is not None and len(vw) > 5:
            ax2.plot(np.asarray(vw, float), np.asarray(yw, float), "b.", ms=2, alpha=0.35, label="wide fit window")
        vfw = ccf_bundle.get("v_fine_wide")
        y1w = ccf_bundle.get("ccf_fit_single_wide")
        if vfw is not None and y1w is not None and len(vfw) > 1:
            ax2.plot(np.asarray(vfw, float), np.asarray(y1w, float), "g-", lw=1.0, alpha=0.9, label="Gauss wide")
        y2w = ccf_bundle.get("ccf_fit_double_wide")
        if ccf_bundle.get("use_double") and vfw is not None and y2w is not None and len(vfw) > 1:
            ax2.plot(np.asarray(vfw, float), np.asarray(y2w, float), "m-", lw=1.0, alpha=0.9, label="double Gauss")
        vc = ccf_bundle.get("vel_core")
        yc = ccf_bundle.get("ccf_core")
        vfc = ccf_bundle.get("v_fine_core")
        y1c = ccf_bundle.get("ccf_fit_core")
        if vc is not None and yc is not None and vfc is not None and y1c is not None and len(vfc) > 1:
            ax2.plot(np.asarray(vc, float), np.asarray(yc, float), color="orange", ls="", marker="o", ms=2, alpha=0.4, label="core window")
            ax2.plot(np.asarray(vfc, float), np.asarray(y1c, float), color="darkorange", ls="--", lw=1.0, alpha=0.95, label="Gauss core")
        sc = ccf_bundle.get("single_core") or {}
        sw = ccf_bundle.get("single_wide") or {}
        dw = ccf_bundle.get("double_wide") or {}
        ann = []
        if sc.get("mu_kms") is not None and np.isfinite(sc.get("mu_kms", np.nan)):
            mu = float(sc["mu_kms"])
            ax2.axvline(mu, color="darkorange", ls="--", lw=0.9)
            ann.append(f"core fit μ={mu:.1f} km/s")
        if sw.get("mu_kms") is not None and np.isfinite(sw.get("mu_kms", np.nan)):
            ax2.axvline(float(sw["mu_kms"]), color="green", ls=":", lw=0.9)
            ann.append(f"wide Gauss μ={float(sw['mu_kms']):.1f} km/s")
        if ccf_bundle.get("use_double") and dw.get("mu1_kms") is not None:
            ax2.axvline(float(dw["mu1_kms"]), color="magenta", ls=":", lw=0.85)
            ax2.axvline(float(dw["mu2_kms"]), color="magenta", ls="-.", lw=0.75, alpha=0.7)
            ann.append(f"dbl: μ1={float(dw['mu1_kms']):.1f} μ2={float(dw['mu2_kms']):.1f} km/s")
        ax2.axvline(float(ccf_bundle.get("peak_vel_grid", np.nan)), color="gray", ls=":", lw=0.7, alpha=0.6)
        ax2.set_xlabel("Lag velocity (km/s)")
        ax2.set_ylabel("Correlation")
        ax2.set_xlim(float(np.min(vf)), float(np.max(vf)))
        ax2.legend(fontsize=6, loc="upper right", ncol=2)
        if ann:
            ax2.set_title("  |  ".join(ann), fontsize=8)

    tag = f"FFT RV = {rv_fft_kms:.2f} km/s"
    if rv_order_kms is not None and np.isfinite(rv_order_kms):
        tag += f"  |  order/stack RV = {rv_order_kms:.2f} km/s"
    if rejected:
        tag += "  (chunk RV rejected: |rv| cap or flat-like CCF)"
    fig.suptitle(tag, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def _plot_strong_line_on_ax(ax, p: dict) -> None:
    """Draw step data, multi-method curves, and vertical markers (full or zoom x-range)."""
    vf = np.asarray(p.get("v_fine_full", p.get("v_fine", [])), float)
    vpf = np.asarray(p.get("v_kms_plot_full", p.get("v_kms_plot", p["v_kms"])), float)
    fpf = np.asarray(p.get("flux_plot_full", p.get("flux_plot", p["flux"])), float)
    ax.step(vpf, fpf, where="mid", color="0.25", lw=0.9, alpha=0.95, label="data (step)")
    fg = p.get("flux_gauss_full", p.get("flux_gauss"))
    if fg is not None and len(vf) == len(fg):
        ax.plot(vf, np.asarray(fg, float), "g-", lw=1.0, alpha=0.9, label="Gaussian")
    flr = p.get("flux_lorentz_full", p.get("flux_lorentz"))
    if flr is not None and len(vf) == len(flr):
        ax.plot(vf, np.asarray(flr, float), "b-", lw=1.0, alpha=0.85, label="Lorentzian")
    fv = p.get("flux_voigt_full", p.get("flux_voigt"))
    if fv is not None and len(vf) == len(fv):
        ax.plot(vf, np.asarray(fv, float), color="darkorange", lw=1.0, alpha=0.9, label="Voigt")
    fcg = p.get("flux_core_gauss_full", p.get("flux_core_gauss"))
    if fcg is not None and len(vf) == len(fcg):
        ax.plot(vf, np.asarray(fcg, float), color="purple", lw=1.0, alpha=0.9, ls="--", label="Gauss core")
    fsm = p.get("flux_smooth_full", p.get("flux_smooth"))
    if fsm is not None and len(vf) == len(fsm):
        ax.plot(vf, np.asarray(fsm, float), color="saddlebrown", lw=1.2, alpha=0.85, label="smoothed")
    rg, eg = float(p["rv_gauss_kms"]), float(p["err_gauss_kms"])
    rl, el = float(p.get("rv_lorentz_kms", np.nan)), float(p.get("err_lorentz_kms", np.nan))
    rv, ev = float(p["rv_voigt_kms"]), float(p["err_voigt_kms"])
    rc, ec = float(p.get("rv_core_gauss_kms", np.nan)), float(p.get("err_core_gauss_kms", np.nan))
    rs, es = float(p.get("rv_smooth_min_kms", np.nan)), float(p.get("err_smooth_min_kms", np.nan))
    if np.isfinite(rg):
        ax.axvline(rg, color="green", ls="--", lw=0.85)
    if np.isfinite(rl):
        ax.axvline(rl, color="blue", ls="--", lw=0.85)
    if np.isfinite(rv):
        ax.axvline(rv, color="darkorange", ls="--", lw=0.85)
    if np.isfinite(rc):
        ax.axvline(rc, color="purple", ls=":", lw=0.9)
    if np.isfinite(rs):
        ax.axvline(rs, color="saddlebrown", ls=":", lw=0.9)


def _strong_line_fit_annotation_text(p: dict) -> str:
    lines_txt = []
    trust_hw = float(p.get("rv_trust_half_width_kms", p.get("fit_half_width_kms", 150)))

    def _line(label, r, e):
        if np.isfinite(r) and np.isfinite(e):
            return f"{label}: {r:+.2f} ± {e:.2f} km/s"
        if np.isfinite(r):
            if abs(r) > trust_hw:
                return f"{label}: {r:+.2f} km/s (σ n/a, |RV|>{trust_hw:.0f})"
            return f"{label}: {r:+.2f} km/s (σ n/a)"
        return None

    for lab, r, e in (
        ("Gauss", float(p["rv_gauss_kms"]), float(p["err_gauss_kms"])),
        ("Lorentz", float(p.get("rv_lorentz_kms", np.nan)), float(p.get("err_lorentz_kms", np.nan))),
        ("Voigt", float(p["rv_voigt_kms"]), float(p["err_voigt_kms"])),
        ("Core G", float(p.get("rv_core_gauss_kms", np.nan)), float(p.get("err_core_gauss_kms", np.nan))),
        ("Sm min", float(p.get("rv_smooth_min_kms", np.nan)), float(p.get("err_smooth_min_kms", np.nan))),
    ):
        s = _line(lab, r, e)
        if s:
            lines_txt.append(s)
    cap = float(p.get("fit_velocity_cap_kms", np.nan))
    if np.isfinite(cap):
        lines_txt.append(f"(fits |v|≤{cap:.0f} km/s; formal σ only if |RV|≤{trust_hw:.0f} km/s)")
    else:
        lines_txt.append(f"(formal σ only if |RV|≤{trust_hw:.0f} km/s)")
    return "\n".join(lines_txt)


def plot_order_strong_line_panels(
    order: int,
    panels: list[dict],
    rv_order_kms: float | None,
    outpath: Path,
    title_stem: str = "",
) -> None:
    """Per echelle order: each line is stacked — capped-|v| fit window, then zoom — same width, taller panels."""
    if not panels:
        return
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    nrows = 2 * n
    row_h = 3.45
    fig, axes = plt.subplots(nrows, 1, figsize=(9.8, row_h * nrows), squeeze=False)
    ax_list = axes.ravel()
    zh = float(panels[0].get("zoom_half_width_kms", 100.0))
    for i, p in enumerate(panels):
        ax_f = ax_list[2 * i]
        ax_z = ax_list[2 * i + 1]
        _plot_strong_line_on_ax(ax_f, p)
        _plot_strong_line_on_ax(ax_z, p)
        vf = np.asarray(p.get("v_fine_full", p.get("v_fine", [0.0, 1.0])), float)
        if vf.size > 1:
            pad = 0.015 * (float(np.nanmax(vf)) - float(np.nanmin(vf))) + 1e-9
            ax_f.set_xlim(float(np.nanmin(vf)) - pad, float(np.nanmax(vf)) + pad)
        zh_i = float(p.get("zoom_half_width_kms", zh))
        ax_z.set_xlim(-zh_i, zh_i)
        vpf = np.asarray(p.get("v_kms_plot_full", p.get("v_kms_plot", p["v_kms"])), float)
        fpf = np.asarray(p.get("flux_plot_full", p.get("flux_plot", p["flux"])), float)
        mz = (vpf >= -zh_i) & (vpf <= zh_i) & np.isfinite(fpf)
        if int(np.sum(mz)) >= 6:
            ylo, yhi = np.nanpercentile(fpf[mz], [4.0, 96.0])
            if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
                pad_y = 0.1 * (yhi - ylo)
                ax_z.set_ylim(ylo - pad_y, yhi + pad_y)
        mf = np.isfinite(fpf)
        if int(np.sum(mf)) >= 8:
            y1, y2 = np.nanpercentile(fpf[mf], [0.5, 99.5])
            if np.isfinite(y1) and np.isfinite(y2) and y2 > y1:
                pdy = 0.04 * (y2 - y1)
                ax_f.set_ylim(max(-0.02, y1 - pdy), y2 + pdy)
        cap = float(p.get("fit_velocity_cap_kms", np.nan))
        cap_s = f"|v|≤{cap:.0f} km/s" if np.isfinite(cap) else "fit window"
        ax_f.set_title(f"{p['name']}  rest={p['rest']:.2f} Å — {cap_s}", fontsize=10)
        ax_z.set_title(f"{p['name']}  ±{zh_i:.0f} km/s", fontsize=10)
        ax_f.set_xlabel("Velocity (km/s)")
        ax_f.set_ylabel("Norm flux (local cont.)")
        ax_z.set_xlabel("Velocity (km/s)")
        ax_z.set_ylabel("Norm flux (local cont.)")
        ax_f.legend(fontsize=6.5, loc="upper right", ncol=2)
        ax_z.legend(fontsize=6.5, loc="upper right", ncol=2)
        ax_z.text(
            0.02,
            0.98,
            _strong_line_fit_annotation_text(p),
            transform=ax_z.transAxes,
            va="top",
            ha="left",
            fontsize=6.5,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
        )
    otag = f"order {order}"
    if rv_order_kms is not None and np.isfinite(rv_order_kms):
        otag += f"  |  order RV = {rv_order_kms:.2f} km/s"
    fig.suptitle(f"{title_stem}  {otag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=135)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def _h_beta_legend_rv_line(label: str, rv: float, err: float) -> str:
    if np.isfinite(rv) and np.isfinite(err) and err > 0:
        return f"{label}: {rv:+.2f} ± {err:.2f} km/s"
    if np.isfinite(rv):
        return f"{label}: {rv:+.2f} km/s"
    return f"{label}: n/a"


def _hb_mask_pseudo_on_wavelengths(
    nw: np.ndarray,
    mask_wave: np.ndarray,
    mask_strength: np.ndarray,
    rv_mask_kms: float,
) -> np.ndarray | None:
    """Schematic mask transmission in the same spirit as the Hβ order overlay (shift by mask RV)."""
    nw = np.asarray(nw, float)
    mw = np.asarray(mask_wave, float)
    ms = np.asarray(mask_strength, float)
    if mw.size < 5 or ms.shape != mw.shape or nw.size < 5:
        return None
    if not np.isfinite(rv_mask_kms):
        return None
    beta = 1.0 + float(rv_mask_kms) / config.C_KMS
    pseudo = np.ones_like(nw, dtype=float)
    sig_a = max(0.05, float(np.nanmedian(np.abs(np.diff(np.sort(nw))))) * 1.8)
    smax = float(np.nanpercentile(np.abs(ms), 90)) + 1e-9
    for w0, s in zip(mw, ms):
        w_shift = float(w0) * beta
        if w_shift < float(np.min(nw)) or w_shift > float(np.max(nw)):
            continue
        amp = min(0.45, 0.42 * (abs(float(s)) / smax) ** 0.5)
        pseudo -= amp * np.exp(-0.5 * ((nw - w_shift) / sig_a) ** 2)
    return np.clip(pseudo, 0.12, 1.45)


def _hb_template_flux_on_wavelengths(
    nw: np.ndarray,
    tpl_wave: np.ndarray,
    tpl_flux_norm: np.ndarray,
    rv_template_kms: float,
    resolving_power: float | None,
) -> np.ndarray | None:
    """Template norm flux interpolated onto ``nw`` with Doppler shift ``rv_template_kms``."""
    nw = np.asarray(nw, float)
    tw = np.asarray(tpl_wave, float)
    tf = np.asarray(tpl_flux_norm, float)
    if tw.size < 30 or tf.shape != tw.shape or nw.size < 5:
        return None
    if not np.isfinite(rv_template_kms):
        return None
    beta = 1.0 + float(rv_template_kms) / config.C_KMS
    o = np.argsort(tw)
    tws, tfs = tw[o], tf[o]
    w_shifted = tws * beta
    tf_i = np.interp(nw, w_shifted, tfs, left=np.nan, right=np.nan)
    if resolving_power is not None and resolving_power > 1.0:
        tf_i = rv_core.degrade_template_flux_lsf(nw, tf_i, float(resolving_power))
    return np.asarray(np.clip(np.nan_to_num(tf_i, nan=np.nan), 0.05, 2.5), dtype=float)


def _hb_scale_template_to_local_cont(
    v_kms: np.ndarray,
    f_obs: np.ndarray,
    f_tpl: np.ndarray,
    *,
    wing_abs_kms: float = 130.0,
) -> np.ndarray:
    """
    Match template level to local-continuum-divided data using line wings (same idea as
    :func:`plot_h_beta_order_method_overlay` order-norm scaling).
    """
    v_kms = np.asarray(v_kms, float)
    f_obs = np.asarray(f_obs, float)
    f_tpl = np.asarray(f_tpl, float)
    if f_tpl.shape != v_kms.shape or f_obs.shape != v_kms.shape:
        return f_tpl
    wing = np.abs(v_kms) > float(wing_abs_kms)
    ok = wing & np.isfinite(f_obs) & np.isfinite(f_tpl) & (f_tpl > 0.08)
    if int(np.sum(ok)) < 8:
        ok = np.isfinite(f_obs) & np.isfinite(f_tpl) & (f_tpl > 0.08) & (np.abs(v_kms) > 70.0)
    if int(np.sum(ok)) < 6:
        ok = np.isfinite(f_obs) & np.isfinite(f_tpl) & (f_tpl > 0.08)
    if int(np.sum(ok)) < 4:
        return f_tpl
    scale = float(np.nanmedian(f_obs[ok] / (f_tpl[ok] + 1e-9)))
    if not np.isfinite(scale) or scale <= 0:
        return f_tpl
    scale = float(np.clip(scale, 0.15, 6.0))
    return np.clip(f_tpl * scale, 0.05, 2.5)


def plot_h_beta_rv_diagnostic(
    title_stem: str,
    hb: dict,
    outpath: Path,
    order_num: int | None = None,
    *,
    rv_mask_kms: float = float("nan"),
    err_mask_kms: float = float("nan"),
    mask_wave: np.ndarray | None = None,
    mask_strength: np.ndarray | None = None,
    tpl_wave: np.ndarray | None = None,
    tpl_flux_norm: np.ndarray | None = None,
    rv_template_kms: float = float("nan"),
    err_template_kms: float = float("nan"),
    resolving_power: float | None = None,
    template_legend_name: str = "Template FFT",
) -> None:
    """
    Hβ window in velocity: black step data, blue shifted mask schematic, red shifted template,
    gold joint Voigt+Lorentz model; legend lists RV ± σ for mask / template / line fit when known.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(hb["v_kms_plot"], float)
    f = np.asarray(hb["flux_plot"], float)
    rest = float(hb.get("rest_a", rv_core.HB_REST_A))
    w_plot = hb.get("wavelength_plot")
    if w_plot is not None:
        wlam = np.asarray(w_plot, dtype=float)
    else:
        wlam = rest * (1.0 + v / config.C_KMS)

    o = np.argsort(v)
    vs, fs, wsort = v[o], f[o], wlam[o]

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.step(vs, fs, where="mid", color="k", lw=1.05, label="data (local cont.)", zorder=2)

    mask_flux = None
    if mask_wave is not None and mask_strength is not None:
        mask_flux = _hb_mask_pseudo_on_wavelengths(wsort, mask_wave, mask_strength, rv_mask_kms)
    if mask_flux is not None:
        ax.plot(
            vs,
            mask_flux,
            "-",
            color="tab:blue",
            lw=1.05,
            alpha=0.9,
            label=_h_beta_legend_rv_line("Mask CCF", rv_mask_kms, err_mask_kms),
            zorder=4,
        )
        if np.isfinite(rv_mask_kms):
            ax.axvline(float(rv_mask_kms), color="tab:blue", ls="-", lw=1.2, alpha=0.88, zorder=10)

    tpl_flux = None
    if tpl_wave is not None and tpl_flux_norm is not None:
        tpl_flux = _hb_template_flux_on_wavelengths(
            wsort, tpl_wave, tpl_flux_norm, rv_template_kms, resolving_power
        )
    if tpl_flux is not None and np.any(np.isfinite(tpl_flux)):
        tpl_flux = _hb_scale_template_to_local_cont(vs, fs, tpl_flux)
        ax.plot(
            vs,
            tpl_flux,
            "-",
            color="tab:red",
            lw=1.05,
            alpha=0.9,
            label=_h_beta_legend_rv_line(template_legend_name, rv_template_kms, err_template_kms),
            zorder=5,
        )
        if np.isfinite(rv_template_kms):
            ax.axvline(float(rv_template_kms), color="tab:red", ls="-", lw=1.2, alpha=0.88, zorder=10)

    wf = np.asarray(hb.get("voigt_wave_fine", []), float)
    p_joint = hb.get("hb_joint_fit_params")
    vf = None
    if p_joint is not None and len(p_joint) == 8 and wf.size > 2:
        vf = rv_core.h_beta_joint_line_model(wf, p_joint, rest=rest)
    elif hb.get("voigt_model_fine") is not None and wf.size > 2:
        vf = np.asarray(hb["voigt_model_fine"], dtype=float)
    rv_vl = float(hb.get("rv_voigt_kms", np.nan))
    err_vl = float(hb.get("err_voigt_kms", np.nan))
    if p_joint is not None and len(p_joint) >= 8:
        ctr = float(p_joint[2])
        rv_vl = float(config.C_KMS * (ctr - rest) / rest)
    if vf is not None and wf.size > 2:
        vvf = config.C_KMS * (wf / rest - 1.0)
        lf_chk = hb.get("lorentz_model_fine")
        vlab = (
            _h_beta_legend_rv_line("Voigt+Lorentz", rv_vl, err_vl)
            if lf_chk is None
            else _h_beta_legend_rv_line("Voigt (+cont.)", rv_vl, err_vl)
        )
        ax.plot(
            vvf,
            np.asarray(vf, float),
            color="goldenrod",
            lw=1.2,
            label=vlab,
            zorder=6,
        )
        if np.isfinite(rv_vl):
            ax.axvline(float(rv_vl), color="goldenrod", ls="-", lw=1.2, alpha=0.9, zorder=10)
    lf = hb.get("lorentz_model_fine")
    if lf is not None and wf.size > 2:
        vlf = config.C_KMS * (wf / rest - 1.0)
        r_lor = float(hb.get("rv_lorentz_kms", np.nan))
        e_lor = float(hb.get("err_lorentz_kms", np.nan))
        ax.plot(
            vlf,
            np.asarray(lf, float),
            color="steelblue",
            lw=1.05,
            alpha=0.9,
            label=_h_beta_legend_rv_line("Lorentz (wings)", r_lor, e_lor),
            zorder=6,
        )

    rv_cc_d = float(hb.get("rv_template_ccf_kms", np.nan))
    if np.isfinite(rv_cc_d) and (not np.isfinite(rv_vl) or abs(rv_cc_d - rv_vl) > 0.6):
        ax.axvline(rv_cc_d, color="0.35", ls="--", lw=1.1, alpha=0.85, zorder=10)

    ann: list[str] = []
    if np.isfinite(float(hb.get("rv_best_kms", np.nan))):
        ann.append(f"best {float(hb['rv_best_kms']):+.1f} ({hb.get('method_used', '')})")
    if np.isfinite(float(hb.get("rv_smoothed_min_kms", np.nan))):
        ann.append(f"sm: {float(hb['rv_smoothed_min_kms']):+.1f}")
    if hb.get("template_ccf_peak") is not None and np.isfinite(float(hb["template_ccf_peak"])):
        ann.append(f"Hβ tpl CCF r={float(hb['template_ccf_peak']):.3f}")
    otag = f"order {order_num}" if order_num is not None else "Hβ window"
    ax.set_title(f"{title_stem}  {otag}" + ("  |  " + "  ".join(ann) if ann else ""), fontsize=9)
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Norm flux (local cont.)")
    ax.legend(fontsize=6.8, loc="upper right")
    m = np.isfinite(v) & np.isfinite(f)
    if int(np.sum(m)) >= 8:
        win = m & (np.abs(v) <= 420.0)
        if int(np.sum(win)) < 6:
            win = m
        ylo, yhi = np.nanpercentile(f[win], [5.0, 95.0])
        seqs: list[np.ndarray] = [f]
        if vf is not None:
            seqs.append(np.asarray(vf, dtype=float))
        if lf is not None:
            seqs.append(np.asarray(lf, dtype=float))
        if mask_flux is not None:
            seqs.append(mask_flux)
        if tpl_flux is not None:
            seqs.append(tpl_flux[np.isfinite(tpl_flux)])
        for seq in seqs:
            sf = seq[np.isfinite(seq)]
            if sf.size and np.isfinite(ylo) and np.isfinite(yhi):
                ylo = float(min(ylo, np.nanpercentile(sf, 3.0)))
                yhi = float(max(yhi, np.nanpercentile(sf, 97.0)))
        if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
            pad = max(0.06 * (yhi - ylo), 0.02)
            ax.set_ylim(ylo - pad, yhi + pad)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def _plot_strong_line_three_models_on_ax(ax, p: dict) -> None:
    """Gaussian / Lorentzian / Voigt only (no core Gaussian or heavy smooth)."""
    vf = np.asarray(p.get("v_fine_full", p.get("v_fine", [])), float)
    vpf = np.asarray(p.get("v_kms_plot_full", p.get("v_kms_plot", p["v_kms"])), float)
    fpf = np.asarray(p.get("flux_plot_full", p.get("flux_plot", p["flux"])), float)
    ax.step(vpf, fpf, where="mid", color="0.25", lw=0.9, alpha=0.95, label="data (step)")
    fg = p.get("flux_gauss_full", p.get("flux_gauss"))
    if fg is not None and len(vf) == len(fg):
        ax.plot(vf, np.asarray(fg, float), "g-", lw=1.0, alpha=0.9, label="Gaussian")
    flr = p.get("flux_lorentz_full", p.get("flux_lorentz"))
    if flr is not None and len(vf) == len(flr):
        ax.plot(vf, np.asarray(flr, float), "b-", lw=1.0, alpha=0.85, label="Lorentzian")
    fv = p.get("flux_voigt_full", p.get("flux_voigt"))
    if fv is not None and len(vf) == len(fv):
        ax.plot(vf, np.asarray(fv, float), color="darkorange", lw=1.0, alpha=0.9, label="Voigt")
    rg, rl = float(p["rv_gauss_kms"]), float(p.get("rv_lorentz_kms", np.nan))
    rv = float(p["rv_voigt_kms"])
    if np.isfinite(rg):
        ax.axvline(rg, color="green", ls="--", lw=0.85)
    if np.isfinite(rl):
        ax.axvline(rl, color="blue", ls="--", lw=0.85)
    if np.isfinite(rv):
        ax.axvline(rv, color="darkorange", ls="--", lw=0.85)


def _strong_line_three_models_annotation(p: dict) -> str:
    trust_hw = float(p.get("rv_trust_half_width_kms", p.get("fit_half_width_kms", 150)))

    def _one(label: str, r: float, e: float) -> str | None:
        if np.isfinite(r) and np.isfinite(e):
            return f"{label}: {r:+.2f} ± {e:.2f} km/s"
        if np.isfinite(r):
            if abs(r) > trust_hw:
                return f"{label}: {r:+.2f} km/s (σ n/a, |RV|>{trust_hw:.0f})"
            return f"{label}: {r:+.2f} km/s (σ n/a)"
        return None

    lines_txt: list[str] = []
    for lab, r, e in (
        ("Gaussian", float(p["rv_gauss_kms"]), float(p["err_gauss_kms"])),
        ("Lorentz", float(p.get("rv_lorentz_kms", np.nan)), float(p.get("err_lorentz_kms", np.nan))),
        ("Voigt", float(p["rv_voigt_kms"]), float(p["err_voigt_kms"])),
    ):
        s = _one(lab, r, e)
        if s:
            lines_txt.append(s)
    cap = float(p.get("fit_velocity_cap_kms", np.nan))
    if np.isfinite(cap):
        lines_txt.append(f"(fits |v|≤{cap:.0f} km/s)")
    return "\n".join(lines_txt)


def plot_h_beta_three_model_epoch_stack(
    epochs: list[tuple[float, str, dict]],
    outpath: Path,
    *,
    title_stem: str = "",
    velocity_half_width_kms: float = 500.0,
) -> None:
    """
    One row per epoch: Hβ local-continuum profile with Gaussian / Lorentzian / Voigt fits and RV
    markers; bottom row: RV vs MJD for those three estimators.

    ``epochs`` entries are ``(mjd, label, panel_dict)`` where ``panel_dict`` is the return value of
    :func:`darkhunter_rv.rv_core.fit_balmer_line_all_methods` for Hβ.

    Spectrum panels share a common velocity x-axis (no per-panel titles). ``title_stem`` is kept
    for API compatibility and ignored.
    """
    _ = title_stem
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if not epochs:
        logger.warning("plot_h_beta_three_model_epoch_stack: no epochs; skip %s", outpath)
        return

    n = len(epochs)
    fig_h = 2.85 * n + 2.65
    fig, axes = plt.subplots(n + 1, 1, figsize=(10.0, fig_h), squeeze=False)
    ax_prof = axes.ravel()[:n]
    ax_rv = axes.ravel()[n]

    v_hw = float(velocity_half_width_kms)
    for i in range(1, n):
        ax_prof[i].sharex(ax_prof[0])
    ax_prof[0].set_xlim(-v_hw, v_hw)

    mjds = np.array([float(e[0]) for e in epochs], dtype=float)
    rg = np.array([float(e[2]["rv_gauss_kms"]) for e in epochs], dtype=float)
    eg = np.array([float(e[2]["err_gauss_kms"]) for e in epochs], dtype=float)
    rl = np.array([float(e[2].get("rv_lorentz_kms", np.nan)) for e in epochs], dtype=float)
    el = np.array([float(e[2].get("err_lorentz_kms", np.nan)) for e in epochs], dtype=float)
    rv = np.array([float(e[2]["rv_voigt_kms"]) for e in epochs], dtype=float)
    ev = np.array([float(e[2]["err_voigt_kms"]) for e in epochs], dtype=float)

    for i, (_mjd, _label, p) in enumerate(epochs):
        ax = ax_prof[i]
        _plot_strong_line_three_models_on_ax(ax, p)
        vpf = np.asarray(p.get("v_kms_plot_full", p.get("v_kms_plot", p["v_kms"])), float)
        fpf = np.asarray(p.get("flux_plot_full", p.get("flux_plot", p["flux"])), float)
        mz = (vpf >= -v_hw) & (vpf <= v_hw) & np.isfinite(fpf)
        if int(np.sum(mz)) >= 6:
            ylo, yhi = np.nanpercentile(fpf[mz], [4.0, 96.0])
            if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
                pad_y = 0.1 * (yhi - ylo)
                ax.set_ylim(ylo - pad_y, yhi + pad_y)
        ax.set_ylabel("Norm flux")
        ax.legend(fontsize=6.5, loc="upper right", ncol=2)
        ax.text(
            0.02,
            0.98,
            _strong_line_three_models_annotation(p),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.5,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
        )
        if i < n - 1:
            ax.tick_params(labelbottom=False)

    ax_prof[-1].set_xlabel("Velocity (km/s)")

    ax_rv.errorbar(
        mjds,
        rg,
        yerr=eg,
        fmt="o-",
        color="green",
        capsize=2,
        ms=3.5,
        lw=0.9,
        label="Gaussian",
    )
    ax_rv.errorbar(
        mjds,
        rl,
        yerr=el,
        fmt="s-",
        color="blue",
        capsize=2,
        ms=3.5,
        lw=0.9,
        label="Lorentzian",
    )
    ax_rv.errorbar(
        mjds,
        rv,
        yerr=ev,
        fmt="^-",
        color="darkorange",
        capsize=2,
        ms=3.5,
        lw=0.9,
        label="Voigt",
    )
    ax_rv.set_xlabel("MJD")
    ax_rv.set_ylabel("RV (km/s)")
    ax_rv.legend(fontsize=7, loc="best")
    ax_rv.grid(True, alpha=0.25)

    fig.tight_layout(h_pad=0.1, w_pad=0.4)
    fig.savefig(outpath, dpi=135)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def plot_rv_scatter_compare(
    x: np.ndarray,
    y: np.ndarray,
    outpath: Path,
    *,
    x_err: np.ndarray | None = None,
    y_err: np.ndarray | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "",
) -> None:
    """Scatter with optional error bars and a 1:1 reference line when ranges overlap."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    if x_err is not None and y_err is not None:
        xe = np.asarray(x_err, float)
        ye = np.asarray(y_err, float)
        ok &= np.isfinite(xe) & np.isfinite(ye)
        ax.errorbar(
            x[ok],
            y[ok],
            xerr=xe[ok],
            yerr=ye[ok],
            fmt="o",
            ms=4,
            capsize=2,
            color="0.2",
            elinewidth=0.6,
        )
    else:
        ax.scatter(x[ok], y[ok], s=22, color="0.2", zorder=3)
    if int(np.sum(ok)) >= 2:
        lo = float(np.nanmin([np.nanmin(x[ok]), np.nanmin(y[ok])]))
        hi = float(np.nanmax([np.nanmax(x[ok]), np.nanmax(y[ok])]))
        if hi > lo:
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.5, label="1:1")
            ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def plot_balmer_voigt_vs_h_beta_voigt(
    hb_v: np.ndarray,
    line_vel: dict[str, np.ndarray],
    outpath: Path,
    *,
    hb_err: np.ndarray | None = None,
    line_err: dict[str, np.ndarray] | None = None,
    title_stem: str = "",
) -> None:
    """
    One column per other Balmer line (e.g. Ha, Hg, Hd): Voigt RV vs Hβ Voigt RV per epoch.
    ``line_vel`` maps line name -> array (same length as ``hb_v``).
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    names = [k for k in ("Ha", "Hg", "Hd") if k in line_vel]
    if not names:
        logger.warning("plot_balmer_voigt_vs_h_beta_voigt: no lines; skip %s", outpath)
        return
    hb_v = np.asarray(hb_v, float)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    axr = axes.ravel()
    he = np.asarray(hb_err, float) if hb_err is not None else None
    for j, name in enumerate(names):
        ax = axr[j]
        y = np.asarray(line_vel[name], float)
        ye = None
        if line_err is not None and name in line_err:
            ye = np.asarray(line_err[name], float)
        ok = np.isfinite(hb_v) & np.isfinite(y)
        if he is not None and ye is not None:
            ok &= np.isfinite(he) & np.isfinite(ye)
            ax.errorbar(
                hb_v[ok],
                y[ok],
                xerr=he[ok],
                yerr=ye[ok],
                fmt="o",
                ms=4,
                capsize=2,
                color="0.2",
                elinewidth=0.6,
            )
        else:
            ax.scatter(hb_v[ok], y[ok], s=24, color="0.2")
        if int(np.sum(ok)) >= 2:
            lo = float(np.nanmin([np.nanmin(hb_v[ok]), np.nanmin(y[ok])]))
            hi = float(np.nanmax([np.nanmax(hb_v[ok]), np.nanmax(y[ok])]))
            if hi > lo:
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.85, alpha=0.45)
        ax.set_xlabel("Hβ Voigt RV (km/s)")
        ax.set_ylabel(f"{name} Voigt RV (km/s)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        (title_stem + "  " if title_stem else "") + "Balmer Voigt vs Hβ Voigt",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def _hb_model_flux_on_order_grid(
    w_model: np.ndarray,
    f_model: np.ndarray,
    nw: np.ndarray,
    nf: np.ndarray,
    w_dat: np.ndarray,
    f_dat: np.ndarray,
) -> np.ndarray | None:
    """Rescale a model curve in λ onto the echelle order norm using (order norm)/(local-cont. data)."""
    w_model = np.asarray(w_model, float)
    f_model = np.asarray(f_model, float)
    if w_model.size < 5 or f_model.shape != w_model.shape:
        return None
    w_dat = np.asarray(w_dat, float)
    f_dat = np.asarray(f_dat, float)
    if w_dat.size < 5 or f_dat.shape != w_dat.shape:
        return None
    o = np.argsort(w_dat)
    w_ds, fp_s = w_dat[o], f_dat[o]
    nf_s = np.interp(w_ds, nw, nf, left=np.nan, right=np.nan)
    den = np.where(np.abs(fp_s) > 0.05, fp_s, np.nan)
    ratio = np.divide(
        nf_s,
        den,
        out=np.ones_like(nf_s, dtype=float),
        where=np.isfinite(nf_s) & np.isfinite(den),
    )
    ratio = np.clip(np.nan_to_num(ratio, nan=1.0, posinf=6.0, neginf=0.15), 0.12, 6.0)
    ratio_f = np.interp(w_model, w_ds, ratio, left=np.nan, right=np.nan)
    med_r = float(np.nanmedian(ratio_f[np.isfinite(ratio_f)]))
    if not np.isfinite(med_r):
        med_r = 1.0
    ratio_f = np.where(np.isfinite(ratio_f), ratio_f, med_r)
    return f_model * ratio_f


def _hb_lab_wavelength_from_rv_kms(rv_kms: float, rest: float) -> float:
    return float(rest * (1.0 + float(rv_kms) / config.C_KMS))


def _hb_tight_ylim_local_cont(ax, w_plot: np.ndarray, flux_series: list[np.ndarray]) -> None:
    """Tight y limits from percentiles of flux in the Hβ window (matches velocity diagnostic style)."""
    w_plot = np.asarray(w_plot, float)
    ok_w = np.isfinite(w_plot)
    if int(np.sum(ok_w)) < 4:
        return
    seqs: list[np.ndarray] = []
    for y in flux_series:
        y = np.asarray(y, float)
        if y.size == w_plot.size:
            m = ok_w & np.isfinite(y)
            if int(np.sum(m)) >= 6:
                seqs.append(y[m])
        else:
            yy = y[np.isfinite(y)]
            if yy.size >= 4:
                seqs.append(yy)
    if not seqs:
        return
    y_all = np.concatenate(seqs)
    ylo, yhi = np.nanpercentile(y_all, [4.0, 96.0])
    if not np.isfinite(ylo) or not np.isfinite(yhi) or yhi <= ylo:
        return
    pad = max(0.06 * (yhi - ylo), 0.02)
    ax.set_ylim(ylo - pad, yhi + pad)


def plot_h_beta_order_method_overlay(
    nw_obs: np.ndarray,
    nf_obs: np.ndarray,
    mask_wave: np.ndarray,
    mask_strength: np.ndarray,
    rv_mask_kms: float,
    err_mask_kms: float,
    tpl_wave: np.ndarray,
    tpl_flux_norm: np.ndarray,
    rv_tpl_kms: float,
    err_tpl_kms: float,
    hb_fit: dict,
    outpath: Path,
    *,
    title_stem: str = "",
    resolving_power: float | None = None,
    hb_measure: dict | None = None,
) -> None:
    """
    Hβ echelle order: mask, shifted/scaled template, and joint Voigt+Lorentz vs data.

    With ``hb_measure`` (valid ``v_kms_plot`` / ``flux_plot``), uses a **single** local-continuum
    flux axis: data drawn first, models on top, tight y-limits, vertical lines at RV centers, and a
    top axis in velocity (km/s). Without that bundle, uses order-normalized flux on the primary axis
    and optional twinx for partial local-cont. data.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    nw = np.asarray(nw_obs, float)
    nf = np.asarray(nf_obs, float)
    rest = float(hb_fit.get("rest", rv_core.HB_REST_A))
    v_obs = config.C_KMS * (nw / rest - 1.0)

    v_pf = (
        np.asarray(hb_measure.get("v_kms_plot", []), float)
        if hb_measure is not None
        else np.array([], dtype=float)
    )
    f_pf = (
        np.asarray(hb_measure.get("flux_plot", []), float)
        if hb_measure is not None
        else np.array([], dtype=float)
    )
    unified_local = hb_measure is not None and v_pf.size > 5 and f_pf.shape == v_pf.shape

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax_hb_local = None
    measure_voigt_plotted = False
    measure_lorentz_plotted = False
    pseudo: np.ndarray | None = None
    tf_plot_scaled: np.ndarray | None = None

    mw = np.asarray(mask_wave, float)
    ms = np.asarray(mask_strength, float)
    tw = np.asarray(tpl_wave, float)
    tf = np.asarray(tpl_flux_norm, float)
    o_t = np.argsort(tw)
    tw, tf = tw[o_t], tf[o_t]

    if unified_local:
        w_loc = rest * (1.0 + v_pf / config.C_KMS)
        o_loc = np.argsort(w_loc)
        ws, fs = w_loc[o_loc], f_pf[o_loc]
        ax.step(
            ws,
            fs,
            where="mid",
            color="0.35",
            lw=1.0,
            alpha=0.88,
            label="data (local cont.)",
            zorder=2,
        )

        if np.isfinite(rv_mask_kms) and len(mw) > 5 and len(ms) == len(mw):
            beta_m = 1.0 + float(rv_mask_kms) / config.C_KMS
            pseudo = np.ones_like(nw, dtype=float)
            sig_a = max(0.05, float(np.nanmedian(np.abs(np.diff(np.sort(nw))))) * 1.8)
            smax = float(np.nanpercentile(np.abs(ms), 90)) + 1e-9
            for w0, s in zip(mw, ms):
                w_shift = float(w0) * beta_m
                if w_shift < float(np.min(nw)) or w_shift > float(np.max(nw)):
                    continue
                amp = min(0.45, 0.42 * (abs(float(s)) / smax) ** 0.5)
                pseudo -= amp * np.exp(-0.5 * ((nw - w_shift) / sig_a) ** 2)
            pseudo = np.clip(pseudo, 0.12, 1.45)
            ax.plot(
                nw,
                pseudo,
                "b-",
                lw=0.9,
                alpha=0.88,
                label="mask (shifted, schematic)",
                zorder=4,
            )

        if np.isfinite(rv_tpl_kms) and len(tw) > 30:
            beta_t = 1.0 + float(rv_tpl_kms) / config.C_KMS
            w_shifted = tw * beta_t
            m = np.isfinite(w_shifted) & np.isfinite(tf)
            w_shifted, tf_s = w_shifted[m], tf[m]
            if len(w_shifted) > 20:
                tf_i = np.interp(nw, w_shifted, tf_s, left=np.nan, right=np.nan)
                if resolving_power is not None and resolving_power > 1.0:
                    tf_i = rv_core.degrade_template_flux_lsf(nw, tf_i, float(resolving_power))
                f_on_nw = np.interp(nw, ws, fs, left=np.nan, right=np.nan)
                tf_plot_scaled = _hb_scale_template_to_local_cont(v_obs, f_on_nw, tf_i)
                ax.plot(
                    nw,
                    tf_plot_scaled,
                    "r-",
                    lw=1.05,
                    alpha=0.92,
                    label="template (shifted, scaled)",
                    zorder=5,
                )

        wf_m = np.asarray(hb_measure.get("voigt_wave_fine", []), float)
        fv_m = hb_measure.get("voigt_model_fine")
        fl_m = hb_measure.get("lorentz_model_fine")
        p_joint = hb_measure.get("hb_joint_fit_params")
        y_m = None
        if p_joint is not None and len(p_joint) == 8 and wf_m.size > 5:
            y_m = rv_core.h_beta_joint_line_model(wf_m, p_joint, rest=rest)
        elif fv_m is not None and wf_m.size > 5 and np.asarray(fv_m, float).shape == wf_m.shape:
            y_m = np.asarray(fv_m, dtype=float)
        if y_m is not None and y_m.shape == wf_m.shape:
            ax.plot(
                wf_m,
                y_m,
                color="goldenrod",
                lw=1.2,
                alpha=0.95,
                label="Hβ Voigt+Lorentz (joint, exact fit)",
                zorder=6,
            )
            measure_voigt_plotted = True
            if fl_m is None:
                measure_lorentz_plotted = True
        if (
            fl_m is not None
            and wf_m.size > 5
            and np.asarray(fl_m, float).shape == wf_m.shape
        ):
            ax.plot(
                wf_m,
                np.asarray(fl_m, dtype=float),
                color="steelblue",
                lw=1.05,
                alpha=0.9,
                label="Hβ Lorentz (exact fit)",
                zorder=6,
            )
            measure_lorentz_plotted = True

        rv_v_line = float(hb_measure.get("rv_voigt_kms", np.nan))
        if p_joint is not None and len(p_joint) >= 8:
            rv_v_line = float(config.C_KMS * (float(p_joint[2]) - rest) / rest)
        rv_cc_line = float(hb_measure.get("rv_template_ccf_kms", np.nan))

        for rv, color, ls in (
            (float(rv_mask_kms), "tab:blue", "-"),
            (float(rv_tpl_kms), "tab:red", "-"),
            (rv_v_line, "goldenrod", "-"),
        ):
            if np.isfinite(rv):
                ax.axvline(
                    _hb_lab_wavelength_from_rv_kms(rv, rest),
                    color=color,
                    ls=ls,
                    lw=1.25,
                    alpha=0.92,
                    zorder=9,
                )
        if np.isfinite(rv_cc_line) and (
            not np.isfinite(rv_v_line) or abs(rv_cc_line - rv_v_line) > 0.6
        ):
            ax.axvline(
                _hb_lab_wavelength_from_rv_kms(rv_cc_line, rest),
                color="0.25",
                ls="--",
                lw=1.1,
                alpha=0.85,
                zorder=9,
            )

        flux_for_ylim: list[np.ndarray] = [fs]
        if y_m is not None:
            flux_for_ylim.append(np.asarray(y_m, float))
        if tf_plot_scaled is not None:
            flux_for_ylim.append(tf_plot_scaled)
        if pseudo is not None:
            inb = (nw >= float(ws[0])) & (nw <= float(ws[-1]))
            if int(np.sum(inb)) > 8:
                flux_for_ylim.append(pseudo[inb])
        _hb_tight_ylim_local_cont(ax, ws, flux_for_ylim)

        def _lam_to_v(lam: np.ndarray) -> np.ndarray:
            return config.C_KMS * (np.asarray(lam, dtype=float) / rest - 1.0)

        def _v_to_lam(v: np.ndarray) -> np.ndarray:
            return rest * (1.0 + np.asarray(v, dtype=float) / config.C_KMS)

        secax = ax.secondary_xaxis("top", functions=(_lam_to_v, _v_to_lam))
        secax.set_xlabel("Velocity (km/s)", fontsize=9)
        ax.set_ylabel("Norm flux (local cont.)")
    else:
        ax.step(nw, nf, where="mid", color="0.35", lw=0.75, alpha=0.9, label="obs (order norm)", zorder=2)

        if np.isfinite(rv_mask_kms) and len(mw) > 5 and len(ms) == len(mw):
            beta_m = 1.0 + float(rv_mask_kms) / config.C_KMS
            pseudo = np.ones_like(nw, dtype=float)
            sig_a = max(0.05, float(np.nanmedian(np.abs(np.diff(np.sort(nw))))) * 1.8)
            smax = float(np.nanpercentile(np.abs(ms), 90)) + 1e-9
            for w0, s in zip(mw, ms):
                w_shift = float(w0) * beta_m
                if w_shift < float(np.min(nw)) or w_shift > float(np.max(nw)):
                    continue
                amp = min(0.45, 0.42 * (abs(float(s)) / smax) ** 0.5)
                pseudo -= amp * np.exp(-0.5 * ((nw - w_shift) / sig_a) ** 2)
            pseudo = np.clip(pseudo, 0.12, 1.45)
            ax.plot(nw, pseudo, "b-", lw=0.85, alpha=0.85, label="mask (shifted, schematic)", zorder=3)

        if np.isfinite(rv_tpl_kms) and len(tw) > 30:
            beta_t = 1.0 + float(rv_tpl_kms) / config.C_KMS
            w_shifted = tw * beta_t
            m = np.isfinite(w_shifted) & np.isfinite(tf)
            w_shifted, tf_s = w_shifted[m], tf[m]
            if len(w_shifted) > 20:
                tf_i = np.interp(
                    nw,
                    w_shifted,
                    tf_s,
                    left=np.nan,
                    right=np.nan,
                )
                if resolving_power is not None and resolving_power > 1.0:
                    tf_i = rv_core.degrade_template_flux_lsf(nw, tf_i, float(resolving_power))
                wing = np.abs(v_obs) > 130.0
                okw = wing & np.isfinite(nf) & np.isfinite(tf_i)
                if int(np.sum(okw)) >= 12:
                    scale = float(np.nanmedian(nf[okw]) / (np.nanmedian(tf_i[okw]) + 1e-9))
                else:
                    scale = float(np.nanmedian(nf) / (np.nanmedian(tf_i) + 1e-9))
                tf_plot = np.clip(tf_i * scale, 0.05, 2.5)
                ax.plot(nw, tf_plot, "r-", lw=0.85, alpha=0.88, label="template (shifted, scaled)", zorder=4)

    if not unified_local and hb_measure is not None:
        wf_m = np.asarray(hb_measure.get("voigt_wave_fine", []), float)
        fv_m = hb_measure.get("voigt_model_fine")
        fl_m = hb_measure.get("lorentz_model_fine")
        p_joint = hb_measure.get("hb_joint_fit_params")
        v_pf2 = np.asarray(hb_measure.get("v_kms_plot", []), float)
        f_pf2 = np.asarray(hb_measure.get("flux_plot", []), float)
        if v_pf2.size > 5 and f_pf2.shape == v_pf2.shape:
            ax_hb_local = ax.twinx()
            w_loc = rest * (1.0 + v_pf2 / config.C_KMS)
            o_loc = np.argsort(w_loc)
            ax_hb_local.step(
                w_loc[o_loc],
                f_pf2[o_loc],
                where="mid",
                color="k",
                lw=0.95,
                alpha=0.95,
                label="obs (Hβ local linear cont.)",
                zorder=2,
            )
            ax_hb_local.set_ylabel("Local-cont. norm flux", color="0.15", fontsize=9)
            ax_hb_local.tick_params(axis="y", labelcolor="0.15")
            y_m2 = None
            if p_joint is not None and len(p_joint) == 8 and wf_m.size > 5:
                y_m2 = rv_core.h_beta_joint_line_model(wf_m, p_joint, rest=rest)
            elif fv_m is not None and wf_m.size > 5 and np.asarray(fv_m, float).shape == wf_m.shape:
                y_m2 = np.asarray(fv_m, dtype=float)
            if y_m2 is not None and y_m2.shape == wf_m.shape:
                ax_hb_local.plot(
                    wf_m,
                    y_m2,
                    color="goldenrod",
                    lw=1.1,
                    alpha=0.95,
                    label="Hβ Voigt+Lorentz (joint, exact fit)",
                    zorder=6,
                )
                measure_voigt_plotted = True
                if fl_m is None:
                    measure_lorentz_plotted = True
            if (
                fl_m is not None
                and wf_m.size > 5
                and np.asarray(fl_m, float).shape == wf_m.shape
            ):
                ax_hb_local.plot(
                    wf_m,
                    np.asarray(fl_m, dtype=float),
                    color="steelblue",
                    lw=1.0,
                    alpha=0.9,
                    label="Hβ Lorentz (exact fit)",
                    zorder=6,
                )
                measure_lorentz_plotted = True

    if not measure_voigt_plotted or not measure_lorentz_plotted:
        vf = np.asarray(hb_fit.get("v_fine_full", []), float)
        fv_raw = hb_fit.get("flux_voigt_full")
        fl_raw = hb_fit.get("flux_lorentz_full")
        vpf = np.asarray(hb_fit.get("v_kms_plot_full", []), float)
        fpf = np.asarray(hb_fit.get("flux_plot_full", []), float)
        if vpf.size > 5 and fpf.size == vpf.size:
            w_dat = rest * (1.0 + vpf / config.C_KMS)
            o = np.argsort(w_dat)
            w_ds_fb, fp_s_fb = w_dat[o], fpf[o]
            if not measure_voigt_plotted and fv_raw is not None and vf.size > 5:
                fv_a = np.asarray(fv_raw, float)
                if fv_a.shape == vf.shape:
                    wv = rest * (1.0 + vf / config.C_KMS)
                    fv_draw = _hb_model_flux_on_order_grid(wv, fv_a, nw, nf, w_ds_fb, fp_s_fb)
                    if fv_draw is not None:
                        ax.plot(
                            wv,
                            fv_draw,
                            color="darkorange",
                            lw=1.0,
                            alpha=0.95,
                            label="Hβ Voigt → order norm",
                            zorder=5,
                        )
            if not measure_lorentz_plotted and fl_raw is not None and vf.size > 5:
                fl_a = np.asarray(fl_raw, float)
                if fl_a.shape == vf.shape:
                    wv = rest * (1.0 + vf / config.C_KMS)
                    fl_draw = _hb_model_flux_on_order_grid(wv, fl_a, nw, nf, w_ds_fb, fp_s_fb)
                    if fl_draw is not None:
                        ax.plot(
                            wv,
                            fl_draw,
                            color="steelblue",
                            lw=1.0,
                            alpha=0.9,
                            label="Hβ Lorentz → order norm",
                            zorder=5,
                        )

    rv_v = ev_v = rv_l = ev_l = rv_s = es_s = rv_cc = rb = float("nan")
    method_used = ""
    joint_hb_line = False
    if hb_measure is not None:
        rv_v = float(hb_measure.get("rv_voigt_kms", np.nan))
        _pj = hb_measure.get("hb_joint_fit_params")
        if _pj is not None and len(_pj) >= 8:
            rv_v = float(config.C_KMS * (float(_pj[2]) - rest) / rest)
        ev_v = float(hb_measure.get("err_voigt_kms", np.nan))
        rv_l = float(hb_measure.get("rv_lorentz_kms", np.nan))
        ev_l = float(hb_measure.get("err_lorentz_kms", np.nan))
        joint_hb_line = hb_measure.get("lorentz_model_fine") is None
        rv_s = float(hb_measure.get("rv_smoothed_min_kms", np.nan))
        es_s = float(hb_measure.get("err_smoothed_kms", np.nan))
        rv_cc = float(hb_measure.get("rv_template_ccf_kms", np.nan))
        rb = float(hb_measure.get("rv_best_kms", np.nan))
        method_used = str(hb_measure.get("method_used", "") or "")
    else:
        rv_v = float(hb_fit.get("rv_voigt_kms", np.nan))
        ev_v = float(hb_fit.get("err_voigt_kms", np.nan))
        rv_l = float(hb_fit.get("rv_lorentz_kms", np.nan))
        ev_l = float(hb_fit.get("err_lorentz_kms", np.nan))
        rv_s = float(hb_fit.get("rv_smooth_min_kms", np.nan))
        es_s = float(hb_fit.get("err_smooth_min_kms", np.nan))

    def _ann_rv(label: str, r: float, e: float) -> str | None:
        if not np.isfinite(r):
            return None
        if np.isfinite(e):
            return f"{label}: {r:+.2f} ± {e:.2f} km/s"
        return f"{label}: {r:+.2f} km/s"

    lines_ann: list[str] = []
    for s in (
        _ann_rv("mask CCF", float(rv_mask_kms), float(err_mask_kms)),
        _ann_rv("template FFT", float(rv_tpl_kms), float(err_tpl_kms)),
    ):
        if s:
            lines_ann.append(s)
    if joint_hb_line:
        t = _ann_rv("Hβ Voigt+Lorentz", rv_v, ev_v)
        if t:
            lines_ann.append(t)
    else:
        for s in (_ann_rv("Hβ Voigt", rv_v, ev_v), _ann_rv("Hβ Lorentz", rv_l, ev_l)):
            if s:
                lines_ann.append(s)
    tsm = _ann_rv("smoothed min", rv_s, es_s)
    if tsm:
        lines_ann.append(tsm)
    if hb_measure is not None and np.isfinite(rv_cc):
        pk = hb_measure.get("template_ccf_peak")
        if pk is not None and np.isfinite(float(pk)):
            lines_ann.append(f"Hβ template CCF: {rv_cc:+.2f} km/s (r={float(pk):.3f})")
        else:
            t = _ann_rv("Hβ template CCF", rv_cc, float("nan"))
            if t:
                lines_ann.append(t)
    if hb_measure is not None and np.isfinite(rb) and method_used:
        lines_ann.append(f"best ({method_used}): {rb:+.2f} km/s")

    ax.text(
        0.02,
        0.02,
        "\n".join(lines_ann),
        transform=ax.transAxes,
        fontsize=7.0,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
    )
    ax.set_xlabel("Wavelength (Å)")
    if not unified_local:
        ax.set_ylabel("Normalized flux (order)")
    h1, l1 = ax.get_legend_handles_labels()
    if ax_hb_local is not None:
        h2, l2 = ax_hb_local.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    else:
        ax.legend(h1, l1, fontsize=7, loc="upper right")
    ttl = (
        "Hβ — mask / template / Voigt+Lorentz (local cont.)"
        if unified_local
        else "Hβ order — mask / template / Voigt+Lorentz (joint)"
    )
    ax.set_title(f"{title_stem}  {ttl}" if title_stem else ttl)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    logger.debug("saved %s", outpath)
