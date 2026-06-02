"""Keplerian RV fit figures (data-only, multi-model, residuals)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.ticker import AutoMinorLocator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fit_apf_rv_keplerian import RVPoint


def _fitmod():
    import fit_apf_rv_keplerian as m

    return m

from darkhunter_rv.rv_point_filters import rv_value_is_valid

FIT_VARIANT_ORDER = ("free", "fix_period", "fix_ecc", "fix_period_ecc")
FIT_VARIANT_STYLE: Dict[str, Tuple[str, str]] = {
    "free": ("tab:blue", "-"),
    "fix_period": ("tab:orange", "--"),
    "fix_ecc": ("tab:green", "-."),
    "fix_period_ecc": ("tab:red", ":"),
}
FIT_VARIANT_LABEL = {
    "free": "RV only",
    "fix_period": "P fixed (Gaia)",
    "fix_ecc": "e fixed (Gaia)",
    "fix_period_ecc": "P & e fixed (Gaia)",
}

TEL_MARKER = {"APF": "o", "KPF": "s", "GHOST": "p", "MAROON-X": "x"}


def filter_valid_points(points: Sequence["RVPoint"]) -> List["RVPoint"]:
    return [p for p in points if rv_value_is_valid(p.rv)]


def our_telescope_points(points: Sequence["RVPoint"]) -> List["RVPoint"]:
    return [p for p in points if not p.is_literature]


def _time_window(
    t: np.ndarray,
    report: dict,
    *,
    include_future: bool,
) -> Tuple[float, float, float]:
    t_ref = float(report["t_ref_mjd"])
    now_mjd = float(report.get("now_mjd", Time.now().mjd))
    t_span = float(np.ptp(t) + 1.0) if t.size else 1.0
    t_start = float(np.min(t) - 0.02 * t_span) if t.size else now_mjd - 30.0

    t_next_max = float(report.get("next_rv_max_mjd", np.nan))
    t_next_min = float(report.get("next_rv_min_mjd", np.nan))
    next_event = t_next_min
    if np.isfinite(t_next_max) and np.isfinite(t_next_min):
        next_event = t_next_max if t_next_max <= t_next_min else t_next_min

    t_end_candidates = [float(np.max(t) + 0.02 * t_span) if t.size else now_mjd + 30.0]
    if include_future:
        t_end_candidates.append(now_mjd)
        if np.isfinite(next_event):
            t_end_candidates.append(next_event)
        obs_win = report.get("observability_window")
        if isinstance(obs_win, dict):
            try:
                obs_end = float(Time(obs_win["end_date"], format="iso", scale="utc").mjd) + 1.0
                t_end_candidates.append(obs_end)
            except Exception:
                pass
    t_end = float(max(t_end_candidates))
    return t_start, t_end, now_mjd


def _observability_span(report: dict) -> Tuple[Optional[float], Optional[float], Optional[dict]]:
    obs_win = report.get("observability_window")
    if not isinstance(obs_win, dict):
        return None, None, None
    try:
        obs_start = float(Time(obs_win["start_date"], format="iso", scale="utc").mjd)
        obs_end = float(Time(obs_win["end_date"], format="iso", scale="utc").mjd) + 1.0
        return obs_start, obs_end, obs_win
    except Exception:
        return None, None, None


def _shade_apf_window(ax, t_start: float, t_end: float, report: dict) -> None:
    obs_start, obs_end, obs_win = _observability_span(report)
    if obs_start is None or obs_end is None or obs_win is None:
        return
    left = max(t_start, obs_start)
    right = min(t_end, obs_end)
    if right <= left:
        return
    ax.axvspan(left, right, color="tab:blue", alpha=0.12, zorder=0)
    ax.text(
        0.985,
        0.02,
        f"APF window {obs_win['start_date']} to {obs_win['end_date']}",
        transform=ax.transAxes,
        fontsize=8.5,
        color="tab:blue",
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="tab:blue", alpha=0.85, boxstyle="round,pad=0.2"),
    )


def _mark_today(ax, now_mjd: float, y_top_in: float) -> None:
    ax.axvline(now_mjd, color="0.35", ls="--", lw=1.2, alpha=0.9)
    ax.text(
        now_mjd,
        y_top_in,
        "Today",
        fontsize=9,
        color="0.25",
        ha="right",
        va="top",
        bbox=dict(facecolor="0.93", edgecolor="0.5", alpha=0.95, boxstyle="round,pad=0.15"),
    )


def _plot_points(ax, points: Sequence["RVPoint"], *, include_literature: bool) -> bool:
    plotted = False
    t = np.array([p.mjd for p in points], dtype=float)
    y = np.array([p.rv for p in points], dtype=float)
    yerr = np.array([p.rms for p in points], dtype=float)

    for tel, marker in TEL_MARKER.items():
        idx = [
            i
            for i, p in enumerate(points)
            if (not p.is_literature and str(p.telescope).upper() == tel)
        ]
        if not idx:
            continue
        ax.errorbar(
            t[idx],
            y[idx],
            yerr=yerr[idx],
            fmt=marker,
            ms=5,
            lw=1,
            capsize=2,
            color="black",
            ecolor="black",
            mec="black",
            mfc=("black" if marker != "x" else "none"),
            label=tel,
        )
        plotted = True

    if include_literature:
        idx_lit = [i for i, p in enumerate(points) if p.is_literature]
        if idx_lit:
            ax.errorbar(
                t[idx_lit],
                y[idx_lit],
                yerr=yerr[idx_lit],
                fmt="D",
                ms=4.8,
                lw=1,
                capsize=2,
                color="0.35",
                ecolor="0.35",
                mec="0.35",
                mfc="0.35",
                label="Literature",
            )
            plotted = True

    if not plotted and len(points):
        ax.errorbar(t, y, yerr=yerr, fmt="o", ms=5, lw=1, capsize=2, color="black")
        plotted = True
    return plotted


def _y_limits_data_and_models(
    t: np.ndarray,
    y: np.ndarray,
    model_curves: Sequence[np.ndarray],
) -> Tuple[float, float]:
    y_low = float(np.min(y))
    y_high = float(np.max(y))
    for curve in model_curves:
        if curve.size:
            y_low = min(y_low, float(np.min(curve)))
            y_high = max(y_high, float(np.max(curve)))
    y_pad = max(1.0, 0.08 * (y_high - y_low if y_high > y_low else 1.0))
    return y_low - y_pad, y_high + y_pad


def _variant_annotation(rep: dict) -> str:
    m = _fitmod()
    p = int(round(float(rep["P_days"])))
    e = float(rep["e"])
    fm = float(
        rep.get("mass_function_msun", m.mass_function_msun(rep["P_days"], rep["K_kms"], rep["e"]))
    )
    return f"P={p} d, e={e:.3f}, f(M)={fm:.4f} M☉"


def plot_rv_data_only(
    summary_path: Path,
    points: Sequence["RVPoint"],
    report: dict,
    out_png: Path,
    *,
    title_prefix: str = "RV data",
) -> None:
    """Our telescope epochs only; Today + APF window markers."""
    pts = our_telescope_points(points)
    if len(pts) < 2:
        raise ValueError("need at least 2 our-telescope points for data plot")

    t = np.array([p.mjd for p in pts], dtype=float)
    y = np.array([p.rv for p in pts], dtype=float)
    t_start, t_end, now_mjd = _time_window(t, report, include_future=True)

    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    _shade_apf_window(ax, t_start, t_end, report)
    _plot_points(ax, pts, include_literature=False)

    y_lim = _y_limits_data_and_models(t, y, [])
    ax.set_ylim(*y_lim)
    y_top_in = y_lim[1] - 0.02 * (y_lim[1] - y_lim[0])
    _mark_today(ax, now_mjd, y_top_in)

    sid = _fitmod().parse_object_id_from_summary(summary_path) or summary_path.stem.replace("_summary", "")
    ax.set_xlabel("MJD")
    ax.set_ylabel("RV (km/s)")
    ax.set_title(f"{title_prefix}: Gaia DR3 {sid}")
    ax.set_xlim(t_start, t_end)
    ax.grid(alpha=0.25)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="major", direction="in", length=7, width=1.1, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=3.5, width=0.9, top=True, right=True)
    ax.legend(loc="best", fontsize=8.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_multi_fit(
    summary_path: Path,
    points: Sequence["RVPoint"],
    fit_variants: Dict[str, Tuple[np.ndarray, dict]],
    report: dict,
    out_png: Path,
    *,
    m1_msun: Optional[float] = None,
) -> None:
    """All epochs (incl. literature) with up to four Keplerian models."""
    if len(points) < 2:
        raise ValueError("need at least 2 points")

    t = np.array([p.mjd for p in points], dtype=float)
    y = np.array([p.rv for p in points], dtype=float)
    t_ref = float(report["t_ref_mjd"])
    t_start, t_end, now_mjd = _time_window(t, report, include_future=True)
    t_dense = np.linspace(t_start, t_end, 2000)

    curves: List[np.ndarray] = []
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    _shade_apf_window(ax, t_start, t_end, report)
    _plot_points(ax, points, include_literature=True)

    for key in FIT_VARIANT_ORDER:
        if key not in fit_variants:
            continue
        params, vrep = fit_variants[key]
        y_dense = _fitmod().rv_model(params, t_dense, t_ref)
        curves.append(y_dense)
        color, ls = FIT_VARIANT_STYLE[key]
        label = f"{FIT_VARIANT_LABEL[key]} ({_variant_annotation(vrep)})"
        ax.plot(t_dense, y_dense, ls=ls, lw=1.8, color=color, alpha=0.92, label=label)

    y_lim = _y_limits_data_and_models(t, y, curves)
    ax.set_ylim(*y_lim)
    y_top_in = y_lim[1] - 0.02 * (y_lim[1] - y_lim[0])
    _mark_today(ax, now_mjd, y_top_in)

    # Nearest upcoming extremum from primary (free) fit if present.
    primary = fit_variants.get("free")
    if primary is not None:
        params0, rep0 = primary
        p_days = float(rep0["P_days"])
        t_next_max, t_next_min = _fitmod().next_extrema_after(params0, t_ref, p_days, now_mjd)
        next_event = t_next_max if (t_next_max is not None and (t_next_min is None or t_next_max <= t_next_min)) else t_next_min
        if next_event is not None:
            ax.axvline(next_event, color="tab:red", ls="--", lw=1.2, alpha=0.9)

    sid = _fitmod().parse_object_id_from_summary(summary_path) or summary_path.stem.replace("_summary", "")
    ax.set_xlabel("MJD")
    ax.set_ylabel("RV (km/s)")
    ax.set_title(f"Keplerian fits: Gaia DR3 {sid}")
    ax.set_xlim(t_start, t_end)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=7.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if m1_msun is not None and primary is not None:
        _rep0 = primary[1]
        fm = float(_rep0.get("mass_function_msun", 0.0))
        m2sini = _fitmod().solve_m2sini_msun(fm, float(m1_msun)) if fm > 0 else None
        if m2sini is not None:
            ax.text(
                0.015,
                0.98,
                f"M₁={m1_msun:.4f} M☉   M₂ sin i={m2sini:.4f} M☉",
                transform=ax.transAxes,
                fontsize=9.0,
                va="top",
                ha="left",
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.95, boxstyle="round,pad=0.2"),
            )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _residual_ylim(
    y: np.ndarray,
    yerr: np.ndarray,
    residuals: Sequence[np.ndarray],
) -> Tuple[float, float]:
    scales = [float(np.nanmax(np.abs(y)))]
    if yerr.size:
        scales.append(float(np.nanmax(yerr[np.isfinite(yerr)])))
    for r in residuals:
        if r.size:
            scales.append(float(np.nanmax(np.abs(r[np.isfinite(r)]))))
    spread = max(scales) if scales else 1.0
    half = min(5.0, max(0.4, 1.6 * spread))
    return -half, half


def plot_fit_residuals(
    summary_path: Path,
    points: Sequence["RVPoint"],
    fit_variants: Dict[str, Tuple[np.ndarray, dict]],
    report: dict,
    out_png: Path,
) -> None:
    if "free" not in fit_variants:
        raise ValueError("residual plot requires a free RV fit")
    if len(points) < 2:
        raise ValueError("need at least 2 points")

    t = np.array([p.mjd for p in points], dtype=float)
    y = np.array([p.rv for p in points], dtype=float)
    yerr = np.array([p.rms for p in points], dtype=float)
    t_ref = float(report["t_ref_mjd"])
    params_free, rep_free = fit_variants["free"]
    t_start, t_end, now_mjd = _time_window(t, report, include_future=True)
    t_dense = np.linspace(t_start, t_end, 2000)

    rv_model = _fitmod().rv_model
    model_free_obs = rv_model(params_free, t, t_ref)
    model_free_dense = rv_model(params_free, t_dense, t_ref)
    resid_map: Dict[str, np.ndarray] = {}
    dense_map: Dict[str, np.ndarray] = {}
    for key in FIT_VARIANT_ORDER:
        if key not in fit_variants:
            continue
        params, _vrep = fit_variants[key]
        model_obs = rv_model(params, t, t_ref)
        resid_map[key] = y - model_obs
        dense_map[key] = rv_model(params, t_dense, t_ref) - model_free_dense

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10.8, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08}
    )
    _shade_apf_window(ax_top, t_start, t_end, report)
    _plot_points(ax_top, points, include_literature=True)
    ax_top.plot(t_dense, model_free_dense, "-", color=FIT_VARIANT_STYLE["free"][0], lw=2.0, label=FIT_VARIANT_LABEL["free"])
    y_lim = _y_limits_data_and_models(t, y, [model_free_dense])
    ax_top.set_ylim(*y_lim)
    y_top_in = y_lim[1] - 0.02 * (y_lim[1] - y_lim[0])
    _mark_today(ax_top, now_mjd, y_top_in)
    sid = _fitmod().parse_object_id_from_summary(summary_path) or summary_path.stem.replace("_summary", "")
    ax_top.set_ylabel("RV (km/s)")
    ax_top.set_title(f"Keplerian fits + residuals: Gaia DR3 {sid}")
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="best", fontsize=8)

    ax_bot.axhline(0.0, color="0.4", lw=1.0)
    ax_bot.errorbar(
        t,
        y - model_free_obs,
        yerr=yerr,
        fmt="o",
        ms=4,
        capsize=2,
        color="black",
        ecolor="black",
        label="Data − RV-only",
    )
    for key in FIT_VARIANT_ORDER:
        if key == "free" or key not in fit_variants:
            continue
        color, ls = FIT_VARIANT_STYLE[key]
        ax_bot.plot(
            t_dense,
            dense_map[key],
            ls=ls,
            lw=1.4,
            color=color,
            alpha=0.9,
            label=f"{FIT_VARIANT_LABEL[key]} − RV-only",
        )

    r_ylim = _residual_ylim(y - model_free_obs, yerr, list(resid_map.values()))
    ax_bot.set_ylim(*r_ylim)
    ax_bot.set_xlabel("MJD")
    ax_bot.set_ylabel("ΔRV (km/s)")
    ax_bot.set_xlim(t_start, t_end)
    ax_bot.grid(alpha=0.25)
    ax_bot.legend(loc="best", fontsize=7)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
