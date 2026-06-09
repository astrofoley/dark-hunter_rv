"""Keplerian RV fit figures (data-only, multi-model, residuals)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.ticker import AutoMinorLocator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fit_apf_rv_keplerian import RVPoint

from darkhunter_rv.rv_point_filters import rv_value_is_valid


def _fitmod():
    import fit_apf_rv_keplerian as m

    return m


FIT_VARIANT_ORDER = ("free", "fix_period", "fix_ecc", "fix_period_ecc")
FIT_VARIANT_STYLE: Dict[str, Tuple[str, str]] = {
    "free": ("tab:blue", "-"),
    "fix_period": ("tab:orange", "--"),
    "fix_ecc": ("tab:green", "-."),
    "fix_period_ecc": ("tab:red", ":"),
}
FIT_VARIANT_LABEL = {
    "free": "RV only",
    "fix_period": "P fixed",
    "fix_ecc": "e fixed",
    "fix_period_ecc": "P & e fixed",
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


def _xlim_from_data(t: np.ndarray, report: dict) -> Tuple[float, float]:
    """X limits spanning all RV epochs (small padding; include Today/APF markers if just outside)."""
    now_mjd = float(report.get("now_mjd", Time.now().mjd))
    if t.size == 0:
        return now_mjd - 30.0, now_mjd + 30.0
    t_span = float(np.ptp(t) + 1.0)
    pad = 0.03 * t_span
    t_lo = float(np.min(t) - pad)
    t_hi = float(np.max(t) + pad)
    t_lo = min(t_lo, now_mjd - 0.01 * t_span)
    t_hi = max(t_hi, now_mjd + 0.01 * t_span)
    obs_start, obs_end, _ = _observability_span(report)
    if obs_start is not None and obs_end is not None:
        t_lo = min(t_lo, obs_start - 0.01 * t_span)
        t_hi = max(t_hi, obs_end + 0.01 * t_span)
    return t_lo, t_hi


def _observability_windows(report: dict) -> List[dict]:
    obs_win = report.get("observability_window")
    if not isinstance(obs_win, dict):
        return []
    windows = obs_win.get("windows")
    if isinstance(windows, list) and windows:
        return [w for w in windows if isinstance(w, dict) and w.get("start_date") and w.get("end_date")]
    if obs_win.get("start_date") and obs_win.get("end_date"):
        return [obs_win]
    return []


def _observability_span(report: dict) -> Tuple[Optional[float], Optional[float], Optional[dict]]:
    obs_win = report.get("observability_window")
    if not isinstance(obs_win, dict):
        return None, None, None
    windows = _observability_windows(report)
    if not windows:
        return None, None, None
    starts: List[float] = []
    ends: List[float] = []
    for w in windows:
        try:
            starts.append(float(Time(w["start_date"], format="iso", scale="utc").mjd))
            ends.append(float(Time(w["end_date"], format="iso", scale="utc").mjd) + 1.0)
        except Exception:
            continue
    if not starts or not ends:
        return None, None, None
    return float(min(starts)), float(max(ends)), obs_win


def _shade_apf_window(
    ax,
    t_start: float,
    t_end: float,
    report: dict,
    *,
    annotate: bool = True,
) -> None:
    obs_win = report.get("observability_window")
    if not isinstance(obs_win, dict):
        return
    windows = _observability_windows(report)
    if not windows:
        return
    label_parts: List[str] = []
    for w in windows:
        try:
            obs_start = float(Time(w["start_date"], format="iso", scale="utc").mjd)
            obs_end = float(Time(w["end_date"], format="iso", scale="utc").mjd) + 1.0
        except Exception:
            continue
        left = max(t_start, obs_start)
        right = min(t_end, obs_end)
        if right <= left:
            continue
        ax.axvspan(left, right, color="tab:blue", alpha=0.12, zorder=0)
        label_parts.append(f"{w['start_date']} to {w['end_date']}")
    if annotate and label_parts:
        if obs_win.get("circumpolar"):
            msg = "Circumpolar (airmass ≤ 2.5): " + "; ".join(label_parts[:2])
        else:
            msg = "APF window " + "; ".join(label_parts[:3])
        ax.text(
            0.985,
            0.02,
            msg,
            transform=ax.transAxes,
            fontsize=8.5,
            color="tab:blue",
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="tab:blue", alpha=0.85, boxstyle="round,pad=0.2"),
        )


def _mark_today(ax, now_mjd: float, y_top_in: float, *, annotate: bool = True) -> None:
    ax.axvline(now_mjd, color="0.35", ls="--", lw=1.2, alpha=0.9, zorder=1)
    if annotate:
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


def _plot_points(
    ax,
    points: Sequence["RVPoint"],
    *,
    include_literature: bool,
    y_values: Optional[np.ndarray] = None,
    yerr: Optional[np.ndarray] = None,
) -> bool:
    plotted = False
    t = np.array([p.mjd for p in points], dtype=float)
    if y_values is None:
        y = np.array([p.rv for p in points], dtype=float)
    else:
        y = np.asarray(y_values, dtype=float)
    if yerr is None:
        yerr = np.array([p.rv_err for p in points], dtype=float)
    else:
        yerr = np.asarray(yerr, dtype=float)

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


def _plot_fit_curves(
    ax,
    t_dense: np.ndarray,
    fit_variants: Dict[str, Tuple[np.ndarray, dict]],
    t_ref: float,
) -> List[np.ndarray]:
    curves: List[np.ndarray] = []
    for key in FIT_VARIANT_ORDER:
        if key not in fit_variants:
            continue
        params, _vrep = fit_variants[key]
        y_dense = _fitmod().rv_model(params, t_dense, t_ref)
        curves.append(y_dense)
        color, ls = FIT_VARIANT_STYLE[key]
        ax.plot(
            t_dense,
            y_dense,
            ls=ls,
            lw=1.8,
            color=color,
            alpha=0.92,
            label=FIT_VARIANT_LABEL[key],
        )
    return curves


def _mark_time_reference(ax, report: dict, t_lo: float, t_hi: float, y_top_in: float, *, annotate: bool) -> None:
    now_mjd = float(report.get("now_mjd", Time.now().mjd))
    if t_lo <= now_mjd <= t_hi:
        _mark_today(ax, now_mjd, y_top_in, annotate=annotate)
    _shade_apf_window(ax, t_lo, t_hi, report, annotate=annotate)


def _m2sini_for_variant(vrep: dict, m1_msun: Optional[float]) -> Optional[float]:
    m = _fitmod()
    fm = vrep.get("mass_function_msun")
    if fm is None or not np.isfinite(fm):
        fm = m.mass_function_msun(vrep["P_days"], vrep["K_kms"], vrep["e"])
    if fm is None or not np.isfinite(fm) or fm <= 0:
        return None
    if m1_msun is not None and np.isfinite(m1_msun) and m1_msun > 0:
        return float(m.solve_m2sini_msun(float(fm), float(m1_msun)))
    return None


def _variant_param_lines(
    fit_variants: Dict[str, Tuple[np.ndarray, dict]],
    m1_msun: Optional[float],
) -> List[str]:
    lines: List[str] = []
    for key in FIT_VARIANT_ORDER:
        if key not in fit_variants:
            continue
        _, vrep = fit_variants[key]
        p_days = int(round(float(vrep["P_days"])))
        ecc = float(vrep["e"])
        m2sini = _m2sini_for_variant(vrep, m1_msun)
        label = FIT_VARIANT_LABEL[key]
        jit = vrep.get("jitter_kms")
        jit_s = ""
        if jit is not None and np.isfinite(jit) and float(jit) > 0:
            jit_s = f",  σ_jit = {float(jit):.3f} km/s"
        if m2sini is not None:
            lines.append(f"{label}:  P = {p_days} d,  e = {ecc:.3f},  M₂ sin i = {m2sini:.4f} M☉{jit_s}")
        else:
            fm = vrep.get("mass_function_msun")
            if fm is None or not np.isfinite(fm):
                fm = _fitmod().mass_function_msun(vrep["P_days"], vrep["K_kms"], vrep["e"])
            fm_s = f"{float(fm):.4f}" if fm is not None and np.isfinite(fm) else "—"
            lines.append(f"{label}:  P = {p_days} d,  e = {ecc:.3f},  f(M) = {fm_s} M☉{jit_s}")
    return lines


def _style_rv_axes(ax) -> None:
    ax.grid(alpha=0.25)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="major", direction="in", length=7, width=1.1, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=3.5, width=0.9, top=True, right=True)


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
    yerr = _fitmod().effective_yerr_for_points(pts, report)
    _plot_points(ax, pts, include_literature=False, yerr=yerr)

    y_lim = _y_limits_data_and_models(t, y, [])
    ax.set_ylim(*y_lim)
    y_top_in = y_lim[1] - 0.02 * (y_lim[1] - y_lim[0])
    _mark_today(ax, now_mjd, y_top_in)

    sid = _fitmod().parse_object_id_from_summary(summary_path) or summary_path.stem.replace("_summary", "")
    ax.set_xlabel("MJD")
    ax.set_ylabel("RV (km/s)")
    ax.set_title(f"{title_prefix}: Gaia DR3 {sid}")
    ax.set_xlim(t_start, t_end)
    _style_rv_axes(ax)
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
    """Fits-only figure: all data and four models; Today/APF markers without text."""
    del m1_msun  # mass annotations only on residuals figure
    if len(points) < 2:
        raise ValueError("need at least 2 points")

    t = np.array([p.mjd for p in points], dtype=float)
    y = np.array([p.rv for p in points], dtype=float)
    yerr = _fitmod().effective_yerr_for_points(points, report)
    t_ref = float(report["t_ref_mjd"])
    t_lo, t_hi = _xlim_from_data(t, report)
    t_dense = np.linspace(t_lo, t_hi, 2000)

    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    _plot_points(ax, points, include_literature=True, yerr=yerr)
    curves = _plot_fit_curves(ax, t_dense, fit_variants, t_ref)

    y_lim = _y_limits_data_and_models(t, y, curves)
    ax.set_ylim(*y_lim)
    y_top_in = y_lim[1] - 0.02 * (y_lim[1] - y_lim[0])
    _mark_time_reference(ax, report, t_lo, t_hi, y_top_in, annotate=False)

    sid = _fitmod().parse_object_id_from_summary(summary_path) or summary_path.stem.replace("_summary", "")
    ax.set_xlabel("MJD")
    ax.set_ylabel("RV (km/s)")
    ax.set_title(f"Keplerian fits: Gaia DR3 {sid}")
    ax.set_xlim(t_lo, t_hi)
    _style_rv_axes(ax)

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
    *,
    m1_msun: Optional[float] = None,
) -> None:
    if "free" not in fit_variants:
        raise ValueError("residual plot requires a free RV fit")
    if len(points) < 2:
        raise ValueError("need at least 2 points")

    if m1_msun is None:
        used = report.get("used_m1_msun")
        if used is not None and np.isfinite(used):
            m1_msun = float(used)

    t = np.array([p.mjd for p in points], dtype=float)
    y = np.array([p.rv for p in points], dtype=float)
    yerr = _fitmod().effective_yerr_for_points(points, report)
    t_ref = float(report["t_ref_mjd"])
    t_lo, t_hi = _xlim_from_data(t, report)
    t_dense = np.linspace(t_lo, t_hi, 2000)
    now_mjd = float(report.get("now_mjd", Time.now().mjd))

    rv_model = _fitmod().rv_model
    params_free, _rep_free = fit_variants["free"]
    model_free_obs = rv_model(params_free, t, t_ref)
    model_free_dense = rv_model(params_free, t_dense, t_ref)
    resid_data = y - model_free_obs
    resid_map: Dict[str, np.ndarray] = {"free": resid_data}
    dense_map: Dict[str, np.ndarray] = {}
    for key in FIT_VARIANT_ORDER:
        if key not in fit_variants:
            continue
        params, _vrep = fit_variants[key]
        model_obs = rv_model(params, t, t_ref)
        resid_map[key] = y - model_obs
        if key != "free":
            dense_map[key] = rv_model(params, t_dense, t_ref) - model_free_dense

    fig = plt.figure(figsize=(10.8, 7.6))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[2.25, 1.05, 0.55],
        hspace=0.06,
    )
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    ax_txt = fig.add_subplot(gs[2])
    ax_txt.axis("off")

    _plot_points(ax_top, points, include_literature=True, yerr=yerr)
    curves = _plot_fit_curves(ax_top, t_dense, fit_variants, t_ref)
    y_lim = _y_limits_data_and_models(t, y, curves)
    ax_top.set_ylim(*y_lim)
    y_top_in = y_lim[1] - 0.02 * (y_lim[1] - y_lim[0])
    _mark_time_reference(ax_top, report, t_lo, t_hi, y_top_in, annotate=True)
    if t_lo <= now_mjd <= t_hi:
        ax_bot.axvline(now_mjd, color="0.35", ls="--", lw=1.2, alpha=0.9, zorder=1)
    _shade_apf_window(ax_bot, t_lo, t_hi, report, annotate=False)

    sid = _fitmod().parse_object_id_from_summary(summary_path) or summary_path.stem.replace("_summary", "")
    ax_top.set_ylabel("RV (km/s)")
    ax_top.set_title(f"Keplerian fits + residuals: Gaia DR3 {sid}")
    _style_rv_axes(ax_top)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_bot.axhline(0.0, color="0.4", lw=1.0, zorder=1)
    _plot_points(ax_bot, points, include_literature=True, y_values=resid_data, yerr=yerr)
    for key in FIT_VARIANT_ORDER:
        if key == "free" or key not in dense_map:
            continue
        color, ls = FIT_VARIANT_STYLE[key]
        ax_bot.plot(
            t_dense,
            dense_map[key],
            ls=ls,
            lw=1.4,
            color=color,
            alpha=0.9,
            label=FIT_VARIANT_LABEL[key],
        )

    r_ylim = _residual_ylim(resid_data, yerr, list(resid_map.values()))
    ax_bot.set_ylim(*r_ylim)
    ax_bot.set_ylabel("ΔRV (km/s)")
    ax_bot.set_xlim(t_lo, t_hi)
    ax_bot.set_xlabel("MJD")
    _style_rv_axes(ax_bot)

    handles, labels = ax_top.get_legend_handles_labels()
    if handles:
        ax_top.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=8.5,
            framealpha=0.95,
        )

    param_lines = _variant_param_lines(fit_variants, m1_msun)
    ax_txt.text(
        0.01,
        0.95,
        "\n".join(param_lines),
        transform=ax_txt.transAxes,
        fontsize=8.5,
        va="top",
        ha="left",
        family="monospace",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(right=0.82)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
