# pipeline.py — main RV processing
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from . import config, instruments, io_utils, continuum, rv_core, templates, chunking, plotting, qc
from darkhunter_rv.summary_paths import is_primary_epoch_spectrum_name

logger = logging.getLogger(__name__)

# Exposure mean uses error-weighted average; clip only obvious outliers (mean-based sigma_clip was too harsh).
_STACK_CLIP_MIN_ORDERS = 12
_STACK_CLIP_SIGMA = 3.2
_STACK_CLIP_MAXITERS = 2
_STACK_CLIP_MIN_KEEP_FRAC = 0.35
_STACK_CLIP_MIN_KEEP_ABS = 4


def _resolve_chunk_layout(args: argparse.Namespace) -> None:
    """Apply env/default production chunk layout when --chunk-layout is omitted."""
    if args.chunk_layout is not None:
        p = Path(args.chunk_layout)
        if not p.is_file():
            logger.error("Chunk layout not found: %s", p)
            sys.exit(2)
        return
    env = os.environ.get("DARKHUNTER_CHUNK_LAYOUT")
    if env:
        p = Path(env)
        if p.is_file():
            args.chunk_layout = p
            return
        logger.warning("DARKHUNTER_CHUNK_LAYOUT=%s not found; trying repo default", env)
    if config.DEFAULT_CHUNK_LAYOUT is not None and config.DEFAULT_CHUNK_LAYOUT.is_file():
        args.chunk_layout = config.DEFAULT_CHUNK_LAYOUT


def _apply_gaia_metadata_to_args(args: argparse.Namespace, gaia_data: dict | None) -> None:
    """Mutate ``args`` teff / logg / mh from Gaia ``metadata`` when present."""
    if not gaia_data:
        return
    md = gaia_data.get("metadata") or {}
    teff_g = md.get("Teff", md.get("teff"))
    if teff_g is not None:
        try:
            tf = float(teff_g)
            if np.isfinite(tf):
                args.teff = tf
        except (TypeError, ValueError):
            pass
    if args.logg is None:
        for k in ("logg", "logg_gspphot"):
            g = md.get(k)
            if g is not None:
                try:
                    gf = float(g)
                    if np.isfinite(gf):
                        args.logg = gf
                        break
                except (TypeError, ValueError):
                    pass
        if args.logg is None:
            args.logg = 4.5
    if args.mh is None:
        for k in ("MH", "mh_gspphot", "M_H", "feh"):
            mh_val = md.get(k)
            if mh_val is not None:
                try:
                    mf = float(mh_val)
                    if np.isfinite(mf):
                        args.mh = mf
                        break
                except (TypeError, ValueError):
                    pass
        if args.mh is None:
            args.mh = 0.0


def _attach_diagnostics_teff(args: argparse.Namespace, gid: int | None, gaia_data: dict | None) -> None:
    """
    Teff stored in per-chunk diagnostics: finite Gaia GSP-Phot Teff when available, else NaN.

    ``args.teff`` may remain ``config.DEFAULT_TEFF`` for template/mask heuristics when Gaia Teff is
    missing; that default must not be written as if it were a measured temperature.
    """
    if gid is None:
        args._diagnostics_teff = float(args.teff)
        return
    if not gaia_data:
        args._diagnostics_teff = float("nan")
        return
    md = gaia_data.get("metadata") or {}
    tv = md.get("Teff", md.get("teff"))
    try:
        tf = float(tv)
        args._diagnostics_teff = tf if np.isfinite(tf) else float("nan")
    except (TypeError, ValueError):
        args._diagnostics_teff = float("nan")


def _hb_joint_fit_params_json(bundle: dict) -> str:
    p = bundle.get("hb_joint_fit_params")
    if not p or len(p) != 8:
        return ""
    return json.dumps([float(x) for x in p])


def _rv_kms_from_hb_joint_line_center(bundle: dict) -> float:
    """Radial velocity (km/s) from stored joint-fit line center λ_c; matches plots and model."""
    p = bundle.get("hb_joint_fit_params")
    if not p or len(p) < 8:
        return float(bundle.get("rv_voigt_kms", np.nan))
    ctr = float(p[2])
    return float(config.C_KMS * (ctr - rv_core.HB_REST_A) / rv_core.HB_REST_A)


def _weighted_rms(
    rv: np.ndarray,
    er: np.ndarray,
    *,
    mu_weighted: float | None = None,
) -> float:
    """Inverse-variance weighted RMS of chunk RVs about the weighted mean."""
    rv = np.asarray(rv, float).ravel()
    er = np.asarray(er, float).ravel()
    n = int(rv.size)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    w = 1.0 / (er**2 + 1e-9)
    mu = float(mu_weighted) if mu_weighted is not None else float(np.average(rv, weights=w))
    return float(np.sqrt(np.average((rv - mu) ** 2, weights=w)))


def _mask_ccf_stack_error_inflated(
    rv: np.ndarray,
    er: np.ndarray,
    formal_combined: float,
    *,
    mu_weighted: float,
) -> float:
    """
    Inverse-variance formal error can be far too small when a few chunks have optimistic Gaussian
    ``sigma_mu`` (broad/noisy lines) while chunk RVs disagree. Floor the combined error using spread
    of the chunk RVs: MAD and std-based terms, plus **weighted RMS of residuals about the weighted mean**
    divided by ``sqrt(N_orders)`` (chunk count), as requested for exposure-level mask stacks.
    """
    rv = np.asarray(rv, float).ravel()
    er = np.asarray(er, float).ravel()
    formal = float(formal_combined)
    n = int(rv.size)
    if n < 2 or er.size != n:
        return formal
    mu = float(mu_weighted)
    resid_rms = _weighted_rms(rv, er, mu_weighted=mu)
    from_rms = resid_rms / np.sqrt(n)
    med = float(np.median(rv))
    mad = float(np.median(np.abs(rv - med))) * 1.4826
    from_mad = mad / np.sqrt(n)
    if n >= 3:
        from_std = float(np.std(rv, ddof=1) / np.sqrt(n))
    else:
        from_std = 0.5 * float(abs(rv[0] - rv[1]))
    return float(max(formal, from_rms, from_mad, from_std, 1e-9))


def _weighted_method_rv_from_rows(rows: list[dict], method: str) -> tuple[float, float]:
    """Error-weighted mean over diagnostics rows for one method (per-chunk or exposure-level)."""
    ck_all_only = method in ("strong_lines",)
    sub: list[dict] = []
    for r in rows:
        if str(r.get("method", "")) != method:
            continue
        ck = str(r.get("chunk_key", ""))
        if ck_all_only:
            if ck != "all":
                continue
        elif ck == "all":
            continue
        rv = float(r.get("rv_kms", np.nan))
        er = float(r.get("rv_err_kms", np.nan))
        if not np.isfinite(rv) or not np.isfinite(er) or er <= 0 or er > 1e27:
            continue
        if method in ("mask_ccf", "template_fft") and not bool(r.get("qc_pass", True)):
            continue
        sub.append(r)
    if not sub:
        return float("nan"), float("nan")
    if method == "template_fft" and len(sub) < int(config.MIN_TEMPLATE_FFT_CHUNKS_FOR_STACK):
        return float("nan"), float("nan")
    if method == "mask_ccf" and len(sub) < int(config.MIN_MASK_CCF_CHUNKS_FOR_STACK):
        return float("nan"), float("nan")
    rv = np.array([float(x["rv_kms"]) for x in sub], dtype=float)
    er = np.array([float(x["rv_err_kms"]) for x in sub], dtype=float)
    if method == "template_fft" and len(rv) >= _STACK_CLIP_MIN_ORDERS:
        keep = _exposure_stack_keep_mask(rv)
        if not np.all(keep):
            rv = rv[keep]
            er = er[keep]
        if len(rv) < int(config.MIN_TEMPLATE_FFT_CHUNKS_FOR_STACK):
            return float("nan"), float("nan")
    w = 1.0 / (er**2 + 1e-18)
    mu = float(np.sum(w * rv) / np.sum(w))
    sig = float(np.sqrt(1.0 / np.sum(w)))
    if method == "mask_ccf":
        sig = _mask_ccf_stack_error_inflated(rv, er, sig, mu_weighted=mu)
    return mu, sig


def _weighted_template_fft_for_order(rows: list[dict], order_num: int) -> tuple[float, float]:
    """Error-weighted template_fft RV for one echelle order (all sub-chunks), no 3-chunk stack rule."""
    prefix = f"{int(order_num)}_"
    sub: list[dict] = []
    for r in rows:
        if str(r.get("method", "")) != "template_fft":
            continue
        ck = str(r.get("chunk_key", ""))
        if not ck.startswith(prefix):
            continue
        rv = float(r.get("rv_kms", np.nan))
        er = float(r.get("rv_err_kms", np.nan))
        if not np.isfinite(rv) or not np.isfinite(er) or er <= 0 or er > 1e27:
            continue
        if not bool(r.get("qc_pass", True)):
            continue
        sub.append(r)
    if not sub:
        return float("nan"), float("nan")
    rv = np.array([float(x["rv_kms"]) for x in sub], dtype=float)
    er = np.array([float(x["rv_err_kms"]) for x in sub], dtype=float)
    w = 1.0 / (er**2 + 1e-18)
    mu = float(np.sum(w * rv) / np.sum(w))
    sig = float(np.sqrt(1.0 / np.sum(w)))
    return mu, sig


def _template_rv_err_from_hb_ccf_peak(hb: dict) -> float:
    """Rough σ for Hβ-window template grid CCF (same scaling as :func:`rv_core.measure_h_beta_rv` best path)."""
    pk = hb.get("template_ccf_peak")
    if pk is None or not np.isfinite(float(pk)):
        return float("nan")
    pk = float(pk)
    if pk <= 0.32:
        return float("nan")
    return float(max(4.0, 90.0 / max(pk, 0.35)))


def _dominant_template_key(rows: list[dict]) -> str:
    keys = [
        str(r.get("template_key", ""))
        for r in rows
        if str(r.get("method", "")) == "template_fft" and str(r.get("template_key", "")).strip()
    ]
    if not keys:
        return ""
    return max(set(keys), key=keys.count)


def _bank_entry_for_template_key_str(bank: dict, tpl_key_str: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not bank:
        return None
    for k, wf in bank.items():
        if str(k) == tpl_key_str:
            return wf
    return next(iter(bank.values())) if bank else None


def _continuum_fit_kw(args: argparse.Namespace, hot_continuum: bool) -> dict:
    """Exclude pixels near strong lines from spline continuum anchors (see continuum.fit_continuum)."""
    kw: dict = {"continuum_mode": args.continuum_mode}
    kw["exclude_near_lines_width"] = float(
        config.HOT_SPLINE_EXCLUDE_NEAR_LINES_WIDTH
        if hot_continuum
        else config.COOL_SPLINE_EXCLUDE_NEAR_LINES_WIDTH
    )
    return kw


def _vote_exposure_fft_template_key(
    chunk_preps: list[dict],
    bank: dict,
    vsini: float,
    args: argparse.Namespace,
    use_fft_primary: bool,
    instrument,
) -> object | None:
    """
    One PHOENIX key for the whole exposure: choose (Teff, log g, [M/H]) using **narrowest** ``v sin i``
    per atmosphere (:func:`templates.narrow_fft_subbank`), then pick ``v sin i`` from the full grid for
    that triple by **minimum mean RMS** (:func:`rv_core.rms_absorption_residual_fft_grid`). Per chunk,
    RV is the template CCF argmax in ``FFT_EXPOSURE_VOTE_RV_SEED_KMS`` ± half-width (before mask CCF).

    Narrow coarse avoids ``coarse_fft_subbank`` (proxy-matched vsini): an inflated ``vsini_proxy`` would
    otherwise broaden every atmosphere before RMS, making wrong giants / wrong Teff win.
    """
    coarse = templates.narrow_fft_subbank(bank)
    if len(coarse) < 2:
        return None
    vote_half = (
        float(config.FFT_EXPOSURE_VOTE_RV_HALF_WIDTH_KMS_HOT)
        if use_fft_primary
        else float(config.FFT_EXPOSURE_VOTE_RV_HALF_WIDTH_KMS_COOL)
    )
    vote_seed = float(config.FFT_EXPOSURE_VOTE_RV_SEED_KMS)
    R = float(getattr(instrument, "resolving_power", 60_000.0))

    sum_rms_st: dict = defaultdict(float)
    cnt_st: dict = defaultdict(int)
    for prep in chunk_preps:
        nw, nf = prep["nw"], prep["nf"]
        nw = np.asarray(nw, float)
        line = rv_core.mask_line_flux_in_excluded_wavelengths(nw, 1.0 - nf)
        obs_resamp, window, fft_obs, _va, mask_vel, vel_win, tpl_grid_wave = rv_core._fft_velocity_window(
            nw,
            line,
            rv_seed_kms=vote_seed,
            rv_search_half_width_kms=vote_half,
        )
        for k, wf in coarse.items():
            tw, tf = np.asarray(wf[0], float), np.asarray(wf[1], float)
            rpk = rv_core._fft_correlation_peak_for_template(
                obs_resamp,
                window,
                fft_obs,
                mask_vel,
                vel_win,
                tpl_grid_wave,
                tw,
                tf,
            )
            if rpk is None:
                continue
            _pk, rv_at_peak, _ = rpk
            rms = rv_core.rms_absorption_residual_fft_grid(
                nw, line, tw, tf, float(rv_at_peak), R
            )
            if not np.isfinite(rms):
                continue
            st = templates.template_key_stellar_tuple(k)
            if st is None:
                continue
            sum_rms_st[st] += rms
            cnt_st[st] += 1

    if not cnt_st:
        return None
    best_st = min(sum_rms_st.keys(), key=lambda st: sum_rms_st[st] / max(cnt_st[st], 1))

    sub = templates.refined_fft_subbank(bank, {best_st})
    if not sub:
        return None
    if len(sub) == 1:
        return next(iter(sub.keys()))

    sum_rms_k: dict = defaultdict(float)
    cnt_k: dict = defaultdict(int)
    for prep in chunk_preps:
        nw, nf = prep["nw"], prep["nf"]
        nw = np.asarray(nw, float)
        line = rv_core.mask_line_flux_in_excluded_wavelengths(nw, 1.0 - nf)
        obs_resamp, window, fft_obs, _va, mask_vel, vel_win, tpl_grid_wave = rv_core._fft_velocity_window(
            nw,
            line,
            rv_seed_kms=vote_seed,
            rv_search_half_width_kms=vote_half,
        )
        for kk, wf in sub.items():
            tw, tf = np.asarray(wf[0], float), np.asarray(wf[1], float)
            rpk = rv_core._fft_correlation_peak_for_template(
                obs_resamp,
                window,
                fft_obs,
                mask_vel,
                vel_win,
                tpl_grid_wave,
                tw,
                tf,
            )
            if rpk is None:
                continue
            _pk, rv_at_peak, _ = rpk
            rms = rv_core.rms_absorption_residual_fft_grid(
                nw, line, tw, tf, float(rv_at_peak), R
            )
            if not np.isfinite(rms):
                continue
            sum_rms_k[kk] += rms
            cnt_k[kk] += 1

    if not cnt_k:
        return next(iter(sub.keys()))

    def _mean_rms(kk: object) -> float:
        return float(sum_rms_k[kk] / max(cnt_k[kk], 1))

    mmin = min(_mean_rms(kk) for kk in cnt_k)
    eps = 1e-5
    candidates = [kk for kk in sub if kk in cnt_k and _mean_rms(kk) <= mmin + eps]

    def _vs(k: object) -> float:
        if isinstance(k, tuple) and len(k) >= 4:
            return float(k[3])
        return 0.0

    return min(candidates, key=_vs) if candidates else next(iter(sub.keys()))


def _exposure_stack_keep_mask(rv_arr: np.ndarray) -> np.ndarray:
    """Median + MAD sigma rejection; skipped for small N; disabled if it would discard most chunks."""
    n = len(rv_arr)
    all_true = np.ones(n, dtype=bool)
    if n < _STACK_CLIP_MIN_ORDERS:
        return all_true
    keep = np.ones(n, dtype=bool)
    for _ in range(_STACK_CLIP_MAXITERS):
        m = float(np.median(rv_arr[keep]))
        mad = float(np.median(np.abs(rv_arr[keep] - m))) + 1e-12
        scale = 1.4826 * mad
        if scale < 1e-8:
            break
        new_keep = np.abs(rv_arr - m) < (_STACK_CLIP_SIGMA * scale)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    min_keep = max(_STACK_CLIP_MIN_KEEP_ABS, int(np.ceil(_STACK_CLIP_MIN_KEEP_FRAC * n)))
    if np.sum(keep) < min_keep:
        logger.warning(
            "Exposure stack clip would keep only %d/%d chunks (min %d); keeping all",
            int(np.sum(keep)),
            n,
            min_keep,
        )
        return all_true
    return keep



def setup_logging(level: str, quiet: bool) -> None:
    if quiet:
        level = "ERROR"
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _mask_tournament(
    spec_data: dict,
    instrument,
    test_orders: list,
    mask_glob_dir: Path,
    continuum_mode: str,
) -> tuple[dict | None, str, list[tuple[str, float]]]:
    """Return ({w,s}, best_name, [(stem, peak_sum), ...])."""
    inst_lower = instrument.name.lower()
    mask_files = list(mask_glob_dir.glob(f"*_{inst_lower}.txt"))
    if not mask_files:
        mask_files = list(mask_glob_dir.glob("*_espresso.txt"))
    if not mask_files:
        return None, "none", []

    scores: list[tuple[str, float]] = []
    best_peak = -np.inf
    best_pack: dict | None = None
    best_name = ""

    for mf in sorted(mask_files):
        try:
            md = np.loadtxt(mf)
            mw, ms = md[:, 0], md[:, 1]
        except Exception:
            continue
        peak_sum = 0.0
        for o in test_orders:
            if o not in spec_data:
                continue
            d = spec_data[o]
            w = np.array(d["wavelength"], float)
            f = np.array(d["flux"], float)
            e = np.array(d["eflux"], float)
            if len(w) < 10:
                continue
            try:
                nw, nf, _ = continuum.fit_continuum(w, f, e, continuum_mode=continuum_mode)
            except Exception:
                continue
            if nw[-1] < mw[0] or nw[0] > mw[-1]:
                continue
            line_t = rv_core.mask_line_flux_in_excluded_wavelengths(nw, 1.0 - nf)
            _, _, _, _, p, _, _ = rv_core.cross_correlate_stellar_mask(nw, line_t, mw, ms)
            peak_sum += float(p)
        scores.append((mf.stem, peak_sum))
        if peak_sum > best_peak:
            best_peak = peak_sum
            best_pack = {"w": mw, "s": ms}
            best_name = mf.stem

    if not best_pack:
        return None, "none", scores
    return best_pack, best_name, scores


def process_spectrum(
    spectrum_file: str,
    args: argparse.Namespace,
    instrument,
    plot_root: Path | None,
) -> dict | None:
    spectrum_file = str(spectrum_file)
    logger.info("Processing instrument=%s file=%s", instrument.name, spectrum_file)

    if instrument.name == "GHOST":
        header, spec_data = io_utils.read_spectrum_ghost(spectrum_file)
    elif instrument.name == "MAROON-X":
        header, spec_data = io_utils.read_spectrum_maroonx(spectrum_file)
    else:
        header, spec_data = io_utils.read_spectrum(spectrum_file)

    mjd = io_utils.extract_mjd_from_header(header, instrument)
    teff = float(args.teff)
    teff_diag = float(getattr(args, "_diagnostics_teff", teff))
    use_fft_primary = teff > config.HOT_STAR_TEFF_THRESHOLD
    run_multi = bool(args.run_all_methods)
    if getattr(args, "mask_only", False):
        # Stellar-mask-only calibration: no PHOENIX bank, no template/strong diagnostics.
        use_fft_primary = False
        run_multi = False
    qc_thresholds = qc.load_qc_config(Path(args.qc_config), instrument.name)

    mask_dir = Path(instrument.mask_directory)
    valid_orders = sorted(o for o in spec_data if o not in instrument.bad_orders)
    if not valid_orders:
        logger.error("No valid orders in %s", spectrum_file)
        return None

    mid = len(valid_orders) // 2
    test_orders = valid_orders[max(0, mid - 2) : min(len(valid_orders), mid + 2)]

    bank = None
    vsini = 10.0
    mask_pack, best_mask_name, tournament_scores = None, "", []
    mw, ms = None, None

    # Mask tournament: always for cool stars; for hot stars only when comparing all methods (so every
    # Teff gets mask_ccf diagnostics when --run-all-methods / --compare-rv-methods is set).
    run_mask_paths = (not use_fft_primary) or run_multi
    if run_mask_paths:
        mask_pack, best_mask_name, tournament_scores = _mask_tournament(
            spec_data, instrument, test_orders, mask_dir, args.continuum_mode
        )
        if mask_pack is not None:
            mw, ms = mask_pack["w"], mask_pack["s"]
            if use_fft_primary:
                logger.info(
                    "Mask tournament (method comparison) winner=%s scores=%s",
                    best_mask_name,
                    tournament_scores[:3],
                )
            else:
                logger.info("Mask tournament winner=%s scores=%s", best_mask_name, tournament_scores[:3])
    elif use_fft_primary:
        logger.info(
            "Teff=%.0f K (hot): primary template FFT only; omit mask (use --run-all-methods for mask CCF)",
            teff,
        )

    # PHOENIX bank: hot stars always; cool stars when comparing methods (template_fft diagnostics).
    if use_fft_primary or run_multi:
        init_bank = templates.build_template_bank_cached(
            teff,
            50.0,
            metallicity=float(args.mh),
            logg=float(args.logg),
            hot_spectrum=bool(use_fft_primary),
            template_grid_wide=bool(args.template_grid_wide),
        )
        if init_bank:
            tw, tf = next(iter(init_bank.values()))
            all_w, all_f = [], []
            for o in valid_orders:
                w = np.array(spec_data[o]["wavelength"], float)
                f = np.array(spec_data[o]["flux"], float)
                qw, qf = continuum.quick_normalize(w, f)
                all_w.append(qw)
                all_f.append(qf)
            if all_w:
                cat_w = np.concatenate(all_w)
                cat_f = np.concatenate(all_f)
                vb, meta_b = rv_core.estimate_broadening(cat_w, cat_f, tw, tf)
                meta_b = meta_b or {}
                rej_raw = meta_b.get("vsini_proxy_rejected_kms")
                rej_f = float(rej_raw) if rej_raw is not None else float("nan")
                if vb is not None and np.isfinite(float(vb)):
                    vsini_raw = float(vb)
                    vmin = float(config.VSINI_PROXY_MIN_KMS)
                    if vsini_raw < vmin:
                        vsini = vmin
                        if vsini_raw + 0.5 < vmin:
                            logger.warning(
                                "Raised vsini_proxy from %.2f to %.2f km/s (floor)",
                                vsini_raw,
                                vmin,
                            )
                    else:
                        vsini = vsini_raw
                elif np.isfinite(rej_f):
                    vrg = float(config.VSINI_PROXY_REJECTED_GRID_KMS)
                    vmax = float(config.VSINI_PROXY_MAX_KMS)
                    logger.warning(
                        "Rejected vsini_proxy estimate %.2f km/s (> %.1f max); using %.1f km/s for template grid",
                        rej_f,
                        vmax,
                        vrg,
                    )
                    vsini = vrg
                elif vb is not None:
                    vnf = float(config.VSINI_PROXY_NONFINITE_GRID_KMS)
                    logger.warning(
                        "Non-finite vsini_proxy estimate; using %.1f km/s for template grid",
                        vnf,
                    )
                    vsini = vnf
                else:
                    vnf = float(config.VSINI_PROXY_NONFINITE_GRID_KMS)
                    logger.warning(
                        "No vsini_proxy from broadening CCF; using %.1f km/s for template grid",
                        vnf,
                    )
                    vsini = vnf
        bank = templates.build_template_bank_cached(
            teff,
            vsini,
            metallicity=float(args.mh),
            logg=float(args.logg),
            hot_spectrum=bool(use_fft_primary),
            template_grid_wide=bool(args.template_grid_wide),
        )
        if bank:
            logger.info(
                "PHOENIX bank priors teff=%.0f logg=%.2f [M/H]=%.2f vsini_proxy=%.2f wide_grid=%s n_templates=%d",
                teff,
                float(args.logg),
                float(args.mh),
                vsini,
                bool(args.template_grid_wide),
                len(bank),
            )
        if not bank:
            logger.warning("No PHOENIX templates (dir %s). Template path skipped.", config.PHOENIX_BASE_DIR)

    bias: dict = {}
    if not args.no_bias and instrument.bias_file:
        bias = io_utils.read_bias(instrument.bias_file)
        if bias:
            logger.info(
                "Per-order debias: %d entries from %s (b0 RV shift; b1/b2 err inflation)",
                len(bias),
                instrument.bias_file,
            )
        elif instrument.bias_file:
            logger.warning(
                "Bias file missing or empty (%s); mask CCF RVs are not debiased",
                instrument.bias_file,
            )

    diagnostics_rows: list[dict] = []
    rv_results: dict = {}
    primary_rv_list: list[float] = []
    primary_err_list: list[float] = []
    chunk_keys_plot: list[str] = []
    primary_chunk_methods: list[str] = []
    stem = Path(spectrum_file).stem
    order_mask_ccf: dict[int, dict] = {}
    plots_focus = bool(getattr(args, "plots_focus", False))
    plot_detail = plot_root is not None and not plots_focus
    plots_only = bool(getattr(args, "plots_only", False))
    plots_skip_chunk_pngs = bool(getattr(args, "plots_skip_chunk_pngs", False))
    chunk_stems_path = getattr(args, "plots_chunk_detail_stems_file", None)
    chunk_detail_stems: set[str] | None = None
    if chunk_stems_path:
        p_stem = Path(str(chunk_stems_path))
        if p_stem.is_file():
            chunk_detail_stems = {
                ln.strip()
                for ln in p_stem.read_text().splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            }
    plot_chunk_pngs = False
    if plot_detail:
        if not plots_skip_chunk_pngs:
            plot_chunk_pngs = True
        elif chunk_detail_stems is not None and stem in chunk_detail_stems:
            plot_chunk_pngs = True

    plots_strong_line_panels = bool(getattr(args, "plots_strong_line_panels", False))
    plots_skip_order_summary = bool(getattr(args, "plots_skip_order_summary", False))

    if plot_detail and tournament_scores:
        plotting.plot_tournament_scores(
            [a for a, _ in tournament_scores],
            [b for _, b in tournament_scores],
            plot_root / f"{stem}_tournament.png",
        )
    if tournament_scores and not plots_only:
        tourn_csv = config.OUTPUT_DIR / f"{stem}_tournament.csv"
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(tourn_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["mask_stem", "peak_score_sum"])
            for name, sc in tournament_scores:
                w.writerow([name, f"{sc:.6f}"])
        logger.info("Tournament table %s", tourn_csv)

    _tpl_fft_abs_cap = float(
        config.TEMPLATE_FFT_MAX_ABS_RV_KMS_HOT
        if use_fft_primary
        else config.TEMPLATE_FFT_MAX_ABS_RV_KMS_COOL
    )

    chunk_preps: list[dict] = []
    chunk_layout_path = getattr(args, "chunk_layout", None)
    if chunk_layout_path:
        from validation.chunk_layout import load_chunk_layout, iter_order_chunks_from_layout

        layout = load_chunk_layout(chunk_layout_path)
        chunk_iter = iter_order_chunks_from_layout(spec_data, instrument.bad_orders, layout)
    else:
        chunk_iter = chunking.iter_order_chunks(spec_data, instrument.bad_orders, args.subchunks)
    for chunk_key, w, f, e in chunk_iter:
        if len(w) < 10:
            continue
        try:
            nw, nf, ne = continuum.fit_continuum(w, f, e, **_continuum_fit_kw(args, use_fft_primary))
            nw, nf, ne = continuum.despike_normalized_pre_ccf(nw, nf, ne)
        except Exception as ex:
            logger.debug("continuum fail chunk=%s: %s", chunk_key, ex)
            continue
        chunk_preps.append(
            {"chunk_key": chunk_key, "w": w, "f": f, "e": e, "nw": nw, "nf": nf, "ne": ne}
        )

    bank_fft = bank
    if (
        bank
        and not bool(getattr(args, "no_fixed_exposure_template", False))
        and len(chunk_preps) >= 1
    ):
        fk = _vote_exposure_fft_template_key(
            chunk_preps, bank, vsini, args, use_fft_primary, instrument
        )
        if fk is not None and fk in bank:
            bank_fft = {fk: bank[fk]}
            logger.info("Exposure template_fft fixed to single PHOENIX key %s", fk)

    for prep in chunk_preps:
        chunk_key = prep["chunk_key"]
        w, f, e = prep["w"], prep["f"], prep["e"]
        nw, nf, ne = prep["nw"], prep["nf"], prep["ne"]
        line_obs = rv_core.mask_line_flux_in_excluded_wavelengths(nw, 1.0 - nf)

        b_order = chunking.bias_order_from_chunk_key(chunk_key)
        bvec = bias.get(b_order, [0.0, 0.0, 0.0]) if b_order is not None else [0.0, 0.0, 0.0]

        tell_frac = qc.telluric_fraction(nw)
        mask_line_count = qc.mask_line_count_in_chunk(nw, mw if "mw" in locals() else None)
        if isinstance(bvec, (list, tuple)) and len(bvec) >= 3:
            b0, b1, b2 = float(bvec[0]), float(bvec[1]), float(bvec[2])
        else:
            b0, b1, b2 = 0.0, 0.0, 0.0

        qc_ok_mask_chunk = False

        # --- Mask CCF ---
        rv_m, err_m, vels_m, ccf_m, peak_m, gauss_p = np.nan, np.nan, None, None, 0.0, None
        peak_snr_m = np.nan
        gauss_ok_m = False
        gauss_plot_ccf = None
        if mw is not None and not (nw[-1] < mw[0] or nw[0] > mw[-1]):
            rv_m, err_m, vels_m, ccf_m, peak_m, gauss_p, peak_snr_m = rv_core.cross_correlate_stellar_mask(
                nw, line_obs, mw, ms
            )
            if gauss_p is not None:
                if len(gauss_p) == 5:
                    c0g, c1g, a_g, mu_g, sig_g = gauss_p
                    gauss_plot_ccf = (
                        c0g + c1g * b0,
                        c1g,
                        a_g,
                        mu_g - b0,
                        sig_g,
                    )
                elif len(gauss_p) == 4:
                    c0g, a_g, mu_g, sig_g = gauss_p
                    gauss_plot_ccf = (c0g, 0.0, a_g, mu_g - b0, sig_g)
                else:
                    a_g, mu_g, sig_g = gauss_p
                    gauss_plot_ccf = (0.0, 0.0, a_g, mu_g - b0, sig_g)
            vels_plot = np.asarray(vels_m, float) - b0
            rv_m -= b0
            err_m = float(np.sqrt(err_m**2 + b1**2 + b2**2)) if np.isfinite(err_m) else err_m
            gauss_ok_m = np.isfinite(err_m) and err_m < 1e29
            _, ord_num, _ = chunking.parse_chunk_key(chunk_key)
            if vels_m is not None and ccf_m is not None and ord_num not in order_mask_ccf:
                peak_for_plot = float(rv_m) if np.isfinite(rv_m) else float("nan")
                if gauss_plot_ccf is not None:
                    lg = len(gauss_plot_ccf)
                    if lg >= 5:
                        gmu = float(gauss_plot_ccf[3])
                    elif lg == 4:
                        gmu = float(gauss_plot_ccf[2])
                    else:
                        gmu = float(gauss_plot_ccf[1])
                    if np.isfinite(gmu):
                        peak_for_plot = gmu
                order_mask_ccf[ord_num] = {
                    "vel": np.asarray(vels_plot, float),
                    "ccf": np.asarray(ccf_m, float),
                    "peak_vel": peak_for_plot,
                    "gauss": gauss_plot_ccf,
                    "label": str(chunk_key),
                }

            diagnostics_rows.append(
                {
                    "file": spectrum_file,
                    "chunk_key": chunk_key,
                    "mjd": mjd,
                    "teff": teff_diag,
                    "continuum_mode": args.continuum_mode,
                    "method": "mask_ccf",
                    "mask_name": best_mask_name,
                    "rv_kms": rv_m,
                    "rv_err_kms": err_m,
                    "ccf_peak": peak_m,
                    "ccf_peak_snr": peak_snr_m,
                    "gauss_ok": gauss_ok_m,
                    "template_key": "",
                    "telluric_fraction": np.nan,
                    "mask_line_count": np.nan,
                    "ccf_width": np.nan,
                    "ccf_asymmetry": np.nan,
                    "qc_pass": True,
                    "qc_reason": "pending",
                    "chunk_scatter_kms": np.nan,
                    "residual_to_exposure_kms": np.nan,
                }
            )
            w_ccf, asym_ccf = qc.ccf_shape_metrics(vels_m, ccf_m)
            qc_ok, qc_reason = qc.evaluate_chunk_qc(
                {
                    "rv_err_kms": err_m,
                    "mask_line_count": mask_line_count,
                    "telluric_fraction": tell_frac,
                    "ccf_asymmetry": asym_ccf,
                    "ccf_peak": peak_m,
                    "ccf_peak_snr": peak_snr_m,
                },
                {**qc_thresholds, "max_chunk_err_kms": min(qc_thresholds["max_chunk_err_kms"], args.max_chunk_err)},
            )
            diagnostics_rows[-1]["ccf_width"] = w_ccf
            diagnostics_rows[-1]["ccf_asymmetry"] = asym_ccf
            diagnostics_rows[-1]["qc_pass"] = qc_ok
            diagnostics_rows[-1]["qc_reason"] = qc_reason
            qc_ok_mask_chunk = bool(qc_ok)

            if plot_chunk_pngs and vels_m is not None and ccf_m is not None:
                plotting.plot_ccf(
                    vels_plot,
                    ccf_m,
                    plot_root / f"{stem}_chunk{chunk_key}_ccf_mask.png",
                    title=f"mask {chunk_key}",
                    peak_vel=float(rv_m) if np.isfinite(rv_m) else None,
                    gauss_params=gauss_plot_ccf,
                )

        # --- Template FFT ---
        rv_t, tpl_key = np.nan, ""
        rv_t_fft_raw = float("nan")
        tpl_key_plot = ""
        fft_chunk_rejected = False
        vel_ccf, ccf_arr = None, None
        fft_rv_seed = None
        if (
            bank_fft
            and (not use_fft_primary)
            and qc_ok_mask_chunk
            and np.isfinite(rv_m)
        ):
            # FFT raw RV is in the frame before pipeline subtracts b0; diagnostics rv_m is after -b0.
            fft_rv_seed = float(rv_m + b0)
        if bank_fft:
            try:
                _top_k = (
                    int(args.fft_coarse_top_k)
                    if getattr(args, "fft_coarse_top_k", None) is not None
                    else None
                )
                _fft_peak_pick = (
                    config.FFT_TEMPLATE_PEAK_PICK_HOT
                    if use_fft_primary
                    else config.FFT_TEMPLATE_PEAK_PICK_COOL
                )
                rv_t_fft_raw, tpl_key_plot, vel_ccf, ccf_arr = rv_core.estimate_rv_fft_with_ccf(
                    nw,
                    line_obs,
                    bank_fft,
                    vsini,
                    fft_two_phase=not bool(getattr(args, "no_fft_two_phase", False)),
                    fft_coarse_top_k=_top_k,
                    rv_seed_kms=fft_rv_seed,
                    rv_search_half_width_kms=float(config.TEMPLATE_FFT_MASK_SEED_HALF_WIDTH_KMS)
                    if fft_rv_seed is not None
                    else None,
                    fft_peak_pick=str(_fft_peak_pick),
                )
            except Exception as ex:
                logger.debug("fft RV chunk=%s: %s", chunk_key, ex)
                rv_t_fft_raw, tpl_key_plot = float("nan"), ""
                vel_ccf, ccf_arr = None, None
            rv_t = float(rv_t_fft_raw) - b0
            fft_chunk_rejected = bool(
                not np.isfinite(rv_t) or abs(float(rv_t)) > _tpl_fft_abs_cap
            )
            fft_ccf_qc_reason = "skipped"
            fft_ccf_rss_ratio = float("nan")
            fft_reject_ccf = False
            if (
                not fft_chunk_rejected
                and vel_ccf is not None
                and ccf_arr is not None
                and len(vel_ccf) == len(ccf_arr)
            ):
                max_rss = float(qc_thresholds.get("max_fft_ccf_flat_rss_ratio", 0.88))
                _ok_c, fft_ccf_qc_reason, _met = qc.fft_ccf_passes_vs_flat(
                    vel_ccf, ccf_arr, max_rss_ratio=max_rss
                )
                fft_ccf_rss_ratio = float(_met.get("fft_ccf_rss_ratio", float("nan")))
                if not _ok_c:
                    fft_chunk_rejected = True
                    fft_reject_ccf = True
                    logger.info(
                        "chunk=%s template_fft CCF QC: %s rss_ratio=%.4f",
                        chunk_key,
                        fft_ccf_qc_reason,
                        fft_ccf_rss_ratio,
                    )
            if fft_chunk_rejected:
                if fft_reject_ccf:
                    tpl_fft_qc_reason = str(fft_ccf_qc_reason)
                elif not np.isfinite(rv_t_fft_raw):
                    tpl_fft_qc_reason = "fft_nonfinite"
                elif not np.isfinite(rv_t):
                    tpl_fft_qc_reason = "bary_nonfinite"
                elif abs(float(rv_t)) > _tpl_fft_abs_cap:
                    tpl_fft_qc_reason = "abs_rv_cap"
                else:
                    tpl_fft_qc_reason = "rejected"
                tpl_fft_qc_pass = False
                rv_t, tpl_key = float("nan"), ""
            else:
                tpl_fft_qc_pass = True
                tpl_fft_qc_reason = (
                    "ok" if str(fft_ccf_qc_reason) in ("skipped", "") else str(fft_ccf_qc_reason)
                )
                tpl_key = tpl_key_plot if tpl_key_plot else ""
            if bank_fft and tpl_key_plot and np.isfinite(rv_t_fft_raw):
                logger.info(
                    "chunk=%s template_fft best_key=%s raw_rv_kms=%.2f (grid search over PHOENIX bank)",
                    chunk_key,
                    tpl_key_plot,
                    float(rv_t_fft_raw),
                )
            err_t = 10.0
            diagnostics_rows.append(
                {
                    "file": spectrum_file,
                    "chunk_key": chunk_key,
                    "mjd": mjd,
                    "teff": teff_diag,
                    "continuum_mode": args.continuum_mode,
                    "method": "template_fft",
                    "mask_name": "",
                    "rv_kms": rv_t,
                    "rv_err_kms": err_t,
                    "ccf_peak": np.nan,
                    "gauss_ok": False,
                    "template_key": str(tpl_key),
                    "fft_ccf_qc_reason": fft_ccf_qc_reason,
                    "fft_ccf_rss_ratio": fft_ccf_rss_ratio,
                    "qc_pass": tpl_fft_qc_pass,
                    "qc_reason": tpl_fft_qc_reason,
                }
            )

        # Primary stacked value (same rule as before)
        if use_fft_primary and bank_fft:
            rv_p, err_p = rv_t, 10.0
        elif mw is not None and not (nw[-1] < mw[0] or nw[0] > mw[-1]):
            rv_p, err_p = rv_m, err_m
        elif bank_fft:
            rv_p, err_p = rv_t, 10.0
        else:
            rv_p, err_p = np.nan, np.nan

        primary_method = "template" if use_fft_primary and bank_fft else "mask"
        qc_primary_ok = True
        qc_primary_reason = "ok"
        if primary_method == "mask" and len(diagnostics_rows) > 0 and diagnostics_rows[-1].get("method") == "mask_ccf":
            qc_primary_ok = bool(diagnostics_rows[-1].get("qc_pass", True))
            qc_primary_reason = str(diagnostics_rows[-1].get("qc_reason", "ok"))

        if np.isfinite(rv_p) and np.isfinite(err_p) and err_p <= args.max_chunk_err and qc_primary_ok:
            rv_results[chunk_key] = {"best_rv": rv_p, "best_rv_err": err_p}
            primary_rv_list.append(float(rv_p))
            primary_err_list.append(float(err_p))
            chunk_keys_plot.append(chunk_key)
            if use_fft_primary and bank_fft:
                primary_chunk_methods.append("template_fft")
            elif mw is not None and not (nw[-1] < mw[0] or nw[0] > mw[-1]):
                primary_chunk_methods.append("mask_ccf")
            elif bank_fft:
                primary_chunk_methods.append("template_fft")
            else:
                primary_chunk_methods.append("mask_ccf")
            logger.info(
                "chunk=%s primary_rv=%.3f err=%.3f method=%s mask=%s",
                chunk_key,
                rv_p,
                err_p,
                "template" if use_fft_primary and bank_fft else "mask",
                best_mask_name,
            )
        else:
            if np.isfinite(rv_p):
                logger.info("chunk=%s rejected rv=%.3f err=%.3f qc=%s", chunk_key, rv_p, err_p, qc_primary_reason)

        if plot_chunk_pngs and bank_fft and tpl_key_plot and np.isfinite(rv_t_fft_raw):
            tw, tf = bank_fft[tpl_key_plot]
            R = float(getattr(instrument, "resolving_power", 60_000.0))
            ser = rv_core.build_fft_match_plot_series(
                nw, line_obs, tw, tf, float(rv_t_fft_raw), R
            )
            rv_ord_plot = float(rv_p) if np.isfinite(rv_p) else None
            ccf_bundle = None
            if vel_ccf is not None and ccf_arr is not None and len(vel_ccf) == len(ccf_arr):
                ccf_bundle = rv_core.fit_fft_ccf_models(vel_ccf, ccf_arr)
            plotting.plot_fft_template_comparison(
                ser,
                str(chunk_key),
                str(tpl_key_plot),
                float(rv_t_fft_raw),
                rv_ord_plot,
                plot_root / f"{stem}_chunk{chunk_key}_fft_template.png",
                rejected=fft_chunk_rejected,
                ccf_bundle=ccf_bundle,
            )

        if plot_chunk_pngs and len(nw) > 10:
            plotting.plot_normalized_order(
                nw,
                nf,
                None,
                plot_root / f"{stem}_chunk{chunk_key}_norm.png",
                title=f"{chunk_key} norm ({args.continuum_mode})",
            )

    if not plots_only:
        io_utils.write_order_results(rv_results, spectrum_file)

    if plot_detail and (use_fft_primary or run_multi) and plots_strong_line_panels:
        balmer = {"Ha": 6562.8, "Hb": 4861.3, "Hg": 4340.5, "Hd": 4101.7}
        for o in valid_orders:
            w = np.array(spec_data[o]["wavelength"], float)
            f = np.array(spec_data[o]["flux"], float)
            e = np.array(spec_data[o]["eflux"], float)
            try:
                onw, onf, _one = continuum.fit_continuum(w, f, e, **_continuum_fit_kw(args, use_fft_primary))
                onw, onf, _one = continuum.despike_normalized_pre_ccf(onw, onf, np.ones_like(onf))
            except Exception:
                continue
            panels: list[dict] = []
            for name, rest in balmer.items():
                if onw[-1] < rest - 1.0 or onw[0] > rest + 1.0:
                    continue
                pan = rv_core.fit_balmer_line_all_methods(
                    onw, onf, rest, name, broad_lines=bool(use_fft_primary)
                )
                if pan is not None:
                    panels.append(pan)
            if not panels:
                continue
            rv_ord = None
            for ck, rec in rv_results.items():
                o_ck, _, _ = chunking.parse_chunk_key(ck)
                if o_ck is not None and int(o_ck) == int(o):
                    rv_ord = float(rec["best_rv"])
                    break
            plotting.plot_order_strong_line_panels(
                int(o),
                panels,
                rv_ord,
                plot_root / f"{stem}_order{o}_strong_lines.png",
                title_stem=stem,
            )

    diag_path = config.OUTPUT_DIR / f"{stem}_diagnostics.csv"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rv_arr = np.array(primary_rv_list, float)
    err_arr = np.array(primary_err_list, float)
    keys_arr = np.array(chunk_keys_plot, dtype=object)
    meth_arr = np.array(primary_chunk_methods, dtype=object)
    stack_pairs: set[tuple[str, str]] = set()
    fallback = False

    if len(rv_arr) == 0:
        logger.warning("No finite RVs for %s", spectrum_file)
        mean_rv, mean_err, rms = np.nan, np.nan, np.nan
    else:
        keep = _exposure_stack_keep_mask(rv_arr)
        if not np.all(keep):
            rv_arr = rv_arr[keep]
            err_arr = err_arr[keep]
            keys_arr = keys_arr[keep]
            meth_arr = meth_arr[keep]
        if len(rv_arr) == 0:
            mean_rv, mean_err, rms = np.nan, np.nan, np.nan
        else:
            stack_pairs = set(zip(keys_arr.tolist(), meth_arr.tolist()))
            wts = 1.0 / (err_arr**2 + 1e-9)
            mean_rv = float(np.average(rv_arr, weights=wts))
            mean_err = float(np.sqrt(1.0 / np.sum(wts)))
            rms = _weighted_rms(rv_arr, err_arr, mu_weighted=mean_rv)

            mask_only = len(meth_arr) > 0 and all(str(m) == "mask_ccf" for m in meth_arr.tolist())
            if (not use_fft_primary) and mask_only:
                if len(rv_arr) < int(config.MIN_MASK_CCF_CHUNKS_FOR_STACK):
                    mean_rv, mean_err, rms = np.nan, np.nan, np.nan
                    stack_pairs = set()
                elif len(rv_arr) >= 2:
                    mean_err = _mask_ccf_stack_error_inflated(
                        rv_arr, err_arr, mean_err, mu_weighted=mean_rv
                    )

            if plot_root and len(keys_arr) == len(rv_arr) and len(rv_arr) > 0:
                plotting.plot_chunk_rvs(
                    list(keys_arr),
                    rv_arr,
                    err_arr,
                    plot_root / f"{stem}_chunk_rvs.png",
                    title=stem,
                )

    if plot_root and order_mask_ccf and not plots_skip_order_summary:
        sorted_o = sorted(order_mask_ccf.keys())
        n_p = min(6, len(sorted_o))
        if n_p > 0:
            pick_idx = [int(round(x)) for x in np.linspace(0, len(sorted_o) - 1, n_p)]
            pick_o = [sorted_o[j] for j in pick_idx]
            panels = []
            for o in pick_o:
                rec = order_mask_ccf[o]
                panels.append(
                    (
                        rec["label"],
                        rec["vel"],
                        rec["ccf"],
                        rec["peak_vel"],
                        rec["gauss"],
                    )
                )
            plotting.plot_ccf_order_grid(
                panels,
                plot_root / f"{stem}_ccf_orders.png",
                title=f"{stem} mask CCF (sample orders)",
            )

    if plot_root and len(rv_arr) > 0 and len(keys_arr) == len(rv_arr) and not plots_skip_order_summary:
        ord_nums = np.array([chunking.parse_chunk_key(str(k))[1] for k in keys_arr], dtype=float)
        plotting.plot_rv_vs_order(
            ord_nums,
            rv_arr,
            err_arr,
            list(meth_arr),
            plot_root / f"{stem}_rv_vs_order.png",
            exposure_rv=float(mean_rv) if np.isfinite(mean_rv) else None,
            title=f"{stem} primary RV vs order",
        )

    hb_bundle: dict | None = None
    hb_order: int | None = None
    nw_hb: np.ndarray | None = None
    nf_hb: np.ndarray | None = None
    if use_fft_primary or run_multi or args.hb_rv_fallback:
        tw_hb, tf_hb = (None, None)
        if bank:
            tw_hb, tf_hb = next(iter(bank.values()))
        R_inst = float(getattr(instrument, "resolving_power", 60_000.0))
        hb_rest = 4861.3
        for o in valid_orders:
            w_raw = np.array(spec_data[o]["wavelength"], float)
            lo, hi = float(min(w_raw[0], w_raw[-1])), float(max(w_raw[0], w_raw[-1]))
            if not (lo <= hb_rest <= hi):
                continue
            f_raw = np.array(spec_data[o]["flux"], float)
            e_raw = np.array(spec_data[o]["eflux"], float)
            try:
                nw_h, nf_h, _neh = continuum.fit_continuum(
                    w_raw, f_raw, e_raw, **_continuum_fit_kw(args, use_fft_primary)
                )
                nw_h, nf_h, _neh = continuum.despike_normalized_pre_ccf(nw_h, nf_h, np.ones_like(nf_h))
            except Exception:
                continue
            hb_try = rv_core.measure_h_beta_rv(
                nw_h,
                nf_h,
                broad_lines=bool(use_fft_primary),
                tpl_wave=tw_hb,
                tpl_flux_norm=tf_hb,
                resolving_power=R_inst,
            )
            if hb_try is not None:
                hb_bundle = hb_try
                hb_order = int(o)
                nw_hb, nf_hb = nw_h, nf_h
                break

    if hb_bundle is not None and (run_multi or use_fft_primary or args.hb_rv_fallback):
        _hbp_json = _hb_joint_fit_params_json(hb_bundle)
        _rv_v_line = _rv_kms_from_hb_joint_line_center(hb_bundle)
        # Single third method: Voigt+Lorentz centroid on Hβ (more strong lines later).
        diagnostics_rows.append(
            {
                "file": spectrum_file,
                "chunk_key": "all",
                "mjd": mjd,
                "teff": teff_diag,
                "continuum_mode": args.continuum_mode,
                "method": "strong_lines",
                "mask_name": "",
                "rv_kms": float(_rv_v_line),
                "rv_err_kms": float(hb_bundle["err_voigt_kms"]),
                "ccf_peak": float(hb_bundle["template_ccf_peak"])
                if hb_bundle.get("template_ccf_peak") is not None
                else np.nan,
                "gauss_ok": np.isfinite(_rv_v_line),
                "template_key": "",
                "telluric_fraction": np.nan,
                "mask_line_count": np.nan,
                "ccf_width": np.nan,
                "ccf_asymmetry": np.nan,
                "qc_pass": True,
                "qc_reason": "pending",
                "chunk_scatter_kms": np.nan,
                "residual_to_exposure_kms": np.nan,
                "hb_joint_fit_params_json": _hbp_json,
                "strong_line_rest_angstrom": float(hb_rest),
            }
        )
        if (
            plot_detail
            and run_multi
            and plots_strong_line_panels
            and plot_root is not None
            and np.isfinite(float(_rv_v_line))
        ):
            plotting.plot_balmer_panels(
                spec_data, mjd, float(_rv_v_line), plot_root / f"{stem}_balmer.png"
            )
    if not run_multi:
        if (
            args.hb_rv_fallback
            and hb_bundle is not None
            and np.isfinite(float(hb_bundle["rv_best_kms"]))
        ):
            hb_rv = float(hb_bundle["rv_best_kms"])
            hb_err = float(hb_bundle["err_best_kms"])
            if not np.isfinite(hb_err) or hb_err <= 0:
                hb_err = 8.0
            if not np.isfinite(mean_rv) or (np.isfinite(rms) and rms > 100.0):
                mean_rv, mean_err, rms = hb_rv, hb_err, 0.0
                fallback = True
                stack_pairs = set()
                logger.warning("Hβ-only fallback RV=%.3f+/-%.3f", mean_rv, mean_err)

        if use_fft_primary and hb_bundle is not None:
            _hb_line_rv_adopt = _rv_kms_from_hb_joint_line_center(hb_bundle)
            if np.isfinite(float(_hb_line_rv_adopt)):
                mean_rv = float(_hb_line_rv_adopt)
                mean_err = float(hb_bundle["err_voigt_kms"])
                if not np.isfinite(mean_err) or mean_err <= 0:
                    mean_err = 5.0
                rms = 0.0
                logger.info(
                    "Hot star: adopted exposure RV = strong_lines (Hβ Voigt+Lorentz) %.3f +/- %.3f km/s",
                    mean_rv,
                    mean_err,
                )

    hb_fit_balmer = None
    if nw_hb is not None and nf_hb is not None:
        hb_fit_balmer = rv_core.fit_balmer_line_all_methods(
            nw_hb,
            nf_hb,
            4861.3,
            "Hb",
            broad_lines=bool(use_fft_primary),
        )
    wrote_hb_overlay = False
    if (
        plot_detail
        and run_multi
        and hb_order is not None
        and nw_hb is not None
        and nf_hb is not None
        and bank
        and mw is not None
        and ms is not None
        and hb_fit_balmer is not None
        and plot_root is not None
    ):
        rv_m_exp, err_m_exp = _weighted_method_rv_from_rows(diagnostics_rows, "mask_ccf")
        rv_t_exp, err_t_exp = _weighted_method_rv_from_rows(diagnostics_rows, "template_fft")
        tpl_k = _dominant_template_key(diagnostics_rows)
        tpl_pair = _bank_entry_for_template_key_str(bank, tpl_k)
        if tpl_pair is not None:
            tw_use, tf_use = tpl_pair
            plotting.plot_h_beta_order_method_overlay(
                nw_hb,
                nf_hb,
                np.asarray(mw, float),
                np.asarray(ms, float),
                rv_m_exp,
                err_m_exp,
                np.asarray(tw_use, float),
                np.asarray(tf_use, float),
                rv_t_exp,
                err_t_exp,
                hb_fit_balmer,
                plot_root / f"{stem}_h_beta_order_mask_template_voigt.png",
                title_stem=stem,
                resolving_power=float(getattr(instrument, "resolving_power", 60_000.0)),
                hb_measure=hb_bundle,
            )
            wrote_hb_overlay = True

    if (
        plot_root
        and hb_bundle is not None
        and (run_multi or use_fft_primary or args.hb_rv_fallback)
        and not wrote_hb_overlay
    ):
        rv_m_hb, err_m_hb = _weighted_method_rv_from_rows(diagnostics_rows, "mask_ccf")
        rv_t_hb, err_t_hb = _weighted_method_rv_from_rows(diagnostics_rows, "template_fft")
        tpl_leg = "Template FFT"
        if not np.isfinite(rv_t_hb) and hb_order is not None:
            rv_t_hb, err_t_hb = _weighted_template_fft_for_order(diagnostics_rows, int(hb_order))
            tpl_leg = "Template FFT (Hβ order)"
        if not np.isfinite(rv_t_hb):
            rv_ccf_hb = float(hb_bundle.get("rv_template_ccf_kms", np.nan))
            if np.isfinite(rv_ccf_hb):
                rv_t_hb = rv_ccf_hb
                err_t_hb = _template_rv_err_from_hb_ccf_peak(hb_bundle)
                tpl_leg = "Template (Hβ-window CCF)"
        tw_hb_plot, tf_hb_plot = None, None
        if bank:
            tpl_k_hb = _dominant_template_key(diagnostics_rows)
            tpl_pr_hb = _bank_entry_for_template_key_str(bank, tpl_k_hb)
            if tpl_pr_hb is not None:
                tw_hb_plot, tf_hb_plot = tpl_pr_hb
        R_hb_plot = float(getattr(instrument, "resolving_power", 60_000.0))
        plotting.plot_h_beta_rv_diagnostic(
            stem,
            hb_bundle,
            plot_root / f"{stem}_h_beta_rv.png",
            order_num=hb_order,
            rv_mask_kms=rv_m_hb,
            err_mask_kms=err_m_hb,
            mask_wave=np.asarray(mw, float) if mw is not None else None,
            mask_strength=np.asarray(ms, float) if ms is not None else None,
            tpl_wave=np.asarray(tw_hb_plot, float) if tw_hb_plot is not None else None,
            tpl_flux_norm=np.asarray(tf_hb_plot, float) if tf_hb_plot is not None else None,
            rv_template_kms=rv_t_hb,
            err_template_kms=err_t_hb,
            resolving_power=R_hb_plot,
            template_legend_name=tpl_leg,
        )

    if run_multi and diagnostics_rows:
        from . import method_evaluation as me

        fl = me.exposure_method_flags(diagnostics_rows)
        mo_arg = getattr(args, "method_offsets_file", None)
        mo_path = Path(mo_arg) if mo_arg is not None else None
        if mo_path is not None and not mo_path.is_file():
            logger.warning("Method offsets file not found: %s", mo_path)
            mo_path = None
        if mo_path is None:
            mo_path = config.METHOD_OFFSETS_FILE
        off_tbl: dict = {}
        if mo_path is not None and mo_path.is_file():
            off_tbl = io_utils.read_method_rv_offsets(mo_path, warn_if_missing=True)
        off_row = off_tbl.get(str(instrument.name)) if off_tbl else None
        fl_off = me.flags_with_method_offsets(fl, off_row)
        teff_d = float(teff_diag)
        snr_m = float(fl["median_mask_ccf_peak_snr"])
        l10 = float(np.log10(snr_m)) if np.isfinite(snr_m) and snr_m > 0 else float("nan")
        ad = me.recommend_adopted_rv(
            fl_off,
            teff=teff_d,
            log10_median_mask_ccf_peak_snr=l10,
        )
        mean_rv = float(ad["adopted_rv_kms"])
        mean_err = float(ad["adopted_err_kms"])
        if not np.isfinite(mean_rv) and args.hb_rv_fallback and hb_bundle is not None:
            hb_best = float(hb_bundle.get("rv_best_kms", np.nan))
            if np.isfinite(hb_best):
                mean_rv = hb_best
                mean_err = float(hb_bundle["err_best_kms"])
                if not np.isfinite(mean_err) or mean_err <= 0:
                    mean_err = 8.0
                rms = 0.0
                fallback = True
                stack_pairs = set()
                logger.warning(
                    "Hβ-only fallback RV=%.3f+/-%.3f km/s (cascade returned no adopted RV)",
                    mean_rv,
                    mean_err,
                )
        elif np.isfinite(mean_rv):
            am = ad.get("adopted_method", "")
            logger.info(
                "Adopted exposure RV (cascade) method=%s rv=%.4f+/-%.4f km/s",
                am,
                mean_rv,
                mean_err,
            )

    # Fill residual/scatter diagnostics + exposure stack metadata
    if diagnostics_rows:
        chunk_scatter = (
            _weighted_rms(rv_arr, err_arr, mu_weighted=float(mean_rv))
            if len(rv_arr) > 1
            else (0.0 if len(rv_arr) == 1 else np.nan)
        )
        for row in diagnostics_rows:
            ck = str(row.get("chunk_key", ""))
            meth = str(row.get("method", ""))
            row["used_in_exposure_stack"] = (ck, meth) in stack_pairs
            row["exposure_rv_kms"] = float(mean_rv) if np.isfinite(mean_rv) else np.nan
            row["exposure_rv_err_kms"] = float(mean_err) if np.isfinite(mean_err) else np.nan
            row["chunk_scatter_kms"] = chunk_scatter if np.isfinite(chunk_scatter) else np.nan
            rv_row = row.get("rv_kms", np.nan)
            row["residual_to_exposure_kms"] = float(rv_row - mean_rv) if np.isfinite(rv_row) and np.isfinite(mean_rv) else np.nan

    # Write per-spectrum diagnostics CSV
    if diagnostics_rows and not plots_only:
        pd.DataFrame(diagnostics_rows).to_csv(diag_path, index=False)

    logger.info(
        "SUMMARY file=%s mjd=%.6f rv=%.4f+/-%.4f rms=%.4f fallback=%s",
        spectrum_file,
        mjd,
        mean_rv if np.isfinite(mean_rv) else float("nan"),
        mean_err if np.isfinite(mean_err) else float("nan"),
        rms if np.isfinite(rms) else float("nan"),
        fallback,
    )

    return {
        "file": spectrum_file,
        "mjd": mjd,
        "rv": mean_rv,
        "rv_err": mean_err,
        "rv_rms": rms,
        "fallback": fallback,
    }


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Dark Hunter echelle RV pipeline")
    parser.add_argument("input_file", nargs="+", help="Spectra (txt, GHOST blue FITS, MAROON-X h5)")
    parser.add_argument("--instrument", default="APF")
    parser.add_argument("--teff", type=float, default=config.DEFAULT_TEFF)
    parser.add_argument(
        "--logg",
        type=float,
        default=None,
        help="PHOENIX bank anchor log g (default: Gaia logg_gspphot if present, else 4.5)",
    )
    parser.add_argument(
        "--mh",
        type=float,
        default=None,
        help="PHOENIX bank anchor [M/H] (default: Gaia mh_gspphot if present, else 0)",
    )
    parser.add_argument(
        "--template-grid-wide",
        action="store_true",
        help="Widen PHOENIX grid (log g, [M/H], more atmospheres and vsini samples) for template FFT",
    )
    parser.add_argument(
        "--no-fft-two-phase",
        action="store_true",
        help="Scan the full PHOENIX bank every chunk (disable coarse Teff/logg/[M/H] then refine vsini pass).",
    )
    parser.add_argument(
        "--fft-coarse-top-k",
        type=int,
        default=None,
        help="After coarse FFT, keep this many atmosphere triples for the full vsini search (default: config).",
    )
    parser.add_argument(
        "--hb-rv-fallback",
        action="store_true",
        help="If primary exposure RV is unusable, use Hβ-only RV as fallback (opt-in)",
    )
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--force-gaia", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--plots", action="store_true", help="Write diagnostic figures")
    parser.add_argument(
        "--plots-minimal",
        "--plots-focus",
        action="store_true",
        dest="plots_focus",
        help="With --plots: only chunk_rvs, rv_vs_order, ccf_orders, and strong_lines Hβ diagnostic PNG (skip per-order norm/CCF/FFT/etc.)",
    )
    parser.add_argument(
        "--plots-skip-chunk-pngs",
        action="store_true",
        help=(
            "With --plots (not --plots-focus): skip per-chunk figures "
            "(*_chunk*_ccf_mask.png, *_chunk*_fft_template.png, *_chunk*_norm.png). "
            "Exposure-level plots (tournament, chunk_rvs, rv_vs_order, ccf_orders sample, Hβ, strong-line panels, …) "
            "are still written. For outliers only, omit this flag on a second run over fewer files, or use "
            "--plots-chunk-detail-stems-file."
        ),
    )
    parser.add_argument(
        "--plots-chunk-detail-stems-file",
        type=Path,
        default=None,
        help=(
            "Optional text file: one spectrum stem per line (basename without .txt, no glob). "
            "With --plots-skip-chunk-pngs, still write per-chunk PNGs for stems listed here (single pass)."
        ),
    )
    parser.add_argument(
        "--plots-skip-order-summary",
        action="store_true",
        help=(
            "With --plots: skip echelle-order summary figures (*_ccf_orders.png sample grid and "
            "*_rv_vs_order.png). Keeps exposure-level chunk_rvs, mask tournament, Hβ diagnostic, etc."
        ),
    )
    parser.add_argument(
        "--plots-strong-line-panels",
        action="store_true",
        help=(
            "With --plots: write per-order strong-line panel PNGs and Balmer 2×2 from the strong-line path. "
            "Default is off (large file count)."
        ),
    )
    parser.add_argument("--plot-dir", default=None, help="Override plot directory")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "With --plots (implied if omitted): skip per-spectrum *_orders.txt, *_diagnostics.csv, and "
            "tournament CSV; still write Gaia_DR3_*_summary.txt ([PIPELINE RESULTS] merge) when source IDs "
            "are known. Full processing still runs so figures match current code and inputs."
        ),
    )
    parser.add_argument("--continuum-mode", choices=["spline", "blaze"], default="spline")
    parser.add_argument("--subchunks", type=int, default=1, help="Split each order into N pixel chunks")
    parser.add_argument(
        "--chunk-layout",
        type=Path,
        default=None,
        help="YAML chunk layout (overrides --subchunks when set). See validation/chunk_layout.py.",
    )
    parser.add_argument("--max-chunk-err", type=float, default=50.0, help="Skip chunk RVs with err > this (km/s)")
    parser.add_argument("--qc-config", default="order_chunk_qc.yaml", help="QC threshold YAML")
    parser.add_argument("--write-qc-config", action="store_true", help="Write default QC config if missing")
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help=(
            "Calibration mode: measure stellar mask CCF RVs per chunk only (no PHOENIX template bank, "
            "no template_fft / strong_lines). Use for per-order bias training sets; pair with --no-bias "
            "so biases are not applied while building bias_statistics.txt."
        ),
    )
    parser.add_argument(
        "--no-run-all-methods",
        action="store_true",
        help=(
            "Legacy: do not run mask+template+strong_lines diagnostics on every spectrum. "
            "Default is multi-method on; adopted RV then uses cool/hot chunk stack + hot-star strong-line rules."
        ),
    )
    parser.add_argument(
        "--run-all-methods",
        "--compare-rv-methods",
        action="store_true",
        dest="run_all_methods_explicit",
        help="No-op if set: multi-method diagnostics are already the default.",
    )
    parser.add_argument(
        "--method-offsets-file",
        type=Path,
        default=None,
        help=(
            "Optional method_rv_offsets.txt from validation.compute_method_rv_offsets "
            "(template/strong RV shifts vs mask). Default: config.METHOD_OFFSETS_FILE or repo method_rv_offsets.txt if present."
        ),
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Skip spectra whose output diagnostics CSV exists and is newer than the input (for cron).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --update: reprocess even when outputs look up to date.",
    )
    parser.add_argument(
        "--no-fixed-exposure-template",
        action="store_true",
        help=(
            "Allow template_fft to pick a different PHOENIX key per chunk (legacy). "
            "Default: one exposure-voted (Teff, log g, [M/H], vsini) key for all chunks."
        ),
    )

    args = parser.parse_args(argv)
    args.run_all_methods = not args.no_run_all_methods
    if getattr(args, "run_all_methods_explicit", False):
        args.run_all_methods = True
    if args.mask_only:
        args.run_all_methods = False
    if args.plots_only:
        args.plots = True
    setup_logging(args.log_level, args.quiet)
    if args.write_qc_config:
        qc.ensure_qc_config(Path(args.qc_config))

    try:
        inst = instruments.get_instrument_profile(args.instrument)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(2)

    _resolve_chunk_layout(args)
    if args.chunk_layout:
        logger.info("Chunk layout: %s", args.chunk_layout)
    if not args.no_bias:
        bias_path = Path(inst.bias_file) if inst.bias_file else None
        if bias_path and bias_path.is_file():
            logger.info("Bias file: %s", bias_path)
        else:
            logger.warning(
                "Bias file missing (%s); per-order debias disabled unless --no-bias set explicitly",
                bias_path,
            )

    plot_root = None
    if args.plots:
        plot_root = Path(args.plot_dir) if args.plot_dir else config.PLOT_DIR
        plot_root.mkdir(parents=True, exist_ok=True)
        logger.info("Plots -> %s", plot_root)

    from . import gaia_utils

    gaia_cache: dict[int, object] = {}
    results_by_gid: dict[int, list[dict]] = defaultdict(list)
    unassigned_results: list[dict] = []

    def _skip_update(inp: str) -> bool:
        if not args.update or args.force:
            return False
        stem = Path(inp).stem
        diag = config.OUTPUT_DIR / f"{stem}_diagnostics.csv"
        if not diag.is_file():
            return False
        try:
            return float(diag.stat().st_mtime) >= float(Path(inp).stat().st_mtime)
        except OSError:
            return False

    for fn in args.input_file:
        fn_path = Path(str(fn))
        if (
            fn_path.suffix.lower() == ".txt"
            and "_epoch_" in fn_path.name
            and not is_primary_epoch_spectrum_name(fn_path.name)
        ):
            logger.info("Skipping per-order spectrum extract: %s", fn)
            continue
        if _skip_update(str(fn)):
            logger.info("Skipping (up to date): %s", fn)
            continue
        args_f = copy.copy(args)
        m_g = re.search(r"Gaia_DR3_(\d{18,19})", str(fn))
        gid: int | None = int(m_g.group(1)) if m_g else None

        if gid is not None:
            if gid not in gaia_cache:
                out_sum = config.OUTPUT_DIR / f"Gaia_DR3_{gid}_summary.txt"
                gaia_cache[gid] = gaia_utils.resolve_gaia_data(gid, out_sum, args.force_gaia)
            _apply_gaia_metadata_to_args(args_f, gaia_cache[gid])
            _attach_diagnostics_teff(args_f, gid, gaia_cache[gid])
            io_utils.write_star_summary(gid, gaia_cache.get(gid), results_by_gid[gid])
            td = float(getattr(args_f, "_diagnostics_teff", float("nan")))
            logger.info(
                "Gaia source %s diagnostics_Teff=%s pipeline_teff=%.0f for %s",
                gid,
                f"{td:.0f}" if np.isfinite(td) else "NaN",
                float(args_f.teff),
                fn,
            )

        if args_f.logg is None:
            args_f.logg = 4.5
        if args_f.mh is None:
            args_f.mh = 0.0

        res = process_spectrum(fn, args_f, inst, plot_root)
        if res:
            if gid is not None:
                results_by_gid[gid].append(res)
                io_utils.write_star_summary(gid, gaia_cache.get(gid), results_by_gid[gid])
            else:
                unassigned_results.append(res)

    if not args.plots_only:
        if unassigned_results:
            io_utils.write_summary(unassigned_results)
        elif not results_by_gid:
            io_utils.write_summary([])


if __name__ == "__main__":
    main()
