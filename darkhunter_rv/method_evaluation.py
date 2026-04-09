# method_evaluation.py
"""Exposure-level method validity, overlap, and adopted-RV recommendation (binary-safe, no true RV)."""
from __future__ import annotations

from typing import Any

import numpy as np

from . import config as dh_config
from .method_regions import (
    region_mask_applicable,
    region_strong_lines_applicable,
    region_template_applicable,
)
from .pipeline import _weighted_method_rv_from_rows

def count_mask_chunks_qc_pass(rows: list[dict]) -> int:
    return sum(
        1
        for r in rows
        if str(r.get("method", "")) == "mask_ccf"
        and str(r.get("chunk_key", "")) != "all"
        and bool(r.get("qc_pass", True))
    )


def count_template_chunks_qc_pass(rows: list[dict]) -> int:
    return sum(
        1
        for r in rows
        if str(r.get("method", "")) == "template_fft"
        and str(r.get("chunk_key", "")) != "all"
        and bool(r.get("qc_pass", True))
    )


def median_mask_ccf_peak_snr(rows: list[dict]) -> float:
    xs: list[float] = []
    for r in rows:
        if str(r.get("method", "")) != "mask_ccf" or str(r.get("chunk_key", "")) == "all":
            continue
        v = r.get("ccf_peak_snr", np.nan)
        try:
            xf = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(xf):
            xs.append(xf)
    return float(np.median(np.asarray(xs, float))) if xs else float("nan")


def _row_method_all(rows: list[dict], method: str) -> dict[str, Any] | None:
    for r in rows:
        if str(r.get("method", "")) == method and str(r.get("chunk_key", "")) == "all":
            return r
    return None


def _rv_err_from_all_row(rows: list[dict], method: str) -> tuple[float, float]:
    """Exposure-level RV/err for single-row methods; requires qc_pass (pipeline does not filter these in weighted)."""
    r = _row_method_all(rows, method)
    if r is None or not bool(r.get("qc_pass", True)):
        return float("nan"), float("nan")
    rv = float(r.get("rv_kms", np.nan))
    er = float(r.get("rv_err_kms", np.nan))
    if not np.isfinite(rv) or not np.isfinite(er) or er <= 0 or er > 1e27:
        return float("nan"), float("nan")
    return rv, er


def exposure_method_flags(rows: list[dict]) -> dict[str, Any]:
    """
    Per-exposure diagnostics: weighted RVs (same rules as pipeline reports) and validity flags.

    ``valid`` for mask/template means :func:`_weighted_method_rv_from_rows` returned finite values
    (includes min chunk count and QC filtering for mask/template).

    ``strong_lines`` is the Voigt+Lorentz centroid row (currently Hβ only).
    """
    rv_m, er_m = _weighted_method_rv_from_rows(rows, "mask_ccf")
    rv_t, er_t = _weighted_method_rv_from_rows(rows, "template_fft")
    rv_sl, er_sl = _rv_err_from_all_row(rows, "strong_lines")

    mask_valid = np.isfinite(rv_m) and np.isfinite(er_m) and er_m > 0
    tpl_valid = np.isfinite(rv_t) and np.isfinite(er_t) and er_t > 0
    sl_valid = np.isfinite(rv_sl) and np.isfinite(er_sl)

    n_ok = int(mask_valid) + int(tpl_valid) + int(sl_valid)

    return {
        "n_mask_chunks_qc_pass": count_mask_chunks_qc_pass(rows),
        "n_template_chunks_qc_pass": count_template_chunks_qc_pass(rows),
        "median_mask_ccf_peak_snr": median_mask_ccf_peak_snr(rows),
        "mask_valid": mask_valid,
        "template_valid": tpl_valid,
        "strong_lines_valid": sl_valid,
        "n_methods_valid": n_ok,
        "overlap_2plus": n_ok >= 2,
        "mask_rv_kms": rv_m,
        "mask_err_kms": er_m,
        "template_rv_kms": rv_t,
        "template_err_kms": er_t,
        "strong_lines_rv_kms": rv_sl,
        "strong_lines_err_kms": er_sl,
    }


def flags_with_method_offsets(
    flags: dict[str, Any],
    instrument_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Shallow copy of ``flags`` with template/strong RVs shifted by calibration offsets (mask = truth).

    ``instrument_row`` is one entry from :func:`darkhunter_rv.io_utils.read_method_rv_offsets`.
    """
    out = dict(flags)
    if not instrument_row:
        return out
    ot = instrument_row.get("offset_template_fft_kms")
    osl = instrument_row.get("offset_strong_lines_kms")
    if ot is not None and bool(out.get("template_valid")):
        out["template_rv_kms"] = float(out["template_rv_kms"]) + float(ot)
    if osl is not None and bool(out.get("strong_lines_valid")):
        out["strong_lines_rv_kms"] = float(out["strong_lines_rv_kms"]) + float(osl)
    return out


def _log10_median_mask_snr(flags: dict[str, Any]) -> float:
    snr = float(flags.get("median_mask_ccf_peak_snr", float("nan")))
    if np.isfinite(snr) and snr > 0:
        return float(np.log10(snr))
    return float("nan")


def recommend_adopted_rv(
    flags: dict[str, Any],
    *,
    teff: float | None = None,
    log10_median_mask_ccf_peak_snr: float | None = None,
    max_sigma_kms: float | None = None,
) -> dict[str, Any]:
    """
    Cascade adoption: mask_ccf → template_fft → strong_lines (stellar mask preferred).

    For each method in order, require **region applicability** (``config.METHOD_REGION_*`` +
    :mod:`darkhunter_rv.method_regions`) and stack **validity** (finite RV/σ, QC rules in
    :func:`exposure_method_flags`). Use the first method with σ ≤ ``max_sigma_kms`` (default
    ``config.ADOPTED_CASCADE_MAX_SIGMA_KMS``). If none meet the σ cut, use the first applicable
    valid method in order and keep its (possibly large) σ.

    Pass ``teff`` (K) and ``log10_median_mask_ccf_peak_snr`` when known (pipeline and overlap
    reports). If omitted, log10 S/N is taken from ``flags``; ``teff`` defaults to NaN (regions
    then exclude most methods).
    """
    t = float(teff) if teff is not None else float("nan")
    s10 = (
        float(log10_median_mask_ccf_peak_snr)
        if log10_median_mask_ccf_peak_snr is not None
        else _log10_median_mask_snr(flags)
    )
    sig_cap = (
        float(max_sigma_kms)
        if max_sigma_kms is not None
        else float(dh_config.ADOPTED_CASCADE_MAX_SIGMA_KMS)
    )

    t_arr = np.array([t], float)
    s_arr = np.array([s10], float)
    reg_m = bool(region_mask_applicable(t_arr, s_arr)[0])
    reg_t = bool(region_template_applicable(t_arr, s_arr)[0])
    reg_sl = bool(region_strong_lines_applicable(t_arr, s_arr)[0])

    steps: list[tuple[str, bool, bool, str, str]] = [
        ("mask_ccf", reg_m, bool(flags.get("mask_valid")), "mask_rv_kms", "mask_err_kms"),
        (
            "template_fft",
            reg_t,
            bool(flags.get("template_valid")),
            "template_rv_kms",
            "template_err_kms",
        ),
        (
            "strong_lines",
            reg_sl,
            bool(flags.get("strong_lines_valid")),
            "strong_lines_rv_kms",
            "strong_lines_err_kms",
        ),
    ]

    loose: tuple[str, float, float] | None = None
    for meth, reg_ok, ok, rv_k, er_k in steps:
        if not (reg_ok and ok):
            continue
        rv = float(flags[rv_k])
        err = float(flags[er_k])
        if not np.isfinite(rv) or not np.isfinite(err) or err <= 0:
            continue
        if loose is None:
            loose = (meth, rv, err)
        if err <= sig_cap:
            return {"adopted_method": meth, "adopted_rv_kms": rv, "adopted_err_kms": err}

    if loose is not None:
        m, rv, err = loose
        return {"adopted_method": m, "adopted_rv_kms": rv, "adopted_err_kms": err}

    return {
        "adopted_method": "",
        "adopted_rv_kms": float("nan"),
        "adopted_err_kms": float("nan"),
    }
