#!/usr/bin/env python3
"""Fit APF RV summaries with a simple 1-companion Keplerian model.

Reads per-star summaries written by ``darkhunter_rv.pipeline`` (flat or nested under
``output/Gaia_DR3_<source_id>/``). RV epochs come from ``[PIPELINE RESULTS]``; NSS
period/eccentricity priors (``--use-gaia-nss``) come from ``[GAIA METADATA]`` in the
same file — no Gaia archive queries by default.

Default behavior:
- Find the newest *_summary.txt under --output-dir (recursive)
- Fit Keplerian RV curve
- Save plot + JSON report in ./rv_fit_reports

Optional:
- --all: fit all summary files under --output-dir
- --summary <file>: fit one specific summary file
- --query-gaia-online: fall back to astroquery Gaia NSS tables (legacy)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.timeseries import LombScargle
from astropy.time import Time

from darkhunter_rv.summary_paths import (
    count_pipeline_rows,
    discover_summary_files,
    discover_summary_path,
    parse_object_id_from_summary,
)
from darkhunter_rv.thiele_innes_inclination import fill_inclination_in_metadata, inclination_from_row_dict
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker, HPacker


@dataclass
class RVPoint:
    mjd: float
    rv: float
    rv_err: float
    rms: float
    file: str
    telescope: str = "APF"
    is_literature: bool = False


def _parse_gaia_metadata(path: Path) -> Optional[Dict[str, Any]]:
    try:
        from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary

        return parse_gaia_metadata_from_star_summary(path)
    except Exception:
        return None


def _meta_float(meta: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key not in meta:
            continue
        try:
            val = float(meta[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            return val
    return None


def load_nss_priors_from_summary(path: Path) -> Optional[Dict[str, float]]:
    """NSS period/eccentricity (and optional masses) from [GAIA METADATA] — no network."""
    meta = _parse_gaia_metadata(path)
    if not meta:
        return None

    nss_type = meta.get("NSS_Solution_Type", meta.get("nss_solution_type"))
    if isinstance(nss_type, str) and nss_type.strip().lower() in ("none", "", "nan"):
        nss_type = None

    period = _meta_float(meta, "Period", "period", "period_days")
    ecc = _meta_float(meta, "Eccentricity", "eccentricity", "ecc")
    if period is None or ecc is None:
        return None
    if period <= 0 or ecc < 0 or ecc >= 1:
        return None

    out: Dict[str, float] = {"period_days": float(period), "eccentricity": float(ecc)}
    if nss_type is not None and str(nss_type).strip():
        out["nss_solution_type"] = str(nss_type).strip()

    m1 = _meta_float(meta, "M1", "m1", "m1_msun", "Mass_Primary", "mass_primary")
    m2 = _meta_float(meta, "M2", "m2", "m2_msun", "Mass_Secondary", "mass_secondary")
    incl = _meta_float(meta, "Inclination", "inclination", "Inclination_Deg")
    if m1 is not None and m1 > 0:
        out["m1_msun"] = m1
    if m2 is not None and m2 > 0:
        out["m2_msun"] = m2
    if incl is not None and np.isfinite(incl):
        out["inclination_deg"] = float(incl)
    return out


def load_mass_priors_from_summary(path: Path) -> Dict[str, float]:
    """M1 / M2 / inclination from [GAIA METADATA] (no NSS period requirement)."""
    meta = _parse_gaia_metadata(path)
    if not meta:
        return {}
    fill_inclination_in_metadata(meta)
    out: Dict[str, float] = {}
    m1 = _meta_float(meta, "M1", "m1", "m1_msun", "Mass_Primary", "mass_primary")
    m2 = _meta_float(meta, "M2", "m2", "m2_msun", "Mass_Secondary", "mass_secondary")
    incl = _meta_float(meta, "Inclination", "inclination", "Inclination_Deg")
    if m1 is not None and m1 > 0:
        out["m1_msun"] = float(m1)
    if m2 is not None and m2 > 0:
        out["m2_msun"] = float(m2)
    if incl is not None and np.isfinite(incl):
        out["inclination_deg"] = float(incl)
    return out


def parse_m1_from_summary(path: Path) -> Optional[float]:
    m1 = load_mass_priors_from_summary(path).get("m1_msun")
    return float(m1) if m1 is not None and m1 > 0 else None


def _free_fit_variant_report(report: dict) -> Optional[dict]:
    variants = report.get("fit_variants")
    if isinstance(variants, dict):
        free = variants.get("free")
        if isinstance(free, dict) and free.get("P_days") is not None:
            return free
    if report.get("P_days") is not None and report.get("K_kms") is not None:
        return report
    return None


def lookup_fit_report_by_gaia_id(reports: Dict[str, dict], gaia_id: str) -> Optional[dict]:
    """Match fit JSON keyed by numeric id or ``Gaia_DR3_<id>`` stem."""
    sid = (gaia_id or "").strip()
    if not sid:
        return None
    if sid in reports:
        return reports[sid]
    prefixed = f"Gaia_DR3_{sid}"
    if prefixed in reports:
        return reports[prefixed]
    return None


def _load_json_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def load_observability_window(source_id: Optional[str], cache_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if source_id is None or cache_path is None or (not cache_path.exists()):
        return None
    data = _load_json_cache(cache_path)
    row = data.get(str(source_id))
    if not isinstance(row, dict):
        return None
    s = row.get("next_window_start_date")
    e = row.get("next_window_end_date")
    windows = row.get("windows")
    circumpolar = bool(row.get("circumpolar", False))
    if isinstance(windows, list) and windows:
        out: Dict[str, Any] = {
            "windows": windows,
            "circumpolar": circumpolar,
        }
        if isinstance(s, str) and s and isinstance(e, str) and e:
            out["start_date"] = s
            out["end_date"] = e
        else:
            w0 = windows[0]
            if isinstance(w0, dict):
                out["start_date"] = w0.get("start_date", "")
                out["end_date"] = windows[-1].get("end_date", "") if isinstance(windows[-1], dict) else ""
        return out if out.get("start_date") and out.get("end_date") else None
    if not isinstance(s, str) or not isinstance(e, str) or (not s) or (not e):
        return None
    return {
        "start_date": s,
        "end_date": e,
        "circumpolar": circumpolar,
        "windows": [{"start_date": s, "end_date": e}],
    }


def _save_json_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _first_finite(row: Any, keys: List[str]) -> Optional[float]:
    for key in keys:
        try:
            val = row[key]
        except Exception:
            continue
        if val is None:
            continue
        try:
            f = float(val)
        except Exception:
            continue
        if np.isfinite(f):
            return f
    return None


def _extract_m1_from_nss_row(row: Any) -> Optional[float]:
    # Common NSS/derived column aliases seen across releases/solutions.
    m1 = _first_finite(
        row,
        [
            "m1_msun",
            "mass_primary",
            "primary_mass",
            "primary_mass_msun",
            "m1",
            "mass1",
            "mass_1",
        ],
    )
    if m1 is not None and np.isfinite(m1) and m1 > 0:
        return float(m1)

    # Fallback: inspect all columns and pick primary-like mass fields.
    names: List[str] = []
    try:
        names = list(getattr(row, "colnames", []) or [])
    except Exception:
        names = []
    if not names:
        try:
            names = list(getattr(getattr(row, "dtype", None), "names", []) or [])
        except Exception:
            names = []
    cand = []
    for n in names:
        ln = str(n).lower()
        if "mass" in ln and ("prim" in ln or "m1" in ln or ln.endswith("1") or "_1" in ln):
            cand.append(str(n))
    m1 = _first_finite(row, cand)
    if m1 is not None and np.isfinite(m1) and m1 > 0:
        return float(m1)
    return None


def _gaia_table_row_as_dict(row: Any) -> Dict[str, Any]:
    try:
        names = list(getattr(row, "colnames", []) or [])
        return {str(n): row[n] for n in names}
    except Exception:
        pass
    if isinstance(row, dict):
        return dict(row)
    return {}


def _resolve_orbit_inclination_deg(row: Any, cached_incl: Optional[float] = None) -> tuple[Optional[float], Optional[float]]:
    """Database inclination when present; otherwise derive from Thiele-Innes (Halbwachs App. A)."""
    incl_o = cached_incl
    if incl_o is None:
        incl_o = _first_finite(row, ["inclination", "incl"])
    if incl_o is not None and np.isfinite(incl_o) and 0.05 < float(incl_o) < 179.95:
        return float(incl_o), None
    return inclination_from_row_dict(_gaia_table_row_as_dict(row))


def fetch_gaia_nss_orbit(source_id: str, cache_path: Optional[Path] = None) -> Optional[Dict[str, float]]:
    cache: Dict[str, Any] = {}
    if cache_path is not None:
        cache = _load_json_cache(cache_path)
        if source_id in cache:
            val = cache[source_id]
            if isinstance(val, dict) and val.get("_none"):
                return None
            # If cache already has M1, trust it. Otherwise re-query to enrich/update.
            if isinstance(val, dict) and val.get("m1_msun") is not None:
                return val
            # For cached None or dict without masses, continue and re-query.

    try:
        from astroquery.gaia import Gaia  # type: ignore
    except Exception:
        return None

    try:
        orbit_query = f"""
        SELECT source_id, nss_solution_type, period, eccentricity, inclination,
               a_thiele_innes, a_thiele_innes_error,
               b_thiele_innes, b_thiele_innes_error,
               f_thiele_innes, f_thiele_innes_error,
               g_thiele_innes, g_thiele_innes_error
        FROM gaiadr3.nss_two_body_orbit
        WHERE source_id = '{source_id}'
        """
        orbit_results = Gaia.launch_job(orbit_query).get_results()

        mass_query = f"""
        SELECT source_id, m1, m2
        FROM gaiadr3.binary_masses
        WHERE source_id = '{source_id}'
        """
        mass_results = Gaia.launch_job(mass_query).get_results()

        m1 = None
        m2 = None
        for row in mass_results:
            m1_try = _first_finite(row, ["m1"])
            m2_try = _first_finite(row, ["m2"])
            if m1 is None and m1_try is not None and m1_try > 0:
                m1 = float(m1_try)
            if m2 is None and m2_try is not None and m2_try > 0:
                m2 = float(m2_try)
            if m1 is not None and m2 is not None:
                break

        out = None
        for row in orbit_results:
            p = _first_finite(row, ["period", "p_orb"])
            e = _first_finite(row, ["eccentricity", "ecc"])
            if p is None or e is None:
                continue
            if not np.isfinite(p) or not np.isfinite(e):
                continue
            if p <= 0 or e < 0 or e >= 1:
                continue
            out = {"period_days": float(p), "eccentricity": float(e)}
            incl_o, incl_err = _resolve_orbit_inclination_deg(row)
            if incl_o is not None and np.isfinite(incl_o):
                out["inclination_deg"] = float(incl_o)
            if incl_err is not None and np.isfinite(incl_err) and incl_err > 0:
                out["inclination_deg_error"] = float(incl_err)
            if m1 is not None:
                out["m1_msun"] = m1
            if m2 is not None:
                out["m2_msun"] = m2
            break
        if cache_path is not None:
            cache[source_id] = out
            _save_json_cache(cache_path, cache)
        return out
    except Exception:
        if cache_path is not None:
            cache[source_id] = {"_none": True, "mass_checked": True}
            _save_json_cache(cache_path, cache)
        return None


def _chunk(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def prefetch_gaia_nss_bulk(source_ids: List[str], cache_path: Path, chunk_size: int = 150) -> None:
    uniq = sorted({str(s) for s in source_ids if s})
    if not uniq:
        return
    cache = _load_json_cache(cache_path)
    to_fetch: List[str] = []
    for sid in uniq:
        v = cache.get(sid)
        if isinstance(v, dict) and (v.get("_none") or v.get("mass_checked")):
            continue
        if isinstance(v, dict) and v.get("m1_msun") is not None:
            continue
        to_fetch.append(sid)
    if not to_fetch:
        return

    try:
        from astroquery.gaia import Gaia  # type: ignore
    except Exception:
        return

    t0 = time.time()
    fetched = 0
    for ids in _chunk(to_fetch, chunk_size):
        id_clause = ",".join(ids)
        orbit_q = (
            "SELECT source_id, period, eccentricity, inclination, "
            "a_thiele_innes, a_thiele_innes_error, "
            "b_thiele_innes, b_thiele_innes_error, "
            "f_thiele_innes, f_thiele_innes_error, "
            "g_thiele_innes, g_thiele_innes_error "
            "FROM gaiadr3.nss_two_body_orbit "
            f"WHERE source_id IN ({id_clause})"
        )
        mass_q = (
            "SELECT source_id, m1, m2 "
            "FROM gaiadr3.binary_masses "
            f"WHERE source_id IN ({id_clause})"
        )
        try:
            orbit_res = Gaia.launch_job(orbit_q).get_results()
            mass_res = Gaia.launch_job(mass_q).get_results()
        except Exception:
            continue

        orbit_by: Dict[str, Dict[str, float]] = {}
        for row in orbit_res:
            try:
                sid = str(int(row["source_id"]))
            except Exception:
                continue
            if sid in orbit_by:
                continue
            p = _first_finite(row, ["period"])
            e = _first_finite(row, ["eccentricity"])
            if p is None or e is None:
                continue
            if not np.isfinite(p) or not np.isfinite(e) or p <= 0 or e < 0 or e >= 1:
                continue
            entry: Dict[str, float] = {"period_days": float(p), "eccentricity": float(e)}
            incl_o, incl_err = _resolve_orbit_inclination_deg(row)
            if incl_o is not None and np.isfinite(incl_o):
                entry["inclination_deg"] = float(incl_o)
            if incl_err is not None and np.isfinite(incl_err) and incl_err > 0:
                entry["inclination_deg_error"] = float(incl_err)
            orbit_by[sid] = entry

        mass_by: Dict[str, Dict[str, float]] = {}
        for row in mass_res:
            try:
                sid = str(int(row["source_id"]))
            except Exception:
                continue
            d = mass_by.setdefault(sid, {})
            m1 = _first_finite(row, ["m1"])
            m2 = _first_finite(row, ["m2"])
            if m1 is not None and np.isfinite(m1) and m1 > 0 and "m1_msun" not in d:
                d["m1_msun"] = float(m1)
            if m2 is not None and np.isfinite(m2) and m2 > 0 and "m2_msun" not in d:
                d["m2_msun"] = float(m2)

        for sid in ids:
            base = orbit_by.get(sid)
            if base is None:
                cache[sid] = {"_none": True, "mass_checked": True}
                continue
            if sid in mass_by:
                base.update(mass_by[sid])
            base["mass_checked"] = True
            cache[sid] = base
            fetched += 1

    _save_json_cache(cache_path, cache)
    print(f"INFO:root:Gaia cache warmup done: {len(uniq)} ids scanned, {fetched} fetched, {time.time()-t0:.2f}s")


def _pipeline_telescope_from_filename(name: str) -> str:
    s = name.lower()
    if "kpf" in s:
        return "KPF"
    if "ghost" in s:
        return "GHOST"
    if "maroon" in s:
        return "MAROON-X"
    return "APF"


def parse_summary(path: Path) -> List[RVPoint]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "[PIPELINE RESULTS]" in text:
        lines = text.split("[PIPELINE RESULTS]", 1)[-1].splitlines()
    else:
        lines = text.splitlines()

    points: List[RVPoint] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or (line.startswith("[") and line.endswith("]")):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        if len(parts) >= 6 and parts[-1] in ("True", "False"):
            parts = parts[:-1]
        if len(parts) < 5:
            continue
        try:
            rv = float(parts[2])
        except ValueError:
            continue
        from darkhunter_rv.rv_point_filters import rv_epoch_is_valid

        try:
            mjd = float(parts[1])
        except ValueError:
            continue
        if not rv_epoch_is_valid(mjd, rv):
            continue
        try:
            points.append(
                RVPoint(
                    file=parts[0],
                    mjd=mjd,
                    rv=rv,
                    rv_err=max(float(parts[3]), 1e-4),
                    rms=max(abs(float(parts[4])), 1e-4),
                    telescope=_pipeline_telescope_from_filename(parts[0]),
                    is_literature=False,
                )
            )
        except ValueError:
            continue
    # Include literature RVs from [EXTERNAL RV DATA] when present.
    try:
        from darkhunter_rv.gaia_utils import parse_external_rvs_from_star_summary

        ext_rows = parse_external_rvs_from_star_summary(path)
    except Exception:
        ext_rows = []
    for r in ext_rows:
        try:
            mjd = float(r.get("mjd", float("nan")))
            rv = float(r.get("rv", float("nan")))
            rv_err = float(r.get("rv_err", float("nan")))
            tel = str(r.get("telescope", "LITERATURE") or "LITERATURE")
        except Exception:
            continue
        from darkhunter_rv.rv_point_filters import rv_epoch_is_valid

        if not rv_epoch_is_valid(mjd, rv):
            continue
        if not np.isfinite(rv_err) or rv_err <= 0:
            rv_err = 1.0
        points.append(
            RVPoint(
                file=f"external:{tel}",
                mjd=mjd,
                rv=rv,
                rv_err=max(rv_err, 1e-4),
                rms=max(rv_err, 1e-4),
                telescope=tel,
                is_literature=True,
            )
        )
    points.sort(key=lambda p: p.mjd)
    return points


def solve_kepler_eccentric_anomaly(M: np.ndarray, e: float, n_iter: int = 30) -> np.ndarray:
    E = np.array(M, dtype=float)
    for _ in range(n_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        E -= f / np.clip(fp, 1e-10, None)
    return E


def rv_model(params: np.ndarray, t: np.ndarray, t_ref: float) -> np.ndarray:
    # params = [logP, K, h, k, M0, gamma]
    logP, K, h, k, M0, gamma = params
    P = np.exp(logP)
    e = np.hypot(h, k)
    e = min(max(e, 1e-8), 0.95)
    omega = math.atan2(k, h)

    n = 2.0 * np.pi / P
    M = n * (t - t_ref) + M0
    E = solve_kepler_eccentric_anomaly(M, e)
    # true anomaly
    cosf = (np.cos(E) - e) / (1.0 - e * np.cos(E))
    sinf = (np.sqrt(1.0 - e * e) * np.sin(E)) / (1.0 - e * np.cos(E))
    f = np.arctan2(sinf, cosf)

    return gamma + K * (np.cos(f + omega) + e * np.cos(omega))


def next_extrema_after(
    params: np.ndarray,
    t_ref: float,
    period_days: float,
    t0_mjd: float,
    n_grid: int = 12000,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (next_max_mjd, next_min_mjd) after t0 for the model curve."""
    if period_days <= 0:
        return None, None
    t_grid = np.linspace(t0_mjd, t0_mjd + 2.5 * period_days, n_grid)
    y = rv_model(params, t_grid, t_ref)
    dy = np.diff(y)

    maxima = []
    minima = []
    for i in range(1, len(dy)):
        if dy[i - 1] > 0 and dy[i] <= 0:
            maxima.append(t_grid[i])
        if dy[i - 1] < 0 and dy[i] >= 0:
            minima.append(t_grid[i])

    t_next_max = maxima[0] if maxima else None
    t_next_min = minima[0] if minima else None
    return t_next_max, t_next_min


def _project_params_to_fixed_e(params: np.ndarray, fix_e: Optional[float]) -> np.ndarray:
    if fix_e is None or not np.isfinite(fix_e):
        return params
    e_fix = float(np.clip(fix_e, 1e-8, 0.95))
    p = np.array(params, dtype=float, copy=True)
    omega = float(np.arctan2(p[3], p[2]))
    p[2] = e_fix * np.cos(omega)
    p[3] = e_fix * np.sin(omega)
    return p


def _clip_initial_guess(x0: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    x = np.array(x0, dtype=float, copy=True)
    bad = ~np.isfinite(x)
    if np.any(bad):
        mid = 0.5 * (lower + upper)
        x[bad] = mid[bad]
    return np.minimum(np.maximum(x, lower), upper)


def _sanitize_rv_arrays(
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    rms_fallback: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep epochs with finite MJD and RV; ensure positive finite errors."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y, yerr = t[ok], y[ok], yerr[ok]
    if t.size == 0:
        return t, y, yerr

    good_err = np.isfinite(yerr) & (yerr > 0)
    if np.any(good_err):
        err_fill = float(np.median(yerr[good_err]))
    elif rms_fallback is not None:
        rms = np.asarray(rms_fallback, dtype=float)[ok]
        err_fill = float(np.nanmedian(rms[np.isfinite(rms) & (rms > 0)]))
    else:
        err_fill = float("nan")
    if not np.isfinite(err_fill) or err_fill <= 0:
        err_fill = 0.1
    bad_err = ~np.isfinite(yerr) | (yerr <= 0)
    yerr = yerr.copy()
    yerr[bad_err] = err_fill
    return t, y, yerr


def fit_keplerian(
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    period_min: Optional[float] = None,
    period_max: Optional[float] = None,
    period_prior: Optional[float] = None,
    period_prior_sigma: float = 0.15,
    fix_period: Optional[float] = None,
    fix_e: Optional[float] = None,
    initial_params: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    if t.size < 3:
        raise ValueError(f"need at least 3 finite RV points, got {t.size}")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("non-finite MJD or RV after sanitization")
    if not np.all(np.isfinite(yerr)) or np.any(yerr <= 0):
        raise ValueError("non-finite or non-positive RV errors after sanitization")

    span = max(float(np.max(t) - np.min(t)), 1.0)
    dt = np.diff(np.sort(t))
    dt = dt[dt > 0]
    median_cadence = float(np.median(dt)) if len(dt) else 1.0

    # Default lower bound avoids physically/numerically silly sub-cadence aliases.
    if period_min is None:
        min_period = max(0.5, 2.0 * median_cadence)
    else:
        min_period = float(period_min)

    if period_max is None:
        max_period = max(5.0, min(span * 2.5, 5000.0))
    else:
        max_period = float(period_max)

    if max_period <= min_period:
        max_period = min_period * 1.5

    ls = LombScargle(t, y, yerr)
    freq, power = ls.autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=10,
    )
    p_default = 0.5 * (min_period + max_period)
    if len(freq) == 0 or not np.any(np.isfinite(power)):
        p_guess = p_default
    else:
        p_guess = float(1.0 / freq[int(np.nanargmax(power))])
        if not np.isfinite(p_guess) or p_guess <= 0:
            p_guess = p_default

    y_span = float(np.max(y) - np.min(y))
    k_guess = max(1.0, 0.5 * y_span) if np.isfinite(y_span) and y_span > 0 else 1.0
    gamma_guess = float(np.median(y))
    if not np.isfinite(gamma_guess):
        gamma_guess = 0.0
    t_ref = float(np.median(t))
    if not np.isfinite(t_ref):
        t_ref = float(np.mean(t))

    if fix_period is not None:
        p_guess = float(np.clip(fix_period, min_period, max_period))
    elif period_prior is not None and np.isfinite(period_prior) and period_prior > 0:
        p_guess = float(np.clip(period_prior, min_period, max_period))

    lower = np.array([np.log(min_period), 0.0, -0.95, -0.95, -np.pi, -1000.0])
    upper = np.array([np.log(max_period), 1000.0, 0.95, 0.95, np.pi, 1000.0])

    def resid(p):
        p_eff = _project_params_to_fixed_e(p, fix_e)
        model = rv_model(p_eff, t, t_ref)
        r = (y - model) / np.clip(yerr, 1e-4, None)
        if fix_period is not None:
            # Hold P effectively fixed by adding a very strong prior.
            r = np.concatenate([r, np.array([(p[0] - np.log(fix_period)) / 1e-6])])
        elif period_prior is not None and np.isfinite(period_prior) and period_prior > 0:
            sigma = max(1e-4, float(period_prior_sigma))
            r = np.concatenate([r, np.array([(p[0] - np.log(period_prior)) / sigma])])
        return r

    # Multi-start on period to reduce local-minimum/alias failures.
    peak_idx = np.argsort(power)[-8:][::-1]
    candidate_periods = [p_guess]
    for idx in peak_idx:
        p0 = float(np.clip(1.0 / freq[idx], min_period, max_period))
        candidate_periods.extend([p0, 0.5 * p0, 2.0 * p0])
    # Deduplicate while preserving order.
    seen = set()
    uniq_periods = []
    for p0 in candidate_periods:
        key = round(float(np.clip(p0, min_period, max_period)), 6)
        if key not in seen:
            seen.add(key)
            uniq_periods.append(key)

    best_res = None
    best_cost = np.inf
    f_scale = float(np.median(yerr))
    if not np.isfinite(f_scale) or f_scale <= 0:
        f_scale = 1.0

    x0_candidates: List[np.ndarray] = []
    if initial_params is not None:
        seed = _clip_initial_guess(np.asarray(initial_params, dtype=float), lower, upper)
        if fix_period is not None:
            seed[0] = np.log(float(np.clip(fix_period, min_period, max_period)))
        x0_candidates.append(seed)

    for p0 in uniq_periods:
        log_p = np.log(np.clip(p0, min_period, max_period))
        if not np.isfinite(log_p):
            log_p = np.log(p_default)
        x0 = _clip_initial_guess(
            np.array([
                log_p,
                np.clip(k_guess, 0.5, 300.0),
                0.2,  # h = e cos(omega)
                0.0,  # k = e sin(omega)
                0.0,  # M0
                gamma_guess,
            ]),
            lower,
            upper,
        )
        x0_candidates.append(x0)

    seen_x0: set = set()
    for x0 in x0_candidates:
        key = tuple(np.round(x0, 5))
        if key in seen_x0:
            continue
        seen_x0.add(key)
        try:
            res = least_squares(
                resid,
                x0=x0,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=f_scale,
                max_nfev=8000,
            )
        except ValueError:
            continue
        if res.cost < best_cost:
            best_cost = res.cost
            best_res = res

    if best_res is None:
        raise RuntimeError("Keplerian fit failed: no optimization result.")

    res = best_res
    params = _project_params_to_fixed_e(res.x, fix_e)
    P = float(np.exp(params[0]))
    K = float(params[1])
    h, k = float(params[2]), float(params[3])
    e = float(min(max(np.hypot(h, k), 1e-8), 0.999))
    omega = float(np.arctan2(k, h))
    M0 = float(params[4])
    gamma = float(params[5])
    n = 2.0 * np.pi / P
    t_peri = float(t_ref - M0 / n)

    model = rv_model(params, t, t_ref)
    chi2 = float(np.sum(((y - model) / np.clip(yerr, 1e-4, None)) ** 2))
    n_fit_params = 5 if fix_e is not None else len(params)
    dof = max(1, len(t) - n_fit_params)

    # Next RV extrema relative to "now" (UTC), for scheduling.
    now_mjd = float(Time.now().mjd)
    t_next_max, t_next_min = next_extrema_after(params, t_ref, P, now_mjd)
    if t_next_max is None or t_next_min is None:
        # Fallback to "after latest observed epoch" if local-extrema detection fails.
        t_start = float(np.max(t))
        t_grid = np.linspace(t_start, t_start + 2.0 * P, 4000)
        y_grid = rv_model(params, t_grid, t_ref)
        t_next_max = float(t_grid[np.argmax(y_grid)])
        rv_next_max = float(np.max(y_grid))
        t_next_min = float(t_grid[np.argmin(y_grid)])
        rv_next_min = float(np.min(y_grid))
    else:
        rv_next_max = float(rv_model(params, np.array([t_next_max]), t_ref)[0])
        rv_next_min = float(rv_model(params, np.array([t_next_min]), t_ref)[0])

    report = {
        "converged": bool(res.success),
        "message": str(res.message),
        "n_points": int(len(t)),
        "chi2": chi2,
        "dof": int(dof),
        "chi2_red": float(chi2 / dof),
        "P_days": P,
        "K_kms": K,
        "e": e,
        "omega_rad": omega,
        "omega_deg": float(np.degrees(omega)),
        "gamma_kms": gamma,
        "t_periastron_mjd": t_peri,
        "t_ref_mjd": t_ref,
        "next_rv_max_mjd": t_next_max,
        "next_rv_max_kms": rv_next_max,
        "next_rv_min_mjd": t_next_min,
        "next_rv_min_kms": rv_next_min,
        "now_mjd": now_mjd,
        "power_peak": float(np.max(power)),
        "period_guess_days": float(p_guess),
        "period_min_days": float(min_period),
        "period_max_days": float(max_period),
        "median_cadence_days": float(median_cadence),
        "period_prior_days": None if period_prior is None else float(period_prior),
        "period_prior_sigma_lnP": float(period_prior_sigma),
        "fixed_period_days": None if fix_period is None else float(fix_period),
        "fixed_eccentricity": None if fix_e is None else float(fix_e),
        "params_raw": params.tolist(),
    }
    return params, report


def mass_function_msun(P_days: float, K_kms: float, e: float) -> float:
    """Mass function (Msun) for P in days, K in km/s. Returns NaN if inputs are non-physical."""
    if not (np.isfinite(P_days) and np.isfinite(K_kms) and np.isfinite(e)):
        return float("nan")
    p = float(P_days)
    k = float(K_kms)
    ecc = float(np.clip(e, 0.0, 0.999))
    if p <= 0.0 or k <= 0.0:
        return float("nan")
    one_minus_e2 = 1.0 - ecc * ecc
    if one_minus_e2 <= 0.0:
        return float("nan")
    val = 1.036149e-7 * (k**3) * p * (one_minus_e2**1.5)
    return float(val) if np.isfinite(val) else float("nan")


def solve_m2sini_msun(f_mass: float, m1: float) -> float:
    # Solve x^3/(m1 + x)^2 = f_mass for x > 0 (x = M2 sin i for sin i=1 lower limit)
    lo, hi = 1e-8, 200.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        g = (mid ** 3) / ((m1 + mid) ** 2) - f_mass
        if g > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def solve_m2_with_inclination_msun(f_mass: float, m1: float, inclination_deg: float) -> Optional[float]:
    """
    Solve (m2^3 sin(i)^3) / (m1 + m2)^2 = f_mass for m2 > 0.
    Returns None for non-physical or non-finite inputs.
    """
    if not np.isfinite(f_mass) or not np.isfinite(m1) or not np.isfinite(inclination_deg):
        return None
    if f_mass <= 0 or m1 <= 0:
        return None
    i_rad = np.deg2rad(float(inclination_deg))
    sin_i = float(np.sin(i_rad))
    if not np.isfinite(sin_i) or abs(sin_i) < 1e-6:
        return None
    s3 = abs(sin_i) ** 3

    lo, hi = 1e-8, 500.0
    for _ in range(260):
        mid = 0.5 * (lo + hi)
        g = (mid ** 3) * s3 / ((m1 + mid) ** 2) - f_mass
        if g > 0:
            hi = mid
        else:
            lo = mid
    out = 0.5 * (lo + hi)
    return float(out) if np.isfinite(out) and out > 0 else None


def _finite_mass_value(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and np.isfinite(value) and float(value) > 0:
        return float(value)
    return None


def _gaia_nss_dict_from_report(report: dict) -> Dict[str, Any]:
    gnss = report.get("gaia_nss")
    return gnss if isinstance(gnss, dict) else {}


def _gaia_cache_entry(gaia_cache: Optional[Dict[str, Any]], gaia_source_id: Optional[str]) -> Optional[dict]:
    if not gaia_cache or not gaia_source_id:
        return None
    sid = str(gaia_source_id).strip()
    for key in (sid, f"Gaia_DR3_{sid}"):
        val = gaia_cache.get(key)
        if isinstance(val, dict) and not val.get("_none"):
            return val
    return None


def estimate_m1_msun_from_hrd(meta: Dict[str, Any]) -> Optional[float]:
    """Rough main-sequence M1 (Msun) from Gaia GSP-PHOT Teff/logg when NSS masses are absent."""
    teff = _meta_float(meta, "Teff", "teff")
    if teff is None or not (3200.0 <= teff <= 9000.0):
        return None
    logg = _meta_float(meta, "logg", "Logg")
    if logg is not None and logg < 3.5:
        return None
    m1 = float((teff / 5772.0) ** 1.8)
    return float(np.clip(m1, 0.5, 2.0))


def resolve_m1_msun_for_rv_mass(
    report: dict,
    *,
    summary_path: Optional[Path] = None,
    gaia_cache: Optional[Dict[str, Any]] = None,
    table_m1_msun: Optional[float] = None,
) -> Optional[float]:
    """
    Primary mass for converting RV f(M) to M2 sin i / M2 at i.

    Order: website table M1 (luminous) → fit report → embedded gaia_nss → cache →
    summary metadata → M2/Mass_Ratio → Teff/logg estimate → 1 Msun if free fit exists.
    """
    m1 = _finite_mass_value(table_m1_msun)
    if m1 is not None:
        return m1

    m1 = _finite_mass_value(report.get("used_m1_msun"))
    if m1 is not None:
        return m1

    gnss = _gaia_nss_dict_from_report(report)
    m1 = _finite_mass_value(gnss.get("m1_msun"))
    if m1 is not None:
        return m1

    sid = report.get("gaia_source_id")
    cached = _gaia_cache_entry(gaia_cache, str(sid) if sid is not None else None)
    if cached is not None:
        m1 = _finite_mass_value(cached.get("m1_msun"))
        if m1 is not None:
            return m1

    meta_mass: Dict[str, float] = {}
    meta_raw: Optional[Dict[str, Any]] = None
    if summary_path is not None and summary_path.is_file():
        meta_mass = load_mass_priors_from_summary(summary_path)
        m1 = _finite_mass_value(meta_mass.get("m1_msun"))
        if m1 is not None:
            return m1
        meta_raw = _parse_gaia_metadata(summary_path)

    m2 = _finite_mass_value(report.get("used_m2_msun"))
    if m2 is None:
        m2 = _finite_mass_value(gnss.get("m2_msun"))
    if m2 is None and cached is not None:
        m2 = _finite_mass_value(cached.get("m2_msun"))
    if m2 is None:
        m2 = _finite_mass_value(meta_mass.get("m2_msun"))
    q = _meta_float(meta_raw, "Mass_Ratio", "mass_ratio") if meta_raw else None
    if m2 is not None and q is not None and q > 0:
        return float(m2 / q)

    if meta_raw:
        est = estimate_m1_msun_from_hrd(meta_raw)
        if est is not None:
            return est

    if _free_fit_variant_report(report) is not None:
        return 1.0
    return None


def _valid_inclination_deg(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return None
    incl = float(value)
    # sin(i) must be usable; Gaia NSS inclinations are in (0, 180) degrees.
    if incl <= 0.05 or incl >= 179.95:
        return None
    return incl


def _json_inclination_is_stale_sin_i_fallback(report: dict) -> bool:
    """
    Detect fit JSON from the old i=90° table-fill era: inclination_deg_used=90 with
    m2_given_inclination_msun copied from m2sini_msun.
    """
    incl = _valid_inclination_deg(report.get("inclination_deg_used"))
    if incl is None or abs(incl - 90.0) > 0.05:
        return False
    stored_at = _finite_mass_value(report.get("m2_given_inclination_msun"))
    stored_sin = _finite_mass_value(report.get("m2sini_msun"))
    if stored_at is None or stored_sin is None:
        return False
    return abs(stored_at - stored_sin) <= max(1e-6, 1e-4 * stored_sin)


def resolve_inclination_deg_for_rv_mass(
    report: dict,
    *,
    summary_path: Optional[Path] = None,
    gaia_cache: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Astrometric inclination (deg) for M2 at i = (M2 sin i) / sin(i)."""
    gnss = _gaia_nss_dict_from_report(report)
    incl = _valid_inclination_deg(gnss.get("inclination_deg"))
    if incl is not None:
        return incl

    sid = report.get("gaia_source_id")
    cached = _gaia_cache_entry(gaia_cache, str(sid) if sid is not None else None)
    if cached is not None:
        incl = _valid_inclination_deg(cached.get("inclination_deg"))
        if incl is not None:
            return incl

    if summary_path is not None and summary_path.is_file():
        incl = _valid_inclination_deg(
            load_mass_priors_from_summary(summary_path).get("inclination_deg")
        )
        if incl is not None:
            return incl
        meta = _parse_gaia_metadata(summary_path)
        if meta:
            incl = _valid_inclination_deg(
                _meta_float(meta, "Inclination", "inclination", "Inclination_Deg")
            )
            if incl is not None:
                return incl

    if not _json_inclination_is_stale_sin_i_fallback(report):
        incl = _valid_inclination_deg(report.get("inclination_deg_used"))
        if incl is not None:
            return incl
    return None


def enrich_gaia_cache_from_summaries(
    gaia_cache: Dict[str, Any],
    out_dir: Path,
    *,
    gaia_ids: Optional[List[str]] = None,
) -> int:
    """Merge [GAIA METADATA] masses/inclination from star summaries into the NSS cache dict."""
    n = 0
    if gaia_ids is None:
        paths = discover_summary_files(out_dir)
    else:
        paths = []
        for gid in gaia_ids:
            p = discover_summary_path(out_dir, gid)
            if p is not None:
                paths.append(p)
    for summ in paths:
        if not summ.is_file():
            continue
        sid = parse_object_id_from_summary(summ)
        if not sid:
            continue
        priors = load_mass_priors_from_summary(summ)
        if not priors:
            continue
        entry = gaia_cache.get(sid)
        if not isinstance(entry, dict) or entry.get("_none"):
            entry = {}
            gaia_cache[sid] = entry
        patched = False
        for key in ("m1_msun", "m2_msun", "inclination_deg"):
            val = priors.get(key)
            if val is None or not np.isfinite(val):
                continue
            if key == "inclination_deg" and _valid_inclination_deg(val) is None:
                continue
            entry[key] = float(val)
            patched = True
        if patched:
            n += 1
    return n


def m2_msun_at_inclination(m2sin_msun: float, inclination_deg: float) -> Optional[float]:
    """M2 at inclination i (Msun) from M2 sin i and astrometric i (degrees)."""
    incl = _valid_inclination_deg(inclination_deg)
    if incl is None or not np.isfinite(m2sin_msun) or float(m2sin_msun) <= 0:
        return None
    sin_i = float(np.sin(np.deg2rad(incl)))
    if sin_i < 1e-6:
        return None
    return float(m2sin_msun / sin_i)


def resolve_m2_astrometric_msun(
    report: dict,
    *,
    summary_path: Optional[Path] = None,
    gaia_cache: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Gaia NSS astrometric secondary mass (not from the RV-only fit)."""
    m2 = _finite_mass_value(report.get("used_m2_msun"))
    if m2 is not None:
        return m2
    m2 = _finite_mass_value(report.get("m2_astrometric_msun"))
    if m2 is not None:
        return m2
    gnss = _gaia_nss_dict_from_report(report)
    m2 = _finite_mass_value(gnss.get("m2_msun"))
    if m2 is not None:
        return m2
    sid = report.get("gaia_source_id")
    cached = _gaia_cache_entry(gaia_cache, str(sid) if sid is not None else None)
    if cached is not None:
        m2 = _finite_mass_value(cached.get("m2_msun"))
        if m2 is not None:
            return m2
    if summary_path is not None and summary_path.is_file():
        m2 = _finite_mass_value(load_mass_priors_from_summary(summary_path).get("m2_msun"))
        if m2 is not None:
            return m2
    return None


def inclination_deg_for_website_table(
    report: dict,
    *,
    summary_path: Optional[Path] = None,
    gaia_cache: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Astrometric inclination (deg) for the website table column."""
    return resolve_inclination_deg_for_rv_mass(
        report, summary_path=summary_path, gaia_cache=gaia_cache
    )


def website_table_masses_from_report(
    report: dict,
    *,
    summary_path: Optional[Path] = None,
    gaia_cache: Optional[Dict[str, Any]] = None,
    table_m1_msun: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    Map a Keplerian fit JSON report to website ``data.csv`` mass columns.

    - ``m2_msun``: Gaia NSS astrometric M2 (with Gaia M1), not from the RV fit.
    - ``m2sin_i_msun``: M2 sin i from the RV-only (free) fit and assumed M1.
    - ``m2_at_i_msun``: M2 at i from the RV-only f(M), assumed M1, and astrometric i.
    """
    incl = resolve_inclination_deg_for_rv_mass(
        report, summary_path=summary_path, gaia_cache=gaia_cache
    )
    m1 = resolve_m1_msun_for_rv_mass(
        report,
        summary_path=summary_path,
        gaia_cache=gaia_cache,
        table_m1_msun=table_m1_msun,
    )

    m2sin: Optional[float] = None
    m2_at_i: Optional[float] = None
    rep_free = _free_fit_variant_report(report)
    if rep_free is not None and m1 is not None:
        calc_sini, calc_at_i = rv_only_mass_estimates(rep_free, m1, incl)
        m2sin = _finite_mass_value(calc_sini)
        if incl is not None and m2sin is not None:
            m2_at_i = _finite_mass_value(calc_at_i)
            if m2_at_i is None:
                m2_at_i = m2_msun_at_inclination(m2sin, incl)
    if m2sin is None:
        m2sin = _finite_mass_value(report.get("m2sini_msun"))
    if m2_at_i is None and m2sin is not None and incl is not None:
        m2_at_i = m2_msun_at_inclination(m2sin, incl)
    if m2_at_i is None:
        stored_at_i = _finite_mass_value(report.get("m2_given_inclination_msun"))
        stored_sin_json = _finite_mass_value(report.get("m2sini_msun"))
        if (
            stored_at_i is not None
            and stored_sin_json is not None
            and abs(stored_at_i - stored_sin_json) > max(1e-6, 1e-4 * stored_sin_json)
        ):
            m2_at_i = stored_at_i

    return {
        "m2_msun": resolve_m2_astrometric_msun(
            report, summary_path=summary_path, gaia_cache=gaia_cache
        ),
        "m2sin_i_msun": m2sin,
        "m2_at_i_msun": m2_at_i,
    }


def rv_only_mass_estimates(
    rep_free: dict,
    m1_msun: Optional[float],
    inclination_deg: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """M2 sin i and M2 at i from the RV-only Keplerian report and astrometric inputs."""
    fm = rep_free.get("mass_function_msun")
    if fm is None or not np.isfinite(fm):
        fm = mass_function_msun(rep_free["P_days"], rep_free["K_kms"], rep_free["e"])
    if fm is None or not np.isfinite(fm) or fm <= 0:
        return None, None
    m1 = _finite_mass_value(m1_msun)
    if m1 is None:
        return None, None
    m2sini = float(solve_m2sini_msun(float(fm), m1))
    m2_at_i = None
    if inclination_deg is not None and np.isfinite(inclination_deg):
        m2_at_i = solve_m2_with_inclination_msun(float(fm), m1, float(inclination_deg))
    return m2sini, m2_at_i


def fit_all_variants(
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    gaia_nss: Optional[Dict[str, float]],
    *,
    period_min: Optional[float],
    period_max: Optional[float],
    period_prior_sigma: float,
) -> Dict[str, Tuple[np.ndarray, dict]]:
    """Four Keplerian fits: free P/e; fixed P; fixed e; fixed P and e (Gaia when available)."""
    out: Dict[str, Tuple[np.ndarray, dict]] = {}
    nss_p = None
    nss_e = None
    if gaia_nss is not None:
        nss_p = gaia_nss.get("period_days")
        nss_e = gaia_nss.get("eccentricity")

    try:
        free_params, free_rep = fit_keplerian(
            t,
            y,
            yerr,
            period_min=period_min,
            period_max=period_max,
            period_prior=None,
            period_prior_sigma=period_prior_sigma,
        )
    except (ValueError, RuntimeError):
        return out
    free_rep["fit_variant"] = "free"
    fm = mass_function_msun(free_rep["P_days"], free_rep["K_kms"], free_rep["e"])
    free_rep["mass_function_msun"] = None if not np.isfinite(fm) else float(fm)
    out["free"] = (free_params, free_rep)

    constrained: List[Tuple[str, Optional[float], Optional[float]]] = []
    if nss_p is not None and nss_e is not None and nss_p > 0 and 0 <= nss_e < 1:
        constrained.extend(
            [
                ("fix_period", float(nss_p), None),
                ("fix_ecc", None, float(nss_e)),
                ("fix_period_ecc", float(nss_p), float(nss_e)),
            ]
        )

    for key, fix_p, fix_e in constrained:
        try:
            params, rep = fit_keplerian(
                t,
                y,
                yerr,
                period_min=period_min,
                period_max=period_max,
                period_prior=None,
                period_prior_sigma=period_prior_sigma,
                fix_period=fix_p,
                fix_e=fix_e,
                initial_params=free_params,
            )
        except (ValueError, RuntimeError):
            continue
        if key == "fix_period":
            chi_free = free_rep.get("chi2_red")
            chi_fp = rep.get("chi2_red")
            e_free = free_rep.get("e")
            e_fp = rep.get("e")
            if (
                isinstance(chi_free, (int, float))
                and isinstance(chi_fp, (int, float))
                and chi_free > 0
                and chi_fp > 2.0 * chi_free
                and isinstance(e_free, (int, float))
                and isinstance(e_fp, (int, float))
                and abs(float(e_fp) - float(e_free)) > 0.15
            ):
                try:
                    params_retry, rep_retry = fit_keplerian(
                        t,
                        y,
                        yerr,
                        period_min=period_min,
                        period_max=period_max,
                        period_prior=None,
                        period_prior_sigma=period_prior_sigma,
                        fix_period=fix_p,
                        fix_e=float(e_free),
                        initial_params=free_params,
                    )
                    chi_retry = rep_retry.get("chi2_red")
                    if isinstance(chi_retry, (int, float)) and chi_retry < chi_fp:
                        params, rep = params_retry, rep_retry
                        rep["fix_period_e_seed_retry"] = True
                except (ValueError, RuntimeError):
                    pass
        rep["fit_variant"] = key
        fm = mass_function_msun(rep["P_days"], rep["K_kms"], rep["e"])
        rep["mass_function_msun"] = None if not np.isfinite(fm) else float(fm)
        if fix_p is not None:
            rep["fixed_period_days"] = float(fix_p)
        if fix_e is not None:
            rep["fixed_eccentricity"] = float(fix_e)
        out[key] = (params, rep)
    return out


def build_plot(
    summary_path: Path,
    points: List[RVPoint],
    params: np.ndarray,
    report: dict,
    out_png: Path,
    m1_msun: Optional[float] = None,
) -> None:
    t = np.array([p.mjd for p in points], dtype=float)
    y = np.array([p.rv for p in points], dtype=float)
    # Plot error bars from RMS column, not formal RV error.
    yerr_rms = np.array([p.rms for p in points], dtype=float)

    t_ref = report["t_ref_mjd"]
    now_mjd = float(report.get("now_mjd", Time.now().mjd))
    t_next_max = float(report["next_rv_max_mjd"])
    t_next_min = float(report["next_rv_min_mjd"])
    if t_next_max <= t_next_min:
        next_event = t_next_max
        next_event_label = "Next max"
        other_event = t_next_min
        other_event_label = "Next min"
    else:
        next_event = t_next_min
        next_event_label = "Next min"
        other_event = t_next_max
        other_event_label = "Next max"

    obs_start = None
    obs_end = None
    obs_win = report.get("observability_window")
    if isinstance(obs_win, dict):
        try:
            obs_start = float(Time(obs_win["start_date"], format="iso", scale="utc").mjd)
            obs_end = float(Time(obs_win["end_date"], format="iso", scale="utc").mjd) + 1.0
        except Exception:
            obs_start = None
            obs_end = None

    t_start = float(np.min(t) - 0.02 * (np.ptp(t) + 1))
    # Extend through nearest upcoming extrema and observability window if available.
    t_end_candidates = [np.max(t) + 0.02 * (np.ptp(t) + 1), now_mjd, next_event]
    if obs_end is not None:
        t_end_candidates.append(obs_end)
    t_end = float(max(t_end_candidates))
    t_dense = np.linspace(t_start, t_end, 2000)
    y_dense = rv_model(params, t_dense, t_ref)
    # Lock y-limits from RV data + fitted curve only (ignore large error bars).
    y_low = float(min(np.min(y), np.min(y_dense)))
    y_high = float(max(np.max(y), np.max(y_dense)))
    y_pad = max(1.0, 0.08 * (y_high - y_low if y_high > y_low else 1.0))
    y_lim = (y_low - y_pad, y_high + y_pad)

    f_mass = mass_function_msun(report["P_days"], report["K_kms"], report["e"])
    m2sini = None
    if m1_msun is not None and np.isfinite(m1_msun) and m1_msun > 0:
        m2sini = solve_m2sini_msun(f_mass, float(m1_msun))

    # Shorter aspect so table rows are not excessively tall on the website.
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    # Do not shade all future times; only shade APF observability window.
    if obs_start is not None and obs_end is not None and isinstance(obs_win, dict):
        try:
            left = max(t_start, obs_start)
            right = min(t_end, obs_end)
            if right > left:
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
        except Exception:
            pass
    ax.axvline(now_mjd, color="0.35", ls="--", lw=1.2, alpha=0.9)
    # Plot symbols/colors:
    # - Our data (black): APF=o, KPF=s, GHOST=p, MAROON-X=x
    # - Literature (dark grey): diamond
    plotted_any = False
    tel_marker = {"APF": "o", "KPF": "s", "GHOST": "p", "MAROON-X": "x"}
    for tel, marker in tel_marker.items():
        idx = [i for i, p in enumerate(points) if (not p.is_literature and str(p.telescope).upper() == tel)]
        if not idx:
            continue
        ax.errorbar(
            t[idx],
            y[idx],
            yerr=yerr_rms[idx],
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
        plotted_any = True
    idx_lit = [i for i, p in enumerate(points) if p.is_literature]
    if idx_lit:
        ax.errorbar(
            t[idx_lit],
            y[idx_lit],
            yerr=yerr_rms[idx_lit],
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
        plotted_any = True
    if not plotted_any:
        ax.errorbar(t, y, yerr=yerr_rms, fmt="o", ms=5, lw=1, capsize=2, color="black")
    ax.plot(t_dense, y_dense, "-", lw=2)
    ax.set_xlabel("MJD")
    ax.set_ylabel("RV (km/s)")
    title_id = parse_object_id_from_summary(summary_path) or summary_path.stem.replace("_summary", "")
    ax.set_title(f"APF RV Fit: Gaia DR3 {title_id}")
    ax.grid(alpha=0.25)
    ax.set_ylim(*y_lim)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="major", direction="in", length=7, width=1.1, top=True, right=True)
    ax.tick_params(axis="both", which="minor", direction="in", length=3.5, width=0.9, top=True, right=True)
    if plotted_any:
        ax.legend(loc="best", fontsize=8.5)

    p_day = int(round(report["P_days"]))
    next_min_mjd = float(report["next_rv_min_mjd"])
    next_max_mjd = float(report["next_rv_max_mjd"])
    next_min_date = Time(next_min_mjd, format="mjd").to_datetime().strftime("%Y-%m-%d")
    next_max_date = Time(next_max_mjd, format="mjd").to_datetime().strftime("%Y-%m-%d")
    in_nss = "Y" if report.get("gaia_nss") else "N"
    e_fixed = "Y" if report.get("fixed_eccentricity") is not None else "N"
    p_prior_used = (
        report.get("fixed_period_days") is not None
        or report.get("used_gaia_period_prior") is not None
        or report.get("period_prior_days") is not None
    )
    p_prior_yn = "Y" if p_prior_used else "N"
    line3 = f"NSS catalog={in_nss}, e fixed in fit={e_fixed}, P prior={p_prior_yn}"
    text_lines = [line3]

    # Mark nearest upcoming extrema on the plot.
    ax.axvline(next_event, color="tab:red", ls="--", lw=1.2, alpha=0.9)

    y_min, y_top = ax.get_ylim()
    y_span = y_top - y_min
    y_top_in = y_top - 0.02 * y_span
    y_bot_in = y_min + 0.02 * y_span
    event_date = Time(next_event, format="mjd").to_datetime().strftime("%Y-%m-%d")
    other_event_date = Time(other_event, format="mjd").to_datetime().strftime("%Y-%m-%d")
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
    row1 = HPacker(
        children=[
            TextArea(f"{next_event_label} ", textprops=dict(color="black", fontsize=9)),
            TextArea(event_date, textprops=dict(color="tab:red", fontsize=9)),
        ],
        align="center",
        pad=0,
        sep=1,
    )
    row2 = HPacker(
        children=[
            TextArea(f"{other_event_label} ", textprops=dict(color="black", fontsize=9)),
            TextArea(other_event_date, textprops=dict(color="tab:red", fontsize=9)),
        ],
        align="center",
        pad=0,
        sep=1,
    )
    packed = VPacker(children=[row1, row2], align="left", pad=0, sep=1)
    ab = AnnotationBbox(
        packed,
        (0.985, 1.01),
        xycoords="axes fraction",
        box_alignment=(1.0, 0.0),
        frameon=True,
        bboxprops=dict(facecolor="white", edgecolor="tab:red", alpha=0.9, boxstyle="round,pad=0.2"),
    )
    ax.add_artist(ab)
    # Keep fit summary above the plotting region so it does not overlap data.
    fig.subplots_adjust(top=0.86)
    fig.text(
        0.13,
        0.965,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=9.5,
    )
    # Highlight period/eccentricity with a style matching the event callout.
    p_highlight = f"P={p_day} d"
    e_highlight = f"e={report['e']:.3f}"
    ax.text(
        0.015,
        0.985,
        f"{p_highlight}   {e_highlight}",
        transform=ax.transAxes,
        fontsize=9.5,
        color="tab:red",
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="tab:red", alpha=0.9, boxstyle="round,pad=0.2"),
    )
    if m2sini is not None and m1_msun is not None:
        ax.text(
            0.015,
            0.935,
            f"M₂ sin(i)={m2sini:.4f} M⊙   M₁={m1_msun:.4f} M⊙",
            transform=ax.transAxes,
            fontsize=9.2,
            color="black",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.95, boxstyle="round,pad=0.2"),
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def newest_summary(output_dir: Path) -> Optional[Path]:
    files = discover_summary_files(output_dir)
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def resolve_summary_files(output_dir: Path, summary: Optional[Path], run_all: bool) -> List[Path]:
    if summary is not None:
        return [summary.resolve()]
    if run_all:
        return discover_summary_files(output_dir)
    newest = newest_summary(output_dir)
    return [newest] if newest is not None else []


def report_stem(summary_path: Path, gaia_source_id: Optional[str]) -> str:
    if gaia_source_id:
        return str(gaia_source_id)
    stem = summary_path.stem.replace("_summary", "")
    m = re.match(r"Gaia_DR3_(\d+)$", stem)
    return m.group(1) if m else stem


def load_table_m1_map_from_csv(data_csv: Path) -> Dict[str, float]:
    """Gaia source id → luminous M1 (Msun) from website tables/data.csv."""
    if not data_csv.is_file():
        return {}
    import csv

    from darkhunter_rv.website_table_csv import gaia_id_from_row, parse_table_m1_msun

    out: Dict[str, float] = {}
    with data_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return out
    hdr = rows[0]
    if "GAIA NAME" not in hdr:
        return out
    gaia_i = hdr.index("GAIA NAME")
    for r in rows[1:]:
        sid = gaia_id_from_row(r[gaia_i] if gaia_i < len(r) else "")
        if not sid:
            continue
        m1 = parse_table_m1_msun(r, hdr)
        if m1 is not None:
            out[sid] = m1
    return out


def run_one(
    summary_path: Path,
    out_dir: Path,
    min_points: int,
    max_points: Optional[int],
    m1_msun: Optional[float],
    period_min: Optional[float],
    period_max: Optional[float],
    period_prior: Optional[float],
    period_prior_sigma: float,
    fix_period: Optional[float],
    fix_e: Optional[float],
    use_gaia_nss: bool,
    gaia_cache_path: Optional[Path],
    observability_cache_path: Optional[Path],
    query_gaia_online: bool,
    plots_root: Optional[Path] = None,
    table_m1_msun: Optional[float] = None,
) -> Optional[dict]:
    from darkhunter_rv.rv_keplerian_plots import (
        our_telescope_points,
        plot_fit_residuals,
        plot_multi_fit,
        plot_rv_data_only,
    )

    points = parse_summary(summary_path)
    n_epochs = len(points)
    if n_epochs < min_points:
        print(
            f"[SKIP] {summary_path}: n_epochs={n_epochs} < min_points={min_points} "
            f"(file rows in [PIPELINE RESULTS]={count_pipeline_rows(summary_path)})"
        )
        return None
    if max_points is not None and n_epochs > max_points:
        print(f"[SKIP] {summary_path}: n_epochs={n_epochs} > max_points={max_points}")
        return None

    t = np.array([p.mjd for p in points], dtype=float)
    y = np.array([p.rv for p in points], dtype=float)
    yerr = np.array([p.rv_err for p in points], dtype=float)
    rms_arr = np.array([p.rms for p in points], dtype=float)

    n_before = len(points)
    ok = np.isfinite(t) & np.isfinite(y)
    points_fit = [p for p, keep in zip(points, ok) if keep]
    t, y, yerr = _sanitize_rv_arrays(t, y, yerr, rms_fallback=rms_arr)
    if len(t) < min_points:
        print(
            f"[SKIP] {summary_path.name}: n_finite_rv={len(t)} < min_points={min_points} "
            f"(parsed {n_before} rows)"
        )
        return None
    if max_points is not None and len(t) > max_points:
        print(f"[SKIP] {summary_path.name}: n_finite_rv={len(t)} > max_points={max_points}")
        return None

    gaia_source_id = parse_object_id_from_summary(summary_path)
    gaia_nss = None
    nss_source: Optional[str] = None
    if use_gaia_nss:
        gaia_nss = load_nss_priors_from_summary(summary_path)
        if gaia_nss is not None:
            nss_source = "summary"
        elif query_gaia_online and gaia_source_id is not None:
            gaia_nss = fetch_gaia_nss_orbit(gaia_source_id, cache_path=gaia_cache_path)
            if gaia_nss is not None:
                nss_source = "online"

    fit_m1_msun = _finite_mass_value(table_m1_msun)
    if fit_m1_msun is None:
        fit_m1_msun = m1_msun
    fit_m2_msun = None
    fit_inclination_deg = None
    if gaia_nss is not None:
        if fit_m1_msun is None:
            fit_m1_msun = gaia_nss.get("m1_msun")
        fit_m2_msun = gaia_nss.get("m2_msun")
        fit_inclination_deg = gaia_nss.get("inclination_deg")
    if fit_m1_msun is None:
        fit_m1_msun = parse_m1_from_summary(summary_path)

    mass_meta = load_mass_priors_from_summary(summary_path)
    if fit_m1_msun is None:
        fit_m1_msun = mass_meta.get("m1_msun")
    if fit_inclination_deg is None:
        fit_inclination_deg = mass_meta.get("inclination_deg")
    fit_inclination_deg = resolve_inclination_deg_for_rv_mass(
        {
            "gaia_source_id": gaia_source_id,
            "inclination_deg_used": fit_inclination_deg,
            "gaia_nss": gaia_nss,
        },
        summary_path=summary_path,
    )

    fit_variants = fit_all_variants(
        t,
        y,
        yerr,
        gaia_nss if use_gaia_nss else None,
        period_min=period_min,
        period_max=period_max,
        period_prior_sigma=period_prior_sigma,
    )
    if "free" not in fit_variants:
        print(f"[SKIP] {summary_path.name}: free RV fit failed")
        return None

    params, _rep_free = fit_variants["free"]
    # Deep copy so report["fit_variants"]["free"] is not the same dict as report (JSON circular ref).
    report: dict = copy.deepcopy(_rep_free)
    report["summary_file"] = str(summary_path)
    report["gaia_source_id"] = gaia_source_id
    report["gaia_nss"] = gaia_nss
    report["nss_priors_source"] = nss_source
    report["observability_window"] = load_observability_window(gaia_source_id, observability_cache_path)
    report["fit_variants"] = {k: copy.deepcopy(v[1]) for k, v in fit_variants.items()}
    report["params_by_variant"] = {k: np.asarray(v[0], dtype=float).tolist() for k, v in fit_variants.items()}

    stem = report_stem(summary_path, gaia_source_id)
    out_png = out_dir / f"{stem}_keplerian_fit.png"
    data_png = out_dir / f"{stem}_rv_data.png"
    resid_png = out_dir / f"{stem}_keplerian_residuals.png"
    out_json = out_dir / f"{stem}_keplerian_fit.json"

    report["used_m1_msun"] = resolve_m1_msun_for_rv_mass(
        report,
        summary_path=summary_path,
        table_m1_msun=table_m1_msun,
    )
    report["table_m1_msun"] = None if table_m1_msun is None else float(table_m1_msun)
    report["used_m2_msun"] = None if fit_m2_msun is None else float(fit_m2_msun)
    report["inclination_deg_used"] = (
        None if fit_inclination_deg is None else float(fit_inclination_deg)
    )

    _, rep_free = fit_variants["free"]
    m1_for_masses = report["used_m1_msun"]
    m2sini, m2_at_i = rv_only_mass_estimates(rep_free, m1_for_masses, fit_inclination_deg)
    report["m2sini_msun"] = m2sini
    report["m2_given_inclination_msun"] = m2_at_i
    report["m2_astrometric_msun"] = report["used_m2_msun"]

    plot_multi_fit(summary_path, points_fit, fit_variants, report, out_png, m1_msun=m1_for_masses)
    plot_fit_residuals(
        summary_path, points_fit, fit_variants, report, resid_png, m1_msun=m1_for_masses
    )
    ours = our_telescope_points(points_fit)
    if len(ours) >= 2:
        plot_rv_data_only(summary_path, ours, report, data_png)

    if plots_root is not None and gaia_source_id:
        import shutil

        star_dir = plots_root / f"Gaia_DR3_{gaia_source_id}"
        star_dir.mkdir(parents=True, exist_ok=True)
        if data_png.is_file():
            shutil.copy2(data_png, star_dir / f"Gaia_DR3_{gaia_source_id}_rv_plot.png")
        shutil.copy2(out_png, star_dir / f"Gaia_DR3_{gaia_source_id}_keplerian_fit.png")
        if resid_png.is_file():
            shutil.copy2(resid_png, star_dir / f"Gaia_DR3_{gaia_source_id}_keplerian_residuals.png")

    out_json.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    try:
        from erfa import ErfaWarning  # type: ignore

        warnings.filterwarnings("ignore", category=ErfaWarning)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Fit APF RV summaries with a simple Keplerian model.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Root directory for summaries (searched recursively, e.g. output/Gaia_DR3_<id>/*_summary.txt).",
    )
    parser.add_argument("--summary", default=None, help="Path to one *_summary.txt file.")
    parser.add_argument("--all", action="store_true", help="Fit all summary files in --output-dir.")
    parser.add_argument("--reports-dir", default="rv_fit_reports", help="Output folder for fit plots/reports.")
    parser.add_argument("--min-points", type=int, default=7, help="Fit only if n_points >= min_points.")
    parser.add_argument("--max-points", type=int, default=None, help="Optional upper bound on n_points.")
    parser.add_argument("--m1-msun", type=float, default=None, help="Optional primary mass for M2 sin(i).")
    parser.add_argument("--period-min", type=float, default=None, help="Optional lower bound on period (days).")
    parser.add_argument("--period-max", type=float, default=None, help="Optional upper bound on period (days).")
    parser.add_argument("--period-prior", type=float, default=None, help="Optional period prior center (days), e.g. Gaia.")
    parser.add_argument(
        "--period-prior-sigma-lnP",
        type=float,
        default=0.15,
        help="Width of log-period prior (smaller = stronger).",
    )
    parser.add_argument("--fix-period", type=float, default=None, help="Fix period to this value (days).")
    parser.add_argument("--fix-e", type=float, default=None, help="Fix eccentricity to this value (0..1).")
    parser.add_argument(
        "--use-gaia-nss",
        action="store_true",
        help=(
            "Use NSS period/eccentricity from summary [GAIA METADATA] (period prior, e fixed unless --fix-e). "
            "No network unless --query-gaia-online."
        ),
    )
    parser.add_argument(
        "--query-gaia-online",
        action="store_true",
        help="If summary lacks NSS fields, query gaiadr3.nss_two_body_orbit / binary_masses via astroquery.",
    )
    parser.add_argument(
        "--gaia-cache",
        default=None,
        help="Cache for online NSS queries only. Default: <reports-dir>/gaia_nss_cache.json",
    )
    parser.add_argument(
        "--observability-cache",
        default=None,
        help="Optional observability cache JSON path. Default: <reports-dir>/observability_windows_cache.json",
    )
    parser.add_argument(
        "--data-csv",
        default=None,
        help="Website tables/data.csv for luminous M1 (Msun) per Gaia id (default: none).",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if report JSON is newer than summary.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary) if args.summary else None
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    gaia_cache_path = Path(args.gaia_cache) if args.gaia_cache else (reports_dir / "gaia_nss_cache.json")
    observability_cache_path = (
        Path(args.observability_cache)
        if args.observability_cache
        else (reports_dir / "observability_windows_cache.json")
    )

    targets = resolve_summary_files(output_dir, summary_path, args.all)
    if not targets:
        print("No summary files found.")
        return

    table_m1_map: Dict[str, float] = {}
    if args.data_csv:
        table_m1_map = load_table_m1_map_from_csv(Path(args.data_csv))
        if table_m1_map:
            print(f"Loaded luminous M1 for {len(table_m1_map)} stars from {args.data_csv}")

    if args.use_gaia_nss and args.query_gaia_online:
        ids: List[str] = []
        for p in targets:
            sid = parse_object_id_from_summary(p)
            if sid:
                ids.append(sid)
        prefetch_gaia_nss_bulk(ids, gaia_cache_path)

    combined = []
    skipped = 0
    n_targets = len(targets)
    if n_targets:
        rows = [count_pipeline_rows(p) for p in targets]
        ge_min = sum(1 for r in rows if r >= args.min_points)
        print(
            f"Fitting {n_targets} summaries (unique sources); "
            f"{ge_min} have >={args.min_points} pipeline rows"
        )
    for path in targets:
        sid = parse_object_id_from_summary(path)
        stem = report_stem(path, sid)
        out_json = reports_dir / f"{stem}_keplerian_fit.json"
        if (not args.force) and out_json.exists():
            try:
                if out_json.stat().st_mtime >= path.stat().st_mtime:
                    skipped += 1
                    continue
            except Exception:
                pass
        report = run_one(
            path,
            reports_dir,
            args.min_points,
            args.max_points,
            args.m1_msun,
            args.period_min,
            args.period_max,
            args.period_prior,
            args.period_prior_sigma_lnP,
            args.fix_period,
            args.fix_e,
            args.use_gaia_nss,
            gaia_cache_path,
            observability_cache_path,
            args.query_gaia_online,
            plots_root=output_dir,
            table_m1_msun=table_m1_map.get(str(sid)) if sid else None,
        )
        if report is None:
            skipped += 1
            continue
        combined.append(report)
        print(
            f"[OK] {path.name}: P={report['P_days']:.3f} d, K={report['K_kms']:.3f} km/s, "
            f"e={report['e']:.3f}, next_max={report['next_rv_max_mjd']:.3f}"
        )

    combined_path = reports_dir / "apf_keplerian_fit_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2))
    print(f"Saved {len(combined)} fit reports to {reports_dir} (skipped {skipped}).")


if __name__ == "__main__":
    main()
