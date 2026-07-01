"""
Parameterized echelle blaze profile for a single order (Hβ lane first).

Model: amplitude × [sinc(π (λ − λ₀) / w) / (π (λ − λ₀) / w)]^p  with sinc(0) = 1.

Fit on many weak-line spectra: per-star amplitude, shared (λ₀, w, p). Pixels near strong
lines are masked before stacking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from darkhunter_rv import config, continuum, io_utils, rv_core

logger = logging.getLogger(__name__)

HB_REST_A = float(rv_core.HB_REST_A)


def _sinc_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.ones_like(x)
    m = np.abs(x) > 1e-12
    y[m] = np.sin(np.pi * x[m]) / (np.pi * x[m])
    return y


def eval_blaze_sinc2(
    wavelength: np.ndarray,
    center: float,
    width: float,
    *,
    power: float = 2.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Parameterized blaze: amplitude × sinc²-like profile in (λ − center) / width."""
    w = np.asarray(wavelength, float)
    x = (w - float(center)) / max(float(width), 1e-9)
    return float(amplitude) * (np.abs(_sinc_pi(x)) ** float(power))


def blaze_line_mask(
    wavelength: np.ndarray,
    *,
    rest_lines: list[float] | None = None,
    half_width_angstrom: float = 22.0,
) -> np.ndarray:
    """True where pixel is usable for blaze fitting (far from strong lines)."""
    rests = rest_lines if rest_lines is not None else list(continuum.STRONG_LINES)
    w = np.asarray(wavelength, float)
    ok = np.isfinite(w)
    hw = float(half_width_angstrom)
    for rest in rests:
        ok &= ~((w >= float(rest) - hw) & (w <= float(rest) + hw))
    return ok


def find_order_covering_rest(
    spec_data: dict,
    rest: float,
    *,
    bad_orders: list[int] | None = None,
) -> int | None:
    bad = set(bad_orders or [])
    for o in sorted(spec_data.keys()):
        if int(o) in bad:
            continue
        w = np.asarray(spec_data[o]["wavelength"], float)
        if w.size < 10:
            continue
        if float(np.min(w)) <= rest <= float(np.max(w)):
            return int(o)
    return None


def order_covers_strong_line(
    wavelength_min: float,
    wavelength_max: float,
    *,
    rest_lines: list[float] | None = None,
) -> bool:
    """True if any entry in STRONG_LINES falls inside the order wavelength span."""
    rests = rest_lines if rest_lines is not None else list(continuum.STRONG_LINES)
    wmn, wmx = float(wavelength_min), float(wavelength_max)
    return any(wmn <= float(rest) <= wmx for rest in rests)


def list_clean_orders(
    spec_data: dict,
    *,
    bad_orders: list[int] | None = None,
    rest_lines: list[float] | None = None,
) -> list[tuple[int, float, float]]:
    """Echelle orders with no STRONG_LINES hit; returns (order, wmin, wmax)."""
    bad = set(bad_orders or [])
    out: list[tuple[int, float, float]] = []
    for o in sorted(spec_data.keys()):
        if int(o) in bad:
            continue
        w = np.asarray(spec_data[o]["wavelength"], float)
        if w.size < 10:
            continue
        wmn, wmx = float(np.min(w)), float(np.max(w))
        if not order_covers_strong_line(wmn, wmx, rest_lines=rest_lines):
            out.append((int(o), wmn, wmx))
    return out


def pick_clean_order_near_wavelength(
    spec_data: dict,
    target_angstrom: float,
    *,
    bad_orders: list[int] | None = None,
) -> tuple[int, float, float] | None:
    """Choose the clean order whose midpoint is closest to ``target_angstrom``."""
    clean = list_clean_orders(spec_data, bad_orders=bad_orders)
    if not clean:
        return None
    target = float(target_angstrom)
    o, wmn, wmx = min(clean, key=lambda row: abs(0.5 * (row[1] + row[2]) - target))
    return o, wmn, wmx


def hbeta_absorption_depth_raw(
    wavelength: np.ndarray,
    flux: np.ndarray,
    *,
    rest: float = HB_REST_A,
    core_half_angstrom: float = 18.0,
    wing_half_angstrom: float = 70.0,
) -> float:
    """
    Shallow-line proxy on raw (blaze-shaped) flux.

    Fits a local linear continuum in the wings (excluding the line core), then
    returns 1 − min(core flux) / median(continuum in core).
    """
    w = np.asarray(wavelength, float)
    f = np.asarray(flux, float)
    m_wing = (w >= rest - wing_half_angstrom) & (w <= rest + wing_half_angstrom) & np.isfinite(f)
    m_core = (w >= rest - core_half_angstrom) & (w <= rest + core_half_angstrom) & np.isfinite(f)
    m_fit = m_wing & ~m_core
    if int(np.sum(m_core)) < 5 or int(np.sum(m_fit)) < 12:
        return float("nan")
    coef = np.polyfit(w[m_fit], f[m_fit], 1)
    cont_core = np.polyval(coef, w[m_core])
    env = float(np.nanmedian(cont_core))
    if not np.isfinite(env) or env <= 0:
        return float("nan")
    return float(1.0 - np.nanmin(f[m_core]) / env)


@dataclass
class OrderBlazeModel:
    """Shared blaze for one echelle order (wavelengths in Å)."""

    echelle_order: int
    model: str  # "sinc2"
    center_angstrom: float
    width_angstrom: float
    power: float
    n_spectra_fit: int
    wavelength_min: float
    wavelength_max: float
    rest_line_angstrom: float = HB_REST_A
    line_mask_half_width_angstrom: float = 22.0

    def blaze_on_grid(self, wavelength: np.ndarray) -> np.ndarray:
        return eval_blaze_sinc2(
            wavelength,
            self.center_angstrom,
            self.width_angstrom,
            power=self.power,
            amplitude=1.0,
        )

    def correct_flux(self, wavelength: np.ndarray, flux: np.ndarray) -> np.ndarray:
        w = np.asarray(wavelength, float)
        f = np.asarray(flux, float)
        b = self.blaze_on_grid(w)
        b = np.maximum(b, 1e-9 * float(np.nanmax(b)))
        return f / b

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> OrderBlazeModel:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> OrderBlazeModel:
        return cls.from_json_dict(json.loads(Path(path).read_text()))


def _interp_to_grid(w: np.ndarray, f: np.ndarray, grid: np.ndarray) -> np.ndarray:
    o = np.argsort(w)
    ws, fs = w[o], f[o]
    m = np.isfinite(ws) & np.isfinite(fs)
    if int(np.sum(m)) < 8:
        return np.full(grid.shape, np.nan, dtype=float)
    return np.interp(grid, ws[m], fs[m], left=np.nan, right=np.nan)


def fit_order_blaze_from_profiles(
    profiles: list[tuple[np.ndarray, np.ndarray]],
    echelle_order: int,
    *,
    line_mask_half_width: float = 22.0,
    rest_lines: list[float] | None = None,
) -> OrderBlazeModel | None:
    """
    Fit shared sinc² blaze to multiple raw-flux profiles on the same order.

    ``profiles`` are (wavelength, flux) per exposure; wavelengths may differ slightly.
    """
    if len(profiles) < 3:
        logger.warning("need at least 3 profiles to fit blaze; got %d", len(profiles))
        return None

    # Common grid: median sampling of all wavelengths
    all_w: list[np.ndarray] = []
    for w, _f in profiles:
        all_w.append(np.asarray(w, float))
    grid = np.unique(np.concatenate(all_w))
    grid.sort()
    if grid.size < 40:
        return None

    mask_fit = blaze_line_mask(grid, rest_lines=rest_lines, half_width_angstrom=line_mask_half_width)
    if int(np.sum(mask_fit)) < 30:
        return None

    stack = np.vstack([_interp_to_grid(w, f, grid) for w, f in profiles])
    median_f = np.nanmedian(stack, axis=0)
    m = mask_fit & np.isfinite(median_f) & (median_f > 0)
    if int(np.sum(m)) < 25:
        return None

    wg, yg = grid[m], median_f[m]
    i_peak = int(np.argmax(yg))
    center0 = float(wg[i_peak])
    # FWHM-ish width from half-max wings
    half = 0.5 * float(yg[i_peak])
    above = wg[yg >= half]
    width0 = float(0.5 * (np.max(above) - np.min(above))) if above.size > 4 else 80.0
    width0 = max(width0, 15.0)

    def model(lam, center, width, power, amp):
        return eval_blaze_sinc2(lam, center, width, power=power, amplitude=amp)

    try:
        popt, _pcov = curve_fit(
            model,
            wg,
            yg,
            p0=[center0, width0, 2.0, float(np.max(yg))],
            bounds=(
                [float(np.min(wg)), 5.0, 1.2, 0.01 * float(np.max(yg))],
                [float(np.max(wg)), 500.0, 4.5, 100.0 * float(np.max(yg))],
            ),
            maxfev=20000,
        )
    except Exception as ex:
        logger.warning("blaze curve_fit failed: %s", ex)
        return None

    center, width, power, _amp = map(float, popt)
    return OrderBlazeModel(
        echelle_order=int(echelle_order),
        model="sinc2",
        center_angstrom=center,
        width_angstrom=width,
        power=power,
        n_spectra_fit=len(profiles),
        wavelength_min=float(grid[0]),
        wavelength_max=float(grid[-1]),
        rest_line_angstrom=HB_REST_A,
        line_mask_half_width_angstrom=float(line_mask_half_width),
    )


def median_profile_and_rms(
    profiles: list[tuple[np.ndarray, np.ndarray]],
    grid: np.ndarray,
    *,
    line_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.vstack([_interp_to_grid(w, f, grid) for w, f in profiles])
    if line_mask is not None:
        stack[:, ~line_mask] = np.nan
    med = np.nanmedian(stack, axis=0)
    rms = np.nanstd(stack, axis=0)
    return med, rms


def strong_lines_in_span(wavelength_min: float, wavelength_max: float) -> list[float]:
    """STRONG_LINES entries that fall inside an order wavelength span."""
    wmn, wmx = float(wavelength_min), float(wavelength_max)
    return [float(r) for r in continuum.STRONG_LINES if wmn <= float(r) <= wmx]


@dataclass
class BlazeCalibration:
    """Per-order sinc² blaze models for one instrument."""

    instrument: str
    n_spectra_fit: int
    min_snr: float
    orders: dict[int, OrderBlazeModel]

    def model_for_order(self, echelle_order: int) -> OrderBlazeModel | None:
        return self.orders.get(int(echelle_order))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "n_spectra_fit": self.n_spectra_fit,
            "min_snr": self.min_snr,
            "orders": {str(k): v.to_json_dict() for k, v in sorted(self.orders.items())},
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> BlazeCalibration:
        orders = {
            int(k): OrderBlazeModel.from_json_dict(v)
            for k, v in (d.get("orders") or {}).items()
        }
        return cls(
            instrument=str(d.get("instrument", "APF")),
            n_spectra_fit=int(d.get("n_spectra_fit", 0)),
            min_snr=float(d.get("min_snr", 0.0)),
            orders=orders,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> BlazeCalibration:
        return cls.from_json_dict(json.loads(Path(path).read_text()))


def build_blaze_calibration(
    spec_paths: list[Path],
    instrument,
    *,
    min_snr: float = 3.5,
    overlap: dict[str, dict] | None = None,
    line_mask_half_width: float = 22.0,
    min_profiles: int = 8,
) -> BlazeCalibration:
    """
    Fit a shared sinc² blaze per echelle order from many spectra.

    Orders containing strong lines mask only the lines present in that order span.
    """
    from collections import defaultdict

    overlap = overlap or {}
    bad = set(instrument.bad_orders or [])
    by_order: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    n_used = 0

    for spec_path in spec_paths:
        stem = Path(spec_path).stem
        meta = overlap.get(stem, {})
        snr = float(meta.get("median_mask_ccf_peak_snr", np.nan))
        if overlap and np.isfinite(snr) and snr < float(min_snr):
            continue
        try:
            _hdr, spec_data = io_utils.read_spectrum(str(spec_path))
        except Exception as ex:
            logger.debug("skip %s: %s", stem, ex)
            continue
        n_used += 1
        for o in spec_data:
            if int(o) in bad:
                continue
            w = np.asarray(spec_data[o]["wavelength"], float)
            f = np.asarray(spec_data[o]["flux"], float)
            if w.size < 20 or not np.any(np.isfinite(f) & (f > 0)):
                continue
            by_order[int(o)].append((w, f))

    models: dict[int, OrderBlazeModel] = {}
    for o in sorted(by_order):
        profiles = by_order[o]
        if len(profiles) < int(min_profiles):
            logger.debug("order %d: only %d profiles", o, len(profiles))
            continue
        wmins = [float(np.min(w)) for w, _f in profiles]
        wmaxs = [float(np.max(w)) for w, _f in profiles]
        wmn, wmx = float(np.min(wmins)), float(np.max(wmaxs))
        rests = strong_lines_in_span(wmn, wmx)
        model = fit_order_blaze_from_profiles(
            profiles,
            o,
            line_mask_half_width=float(line_mask_half_width),
            rest_lines=rests if rests else [],
        )
        if model is not None:
            models[o] = model

    return BlazeCalibration(
        instrument=str(getattr(instrument, "name", "APF")),
        n_spectra_fit=n_used,
        min_snr=float(min_snr),
        orders=models,
    )
