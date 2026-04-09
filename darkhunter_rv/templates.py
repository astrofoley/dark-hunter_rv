# templates.py
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import astropy.io.fits as fits
import numpy as np

from collections import defaultdict

from . import config, continuum, physics

# Global cache of (wave, norm_flux) per PHOENIX key (Teff, logg, [M/H], vsini); reused across
# :func:`build_template_bank` calls in the same Python process so multi-star in-process campaigns
# do not re-read FITS for the same model keys.
MODEL_CACHE: dict = {}

_WAVE_HIRES_CACHE: np.ndarray | None = None
_WAVE_HIRES_PATH: Path | None = None

# Hot-star PHOENIX: exclude this half-width (Å) around each entry in STRONG_LINES when fitting the
# template upper-envelope continuum. The default 20 Å mask leaves the entire broad Balmer wing in
# the “continuum” sample, so flux/continuum ≈ 1 and FFT templates plot as a flat line.
_HOT_PHOENIX_LINE_MASK_A = 175.0


def _norm_template_flux_fft(wave: np.ndarray, f_broad: np.ndarray, teff: float) -> np.ndarray:
    """
    Continuum-normalize PHOENIX for FFT matching. Cool stars use global continuum plus
    ``renormalize_local``; hot stars (Teff above threshold) use **global continuum division only**
    scaled by a high percentile, so very broad Balmer features are not flattened by the local
    upper-envelope polynomial.

    For hot stars the global continuum uses a **wide** mask around strong lines so broad Balmer
    wings are not mistaken for continuum (see :data:`_HOT_PHOENIX_LINE_MASK_A`).
    """
    wave = np.asarray(wave, float)
    f_broad = np.asarray(f_broad, float)
    hot = float(teff) > float(config.HOT_STAR_TEFF_THRESHOLD)
    line_mask_a = _HOT_PHOENIX_LINE_MASK_A if hot else 20.0
    win_frac = 0.028 if hot else 0.02
    cont = continuum.compute_template_global_continuum(
        wave, f_broad, mask_width=line_mask_a, window_frac=win_frac
    )
    floor = float(np.nanpercentile(cont, 5)) * 1e-6 + 1e-24
    raw = f_broad / np.maximum(cont, floor)
    if hot:
        scale = float(np.nanpercentile(raw, 90))
        scale = max(scale, float(np.nanmedian(raw)) * 0.45, 1e-30)
        out = raw / scale
        return np.clip(out, 0.03, 2.8).astype(float)
    _, f_norm, _ = continuum.renormalize_local(wave, f_broad, cont, poly_order=2)
    return np.asarray(f_norm, dtype=float)


def parse_phoenix_dirname(dirname):
    name = dirname.lower()
    if "phoenixm" in name:
        return -float(name.split("phoenixm")[1]) / 10.0
    if "phoenixp" in name:
        return float(name.split("phoenixp")[1]) / 10.0
    return 0.0


def _hires_wave_file(base_dir: Path) -> Path | None:
    p = base_dir / "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    if p.is_file():
        return p
    matches = sorted(base_dir.glob("WAVE_PHOENIX*.fits"))
    return matches[0] if matches else None


def _hires_model_root(base_dir: Path) -> Path | None:
    r = base_dir / "PHOENIX-ACES-AGSS-COND-2011"
    return r if r.is_dir() else None


def _is_hires_phoenix_layout(base_dir: Path) -> bool:
    return _hires_wave_file(base_dir) is not None and _hires_model_root(base_dir) is not None


def _load_hires_wavelength(wave_path: Path) -> np.ndarray:
    global _WAVE_HIRES_CACHE, _WAVE_HIRES_PATH
    if _WAVE_HIRES_CACHE is not None and _WAVE_HIRES_PATH == wave_path:
        return _WAVE_HIRES_CACHE
    with fits.open(wave_path) as hdul:
        w = np.asarray(hdul[0].data, dtype=float)
    _WAVE_HIRES_CACHE = w
    _WAVE_HIRES_PATH = wave_path
    return w


def _parse_z_metallicity_folder(name: str) -> float | None:
    if not name.startswith("Z"):
        return None
    try:
        return float(name[1:])
    except ValueError:
        return None


_LTE_GLUE_MH = re.compile(r"^(\d+\.\d{2})([+-]\d+\.\d+)$")


def _parse_lte_hires_filename(fname: str) -> tuple[float, float, float] | None:
    """
    Parse PHOENIX HiRes names, e.g.
    lte12000-3.00-0.0.PHOENIX-... (solar [M/H] uses a third dash field)
    lte12000-3.50+0.5.PHOENIX-... ([M/H] sign glued to logg field for non-solar grids)
    """
    if not fname.startswith("lte") or not fname.lower().endswith(".fits"):
        return None
    # Drop .fits; Teff/logg contain dots. HiRes names look like
    # lte12000-4.00-0.0.PHOENIX-ACES-...-HiRes.fits — keep only the lte… segment before .PHOENIX.
    no_fits = fname[:-5] if fname.lower().endswith(".fits") else fname
    stem = re.split(r"\.phoenix", no_fits, maxsplit=1, flags=re.I)[0]
    body = stem[3:]  # after 'lte'
    parts = body.split("-", 2)
    try:
        teff = float(parts[0])
        if len(parts) == 3:
            logg = float(parts[1])
            mh = float(parts[2])
            return teff, logg, mh
        if len(parts) == 2:
            m = _LTE_GLUE_MH.match(parts[1])
            if m:
                logg = float(m.group(1))
                mh = float(m.group(2))
                return teff, logg, mh
    except (ValueError, IndexError):
        return None
    return None


def _broadening_velocity_grid(vsini_proxy: float | None, *, wide: bool) -> list[float]:
    vp = float(vsini_proxy if vsini_proxy is not None else 10.0)
    vp = max(vp, 0.0)
    if not wide:
        # Small spread around the proxy only; avoid 1.2× which over-broadens slow rotators.
        lo = max(0.0, vp - min(5.0, 0.15 * max(vp, 1.0)))
        hi = vp + min(8.0, 0.2 * max(vp, 1.0))
        return sorted({float(lo), float(vp), float(hi)})
    # Wide grid: cap extent so the bank is not dominated by extreme rotational smearing.
    cap = float(getattr(config, "VSINI_WIDE_GRID_CAP_KMS", 160.0))
    vmax = min(max(vp * 1.45, vp + 25.0), cap)
    vmax = max(vmax, min(vp + 35.0, cap))
    pts = np.linspace(0.0, vmax, 6)
    return sorted({float(round(x, 4)) for x in pts})


def _build_template_bank_hires(
    teff: float,
    vsini_proxy: float,
    metallicity: float,
    logg: float,
    wave_range: tuple[float, float],
    air: bool,
    base_dir: Path,
    *,
    hot_spectrum: bool = False,
    template_grid_wide: bool = False,
) -> dict:
    global MODEL_CACHE
    templates: dict = {}

    wave_path = _hires_wave_file(base_dir)
    model_root = _hires_model_root(base_dir)
    assert wave_path is not None and model_root is not None

    wave_full = _load_hires_wavelength(wave_path)
    if vsini_proxy is None:
        vsini_proxy = 10.0
    wide = bool(hot_spectrum or template_grid_wide)
    vb_values = _broadening_velocity_grid(vsini_proxy, wide=wide)
    mh_tol = 0.52 if wide else 0.36
    dg_max = 2.05 if wide else 1.55
    dt_max = 6000.0 if wide else 5500.0
    n_keep = 32 if wide else 24

    entries: list[tuple[float, float, float, float, float, Path]] = []
    for zsub in sorted(model_root.iterdir()):
        if not zsub.is_dir():
            continue
        mh_dir = _parse_z_metallicity_folder(zsub.name)
        if mh_dir is None:
            continue
        if abs(mh_dir - metallicity) > mh_tol:
            continue
        for fp in zsub.glob("lte*.fits"):
            parsed = _parse_lte_hires_filename(fp.name)
            if parsed is None:
                continue
            t_m, g_m, mh_f = parsed
            if abs(mh_f - mh_dir) > 0.09:
                continue
            dg = abs(g_m - logg)
            if dg > dg_max:
                continue
            dt = abs(t_m - teff)
            if dt > dt_max:
                continue
            entries.append((dt, dg, t_m, g_m, mh_f, fp))

    entries.sort(key=lambda x: (x[0], x[1]))
    # Each entry × N vb values: FFT picks the best key (Teff / log g / [M/H] / vsini).
    entries = entries[:n_keep]
    if not entries:
        logging.warning(
            "HiRes PHOENIX grid under %s: no models within Teff/logg/[M/H] cuts (teff=%s logg=%s [M/H]=%s).",
            model_root,
            teff,
            logg,
            metallicity,
        )
        return {}

    for _dt, _dg, t_file, g_val, mval, full_path in entries:
        try:
            with fits.open(full_path) as hdul:
                flux_full = np.asarray(hdul[0].data, dtype=float)
        except Exception:
            continue
        if flux_full.shape != wave_full.shape:
            logging.warning("HiRes flux length mismatch %s vs wave", full_path.name)
            continue

        mask = (wave_full >= wave_range[0]) & (wave_full <= wave_range[1])
        if int(np.sum(mask)) < 50:
            continue
        wave = wave_full[mask]
        flux = flux_full[mask]
        if air:
            wave = physics.vac_to_air(wave)
        if np.all(flux == 0):
            continue
        # Million-point grids make rotational broadening and caching very slow; uniform downsample.
        max_pts = 32000
        if len(wave) > max_pts:
            step = max(1, int(np.ceil(len(wave) / max_pts)))
            wave = wave[::step]
            flux = flux[::step]

        for vb in vb_values:
            key = (t_file, g_val, mval, vb)
            if key in MODEL_CACHE:
                templates[key] = MODEL_CACHE[key]
                continue
            f_broad = physics.broaden_spectrum(wave, flux, vb)
            f_norm = _norm_template_flux_fft(wave, f_broad, float(teff))
            MODEL_CACHE[key] = (wave, f_norm)
            templates[key] = (wave, f_norm)

    logging.info("HiRes PHOENIX: loaded %d template keys from %s", len(templates), base_dir)
    return templates


def load_hires_phoenix_raw_nearest(
    teff: float,
    logg: float,
    metallicity: float,
    wave_range: tuple[float, float],
    *,
    air: bool = True,
    mh_tol: float = 0.52,
    dg_max: float = 2.05,
    dt_max: float = 8000.0,
    max_points: int = 32000,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]] | None:
    """
    Load the **single** nearest HiRes PHOENIX atmosphere (no rotational broadening, no FFT normalization).

    Returns ``(wavelength, flux, (teff_file, logg_file, mh_file))`` clipped to ``wave_range`` (with the
    same downsampling cap used when building template banks). Intended for validation / parameter sweeps.
    """
    base_dir = Path(config.PHOENIX_BASE_DIR)
    if not base_dir.exists() or not _is_hires_phoenix_layout(base_dir):
        logging.warning("HiRes PHOENIX layout not found under %s", base_dir)
        return None

    wave_path = _hires_wave_file(base_dir)
    model_root = _hires_model_root(base_dir)
    assert wave_path is not None and model_root is not None

    wave_full = _load_hires_wavelength(wave_path)
    entries: list[tuple[float, float, float, float, float, Path]] = []
    for zsub in sorted(model_root.iterdir()):
        if not zsub.is_dir():
            continue
        mh_dir = _parse_z_metallicity_folder(zsub.name)
        if mh_dir is None:
            continue
        if abs(mh_dir - float(metallicity)) > mh_tol:
            continue
        for fp in zsub.glob("lte*.fits"):
            parsed = _parse_lte_hires_filename(fp.name)
            if parsed is None:
                continue
            t_m, g_m, mh_f = parsed
            if abs(mh_f - mh_dir) > 0.09:
                continue
            dg = abs(g_m - float(logg))
            if dg > dg_max:
                continue
            dt = abs(t_m - float(teff))
            if dt > dt_max:
                continue
            entries.append((dt, dg, t_m, g_m, mh_f, fp))

    entries.sort(key=lambda x: (x[0], x[1]))
    if not entries:
        logging.warning(
            "load_hires_phoenix_raw_nearest: no model within cuts (teff=%s logg=%s [M/H]=%s under %s).",
            teff,
            logg,
            metallicity,
            model_root,
        )
        return None

    _dt, _dg, t_file, g_val, mval, full_path = entries[0]
    try:
        with fits.open(full_path) as hdul:
            flux_full = np.asarray(hdul[0].data, dtype=float)
    except Exception as ex:
        logging.warning("Failed to read %s: %s", full_path, ex)
        return None
    if flux_full.shape != wave_full.shape:
        logging.warning("HiRes flux length mismatch %s vs wave", full_path.name)
        return None

    mask = (wave_full >= wave_range[0]) & (wave_full <= wave_range[1])
    if int(np.sum(mask)) < 50:
        logging.warning("load_hires_phoenix_raw_nearest: <50 pixels in wave_range %s", wave_range)
        return None
    wave = wave_full[mask]
    flux = flux_full[mask]
    if air:
        wave = physics.vac_to_air(wave)
    if np.all(flux == 0):
        return None
    n = len(wave)
    if n > int(max_points):
        step = max(1, int(np.ceil(n / float(max_points))))
        wave = wave[::step]
        flux = flux[::step]
    return wave, flux, (float(t_file), float(g_val), float(mval))


def _build_template_bank_legacy(
    teff: float,
    vsini_proxy: float,
    metallicity: float,
    logg: float,
    wave_range: tuple[float, float],
    air: bool,
    base_dir: Path,
    *,
    template_grid_wide: bool = False,
) -> dict:
    global MODEL_CACHE
    templates: dict = {}

    if vsini_proxy is None:
        vsini_proxy = 10.0
    vb_values = _broadening_velocity_grid(vsini_proxy, wide=bool(template_grid_wide))

    for d in os.listdir(base_dir):
        if not d.lower().startswith("phoenix"):
            continue
        mval = parse_phoenix_dirname(d)
        if abs(mval - metallicity) > 0.5:
            continue

        path_d = base_dir / d
        if not path_d.is_dir():
            continue

        for fname in os.listdir(path_d):
            if not fname.endswith(".fits"):
                continue
            try:
                t_file = float(fname.split("_")[1].replace(".fits", ""))
                if abs(t_file - teff) > 1000:
                    continue

                full_path = path_d / fname
                with fits.open(full_path) as hdul:
                    data = hdul[1].data
                    wave = data["WAVELENGTH"]
                    if air:
                        wave = physics.vac_to_air(wave)

                    mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
                    wave = wave[mask]

                    for col in hdul[1].columns.names:
                        if not col.startswith("g"):
                            continue
                        g_val = float(col[1:]) / 10.0
                        if abs(g_val - logg) > 0.5:
                            continue

                        flux = data[col][mask]
                        if np.all(flux == 0):
                            continue

                        for vb in vb_values:
                            key = (t_file, g_val, mval, vb)
                            if key in MODEL_CACHE:
                                templates[key] = MODEL_CACHE[key]
                            else:
                                f_broad = physics.broaden_spectrum(wave, flux, vb)
                                f_norm = _norm_template_flux_fft(wave, f_broad, float(teff))

                                MODEL_CACHE[key] = (wave, f_norm)
                                templates[key] = (wave, f_norm)

            except Exception:
                continue

    return templates


def build_template_bank(
    teff,
    vsini_proxy,
    metallicity=0.0,
    logg=4.5,
    wave_range=(3300, 9000),
    air=True,
    *,
    hot_spectrum: bool = False,
    template_grid_wide: bool = False,
):
    base_dir = Path(config.PHOENIX_BASE_DIR)

    if not base_dir.exists():
        logging.warning("Phoenix dir %s not found.", base_dir)
        return {}

    if _is_hires_phoenix_layout(base_dir):
        return _build_template_bank_hires(
            float(teff),
            vsini_proxy,
            float(metallicity),
            float(logg),
            wave_range,
            air,
            base_dir,
            hot_spectrum=bool(hot_spectrum),
            template_grid_wide=bool(template_grid_wide),
        )

    return _build_template_bank_legacy(
        float(teff),
        vsini_proxy,
        float(metallicity),
        float(logg),
        wave_range,
        air,
        base_dir,
        template_grid_wide=bool(hot_spectrum or template_grid_wide),
    )


_SESSION_BANK_CACHE: dict[tuple, dict] = {}


def build_template_bank_cached(
    teff,
    vsini_proxy,
    metallicity=0.0,
    logg=4.5,
    wave_range=(3300, 9000),
    air=True,
    *,
    hot_spectrum: bool = False,
    template_grid_wide: bool = False,
) -> dict:
    """
    Same as :func:`build_template_bank`, but reuses the in-process dict when priors match a prior
    call (vsini rounded to 0.5 km/s). ``MODEL_CACHE`` still deduplicates FITS loads; this avoids
    rebuilding the key list and copying references for every spectrum in a campaign.
    """
    vp = float(vsini_proxy if vsini_proxy is not None else 10.0)
    key = (
        int(round(float(teff))),
        round(vp * 2.0) / 2.0,
        round(float(metallicity), 2),
        round(float(logg), 2),
        bool(hot_spectrum),
        bool(template_grid_wide),
        tuple(wave_range),
        bool(air),
    )
    if key in _SESSION_BANK_CACHE:
        return _SESSION_BANK_CACHE[key]
    bank = build_template_bank(
        teff,
        vsini_proxy,
        metallicity=metallicity,
        logg=logg,
        wave_range=wave_range,
        air=air,
        hot_spectrum=hot_spectrum,
        template_grid_wide=template_grid_wide,
    )
    _SESSION_BANK_CACHE[key] = bank
    return bank


def template_key_stellar_tuple(key: object) -> tuple[float, float, float] | None:
    """PHOENIX bank keys are (Teff, log g, [M/H], v_sini); return the atmosphere triple or None."""
    if isinstance(key, tuple) and len(key) == 4:
        try:
            return (float(key[0]), float(key[1]), float(key[2]))
        except (TypeError, ValueError):
            return None
    return None


def max_vsini_variants_per_atmosphere(templates: dict) -> int:
    """Largest number of rotational-broadening keys sharing one (Teff, log g, [M/H])."""
    counts: dict[tuple[float, float, float], int] = defaultdict(int)
    for k in templates:
        st = template_key_stellar_tuple(k)
        if st is not None:
            counts[st] += 1
    return max(counts.values()) if counts else 1


def coarse_fft_subbank(templates: dict, vsini_proxy: float | None) -> dict:
    """
    One template per (Teff, log g, [M/H]): the broadening velocity closest to ``vsini_proxy``.
    Used for a fast first FFT pass in stellar-parameter space before expanding vsini.
    """
    vp = float(vsini_proxy if vsini_proxy is not None else 10.0)
    by_stellar: dict[tuple[float, float, float], list[object]] = defaultdict(list)
    for k in templates:
        st = template_key_stellar_tuple(k)
        if st is None:
            continue
        if not isinstance(k, tuple) or len(k) != 4:
            continue
        by_stellar[st].append(k)
    out: dict = {}
    for _st, keys in by_stellar.items():
        best_k = min(keys, key=lambda kk: abs(float(kk[3]) - vp))
        out[best_k] = templates[best_k]
    return out


def narrow_fft_subbank(templates: dict) -> dict:
    """
    One template per (Teff, log g, [M/H]): the **lowest** broadening velocity in the bank for that triple.

    Used for exposure-level atmosphere scoring so a bad ``vsini_proxy`` does not smear **every** candidate
    before RMS comparison (``coarse_fft_subbank`` matches proxy and can force all trials to ~200 km/s).
    """
    by_stellar: dict[tuple[float, float, float], list[object]] = defaultdict(list)
    for k in templates:
        st = template_key_stellar_tuple(k)
        if st is None:
            continue
        if not isinstance(k, tuple) or len(k) != 4:
            continue
        by_stellar[st].append(k)
    out: dict = {}
    for _st, keys in by_stellar.items():
        best_k = min(keys, key=lambda kk: float(kk[3]))
        out[best_k] = templates[best_k]
    return out


def refined_fft_subbank(templates: dict, stellar_triples: set[tuple[float, float, float]]) -> dict:
    """All bank entries whose atmosphere triple is in ``stellar_triples`` (full vsini set for each)."""
    out: dict = {}
    for k, v in templates.items():
        st = template_key_stellar_tuple(k)
        if st is not None and st in stellar_triples:
            out[k] = v
    return out
