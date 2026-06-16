#!/usr/bin/env python3
"""
Fixed atmosphere (default: K6 V MS), scan rotational broadening only.

Loads the nearest HiRes PHOENIX model to (Teff, log g, [M/H]), applies ``physics.broaden_spectrum`` for
each trial v sin i, FFT-normalizes like the pipeline, then compares to one spectral chunk.

Writes one PNG per broadening trial (observation vs shifted+LSF template) plus a summary metric plot.

Example (Gaia DR3 468391369318487040 epoch 1, chunk 22; paths relative to repo root)::

  python validation/fit_k6_template_vsini_sweep.py \\
    --spectrum data/Gaia_DR3_468391369318487040_epoch_1.txt \\
    --instrument APF --chunk-key 22 \\
    --obs-teff 5117.70166016 \\
    --diagnostics-csv output/Gaia_DR3_468391369318487040_epoch_1_diagnostics.csv \\
    --out-dir validation_output/k6_vsini_chunk22_Gaia_DR3_468391369318487040_epoch_1

Override the default K6 dwarf priors with ``--teff``, ``--logg``, ``--mh`` if needed.
``--obs-teff`` should match the **target** star (continuum mode); template grid still uses ``--teff``/``--logg``/``--mh``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import chunking, config, continuum, instruments, io_utils, physics, rv_core, templates

# Pecaut & Mamajek (2013) style: K6V ~ 4120 K, MS log g ~ 4.5
DEFAULT_K6_MS_TEFF = 4120.0
DEFAULT_K6_MS_LOGG = 4.5
DEFAULT_K6_MS_MH = 0.0


def _parse_vb_list(s: str) -> list[float]:
    s = str(s).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(x) for x in parts]


def _continuum_kw(teff: float) -> dict:
    hot = float(teff) > float(config.HOT_STAR_TEFF_THRESHOLD)
    return {
        "continuum_mode": "spline",
        "exclude_near_lines_width": float(
            config.HOT_SPLINE_EXCLUDE_NEAR_LINES_WIDTH if hot else config.COOL_SPLINE_EXCLUDE_NEAR_LINES_WIDTH
        ),
    }


def _mask_rv_from_diagnostics(
    diagnostics_csv: Path,
    chunk_key: str,
    inst: instruments.InstrumentProfile,
    *,
    no_bias: bool,
) -> float | None:
    df = pd.read_csv(diagnostics_csv)
    m = df.loc[df["method"] == "mask_ccf", ["chunk_key", "rv_kms"]].dropna(subset=["rv_kms"])
    row = m.loc[m["chunk_key"].astype(str) == str(chunk_key)]
    if row.empty:
        return None
    rv = float(row["rv_kms"].iloc[0])
    if no_bias or not inst.bias_file:
        return rv
    bias = io_utils.read_bias(inst.bias_file)
    bvec = io_utils.lookup_bias(bias, chunk_key)
    b0 = float(bvec[0]) if isinstance(bvec, (list, tuple)) and len(bvec) >= 1 else 0.0
    return rv + b0


def _template_on_obs(
    obs_wave: np.ndarray,
    tpl_wave: np.ndarray,
    tpl_flux_norm: np.ndarray,
    rv_kms: float,
    resolving_power: float,
) -> np.ndarray:
    beta = 1.0 + float(rv_kms) / float(config.C_KMS)
    w_map = np.asarray(tpl_wave, float) * beta
    order = np.argsort(w_map)
    w_s, tf_s = w_map[order], np.asarray(tpl_flux_norm, float)[order]
    tpl_on_obs = np.interp(obs_wave, w_s, tf_s, left=np.nan, right=np.nan)
    return rv_core.degrade_template_flux_lsf(obs_wave, tpl_on_obs, resolving_power)


def _fft_log_grid_shape_metrics(
    obs_wave: np.ndarray,
    obs_absorption: np.ndarray,
    tpl_wave: np.ndarray,
    tpl_flux_norm: np.ndarray,
    rv_kms: float,
    resolving_power: float,
) -> tuple[float, float]:
    """
    Metrics on the same log-λ grid as template FFT (shift+LSF template → absorption ``1 - F_norm``).

    ``corr_highpass_z_hann``: Pearson r between **Hanning-weighted z-scores** of a **high-pass**
    filtered absorption (obs vs template) on the log grid, **before** affine match. Plain z-scored
    absorption is dominated by correlated low-frequency continuum/line density and often stays r≈1;
    subtracting a long Gaussian smooth emphasizes line cores vs broadening mismatch.

    ``rms_absorption_residual``: RMS of ``obs − (a·tpl + b)`` on the log grid after the affine match
    used in FFT (lower = better line shape agreement).
    """
    obs_wave = np.asarray(obs_wave, float)
    obs_line = np.asarray(obs_absorption, float)
    tpl_wave = np.asarray(tpl_wave, float)
    tpl_flux_norm = np.asarray(tpl_flux_norm, float)

    beta = 1.0 + float(rv_kms) / float(config.C_KMS)
    w_map = tpl_wave * beta
    order = np.argsort(w_map)
    w_s, tf_s = w_map[order], tpl_flux_norm[order]
    tpl_on_obs = np.interp(obs_wave, w_s, tf_s, left=np.nan, right=np.nan)
    tpl_sm = rv_core.degrade_template_flux_lsf(obs_wave, tpl_on_obs, resolving_power)
    tpl_line_raw = 1.0 - tpl_sm

    loglam_obs = np.log10(obs_wave)
    npts = max(2 ** int(np.ceil(np.log2(len(obs_wave)))), 512)
    log_grid = np.linspace(loglam_obs.min(), loglam_obs.max(), npts)
    obs_r = np.interp(log_grid, loglam_obs, obs_line)
    tpl_r = np.interp(log_grid, loglam_obs, tpl_line_raw)
    valid = np.isfinite(obs_r) & np.isfinite(tpl_r)
    if int(np.sum(valid)) < 16:
        return float("nan"), float("nan")

    obs_v = obs_r[valid]
    tpl_v = tpl_r[valid]
    nv = len(obs_v)
    sig = float(max(2.5, min(35.0, nv * 0.035)))
    obs_hp = obs_v - gaussian_filter1d(obs_v, sig, mode="nearest")
    tpl_hp = tpl_v - gaussian_filter1d(tpl_v, sig, mode="nearest")
    so = float(np.std(obs_hp))
    st = float(np.std(tpl_hp))
    if so < 1e-12 or st < 1e-12:
        corr_hp = float("nan")
    else:
        obs_z = (obs_hp - float(np.mean(obs_hp))) / (so + 1e-9)
        tpl_z = (tpl_hp - float(np.mean(tpl_hp))) / (st + 1e-9)
        window = np.hanning(nv)
        oz = obs_z * window
        tz = tpl_z * window
        corr_hp = float(np.corrcoef(oz, tz)[0, 0])

    rms = rv_core.rms_absorption_residual_fft_grid(
        obs_wave, obs_line, tpl_wave, tpl_flux_norm, float(rv_kms), resolving_power
    )

    return corr_hp, rms


def main() -> None:
    ap = argparse.ArgumentParser(description="Fixed PHOENIX atmosphere, scan v sin i (chunk diagnostic plots)")
    ap.add_argument("--spectrum", type=Path, required=True)
    ap.add_argument("--instrument", type=str, default="APF")
    ap.add_argument("--chunk-key", type=str, default="22", help='Chunk id, e.g. "22" or "22_0"')
    ap.add_argument("--subchunks", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--teff", type=float, default=DEFAULT_K6_MS_TEFF, help="Target Teff for nearest grid model")
    ap.add_argument("--logg", type=float, default=DEFAULT_K6_MS_LOGG)
    ap.add_argument("--mh", type=float, default=DEFAULT_K6_MS_MH, help="Target [M/H] for nearest Z folder")
    ap.add_argument(
        "--obs-teff",
        type=float,
        default=None,
        help="Observation Teff for spline hot/cool continuum (default: same as --teff). "
        "Set to Gaia/known Teff when the target star differs from the PHOENIX grid star.",
    )
    ap.add_argument(
        "--vb-kms",
        type=str,
        default="0,2,4,6,8,10,15,20,25,30,40,50,65,80,100,120,150,180,200",
        help="Comma-separated rotational broadening trials (km/s)",
    )
    ap.add_argument("--rv-kms", type=float, default=None, help="Fixed heliocentric RV for template shift (km/s)")
    ap.add_argument(
        "--diagnostics-csv",
        type=Path,
        default=None,
        help="Pipeline diagnostics CSV: use mask_ccf RV for this chunk if --rv-kms omitted",
    )
    ap.add_argument("--no-bias", action="store_true", help="Do not add order b0 to mask RV (see diagnose_template_fft_star)")
    ap.add_argument(
        "--rv-search-half-width-kms",
        type=float,
        default=120.0,
        help="Lag half-width around seed when measuring FFT CCF peak (km/s)",
    )
    ap.add_argument("--dt-max", type=float, default=8000.0, help="Max |ΔTeff| when searching the PHOENIX grid")
    args = ap.parse_args()

    inst = instruments.get_instrument_profile(args.instrument)
    vb_list = _parse_vb_list(args.vb_kms)
    if not vb_list:
        raise SystemExit("Empty --vb-kms")

    rv_fixed = args.rv_kms
    if rv_fixed is None:
        if args.diagnostics_csv is None:
            raise SystemExit("Provide --rv-kms or --diagnostics-csv to set the fixed RV shift.")
        rv_fixed = _mask_rv_from_diagnostics(
            Path(args.diagnostics_csv), args.chunk_key, inst, no_bias=bool(args.no_bias)
        )
        if rv_fixed is None or not np.isfinite(rv_fixed):
            raise SystemExit(f"No mask_ccf RV for chunk_key={args.chunk_key!r} in {args.diagnostics_csv}")

    _, spec_data = io_utils.read_spectrum(str(args.spectrum))
    chunk_wave = chunk_flux = chunk_eflux = None
    for ck, w, f, e in chunking.iter_order_chunks(spec_data, inst.bad_orders, int(args.subchunks)):
        if str(ck) == str(args.chunk_key):
            chunk_wave, chunk_flux, chunk_eflux = w, f, e
            break
    if chunk_wave is None:
        raise SystemExit(f"Chunk {args.chunk_key!r} not found (check --subchunks and bad orders).")

    w_arr = np.asarray(chunk_wave, float)
    wl0, wl1 = float(np.min(w_arr)), float(np.max(w_arr))
    margin = 80.0
    raw = templates.load_hires_phoenix_raw_nearest(
        float(args.teff),
        float(args.logg),
        float(args.mh),
        (wl0 - margin, wl1 + margin),
        air=True,
        dt_max=float(args.dt_max),
    )
    if raw is None:
        raise SystemExit(
            f"No PHOENIX model (set DARKHUNTER_PHOENIX_DIR; nearest to Teff={args.teff} logg={args.logg} [M/H]={args.mh})."
        )
    wave_m, flux_m, (t_file, g_file, mh_file) = raw
    teff_norm = float(t_file)
    obs_teff = float(args.obs_teff) if args.obs_teff is not None else float(args.teff)

    ckw = _continuum_kw(obs_teff)
    nw, nf, ne = continuum.fit_continuum(chunk_wave, chunk_flux, chunk_eflux, **ckw)
    nw, nf, ne = continuum.despike_normalized_pre_ccf(nw, nf, ne)
    obs_abs = 1.0 - np.asarray(nf, float)
    obs_abs = rv_core.mask_line_flux_in_excluded_wavelengths(np.asarray(nw, float), obs_abs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w.\-]+", "_", Path(args.spectrum).stem)

    records: list[dict] = []
    R = float(inst.resolving_power)

    for vb in vb_list:
        vb = float(vb)
        f_broad = physics.broaden_spectrum(wave_m, flux_m, vb)
        f_norm = templates._norm_template_flux_fft(wave_m, f_broad, teff_norm)

        obs_resamp, window, fft_obs, _vel_axis, mask_vel, vel_win, tpl_grid_wave = rv_core._fft_velocity_window(
            nw,
            obs_abs,
            rv_seed_kms=float(rv_fixed),
            rv_search_half_width_kms=float(args.rv_search_half_width_kms),
        )
        r_pk = rv_core._fft_correlation_peak_for_template(
            obs_resamp,
            window,
            fft_obs,
            mask_vel,
            vel_win,
            tpl_grid_wave,
            np.asarray(wave_m, float),
            np.asarray(f_norm, float),
        )
        if r_pk is None:
            ccf_peak = float("nan")
            rv_at_peak = float("nan")
        else:
            ccf_peak, rv_at_peak, _ccf_w = r_pk

        corr_hp, rms_abs = _fft_log_grid_shape_metrics(
            nw, obs_abs, wave_m, f_norm, float(rv_fixed), R
        )
        tpl_sm = _template_on_obs(np.asarray(nw, float), wave_m, f_norm, float(rv_fixed), R)

        rec = {
            "vb_kms": vb,
            "fft_ccf_peak": ccf_peak,
            "rv_fft_peak_kms": rv_at_peak,
            "corr_highpass_z_hann": corr_hp,
            "rms_absorption_residual_fft_grid": rms_abs,
            "phoenix_teff": t_file,
            "phoenix_logg": g_file,
            "phoenix_mh": mh_file,
            "rv_fixed_kms": float(rv_fixed),
        }
        records.append(rec)

        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(nw, nf, "k-", lw=0.55, label="obs norm")
        ax.plot(nw, tpl_sm, "r-", lw=0.65, alpha=0.85, label="template (shift+LSF)")
        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Norm flux")
        ttl = (
            f"chunk {args.chunk_key} | PHOENIX ({t_file:.0f}, {g_file:.2f}, {mh_file:+.2f}) | "
            f"v_sin i = {vb:.1f} km/s | CCF peak = {ccf_peak:.1f} | "
            f"corr_hp = {corr_hp:.3f} | RMS_abs = {rms_abs:.4f}"
        )
        ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        p_png = args.out_dir / f"{stem}_chunk{args.chunk_key}_vsini_{vb:g}.png"
        fig.savefig(p_png, dpi=140)
        plt.close(fig)

    with open(args.out_dir / f"{stem}_chunk{args.chunk_key}_vsini_metrics.json", "w") as fp:
        json.dump(
            {
                "spectrum": str(args.spectrum),
                "chunk_key": str(args.chunk_key),
                "rv_fixed_kms": float(rv_fixed),
                "requested_teff_logg_mh": [args.teff, args.logg, args.mh],
                "obs_teff_continuum": obs_teff,
                "phoenix_file_atmosphere": [t_file, g_file, mh_file],
                "metrics_note": (
                    "corr_highpass_z_hann: Hanning-weighted z-scores of Gaussian high-pass absorption on the "
                    "FFT log grid (before affine). rms_absorption_residual_fft_grid: RMS(obs - affine_tpl); "
                    "lower is better. fft_ccf_peak: pipeline-style FFT CCF peak (higher better)."
                ),
                "trials": records,
            },
            fp,
            indent=2,
        )

    vb_a = np.array([r["vb_kms"] for r in records], float)
    peak_a = np.array([r["fft_ccf_peak"] for r in records], float)
    shape_a = np.array([r["corr_highpass_z_hann"] for r in records], float)
    rms_a = np.array([r["rms_absorption_residual_fft_grid"] for r in records], float)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8.5, 7.0), sharex=True)
    ax0.plot(vb_a, peak_a, "o-", ms=4, lw=0.9, color="C0")
    ax0.set_ylabel("FFT CCF peak\n(narrow lag window)")
    ax0.grid(True, alpha=0.3)
    ax0.set_title(
        f"{stem} chunk {args.chunk_key} | fixed RV = {rv_fixed:.2f} km/s | "
        f"PHOENIX ({t_file:.0f}/{g_file:.2f}/{mh_file:+.2f})",
        fontsize=10,
    )

    ax1.plot(vb_a, shape_a, "s-", ms=4, lw=0.9, color="C1")
    ax1.set_ylabel("Corr (high-pass abs.)\nz-scored, Hanning")
    ax1.grid(True, alpha=0.3)

    ax2.plot(vb_a, rms_a, "^-", ms=4, lw=0.9, color="C2")
    ax2.set_xlabel("Rotational broadening v sin i (km/s)")
    ax2.set_ylabel("RMS absorption residual\n(after affine, log grid)")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"{stem}_chunk{args.chunk_key}_vsini_metrics.png", dpi=140)
    plt.close(fig)

    print(f"Wrote {len(records)} per-vsini plots and summary to {args.out_dir}")


if __name__ == "__main__":
    main()
