#!/usr/bin/env python3
"""Build Gaia_DR3_<id>_28_hbeta.png: all epochs on one axes, color by MJD."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from darkhunter_rv import continuum, io_utils, rv_core
from fit_apf_rv_keplerian import discover_summary_files, parse_object_id_from_summary


def _epoch_rows(summary_path: Path) -> list[tuple[float, str]]:
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    if "[PIPELINE RESULTS]" not in text:
        return []
    lines = text.split("[PIPELINE RESULTS]", 1)[-1].splitlines()
    rows: list[tuple[float, str]] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[1]), parts[0]))
        except ValueError:
            continue
    rows.sort(key=lambda x: x[0])
    return rows


def _parse_teff_from_summary(summary_path: Path) -> float:
    try:
        from darkhunter_rv.gaia_utils import parse_gaia_metadata_from_star_summary

        meta = parse_gaia_metadata_from_star_summary(summary_path) or {}
        for key in ("Teff", "teff", "TEFF"):
            if key in meta:
                return float(meta[key])
    except Exception:
        pass
    return 5500.0


def _find_spectrum(basename: str, roots: list[Path]) -> Path | None:
    name = Path(basename).name
    for root in roots:
        if not root.is_dir():
            continue
        direct = root / name
        if direct.is_file():
            return direct
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def _hbeta_profile_from_spectrum(spec_path: Path, teff: float) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        if "ghost" in spec_path.name.lower():
            _, spec_data = io_utils.read_spectrum_ghost(str(spec_path))
        elif "maroon" in spec_path.name.lower():
            _, spec_data = io_utils.read_spectrum_maroonx(str(spec_path))
        else:
            _, spec_data = io_utils.read_spectrum(str(spec_path))
    except Exception:
        return None

    broad = teff > 5200.0
    for order in sorted(spec_data.keys()):
        w_raw = np.asarray(spec_data[order]["wavelength"], dtype=float)
        f_raw = np.asarray(spec_data[order]["flux"], dtype=float)
        e_raw = np.asarray(spec_data[order]["eflux"], dtype=float)
        if w_raw.size < 30:
            continue
        if not (np.nanmin(w_raw) <= rv_core.HB_REST_A <= np.nanmax(w_raw)):
            continue
        try:
            nw, nf, ne = continuum.fit_continuum(
                w_raw,
                f_raw,
                e_raw,
                continuum_mode="spline",
            )
            nw, nf, _ = continuum.despike_normalized_pre_ccf(nw, nf, np.ones_like(nf))
        except Exception:
            continue
        hb = rv_core.measure_h_beta_rv(nw, nf, broad_lines=broad)
        if hb is None:
            continue
        v = np.asarray(hb.get("v_kms_plot", hb.get("v_kms")), dtype=float)
        f = np.asarray(hb.get("flux_plot", hb.get("flux")), dtype=float)
        if v.size < 10 or f.size != v.size:
            continue
        order = np.argsort(v)
        return v[order], f[order]
    return None


def build_overlay(
    epochs: list[tuple[float, str]],
    summary_path: Path,
    spec_roots: list[Path],
    out_png: Path,
) -> bool:
    teff = _parse_teff_from_summary(summary_path)
    curves: list[tuple[float, np.ndarray, np.ndarray]] = []
    for mjd, bn in epochs:
        spec = _find_spectrum(bn, spec_roots)
        if spec is None:
            continue
        prof = _hbeta_profile_from_spectrum(spec, teff)
        if prof is None:
            continue
        v, f = prof
        curves.append((mjd, v, f))
    if not curves:
        return False

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    mjds = np.array([c[0] for c in curves], dtype=float)
    mmin, mmax = float(np.min(mjds)), float(np.max(mjds))
    cmap = plt.get_cmap("viridis")
    for mjd, v, f in curves:
        frac = 0.0 if mmax <= mmin else (float(mjd) - mmin) / (mmax - mmin)
        color = cmap(frac)
        ax.plot(v, f, "-", lw=1.1, color=color, alpha=0.88, label=f"MJD {mjd:.1f}")

    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Normalized flux")
    sid = parse_object_id_from_summary(summary_path) or summary_path.stem
    ax.set_title(f"Hβ profiles — Gaia DR3 {sid}")
    ax.grid(alpha=0.25)
    if len(curves) <= 12:
        ax.legend(loc="best", fontsize=7, ncol=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=mmin, vmax=mmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("MJD")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Build multi-epoch Hβ overlay PNGs for the website.")
    ap.add_argument("--summary-dir", required=True)
    ap.add_argument("--plots-root", required=True, help="Per-star plot directory root (output/Gaia_DR3_<id>)")
    ap.add_argument(
        "--spec-root",
        action="append",
        default=[],
        help="Spectrum search root (repeatable). Default: SPEC_ROOT env or /data2/gaia_stars/apf_reductions",
    )
    ap.add_argument("--star-id", default=None)
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    plots_root = Path(args.plots_root)
    spec_roots = [Path(p) for p in args.spec_root]
    if not spec_roots:
        env_root = os.environ.get("SPEC_ROOT", "/data2/gaia_stars/apf_reductions")
        spec_roots = [Path(env_root), summary_dir, summary_dir.parent]

    if args.star_id:
        summaries = [summary_dir / f"Gaia_DR3_{args.star_id}_summary.txt"]
    else:
        summaries = discover_summary_files(summary_dir)

    built = 0
    skipped = 0
    for summ in summaries:
        if not summ.is_file():
            skipped += 1
            continue
        sid = parse_object_id_from_summary(summ)
        if not sid:
            skipped += 1
            continue
        plot_dir = plots_root / f"Gaia_DR3_{sid}"
        out_png = plot_dir / f"Gaia_DR3_{sid}_28_hbeta.png"
        epochs = _epoch_rows(summ)
        if build_overlay(epochs, summ, spec_roots, out_png):
            built += 1
        else:
            skipped += 1
    print(f"Built {built} Hβ overlay plots (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
