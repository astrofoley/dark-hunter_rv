#!/usr/bin/env python3
"""
Fit a shared parameterized blaze (sinc²) on one echelle order.

Modes:
  * ``hbeta`` — Hβ order (28); weak-line cohort; mask Balmer pixels for stacking/fit.
  * ``clean`` — order with no STRONG_LINES; use full order for fit and comparison.

Example (clean order, no line gaps)::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python3 -m validation.fit_hbeta_order_blaze \\
    --fit-mode clean --echelle-order 35 \\
    --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \\
    --overlap-csv validation_output/template_fft_baseline/overlap/overlap_enriched_per_exposure.csv \\
    --out-dir validation_output/template_fft_baseline/order35_blaze_fit
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import blaze, config, continuum, instruments, io_utils

logger = logging.getLogger(__name__)


def _stem_from_path(p: str) -> str:
    return Path(p).stem.replace("_diagnostics", "")


def _load_overlap_index(overlap_csv: Path | None) -> dict[str, dict]:
    if overlap_csv is None or not overlap_csv.is_file():
        return {}
    df = pd.read_csv(overlap_csv)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        stem = _stem_from_path(str(row.get("basename", row.get("diagnostics_csv", ""))))
        if stem:
            out[stem] = row.to_dict()
    return out


def _select_weak_narrow_profiles(
    spec_paths: list[Path],
    overlap: dict[str, dict],
    inst,
    *,
    teff_min: float,
    teff_max: float,
    min_snr: float,
    max_hbeta_depth: float | None,
    depth_percentile: float | None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, list[str], list[float]]:
    """Return profiles, Hβ order, stems, and per-spectrum Hβ depth proxy."""
    candidates: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    hb_order: int | None = None

    for spec_path in spec_paths:
        stem = spec_path.stem
        meta = overlap.get(stem, {})
        teff = float(meta.get("teff", np.nan))
        if np.isfinite(teff) and (teff < teff_min or teff > teff_max):
            continue
        snr = float(meta.get("median_mask_ccf_peak_snr", np.nan))
        if np.isfinite(snr) and snr < min_snr:
            continue

        try:
            _hdr, spec_data = io_utils.read_spectrum(str(spec_path))
        except Exception as ex:
            logger.debug("skip %s: %s", stem, ex)
            continue

        o = blaze.find_order_covering_rest(spec_data, blaze.HB_REST_A, bad_orders=inst.bad_orders)
        if o is None:
            continue
        if hb_order is None:
            hb_order = int(o)
        elif int(o) != hb_order:
            logger.debug("skip %s: Hβ order %s != %s", stem, o, hb_order)
            continue

        w = np.asarray(spec_data[o]["wavelength"], float)
        f = np.asarray(spec_data[o]["flux"], float)
        depth = blaze.hbeta_absorption_depth_raw(w, f)
        if not np.isfinite(depth) and np.isfinite(teff) and teff > config.HOT_STAR_TEFF_THRESHOLD:
            continue
        candidates.append((stem, w, f, float(depth)))

    if hb_order is None or not candidates:
        return [], -1, [], []

    depths = np.asarray([c[3] for c in candidates], float)
    depth_thr = float("inf")
    if depth_percentile is not None and np.isfinite(depth_percentile):
        depth_thr = float(np.nanpercentile(depths, float(depth_percentile)))
    if max_hbeta_depth is not None and np.isfinite(max_hbeta_depth):
        depth_thr = min(depth_thr, float(max_hbeta_depth))

    profiles: list[tuple[np.ndarray, np.ndarray]] = []
    stems_used: list[str] = []
    depths_used: list[float] = []
    for stem, w, f, depth in candidates:
        if np.isfinite(depth) and depth > depth_thr:
            continue
        profiles.append((w, f))
        stems_used.append(stem)
        depths_used.append(depth)

    return profiles, hb_order, stems_used, depths_used


def _select_order_profiles(
    spec_paths: list[Path],
    overlap: dict[str, dict],
    inst,
    echelle_order: int,
    *,
    min_snr: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    """All spectra on a fixed order (clean-order blaze validation)."""
    profiles: list[tuple[np.ndarray, np.ndarray]] = []
    stems_used: list[str] = []

    for spec_path in spec_paths:
        stem = spec_path.stem
        meta = overlap.get(stem, {})
        snr = float(meta.get("median_mask_ccf_peak_snr", np.nan))
        if overlap and np.isfinite(snr) and snr < min_snr:
            continue

        try:
            _hdr, spec_data = io_utils.read_spectrum(str(spec_path))
        except Exception as ex:
            logger.debug("skip %s: %s", stem, ex)
            continue

        if int(echelle_order) not in spec_data:
            continue
        if int(echelle_order) in set(inst.bad_orders or []):
            continue

        w = np.asarray(spec_data[echelle_order]["wavelength"], float)
        f = np.asarray(spec_data[echelle_order]["flux"], float)
        if w.size < 20 or not np.any(np.isfinite(f) & (f > 0)):
            continue

        profiles.append((w, f))
        stems_used.append(stem)

    return profiles, stems_used


def _plot_fit(
    model: blaze.OrderBlazeModel,
    grid: np.ndarray,
    median_f: np.ndarray,
    out_path: Path,
    *,
    profiles: list[tuple[np.ndarray, np.ndarray]],
    n_show: int = 12,
    use_line_mask: bool = True,
    ylim_bottom: tuple[float, float] = (0.0, 1.3),
    title_label: str = "",
) -> None:
    mask = (
        blaze.blaze_line_mask(
            grid,
            half_width_angstrom=model.line_mask_half_width_angstrom,
        )
        if use_line_mask
        else np.isfinite(grid)
    )
    b = model.blaze_on_grid(grid)
    scale = float(np.nanmedian(median_f[mask] / np.maximum(b[mask], 1e-12)))
    b_scaled = b * scale

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    ax0, ax1 = axes

    ax0.plot(grid, median_f, "k-", lw=1.2, label="median raw flux (fit sample)")
    ax0.plot(grid, b_scaled, "r--", lw=1.1, label=f"sinc² fit (p={model.power:.2f})")
    ax0.set_ylabel("Raw flux")
    label = title_label or f"order {model.echelle_order}"
    ax0.set_title(
        f"{label} blaze fit  "
        f"λ₀={model.center_angstrom:.1f} Å  w={model.width_angstrom:.1f} Å  N={model.n_spectra_fit}"
    )
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.25)

    show = profiles[:n_show]
    median_norm = np.full(grid.shape, np.nan, dtype=float)
    for w, f in show:
        fc = model.correct_flux(w, f)
        m = np.isfinite(fc) & (fc > 0)
        level = float(np.nanmedian(fc[m])) if int(np.sum(m)) else float("nan")
        if not np.isfinite(level) or level <= 0:
            continue
        fc_norm = fc / level
        ax1.plot(w, fc_norm, lw=0.7, alpha=0.55)
        median_norm = np.nanmedian(
            np.vstack([median_norm, np.interp(grid, w, fc_norm, left=np.nan, right=np.nan)]),
            axis=0,
        )

    if np.any(np.isfinite(median_norm)):
        ax1.plot(grid, median_norm, "k-", lw=1.4, alpha=0.9, label="median corrected")
    ax1.axhline(1.0, color="0.4", ls=":", lw=0.8)
    ax1.set_xlabel("Wavelength (Å)")
    ax1.set_ylabel("Flux / blaze / median")
    ax1.set_ylim(*ylim_bottom)
    ax1.set_title(f"Blaze-corrected, median-normalized ({len(show)} sample spectra)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit shared echelle-order blaze from campaign spectra")
    ap.add_argument(
        "--fit-mode",
        choices=("hbeta", "clean"),
        default="hbeta",
        help="hbeta: Hβ order + weak-line cohort; clean: fixed order without strong lines",
    )
    ap.add_argument("--echelle-order", type=int, default=None, help="Required for clean mode (e.g. 35)")
    ap.add_argument(
        "--pick-clean-near-angstrom",
        type=float,
        default=None,
        help="If set with clean mode and no --echelle-order, auto-pick clean order near this λ",
    )
    ap.add_argument("--spectrum-list", type=Path, required=True)
    ap.add_argument("--overlap-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--calibration-out",
        type=Path,
        default=None,
        help="Defaults: calibration/blaze_hbeta_order.json or blaze_order{N}.json",
    )
    ap.add_argument("--instrument", default="APF")
    ap.add_argument("--teff-min", type=float, default=5200.0)
    ap.add_argument("--teff-max", type=float, default=6500.0)
    ap.add_argument("--min-snr", type=float, default=3.5)
    ap.add_argument(
        "--depth-percentile",
        type=float,
        default=60.0,
        help="Keep spectra with Hβ depth at or below this percentile among Teff/SNR passers",
    )
    ap.add_argument(
        "--max-hbeta-depth",
        type=float,
        default=None,
        help="Optional absolute cap on wing-normalized Hβ depth (in addition to percentile)",
    )
    ap.add_argument("--line-mask-half-width", type=float, default=22.0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    spec_paths = [
        Path(ln.strip())
        for ln in args.spectrum_list.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    overlap = _load_overlap_index(args.overlap_csv)
    inst = instruments.get_instrument_profile(args.instrument)

    fit_mode = str(args.fit_mode)
    use_line_mask = fit_mode == "hbeta"
    rest_lines: list[float] | None = list(continuum.STRONG_LINES) if use_line_mask else []
    depths_used: list[float] = []
    stems: list[str] = []

    if fit_mode == "hbeta":
        profiles, hb_order, stems, depths_used = _select_weak_narrow_profiles(
            spec_paths,
            overlap,
            inst,
            teff_min=float(args.teff_min),
            teff_max=float(args.teff_max),
            min_snr=float(args.min_snr),
            max_hbeta_depth=args.max_hbeta_depth,
            depth_percentile=float(args.depth_percentile) if args.depth_percentile is not None else None,
        )
        logger.info(
            "Selected %d weak/narrow spectra on Hβ order %d (Teff %.0f–%.0f, depth ≤ p%.0f)",
            len(profiles),
            hb_order,
            args.teff_min,
            args.teff_max,
            args.depth_percentile or 0,
        )
        title_label = f"Hβ order {hb_order}"
        if args.calibration_out is None:
            args.calibration_out = _REPO_ROOT / "calibration" / "blaze_hbeta_order.json"
    else:
        echelle_order = args.echelle_order
        if echelle_order is None:
            if args.pick_clean_near_angstrom is None:
                args.pick_clean_near_angstrom = 5250.0
            probe = next(
                (p for p in spec_paths if p.is_file()),
                None,
            )
            if probe is None:
                logging.error("No readable spectra in --spectrum-list")
                return 1
            _hdr, spec_data = io_utils.read_spectrum(str(probe))
            picked = blaze.pick_clean_order_near_wavelength(
                spec_data,
                float(args.pick_clean_near_angstrom),
                bad_orders=inst.bad_orders,
            )
            if picked is None:
                logging.error("No clean order found near %.1f Å", args.pick_clean_near_angstrom)
                return 1
            echelle_order, wmn, wmx = picked
            logger.info(
                "Auto-picked clean order %d (%.1f–%.1f Å, no strong lines)",
                echelle_order,
                wmn,
                wmx,
            )
        else:
            probe = next((p for p in spec_paths if p.is_file()), None)
            if probe is not None:
                _hdr, spec_data = io_utils.read_spectrum(str(probe))
                w = np.asarray(spec_data[echelle_order]["wavelength"], float)
                wmn, wmx = float(np.min(w)), float(np.max(w))
                if blaze.order_covers_strong_line(wmn, wmx):
                    logging.warning(
                        "Order %d (%.1f–%.1f Å) contains a STRONG_LINES rest wavelength",
                        echelle_order,
                        wmn,
                        wmx,
                    )

        profiles, stems = _select_order_profiles(
            spec_paths,
            overlap,
            inst,
            int(echelle_order),
            min_snr=float(args.min_snr),
        )
        hb_order = int(echelle_order)
        logger.info("Selected %d spectra on clean order %d (S/N ≥ %.1f)", len(profiles), hb_order, args.min_snr)
        title_label = f"Clean order {hb_order}"
        if args.calibration_out is None:
            args.calibration_out = _REPO_ROOT / "calibration" / f"blaze_order{hb_order}.json"
    if len(profiles) < 5:
        logging.error("Too few spectra for blaze fit (%d). Relax cuts or add overlap metadata.", len(profiles))
        return 1

    model = blaze.fit_order_blaze_from_profiles(
        profiles,
        hb_order,
        line_mask_half_width=float(args.line_mask_half_width),
        rest_lines=rest_lines,
    )
    if model is None:
        logging.error("Blaze fit failed")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.calibration_out)

    grid = np.linspace(model.wavelength_min, model.wavelength_max, 500)
    if use_line_mask:
        mask = blaze.blaze_line_mask(grid, half_width_angstrom=model.line_mask_half_width_angstrom)
    else:
        mask = np.isfinite(grid)
    median_f, _rms = blaze.median_profile_and_rms(profiles, grid, line_mask=mask if use_line_mask else None)

    stem_rows = {"stem": stems}
    if depths_used:
        stem_rows["hbeta_depth_proxy"] = depths_used
    pd.DataFrame(stem_rows).to_csv(args.out_dir / "blaze_fit_stems.csv", index=False)

    plot_name = "hbeta_order_blaze_fit.png" if fit_mode == "hbeta" else f"order{hb_order}_blaze_fit.png"
    _plot_fit(
        model,
        grid,
        median_f,
        args.out_dir / plot_name,
        profiles=profiles,
        use_line_mask=use_line_mask,
        title_label=title_label,
    )

    # Residual panel
    b = model.blaze_on_grid(grid)
    scale = float(np.nanmedian(median_f[mask] / np.maximum(b[mask], 1e-12)))
    resid = (median_f - scale * b) / np.maximum(scale * b, 1e-12)
    fig, ax = plt.subplots(figsize=(10.0, 3.2))
    ax.plot(grid[mask], resid[mask], "k-", lw=0.9)
    ax.axhline(0.0, color="0.5", ls="--")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("(median − blaze) / blaze")
    resid_title = "masked pixels" if use_line_mask else "full order"
    ax.set_title(f"Fractional residual after shared blaze ({resid_title})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    resid_name = "hbeta_order_blaze_residual.png" if fit_mode == "hbeta" else f"order{hb_order}_blaze_residual.png"
    fig.savefig(args.out_dir / resid_name, dpi=130)
    plt.close(fig)

    summary = [
        f"# {title_label} blaze fit",
        f"- fit_mode: {fit_mode}",
        f"- echelle_order: {model.echelle_order}",
        f"- wavelength_span_angstrom: {model.wavelength_min:.2f} – {model.wavelength_max:.2f}",
        f"- center_angstrom: {model.center_angstrom:.4f}",
        f"- width_angstrom: {model.width_angstrom:.4f}",
        f"- power: {model.power:.4f}",
        f"- n_spectra: {model.n_spectra_fit}",
        f"- line_mask_during_fit: {use_line_mask}",
        f"- calibration: {args.calibration_out}",
    ]
    (args.out_dir / "BLAZE_FIT_SUMMARY.md").write_text("\n".join(summary) + "\n")

    logger.info("Wrote %s", args.calibration_out.resolve())
    logger.info("Wrote plots -> %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
