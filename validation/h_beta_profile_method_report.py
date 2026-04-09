#!/usr/bin/env python3
"""
Per-epoch Hβ profile fits (Gaussian / Lorentzian / Voigt only) and comparison figures.

Reads echelle spectra, applies the same hot-star continuum settings as the main pipeline, runs
:class:`darkhunter_rv.rv_core.fit_balmer_line_all_methods` on the echelle order that covers each
Balmer line, and writes:

  * stacked Hβ panels + RV-vs-MJD curves for the three profile models
  * template FFT RV vs Hβ Voigt RV (template RVs from an optional method summary CSV)
  * Ha / Hg / Hd Voigt vs Hβ Voigt scatter panels

Example (Gaia DR3 412195879777348480; run from the repository root — spectra path matches the
files referenced in pipeline diagnostics, typically under ``.../darkhunter/rvs/data/``):

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.h_beta_profile_method_report \\
    --spectra-glob '/Users/rfoley/darkhunter/rvs/data/Gaia_DR3_412195879777348480_epoch_*.txt' \\
    --method-summary-csv validation_output/diagnose_hot412/method_exposure_summary.csv \\
    --out-dir validation_output/h_beta_method_plots_hot412 \\
    --title-stem Gaia_412195879777348480
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from glob import glob as glob_paths
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config, continuum, instruments, io_utils, plotting, rv_core

logger = logging.getLogger(__name__)

BALMER = {"Ha": 6562.8, "Hb": rv_core.HB_REST_A, "Hg": 4340.5, "Hd": 4101.7}


def _sort_spectra(paths: list[Path]) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        m = re.search(r"epoch_(\d+)", p.name, re.I)
        return (int(m.group(1)) if m else 0, p.name)

    return sorted(paths, key=key)


def _order_covering_rest(spec_data: dict, valid_orders: list[int], rest: float) -> int | None:
    for o in valid_orders:
        w = np.array(spec_data[o]["wavelength"], float)
        lo, hi = float(np.min(w)), float(np.max(w))
        if lo <= rest <= hi:
            return int(o)
    return None


def _norm_order(
    spec_data: dict,
    order: int,
    *,
    continuum_mode: str,
    hot_continuum: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    w = np.array(spec_data[order]["wavelength"], float)
    f = np.array(spec_data[order]["flux"], float)
    e = np.array(spec_data[order]["eflux"], float)
    kw: dict = {"continuum_mode": continuum_mode}
    if hot_continuum:
        kw["exclude_near_lines_width"] = 78.0
    try:
        nw, nf, _ = continuum.fit_continuum(w, f, e, **kw)
        nw, nf, _ = continuum.despike_normalized_pre_ccf(nw, nf, np.ones_like(nf))
    except Exception as ex:
        logger.warning("continuum failed order=%s: %s", order, ex)
        return None
    return nw, nf


def _fit_balmer_line(
    spec_data: dict,
    valid_orders: list[int],
    line_name: str,
    rest: float,
    *,
    continuum_mode: str,
    broad_lines: bool,
) -> dict | None:
    o = _order_covering_rest(spec_data, valid_orders, rest)
    if o is None:
        return None
    pair = _norm_order(spec_data, o, continuum_mode=continuum_mode, hot_continuum=broad_lines)
    if pair is None:
        return None
    nw, nf = pair
    return rv_core.fit_balmer_line_all_methods(nw, nf, rest, line_name, broad_lines=broad_lines)


def _load_template_rv_by_basename(path: Path) -> dict[str, tuple[float, float]]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    if "method" not in df.columns or "basename" not in df.columns:
        logger.warning("method summary missing columns: %s", path)
        return {}
    sub = df[df["method"].astype(str) == "template_fft"]
    out: dict[str, tuple[float, float]] = {}
    for _, row in sub.iterrows():
        b = str(row["basename"])
        try:
            rv = float(row["rv_kms"])
            er = float(row["rv_err_kms"])
        except (TypeError, ValueError):
            continue
        out[b] = (rv, er)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hβ three-model profile report and Balmer comparisons")
    p.add_argument(
        "--spectra-glob",
        required=True,
        help=(
            "Glob for spectrum files (quoted). Example: "
            "'/Users/rfoley/darkhunter/rvs/data/Gaia_DR3_412195879777348480_epoch_*.txt'"
        ),
    )
    p.add_argument("--instrument", default="APF", help="Instrument profile name (default APF)")
    p.add_argument("--teff", type=float, default=14000.0, help="Teff for broad-line mode (default 14000)")
    p.add_argument("--continuum-mode", choices=["spline", "blaze"], default="spline")
    p.add_argument(
        "--method-summary-csv",
        default="",
        help="Optional CSV (basename,method,rv_kms,rv_err_kms,...) with template_fft rows",
    )
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory for PNGs")
    p.add_argument("--title-stem", default="", help="Prefix for figure titles")
    p.add_argument("--log-level", default="INFO", help="logging level")
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    paths = [Path(x) for x in glob_paths(args.spectra_glob)]
    paths = _sort_spectra(paths)
    if not paths:
        logger.error("No files matched --spectra-glob %r", args.spectra_glob)
        return 2

    inst = instruments.get_instrument_profile(args.instrument)
    valid_template: dict[str, tuple[float, float]] = {}
    if args.method_summary_csv:
        csv_p = Path(args.method_summary_csv)
        if not csv_p.is_absolute():
            csv_p = _REPO_ROOT / csv_p
        valid_template = _load_template_rv_by_basename(csv_p)

    broad = float(args.teff) > float(config.HOT_STAR_TEFF_THRESHOLD)
    if not broad:
        logger.warning("teff=%.0f ≤ hot threshold %.0f; using narrow-line Balmer caps", args.teff, config.HOT_STAR_TEFF_THRESHOLD)

    epochs: list[tuple[float, str, dict]] = []
    hb_voigt: list[float] = []
    hb_voigt_err: list[float] = []
    tpl_rv: list[float] = []
    tpl_err: list[float] = []

    line_voigt: dict[str, list[float]] = {k: [] for k in ("Ha", "Hg", "Hd")}
    line_voigt_err: dict[str, list[float]] = {k: [] for k in ("Ha", "Hg", "Hd")}

    for spec_path in paths:
        try:
            header, spec_data = io_utils.read_spectrum(str(spec_path))
        except Exception as ex:
            logger.warning("skip %s: %s", spec_path, ex)
            continue
        mjd = float(io_utils.extract_mjd_from_header(header, inst))

        valid_orders = sorted(o for o in spec_data if o not in inst.bad_orders)
        hb = _fit_balmer_line(
            spec_data,
            valid_orders,
            "Hb",
            BALMER["Hb"],
            continuum_mode=args.continuum_mode,
            broad_lines=broad,
        )
        tr = valid_template.get(spec_path.name, (float("nan"), float("nan")))
        tpl_rv.append(tr[0])
        tpl_err.append(tr[1])

        for name in ("Ha", "Hg", "Hd"):
            rest = BALMER[name]
            pan = _fit_balmer_line(
                spec_data,
                valid_orders,
                name,
                rest,
                continuum_mode=args.continuum_mode,
                broad_lines=broad,
            )
            if pan is None:
                line_voigt[name].append(float("nan"))
                line_voigt_err[name].append(float("nan"))
            else:
                line_voigt[name].append(float(pan["rv_voigt_kms"]))
                line_voigt_err[name].append(float(pan["err_voigt_kms"]))

        if hb is None:
            logger.warning("no Hβ fit for %s", spec_path.name)
            hb_voigt.append(float("nan"))
            hb_voigt_err.append(float("nan"))
            continue

        label = spec_path.stem
        epochs.append((mjd, label, hb))
        hb_voigt.append(float(hb["rv_voigt_kms"]))
        hb_voigt_err.append(float(hb["err_voigt_kms"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.title_stem or (paths[0].stem.split("_epoch_")[0] if paths else "h_beta_report")
    file_slug = re.sub(r"\s+", "_", str(stem).strip()) or "h_beta_report"

    if epochs:
        plotting.plot_h_beta_three_model_epoch_stack(
            epochs,
            args.out_dir / f"{file_slug}_h_beta_three_model_epochs.png",
            title_stem=args.title_stem or stem,
        )
    else:
        logger.error("No successful Hβ fits; no epoch stack plot")

    hb_v = np.array(hb_voigt, float)
    hb_e = np.array(hb_voigt_err, float)
    t_rv = np.array(tpl_rv, float)
    t_er = np.array(tpl_err, float)
    ok_tpl = np.isfinite(hb_v) & np.isfinite(t_rv) & np.isfinite(t_er) & np.isfinite(hb_e)
    if int(np.sum(ok_tpl)) >= 2:
        plotting.plot_rv_scatter_compare(
            hb_v[ok_tpl],
            t_rv[ok_tpl],
            args.out_dir / f"{file_slug}_template_fft_vs_hb_voigt.png",
            x_err=hb_e[ok_tpl],
            y_err=t_er[ok_tpl],
            xlabel="Hβ Voigt RV (km/s)",
            ylabel="Template FFT exposure RV (km/s)",
            title=args.title_stem or stem,
        )
    else:
        logger.warning("Not enough paired template/Hβ points for scatter (need method summary CSV)")

    lv = {k: np.array(v, float) for k, v in line_voigt.items()}
    le = {k: np.array(v, float) for k, v in line_voigt_err.items()}
    if np.any(np.isfinite(hb_v)):
        plotting.plot_balmer_voigt_vs_h_beta_voigt(
            hb_v,
            lv,
            args.out_dir / f"{file_slug}_balmer_voigt_vs_hb_voigt.png",
            hb_err=hb_e,
            line_err=le,
            title_stem=args.title_stem or stem,
        )

    logger.info("Wrote figures under %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
