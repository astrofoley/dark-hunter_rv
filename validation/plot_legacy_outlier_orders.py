#!/usr/bin/env python3
"""
Per-exposure PDFs: every echelle order with continuum-normalized flux and the **mask schematic**
(``plotting._hb_mask_pseudo_on_wavelengths``) drawn at **two RVs** — same layout as the Hβ
order-norm branch (tight y-limits on data + schematics).

**Legacy mode** (default): RVs are exposure-level mask CCF (recomputed from diagnostics) vs legacy
RV from the star summary row, when |mask − legacy| exceeds a threshold. Input:
``method_comparison_per_exposure.csv`` from ``rv_method_diagnostics_report``.

**Method-pair mode**: RVs are any two pipeline methods that are both valid in
``overlap_enriched_per_exposure.csv`` (from ``rv_method_overlap_report``) and differ by more than
the threshold. The schematic is still the **stellar mask** shifted to each method's reported
velocity (including template / Hβ / strong-line RVs), so you see the same comparison style as
legacy-vs-current on every order.

Examples::

  # Legacy vs current mask (as before)
  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.plot_legacy_outlier_orders \\
    --comparison-csv validation_output/rv_method_teff_report_epoch1/method_comparison_per_exposure.csv \\
    --threshold-kms 5 \\
    --out-dir validation_output/legacy_outliers_epoch1 \\
    --max-legacy-err 0.5

  # Two “good” methods disagreeing by >50 km/s (one PDF per exposure per pair)
  python -m validation.plot_legacy_outlier_orders \\
    --overlap-csv validation_output/rv_method_overlap_report_epoch1/overlap_enriched_per_exposure.csv \\
    --threshold-kms 50 \\
    --out-dir validation_output/method_discrep_orders_epoch1
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config, continuum, instruments, io_utils
from darkhunter_rv.pipeline import _weighted_method_rv_from_rows
from darkhunter_rv.plotting import (  # noqa: PLC2701
    _hb_mask_pseudo_on_wavelengths,
    _hb_tight_ylim_local_cont,
)

METHOD_ORDER = ("mask_ccf", "template_fft", "strong_lines")

METHOD_SPEC: dict[str, tuple[str, str]] = {
    "mask_ccf": ("mask_valid", "mask_rv_kms"),
    "template_fft": ("template_valid", "template_rv_kms"),
    "strong_lines": ("strong_lines_valid", "strong_lines_rv_kms"),
}

METHOD_LABEL = {
    "mask_ccf": "mask CCF",
    "template_fft": "template FFT",
    "strong_lines": "strong lines (Voigt+Lorentz Hβ)",
}


def _resolve_spectrum_path(raw: str, data_root: Path | None) -> Path:
    p = Path(raw)
    if p.is_file():
        return p
    if data_root is not None:
        q = data_root / p.name
        if q.is_file():
            return q
    raise FileNotFoundError(f"Spectrum not found: {raw}")


def _load_mask_arrays(mask_dir: Path, mask_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = mask_dir / f"{mask_name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Mask file missing: {path}")
    md = np.loadtxt(path)
    return np.asarray(md[:, 0], float), np.asarray(md[:, 1], float)


def plot_exposure_pdf_dual_mask_schematic(
    diag_csv: Path,
    out_pdf: Path,
    rv_a: float,
    rv_b: float,
    legend_a: str,
    legend_b: str,
    title_line_methods: str,
    mask_name: str,
    instrument_name: str,
    continuum_mode: str,
    data_root: Path | None,
) -> None:
    """
    One PDF: each page is one echelle order; obs norm flux + mask schematic at ``rv_a`` and ``rv_b``.
    """
    inst = instruments.get_instrument_profile(instrument_name)
    mask_dir = Path(inst.mask_directory)
    mw, ms = _load_mask_arrays(mask_dir, mask_name)

    df = pd.read_csv(diag_csv)
    fn = str(df["file"].iloc[0]) if "file" in df.columns and len(df) else ""
    spec_path = _resolve_spectrum_path(fn, data_root)
    _header, spec_data = io_utils.read_spectrum(str(spec_path))

    valid_orders = sorted(o for o in spec_data if o not in inst.bad_orders)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    dvl = float(rv_a) - float(rv_b)

    with PdfPages(out_pdf) as pdf:
        for o in valid_orders:
            d = spec_data[o]
            w = np.asarray(d["wavelength"], float)
            f = np.asarray(d["flux"], float)
            e = np.asarray(d["eflux"], float)
            if len(w) < 10:
                continue
            try:
                ckw: dict = {"continuum_mode": continuum_mode}
                if continuum_mode == "spline":
                    ckw["exclude_near_lines_width"] = float(config.COOL_SPLINE_EXCLUDE_NEAR_LINES_WIDTH)
                nw, nf, _ne = continuum.fit_continuum(w, f, e, **ckw)
            except Exception:
                continue

            fig, ax = plt.subplots(figsize=(10, 3.2))
            ax.step(nw, nf, where="mid", color="0.35", lw=0.75, alpha=0.9, label="obs (order norm)", zorder=2)

            pseudo_a = _hb_mask_pseudo_on_wavelengths(nw, mw, ms, float(rv_a))
            pseudo_b = _hb_mask_pseudo_on_wavelengths(nw, mw, ms, float(rv_b))
            if pseudo_a is not None:
                ax.plot(
                    nw,
                    pseudo_a,
                    "b-",
                    lw=0.9,
                    alpha=0.88,
                    label=f"mask schematic @ {legend_a} ({float(rv_a):+.2f} km/s)",
                    zorder=4,
                )
            if pseudo_b is not None:
                ax.plot(
                    nw,
                    pseudo_b,
                    color="tab:orange",
                    ls="--",
                    lw=0.95,
                    alpha=0.9,
                    label=f"mask schematic @ {legend_b} ({float(rv_b):+.2f} km/s)",
                    zorder=5,
                )

            flux_for_ylim: list[np.ndarray] = [nf]
            if pseudo_a is not None:
                flux_for_ylim.append(pseudo_a)
            if pseudo_b is not None:
                flux_for_ylim.append(pseudo_b)
            _hb_tight_ylim_local_cont(ax, nw, flux_for_ylim)

            stem = spec_path.stem
            ax.set_xlabel("Wavelength (Å)")
            ax.set_ylabel("Norm flux")
            ax.set_title(
                f"{stem}  order {o}  mask={mask_name}\n"
                f"{title_line_methods}\n"
                f"RV_a = {float(rv_a):+.3f} km/s   RV_b = {float(rv_b):+.3f} km/s   Δ = {dvl:+.3f} km/s"
            )
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _mask_name_from_diag(diag: Path) -> str | None:
    df_one = pd.read_csv(diag)
    recs = df_one.to_dict("records")
    mask_rows = [r for r in recs if str(r.get("method")) == "mask_ccf" and str(r.get("chunk_key")) != "all"]
    if not mask_rows:
        return None
    mask_name = str(mask_rows[0].get("mask_name") or "")
    if not mask_name or mask_name == "none":
        return None
    return mask_name


def _run_legacy_mode(args: argparse.Namespace) -> int:
    tab = pd.read_csv(args.comparison_csv)
    need = ["diagnostics_csv", "mask_rv_kms", "legacy_rv_kms", "legacy_err_kms"]
    for c in need:
        if c not in tab.columns:
            logging.error("comparison CSV missing column %r", c)
            return 2

    m = (
        np.isfinite(tab["mask_rv_kms"].astype(float).values)
        & np.isfinite(tab["legacy_rv_kms"].astype(float).values)
        & np.isfinite(tab["legacy_err_kms"].astype(float).values)
        & (tab["legacy_err_kms"].astype(float).values <= args.max_legacy_err)
    )
    sub = tab.loc[m].copy()
    sub["residual_mask_minus_legacy"] = sub["mask_rv_kms"].astype(float) - sub["legacy_rv_kms"].astype(
        float
    )
    sub["abs_residual"] = np.abs(sub["residual_mask_minus_legacy"].values)
    out = sub[sub["abs_residual"] > float(args.threshold_kms)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_list = args.out_dir / "legacy_outliers_mask_minus_legacy.csv"
    out.to_csv(out_list, index=False)
    logging.info("Wrote %d outlier rows -> %s", len(out), out_list.resolve())

    if len(out) == 0:
        return 0

    n_done = 0
    for _, row in out.iterrows():
        if args.max_exposures is not None and n_done >= int(args.max_exposures):
            logging.info("Stopped after --max-exposures=%d", args.max_exposures)
            break
        diag = Path(str(row["diagnostics_csv"]))
        if not diag.is_file():
            logging.warning("skip missing diagnostics %s", diag)
            continue
        mask_name = _mask_name_from_diag(diag)
        if mask_name is None:
            logging.warning("skip %s: no mask_name", diag)
            continue

        df_one = pd.read_csv(diag)
        recs = df_one.to_dict("records")
        rv_m, _ = _weighted_method_rv_from_rows(recs, "mask_ccf")
        leg = float(row["legacy_rv_kms"])

        stem = diag.stem.replace("_diagnostics", "")
        pdf_path = args.out_dir / f"{stem}_legacy_outlier_orders.pdf"
        try:
            plot_exposure_pdf_dual_mask_schematic(
                diag,
                pdf_path,
                rv_m,
                leg,
                "mask CCF (exposure stack)",
                "legacy",
                "mask CCF vs legacy",
                mask_name,
                args.instrument,
                args.continuum_mode,
                args.data_root,
            )
            logging.info("Wrote %s", pdf_path.resolve())
            n_done += 1
        except Exception as ex:
            logging.warning("failed %s: %s", diag, ex)

    return 0


def _run_overlap_method_pairs(args: argparse.Namespace) -> int:
    tab = pd.read_csv(args.overlap_csv)
    for method in METHOD_ORDER:
        vcol, rcol = METHOD_SPEC[method]
        if vcol not in tab.columns or rcol not in tab.columns:
            logging.error("overlap CSV missing columns for %s (%s, %s)", method, vcol, rcol)
            return 2
    if "diagnostics_csv" not in tab.columns:
        logging.error("overlap CSV missing diagnostics_csv")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []

    for _, row in tab.iterrows():
        diag = Path(str(row["diagnostics_csv"]))
        for a, b in itertools.combinations(METHOD_ORDER, 2):
            va, ra = METHOD_SPEC[a]
            vb, rb = METHOD_SPEC[b]
            if not bool(row[va]) or not bool(row[vb]):
                continue
            rv_a = float(row[ra])
            rv_b = float(row[rb])
            if not np.isfinite(rv_a) or not np.isfinite(rv_b):
                continue
            if abs(rv_a - rv_b) <= float(args.threshold_kms):
                continue
            out_rows.append(
                {
                    "diagnostics_csv": str(diag),
                    "method_a": a,
                    "method_b": b,
                    "rv_a_kms": rv_a,
                    "rv_b_kms": rv_b,
                    "delta_a_minus_b_kms": rv_a - rv_b,
                }
            )

    listing = pd.DataFrame(out_rows)
    list_path = args.out_dir / "method_pair_discrepancy_listing.csv"
    listing.to_csv(list_path, index=False)
    logging.info("Wrote %d pair rows -> %s", len(listing), list_path.resolve())

    if len(listing) == 0:
        return 0

    n_done = 0
    for _, prow in listing.iterrows():
        if args.max_exposures is not None and n_done >= int(args.max_exposures):
            logging.info("Stopped after --max-exposures=%d", args.max_exposures)
            break
        diag = Path(str(prow["diagnostics_csv"]))
        if not diag.is_file():
            logging.warning("skip missing diagnostics %s", diag)
            continue
        mask_name = _mask_name_from_diag(diag)
        if mask_name is None:
            logging.warning("skip %s: no mask_name", diag)
            continue

        ma, mb = str(prow["method_a"]), str(prow["method_b"])
        rv_a = float(prow["rv_a_kms"])
        rv_b = float(prow["rv_b_kms"])
        stem = diag.stem.replace("_diagnostics", "")
        safe_a = ma.replace(" ", "_")
        safe_b = mb.replace(" ", "_")
        pdf_path = args.out_dir / f"{stem}__{safe_a}__{safe_b}_discrep_orders.pdf"
        try:
            plot_exposure_pdf_dual_mask_schematic(
                diag,
                pdf_path,
                rv_a,
                rv_b,
                METHOD_LABEL[ma],
                METHOD_LABEL[mb],
                f"{METHOD_LABEL[ma]} vs {METHOD_LABEL[mb]} (both valid)",
                mask_name,
                args.instrument,
                args.continuum_mode,
                args.data_root,
            )
            logging.info("Wrote %s", pdf_path.resolve())
            n_done += 1
        except Exception as ex:
            logging.warning("failed %s: %s", pdf_path, ex)

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="PDF per exposure: all orders, mask schematic at two RVs (legacy or method-pair mode)"
    )
    ap.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help="Legacy mode: method_comparison_per_exposure.csv from rv_method_diagnostics_report",
    )
    ap.add_argument(
        "--overlap-csv",
        type=Path,
        default=None,
        help="Method-pair mode: overlap_enriched_per_exposure.csv from rv_method_overlap_report",
    )
    ap.add_argument("--threshold-kms", type=float, default=5.0)
    ap.add_argument(
        "--max-legacy-err",
        type=float,
        default=float("inf"),
        help="Legacy mode only: max legacy σ (km/s) to include a row",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--instrument", default="APF")
    ap.add_argument("--continuum-mode", choices=["spline", "blaze"], default="spline")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="If spectrum paths in diagnostics are not found, try data-root / basename",
    )
    ap.add_argument(
        "--max-exposures",
        type=int,
        default=None,
        help="Cap how many PDFs to build (legacy: outlier exposures; overlap: pair-PDFs, in listing order)",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    if args.overlap_csv is not None and args.comparison_csv is not None:
        logging.error("Pass only one of --overlap-csv or --comparison-csv")
        return 2
    if args.overlap_csv is not None:
        if not args.overlap_csv.is_file():
            logging.error("overlap CSV not found: %s", args.overlap_csv)
            return 2
        return _run_overlap_method_pairs(args)
    if args.comparison_csv is None:
        logging.error("Provide --comparison-csv (legacy) or --overlap-csv (method pairs)")
        return 2
    if not args.comparison_csv.is_file():
        logging.error("comparison CSV not found: %s", args.comparison_csv)
        return 2
    return _run_legacy_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
