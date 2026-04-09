#!/usr/bin/env python3
"""
Run the pipeline on the first spectrum of each Gaia source matched by a glob.

Uses a **single Python process** so :mod:`darkhunter_rv.templates` ``MODEL_CACHE`` reuses loaded
PHOENIX models across stars (keys still depend on Teff / vsini proxy per star).

Each run sets ``run_all_methods`` (same as ``python -m darkhunter_rv.pipeline … --run-all-methods`` /
``--compare-rv-methods``): mask CCF, template FFT, strong lines, and Hβ diagnostics for **every** star
regardless of Teff.

Example (first epoch per Gaia source; template FFT uses a **coarse (Teff, log g, [M/H]) pass**
then full vsini on the top few atmospheres when the bank is large — see ``config.FFT_COARSE_TOP_K``)::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.run_first_epoch_campaign \\
    --data-dir /Users/rfoley/darkhunter/rvs/data \\
    --spectrum-glob 'Gaia_DR3_*_epoch_*.txt' \\
    --plots \\
    --plot-dir validation_output/first_epoch_plots

Wide PHOENIX grid (optional)::

  python -m validation.run_first_epoch_campaign \\
    --data-dir /Users/rfoley/darkhunter/rvs/data \\
    --spectrum-glob 'Gaia_DR3_*_epoch_*.txt' \\
    --template-grid-wide

**Teff vs method** residual plots are **not** part of per-spectrum ``--plots``; they come from
aggregating ``*_diagnostics.csv``. After a campaign, use ``--method-teff-report`` or run
``python -m validation.rv_method_diagnostics_report`` (see that module's docstring).
Use ``--overlap-report`` to also write validity/overlap/adopted-RV tables and S/N-binned bias
(``validation.rv_method_overlap_report``).

Limit to **N** sources (sorted Gaia id) and full comparison reports::

  python -m validation.run_first_epoch_campaign \\
    --data-dir ../data \\
    --spectrum-glob 'Gaia_DR3_*_epoch_1.txt' \\
    --max-sources 20 \\
    --plots \\
    --plot-dir validation_output/batch20_plots \\
    --method-teff-report \\
    --method-report-out-dir validation_output/rv_method_teff_report_batch20 \\
    --overlap-report \\
    --overlap-report-out-dir validation_output/rv_method_overlap_report_batch20

With ``--max-sources``, Teff/overlap reports use a written ``diagnostics_list.txt`` so they include
**only** spectra from this run (not every CSV under ``OUTPUT_DIR`` matching the glob).
"""
from __future__ import annotations

import argparse
import copy
import logging
import re
import sys
from collections import defaultdict
from glob import glob as glob_paths
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config, instruments, io_utils, pipeline


def _default_diagnostics_glob(spectrum_glob: str) -> str:
    """Map e.g. Gaia_DR3_*_epoch_1.txt -> OUTPUT_DIR/Gaia_DR3_*_epoch_1_diagnostics.csv."""
    g = spectrum_glob.strip()
    if g.endswith(".txt") and "diagnostics" not in g.lower():
        return str(config.OUTPUT_DIR / g.replace(".txt", "_diagnostics.csv"))
    return str(config.OUTPUT_DIR / "Gaia_DR3_*_diagnostics.csv")


def _parse_gaia_id(name: str) -> int | None:
    m = re.search(r"Gaia_DR3_(\d{18,19})", str(name))
    return int(m.group(1)) if m else None


def _epoch_key(name: str) -> tuple[int, str]:
    m = re.search(r"epoch_(\d+)", name, re.I)
    return (int(m.group(1)) if m else 0, name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="First-epoch pipeline campaign (one Python process)")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--spectrum-glob", type=str, required=True, help="Glob under data-dir, quoted")
    ap.add_argument("--instrument", default="APF")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument(
        "--plots-minimal",
        "--plots-focus",
        action="store_true",
        dest="plots_focus",
        help="Match pipeline --plots-focus: chunk_rvs, rv_vs_order, ccf_orders, h_beta_rv only",
    )
    ap.add_argument(
        "--plots-skip-chunk-pngs",
        action="store_true",
        help="Match pipeline --plots-skip-chunk-pngs (see python -m darkhunter_rv.pipeline --help).",
    )
    ap.add_argument(
        "--plots-chunk-detail-stems-file",
        type=Path,
        default=None,
        help="Optional stems file for per-chunk PNGs when using --plots-skip-chunk-pngs.",
    )
    ap.add_argument("--plot-dir", type=Path, default=None)
    ap.add_argument("--force-gaia", action="store_true")
    ap.add_argument(
        "--query-gaia",
        action="store_true",
        help="Same as --force-gaia: always call the Gaia query path (do not rely only on cached star summary).",
    )
    ap.add_argument("--continuum-mode", choices=["spline", "blaze"], default="spline")
    ap.add_argument("--qc-config", default="order_chunk_qc.yaml")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument(
        "--template-grid-wide",
        action="store_true",
        help="Widen PHOENIX bank (more atmospheres and vsini samples); two-phase FFT helps most here.",
    )
    ap.add_argument(
        "--no-fft-two-phase",
        action="store_true",
        help="Full bank FFT every chunk (forwarded to pipeline).",
    )
    ap.add_argument(
        "--fft-coarse-top-k",
        type=int,
        default=None,
        help="Top atmosphere triples after coarse pass (default: package config).",
    )
    ap.add_argument(
        "--method-teff-report",
        action="store_true",
        help=(
            "After processing, run validation.rv_method_diagnostics_report to write Teff vs method "
            "residual plots (requires --run-all-methods data: this campaign always enables it)."
        ),
    )
    ap.add_argument(
        "--method-report-glob",
        type=str,
        default=None,
        help=(
            "Quoted glob for *_diagnostics.csv under OUTPUT_DIR. "
            "Default: same pattern as --spectrum-glob with .txt -> _diagnostics.csv in OUTPUT_DIR."
        ),
    )
    ap.add_argument(
        "--method-report-out-dir",
        type=Path,
        default=None,
        help="Report output directory (default: validation_output/rv_method_teff_report under repo root).",
    )
    ap.add_argument(
        "--overlap-report",
        action="store_true",
        help=(
            "After processing, run validation.rv_method_overlap_report (validity flags, adopted RV, "
            "Gaia RUWE/MH/log g merge, S/N-binned mask−template). Implies same diagnostics glob as "
            "the method Teff report unless overridden."
        ),
    )
    ap.add_argument(
        "--overlap-report-out-dir",
        type=Path,
        default=None,
        help="Overlap report directory (default: sibling rv_method_overlap_report next to method report dir).",
    )
    ap.add_argument(
        "--max-sources",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N sources (first epoch each), Gaia id order; use with reports to limit comparison set.",
    )
    ap.add_argument(
        "--all-epochs",
        action="store_true",
        help="Process every spectrum matching the glob (all epochs), not only the first epoch per Gaia source.",
    )
    ap.add_argument(
        "--plots-skip-order-summary",
        action="store_true",
        help="Match pipeline --plots-skip-order-summary (no *_ccf_orders.png / *_rv_vs_order.png).",
    )
    args_ns = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args_ns.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    pattern = str(args_ns.data_dir / args_ns.spectrum_glob)
    paths = [Path(p) for p in glob_paths(pattern)]
    if not paths:
        logging.error("No files matched %s", pattern)
        return 2

    by_id: dict[int, list[Path]] = defaultdict(list)
    for p in paths:
        gid = _parse_gaia_id(p.name)
        if gid is not None:
            by_id[gid].append(p)

    if not by_id:
        logging.error("No Gaia_DR3_<id> filenames in glob.")
        return 2

    first_paths: list[Path] = []
    for gid in sorted(by_id.keys()):
        sp = sorted(by_id[gid], key=lambda x: _epoch_key(x.name))
        first_paths.append(sp[0])

    if args_ns.all_epochs:
        run_paths: list[Path] = []
        for gid in sorted(by_id.keys()):
            run_paths.extend(sorted(by_id[gid], key=lambda x: _epoch_key(x.name)))
    else:
        run_paths = list(first_paths)

    if args_ns.max_sources is not None and not args_ns.all_epochs:
        n = max(0, int(args_ns.max_sources))
        run_paths = run_paths[:n]
        logging.info("Limited to --max-sources %d (first epoch per source)", n)
    elif args_ns.max_sources is not None and args_ns.all_epochs:
        logging.warning("--max-sources is ignored when --all-epochs is set")

    logging.info(
        "Campaign: %d spectra, PHOENIX MODEL_CACHE shared in this process%s",
        len(run_paths),
        "; all epochs" if args_ns.all_epochs else "; first epoch per source",
    )

    base = argparse.Namespace(
        instrument=args_ns.instrument,
        teff=config.DEFAULT_TEFF,
        logg=None,
        mh=None,
        template_grid_wide=bool(args_ns.template_grid_wide),
        no_fft_two_phase=bool(args_ns.no_fft_two_phase),
        fft_coarse_top_k=args_ns.fft_coarse_top_k,
        hb_rv_fallback=False,
        no_bias=False,
        force_gaia=bool(args_ns.force_gaia or args_ns.query_gaia),
        quiet=False,
        log_level=args_ns.log_level,
        plots=args_ns.plots,
        plots_focus=bool(args_ns.plots_focus),
        plots_skip_chunk_pngs=bool(args_ns.plots_skip_chunk_pngs),
        plots_skip_order_summary=bool(args_ns.plots_skip_order_summary),
        plots_chunk_detail_stems_file=args_ns.plots_chunk_detail_stems_file,
        plot_dir=str(args_ns.plot_dir) if args_ns.plot_dir else None,
        continuum_mode=args_ns.continuum_mode,
        subchunks=1,
        max_chunk_err=50.0,
        qc_config=args_ns.qc_config,
        write_qc_config=False,
        run_all_methods=True,
        plots_strong_line_panels=False,
    )

    inst = instruments.get_instrument_profile(base.instrument)
    plot_root = None
    if base.plots:
        plot_root = Path(base.plot_dir) if base.plot_dir else config.PLOT_DIR
        plot_root.mkdir(parents=True, exist_ok=True)

    from darkhunter_rv import gaia_utils

    gaia_cache: dict[int, object] = {}
    results_by_gid: dict[int, list[dict]] = defaultdict(list)
    diagnostic_csv_paths: list[Path] = []
    for p in run_paths:
        args_f = copy.copy(base)
        gid = _parse_gaia_id(p.name)
        if gid is None:
            continue
        if gid not in gaia_cache:
            out_sum = config.OUTPUT_DIR / f"Gaia_DR3_{gid}_summary.txt"
            gaia_cache[gid] = gaia_utils.resolve_gaia_data(gid, out_sum, args_f.force_gaia)
        pipeline._apply_gaia_metadata_to_args(args_f, gaia_cache[gid])
        pipeline._attach_diagnostics_teff(args_f, gid, gaia_cache[gid])
        io_utils.write_star_summary(gid, gaia_cache.get(gid), results_by_gid[gid])
        if args_f.logg is None:
            args_f.logg = 4.5
        if args_f.mh is None:
            args_f.mh = 0.0
        res = pipeline.process_spectrum(str(p), args_f, inst, plot_root)
        if res:
            results_by_gid[gid].append(res)
            diagnostic_csv_paths.append(config.OUTPUT_DIR / f"{p.stem}_diagnostics.csv")
            io_utils.write_star_summary(gid, gaia_cache.get(gid), results_by_gid[gid])

    logging.info(
        "Done. Wrote %d star summaries under %s; diagnostics CSVs per spectrum under OUTPUT_DIR.",
        len(results_by_gid),
        config.OUTPUT_DIR,
    )

    if args_ns.method_teff_report:
        from validation import rv_method_diagnostics_report

        dg = args_ns.method_report_glob or _default_diagnostics_glob(args_ns.spectrum_glob)
        out_rep = Path(
            args_ns.method_report_out_dir or (_REPO_ROOT / "validation_output" / "rv_method_teff_report")
        )
        out_rep.mkdir(parents=True, exist_ok=True)
        use_list = bool(diagnostic_csv_paths)
        if use_list:
            manifest = out_rep / "diagnostics_list.txt"
            manifest.write_text(
                "\n".join(str(p.resolve()) for p in diagnostic_csv_paths if p) + "\n",
                encoding="utf-8",
            )
            logging.info(
                "Running method-vs-Teff report: diagnostics_list=%s (%d files) out_dir=%s",
                manifest,
                len(diagnostic_csv_paths),
                out_rep,
            )
            diag_arg = ["--diagnostics-list", str(manifest)]
        else:
            logging.info("Running method-vs-Teff report: diagnostics_glob=%s out_dir=%s", dg, out_rep)
            diag_arg = ["--diagnostics-glob", dg]
        rc = rv_method_diagnostics_report.main(
            diag_arg
            + [
                "--legacy-summary-dir",
                str(config.OUTPUT_DIR),
                "--max-legacy-err",
                "0.5",
                "--max-method-err",
                str(float(config.COMPARISON_REPORT_MAX_RV_ERR_KMS)),
                "--teff-bin-lo",
                str(float(config.COMPARISON_REPORT_TEFF_BIN_LO_K)),
                "--teff-bin-hi",
                str(float(config.COMPARISON_REPORT_TEFF_BIN_HI_K)),
                "--out-dir",
                str(out_rep),
            ]
        )
        if rc != 0:
            logging.warning("rv_method_diagnostics_report exited with code %s", rc)
        else:
            logging.info(
                "Teff / method plots (e.g. residual_mask_minus_template_vs_teff.png) -> %s",
                out_rep.resolve(),
            )

    if args_ns.overlap_report:
        from validation import rv_method_overlap_report

        dg = args_ns.method_report_glob or _default_diagnostics_glob(args_ns.spectrum_glob)
        out_ol = args_ns.overlap_report_out_dir
        if out_ol is None:
            base = args_ns.method_report_out_dir or (_REPO_ROOT / "validation_output" / "rv_method_teff_report")
            out_ol = base.resolve().parent / "rv_method_overlap_report"
        out_ol = Path(out_ol)
        out_ol.mkdir(parents=True, exist_ok=True)
        if diagnostic_csv_paths:
            manifest_ol = out_ol / "diagnostics_list.txt"
            manifest_ol.write_text(
                "\n".join(str(p.resolve()) for p in diagnostic_csv_paths if p) + "\n",
                encoding="utf-8",
            )
            logging.info(
                "Running overlap report: diagnostics_list=%s (%d files) out_dir=%s",
                manifest_ol,
                len(diagnostic_csv_paths),
                out_ol,
            )
            diag_arg_ol = ["--diagnostics-list", str(manifest_ol)]
        else:
            logging.info("Running overlap report: diagnostics_glob=%s out_dir=%s", dg, out_ol)
            diag_arg_ol = ["--diagnostics-glob", dg]
        overlap_argv = diag_arg_ol + [
            "--gaia-summary-dir",
            str(config.OUTPUT_DIR),
            "--max-method-err",
            str(float(config.COMPARISON_REPORT_MAX_RV_ERR_KMS)),
            "--out-dir",
            str(out_ol),
        ]
        if diagnostic_csv_paths and len(diagnostic_csv_paths) < 30:
            overlap_argv += ["--min-bin-count", "3"]
        rc_ol = rv_method_overlap_report.main(overlap_argv)
        if rc_ol != 0:
            logging.warning("rv_method_overlap_report exited with code %s", rc_ol)
        else:
            logging.info("Overlap report -> %s", out_ol.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
