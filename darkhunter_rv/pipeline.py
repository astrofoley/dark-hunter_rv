# pipeline.py — main RV processing
from __future__ import annotations

import argparse
import csv
import re
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip

from . import config, instruments, io_utils, continuum, rv_core, templates, chunking, plotting

logger = logging.getLogger(__name__)


def setup_logging(level: str, quiet: bool) -> None:
    if quiet:
        level = "ERROR"
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _mask_tournament(
    spec_data: dict,
    instrument,
    test_orders: list,
    mask_glob_dir: Path,
    continuum_mode: str,
) -> tuple[dict | None, str, list[tuple[str, float]]]:
    """Return ({w,s}, best_name, [(stem, peak_sum), ...])."""
    inst_lower = instrument.name.lower()
    mask_files = list(mask_glob_dir.glob(f"*_{inst_lower}.txt"))
    if not mask_files:
        mask_files = list(mask_glob_dir.glob("*_espresso.txt"))
    if not mask_files:
        return None, "none", []

    scores: list[tuple[str, float]] = []
    best_peak = -np.inf
    best_pack: dict | None = None
    best_name = ""

    for mf in sorted(mask_files):
        try:
            md = np.loadtxt(mf)
            mw, ms = md[:, 0], md[:, 1]
        except Exception:
            continue
        peak_sum = 0.0
        for o in test_orders:
            if o not in spec_data:
                continue
            d = spec_data[o]
            w = np.array(d["wavelength"], float)
            f = np.array(d["flux"], float)
            e = np.array(d["eflux"], float)
            if len(w) < 10:
                continue
            try:
                nw, nf, _ = continuum.fit_continuum(w, f, e, continuum_mode=continuum_mode)
            except Exception:
                continue
            if nw[-1] < mw[0] or nw[0] > mw[-1]:
                continue
            _, _, _, _, p = rv_core.cross_correlate_stellar_mask(nw, 1.0 - nf, mw, ms)
            peak_sum += float(p)
        scores.append((mf.stem, peak_sum))
        if peak_sum > best_peak:
            best_peak = peak_sum
            best_pack = {"w": mw, "s": ms}
            best_name = mf.stem

    if not best_pack:
        return None, "none", scores
    return best_pack, best_name, scores


def process_spectrum(
    spectrum_file: str,
    args: argparse.Namespace,
    instrument,
    plot_root: Path | None,
) -> dict | None:
    spectrum_file = str(spectrum_file)
    logger.info("Processing instrument=%s file=%s", instrument.name, spectrum_file)

    if instrument.name == "GHOST":
        header, spec_data = io_utils.read_spectrum_ghost(spectrum_file)
    elif instrument.name == "MAROON-X":
        header, spec_data = io_utils.read_spectrum_maroonx(spectrum_file)
    else:
        header, spec_data = io_utils.read_spectrum(spectrum_file)

    mjd = io_utils.extract_mjd_from_header(header, instrument)
    teff = float(args.teff)
    use_fft_primary = teff > config.HOT_STAR_TEFF_THRESHOLD

    mask_dir = Path(instrument.mask_directory)
    valid_orders = sorted(o for o in spec_data if o not in instrument.bad_orders)
    if not valid_orders:
        logger.error("No valid orders in %s", spectrum_file)
        return None

    mid = len(valid_orders) // 2
    test_orders = valid_orders[max(0, mid - 2) : min(len(valid_orders), mid + 2)]

    bank = None
    vsini = 10.0
    mask_pack, best_mask_name, tournament_scores = None, "", []

    if not use_fft_primary:
        mask_pack, best_mask_name, tournament_scores = _mask_tournament(
            spec_data, instrument, test_orders, mask_dir, args.continuum_mode
        )
        mw = mask_pack["w"] if mask_pack is not None else None
        ms = mask_pack["s"] if mask_pack is not None else None
        if mask_pack is not None:
            logger.info("Mask tournament winner=%s scores=%s", best_mask_name, tournament_scores[:3])
    else:
        mw, ms = None, None
        logger.info("Teff=%s > hot threshold: primary path template FFT", teff)

    if use_fft_primary or args.run_all_methods:
        init_bank = templates.build_template_bank(teff, 50.0)
        if init_bank:
            tw, tf = next(iter(init_bank.values()))
            all_w, all_f = [], []
            for o in valid_orders:
                w = np.array(spec_data[o]["wavelength"], float)
                f = np.array(spec_data[o]["flux"], float)
                qw, qf = continuum.quick_normalize(w, f)
                all_w.append(qw)
                all_f.append(qf)
            if all_w:
                cat_w = np.concatenate(all_w)
                cat_f = np.concatenate(all_f)
                vb, _ = rv_core.estimate_broadening(cat_w, cat_f, tw, tf)
                if vb is not None:
                    vsini = float(vb)
        bank = templates.build_template_bank(teff, vsini)
        if not bank:
            logger.warning("No PHOENIX templates (dir %s). Template path skipped.", config.PHOENIX_BASE_DIR)

    if mask_pack is not None:
        mw, ms = mask_pack["w"], mask_pack["s"]

    bias: dict = {}
    if not args.no_bias and instrument.bias_file:
        bias = io_utils.read_bias(instrument.bias_file)

    diagnostics_rows: list[dict] = []
    rv_results: dict = {}
    primary_rv_list: list[float] = []
    primary_err_list: list[float] = []
    chunk_keys_plot: list[str] = []
    stem = Path(spectrum_file).stem

    if plot_root and tournament_scores:
        plotting.plot_tournament_scores(
            [a for a, _ in tournament_scores],
            [b for _, b in tournament_scores],
            plot_root / f"{stem}_tournament.png",
        )
    if tournament_scores:
        tourn_csv = config.OUTPUT_DIR / f"{stem}_tournament.csv"
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(tourn_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["mask_stem", "peak_score_sum"])
            for name, sc in tournament_scores:
                w.writerow([name, f"{sc:.6f}"])
        logger.info("Tournament table %s", tourn_csv)

    for chunk_key, w, f, e in chunking.iter_order_chunks(spec_data, instrument.bad_orders, args.subchunks):
        if len(w) < 10:
            continue
        try:
            nw, nf, ne = continuum.fit_continuum(w, f, e, continuum_mode=args.continuum_mode)
        except Exception as ex:
            logger.debug("continuum fail chunk=%s: %s", chunk_key, ex)
            continue

        bvec = bias.get(int(chunk_key.split("_")[0]), [0.0, 0.0, 0.0])
        if isinstance(bvec, (list, tuple)) and len(bvec) >= 3:
            b0, b1, b2 = float(bvec[0]), float(bvec[1]), float(bvec[2])
        else:
            b0, b1, b2 = 0.0, 0.0, 0.0

        # --- Mask CCF ---
        rv_m, err_m, vels_m, ccf_m, peak_m = np.nan, np.nan, None, None, 0.0
        gauss_ok_m = False
        if mw is not None and not (nw[-1] < mw[0] or nw[0] > mw[-1]):
            rv_m, err_m, vels_m, ccf_m, peak_m = rv_core.cross_correlate_stellar_mask(
                nw, 1.0 - nf, mw, ms
            )
            rv_m -= b0
            err_m = float(np.sqrt(err_m**2 + b1**2 + b2**2)) if np.isfinite(err_m) else err_m
            gauss_ok_m = np.isfinite(err_m) and err_m < 1e29

            diagnostics_rows.append(
                {
                    "file": spectrum_file,
                    "chunk_key": chunk_key,
                    "mjd": mjd,
                    "teff": teff,
                    "continuum_mode": args.continuum_mode,
                    "method": "mask_ccf",
                    "mask_name": best_mask_name,
                    "rv_kms": rv_m,
                    "rv_err_kms": err_m,
                    "ccf_peak": peak_m,
                    "gauss_ok": gauss_ok_m,
                    "template_key": "",
                }
            )
            if plot_root and vels_m is not None and ccf_m is not None:
                plotting.plot_ccf(
                    vels_m,
                    ccf_m,
                    plot_root / f"{stem}_chunk{chunk_key}_ccf_mask.png",
                    title=f"mask {chunk_key}",
                    peak_vel=float(rv_m) if np.isfinite(rv_m) else None,
                )

        # --- Template FFT ---
        rv_t, tpl_key = np.nan, ""
        if bank:
            try:
                rv_t, tpl_key = rv_core.estimate_rv_fft_vectorized(nw, 1.0 - nf, bank, vsini)
            except Exception as ex:
                logger.debug("fft RV chunk=%s: %s", chunk_key, ex)
            rv_t -= b0
            err_t = 10.0
            diagnostics_rows.append(
                {
                    "file": spectrum_file,
                    "chunk_key": chunk_key,
                    "mjd": mjd,
                    "teff": teff,
                    "continuum_mode": args.continuum_mode,
                    "method": "template_fft",
                    "mask_name": "",
                    "rv_kms": rv_t,
                    "rv_err_kms": err_t,
                    "ccf_peak": np.nan,
                    "gauss_ok": False,
                    "template_key": str(tpl_key),
                }
            )

        # Primary stacked value (same rule as before)
        if use_fft_primary and bank:
            rv_p, err_p = rv_t, 10.0
        elif mw is not None and not (nw[-1] < mw[0] or nw[0] > mw[-1]):
            rv_p, err_p = rv_m, err_m
        elif bank:
            rv_p, err_p = rv_t, 10.0
        else:
            rv_p, err_p = np.nan, np.nan

        if np.isfinite(rv_p) and np.isfinite(err_p) and err_p <= args.max_chunk_err:
            rv_results[chunk_key] = {"best_rv": rv_p, "best_rv_err": err_p}
            primary_rv_list.append(float(rv_p))
            primary_err_list.append(float(err_p))
            chunk_keys_plot.append(chunk_key)
            logger.info(
                "chunk=%s primary_rv=%.3f err=%.3f method=%s mask=%s",
                chunk_key,
                rv_p,
                err_p,
                "template" if use_fft_primary and bank else "mask",
                best_mask_name,
            )
        else:
            if np.isfinite(rv_p):
                logger.info("chunk=%s rejected rv=%.3f err=%.3f", chunk_key, rv_p, err_p)

        if plot_root and len(nw) > 10:
            plotting.plot_normalized_order(
                nw,
                nf,
                None,
                plot_root / f"{stem}_chunk{chunk_key}_norm.png",
                title=f"{chunk_key} norm ({args.continuum_mode})",
            )

    io_utils.write_order_results(rv_results, spectrum_file)

    diag_path = config.OUTPUT_DIR / f"{stem}_diagnostics.csv"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if diagnostics_rows:
        pd.DataFrame(diagnostics_rows).to_csv(diag_path, index=False)
        logger.info("Diagnostics CSV %s (%d rows)", diag_path, len(diagnostics_rows))

    rv_arr = np.array(primary_rv_list, float)
    err_arr = np.array(primary_err_list, float)
    if len(rv_arr) == 0:
        logger.warning("No finite RVs for %s", spectrum_file)
        mean_rv, mean_err, rms = np.nan, np.nan, np.nan
        fallback = False
    else:
        clipped = sigma_clip(rv_arr, sigma=2.2, masked=True)
        keys_arr = np.array(chunk_keys_plot, dtype=object)
        if np.ma.is_masked(clipped):
            keep = ~clipped.mask
            rv_arr = rv_arr[keep]
            err_arr = err_arr[keep]
            keys_arr = keys_arr[keep]
        wts = 1.0 / (err_arr**2 + 1e-9)
        mean_rv = float(np.average(rv_arr, weights=wts))
        mean_err = float(np.sqrt(1.0 / np.sum(wts)))
        rms = float(np.std(rv_arr))
        fallback = False

        if plot_root and len(keys_arr) == len(rv_arr) and len(rv_arr) > 0:
            plotting.plot_chunk_rvs(
                list(keys_arr),
                rv_arr,
                err_arr,
                plot_root / f"{stem}_chunk_rvs.png",
                title=stem,
            )

    # Strong-line fallback (exposure-level)
    if (use_fft_primary and (not np.isfinite(mean_rv) or rms > 100)) or args.run_all_methods:
        all_w, all_f, all_e = [], [], []
        for o in valid_orders:
            w = np.array(spec_data[o]["wavelength"], float)
            f = np.array(spec_data[o]["flux"], float)
            e = np.array(spec_data[o]["eflux"], float)
            nw, nf, _ = continuum.fit_continuum(w, f, e, continuum_mode=args.continuum_mode)
            all_w.append(nw)
            all_f.append(nf)
            all_e.append(np.ones_like(nf))
        if all_w:
            cw = np.concatenate(all_w)
            cf = np.concatenate(all_f)
            ix = np.argsort(cw)
            rv_sl, err_sl = rv_core.measure_strong_line_centroids(cw[ix], cf[ix])
            diagnostics_rows.append(
                {
                    "file": spectrum_file,
                    "chunk_key": "all",
                    "mjd": mjd,
                    "teff": teff,
                    "continuum_mode": args.continuum_mode,
                    "method": "strong_lines",
                    "mask_name": "",
                    "rv_kms": rv_sl,
                    "rv_err_kms": err_sl,
                    "ccf_peak": np.nan,
                    "gauss_ok": np.isfinite(err_sl),
                    "template_key": "",
                }
            )
            if use_fft_primary and (not np.isfinite(mean_rv) or (np.isfinite(rms) and rms > 100)):
                mean_rv, mean_err, rms = float(rv_sl), float(err_sl), 0.0
                fallback = True
                logger.warning("Strong-line fallback RV=%.3f", mean_rv)
            if plot_root and np.isfinite(rv_sl):
                plotting.plot_balmer_panels(spec_data, mjd, rv_sl, plot_root / f"{stem}_balmer.png")

    # Refresh diagnostics if strong_lines added
    if diagnostics_rows:
        pd.DataFrame(diagnostics_rows).to_csv(diag_path, index=False)

    logger.info(
        "SUMMARY file=%s mjd=%.6f rv=%.4f+/-%.4f rms=%.4f fallback=%s",
        spectrum_file,
        mjd,
        mean_rv if np.isfinite(mean_rv) else float("nan"),
 mean_err if np.isfinite(mean_err) else float("nan"),
        rms if np.isfinite(rms) else float("nan"),
        fallback,
    )

    return {
        "file": spectrum_file,
        "mjd": mjd,
        "rv": mean_rv,
        "rv_err": mean_err,
        "rv_rms": rms,
        "fallback": fallback,
    }


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Dark Hunter echelle RV pipeline")
    parser.add_argument("input_file", nargs="+", help="Spectra (txt, GHOST blue FITS, MAROON-X h5)")
    parser.add_argument("--instrument", default="APF")
    parser.add_argument("--teff", type=float, default=config.DEFAULT_TEFF)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--force-gaia", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--plots", action="store_true", help="Write diagnostic figures")
    parser.add_argument("--plot-dir", default=None, help="Override plot directory")
    parser.add_argument("--continuum-mode", choices=["spline", "blaze"], default="spline")
    parser.add_argument("--subchunks", type=int, default=1, help="Split each order into N pixel chunks")
    parser.add_argument("--max-chunk-err", type=float, default=50.0, help="Skip chunk RVs with err > this (km/s)")
    parser.add_argument(
        "--run-all-methods",
        action="store_true",
        help="Also run template / strong-line paths for diagnostics table",
    )

    args = parser.parse_args(argv)
    setup_logging(args.log_level, args.quiet)

    try:
        inst = instruments.get_instrument_profile(args.instrument)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(2)

    plot_root = None
    if args.plots:
        plot_root = Path(args.plot_dir) if args.plot_dir else config.PLOT_DIR
        plot_root.mkdir(parents=True, exist_ok=True)
        logger.info("Plots -> %s", plot_root)

    gaia_id = None
    gaia_data = None
    m = re.search(r"Gaia_DR3_(\d{18,19})", str(args.input_file[0]))
    if m:
        gaia_id = int(m.group(1))

    if gaia_id:
        from . import gaia_utils

        out_sum = config.OUTPUT_DIR / f"Gaia_DR3_{gaia_id}_summary.txt"
        if args.force_gaia or not out_sum.exists():
            gaia_data = gaia_utils.query_gaia_data(gaia_id)
            if gaia_data and gaia_data.get("metadata", {}).get("teff"):
                args.teff = float(gaia_data["metadata"]["teff"])
                logger.info("Gaia teff override %s", args.teff)

    results: list[dict] = []
    for fn in args.input_file:
        res = process_spectrum(fn, args, inst, plot_root)
        if res:
            results.append(res)

    if gaia_id:
        from . import gaia_utils

        io_utils.write_star_summary(gaia_id, gaia_data, results)
    else:
        io_utils.write_summary(results)


if __name__ == "__main__":
    main()
