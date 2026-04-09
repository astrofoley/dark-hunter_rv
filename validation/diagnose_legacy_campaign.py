#!/usr/bin/env python3
"""Compare legacy summaries to darkhunter_rv; optional Gaia (+LAMOST/RAVE)."""
from __future__ import annotations

import argparse
import json
import re
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VAL_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _VAL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from method_pair_stats import compute_method_pair_table  # noqa: E402


def _parse_gaia_id(filename: str):
    m = re.search(r"Gaia_DR3_(\d{18,19})", str(filename))
    return int(m.group(1)) if m else None


def parse_summary_file(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    text = path.read_text()
    if "[PIPELINE RESULTS]" in text:
        chunk = text.split("[PIPELINE RESULTS]", 1)[-1]
        lines = chunk.splitlines()
    else:
        lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        if len(parts) >= 6 and parts[-1] in ("True", "False"):
            parts = parts[:-1]
        if len(parts) < 5:
            continue
        try:
            rows.append({
                "basename": parts[0],
                "mjd": float(parts[1]),
                "rv": float(parts[2]),
                "rv_err": float(parts[3]),
                "rms": float(parts[4]),
            })
        except ValueError:
            continue
    return rows


def summary_rows_by_basename(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for r in rows:
        out[Path(r["basename"]).name] = r
    return out


def apply_legacy_filters(rows, max_abs_rv, max_err, max_rms):
    out = []
    for r in rows:
        ok = (
            np.isfinite(r["rv"])
            and np.isfinite(r["rv_err"])
            and np.isfinite(r["rms"])
            and abs(r["rv"]) <= max_abs_rv
            and r["rv_err"] <= max_err
            and r["rms"] <= max_rms
        )
        out.append({**r, "passes_legacy_filter": bool(ok)})
    return out


def gaia_to_jsonable(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: gaia_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [gaia_to_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, str):
        return obj
    try:
        f = float(obj)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return str(obj)


def weighted_rv(g: pd.DataFrame):
    rv = g["rv_kms"].astype(float).values
    err = g["rv_err_kms"].astype(float).values
    ok = np.isfinite(rv) & np.isfinite(err) & (err > 0) & (err < 1e28)
    if not np.any(ok):
        return float("nan"), float("nan")
    rv, err = rv[ok], err[ok]
    w = 1.0 / (err ** 2 + 1e-18)
    mu = float(np.sum(w * rv) / np.sum(w))
    sig = float(np.sqrt(1.0 / np.sum(w)))
    return mu, sig


def exposure_rv_from_diag_csv(path: Path, require_mask_qc: bool) -> tuple[float, float, float]:
    if not path.exists():
        return (float("nan"), float("nan"), float("nan"))
    df = pd.read_csv(path)
    if "exposure_rv_kms" in df.columns and df["exposure_rv_kms"].notna().any():
        v = float(df["exposure_rv_kms"].dropna().iloc[0])
        e = (
            float(df["exposure_rv_err_kms"].dropna().iloc[0])
            if "exposure_rv_err_kms" in df.columns and df["exposure_rv_err_kms"].notna().any()
            else float("nan")
        )
        sm = df[(df["method"] == "mask_ccf")]
        if "used_in_exposure_stack" in sm.columns:
            u = sm[sm["used_in_exposure_stack"].fillna(False).astype(bool)]
            if len(u) > 0:
                sm = u
        rvs = sm["rv_kms"].astype(float).values
        rvs = rvs[np.isfinite(rvs)]
        rms = float(np.std(rvs)) if len(rvs) > 1 else (0.0 if len(rvs) == 1 else float("nan"))
        if not np.isfinite(rms):
            rms = 0.0
        return v, e, rms
    m = df[df["method"] == "mask_ccf"]
    if m.empty:
        return (float("nan"), float("nan"), float("nan"))
    if "used_in_exposure_stack" in m.columns:
        u = m[m["used_in_exposure_stack"].fillna(False).astype(bool)]
        if len(u) > 0:
            m = u
    if require_mask_qc and "qc_pass" in m.columns:
        m = m[m["qc_pass"].fillna(False).astype(bool)]
    if m.empty:
        return (float("nan"), float("nan"), float("nan"))
    mu, sig = weighted_rv(m)
    rvs = m["rv_kms"].astype(float).values
    rvs = rvs[np.isfinite(rvs)]
    rms = float(np.std(rvs)) if len(rvs) > 1 else 0.0
    return mu, sig, rms


def exposure_method_summary(df: pd.DataFrame, require_mask_qc: bool) -> pd.DataFrame:
    rows = []
    for fn, g in df.groupby("file"):
        base = Path(str(fn)).name
        for method, gm in g.groupby("method"):
            gm2 = gm
            if method == "mask_ccf" and "used_in_exposure_stack" in gm.columns:
                u = gm[gm["used_in_exposure_stack"].fillna(False).astype(bool)]
                if len(u) > 0:
                    gm2 = u
                else:
                    gm2 = gm
            else:
                gm2 = gm
            if method == "mask_ccf" and require_mask_qc and "qc_pass" in gm2.columns:
                gm2 = gm2[gm2["qc_pass"].fillna(False).astype(bool)]
            if gm2.empty:
                continue
            mu, sig = weighted_rv(gm2)
            rows.append({
                "basename": base,
                "method": str(method),
                "rv_kms": mu,
                "rv_err_kms": sig,
                "n_chunks": int(len(gm2)),
            })
    return pd.DataFrame(rows)


def teff_bin_edges(teffs: np.ndarray, n_bins: int) -> np.ndarray:
    t = teffs[np.isfinite(teffs)]
    if len(t) < 2:
        return np.array([4000.0, 8000.0])
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(t, q))
    if len(edges) < 2:
        edges = np.array([float(np.min(t)), float(np.max(t)) + 1.0])
    return edges


def build_by_teff_bin(comparison, diag_concat, teff_col, n_bins):
    sub = comparison[comparison["passes_legacy_filter"]].copy()
    if sub.empty or teff_col not in sub.columns:
        return pd.DataFrame()
    edges = teff_bin_edges(sub[teff_col].astype(float).values, n_bins)
    sub["_bin"] = pd.cut(sub[teff_col], bins=edges, include_lowest=True)

    def rms_delta(x: pd.Series) -> float:
        a = x.dropna().values.astype(float)
        return float(np.std(a, ddof=1)) if len(a) > 1 else 0.0

    agg = sub.groupby("_bin", observed=False).agg(
        n=("delta_rv", "count"),
        mean_delta=("delta_rv", "mean"),
        rms_delta=("delta_rv", rms_delta),
        mean_teff=(teff_col, "mean"),
    ).reset_index()

    if diag_concat is not None and not diag_concat.empty and "method" in diag_concat.columns:
        mask_rows = diag_concat[diag_concat["method"] == "mask_ccf"]
        if "qc_pass" in mask_rows.columns and "teff" in mask_rows.columns:
            bins = pd.cut(mask_rows["teff"], bins=edges, include_lowest=True)
            qc_rate = (
                mask_rows.assign(_bin=bins)
                .groupby("_bin", observed=False)["qc_pass"]
                .apply(lambda s: float(s.fillna(False).astype(bool).mean()))
                .reset_index(name="mask_qc_pass_rate")
            )
            agg = agg.merge(qc_rate, on="_bin", how="left")
    return agg


def run_pipeline_on_file(repo_root, spectrum_path, new_output_dir, plot_dir, teff, extra_argv):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env["DARKHUNTER_OUTPUT_DIR"] = str(new_output_dir)
    if plot_dir is not None:
        env["DARKHUNTER_PLOT_DIR"] = str(plot_dir)
    cmd = [sys.executable, "-m", "darkhunter_rv.pipeline", str(spectrum_path)]
    if teff is not None and np.isfinite(teff):
        cmd.extend(["--teff", str(teff)])
    cmd.extend(extra_argv)
    p = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True)
    if p.returncode != 0:
        spec = Path(spectrum_path).name
        err = (p.stderr or "").strip()
        out = (p.stdout or "").strip()
        tail = 12000
        if err or out:
            parts = []
            if err:
                parts.append(f"stderr (last {tail} chars):\n{err[-tail:]}")
            if out:
                parts.append(f"stdout (last {tail} chars):\n{out[-tail:]}")
            print(
                f"--- pipeline failed exit {p.returncode} {spec} ---\n"
                + "\n\n".join(parts)
                + "\n---",
                file=sys.stderr,
            )
        else:
            print(
                f"--- pipeline failed exit {p.returncode} {spec} (no captured output) ---",
                file=sys.stderr,
            )
    return p.returncode


def _split_argv(argv: list[str]):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def diagnose_one_star(
    args: argparse.Namespace,
    gaia_id: int,
    spectra: list[Path],
    report_dir: Path,
    legacy_summary_path: Path,
    pipe_extra: list[str],
    plot_dir: Path | None,
) -> None:
    """Build comparison + plots under report_dir for one Gaia source_id."""
    report_dir.mkdir(parents=True, exist_ok=True)

    legacy_rows = apply_legacy_filters(
        parse_summary_file(legacy_summary_path),
        args.max_abs_legacy_rv,
        args.max_legacy_err,
        args.max_legacy_rms,
    )
    legacy_by_base = summary_rows_by_basename(legacy_rows)

    gaia_data = None
    teff_star = float(args.default_teff)
    new_summary_file = args.new_output_dir / f"Gaia_DR3_{gaia_id}_summary.txt"
    if args.query_gaia:
        from darkhunter_rv import gaia_utils  # noqa: E402

        gaia_data = gaia_utils.resolve_gaia_data(gaia_id, new_summary_file, args.force_gaia)
        if args.dump_gaia_json and gaia_data is not None:
            (report_dir / f"gaia_query_{gaia_id}.json").write_text(
                json.dumps(gaia_to_jsonable(gaia_data), indent=2)
            )
        if gaia_data and gaia_data.get("metadata"):
            md = gaia_data["metadata"]
            tg = md.get("Teff", md.get("teff"))
            if tg is not None:
                try:
                    tf = float(tg)
                    if np.isfinite(tf):
                        teff_star = tf
                except (TypeError, ValueError):
                    pass

    pipe_args = list(pipe_extra)
    if args.force_gaia:
        pipe_args.append("--force-gaia")
    if args.plots:
        pipe_args.extend(["--plots", "--plot-dir", str(plot_dir)])

    if args.run_pipeline:
        for sp in spectra:
            rc = run_pipeline_on_file(_REPO_ROOT, sp, args.new_output_dir, plot_dir, teff_star, pipe_args)
            if rc != 0:
                print(f"WARNING pipeline exit {rc} for {sp.name}", file=sys.stderr)

    new_rows = parse_summary_file(new_summary_file)
    new_by_base = summary_rows_by_basename(new_rows)

    md_flat = {}
    if gaia_data and gaia_data.get("metadata"):
        for k, v in gaia_data["metadata"].items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                try:
                    fv = float(v)
                    md_flat[f"gaia_{k}"] = fv if np.isfinite(fv) else None
                except (TypeError, ValueError):
                    md_flat[f"gaia_{k}"] = None
            else:
                md_flat[f"gaia_{k}"] = v

    comp_rows = []
    for sp in spectra:
        base = sp.name
        leg = legacy_by_base.get(base)
        neu = new_by_base.get(base)
        diag_csv = args.new_output_dir / f"{Path(base).stem}_diagnostics.csv"
        if neu is None or not np.isfinite(neu.get("rv", float("nan"))):
            dr, de, drm = exposure_rv_from_diag_csv(diag_csv, args.require_mask_qc)
            if np.isfinite(dr):
                if neu is None:
                    neu = {"basename": base, "mjd": float("nan"), "rv": dr, "rv_err": de, "rms": drm}
                else:
                    neu = {**neu, "rv": dr, "rv_err": de, "rms": drm}
        mjd_v = np.nan
        if leg and np.isfinite(leg.get("mjd", np.nan)):
            mjd_v = float(leg["mjd"])
        elif neu and np.isfinite(neu.get("mjd", np.nan)):
            mjd_v = float(neu["mjd"])
        row = {
            "gaia_source_id": gaia_id,
            "basename": base,
            "mjd": mjd_v,
            "legacy_rv": leg["rv"] if leg else np.nan,
            "legacy_err": leg["rv_err"] if leg else np.nan,
            "legacy_rms": leg["rms"] if leg else np.nan,
            "passes_legacy_filter": leg["passes_legacy_filter"] if leg else False,
            "new_rv": neu["rv"] if neu else np.nan,
            "new_err": neu["rv_err"] if neu else np.nan,
            "new_rms": neu["rms"] if neu else np.nan,
            "teff_used": teff_star,
        }
        row.update(md_flat)
        if leg and neu and np.isfinite(leg["rv"]) and np.isfinite(neu["rv"]):
            row["delta_rv"] = float(neu["rv"] - leg["rv"])
        else:
            row["delta_rv"] = np.nan
        comp_rows.append(row)

    comparison = pd.DataFrame(comp_rows)
    comparison.to_csv(report_dir / "exposure_comparison.csv", index=False)

    diag_files = sorted(args.new_output_dir.glob(f"Gaia_DR3_{gaia_id}_*_diagnostics.csv"))
    diag_concat = (
        pd.concat([pd.read_csv(p) for p in diag_files], ignore_index=True) if diag_files else pd.DataFrame()
    )

    if not diag_concat.empty:
        exposure_method_summary(diag_concat, args.require_mask_qc).to_csv(
            report_dir / "method_exposure_summary.csv", index=False
        )
        compute_method_pair_table(diag_concat).to_csv(report_dir / "method_pair_stats.csv", index=False)

    teff_col = "gaia_Teff" if "gaia_Teff" in comparison.columns else "teff_used"
    by_bin = build_by_teff_bin(comparison, diag_concat, teff_col, args.teff_bins)
    if not by_bin.empty:
        by_bin.to_csv(report_dir / "by_teff_bin.csv", index=False)

    good = comparison[comparison["passes_legacy_filter"]].dropna(subset=["new_rv", "legacy_rv"])
    min_scatter = max(1, int(args.legacy_scatter_min_points))
    if len(good) >= min_scatter:
        fig, (ax0, ax1) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(5, 6.5),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
            layout="constrained",
        )
        ax0.errorbar(
            good["legacy_rv"],
            good["new_rv"],
            xerr=good["legacy_err"],
            yerr=good["new_err"],
            fmt="o",
            ms=4,
            alpha=0.8,
        )
        lims = [
            float(min(good["legacy_rv"].min(), good["new_rv"].min())) - 5,
            float(max(good["legacy_rv"].max(), good["new_rv"].max())) + 5,
        ]
        if lims[1] > lims[0]:
            ax0.plot(lims, lims, "k--", lw=0.8)
        ax0.set_ylabel("New RV (km/s)")
        ax0.set_title(f"Legacy vs new (Gaia {gaia_id}, legacy-filtered)")
        ax0.grid(True, alpha=0.25)

        leg = good["legacy_rv"].astype(float).values
        newv = good["new_rv"].astype(float).values
        d_rv = newv - leg
        err_comb = np.sqrt(
            np.clip(good["new_err"].astype(float).values ** 2 + good["legacy_err"].astype(float).values ** 2, 0, None)
        )
        ax1.axhline(0, color="k", lw=0.6)
        ax1.errorbar(leg, d_rv, yerr=err_comb, fmt="o", ms=4, alpha=0.8, color="C0")
        ax1.set_xlabel("Legacy RV (km/s)")
        ax1.set_ylabel("ΔRV (km/s)")
        ax1.grid(True, alpha=0.25)
        fig.savefig(report_dir / "legacy_vs_new_rv.png", dpi=130)
        plt.close(fig)

    if len(good) >= min_scatter and good["delta_rv"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 4))
        tc = good[teff_col] if teff_col in good.columns else good["teff_used"]
        ax.scatter(tc, good["delta_rv"], alpha=0.8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(teff_col)
        ax.set_ylabel("new - legacy RV (km/s)")
        fig.tight_layout()
        fig.savefig(report_dir / "delta_rv_vs_teff.png", dpi=130)
        plt.close(fig)

    print(f"Wrote report under {report_dir}")

    if not args.skip_interpretation_report:
        from legacy_interpretation_report import generate_interpretation

        try:
            ip = generate_interpretation(
                report_dir,
                diag_dir=args.new_output_dir,
                pipeline_summary=new_summary_file,
                legacy_summary=legacy_summary_path,
            )
            print(f"Interpretation -> {ip}")
        except Exception as ex:
            print(f"WARNING interpretation report: {ex}", file=sys.stderr)


def main() -> None:
    main_argv, pipeline_argv = _split_argv(sys.argv[1:])
    ap = argparse.ArgumentParser(description="Diagnose pipeline vs legacy APF summaries")
    parent = _REPO_ROOT.parent
    ap.add_argument("--data-dir", type=Path, default=parent / "data")
    ap.add_argument("--legacy-output-dir", type=Path, default=parent / "output")
    ap.add_argument("--new-output-dir", type=Path, default=_REPO_ROOT / "validation_output" / "pipeline_rerun")
    ap.add_argument("--report-dir", type=Path, default=_REPO_ROOT / "validation_output" / "diagnose_legacy")
    ap.add_argument("--spectrum-glob", default="Gaia_DR3_1702*.txt")
    ap.add_argument("--legacy-summary", type=Path, default=None)
    ap.add_argument("--run-pipeline", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--query-gaia", action="store_true")
    ap.add_argument(
        "--force-gaia",
        action="store_true",
        help="Always query Gaia (also passed to pipeline with --run-pipeline)",
    )
    ap.add_argument("--dump-gaia-json", action="store_true")
    ap.add_argument("--default-teff", type=float, default=5800.0)
    ap.add_argument("--max-abs-legacy-rv", type=float, default=200.0)
    ap.add_argument("--max-legacy-err", type=float, default=0.5)
    ap.add_argument("--max-legacy-rms", type=float, default=0.5)
    ap.add_argument(
        "--legacy-scatter-min-points",
        type=int,
        default=2,
        help="Minimum legacy-filtered points with finite legacy+new RV to write legacy_vs_new_rv.png (use 1 for debugging)",
    )
    ap.add_argument("--require-mask-qc", action="store_true")
    ap.add_argument("--teff-bins", type=int, default=5)
    ap.add_argument(
        "--skip-interpretation-report",
        action="store_true",
        help="Do not write interpretation_summary.txt and extra plots",
    )
    ap.add_argument(
        "--multi-star",
        action="store_true",
        help="Allow multiple Gaia IDs in glob; write per-star subdirs under report-dir",
    )
    ap.add_argument("--min-epochs", type=int, default=1, help="With --multi-star, skip stars with fewer epochs")
    ap.add_argument(
        "--write-combined-csv",
        action="store_true",
        help="Write exposure_comparison_all_stars.csv at report-dir root (multi-star)",
    )
    args = ap.parse_args(main_argv)

    extra = list(pipeline_argv)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.new_output_dir.mkdir(parents=True, exist_ok=True)

    spectra = sorted(args.data_dir.glob(args.spectrum_glob))
    if not spectra:
        raise SystemExit(f"No spectra matching {args.spectrum_glob} under {args.data_dir}")

    from collections import defaultdict

    by_id: dict[int, list[Path]] = defaultdict(list)
    for sp in spectra:
        gid = _parse_gaia_id(sp.name)
        if gid is not None:
            by_id[gid].append(sp)
    if not by_id:
        raise SystemExit("No Gaia_DR3_<id> spectra in glob.")

    ids_sorted = sorted(by_id.keys(), key=lambda g: -len(by_id[g]))
    ids_sorted = [g for g in ids_sorted if len(by_id[g]) >= args.min_epochs]

    if len(ids_sorted) > 1 and not args.multi_star:
        raise SystemExit(
            f"Glob matched {len(ids_sorted)} source IDs {ids_sorted[:5]}... "
            "Use --multi-star or narrow --spectrum-glob."
        )

    plot_dir = (args.new_output_dir.parent / "pipeline_rerun_plots") if args.plots else None
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    combined_frames = []
    for gaia_id in ids_sorted:
        sp_list = sorted(by_id[gaia_id], key=lambda p: p.name)
        rdir = args.report_dir / str(gaia_id) if len(ids_sorted) > 1 else args.report_dir
        if args.legacy_summary is not None and len(ids_sorted) == 1:
            leg_path = args.legacy_summary
        else:
            leg_path = args.legacy_output_dir / f"{gaia_id}_summary.txt"
        diagnose_one_star(args, gaia_id, sp_list, rdir, leg_path, extra, plot_dir)
        if args.write_combined_csv and len(ids_sorted) > 1:
            combined_frames.append(pd.read_csv(rdir / "exposure_comparison.csv"))

    if combined_frames:
        pd.concat(combined_frames, ignore_index=True).to_csv(args.report_dir / "exposure_comparison_all_stars.csv")
        print(f"Wrote combined {args.report_dir / 'exposure_comparison_all_stars.csv'}")
if __name__ == "__main__":
    main()
