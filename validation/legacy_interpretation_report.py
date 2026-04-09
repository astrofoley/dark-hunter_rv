#!/usr/bin/env python3
"""Build interpretation_summary.txt and extra plots from diagnose_legacy_campaign outputs."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "validation") not in sys.path:
    sys.path.insert(0, str(_REPO / "validation"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from darkhunter_rv import plotting as dh_plotting


def _norm_base(s: str) -> str:
    return Path(str(s)).name


def parse_summary_mjd(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    text = path.read_text()
    if "[PIPELINE RESULTS]" in text:
        chunk = text.split("[PIPELINE RESULTS]", 1)[-1]
        lines = chunk.splitlines()
    else:
        lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        if len(parts) >= 6 and parts[-1] in ("True", "False"):
            parts = parts[:-1]
        if len(parts) < 5:
            continue
        try:
            out[_norm_base(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return out


def _finite_series(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").values
    return x[np.isfinite(x)]


def _plot_method_chunk_finite_counts(
    diag_dir: Path,
    gaia_source_id: int,
    report_dir: Path,
    title_suffix: str = "",
) -> None:
    """Per-epoch bar chart: chunks with finite RV per method (diagnostics CSV rows)."""
    pat = f"Gaia_DR3_{gaia_source_id}_*_diagnostics.csv"
    files = sorted(diag_dir.glob(pat))
    if not files:
        return
    methods = ("mask_ccf", "template_fft", "strong_lines")
    epoch_labels: list[str] = []
    counts = {m: [] for m in methods}
    for p in files:
        epoch_labels.append(
            re.sub(r"^Gaia_DR3_\d+_", "", p.name).replace("_diagnostics.csv", "")
        )
        dfc = pd.read_csv(p)
        for m in methods:
            if m not in dfc["method"].values:
                counts[m].append(0)
                continue
            v = pd.to_numeric(dfc.loc[dfc["method"] == m, "rv_kms"], errors="coerce")
            counts[m].append(int(np.sum(np.isfinite(v.values))))
    x = np.arange(len(files))
    fig, ax = plt.subplots(figsize=(max(8, len(files) * 0.5), 4))
    w = 0.22
    ax.bar(x - w, counts["mask_ccf"], w, label="mask_ccf", color="C0")
    ax.bar(x, counts["template_fft"], w, label="template_fft", color="C1")
    ax.bar(x + w, counts["strong_lines"], w, label="strong_lines", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(epoch_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Chunks with finite RV")
    ax.set_xlabel("Exposure")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ttl = f"Method coverage (finite chunk RVs) Gaia {gaia_source_id}"
    if title_suffix:
        ttl = f"{ttl} — {title_suffix}"
    ax.set_title(ttl)
    fig.tight_layout()
    fig.savefig(report_dir / "method_chunk_finite_counts.png", dpi=140)
    plt.close(fig)


def generate_interpretation(
    report_dir: Path,
    diag_dir: Path | None = None,
    pipeline_summary: Path | None = None,
    legacy_summary: Path | None = None,
) -> Path:
    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    comp_path = report_dir / "exposure_comparison.csv"
    if not comp_path.exists():
        raise FileNotFoundError(f"Missing {comp_path}; run diagnose_legacy_campaign first.")

    df = pd.read_csv(comp_path)
    df["basename"] = df["basename"].map(_norm_base)

    gaia_id_int: int | None = None
    if "gaia_source_id" in df.columns and df["gaia_source_id"].notna().any():
        try:
            gaia_id_int = int(float(df["gaia_source_id"].dropna().iloc[0]))
        except (TypeError, ValueError):
            gaia_id_int = None

    mjd_map: dict[str, float] = {}
    if pipeline_summary and pipeline_summary.exists():
        mjd_map.update(parse_summary_mjd(pipeline_summary))
    if legacy_summary and legacy_summary.exists():
        for k, v in parse_summary_mjd(legacy_summary).items():
            mjd_map.setdefault(k, v)

    if "mjd" not in df.columns or df["mjd"].isna().all():
        df["mjd"] = df["basename"].map(lambda b: mjd_map.get(b, float("nan")))
    else:
        df["mjd"] = pd.to_numeric(df["mjd"], errors="coerce")

    msum_path = report_dir / "method_exposure_summary.csv"
    msum = pd.read_csv(msum_path) if msum_path.exists() else pd.DataFrame()
    if not msum.empty:
        msum["basename"] = msum["basename"].map(_norm_base)

    mpair_path = report_dir / "method_pair_stats.csv"
    mpair = (
        pd.read_csv(mpair_path)
        if mpair_path.exists() and mpair_path.stat().st_size > 2
        else pd.DataFrame()
    )

    bybin_path = report_dir / "by_teff_bin.csv"
    bybin = pd.read_csv(bybin_path) if bybin_path.exists() else pd.DataFrame()

    good = df[df["passes_legacy_filter"] == True].copy()  # noqa: E712
    both = good.dropna(subset=["legacy_rv", "new_rv"])
    d = _finite_series(both["delta_rv"]) if len(both) else np.array([])

    lines: list[str] = []
    lines.append("Legacy vs new pipeline — interpretation")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Exposures in comparison table: {len(df)}")
    lines.append(
        "Note: Re-run pipeline writes exposure_rv_kms / used_in_exposure_stack in diagnostics; "
        "diagnose uses those for parity with the sigma-clipped stack. "
        "If you only have older CSVs, new_rms may still be chunk scatter."
    )
    lines.append("")
    lines.append(f"Pass legacy quality cuts (|RV|, err, RMS): {int(good['passes_legacy_filter'].sum())}")
    lines.append(f"Subset with both legacy and new RV finite: {len(both)}")
    lines.append("")

    new_finite = int(pd.to_numeric(df["new_rv"], errors="coerce").notna().sum())
    if new_finite == 0 and len(df) > 0:
        lines.append(
            "No finite exposure-level new_rv in the comparison table. For hot stars (Teff≳6500 K), "
            "the primary path is template FFT and requires PHOENIX models under DARKHUNTER_PHOENIX_DIR "
            "(or repo phoenix_models). Without templates, mask CCF is skipped and RVs stay NaN unless "
            "strong-line fallback succeeds."
        )
        lines.append("")

    if len(d) > 0:
        lines.append("Delta = new_rv - legacy_rv (km/s), good legacy rows with finite new RV:")
        lines.append(f"  Mean (bias):     {float(np.mean(d)):+.4f}")
        std = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        lines.append(f"  Std (RMS):       {std:.4f}")
        lines.append(f"  Median:          {float(np.median(d)):+.4f}")
        lines.append(f"  MAD (robust):    {float(np.median(np.abs(d - np.median(d)))):.4f}")
        lines.append("")
    else:
        lines.append(
            "No finite new-vs-legacy pairs in the good subset. "
            "Often this means the star summary used full paths for filenames — "
            "re-run diagnose_legacy_campaign (basename matching is fixed) or check new-output summary."
        )
        lines.append("")

    nss = None
    for col in ("gaia_NSS_Solution_Type",):
        if col in df.columns and df[col].notna().any():
            nss = str(df[col].dropna().iloc[0])
            break
    if nss and str(nss) not in ("None", "nan", ""):
        lines.append(f"Gaia NSS solution type: {nss}")
        lines.append(
            "  If this is an orbital (binary) solution, single-epoch RVs should follow the orbit; "
            "a non-zero 'bias' vs legacy can be phase/sampling, not a pipeline offset."
        )
        lines.append("")

    if not msum.empty:
        lines.append("Per-method exposure means (from diagnostics):")
        for method in sorted(msum["method"].dropna().unique()):
            sub = msum[msum["method"] == method]
            rv = _finite_series(sub["rv_kms"])
            if len(rv) == 0:
                continue
            stdm = float(np.std(rv, ddof=1)) if len(rv) > 1 else 0.0
            lines.append(
                f"  {method}: N={len(rv)} epochs, "
                f"RV mean={float(np.mean(rv)):.3f} km/s, std={stdm:.3f} km/s"
            )
        lines.append("")
        piv = msum.pivot_table(index="basename", columns="method", values="rv_kms", aggfunc="first")
        if piv.shape[1] >= 2:
            row_spread = []
            for _, row in piv.iterrows():
                vals = _finite_series(pd.Series(row))
                if len(vals) >= 2:
                    row_spread.append(float(np.max(vals) - np.min(vals)))
            if row_spread:
                lines.append(
                    f"Cross-method spread per epoch (max-min RV, finite methods): "
                    f"median={float(np.median(row_spread)):.3f} km/s, "
                    f"90th pct={float(np.percentile(row_spread, 90)):.3f} km/s"
                )
                lines.append("")

    if not mpair.empty and "median_offset_kms" in mpair.columns:
        lines.append("Chunk-level method pair offsets (median offset, km/s):")
        for _, r in mpair.iterrows():
            lines.append(
                f"  {r.get('method_a','?')} vs {r.get('method_b','?')}: "
                f"median offset={float(r['median_offset_kms']):+.4f}, n_chunks={int(r.get('n',0))}"
            )
        lines.append("")
    else:
        lines.append("No method_pair_stats (need >=2 methods on >=2 chunks with overlapping chunk_keys).")
        lines.append("")

    if not bybin.empty:
        lines.append("By Teff bin (legacy-filtered rows; see by_teff_bin.csv for details):")
        lines.append(bybin.to_string(index=False))
        lines.append("")

    out_txt = report_dir / "interpretation_summary.txt"
    out_txt.write_text("\n".join(lines) + "\n")

    plot_df = df.sort_values("mjd")
    mjd = pd.to_numeric(plot_df["mjd"], errors="coerce").values
    ok_t = np.isfinite(mjd)

    if np.sum(ok_t) >= 2:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        leg_rv = pd.to_numeric(plot_df["legacy_rv"], errors="coerce").values
        leg_e = pd.to_numeric(plot_df["legacy_err"], errors="coerce").values
        new_rv = pd.to_numeric(plot_df["new_rv"], errors="coerce").values
        new_e = pd.to_numeric(plot_df["new_err"], errors="coerce").values
        m = mjd[ok_t]
        ax.errorbar(
            m,
            leg_rv[ok_t],
            yerr=np.where(np.isfinite(leg_e[ok_t]), leg_e[ok_t], 0),
            fmt="o",
            ms=4,
            capsize=2,
            label="Legacy",
            color="C0",
            alpha=0.85,
        )
        mask_new = ok_t & np.isfinite(new_rv)
        if np.any(mask_new):
            ax.errorbar(
                mjd[mask_new],
                new_rv[mask_new],
                yerr=np.where(np.isfinite(new_e[mask_new]), new_e[mask_new], 0),
                fmt="s",
                ms=4,
                capsize=2,
                label="New pipeline",
                color="C1",
                alpha=0.85,
            )
        if not msum.empty and "mask_ccf" in msum["method"].values:
            mm = msum[msum["method"] == "mask_ccf"].set_index("basename")
            mx, my = [], []
            for i, b in enumerate(plot_df["basename"]):
                if b in mm.index and np.isfinite(mjd[i]):
                    v = mm.loc[b, "rv_kms"]
                    if isinstance(v, pd.Series):
                        v = v.iloc[0]
                    if np.isfinite(float(v)):
                        mx.append(mjd[i])
                        my.append(float(v))
            if len(mx) >= 2:
                order = np.argsort(mx)
                ax.plot(np.array(mx)[order], np.array(my)[order], ":", lw=1.2, color="C2", label="New mask_ccf (mean)")
        ax.set_xlabel("MJD")
        ax.set_ylabel("RV (km/s)")
        ax.legend(loc="best", fontsize=8)
        ax.set_title("Radial velocity vs time (all epochs in comparison table)")
        fig.tight_layout()
        fig.savefig(report_dir / "rv_vs_mjd.png", dpi=140)
        plt.close(fig)

    if len(d) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(d, bins=min(20, max(5, len(d) // 2)), color="steelblue", edgecolor="white")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("new - legacy RV (km/s)")
        ax.set_ylabel("Count")
        ax.set_title("Delta RV (good legacy subset, finite new)")
        fig.tight_layout()
        fig.savefig(report_dir / "delta_rv_histogram.png", dpi=140)
        plt.close(fig)

    if not msum.empty:
        methods = [m for m in msum["method"].unique() if pd.notna(m)]
        epochs = sorted(msum["basename"].unique())
        if len(epochs) and len(methods) >= 2:
            piv = msum.pivot_table(index="basename", columns="method", values="rv_kms", aggfunc="first")
            piv = piv.reindex(epochs)
            arr = piv.values.astype(float)
            for j in range(arr.shape[1]):
                col = arr[:, j]
                bad = ~np.isfinite(col) | (np.abs(col) > 200)
                arr[bad, j] = np.nan
            fig_h = max(4, min(18, 0.35 * len(epochs) + 2))
            fig, ax = plt.subplots(figsize=(8, fig_h))
            im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-50, vmax=50)
            ax.set_yticks(np.arange(len(epochs)))
            ax.set_yticklabels([re.sub(r"Gaia_DR3_\d+_epoch_", "ep", e) for e in epochs], fontsize=7)
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels(methods, rotation=25, ha="right")
            ax.set_title("Method RV means (km/s); blank if missing or |RV|>200")
            plt.colorbar(im, ax=ax, label="RV km/s", shrink=0.6)
            fig.tight_layout()
            fig.savefig(report_dir / "methods_heatmap.png", dpi=140)
            plt.close(fig)

    if not msum.empty and "mjd" in df.columns:
        base_mjd = df[["basename", "mjd"]].drop_duplicates(subset=["basename"])
        mm = msum.merge(base_mjd, on="basename", how="left")
        if mm["mjd"].notna().any() and mm["rv_kms"].notna().any():
            gid = ""
            if "gaia_source_id" in df.columns and df["gaia_source_id"].notna().any():
                try:
                    gid = str(int(float(df["gaia_source_id"].dropna().iloc[0])))
                except (TypeError, ValueError):
                    gid = ""
            title = f"Per-method exposure RV vs MJD (Gaia {gid})" if gid else "Per-method exposure RV vs MJD"
            dh_plotting.plot_methods_rv_vs_mjd(mm, report_dir / "methods_rv_vs_mjd.png", title=title)

    if diag_dir is not None and gaia_id_int is not None and Path(diag_dir).is_dir():
        teff_u = float("nan")
        if "teff_used" in df.columns and df["teff_used"].notna().any():
            try:
                teff_u = float(pd.to_numeric(df["teff_used"], errors="coerce").dropna().iloc[0])
            except (TypeError, ValueError, IndexError):
                teff_u = float("nan")
        suff = f"teff_used={teff_u:.0f}" if np.isfinite(teff_u) else ""
        _plot_method_chunk_finite_counts(Path(diag_dir), gaia_id_int, report_dir, suff)

    teff_plot = "gaia_Teff" if "gaia_Teff" in df.columns else "teff_used"
    if (
        not msum.empty
        and teff_plot in df.columns
        and "legacy_rv" in df.columns
    ):
        leg = df[df["passes_legacy_filter"] & df["legacy_rv"].notna()][["basename", "legacy_rv", teff_plot]]
        if len(leg) >= 2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            cmap = plt.cm.tab10(np.linspace(0, 1, max(3, msum["method"].nunique())))
            for i, method in enumerate(sorted(msum["method"].dropna().unique())):
                sub = msum[msum["method"] == method]
                xs, ys = [], []
                for _, r in sub.iterrows():
                    b = r["basename"]
                    hit = leg[leg["basename"] == b]
                    if hit.empty:
                        continue
                    lv = float(hit["legacy_rv"].iloc[0])
                    tv = float(hit[teff_plot].iloc[0])
                    mv = float(r["rv_kms"])
                    if not (np.isfinite(lv) and np.isfinite(tv) and np.isfinite(mv)):
                        continue
                    xs.append(tv)
                    ys.append(mv - lv)
                if len(xs) >= 1:
                    ax.scatter(
                        xs,
                        ys,
                        alpha=0.75,
                        label=str(method),
                        color=cmap[i % len(cmap)],
                        s=28,
                    )
            ax.axhline(0, color="k", lw=0.7)
            ax.set_xlabel(teff_plot)
            ax.set_ylabel("method RV - legacy RV (km/s)")
            ax.set_title("Per-method offset vs Teff (legacy-filtered rows)")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(report_dir / "delta_method_vs_teff.png", dpi=140)
            plt.close(fig)

    return out_txt


def main() -> None:
    ap = argparse.ArgumentParser(description="Interpret diagnose_legacy outputs -> summary + plots")
    ap.add_argument("--report-dir", type=Path, default=_REPO / "validation_output" / "diagnose_legacy")
    ap.add_argument(
        "--pipeline-summary",
        type=Path,
        default=None,
        help="e.g. pipeline_rerun/Gaia_DR3_*_summary.txt for MJD",
    )
    ap.add_argument("--legacy-summary", type=Path, default=None)
    ap.add_argument(
        "--diag-dir",
        type=Path,
        default=None,
        help="Directory containing Gaia_DR3_*_diagnostics.csv (defaults next to --pipeline-summary, else report-dir)",
    )
    args = ap.parse_args()
    diag_dir = args.diag_dir
    if diag_dir is None and args.pipeline_summary is not None:
        diag_dir = args.pipeline_summary.parent
    if diag_dir is None:
        diag_dir = args.report_dir
    out = generate_interpretation(
        args.report_dir,
        diag_dir=diag_dir,
        pipeline_summary=args.pipeline_summary,
        legacy_summary=args.legacy_summary,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
