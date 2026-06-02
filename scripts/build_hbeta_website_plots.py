#!/usr/bin/env python3
"""Stack per-epoch Hβ diagnostic PNGs into website Gaia_DR3_<id>_28_hbeta.png."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

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


def _find_hbeta_png(plot_dir: Path, basename: str) -> Path | None:
    stem = Path(basename).stem
    candidates = sorted(plot_dir.glob(f"{stem}*h_beta*.png"))
    if not candidates:
        candidates = sorted(plot_dir.glob(f"*{stem}*h_beta*.png"))
    return candidates[0] if candidates else None


def build_stack(epochs: list[tuple[float, str]], plot_dir: Path, out_png: Path) -> bool:
    images: list[tuple[float, np.ndarray]] = []
    for mjd, bn in epochs:
        p = _find_hbeta_png(plot_dir, bn)
        if p is None:
            continue
        try:
            images.append((mjd, mpimg.imread(p)))
        except Exception:
            continue
    if not images:
        return False

    n = len(images)
    cmap = plt.get_cmap("viridis")
    fig_h = max(3.0, 2.35 * n)
    fig, axes = plt.subplots(n, 1, figsize=(10.0, fig_h), squeeze=False)
    axs = axes.ravel()
    mjds = np.array([im[0] for im in images], dtype=float)
    mmin, mmax = float(np.min(mjds)), float(np.max(mjds))
    for i, (mjd, img) in enumerate(images):
        ax = axs[i]
        ax.imshow(img, aspect="auto")
        frac = 0.0 if mmax <= mmin else (mjd - mmin) / (mmax - mmin)
        edge = cmap(frac)
        for spine in ax.spines.values():
            spine.set_edgecolor(edge)
            spine.set_linewidth(3.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f"MJD {mjd:.1f}", fontsize=8, rotation=0, labelpad=42, va="center")
    fig.suptitle("Hβ epochs (color = time)", fontsize=11, y=0.995)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.08, 0.02, 1.0, 0.98])
    fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Build stacked Hβ website PNGs from epoch diagnostics.")
    ap.add_argument("--summary-dir", required=True)
    ap.add_argument("--plots-root", required=True, help="Per-star plot directory root (output/Gaia_DR3_<id>)")
    ap.add_argument("--star-id", default=None)
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    plots_root = Path(args.plots_root)
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
        if build_stack(epochs, plot_dir, out_png):
            built += 1
        else:
            skipped += 1
    print(f"Built {built} Hβ stack plots (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
