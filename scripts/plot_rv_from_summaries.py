#!/usr/bin/env python3
"""Generate APF RV-vs-MJD plots directly from summary files (no pipeline rerun)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_gaia_id(path: Path) -> str | None:
    m = re.search(r"Gaia_DR3_(\d{8,19})", f"{path.parent.name}/{path.stem}")
    if m:
        return m.group(1)
    m2 = re.match(r"(\d{8,19})_summary$", path.stem)
    return m2.group(1) if m2 else None


def parse_points(summary_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("[PIPELINE RESULTS]", 1)[-1].splitlines() if "[PIPELINE RESULTS]" in text else text.splitlines()
    t, y, e = [], [], []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#") or (s.startswith("[") and s.endswith("]")):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        if len(parts) >= 6 and parts[-1] in ("True", "False"):
            parts = parts[:-1]
        if len(parts) < 5:
            continue
        try:
            mjd = float(parts[1])
            rv = float(parts[2])
            err = float(parts[3])
            rms = float(parts[4])
        except ValueError:
            continue
        if not np.isfinite(mjd) or not np.isfinite(rv):
            continue
        t.append(mjd)
        y.append(rv)
        if np.isfinite(rms) and rms > 0:
            e.append(rms)
        elif np.isfinite(err) and err > 0:
            e.append(err)
        else:
            e.append(0.25)
    return np.array(t, dtype=float), np.array(y, dtype=float), np.array(e, dtype=float)


def build_plot(summary_path: Path, out_png: Path, fit_json: Path | None = None) -> bool:
    t, y, e = parse_points(summary_path)
    if t.size < 2:
        return False
    order = np.argsort(t)
    t, y, e = t[order], y[order], e[order]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.errorbar(t, y, yerr=e, fmt="o", ms=4.5, capsize=2, lw=1, label="APF epochs")
    ax.set_xlabel("MJD")
    ax.set_ylabel("RV (km/s)")
    gaia_id = parse_gaia_id(summary_path) or summary_path.stem.replace("_summary", "")
    ax.set_title(f"Gaia DR3 {gaia_id} RV vs MJD")
    ax.grid(alpha=0.25)

    # Optional model overlay from keplerian fit JSON.
    if fit_json is not None and fit_json.is_file():
        try:
            rep = json.loads(fit_json.read_text(encoding="utf-8"))
            p = rep.get("params_raw")
            t_ref = float(rep.get("t_ref_mjd"))
            if isinstance(p, list) and len(p) == 6:
                arr = np.array(p, dtype=float)
                td = np.linspace(float(np.min(t) - 0.02 * (np.ptp(t) + 1)), float(np.max(t) + 0.02 * (np.ptp(t) + 1)), 1200)
                logp, k, h, kk, m0, gamma = arr
                period = np.exp(logp)
                ecc = np.clip(np.hypot(h, kk), 1e-8, 0.95)
                omega = float(np.arctan2(kk, h))
                n = 2.0 * np.pi / period
                mean_anom = n * (td - t_ref) + m0
                E = np.array(mean_anom, dtype=float)
                for _ in range(30):
                    f = E - ecc * np.sin(E) - mean_anom
                    fp = 1.0 - ecc * np.cos(E)
                    E -= f / np.clip(fp, 1e-10, None)
                cosf = (np.cos(E) - ecc) / (1.0 - ecc * np.cos(E))
                sinf = (np.sqrt(1.0 - ecc * ecc) * np.sin(E)) / (1.0 - ecc * np.cos(E))
                true_anom = np.arctan2(sinf, cosf)
                model = gamma + k * (np.cos(true_anom + omega) + ecc * np.cos(omega))
                ax.plot(td, model, "-", lw=1.6, alpha=0.9, label="Keplerian fit")
                ax.legend(loc="best", fontsize=9)
        except Exception:
            pass

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Build RV plots from Gaia summary files only.")
    ap.add_argument("--summary-dir", required=True, help="Directory containing Gaia_DR3_*_summary.txt")
    ap.add_argument("--plots-root", required=True, help="Output root for per-star plot subdirs")
    ap.add_argument("--reports-dir", default=None, help="Optional dir with <id>_keplerian_fit.json to overlay model")
    ap.add_argument("--star-id", default=None, help="Optional single Gaia source id")
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir)
    plots_root = Path(args.plots_root)
    reports_dir = Path(args.reports_dir) if args.reports_dir else None

    pattern = f"Gaia_DR3_{args.star_id}_summary.txt" if args.star_id else "Gaia_DR3_*_summary.txt"
    files = sorted(summary_dir.glob(pattern))
    if not files:
        print("No summary files found.")
        return 2

    built = 0
    skipped = 0
    for summ in files:
        sid = parse_gaia_id(summ)
        if not sid:
            skipped += 1
            continue
        out_png = plots_root / f"Gaia_DR3_{sid}" / f"Gaia_DR3_{sid}_rv_plot.png"
        fit_json = (reports_dir / f"{sid}_keplerian_fit.json") if reports_dir else None
        ok = build_plot(summ, out_png, fit_json=fit_json)
        if ok:
            built += 1
        else:
            skipped += 1
    print(f"Built {built} summary-based RV plots (skipped {skipped}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
