#!/usr/bin/env python3
"""Build per-order/chunk bias tables from *_orders.txt files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from darkhunter_rv import chunking

# Match legacy rv_bias.py per-exposure clipping before debias aggregation.
DEFAULT_SIGMA_CLIP = 2.2
DEFAULT_MAX_CLIP_ITER = 20
DEFAULT_CLIP_TOL = 1e-3


def parse_orders_file(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 3:
                continue
            ck = p[0]
            try:
                rv = float(p[1])
                er = float(p[2])
            except ValueError:
                continue
            rows.append({"chunk_key": ck, "rv_kms": rv, "rv_err_kms": er, "file": path.name})
    return pd.DataFrame(rows)


def _sigma_clip_exposure(
    df: pd.DataFrame,
    *,
    sigma: float = DEFAULT_SIGMA_CLIP,
    max_iter: int = DEFAULT_MAX_CLIP_ITER,
    tol: float = DEFAULT_CLIP_TOL,
) -> pd.DataFrame:
    """Keep inlier mask chunks for one exposure (same logic as rv_bias.compute_bias)."""
    if df.empty:
        return df
    g = df.rename(columns={"rv_kms": "RV", "rv_err_kms": "RV_Error"}).copy()
    prev = len(g)
    for _ in range(max_iter):
        med = float(np.median(g["RV"]))
        std = float(np.std(g["RV"])) or 1e-9
        filt = g[np.abs(g["RV"] - med) <= sigma * std]
        if len(filt) == prev or abs(len(filt) - prev) / max(prev, 1) < tol:
            return filt.rename(columns={"RV": "rv_kms", "RV_Error": "rv_err_kms"})
        prev = len(filt)
        g = filt
    return g.rename(columns={"RV": "rv_kms", "RV_Error": "rv_err_kms"})


def _weighted_order_stats(vals: np.ndarray, errs: np.ndarray) -> tuple[float, float, float]:
    w = 1.0 / np.clip(errs, 1e-6, np.inf) ** 2
    bm = float(np.average(vals, weights=w))
    be = float(np.sqrt(1.0 / np.sum(w)))
    br = float(np.sqrt(np.average((vals - bm) ** 2, weights=w)))
    return bm, be, br


def bootstrap_mean(vals: np.ndarray, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    if len(vals) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        s = rng.choice(vals, size=len(vals), replace=True)
        draws.append(np.mean(s))
    return float(np.mean(draws)), float(np.std(draws))


def _orders_file_stem(path: Path) -> str:
    name = path.name
    if name.endswith("_orders.txt"):
        return name[: -len("_orders.txt")]
    return path.stem


def build_bias(
    input_dir: Path,
    n_boot: int,
    out_dir: Path,
    *,
    spectrum_stems: set[str] | None = None,
    sigma_clip: float = DEFAULT_SIGMA_CLIP,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    residual_frames = []
    chunk_frames = []

    for fp in sorted(input_dir.glob("*_orders.txt")):
        if spectrum_stems is not None and _orders_file_stem(fp) not in spectrum_stems:
            continue
        df = parse_orders_file(fp)
        if df.empty:
            continue
        kept = _sigma_clip_exposure(df, sigma=sigma_clip)
        if kept.empty:
            continue
        w = 1.0 / np.clip(kept["rv_err_kms"].values, 1e-6, np.inf) ** 2
        center = float(np.average(kept["rv_kms"].values, weights=w))
        kept = kept.copy()
        kept["resid_kms"] = kept["rv_kms"] - center
        kept["order"] = kept["chunk_key"].map(
            lambda ck: chunking.bias_order_from_chunk_key(str(ck))
        )
        kept = kept[kept["order"].notna()].copy()
        if kept.empty:
            continue
        kept["order"] = kept["order"].astype(int)
        residual_frames.append(kept[["order", "chunk_key", "resid_kms", "rv_err_kms", "file"]])
        chunk_frames.append(kept)

    if not residual_frames:
        empty = pd.DataFrame(
            columns=["chunk_key", "bias_mean_kms", "bias_rms_kms", "n", "bootstrap_mean_kms", "bootstrap_std_kms"]
        )
        empty.to_csv(out_dir / "bias_by_chunk.csv", index=False)
        with open(out_dir / "bias_statistics.txt", "w", encoding="utf-8") as fh:
            fh.write("# chunk_key bias_dv bias_err_stat bias_rms_stat\n")
        summary = {"n_files": 0, "n_rows": 0, "n_chunk_keys": 0}
        (out_dir / "bias_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    all_df = pd.concat(residual_frames, ignore_index=True)
    chunk_df = pd.concat(chunk_frames, ignore_index=True)

    chunk_stats = []
    for ck, g in chunk_df.groupby("chunk_key"):
        vals = g["resid_kms"].values
        errs = g["rv_err_kms"].values
        m, _, s = _weighted_order_stats(vals, errs)
        bm, bs = bootstrap_mean(vals, n_boot=n_boot)
        chunk_stats.append(
            {
                "chunk_key": ck,
                "bias_mean_kms": m,
                "bias_rms_kms": s,
                "n": int(len(vals)),
                "bootstrap_mean_kms": bm,
                "bootstrap_std_kms": bs,
            }
        )
    sdf = pd.DataFrame(chunk_stats).sort_values("chunk_key")
    sdf.to_csv(out_dir / "bias_by_chunk.csv", index=False)

    chunk_rows: list[tuple[str, float, float, float]] = []
    for ck, g in all_df.groupby("chunk_key"):
        vals = g["resid_kms"].values
        errs = g["rv_err_kms"].values
        bm, be, br = _weighted_order_stats(vals, errs)
        chunk_rows.append((str(ck), bm, be, br))

    with open(out_dir / "bias_statistics.txt", "w", encoding="utf-8") as fh:
        fh.write("# chunk_key bias_dv bias_err_stat bias_rms_stat\n")
        for ck, b, e, r in sorted(chunk_rows, key=lambda row: chunking.chunk_sort_key(row[0])):
            fh.write(f"{ck} {b:.8f} {e:.8f} {r:.8f}\n")

    summary = {
        "n_files": int(all_df["file"].nunique()),
        "n_rows": int(len(all_df)),
        "n_chunk_keys": int(sdf["chunk_key"].nunique()),
        "spectrum_stems_filtered": spectrum_stems is not None,
    }
    (out_dir / "bias_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build robust bias tables from order/chunk RV files")
    ap.add_argument("--input-dir", default="output", type=Path)
    ap.add_argument("--out-dir", default=Path("validation_output/bias"), type=Path)
    ap.add_argument("--bootstrap", default=200, type=int)
    ap.add_argument(
        "--spectrum-list",
        type=Path,
        default=None,
        help="Text file of spectrum paths; only matching *_orders.txt stems are used",
    )
    ap.add_argument("--sigma-clip", default=DEFAULT_SIGMA_CLIP, type=float)
    args = ap.parse_args()

    stems: set[str] | None = None
    if args.spectrum_list is not None:
        stems = set()
        for line in args.spectrum_list.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            stems.add(Path(s).stem)

    s = build_bias(
        args.input_dir,
        args.bootstrap,
        args.out_dir,
        spectrum_stems=stems,
        sigma_clip=args.sigma_clip,
    )
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
