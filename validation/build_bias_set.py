#!/usr/bin/env python3
"""Build per-order/chunk bias tables from *_orders.txt files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def bootstrap_mean(vals: np.ndarray, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    if len(vals) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        s = rng.choice(vals, size=len(vals), replace=True)
        draws.append(np.mean(s))
    return float(np.mean(draws)), float(np.std(draws))


def build_bias(input_dir: Path, n_boot: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for fp in sorted(input_dir.glob("*_orders.txt")):
        df = parse_orders_file(fp)
        if len(df):
            # per-file weighted center
            w = 1.0 / np.clip(df["rv_err_kms"].values, 1e-6, np.inf) ** 2
            center = np.average(df["rv_kms"].values, weights=w)
            df["resid_kms"] = df["rv_kms"] - center
            frames.append(df)
    if not frames:
        empty = pd.DataFrame(columns=["chunk_key","bias_mean_kms","bias_rms_kms","n","bootstrap_mean_kms","bootstrap_std_kms"])
        empty.to_csv(out_dir / "bias_by_chunk.csv", index=False)
        with open(out_dir / "bias_statistics.txt", "w", encoding="utf-8") as fh:
            fh.write("# order bias_dv bias_err_stat bias_rms_stat\n")
        summary = {"n_files": 0, "n_rows": 0, "n_chunk_keys": 0}
        (out_dir / "bias_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    all_df = pd.concat(frames, ignore_index=True)
    stats = []
    for ck, g in all_df.groupby("chunk_key"):
        vals = g["resid_kms"].values
        m = float(np.mean(vals))
        s = float(np.std(vals))
        bm, bs = bootstrap_mean(vals, n_boot=n_boot)
        stats.append({
            "chunk_key": ck,
            "bias_mean_kms": m,
            "bias_rms_kms": s,
            "n": int(len(vals)),
            "bootstrap_mean_kms": bm,
            "bootstrap_std_kms": bs,
        })
    sdf = pd.DataFrame(stats).sort_values("chunk_key")
    sdf.to_csv(out_dir / "bias_by_chunk.csv", index=False)

    # order-level file expected by pipeline reader
    order_rows = []
    tmp = sdf.copy()
    tmp["order"] = tmp["chunk_key"].astype(str).str.split("_").str[0].astype(int)
    for order, g in tmp.groupby("order"):
        order_rows.append((order, float(g["bias_mean_kms"].mean()), float(g["bootstrap_std_kms"].mean()), float(g["bias_rms_kms"].mean())))
    with open(out_dir / "bias_statistics.txt", "w", encoding="utf-8") as fh:
        fh.write("# order bias_dv bias_err_stat bias_rms_stat\n")
        for o, b, e, r in sorted(order_rows):
            fh.write(f"{o} {b:.8f} {e:.8f} {r:.8f}\n")

    summary = {
        "n_files": int(all_df["file"].nunique()),
        "n_rows": int(len(all_df)),
        "n_chunk_keys": int(sdf["chunk_key"].nunique()),
    }
    (out_dir / "bias_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build robust bias tables from order/chunk RV files")
    ap.add_argument("--input-dir", default="output", type=Path)
    ap.add_argument("--out-dir", default=Path("validation_output/bias"), type=Path)
    ap.add_argument("--bootstrap", default=200, type=int)
    args = ap.parse_args()

    s = build_bias(args.input_dir, args.bootstrap, args.out_dir)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
