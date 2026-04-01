"""Aggregate per-order RV files and write bias_statistics.txt for debiassing."""

import argparse
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_suborder(filepath: str) -> bool:
    if not filepath.endswith("_orders.txt") or "epoch" not in filepath:
        return False
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") and "Suborder" in line:
                return True
    return False


def read_data(directory: str):
    first_suborder = False
    for fn in os.listdir(directory):
        fp = os.path.join(directory, fn)
        if os.path.isfile(fp) and fn.endswith("_orders.txt") and "epoch" in fn:
            if check_suborder(fp):
                first_suborder = True
                break

    rows = []
    for fn in os.listdir(directory):
        if not fn.endswith("_orders.txt") or "epoch" not in fn:
            continue
        fp = os.path.join(directory, fn)
        sub_here = check_suborder(fp)
        star_name = "_".join(fn.split("_")[:3])
        try:
            epoch = int(fn.split("epoch_")[1].split("_")[0])
        except (IndexError, ValueError):
            logger.warning("skip %s (epoch parse)", fn)
            continue

        if not sub_here and first_suborder:
            continue
        if sub_here or first_suborder:
            df = pd.read_csv(fp, sep=r"\s+", comment="#", names=["Order", "Suborder", "RV", "RV_Error"])
            df["Suborder"] = pd.to_numeric(df["Suborder"], errors="coerce").fillna(0).astype(int)
        else:
            df = pd.read_csv(fp, sep=r"\s+", comment="#", names=["Order", "RV", "RV_Error"])
            df["Suborder"] = 0

        df["Star"] = star_name
        df["Epoch"] = epoch
        rows.append(df)

    if not rows:
        return pd.DataFrame(), False
    return pd.concat(rows, ignore_index=True), first_suborder


def compute_bias(df: pd.DataFrame, sigma: float = 2.2, max_iter: int = 20, tol: float = 1e-3):
    if df.empty:
        return df
    out = []
    for (_, _), group in df.groupby(["Star", "Epoch"]):
        g = group.copy()
        prev = len(g)
        for _ in range(max_iter):
            med = np.median(g["RV"])
            std = np.std(g["RV"]) or 1e-9
            filt = g[np.abs(g["RV"] - med) <= sigma * std]
            if len(filt) == prev or abs(len(filt) - prev) / prev < tol:
                g = filt
                break
            prev = len(filt)
            g = filt
        if len(g) == 0:
            continue
        w = 1.0 / (g["RV_Error"] ** 2 + 1e-18)
        wmean = np.sum(g["RV"] * w) / np.sum(w)
        g = g.copy()
        g["Bias"] = g["RV"] - wmean
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=df.columns)


def compute_statistics(df: pd.DataFrame, groupby_cols: list):
    def _agg(g):
        w = 1.0 / (g["RV_Error"] ** 2 + 1e-18)
        bm = np.average(g["Bias"], weights=w)
        be = np.sqrt(1.0 / np.sum(w))
        br = np.sum((g["Bias"] - bm) ** 2 * w) / np.sum(w)
        return pd.Series({"Bias_Mean": bm, "Bias_Error": be, "Bias_RMS": np.sqrt(max(br, 0.0))})

    return df.groupby(groupby_cols).apply(_agg).reset_index()


def write_bias_file(stats: pd.DataFrame, outpath: str) -> None:
    stats = stats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    with open(outpath, "w", encoding="utf-8") as fh:
        fh.write("# order bias_dv bias_err_stat bias_rms_stat\n")
        for _, row in stats.iterrows():
            o = int(row["Order"])
            fh.write(
                f"{o} {row['Bias_Mean']:.8f} {row['Bias_Error']:.8f} {row['Bias_RMS']:.8f}\n"
            )
    logger.info("Wrote %s", outpath)


def main():
    parser = argparse.ArgumentParser(description="Build bias_statistics.txt from order RV files.")
    parser.add_argument("--input-dir", default="output")
    parser.add_argument("--output", default="bias_statistics.txt")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    df, sub = read_data(args.input_dir)
    if df.empty:
        logger.error("No order files in %s", args.input_dir)
        return
    filt = compute_bias(df)
    if filt.empty:
        logger.error("No bias rows after filtering")
        return
    stats = compute_statistics(filt, ["Order"])
    write_bias_file(stats, args.output)


if __name__ == "__main__":
    main()
