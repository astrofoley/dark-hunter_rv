#!/usr/bin/env python3
"""Pair overlap_enriched_per_exposure.csv from continuum campaign arms. Join on basename."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config as dh_config

_METHODS = (
    ("mask", "mask_rv_kms", "mask_err_kms", "mask_valid"),
    ("template", "template_rv_kms", "template_err_kms", "template_valid"),
    ("strong_lines", "strong_lines_rv_kms", "strong_lines_err_kms", "strong_lines_valid"),
)

def _mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))

def _summary_stats(d: np.ndarray) -> dict[str, float]:
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0, "median_kms": float("nan"), "mad_kms": float("nan"),
                "mean_abs_kms": float("nan"), "p95_abs_kms": float("nan")}
    ad = np.abs(d)
    return {"n": int(d.size), "median_kms": float(np.median(d)), "mad_kms": _mad(d),
            "mean_abs_kms": float(np.mean(ad)), "p95_abs_kms": float(np.percentile(ad, 95))}

def _method_ok(row: pd.Series, rv_col: str, err_col: str, valid_col: str, max_err: float) -> bool:
    if not bool(row.get(valid_col, False)):
        return False
    rv, err = float(row.get(rv_col, float("nan"))), float(row.get(err_col, float("nan")))
    return np.isfinite(rv) and np.isfinite(err) and err > 0.0 and err <= max_err

def _load_arm(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "basename" not in df.columns:
        raise ValueError(f"arm {name!r}: missing basename in {path}")
    df = df.drop_duplicates(subset=["basename"], keep="first").copy()
    df["basename"] = df["basename"].astype(str)
    return df.set_index("basename", drop=False)

def _valid_fraction(df: pd.DataFrame, max_err: float) -> pd.DataFrame:
    n = len(df)
    rows = []
    for method, rv_col, err_col, valid_col in _METHODS:
        ok = df.apply(lambda r: _method_ok(r, rv_col, err_col, valid_col, max_err), axis=1) if n else pd.Series([], dtype=bool)
        rows.append({"method": method, "n_exposures": n, "valid_fraction": float(ok.sum()) / n if n else float("nan")})
    mt = df["mask_valid"].astype(bool) & df["template_valid"].astype(bool) if n else pd.Series([], dtype=bool)
    rows.append({"method": "mask_and_template", "n_exposures": n, "valid_fraction": float(mt.sum()) / n if n else float("nan")})
    return pd.DataFrame(rows)

def _paired_delta(base, cand, method, rv_col, err_col, valid_col, max_err):
    stems = sorted(set(base.index) & set(cand.index))
    rows, deltas = [], []
    for stem in stems:
        br, cr = base.loc[stem], cand.loc[stem]
        if not (_method_ok(br, rv_col, err_col, valid_col, max_err) and _method_ok(cr, rv_col, err_col, valid_col, max_err)):
            continue
        d = float(cr[rv_col]) - float(br[rv_col])
        deltas.append(d)
        rows.append({"basename": stem, "method": method, "rv_baseline_kms": float(br[rv_col]),
                     "rv_candidate_kms": float(cr[rv_col]), "delta_candidate_minus_baseline_kms": d})
    stats = _summary_stats(np.asarray(deltas, dtype=float))
    stats.update(method=method, n_stems_joined=len(stems))
    return pd.DataFrame(rows), stats

def _mask_minus_template_stats(df: pd.DataFrame, max_err: float) -> dict[str, float]:
    deltas = []
    for _, row in df.iterrows():
        if not (bool(row["mask_valid"]) and bool(row["template_valid"])):
            continue
        if not (_method_ok(row, "mask_rv_kms", "mask_err_kms", "mask_valid", max_err)
                and _method_ok(row, "template_rv_kms", "template_err_kms", "template_valid", max_err)):
            continue
        deltas.append(float(row["mask_rv_kms"]) - float(row["template_rv_kms"]))
    out = _summary_stats(np.asarray(deltas, dtype=float))
    out["metric"] = "mask_minus_template"
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", nargs=2, metavar=("NAME", "CSV"), required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-method-err", type=float, default=float(dh_config.COMPARISON_REPORT_MAX_RV_ERR_KMS))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    arms = {n: Path(p) for n, p in args.arm}
    if args.baseline not in arms:
        logging.error("baseline %r not in %s", args.baseline, sorted(arms))
        return 2
    tables = {n: _load_arm(p, n) for n, p in arms.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    validity = pd.concat([_valid_fraction(df, args.max_method_err).assign(arm=n) for n, df in tables.items()])
    validity.to_csv(args.out_dir / "valid_fraction_by_arm.csv", index=False)

    base = tables[args.baseline]
    delta_stats = []
    for cand_name, cand_df in tables.items():
        if cand_name == args.baseline:
            continue
        for method, rv_col, err_col, valid_col in _METHODS:
            tab, stats = _paired_delta(base, cand_df, method, rv_col, err_col, valid_col, args.max_method_err)
            stats["candidate"] = cand_name
            delta_stats.append(stats)
            tab.to_csv(args.out_dir / f"paired_delta_{cand_name}_vs_{args.baseline}_{method}.csv", index=False)

    pd.DataFrame(delta_stats).to_csv(args.out_dir / "delta_rv_summary_vs_baseline.csv", index=False)
    mt = pd.DataFrame([{**_mask_minus_template_stats(df, args.max_method_err), "arm": n} for n, df in tables.items()])
    mt.to_csv(args.out_dir / "mask_minus_template_by_arm.csv", index=False)
    logging.info("Wrote %s", args.out_dir.resolve())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
