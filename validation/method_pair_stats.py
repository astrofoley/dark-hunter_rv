"""Shared helpers for mask/template/line RV consistency from diagnostics CSV rows."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {
        "n": int(len(d)),
        "median_offset_kms": float(np.median(d)),
        "mean_offset_kms": float(np.mean(d)),
        "std_offset_kms": float(np.std(d)),
    }


def compute_method_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise chunk-level RV differences between methods (same file, chunk_key)."""
    key = ["file", "chunk_key"]
    p = df.pivot_table(index=key, columns="method", values="rv_kms", aggfunc="first").reset_index()
    pairs = []
    methods = [c for c in p.columns if c not in key]
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            sub = p[[m1, m2]].dropna()
            if len(sub) < 2:
                continue
            st = pair_stats(sub[m1].values, sub[m2].values)
            st["method_a"] = m1
            st["method_b"] = m2
            pairs.append(st)
    return pd.DataFrame(pairs)
