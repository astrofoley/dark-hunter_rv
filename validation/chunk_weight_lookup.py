#!/usr/bin/env python3
"""
Persistent chunk weight lookup: (layout_name, chunk_key) → bias, stat err, IVW weight.

Incremental merge: new layouts add/update rows; other layouts are preserved.

Example::

  PYTHONPATH=. python3 -m validation.chunk_weight_lookup \\
    --campaign-dir validation_output/chunk_campaign \\
    --layouts subchunks_4,merge_w4 \\
    --lookup-csv validation_output/chunk_campaign/chunk_weight_lookup.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.chunk_adaptive_stack import build_multi_layout_bias_tables, load_layouts  # noqa: E402
from validation.chunk_measurement_cache import diagnostics_glob_for_layout  # noqa: E402
from validation.plot_chunk_residuals import (  # noqa: E402
    DEFAULT_CHUNK_MAX_DELTA_KMS,
    DEFAULT_CHUNK_OUTLIER_SIGMA,
    _load_chunk_rows,
    apply_spectrum_chunk_outlier_clip,
)

logger = logging.getLogger(__name__)

LOOKUP_COLUMNS = (
    "layout_name",
    "chunk_key",
    "bias_kms",
    "statistical_err_kms",
    "intrinsic_scatter_kms",
    "ivw_weight",
    "updated_at",
)


def build_lookup_from_fallback(fallback: pd.DataFrame, *, layout_name: str) -> pd.DataFrame:
    """One layout slice from multi-layout fallback bias table."""
    sub = fallback[fallback["layout_name"].astype(str) == str(layout_name)].copy()
    if sub.empty:
        return pd.DataFrame(columns=LOOKUP_COLUMNS)
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, r in sub.iterrows():
        stat = float(r.get("statistical_err_kms", np.nan))
        intrinsic = float(r.get("intrinsic_scatter_kms", 0.0) or 0.0)
        sigma = float(np.hypot(stat, intrinsic)) if np.isfinite(stat) and stat > 0 else float("nan")
        w = float(1.0 / sigma**2) if np.isfinite(sigma) and sigma > 0 else float("nan")
        rows.append(
            {
                "layout_name": str(layout_name),
                "chunk_key": str(r["chunk_key"]),
                "bias_kms": float(r.get("bias_kms", np.nan)),
                "statistical_err_kms": stat,
                "intrinsic_scatter_kms": intrinsic,
                "ivw_weight": w,
                "updated_at": now,
            }
        )
    return pd.DataFrame(rows)


def merge_lookup(existing: pd.DataFrame | None, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Upsert on (layout_name, chunk_key); preserve rows not in new_rows."""
    if new_rows.empty and (existing is None or existing.empty):
        return pd.DataFrame(columns=LOOKUP_COLUMNS)
    new_rows = new_rows[list(LOOKUP_COLUMNS)].copy()
    if existing is None or existing.empty:
        return new_rows.sort_values(["layout_name", "chunk_key"]).reset_index(drop=True)
    old = existing[list(LOOKUP_COLUMNS)].copy()
    key = ["layout_name", "chunk_key"]
    layouts_touched = set(new_rows["layout_name"].astype(str).unique())
    keep = old[~old["layout_name"].astype(str).isin(layouts_touched)]
    combined = pd.concat([keep, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=key, keep="last")
    return combined.sort_values(key).reset_index(drop=True)


def load_lookup(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=LOOKUP_COLUMNS)
    df = pd.read_csv(path)
    for col in LOOKUP_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[list(LOOKUP_COLUMNS)]


def save_lookup(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_lookup_for_campaign(
    campaign_dir: Path,
    layout_names: list[str],
    *,
    lookup_path: Path | None = None,
) -> pd.DataFrame:
    campaign_dir = Path(campaign_dir)
    layouts = load_layouts(campaign_dir)
    _per_object, fallback, _intrinsic = build_multi_layout_bias_tables(campaign_dir, layouts)
    existing = load_lookup(lookup_path) if lookup_path else pd.DataFrame(columns=LOOKUP_COLUMNS)
    merged = existing
    for name in layout_names:
        if name not in layouts:
            logger.warning("Layout %s not in campaign layouts/", name)
            continue
        piece = build_lookup_from_fallback(fallback, layout_name=name)
        merged = merge_lookup(merged, piece)
        logger.info("Lookup: %s → %d rows", name, len(piece))
    if lookup_path:
        save_lookup(merged, lookup_path)
    return merged


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build or merge chunk weight lookup CSV.")
    p.add_argument("--campaign-dir", type=Path, required=True)
    p.add_argument("--layouts", type=str, required=True, help="Comma-separated layout names")
    p.add_argument("--lookup-csv", type=Path, required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    names = [x.strip() for x in args.layouts.split(",") if x.strip()]
    build_lookup_for_campaign(args.campaign_dir, names, lookup_path=args.lookup_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
