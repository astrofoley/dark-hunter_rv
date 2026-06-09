"""CSV cache for per-chunk RV measurements across chunk layout campaigns."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from validation.chunk_layout import ChunkLayout, layout_to_dict, measurement_id_for_chunk

CACHE_COLUMNS = [
    "measurement_id",
    "layout_name",
    "file",
    "gaia_dr3_id",
    "chunk_key",
    "method",
    "rv_kms",
    "rv_err_kms",
    "mjd",
    "teff",
    "qc_pass",
    "pixel_edges_json",
    "measured_at",
]


def load_cache(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    df = pd.read_csv(path)
    for c in CACHE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[CACHE_COLUMNS]


def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _layout_edges_json(layout: ChunkLayout) -> str:
    return json.dumps(layout_to_dict(layout))


def ingest_diagnostics_csv(
    diag_path: Path,
    *,
    layout: ChunkLayout,
    cache: pd.DataFrame,
) -> pd.DataFrame:
    """Append mask_ccf chunk rows from one diagnostics file into cache (deduplicated)."""
    if not diag_path.is_file():
        return cache
    df = pd.read_csv(diag_path)
    if df.empty:
        return cache
    edges_json = _layout_edges_json(layout)
    merge_mode = layout.merge_orders is not None
    rows: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    for _, r in df.iterrows():
        if str(r.get("chunk_key", "")) == "all":
            continue
        if str(r.get("method", "")) != "mask_ccf":
            continue
        ck = str(r.get("chunk_key", ""))
        file_label = str(r.get("file", diag_path.stem.replace("_diagnostics", "")))
        mid = measurement_id_for_chunk(ck, layout.normalized_pixel_edges().tolist(), merge_mode)
        gid = ""
        m = __import__("re").search(r"Gaia_DR3_(\d{18,19})", file_label)
        if m:
            gid = m.group(1)
        rows.append(
            {
                "measurement_id": mid,
                "layout_name": layout.name,
                "file": file_label,
                "gaia_dr3_id": gid,
                "chunk_key": ck,
                "method": "mask_ccf",
                "rv_kms": float(r.get("rv_kms", np.nan)),
                "rv_err_kms": float(r.get("rv_err_kms", np.nan)),
                "mjd": float(r.get("mjd", np.nan)),
                "teff": float(r.get("teff", np.nan)),
                "qc_pass": bool(r.get("qc_pass", True)),
                "pixel_edges_json": edges_json,
                "measured_at": now,
            }
        )
    if not rows:
        return cache
    new_df = pd.DataFrame(rows)
    combined = pd.concat([cache, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["measurement_id", "file"], keep="last")
    return combined[CACHE_COLUMNS]


def ingest_layout_diagnostics_dir(
    diag_dir: Path,
    *,
    layout: ChunkLayout,
    cache: pd.DataFrame,
) -> pd.DataFrame:
    for p in sorted(diag_dir.glob("*_diagnostics.csv")):
        cache = ingest_diagnostics_csv(p, layout=layout, cache=cache)
    return cache


def layout_files_complete(
    cache: pd.DataFrame,
    *,
    layout_name: str,
    spectrum_files: list[Path],
    min_chunks_per_file: int = 3,
) -> set[Path]:
    """Return spectrum files that already have >= min_chunks in cache for this layout."""
    done: set[Path] = set()
    sub = cache[cache["layout_name"].astype(str) == str(layout_name)]
    if sub.empty:
        return done
    for sf in spectrum_files:
        file_label = str(sf)
        n = int((sub["file"].astype(str) == file_label).sum())
        if n == 0:
            stem = sf.stem
            n = int(sub["file"].astype(str).str.contains(stem, regex=False).sum())
        if n >= min_chunks_per_file:
            done.add(sf)
    return done


def diagnostics_glob_for_layout(campaign_dir: Path, layout_name: str) -> str:
    d = campaign_dir / "diagnostics" / layout_name
    return str(d / "Gaia_DR3_*_diagnostics.csv")
