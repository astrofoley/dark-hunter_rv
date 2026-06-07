"""Shared utilities for literature/APF overlap inventory and calibration gates (Phase A)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from glob import glob as glob_paths
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from darkhunter_rv.summary_paths import discover_summary_files, parse_object_id_from_summary
from validation.diagnose_legacy_campaign import parse_summary_file

BJD_TO_MJD_OFFSET = 2400000.5
PAIR_TYPES = ("apf_literature", "apf_apf", "literature_literature")
ABSOLUTE_GATE_KMS = 1.0
RELATIVE_GOAL_KMS = 0.1


@dataclass
class PhaseAGoals:
    pair_window_days: float = 7.0
    absolute_gate_kms: float = ABSOLUTE_GATE_KMS
    relative_goal_kms: float = RELATIVE_GOAL_KMS

    @classmethod
    def from_yaml_path(cls, path: Path) -> PhaseAGoals:
        if not path.is_file():
            return cls()
        try:
            import yaml
        except ImportError:
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            pair_window_days=float(data.get("pair_window_days", 7.0)),
            absolute_gate_kms=float(data.get("absolute_gate_kms", ABSOLUTE_GATE_KMS)),
            relative_goal_kms=float(data.get("relative_goal_kms", RELATIVE_GOAL_KMS)),
        )


def bjd_to_mjd(bjd: float) -> float:
    return float(bjd) - BJD_TO_MJD_OFFSET


def mjd_to_bjd(mjd: float) -> float:
    return float(mjd) + BJD_TO_MJD_OFFSET


def _parse_gaia_from_basename(basename: str) -> str | None:
    m = re.search(r"Gaia_DR3_(\d{18,19})", basename)
    return m.group(1) if m else None


def load_literature_epochs(master_path: Path) -> pd.DataFrame:
    df = pd.read_csv(master_path)
    df["gaia_dr3_id"] = df["gaia_dr3_id"].astype(str)
    df["bjd"] = pd.to_numeric(df["bjd"], errors="coerce")
    df["rv_kms"] = pd.to_numeric(df["rv_kms"], errors="coerce")
    df["rv_err_kms"] = pd.to_numeric(df["rv_err_kms"], errors="coerce")
    df["mjd"] = df["bjd"].apply(lambda x: bjd_to_mjd(x) if np.isfinite(x) else np.nan)
    df["epoch_id"] = (
        df["reference_key"].astype(str)
        + ":"
        + df["obs_index"].astype(str)
    )
    return df


def _diagnostics_path_for_basename(diagnostics_by_stem: dict[str, Path], basename: str) -> Path | None:
    stem = Path(basename).stem
    if stem in diagnostics_by_stem:
        return diagnostics_by_stem[stem]
    m = re.search(r"Gaia_DR3_\d+_epoch_\d+", stem)
    if m and m.group(0) in diagnostics_by_stem:
        return diagnostics_by_stem[m.group(0)]
    return None


def _exposure_from_diagnostics(path: Path) -> dict[str, float]:
    out: dict[str, float] = {
        "exposure_rv_kms": np.nan,
        "exposure_rv_err_kms": np.nan,
        "chunk_scatter_kms": np.nan,
        "teff": np.nan,
        "log10_median_mask_ccf_peak_snr": np.nan,
    }
    if not path.is_file():
        return out
    df = pd.read_csv(path)
    if df.empty:
        return out
    if "exposure_rv_kms" in df.columns and df["exposure_rv_kms"].notna().any():
        out["exposure_rv_kms"] = float(df["exposure_rv_kms"].dropna().iloc[0])
    if "exposure_rv_err_kms" in df.columns and df["exposure_rv_err_kms"].notna().any():
        out["exposure_rv_err_kms"] = float(df["exposure_rv_err_kms"].dropna().iloc[0])
    row0 = df.iloc[0]
    for col in ("chunk_scatter_kms", "teff"):
        if col in df.columns and pd.notna(row0.get(col)):
            out[col] = float(row0[col])
    mask = df[df["method"] == "mask_ccf"] if "method" in df.columns else df
    if "ccf_peak_snr" in mask.columns:
        snrs = pd.to_numeric(mask["ccf_peak_snr"], errors="coerce").dropna()
        if len(snrs):
            med = float(np.median(snrs))
            if med > 0:
                out["log10_median_mask_ccf_peak_snr"] = float(np.log10(med))
    return out


def load_apf_epochs(
    summary_dir: Path,
    *,
    diagnostics_glob: str | None = None,
    bias_correction_applied: bool = True,
    prefer_diagnostics_rv: bool = True,
) -> pd.DataFrame:
    diagnostics_by_stem: dict[str, Path] = {}
    if diagnostics_glob:
        for p in sorted(glob_paths(diagnostics_glob)):
            diagnostics_by_stem[Path(p).stem.replace("_diagnostics", "")] = Path(p)

    rows: list[dict[str, Any]] = []
    for summary_path in discover_summary_files(summary_dir):
        gaia_id = parse_object_id_from_summary(summary_path)
        if not gaia_id:
            continue
        name = summary_path.stem.replace("_summary", "").replace(f"Gaia_DR3_{gaia_id}", "").strip("_")
        for rec in parse_summary_file(summary_path):
            if not np.isfinite(rec.get("rv", np.nan)):
                continue
            basename = rec["basename"]
            diag = _diagnostics_path_for_basename(diagnostics_by_stem, basename)
            diag_fields = _exposure_from_diagnostics(diag) if diag else _exposure_from_diagnostics(Path())
            rv = float(rec["rv"])
            rv_err = float(rec["rv_err"]) if np.isfinite(rec.get("rv_err", np.nan)) else np.nan
            if prefer_diagnostics_rv and np.isfinite(diag_fields["exposure_rv_kms"]):
                rv = float(diag_fields["exposure_rv_kms"])
                if np.isfinite(diag_fields["exposure_rv_err_kms"]):
                    rv_err = float(diag_fields["exposure_rv_err_kms"])
            rows.append(
                {
                    "gaia_dr3_id": str(gaia_id),
                    "name": name or f"Gaia_DR3_{gaia_id}",
                    "basename": basename,
                    "mjd": float(rec["mjd"]),
                    "bjd": mjd_to_bjd(float(rec["mjd"])),
                    "rv_kms": rv,
                    "rv_err_kms": rv_err,
                    "rms_kms": float(rec.get("rms", np.nan)),
                    "summary_path": str(summary_path),
                    "diagnostics_path": str(diag) if diag else "",
                    "bias_correction_applied": bool(bias_correction_applied),
                    "chunk_scatter_kms": diag_fields["chunk_scatter_kms"],
                    "teff": diag_fields["teff"],
                    "log10_median_mask_ccf_peak_snr": diag_fields["log10_median_mask_ccf_peak_snr"],
                    "epoch_id": f"apf:{basename}",
                }
            )
    return pd.DataFrame(rows)


def min_cross_epoch_delta_days(left: pd.DataFrame, right: pd.DataFrame) -> float:
    if left.empty or right.empty:
        return float("nan")
    lm = left["mjd"].astype(float).values
    rm = right["mjd"].astype(float).values
    return float(np.min(np.abs(lm[:, None] - rm[None, :])))


def build_overlap_stars(literature: pd.DataFrame, apf: pd.DataFrame) -> pd.DataFrame:
    lit_ids = set(literature["gaia_dr3_id"].astype(str))
    apf_ids = set(apf["gaia_dr3_id"].astype(str))
    overlap = sorted(lit_ids & apf_ids)
    rows = []
    for gid in overlap:
        lg = literature[literature["gaia_dr3_id"] == gid]
        ag = apf[apf["gaia_dr3_id"] == gid]
        name = ""
        if "name" in lg.columns and lg["name"].notna().any():
            name = str(lg["name"].dropna().iloc[0])
        elif len(ag):
            name = str(ag["name"].iloc[0])
        rows.append(
            {
                "gaia_dr3_id": gid,
                "name": name,
                "n_literature_epochs": int(len(lg)),
                "n_apf_epochs": int(len(ag)),
                "n_literature_references": int(lg["reference_key"].nunique()) if "reference_key" in lg.columns else 0,
                "literature_mjd_min": float(lg["mjd"].min()) if len(lg) else np.nan,
                "literature_mjd_max": float(lg["mjd"].max()) if len(lg) else np.nan,
                "apf_mjd_min": float(ag["mjd"].min()) if len(ag) else np.nan,
                "apf_mjd_max": float(ag["mjd"].max()) if len(ag) else np.nan,
                "min_apf_literature_delta_days": min_cross_epoch_delta_days(ag, lg),
            }
        )
    return pd.DataFrame(rows)


def pair_counts_by_window(
    literature: pd.DataFrame,
    apf: pd.DataFrame,
    overlap_stars: pd.DataFrame,
    windows_days: Iterable[float],
) -> pd.DataFrame:
    rows = []
    for w in windows_days:
        pairs = find_pair_candidates(literature, apf, overlap_stars, window_days=float(w))
        rows.append(
            {
                "window_days": float(w),
                "n_apf_literature": int((pairs["pair_type"] == "apf_literature").sum()) if len(pairs) else 0,
                "n_apf_apf": int((pairs["pair_type"] == "apf_apf").sum()) if len(pairs) else 0,
                "n_literature_literature": int((pairs["pair_type"] == "literature_literature").sum()) if len(pairs) else 0,
            }
        )
    return pd.DataFrame(rows)


def _pair_row(
    pair_type: str,
    gaia_dr3_id: str,
    name: str,
    left: dict[str, Any],
    right: dict[str, Any],
    delta_days: float,
) -> dict[str, Any]:
    return {
        "pair_type": pair_type,
        "gaia_dr3_id": gaia_dr3_id,
        "name": name,
        "delta_days": float(delta_days),
        "left_epoch_id": left.get("epoch_id", ""),
        "right_epoch_id": right.get("epoch_id", ""),
        "left_mjd": float(left.get("mjd", np.nan)),
        "right_mjd": float(right.get("mjd", np.nan)),
        "left_rv_kms": float(left.get("rv_kms", np.nan)),
        "right_rv_kms": float(right.get("rv_kms", np.nan)),
        "left_rv_err_kms": float(left.get("rv_err_kms", np.nan)),
        "right_rv_err_kms": float(right.get("rv_err_kms", np.nan)),
        "left_reference_key": left.get("reference_key", ""),
        "right_reference_key": right.get("reference_key", ""),
        "left_instrument": left.get("instrument", "APF"),
        "right_instrument": right.get("instrument", "APF"),
        "left_bias_correction_applied": left.get("bias_correction_applied", np.nan),
        "right_bias_correction_applied": right.get("bias_correction_applied", np.nan),
    }


def find_pair_candidates(
    literature: pd.DataFrame,
    apf: pd.DataFrame,
    overlap_stars: pd.DataFrame,
    *,
    window_days: float = 7.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    star_names = {
        str(r["gaia_dr3_id"]): str(r.get("name", ""))
        for _, r in overlap_stars.iterrows()
    }
    for gid in overlap_stars["gaia_dr3_id"].astype(str):
        lg = literature[literature["gaia_dr3_id"] == gid].to_dict("records")
        ag = apf[apf["gaia_dr3_id"] == gid].to_dict("records")
        name = star_names.get(gid, "")

        for a, b in combinations(lg, 2):
            dt = abs(float(a["mjd"]) - float(b["mjd"]))
            if dt <= window_days:
                rows.append(_pair_row("literature_literature", gid, name, a, b, dt))

        for a, b in combinations(ag, 2):
            dt = abs(float(a["mjd"]) - float(b["mjd"]))
            if dt <= window_days and a.get("epoch_id") != b.get("epoch_id"):
                rows.append(_pair_row("apf_apf", gid, name, a, b, dt))

        for a in ag:
            for b in lg:
                dt = abs(float(a["mjd"]) - float(b["mjd"]))
                if dt <= window_days:
                    rows.append(_pair_row("apf_literature", gid, name, a, b, dt))

    if not rows:
        return pd.DataFrame(columns=[
            "pair_type", "gaia_dr3_id", "name", "delta_days",
            "left_epoch_id", "right_epoch_id", "left_mjd", "right_mjd",
            "left_rv_kms", "right_rv_kms", "left_rv_err_kms", "right_rv_err_kms",
            "left_reference_key", "right_reference_key",
            "left_instrument", "right_instrument",
            "left_bias_correction_applied", "right_bias_correction_applied",
        ])
    return pd.DataFrame(rows)


def enrich_pairs_with_deltas(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    out = pairs.copy()
    out["delta_rv_kms"] = out["left_rv_kms"].astype(float) - out["right_rv_kms"].astype(float)
    out["abs_delta_rv_kms"] = out["delta_rv_kms"].abs()
    le = out["left_rv_err_kms"].astype(float).fillna(0.0)
    re = out["right_rv_err_kms"].astype(float).fillna(0.0)
    out["combined_err_kms"] = np.sqrt(le**2 + re**2)
    out.loc[(le <= 0) & (re <= 0), "combined_err_kms"] = np.nan
    return out


def summarize_absolute_gate(
    pairs: pd.DataFrame,
    *,
    threshold_kms: float = ABSOLUTE_GATE_KMS,
) -> dict[str, Any]:
    sub = pairs[pairs["pair_type"] == "apf_literature"].copy()
    if sub.empty:
        return {"n_pairs": 0, "threshold_kms": threshold_kms}
    abs_dv = sub["abs_delta_rv_kms"].astype(float)
    passed = abs_dv < threshold_kms
    return {
        "n_pairs": int(len(sub)),
        "n_stars": int(sub["gaia_dr3_id"].nunique()),
        "threshold_kms": float(threshold_kms),
        "pass_rate": float(passed.mean()),
        "n_pass": int(passed.sum()),
        "n_fail": int((~passed).sum()),
        "median_abs_delta_rv_kms": float(np.median(abs_dv)),
        "p90_abs_delta_rv_kms": float(np.percentile(abs_dv, 90)),
        "max_abs_delta_rv_kms": float(np.max(abs_dv)),
        "rms_delta_rv_kms": float(np.sqrt(np.mean(sub["delta_rv_kms"].astype(float) ** 2))),
    }


def summarize_relative_gate(
    pairs: pd.DataFrame,
    *,
    goal_kms: float = RELATIVE_GOAL_KMS,
) -> dict[str, Any]:
    sub = pairs[pairs["pair_type"] == "apf_apf"].copy()
    if sub.empty:
        return {"n_pairs": 0, "goal_kms": goal_kms}
    abs_dv = sub["abs_delta_rv_kms"].astype(float)
    return {
        "n_pairs": int(len(sub)),
        "n_stars": int(sub["gaia_dr3_id"].nunique()),
        "goal_kms": float(goal_kms),
        "frac_below_goal": float(np.mean(abs_dv < goal_kms)),
        "median_abs_delta_rv_kms": float(np.median(abs_dv)),
        "p90_abs_delta_rv_kms": float(np.percentile(abs_dv, 90)),
        "max_abs_delta_rv_kms": float(np.max(abs_dv)),
        "rms_delta_rv_kms": float(np.sqrt(np.mean(sub["delta_rv_kms"].astype(float) ** 2))),
    }


def per_star_gate_table(pairs: pd.DataFrame, *, absolute_threshold_kms: float) -> pd.DataFrame:
    rows = []
    for gid, g in pairs.groupby("gaia_dr3_id"):
        name = str(g["name"].iloc[0]) if "name" in g.columns else ""
        abs_g = g[g["pair_type"] == "apf_literature"]
        rel_g = g[g["pair_type"] == "apf_apf"]
        row: dict[str, Any] = {"gaia_dr3_id": str(gid), "name": name}
        if len(abs_g):
            ad = abs_g["abs_delta_rv_kms"].astype(float)
            row.update(
                {
                    "n_apf_literature_pairs": int(len(abs_g)),
                    "abs_pass_rate": float((ad < absolute_threshold_kms).mean()),
                    "abs_median_delta_kms": float(np.median(ad)),
                    "abs_max_delta_kms": float(np.max(ad)),
                }
            )
        else:
            row.update(
                {
                    "n_apf_literature_pairs": 0,
                    "abs_pass_rate": np.nan,
                    "abs_median_delta_kms": np.nan,
                    "abs_max_delta_kms": np.nan,
                }
            )
        if len(rel_g):
            rd = rel_g["abs_delta_rv_kms"].astype(float)
            row.update(
                {
                    "n_apf_apf_pairs": int(len(rel_g)),
                    "rel_median_delta_kms": float(np.median(rd)),
                    "rel_p90_delta_kms": float(np.percentile(rd, 90)),
                    "rel_rms_delta_kms": float(
                        np.sqrt(np.mean(rel_g["delta_rv_kms"].astype(float) ** 2))
                    ),
                }
            )
        else:
            row.update(
                {
                    "n_apf_apf_pairs": 0,
                    "rel_median_delta_kms": np.nan,
                    "rel_p90_delta_kms": np.nan,
                    "rel_rms_delta_kms": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PhaseARunManifest:
    run_id: str
    created_utc: str
    master_path: str
    summary_dir: str
    diagnostics_glob: str
    bias_correction_applied: bool
    goals: dict[str, float] = field(default_factory=dict)
    inventory: dict[str, Any] = field(default_factory=dict)
    absolute_gate: dict[str, Any] = field(default_factory=dict)
    relative_gate: dict[str, Any] = field(default_factory=dict)
    output_files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "master_path": self.master_path,
            "summary_dir": self.summary_dir,
            "diagnostics_glob": self.diagnostics_glob,
            "bias_correction_applied": self.bias_correction_applied,
            "goals": self.goals,
            "inventory": self.inventory,
            "absolute_gate": self.absolute_gate,
            "relative_gate": self.relative_gate,
            "output_files": self.output_files,
        }

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def inventory_summary_counts(
    literature: pd.DataFrame,
    apf: pd.DataFrame,
    overlap_stars: pd.DataFrame,
    pairs: pd.DataFrame,
) -> dict[str, int]:
    return {
        "n_literature_stars": int(literature["gaia_dr3_id"].nunique()),
        "n_apf_stars": int(apf["gaia_dr3_id"].nunique()),
        "n_overlap_stars": int(len(overlap_stars)),
        "n_literature_epochs": int(len(literature)),
        "n_apf_epochs": int(len(apf)),
        "n_pair_candidates": int(len(pairs)),
        "n_apf_literature_pairs": int((pairs["pair_type"] == "apf_literature").sum()) if len(pairs) else 0,
        "n_apf_apf_pairs": int((pairs["pair_type"] == "apf_apf").sum()) if len(pairs) else 0,
        "n_literature_literature_pairs": int((pairs["pair_type"] == "literature_literature").sum()) if len(pairs) else 0,
    }
