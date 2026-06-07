#!/usr/bin/env python3
"""Build a unified literature RV reference table from compact-object follow-up papers.

Sources (default paths point at local arXiv source trees):
  - El-Badry et al. 2024 (arXiv:2405.00089v2): Table 1 sample metadata + Table 5 RVs
  - El-Badry et al. 2024 Gaia NS1 (arXiv:2402.06722v2): tab:rvs
  - El-Badry et al. 2022 Gaia BH1 (arXiv:2209.06833v3): tab:rvs
  - Simon et al. 2026 (arXiv:2603.20371): Table 2 velocity stub (subset)

Output: calibration/literature_rv_master.csv with a uniform column set; missing
fields are empty strings (not NaN) so downstream tools can treat them as absent.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO / "calibration" / "literature_rv_master.csv"

# El-Badry et al. 2024 (NS population) — Table 1 (tab:sample), parsed central ± err.
_EL_BADRY_2024_SAMPLE: list[dict[str, str]] = [
    {
        "name": "J0553-1349",
        "gaia_dr3_id": "2995961897685517312",
        "P_orb_days": "189.10",
        "P_orb_err": "0.05",
        "M_star_msun": "0.98",
        "M_star_err": "0.06",
        "M2_msun": "1.33",
        "M2_err": "0.05",
        "eccentricity": "0.3879",
        "eccentricity_err": "0.0007",
        "parallax_mas": "2.505",
        "parallax_err_mas": "0.015",
        "G_mag": "13.00",
        "N_rvs_table1": "20",
    },
    {
        "name": "J2057-4742",
        "gaia_dr3_id": "6481502062263141504",
        "P_orb_days": "230.15",
        "P_orb_err": "0.07",
        "M_star_msun": "1.048",
        "M_star_err": "0.031",
        "M2_msun": "1.31",
        "M2_err": "0.04",
        "eccentricity": "0.3095",
        "eccentricity_err": "0.0026",
        "parallax_mas": "1.745",
        "parallax_err_mas": "0.019",
        "G_mag": "13.58",
        "N_rvs_table1": "11",
    },
    {
        "name": "J1553-6846",
        "gaia_dr3_id": "5820382041374661888",
        "P_orb_days": "310.17",
        "P_orb_err": "0.11",
        "M_star_msun": "1.04",
        "M_star_err": "0.05",
        "M2_msun": "1.323",
        "M2_err": "0.032",
        "eccentricity": "0.5314",
        "eccentricity_err": "0.0021",
        "parallax_mas": "1.344",
        "parallax_err_mas": "0.012",
        "G_mag": "14.19",
        "N_rvs_table1": "16",
    },
    {
        "name": "J2102+3703",
        "gaia_dr3_id": "1871419337958702720",
        "P_orb_days": "481.04",
        "P_orb_err": "0.26",
        "M_star_msun": "1.03",
        "M_star_err": "0.03",
        "M2_msun": "1.473",
        "M2_err": "0.034",
        "eccentricity": "0.448",
        "eccentricity_err": "0.009",
        "parallax_mas": "1.521",
        "parallax_err_mas": "0.013",
        "G_mag": "13.70",
        "N_rvs_table1": "10",
    },
    {
        "name": "J0742-4749",
        "gaia_dr3_id": "5530442371304582912",
        "P_orb_days": "497.6",
        "P_orb_err": "0.4",
        "M_star_msun": "0.90",
        "M_star_err": "0.05",
        "M2_msun": "1.28",
        "M2_err": "0.04",
        "eccentricity": "0.168",
        "eccentricity_err": "0.004",
        "parallax_mas": "1.035",
        "parallax_err_mas": "0.014",
        "G_mag": "14.60",
        "N_rvs_table1": "8",
    },
    {
        "name": "J0152-2049",
        "gaia_dr3_id": "5136025521527939072",
        "P_orb_days": "536.14",
        "P_orb_err": "0.18",
        "M_star_msun": "0.782",
        "M_star_err": "0.03",
        "M2_msun": "1.291",
        "M2_err": "0.024",
        "eccentricity": "0.6615",
        "eccentricity_err": "0.0010",
        "parallax_mas": "2.453",
        "parallax_err_mas": "0.017",
        "G_mag": "12.05",
        "N_rvs_table1": "15",
    },
    {
        "name": "J0003-5604",
        "gaia_dr3_id": "4922744974687373440",
        "P_orb_days": "561.83",
        "P_orb_err": "0.29",
        "M_star_msun": "0.802",
        "M_star_err": "0.03",
        "M2_msun": "1.34",
        "M2_err": "0.04",
        "eccentricity": "0.795",
        "eccentricity_err": "0.005",
        "parallax_mas": "2.183",
        "parallax_err_mas": "0.016",
        "G_mag": "14.48",
        "N_rvs_table1": "12",
    },
    {
        "name": "J1733+5808",
        "gaia_dr3_id": "1434445448240677376",
        "P_orb_days": "570.94",
        "P_orb_err": "0.31",
        "M_star_msun": "1.16",
        "M_star_err": "0.05",
        "M2_msun": "1.362",
        "M2_err": "0.030",
        "eccentricity": "0.3093",
        "eccentricity_err": "0.0010",
        "parallax_mas": "1.452",
        "parallax_err_mas": "0.010",
        "G_mag": "13.65",
        "N_rvs_table1": "13",
    },
    {
        "name": "J1150-2203",
        "gaia_dr3_id": "3494029910469026432",
        "P_orb_days": "631.81",
        "P_orb_err": "0.22",
        "M_star_msun": "1.18",
        "M_star_err": "0.06",
        "M2_msun": "1.39",
        "M2_err": "0.04",
        "eccentricity": "0.552",
        "eccentricity_err": "0.004",
        "parallax_mas": "1.738",
        "parallax_err_mas": "0.016",
        "G_mag": "12.66",
        "N_rvs_table1": "20",
    },
    {
        "name": "J1449+6919",
        "gaia_dr3_id": "1694708646628402048",
        "P_orb_days": "632.65",
        "P_orb_err": "0.21",
        "M_star_msun": "0.91",
        "M_star_err": "0.05",
        "M2_msun": "1.258",
        "M2_err": "0.032",
        "eccentricity": "0.2668",
        "eccentricity_err": "0.0010",
        "parallax_mas": "1.812",
        "parallax_err_mas": "0.010",
        "G_mag": "13.20",
        "N_rvs_table1": "19",
    },
    {
        "name": "J0217-7541",
        "gaia_dr3_id": "4637171465304969216",
        "P_orb_days": "636.1",
        "P_orb_err": "0.7",
        "M_star_msun": "0.996",
        "M_star_err": "0.033",
        "M2_msun": "1.396",
        "M2_err": "0.033",
        "eccentricity": "0.3228",
        "eccentricity_err": "0.0033",
        "parallax_mas": "1.193",
        "parallax_err_mas": "0.012",
        "G_mag": "14.01",
        "N_rvs_table1": "10",
    },
    {
        "name": "J0639-3655",
        "gaia_dr3_id": "5580526947012630912",
        "P_orb_days": "654.6",
        "P_orb_err": "0.6",
        "M_star_msun": "1.32",
        "M_star_err": "0.06",
        "M2_msun": "1.70",
        "M2_err": "0.07",
        "eccentricity": "0.721",
        "eccentricity_err": "0.013",
        "parallax_mas": "1.130",
        "parallax_err_mas": "0.011",
        "G_mag": "13.36",
        "N_rvs_table1": "10",
    },
    {
        "name": "J1739+4502",
        "gaia_dr3_id": "1350295047363872512",
        "P_orb_days": "657.4",
        "P_orb_err": "0.6",
        "M_star_msun": "0.781",
        "M_star_err": "0.03",
        "M2_msun": "1.38",
        "M2_err": "0.04",
        "eccentricity": "0.6777",
        "eccentricity_err": "0.0018",
        "parallax_mas": "1.126",
        "parallax_err_mas": "0.013",
        "G_mag": "13.52",
        "N_rvs_table1": "18",
    },
    {
        "name": "J0036-0932",
        "gaia_dr3_id": "2426116249713980416",
        "P_orb_days": "719.8",
        "P_orb_err": "0.9",
        "M_star_msun": "0.94",
        "M_star_err": "0.04",
        "M2_msun": "1.362",
        "M2_err": "0.034",
        "eccentricity": "0.3993",
        "eccentricity_err": "0.0021",
        "parallax_mas": "1.661",
        "parallax_err_mas": "0.019",
        "G_mag": "13.02",
        "N_rvs_table1": "16",
    },
    {
        "name": "J1432-1021",
        "gaia_dr3_id": "6328149636482597888",
        "P_orb_days": "730.9",
        "P_orb_err": "0.5",
        "M_star_msun": "0.79",
        "M_star_err": "0.03",
        "M2_msun": "1.898",
        "M2_err": "0.030",
        "eccentricity": "0.1203",
        "eccentricity_err": "0.0022",
        "parallax_mas": "1.367",
        "parallax_err_mas": "0.011",
        "G_mag": "13.34",
        "N_rvs_table1": "34",
    },
    {
        "name": "J1048+6547",
        "gaia_dr3_id": "1058875159778407808",
        "P_orb_days": "827",
        "P_orb_err": "5",
        "M_star_msun": "0.99",
        "M_star_err": "0.05",
        "M2_msun": "1.52",
        "M2_err": "0.07",
        "eccentricity": "0.357",
        "eccentricity_err": "0.009",
        "parallax_mas": "0.916",
        "parallax_err_mas": "0.016",
        "G_mag": "14.52",
        "N_rvs_table1": "9",
    },
    {
        "name": "J2145+2837",
        "gaia_dr3_id": "1801110822095134848",
        "P_orb_days": "889.5",
        "P_orb_err": "0.7",
        "M_star_msun": "0.95",
        "M_star_err": "0.05",
        "M2_msun": "1.396",
        "M2_err": "0.035",
        "eccentricity": "0.5840",
        "eccentricity_err": "0.0035",
        "parallax_mas": "4.137",
        "parallax_err_mas": "0.016",
        "G_mag": "12.19",
        "N_rvs_table1": "11",
    },
    {
        "name": "J2244-2236",
        "gaia_dr3_id": "2397135910639986304",
        "P_orb_days": "938.3",
        "P_orb_err": "0.5",
        "M_star_msun": "1.002",
        "M_star_err": "0.03",
        "M2_msun": "1.443",
        "M2_err": "0.023",
        "eccentricity": "0.5666",
        "eccentricity_err": "0.0011",
        "parallax_mas": "2.079",
        "parallax_err_mas": "0.019",
        "G_mag": "13.35",
        "N_rvs_table1": "13",
    },
    {
        "name": "J0824+5254",
        "gaia_dr3_id": "1028887114002082432",
        "P_orb_days": "1026.7",
        "P_orb_err": "3.3",
        "M_star_msun": "1.102",
        "M_star_err": "0.03",
        "M2_msun": "1.604",
        "M2_err": "0.034",
        "eccentricity": "0.686",
        "eccentricity_err": "0.012",
        "parallax_mas": "1.643",
        "parallax_err_mas": "0.015",
        "G_mag": "13.59",
        "N_rvs_table1": "13",
    },
    {
        "name": "J0230+5950",
        "gaia_dr3_id": "465093354131112960",
        "P_orb_days": "1029",
        "P_orb_err": "5",
        "M_star_msun": "1.114",
        "M_star_err": "0.03",
        "M2_msun": "1.401",
        "M2_err": "0.034",
        "eccentricity": "0.753",
        "eccentricity_err": "0.011",
        "parallax_mas": "2.523",
        "parallax_err_mas": "0.015",
        "G_mag": "13.09",
        "N_rvs_table1": "15",
    },
    {
        "name": "J0634+6256",
        "gaia_dr3_id": "1007185297091149824",
        "P_orb_days": "1046.0",
        "P_orb_err": "2.1",
        "M_star_msun": "1.18",
        "M_star_err": "0.06",
        "M2_msun": "1.48",
        "M2_err": "0.09",
        "eccentricity": "0.564",
        "eccentricity_err": "0.011",
        "parallax_mas": "0.689",
        "parallax_err_mas": "0.019",
        "G_mag": "14.62",
        "N_rvs_table1": "10",
    },
]

_REF_EL_BADRY_2024 = {
    "reference_key": "ElBadry2024_NS_population",
    "reference_title": "A population of neutron star candidates in wide orbits from Gaia astrometry",
    "reference_arxiv": "2405.00089v2",
    "reference_url": "https://arxiv.org/abs/2405.00089v2",
}

_REF_EL_BADRY_2024_NS1 = {
    "reference_key": "ElBadry2024_GaiaNS1",
    "reference_title": "A 1.9 Msun neutron star candidate in a 2-year orbit",
    "reference_arxiv": "2402.06722v2",
    "reference_url": "https://arxiv.org/abs/2402.06722",
}

_REF_EL_BADRY_2022_BH1 = {
    "reference_key": "ElBadry2022_GaiaBH1",
    "reference_title": "A Sun-like star orbiting a black hole",
    "reference_arxiv": "2209.06833v3",
    "reference_url": "https://arxiv.org/abs/2209.06833",
}

_REF_SIMON_2026 = {
    "reference_key": "Simon2026_DR3_compact_followup",
    "reference_title": (
        "Radial Velocity Orbital Solutions for Candidate Black Hole and "
        "Neutron Star Binary Systems in the Gaia Data Release 3 Catalog"
    ),
    "reference_arxiv": "2603.20371v1",
    "reference_url": "https://arxiv.org/abs/2603.20371",
}

# System metadata for papers that publish RV tables without El-Badry 2024 Table 1 rows.
_EXTRA_SYSTEM: dict[str, dict[str, str]] = {
    "4373465352415301632": {
        "name": "Gaia BH1",
        "P_orb_days": "185.6",
        "P_orb_err": "",
        "M_star_msun": "0.93",
        "M_star_err": "",
        "M2_msun": "9.62",
        "M2_err": "0.18",
        "eccentricity": "0.45",
        "eccentricity_err": "",
        "parallax_mas": "",
        "parallax_err_mas": "",
        "G_mag": "13.8",
        "N_rvs_table1": "",
    },
    "6328149636482597888": {
        "name": "Gaia NS1",
        "P_orb_days": "730.7",
        "P_orb_err": "2.2",
        "M_star_msun": "0.79",
        "M_star_err": "0.01",
        "M2_msun": "1.90",
        "M2_err": "0.04",
        "eccentricity": "0.122",
        "eccentricity_err": "0.004",
        "parallax_mas": "1.36",
        "parallax_err_mas": "0.03",
        "G_mag": "13.3",
        "N_rvs_table1": "",
    },
    "5593444799901901696": {
        "name": "",
        "P_orb_days": "1038.8",
        "P_orb_err": "",
        "M_star_msun": "1.28",
        "M_star_err": "",
        "M2_msun": "1.8",
        "M2_err": "",
        "eccentricity": "",
        "eccentricity_err": "",
        "parallax_mas": "",
        "parallax_err_mas": "",
        "G_mag": "14.42",
        "N_rvs_table1": "14",
    },
}

MASTER_COLUMNS = [
    "reference_key",
    "reference_title",
    "reference_arxiv",
    "reference_url",
    "reference_table",
    "obs_index",
    "gaia_dr3_id",
    "name",
    "bjd",
    "rv_kms",
    "rv_err_kms",
    "instrument",
    "spectral_resolution_R",
    "snr_per_pixel",
    "P_orb_days",
    "P_orb_err",
    "M_star_msun",
    "M_star_err",
    "M2_msun",
    "M2_err",
    "eccentricity",
    "eccentricity_err",
    "parallax_mas",
    "parallax_err_mas",
    "G_mag",
    "N_rvs_table1",
]

META_FIELDS = [
    "P_orb_days",
    "P_orb_err",
    "M_star_msun",
    "M_star_err",
    "M2_msun",
    "M2_err",
    "eccentricity",
    "eccentricity_err",
    "parallax_mas",
    "parallax_err_mas",
    "G_mag",
    "N_rvs_table1",
]


def _blank_row() -> dict[str, str]:
    return {c: "" for c in MASTER_COLUMNS}


def _meta_for_name(name: str) -> dict[str, str]:
    for row in _EL_BADRY_2024_SAMPLE:
        if row["name"] == name:
            return {k: row.get(k, "") for k in META_FIELDS + ["gaia_dr3_id", "name"]}
    return {}


def _meta_for_gaia(gaia_id: str) -> dict[str, str]:
    for row in _EL_BADRY_2024_SAMPLE:
        if row["gaia_dr3_id"] == gaia_id:
            return {k: row.get(k, "") for k in META_FIELDS + ["gaia_dr3_id", "name"]}
    if gaia_id in _EXTRA_SYSTEM:
        out = dict(_EXTRA_SYSTEM[gaia_id])
        out["gaia_dr3_id"] = gaia_id
        return {k: out.get(k, "") for k in META_FIELDS + ["gaia_dr3_id", "name"]}
    return {"gaia_dr3_id": gaia_id}


def _parse_pm(value: str) -> tuple[str, str]:
    """Parse 'rv ± err' or bare float from LaTeX-ish strings."""
    s = value.strip().replace("$", "").replace("\\pm", "±")
    if "±" in s:
        a, b = s.split("±", 1)
        return a.strip(), b.strip()
    return s.strip(), ""


def load_elbadry_2024_rvs(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ref = dict(_REF_EL_BADRY_2024)
    ref["reference_table"] = "Table 5 (all_ns_rvs.txt)"
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        name, bjd, rv, err, inst = parts[0], parts[1], parts[2], parts[3], parts[4]
        meta = _meta_for_name(name)
        row = _blank_row()
        row.update(ref)
        row["obs_index"] = str(i)
        row["name"] = name
        row["gaia_dr3_id"] = meta.get("gaia_dr3_id", "")
        row["bjd"] = bjd
        row["rv_kms"] = rv
        row["rv_err_kms"] = err
        row["instrument"] = inst
        for k in META_FIELDS:
            row[k] = meta.get(k, "")
        rows.append(row)
    return rows


_RV_LINE_RE = re.compile(
    r"^(\d+\.\d+)\s*&\s*\$?([^$&]+?)\$?\s*&\s*([^&]+?)\s*(?:&\s*([^&]+?)\s*(?:&\s*(\d+))?)?\s*\\\\?\s*$"
)


def _parse_bh1_rv_table(tex: str) -> list[tuple[str, str, str, str, str]]:
    """Return list of (bjd, rv, rv_err, instrument, R, snr)."""
    start = tex.find(r"\label{tab:rvs}")
    if start < 0:
        raise ValueError("tab:rvs not found in Gaia BH1 tex")
    chunk = tex[:start]
    hline = chunk.rfind(r"\hline")
    if hline < 0:
        raise ValueError("BH1 RV table header not found")
    body = chunk[hline:]
    out: list[tuple[str, str, str, str, str, str]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("%") or "HJD" in line or r"\hline" in line:
            continue
        if r"\end{tabular}" in line or r"\caption" in line:
            break
        if "&" not in line:
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 3:
            continue
        bjd = parts[0]
        rv_s, rv_e = _parse_pm(parts[1])
        inst = parts[2].strip().rstrip("\\")
        res = parts[3].strip() if len(parts) > 3 else ""
        snr = parts[4].strip().rstrip("\\") if len(parts) > 4 else ""
        out.append((bjd, rv_s, rv_e, inst, res, snr))
    return out


def load_gaia_bh1_rvs(path: Path) -> list[dict[str, str]]:
    tex = path.read_text()
    gaia_id = "4373465352415301632"
    meta = _meta_for_gaia(gaia_id)
    ref = dict(_REF_EL_BADRY_2022_BH1)
    ref["reference_table"] = "tab:rvs"
    rows: list[dict[str, str]] = []
    for i, (bjd, rv, err, inst, res, snr) in enumerate(_parse_bh1_rv_table(tex), start=1):
        row = _blank_row()
        row.update(ref)
        row["obs_index"] = str(i)
        row["gaia_dr3_id"] = gaia_id
        row["name"] = meta.get("name", "Gaia BH1")
        row["bjd"] = bjd
        row["rv_kms"] = rv
        row["rv_err_kms"] = err
        row["instrument"] = inst
        row["spectral_resolution_R"] = res
        row["snr_per_pixel"] = snr
        for k in META_FIELDS:
            row[k] = meta.get(k, "")
        rows.append(row)
    return rows


def _parse_ns1_rv_table(tex: str) -> list[tuple[str, str, str, str]]:
    """Parse El-Badry 2024 Gaia NS1 tab:rvs (HJD, RV±err, instrument)."""
    m = re.search(
        r"\\begin\{tabular\}\{lll\}\s*HJD UTC.*?\\hline\s*(.*?)\\hline\s*\\hline",
        tex,
        flags=re.DOTALL,
    )
    if not m:
        raise ValueError("Gaia NS1 tab:rvs body not found")
    out: list[tuple[str, str, str, str]] = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or "&" not in line:
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) != 3:
            continue
        bjd = parts[0]
        rv_s, rv_e = _parse_pm(parts[1])
        inst = parts[2].strip().rstrip("\\")
        out.append((bjd, rv_s, rv_e, inst))
    return out


def load_gaia_ns1_rvs(path: Path) -> list[dict[str, str]]:
    tex = path.read_text()
    gaia_id = "6328149636482597888"
    meta = _meta_for_gaia(gaia_id)
    ref = dict(_REF_EL_BADRY_2024_NS1)
    ref["reference_table"] = "tab:rvs"
    rows: list[dict[str, str]] = []
    for i, (bjd, rv, err, inst) in enumerate(_parse_ns1_rv_table(tex), start=1):
        row = _blank_row()
        row.update(ref)
        row["obs_index"] = str(i)
        row["gaia_dr3_id"] = gaia_id
        row["name"] = meta.get("name", "Gaia NS1")
        row["bjd"] = bjd
        row["rv_kms"] = rv
        row["rv_err_kms"] = err
        row["instrument"] = inst
        for k in META_FIELDS:
            row[k] = meta.get(k, "")
        rows.append(row)
    return rows


_GAIA_ID_RE = re.compile(r"^\d{10,}$")


def load_simon_2026_rvs(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ref = dict(_REF_SIMON_2026)
    ref["reference_table"] = "Table 2 (velocity_table_stub.tex)"
    obs_i = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("%") or "startdata" in line or "Gaia source" in line:
            continue
        if "&" not in line:
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 5:
            continue
        gaia_id = parts[0].strip()
        if not _GAIA_ID_RE.match(gaia_id):
            continue
        obs_i += 1
        bjd = parts[1].strip()
        rv = parts[2].strip()
        err = parts[3].strip()
        inst = parts[4].strip().rstrip("\\")
        meta = _meta_for_gaia(gaia_id)
        row = _blank_row()
        row.update(ref)
        row["obs_index"] = str(obs_i)
        row["gaia_dr3_id"] = gaia_id
        row["name"] = meta.get("name", "")
        row["bjd"] = bjd
        row["rv_kms"] = rv
        row["rv_err_kms"] = err
        row["instrument"] = inst
        for k in META_FIELDS:
            row[k] = meta.get(k, "")
        rows.append(row)
    return rows


def build_master(
    *,
    elbadry_2024_rvs: Path,
    elbadry_2024_sample_tex: Path | None,
    elbadry_ns1_tex: Path,
    elbadry_bh1_tex: Path,
    simon_2026_vel: Path,
) -> list[dict[str, str]]:
    del elbadry_2024_sample_tex  # Table 1 encoded in _EL_BADRY_2024_SAMPLE
    rows: list[dict[str, str]] = []
    rows.extend(load_elbadry_2024_rvs(elbadry_2024_rvs))
    rows.extend(load_gaia_ns1_rvs(elbadry_ns1_tex))
    rows.extend(load_gaia_bh1_rvs(elbadry_bh1_tex))
    rows.extend(load_simon_2026_rvs(simon_2026_vel))
    return rows


def write_master(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--elbadry-2024-rvs",
        type=Path,
        default=home / "Downloads/arXiv-2405.00089v2/all_ns_rvs.txt",
    )
    ap.add_argument(
        "--elbadry-ns1-tex",
        type=Path,
        default=home / "Downloads/arXiv-2402.06722v2/manuscript.tex",
    )
    ap.add_argument(
        "--elbadry-bh1-tex",
        type=Path,
        default=home / "Downloads/arXiv-2209.06833v3/gaia_bh1.tex",
    )
    ap.add_argument(
        "--simon-2026-vel",
        type=Path,
        default=home / "Downloads/arXiv-2603.20371v1/velocity_table_stub.tex",
    )
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = ap.parse_args()

    rows = build_master(
        elbadry_2024_rvs=args.elbadry_2024_rvs,
        elbadry_2024_sample_tex=None,
        elbadry_ns1_tex=args.elbadry_ns1_tex,
        elbadry_bh1_tex=args.elbadry_bh1_tex,
        simon_2026_vel=args.simon_2026_vel,
    )
    write_master(rows, args.out)
    n_ref = len({r["reference_key"] for r in rows})
    n_star = len({r["gaia_dr3_id"] or r["name"] for r in rows})
    print(f"Wrote {len(rows)} observations ({n_star} systems, {n_ref} references) -> {args.out}")


if __name__ == "__main__":
    main()
