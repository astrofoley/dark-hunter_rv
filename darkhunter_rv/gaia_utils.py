# gaia_utils.py
from __future__ import annotations

import logging
import re
from pathlib import Path
import numpy as np
from astropy.time import Time
from . import config
from .apogee_rv_query import APOGEE_TELESCOPE_PREFIX, query_apogee_rvs_for_gaia_ids
from .desi_rv_query import DESI_TELESCOPE_PREFIX, query_desi_rvs_for_gaia_ids
from .galah_rv_query import GALAH_TELESCOPE_PREFIX, query_galah_rvs_for_gaia_ids
from .ges_rv_query import GES_TELESCOPE_PREFIX, query_ges_rvs_for_gaia_ids
from .rv_frame import (
    EXTERNAL_RV_HEADER_COMMENT,
    NativeFrame,
    finalize_external_rv_record,
    mjd_from_rave_obs_id,
)
from .rv_point_filters import mjd_is_valid
from .thiele_innes_inclination import fill_inclination_in_metadata

EXTERNAL_CATALOG_PREFIXES: tuple[str, ...] = (
    "LAMOST_LRS",
    "LAMOST_MRS",
    "RAVE_DR6",
    DESI_TELESCOPE_PREFIX,
    GALAH_TELESCOPE_PREFIX,
    APOGEE_TELESCOPE_PREFIX,
    GES_TELESCOPE_PREFIX,
)

# DR3 source_id is up to 19 digits; some APF targets use 16–17 digit ids.
GAIA_DR3_ID_RE = re.compile(r"Gaia_DR3_(\d{16,19})")


def parse_gaia_id_from_path(path: str | Path) -> int | None:
    """Extract Gaia DR3 source_id from a path or filename (16–19 digit ids)."""
    m = GAIA_DR3_ID_RE.search(str(path))
    if m:
        return int(m.group(1))
    return None


def _gaia_class():
    """Lazy import: avoids astroquery's noisy archive banner when only reading summaries from disk."""
    try:
        from astroquery.gaia import Gaia

        return Gaia
    except ImportError:
        return None


def parse_gaia_id(filename):
    """
    Extracts the Gaia DR3 Source ID from a filename.
    Matches patterns like 'Gaia_DR3_1702370142434513152_epoch_1.txt'.
    """
    return parse_gaia_id_from_path(filename)


def execute_gaia_adql(query: str, name: str) -> list:
    """
    Run ADQL on the Gaia archive via astroquery only (ESA TAP sync HTTP fallback was removed:
    post-upgrade archives reject valid DR3 queries on that endpoint).
    """
    logging.info("Querying %s...", name)
    Gaia = _gaia_class()
    if Gaia is None:
        logging.warning("%s: astroquery.gaia is not installed", name)
        return []
    try:
        job = Gaia.launch_job_async(query, dump_to_file=False)
        r = job.get_results()
        return [dict(zip(r.colnames, row)) for row in r]
    except Exception as e:
        logging.warning("%s query failed: %s", name, e)
        return []


def query_gaia_data(source_id):
    if not source_id:
        return None

    cols_nss = "n.nss_solution_type, n.ra, n.dec, n.parallax, n.parallax_error, n.a_thiele_innes, n.a_thiele_innes_error, n.b_thiele_innes, n.b_thiele_innes_error, n.f_thiele_innes, n.f_thiele_innes_error, n.g_thiele_innes, n.g_thiele_innes_error, n.c_thiele_innes, n.c_thiele_innes_error, n.h_thiele_innes, n.h_thiele_innes_error, n.period, n.period_error, n.t_periastron, n.t_periastron_error, n.eccentricity, n.eccentricity_error, n.center_of_mass_velocity, n.center_of_mass_velocity_error, n.semi_amplitude_primary, n.semi_amplitude_primary_error, n.semi_amplitude_secondary, n.semi_amplitude_secondary_error, n.mass_ratio, n.mass_ratio_error, n.inclination, n.inclination_error, n.arg_periastron, n.arg_periastron_error"
    cols_source = "s.ruwe, s.teff_gspphot, s.teff_gspphot_lower, s.teff_gspphot_upper, s.logg_gspphot, s.logg_gspphot_lower, s.logg_gspphot_upper, s.mh_gspphot, s.mh_gspphot_lower, s.mh_gspphot_upper, s.phot_g_mean_mag, s.phot_bp_mean_mag, s.phot_rp_mean_mag, s.ra AS ra_source, s.dec AS dec_source, s.parallax AS plx_source, s.pmra, s.pmdec, s.radial_velocity, s.radial_velocity_error, s.source_id"

    q_main = f"""
    SELECT {cols_nss}, {cols_source}
    FROM gaiadr3.gaia_source AS s
    LEFT JOIN gaiadr3.nss_two_body_orbit AS n ON s.source_id = n.source_id
    WHERE s.source_id = {source_id}
    """

    main_rows = execute_gaia_adql(q_main.strip(), "Gaia Core")
    if not main_rows:
        return None

    base = main_rows[0]
    ra = base.get("ra") if base.get("ra") else base.get("ra_source")
    dec = base.get("dec") if base.get("dec") else base.get("dec_source")
    plx = base.get("parallax") if base.get("parallax") else base.get("plx_source")
    pmra = base.get("pmra", 0.0)
    pmdec = base.get("pmdec", 0.0)
    rv_est = base.get("radial_velocity", 0.0)

    if ra is None or dec is None:
        logging.warning("Missing coordinates; cannot query external catalogs.")
        return process_query_results(main_rows, [])

    def clean(val):
        return val if (val is not None and np.isfinite(float(val))) else 0.0

    prop_args = (
        f"{clean(ra)}, {clean(dec)}, {clean(plx)}, {clean(pmra)}, {clean(pmdec)}, "
        f"{clean(rv_est)}, 2016.0, 2000.0"
    )

    ext_rows = _query_external_rvs_combined(prop_args)
    if not ext_rows:
        ext_rows = _query_external_rvs_sequential(prop_args)

    result = process_query_results(main_rows, ext_rows)
    meta = result.get("metadata") or {}
    ra = meta.get("RA")
    dec = meta.get("Dec")
    positions = None
    try:
        if ra is not None and dec is not None and np.isfinite(float(ra)) and np.isfinite(float(dec)):
            positions = {int(source_id): (float(ra), float(dec))}
    except (TypeError, ValueError):
        positions = None
    sid = int(source_id)
    desi_rows = query_desi_rvs_for_gaia_ids([sid], positions_by_id=positions).get(sid, [])
    galah_rows = query_galah_rvs_for_gaia_ids([sid], positions_by_id=positions).get(sid, [])
    apogee_rows = query_apogee_rvs_for_gaia_ids([sid], positions_by_id=positions).get(sid, [])
    result["external_rvs"] = merge_external_rv_lists(result.get("external_rvs", []), desi_rows)
    result["external_rvs"] = merge_external_rv_lists(
        result["external_rvs"], galah_rows, replace_prefixes=(GALAH_TELESCOPE_PREFIX,)
    )
    result["external_rvs"] = merge_external_rv_lists(
        result["external_rvs"], apogee_rows, replace_prefixes=(APOGEE_TELESCOPE_PREFIX,)
    )
    return result


def _query_external_rvs_combined(prop_args: str) -> list:
    """
    Single ADQL job: LAMOST LRS + MRS + RAVE cone rows (UNION ALL).
    Falls back to :func:`_query_external_rvs_sequential` if the archive rejects the union.
    """
    cone = f"""1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS',
            COORD1(EPOCH_PROP_POS({prop_args})),
            COORD2(EPOCH_PROP_POS({prop_args})),
            0.0013888)
    )"""
    q = f"""
    SELECT 'LAMOST_LRS' AS ext_cat,
           CAST(obsdate AS VARCHAR) AS obs_str,
           z AS rv_z,
           z_err AS err_z,
           CAST('z_meas' AS VARCHAR(16)) AS flag_raw
    FROM external.lamost_dr9_lrs
    WHERE {cone}

    UNION ALL

    SELECT 'LAMOST_MRS' AS ext_cat,
           CAST(obsdate AS VARCHAR) AS obs_str,
           rv_br1 AS rv_z,
           rv_br1_err AS err_z,
           CAST(rv_br_flag AS VARCHAR(16)) AS flag_raw
    FROM external.lamost_dr9_mrs
    WHERE {cone}

    UNION ALL

    SELECT 'RAVE_DR6' AS ext_cat,
           CAST('' AS VARCHAR) AS obs_str,
           hrv_sparv AS rv_z,
           hrv_error_sparv AS err_z,
           CAST(rave_obs_id AS VARCHAR(32)) AS flag_raw
    FROM external.ravedr6
    WHERE 1=CONTAINS(
        POINT('ICRS', external.ravedr6.ra_input, external.ravedr6.dec_input),
        CIRCLE('ICRS',
            COORD1(EPOCH_PROP_POS({prop_args})),
            COORD2(EPOCH_PROP_POS({prop_args})),
            0.001388888888888889)
    )
    """
    rows = execute_gaia_adql(q.strip(), "External RVs (LAMOST+RAVE)")
    if not rows:
        return []
    return rows


def _query_external_rvs_sequential(prop_args: str) -> list:
    out: list = []
    cone_l = f"""1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS',
            COORD1(EPOCH_PROP_POS({prop_args})),
            COORD2(EPOCH_PROP_POS({prop_args})),
            0.0013888)
    )"""
    q_lrs = f"""
    SELECT 'LAMOST_LRS' AS ext_cat, CAST(obsdate AS VARCHAR) AS obs_str, z AS rv_z, z_err AS err_z,
           CAST('z_meas' AS VARCHAR(16)) AS flag_raw
    FROM external.lamost_dr9_lrs
    WHERE {cone_l}
    """
    q_mrs = f"""
    SELECT 'LAMOST_MRS' AS ext_cat, CAST(obsdate AS VARCHAR) AS obs_str, rv_br1 AS rv_z, rv_br1_err AS err_z,
           CAST(rv_br_flag AS VARCHAR(16)) AS flag_raw
    FROM external.lamost_dr9_mrs
    WHERE {cone_l}
    """
    q_rave = f"""
    SELECT 'RAVE_DR6' AS ext_cat, CAST('' AS VARCHAR) AS obs_str, hrv_sparv AS rv_z, hrv_error_sparv AS err_z,
           CAST(rave_obs_id AS VARCHAR(32)) AS flag_raw
    FROM external.ravedr6
    WHERE 1=CONTAINS(
        POINT('ICRS', external.ravedr6.ra_input, external.ravedr6.dec_input),
        CIRCLE('ICRS',
            COORD1(EPOCH_PROP_POS({prop_args})),
            COORD2(EPOCH_PROP_POS({prop_args})),
            0.001388888888888889)
    )
    ORDER BY DISTANCE(
        POINT('ICRS', external.ravedr6.ra_input, external.ravedr6.dec_input),
        POINT('ICRS', COORD1(EPOCH_PROP_POS({prop_args})), COORD2(EPOCH_PROP_POS({prop_args})))
    )
    """
    for q, label in (
        (q_lrs, "LAMOST LRS"),
        (q_mrs, "LAMOST MRS"),
        (q_rave, "RAVE DR6"),
    ):
        part = execute_gaia_adql(q.strip(), label)
        out.extend(part)
    return out


def process_query_results(main_rows, unified_external_rows):
    base = main_rows[0]

    def get_val(row, key, default=np.nan):
        val = row.get(key)
        if val is None:
            return default
        if np.ma.is_masked(val):
            return default
        try:
            return float(val) if np.isfinite(float(val)) else default
        except (TypeError, ValueError):
            return val

    sid_raw = base.get("source_id")
    try:
        source_id_meta = int(sid_raw) if sid_raw is not None else 0
    except (TypeError, ValueError):
        source_id_meta = 0

    metadata = {
        "Source_ID": source_id_meta,
        "RA": get_val(base, "ra", get_val(base, "ra_source")),
        "Dec": get_val(base, "dec", get_val(base, "dec_source")),
        "G": get_val(base, "phot_g_mean_mag"),
        "BP": get_val(base, "phot_bp_mean_mag"),
        "RP": get_val(base, "phot_rp_mean_mag"),
        "Parallax": get_val(base, "parallax", get_val(base, "plx_source")),
        "Parallax_Error": get_val(base, "parallax_error"),
        "PMRA": get_val(base, "pmra"),
        "PMDec": get_val(base, "pmdec"),
        "RUWE": get_val(base, "ruwe"),
        "Teff": get_val(base, "teff_gspphot"),
        "Teff_Lower": get_val(base, "teff_gspphot_lower"),
        "Teff_Upper": get_val(base, "teff_gspphot_upper"),
        "logg": get_val(base, "logg_gspphot"),
        "logg_Lower": get_val(base, "logg_gspphot_lower"),
        "logg_Upper": get_val(base, "logg_gspphot_upper"),
        "MH": get_val(base, "mh_gspphot"),
        "MH_Lower": get_val(base, "mh_gspphot_lower"),
        "MH_Upper": get_val(base, "mh_gspphot_upper"),
        "Radial_Velocity": get_val(base, "radial_velocity"),
        "Radial_Velocity_Error": get_val(base, "radial_velocity_error"),
        "NSS_Solution_Type": base.get("nss_solution_type", "None"),
        "Period": get_val(base, "period"),
        "Period_Error": get_val(base, "period_error"),
        "Eccentricity": get_val(base, "eccentricity"),
        "Eccentricity_Error": get_val(base, "eccentricity_error"),
        "T_Periastron": get_val(base, "t_periastron"),
        "T_Periastron_Error": get_val(base, "t_periastron_error"),
        "Mass_Ratio": get_val(base, "mass_ratio"),
        "Mass_Ratio_Error": get_val(base, "mass_ratio_error"),
        "Center_Mass_Velocity": get_val(base, "center_of_mass_velocity"),
        "Center_Mass_Velocity_Error": get_val(base, "center_of_mass_velocity_error"),
        "Semi_Amp_Primary": get_val(base, "semi_amplitude_primary"),
        "Semi_Amp_Primary_Error": get_val(base, "semi_amplitude_primary_error"),
        "Semi_Amp_Secondary": get_val(base, "semi_amplitude_secondary"),
        "Semi_Amp_Secondary_Error": get_val(base, "semi_amplitude_secondary_error"),
        "Inclination": get_val(base, "inclination"),
        "Inclination_Error": get_val(base, "inclination_error"),
        "Arg_Periastron": get_val(base, "arg_periastron"),
        "Arg_Periastron_Error": get_val(base, "arg_periastron_error"),
        "A_Thiele_Innes": get_val(base, "a_thiele_innes"),
        "A_Thiele_Innes_Error": get_val(base, "a_thiele_innes_error"),
        "B_Thiele_Innes": get_val(base, "b_thiele_innes"),
        "B_Thiele_Innes_Error": get_val(base, "b_thiele_innes_error"),
        "F_Thiele_Innes": get_val(base, "f_thiele_innes"),
        "F_Thiele_Innes_Error": get_val(base, "f_thiele_innes_error"),
        "G_Thiele_Innes": get_val(base, "g_thiele_innes"),
        "G_Thiele_Innes_Error": get_val(base, "g_thiele_innes_error"),
    }

    fill_inclination_in_metadata(metadata)

    ra_meta = metadata.get("RA")
    dec_meta = metadata.get("Dec")
    external_rvs = _external_rvs_from_unified_rows(
        unified_external_rows, ra_deg=ra_meta, dec_deg=dec_meta
    )
    return {"metadata": metadata, "external_rvs": external_rvs}


def _external_rvs_from_unified_rows(
    rows: list,
    *,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
) -> list:
    external_rvs = []
    for r in rows:
        cat = str(r.get("ext_cat", "") or "")
        obs_str = r.get("obs_str")
        z = r.get("rv_z")
        z_err = r.get("err_z")
        flag = str(r.get("flag_raw", "") or "")
        if cat == "LAMOST_LRS":
            zf = float(z) if z is not None and np.isfinite(float(z)) else np.nan
            if np.isfinite(zf):
                rv = zf * config.C_KMS
                err = (
                    float(z_err) * config.C_KMS
                    if z_err is not None and np.isfinite(float(z_err))
                    else 0.0
                )
                t = 0.0
                if obs_str and str(obs_str).strip():
                    try:
                        t = Time(str(obs_str).strip(), format="isot", scale="utc").mjd
                    except Exception:
                        t = 0.0
                rec = finalize_external_rv_record(
                    {
                        "telescope": "LAMOST_LRS",
                        "mjd": t,
                        "rv": rv,
                        "rv_err": err,
                        "flag": "z_meas",
                    },
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    native_frame=NativeFrame.HELIOCENTRIC,
                    site_key="LAMOST",
                )
                if rec is not None:
                    external_rvs.append(rec)
        elif cat == "LAMOST_MRS":
            rv = float(z) if z is not None and np.isfinite(float(z)) else np.nan
            if np.isfinite(rv):
                t = 0.0
                if obs_str and str(obs_str).strip():
                    try:
                        t = Time(str(obs_str).strip(), format="isot", scale="utc").mjd
                    except Exception:
                        t = 0.0
                err = (
                    float(z_err)
                    if z_err is not None and np.isfinite(float(z_err))
                    else 0.0
                )
                rec = finalize_external_rv_record(
                    {
                        "telescope": "LAMOST_MRS",
                        "mjd": t,
                        "rv": rv,
                        "rv_err": err,
                        "flag": flag,
                    },
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    native_frame=NativeFrame.BARYCENTRIC,
                    site_key="LAMOST",
                )
                if rec is not None:
                    external_rvs.append(rec)
        elif cat == "RAVE_DR6":
            rv = float(z) if z is not None and np.isfinite(float(z)) else np.nan
            if np.isfinite(rv):
                err = (
                    float(z_err)
                    if z_err is not None and np.isfinite(float(z_err))
                    else 0.0
                )
                t = mjd_from_rave_obs_id(flag)
                if not mjd_is_valid(t):
                    continue
                rec = finalize_external_rv_record(
                    {
                        "telescope": "RAVE_DR6",
                        "mjd": t,
                        "rv": rv,
                        "rv_err": err,
                        "flag": flag,
                    },
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    native_frame=NativeFrame.HELIOCENTRIC,
                    site_key="RAVE",
                )
                if rec is not None:
                    external_rvs.append(rec)
    return external_rvs


# Keys that must be present (and valid) in [GAIA METADATA] to skip re-querying Gaia.
REQUIRED_GAIA_SUMMARY_KEYS = ("Source_ID", "RA", "Dec")

_GAIA_METADATA_FAILURE_MARKERS = (
    "not found or query failed",
    "query failed",
)


def _parse_scalar(val: str):
    s = val.strip()
    if not s:
        return None
    low = s.lower()
    if low in ("nan", "none", "null"):
        return None
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    try:
        x = float(s)
        return x if np.isfinite(x) else None
    except ValueError:
        return s


def _metadata_value_ok(key: str, val) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        if not val.strip():
            return False
        if val.strip().lower() in _GAIA_METADATA_FAILURE_MARKERS:
            return False
        return True
    if isinstance(val, (float, np.floating)):
        if key == "Teff" and not np.isfinite(val):
            return False
        if key in ("RA", "Dec", "Parallax", "PMRA", "PMDec", "G", "BP", "RP") and not np.isfinite(val):
            return False
        return True
    if isinstance(val, (int, np.integer)):
        return True
    return bool(val)


def _source_id_matches_expected(metadata: dict, expected_source_id: int) -> bool:
    meta = normalize_parsed_star_metadata(metadata)
    sid = meta.get("Source_ID")
    if sid is None:
        return True
    try:
        si = int(sid)
    except (TypeError, ValueError):
        return False
    if si == 0:
        return True
    return si == int(expected_source_id)


def parse_gaia_metadata_from_star_summary(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "[GAIA METADATA]":
            i += 1
            break
        i += 1
    else:
        return None
    if i >= len(lines):
        return None
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return None
    first_raw = lines[i].strip()
    if first_raw.lower() in ("not found or query failed.", "not found or query failed"):
        return None
    meta = {}
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("[") and line.endswith("]"):
            break
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            meta[k] = _parse_scalar(v)
        i += 1
    return meta if meta else None


def is_desi_external_telescope(name: str) -> bool:
    return str(name or "").startswith("DESI_")


def format_external_rv_line(ext: dict) -> str:
    flag = str(ext.get("flag", "") or "").strip()
    base = (
        f"{ext['telescope']} {float(ext['mjd']):.5f} "
        f"{float(ext['rv']):.3f} {float(ext['rv_err']):.3f}"
    )
    return f"{base} {flag}".rstrip()


def merge_external_rv_lists(
    existing: list,
    new_rows: list,
    *,
    replace_prefixes: tuple[str, ...] = (DESI_TELESCOPE_PREFIX,),
) -> list:
    """Drop rows whose telescope matches a replace prefix, then append new_rows."""
    kept: list = []
    for row in existing:
        tele = str(row.get("telescope", "") or "")
        if any(tele.startswith(p) for p in replace_prefixes):
            continue
        kept.append(row)
    kept.extend(new_rows)
    kept.sort(key=lambda r: (float(r.get("mjd", 0.0)), str(r.get("telescope", ""))))
    return kept


def replace_external_rv_section_in_summary(path: Path, external_rvs: list) -> None:
    """Rewrite [EXTERNAL RV DATA] in a star summary, preserving other sections."""
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "[EXTERNAL RV DATA]"
    if marker not in text:
        raise ValueError(f"{path}: missing {marker}")
    start = text.index(marker)
    rest = text[start + len(marker) :]
    end = len(text)
    for off, ch in enumerate(rest):
        if ch == "\n" and off + 1 < len(rest) and rest[off + 1] == "[":
            end = start + len(marker) + off + 1
            break
    lines = [marker, EXTERNAL_RV_HEADER_COMMENT]
    if external_rvs:
        for ext in external_rvs:
            lines.append(format_external_rv_line(ext))
    else:
        lines.append("# No external data found.")
    new_block = "\n".join(lines) + "\n"
    path.write_text(text[:start] + new_block + text[end:], encoding="utf-8")


def parse_external_rvs_from_star_summary(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "[EXTERNAL RV DATA]":
            i += 1
            break
        i += 1
    else:
        return []
    out = []
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("[") and line.endswith("]"):
            break
        if line.startswith("#") or line.lower().startswith("# no external"):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 4:
            i += 1
            continue
        try:
            tele = parts[0]
            mjd = float(parts[1])
            rv = float(parts[2])
            rv_err = float(parts[3])
            flag = " ".join(parts[4:]) if len(parts) > 4 else ""
            out.append({"telescope": tele, "mjd": mjd, "rv": rv, "rv_err": rv_err, "flag": flag})
        except ValueError:
            pass
        i += 1
    return out


def star_summary_has_external_rv_section(path) -> bool:
    """
    True if the summary was written with an [EXTERNAL RV DATA] block (even when empty /
    ``# No external data``), so we skip re-querying LAMOST/RAVE.
    """
    path = Path(path)
    if not path.exists():
        return False
    text = path.read_text()
    if "[EXTERNAL RV DATA]" not in text:
        return False
    i = text.find("[EXTERNAL RV DATA]")
    j = text.find("\n[", i + 1)
    if j == -1:
        block = text[i:]
    else:
        block = text[i:j]
    for raw in block.splitlines()[1:]:
        s = raw.strip()
        if not s:
            continue
        return True
    return True


def normalize_parsed_star_metadata(metadata: dict) -> dict:
    m = dict(metadata)
    if "Source_ID" not in m:
        for alt in ("source_id", "SOURCE_ID"):
            if alt in m:
                m["Source_ID"] = m[alt]
                break
    if "RA" not in m:
        for alt in ("ra", "ra_source", "RA_ICRS"):
            if alt in m and m[alt] is not None:
                m["RA"] = m[alt]
                break
    if "Dec" not in m:
        for alt in ("dec", "dec_source", "Dec_ICRS"):
            if alt in m and m[alt] is not None:
                m["Dec"] = m[alt]
                break
    return m


def star_summary_metadata_complete(metadata: dict) -> bool:
    if not metadata:
        return False
    meta = normalize_parsed_star_metadata(metadata)
    for key in REQUIRED_GAIA_SUMMARY_KEYS:
        if key not in meta:
            return False
        if not _metadata_value_ok(key, meta[key]):
            return False
    return True


def _propagation_args_from_disk_metadata(meta: dict) -> str | None:
    """Build EPOCH_PROP_POS argument string from summary metadata (for external-only query)."""
    m = normalize_parsed_star_metadata(meta)

    def fget(key, alts, default=0.0):
        v = m.get(key)
        if v is not None and np.isfinite(float(v)):
            return float(v)
        for a in alts:
            if a in m and m[a] is not None:
                try:
                    vf = float(m[a])
                    if np.isfinite(vf):
                        return vf
                except (TypeError, ValueError):
                    pass
        return default

    ra = m.get("RA")
    dec = m.get("Dec")
    if ra is None or dec is None:
        return None
    try:
        ra, dec = float(ra), float(dec)
    except (TypeError, ValueError):
        return None
    plx = fget("Parallax", [], 0.0)
    pmra = fget("PMRA", [], 0.0)
    pmdec = fget("PMDec", [], 0.0)
    rv = fget("Radial_Velocity", ["radial_velocity"], 0.0)
    return f"{ra}, {dec}, {plx}, {pmra}, {pmdec}, {rv}, 2016.0, 2000.0"


def _query_lamost_rave_from_metadata(metadata: dict) -> list:
    prop = _propagation_args_from_disk_metadata(metadata)
    if prop is None:
        return []
    rows = _query_external_rvs_combined(prop)
    if not rows:
        rows = _query_external_rvs_sequential(prop)
    meta = normalize_parsed_star_metadata(metadata)
    return _external_rvs_from_unified_rows(
        rows, ra_deg=meta.get("RA"), dec_deg=meta.get("Dec")
    )


def query_external_rvs_only_from_disk_metadata(metadata: dict) -> list:
    """LAMOST + RAVE via Gaia archive TAP (cone on propagated position)."""
    return _query_lamost_rave_from_metadata(metadata)


def _positions_from_metadata(metadata: dict | None, sid: int) -> dict[int, tuple[float, float]] | None:
    if metadata is None:
        return None
    m = normalize_parsed_star_metadata(metadata)
    try:
        ra = float(m.get("RA"))
        dec = float(m.get("Dec"))
        if np.isfinite(ra) and np.isfinite(dec):
            return {sid: (ra, dec)}
    except (TypeError, ValueError):
        pass
    return None


def query_external_rvs_for_source(
    source_id: int | None,
    metadata: dict | None,
    *,
    sources: tuple[str, ...] | None = None,
) -> list:
    """External RV rows for one Gaia source (LAMOST, RAVE, DESI, GALAH, APOGEE, GES)."""
    enabled = set(sources) if sources else {
        "lamost",
        "rave",
        "desi",
        "galah",
        "apogee",
        "ges",
    }
    rows: list = []
    if metadata and ("lamost" in enabled or "rave" in enabled):
        rows.extend(_query_lamost_rave_from_metadata(metadata))
    if source_id is not None:
        try:
            sid = int(source_id)
        except (TypeError, ValueError):
            sid = 0
        if sid > 0:
            positions = _positions_from_metadata(metadata, sid)
            if "desi" in enabled:
                desi_rows = query_desi_rvs_for_gaia_ids([sid], positions_by_id=positions).get(sid, [])
                rows = merge_external_rv_lists(rows, desi_rows, replace_prefixes=(DESI_TELESCOPE_PREFIX,))
            if "galah" in enabled:
                galah_rows = query_galah_rvs_for_gaia_ids([sid], positions_by_id=positions).get(sid, [])
                rows = merge_external_rv_lists(rows, galah_rows, replace_prefixes=(GALAH_TELESCOPE_PREFIX,))
            if "apogee" in enabled:
                apogee_rows = query_apogee_rvs_for_gaia_ids([sid], positions_by_id=positions).get(sid, [])
                rows = merge_external_rv_lists(rows, apogee_rows, replace_prefixes=(APOGEE_TELESCOPE_PREFIX,))
            if "ges" in enabled:
                ges_rows = query_ges_rvs_for_gaia_ids([sid], positions_by_id=positions).get(sid, [])
                rows = merge_external_rv_lists(rows, ges_rows, replace_prefixes=(GES_TELESCOPE_PREFIX,))
    return rows


def update_summary_desi_external_rvs(summary_path: Path, *, source_id: int | None = None) -> int:
    """Query DESI and patch [EXTERNAL RV DATA] in one summary. Returns number of DESI rows written."""
    summary_path = Path(summary_path)
    meta = parse_gaia_metadata_from_star_summary(summary_path)
    if meta is None:
        return 0
    meta = normalize_parsed_star_metadata(meta)
    sid = source_id
    if sid is None:
        try:
            sid = int(meta.get("Source_ID", 0))
        except (TypeError, ValueError):
            sid = 0
    if not sid:
        return 0
    positions = None
    try:
        ra = float(meta.get("RA"))
        dec = float(meta.get("Dec"))
        if np.isfinite(ra) and np.isfinite(dec):
            positions = {int(sid): (ra, dec)}
    except (TypeError, ValueError):
        positions = None
    if positions is None:
        logging.warning(
            "Gaia_DR3_%s: no RA/Dec in summary; DESI lookup uses source_id only",
            sid,
        )
    desi_rows = query_desi_rvs_for_gaia_ids([int(sid)], positions_by_id=positions).get(int(sid), [])
    existing = parse_external_rvs_from_star_summary(summary_path)
    merged = merge_external_rv_lists(existing, desi_rows)
    replace_external_rv_section_in_summary(summary_path, merged)
    return len(desi_rows)


def load_gaia_data_from_star_summary(path, expected_source_id: int | None = None) -> dict | None:
    meta = parse_gaia_metadata_from_star_summary(path)
    if meta is None:
        return None
    meta = normalize_parsed_star_metadata(meta)
    if not star_summary_metadata_complete(meta):
        return None
    if expected_source_id is not None and not _source_id_matches_expected(meta, expected_source_id):
        return None
    ext = parse_external_rvs_from_star_summary(path)
    return {"metadata": meta, "external_rvs": ext}


def star_summary_metadata_needs_photometry(metadata: dict) -> bool:
    meta = normalize_parsed_star_metadata(metadata or {})
    for key in ("G", "BP", "RP"):
        val = meta.get(key)
        if val is None:
            return True
        try:
            if not np.isfinite(float(val)):
                return True
        except (TypeError, ValueError):
            return True
    return False


def resolve_gaia_data(source_id: int, summary_path, force_query: bool):
    summary_path = Path(summary_path)
    if force_query:
        return query_gaia_data(source_id)
    if not summary_path.exists():
        return query_gaia_data(source_id)
    loaded = load_gaia_data_from_star_summary(summary_path, expected_source_id=source_id)
    if loaded is None:
        return query_gaia_data(source_id)
    # Disk-only when the star summary parses: do not query Core or LAMOST/RAVE TAP here.
    # Use --force-gaia to refresh from the network. External RVs stay whatever is in the file
    # (possibly empty if [EXTERNAL RV DATA] is missing).
    return loaded
