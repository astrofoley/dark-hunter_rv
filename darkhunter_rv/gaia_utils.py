# gaia_utils.py
import logging
import re
import numpy as np
import requests
from astropy.time import Time
from . import config

# Try to import astroquery, fallback to requests if necessary
try:
    from astroquery.gaia import Gaia
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False

def parse_gaia_id(filename):
    """
    Extracts the Gaia DR3 Source ID from a filename.
    Matches patterns like 'Gaia_DR3_1702370142434513152_epoch_1.txt'.
    """
    match = re.search(r"Gaia_DR3_(\d{18,19})", str(filename))
    if match:
        return int(match.group(1))
    return None

def execute_query_safe(name, query):
    """
    Executes a query defensively. If it fails, returns empty list
    instead of crashing the whole pipeline.
    """
    logging.info(f"Querying {name}...")
    
    # 1. Try Astroquery
    if ASTROQUERY_AVAILABLE:
        try:
            job = Gaia.launch_job_async(query)
            r = job.get_results()
            return [dict(zip(r.colnames, row)) for row in r]
        except Exception:
            pass

    # 2. Fallback to Direct TAP
    try:
        url = "https://gea.esac.esa.int/tap-server/tap/sync"
        params = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": query}
        resp = requests.post(url, data=params)
        
        if resp.status_code != 200:
            logging.warning(f"{name} request failed (Code {resp.status_code}): {resp.text}")
            return []
            
        data = resp.json()
        if not data.get('data'): return []
        cols = [c['name'] for c in data['metadata']]
        return [dict(zip(cols, row)) for row in data['data']]
    except Exception as e:
        logging.error(f"{name} query fatal error: {e}")
        return []

def query_gaia_data(source_id):
    if not source_id: return None

    # ---------------------------------------------------------
    # 1. Core Gaia Query (Source + NSS)
    # ---------------------------------------------------------
    cols_nss = "n.nss_solution_type, n.ra, n.dec, n.parallax, n.parallax_error, n.a_thiele_innes, n.a_thiele_innes_error, n.b_thiele_innes, n.b_thiele_innes_error, n.f_thiele_innes, n.f_thiele_innes_error, n.g_thiele_innes, n.g_thiele_innes_error, n.c_thiele_innes, n.c_thiele_innes_error, n.h_thiele_innes, n.h_thiele_innes_error, n.period, n.period_error, n.t_periastron, n.t_periastron_error, n.eccentricity, n.eccentricity_error, n.center_of_mass_velocity, n.center_of_mass_velocity_error, n.semi_amplitude_primary, n.semi_amplitude_primary_error, n.semi_amplitude_secondary, n.semi_amplitude_secondary_error, n.mass_ratio, n.mass_ratio_error, n.inclination, n.inclination_error, n.arg_periastron, n.arg_periastron_error"
    cols_source = "s.ruwe, s.teff_gspphot, s.teff_gspphot_lower, s.teff_gspphot_upper, s.logg_gspphot, s.logg_gspphot_lower, s.logg_gspphot_upper, s.mh_gspphot, s.mh_gspphot_lower, s.mh_gspphot_upper, s.ra as ra_source, s.dec as dec_source, s.parallax as plx_source, s.pmra, s.pmdec, s.radial_velocity, s.radial_velocity_error"
    
    q_main = f"""
    SELECT {cols_nss}, {cols_source}
    FROM gaiadr3.gaia_source AS s
    LEFT JOIN gaiadr3.nss_two_body_orbit AS n ON s.source_id = n.source_id
    WHERE s.source_id = {source_id}
    """
    
    main_rows = execute_query_safe("Gaia Core", q_main)
    if not main_rows:
        return None

    # Extract Astrometry for Propagation
    base = main_rows[0]
    
    # Use Source values if NSS values are missing/null
    ra = base.get('ra') if base.get('ra') else base.get('ra_source')
    dec = base.get('dec') if base.get('dec') else base.get('dec_source')
    plx = base.get('parallax') if base.get('parallax') else base.get('plx_source')
    pmra = base.get('pmra', 0.0)
    pmdec = base.get('pmdec', 0.0)
    rv_est = base.get('radial_velocity', 0.0)

    # Validate for propagation
    if ra is None or dec is None:
        logging.warning("Missing coordinates; cannot query external catalogs.")
        return process_query_results(main_rows, [], [], [])
        
    # Clean NaNs
    def clean(val): return val if (val is not None and np.isfinite(val)) else 0.0
    
    # Arguments for EPOCH_PROP_POS: ra, dec, plx, pmra, pmdec, rv, t_obs, t_target
    prop_args = f"{clean(ra)}, {clean(dec)}, {clean(plx)}, {clean(pmra)}, {clean(pmdec)}, {clean(rv_est)}, 2016.0, 2000.0"

    # ---------------------------------------------------------
    # 2. External Queries
    # ---------------------------------------------------------
    
    # LAMOST LRS (J2000)
    q_lrs = f"""
    SELECT obsdate, z, z_err
    FROM external.lamost_dr9_lrs
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS',
            COORD1(EPOCH_PROP_POS({prop_args})),
            COORD2(EPOCH_PROP_POS({prop_args})),
            0.0013888)
    )
    """
    
    # LAMOST MRS (J2000)
    q_mrs = f"""
    SELECT obsdate, rv_br1, rv_br1_err, rv_br_flag
    FROM external.lamost_dr9_mrs
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS',
            COORD1(EPOCH_PROP_POS({prop_args})),
            COORD2(EPOCH_PROP_POS({prop_args})),
            0.0013888)
    )
    """

    # RAVE DR6 (Exact User Logic)
    # Using ra_input/dec_input, EPOCH_PROP_POS, and ORDER BY separation
    q_rave = f"""
    SELECT
        ravedr6.rave_obs_id,
        ravedr6.hrv_sparv,
        ravedr6.hrv_error_sparv,
        DISTANCE(
            POINT('ICRS', external.ravedr6.ra_input, external.ravedr6.dec_input),
            POINT('ICRS',
                COORD1(EPOCH_PROP_POS({prop_args})),
                COORD2(EPOCH_PROP_POS({prop_args}))
            )
        ) AS dist_deg
    FROM external.ravedr6
    WHERE 1=CONTAINS(
        POINT('ICRS', external.ravedr6.ra_input, external.ravedr6.dec_input),
        CIRCLE('ICRS',
            COORD1(EPOCH_PROP_POS({prop_args})),
            COORD2(EPOCH_PROP_POS({prop_args})),
            0.001388888888888889)
    )
    ORDER BY dist_deg ASC
    """

    lrs_rows = execute_query_safe("LAMOST LRS", q_lrs)
    mrs_rows = execute_query_safe("LAMOST MRS", q_mrs)
    rave_rows = execute_query_safe("RAVE DR6", q_rave)

    return process_query_results(main_rows, lrs_rows, mrs_rows, rave_rows)


def process_query_results(main_rows, lrs_rows, mrs_rows, rave_rows):
    base = main_rows[0]
    
    def get_val(row, key, default=np.nan):
        val = row.get(key)
        if val is None: return default
        if np.ma.is_masked(val): return default
        try: return float(val) if np.isfinite(float(val)) else default
        except: return val

    # Collect ALL requested columns + Errors
    metadata = {
        "Source_ID": get_val(base, 'source_id', 0),
        "RA": get_val(base, 'ra', get_val(base, 'ra_source')),
        "Dec": get_val(base, 'dec', get_val(base, 'dec_source')),
        "Parallax": get_val(base, 'parallax', get_val(base, 'plx_source')),
        "Parallax_Error": get_val(base, 'parallax_error'),
        "PMRA": get_val(base, 'pmra'),
        "PMDec": get_val(base, 'pmdec'),
        "RUWE": get_val(base, 'ruwe'),
        
        "Teff": get_val(base, 'teff_gspphot'),
        "Teff_Lower": get_val(base, 'teff_gspphot_lower'),
        "Teff_Upper": get_val(base, 'teff_gspphot_upper'),
        
        "logg": get_val(base, 'logg_gspphot'),
        "logg_Lower": get_val(base, 'logg_gspphot_lower'),
        "logg_Upper": get_val(base, 'logg_gspphot_upper'),
        
        "MH": get_val(base, 'mh_gspphot'),
        "MH_Lower": get_val(base, 'mh_gspphot_lower'),
        "MH_Upper": get_val(base, 'mh_gspphot_upper'),
        
        "Radial_Velocity": get_val(base, 'radial_velocity'),
        "Radial_Velocity_Error": get_val(base, 'radial_velocity_error'),
        
        # NSS Parameters
        "NSS_Solution_Type": base.get('nss_solution_type', "None"),
        "Period": get_val(base, 'period'),
        "Period_Error": get_val(base, 'period_error'),
        "Eccentricity": get_val(base, 'eccentricity'),
        "Eccentricity_Error": get_val(base, 'eccentricity_error'),
        "T_Periastron": get_val(base, 't_periastron'),
        "T_Periastron_Error": get_val(base, 't_periastron_error'),
        "Mass_Ratio": get_val(base, 'mass_ratio'),
        "Mass_Ratio_Error": get_val(base, 'mass_ratio_error'),
        "Center_Mass_Velocity": get_val(base, 'center_of_mass_velocity'),
        "Center_Mass_Velocity_Error": get_val(base, 'center_of_mass_velocity_error'),
        "Semi_Amp_Primary": get_val(base, 'semi_amplitude_primary'),
        "Semi_Amp_Primary_Error": get_val(base, 'semi_amplitude_primary_error'),
        "Semi_Amp_Secondary": get_val(base, 'semi_amplitude_secondary'),
        "Semi_Amp_Secondary_Error": get_val(base, 'semi_amplitude_secondary_error'),
        "Inclination": get_val(base, 'inclination'),
        "Inclination_Error": get_val(base, 'inclination_error'),
        "Arg_Periastron": get_val(base, 'arg_periastron'),
        "Arg_Periastron_Error": get_val(base, 'arg_periastron_error'),
        
        # Thiele-Innes
        "A_Thiele_Innes": get_val(base, 'a_thiele_innes'),
        "A_Thiele_Innes_Error": get_val(base, 'a_thiele_innes_error'),
        "B_Thiele_Innes": get_val(base, 'b_thiele_innes'),
        "B_Thiele_Innes_Error": get_val(base, 'b_thiele_innes_error'),
        "F_Thiele_Innes": get_val(base, 'f_thiele_innes'),
        "F_Thiele_Innes_Error": get_val(base, 'f_thiele_innes_error'),
        "G_Thiele_Innes": get_val(base, 'g_thiele_innes'),
        "G_Thiele_Innes_Error": get_val(base, 'g_thiele_innes_error'),
    }
    
    external_rvs = []
    
    # Process LAMOST LRS
    for r in lrs_rows:
        z = get_val(r, 'z')
        if np.isfinite(z):
            rv = z * config.C_KMS
            err = get_val(r, 'z_err', 0) * config.C_KMS
            try: t = Time(r['obsdate'], format='isot', scale='utc').mjd
            except: t = 0.0
            external_rvs.append({"telescope": "LAMOST_LRS", "mjd": t, "rv": rv, "rv_err": err, "flag": "z_meas"})

    # Process LAMOST MRS
    for r in mrs_rows:
        rv = get_val(r, 'rv_br1')
        if np.isfinite(rv):
            try: t = Time(r['obsdate'], format='isot', scale='utc').mjd
            except: t = 0.0
            external_rvs.append({"telescope": "LAMOST_MRS", "mjd": t, "rv": rv, "rv_err": get_val(r, 'rv_br1_err', 0), "flag": str(r.get('rv_br_flag', ''))})

    # Process RAVE
    for r in rave_rows:
        rv = get_val(r, 'hrv_sparv')
        if np.isfinite(rv):
            # Your specific query does not return mjd_obs, so we default to 0.0
            external_rvs.append({"telescope": "RAVE_DR6", "mjd": 0.0, "rv": rv, "rv_err": get_val(r, 'hrv_error_sparv', 0), "flag": str(r.get('rave_obs_id', ''))})

    return {"metadata": metadata, "external_rvs": external_rvs}
