# io_utils.py
import os
import logging
import numpy as np
import pandas as pd
import astropy.io.fits as fits
from astropy.time import Time
from pathlib import Path
from . import config

def extract_mjd_from_header(header, instrument=None):
    if instrument is not None and hasattr(instrument, "header_keywords"):
        if hasattr(header, 'get') or isinstance(header, (dict, fits.Header)):
            if "bjd" in instrument.header_keywords:
                key = instrument.header_keywords["bjd"]
                if key in header:
                    return Time(header[key], format='jd', scale='utc').mjd
            if "jd" in instrument.header_keywords:
                key = instrument.header_keywords["jd"]
                if key in header:
                    return Time(header[key], format='jd', scale='utc').mjd

    if isinstance(header, (list, tuple)):
        for line in header:
            if isinstance(line, str):
                if line.startswith("# THEMIDPT") or line.startswith("# AMIDPT"):
                    try:
                        date_str = line.split('=')[1].strip()
                        return Time(date_str, format='isot', scale='utc').mjd
                    except Exception:
                        pass
    logging.warning("Could not extract MJD from header. Returning 0.0")
    return 0.0

def read_spectrum(filename):
    header = []
    spectrum_data = {}
    current_order = None
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith("#"):
                    if "Order" in line:
                        current_order = int(line.split()[-1])
                        spectrum_data[current_order] = {"wavelength": [], "flux": [], "eflux": []}
                    else:
                        header.append(line)
                elif current_order is not None:
                    values = line.replace(',', ' ').split()
                    if len(values) == 2:
                        val = float(values[1])
                        spectrum_data[current_order]["wavelength"].append(float(values[0]))
                        spectrum_data[current_order]["flux"].append(val)
                        spectrum_data[current_order]["eflux"].append(np.sqrt(np.abs(val)))
    except Exception as e:
        logging.error(f"Error reading APF file {filename}: {e}")
        raise
    return header, spectrum_data

def read_spectrum_ghost(blue_filename):
    if isinstance(blue_filename, list): blue_filename = blue_filename[0]
    if "blue" not in str(blue_filename).lower():
        raise ValueError(f"File {blue_filename} must contain 'blue' in name.")
    red_filename = str(blue_filename).replace("blue", "red")
    spectrum_data = {}
    order_index = 0
    header = None
    def process_arm(fname, chunk_count):
        nonlocal order_index, header
        with fits.open(fname) as hdul:
            if header is None: header = hdul["SCI"].header
            sci = hdul["SCI"]
            flux = sci.data
            hdr = sci.header
            wave = hdr["CRVAL1"] * np.exp(hdr["CD1_1"] * (np.arange(hdr["NAXIS1"]) + 1 - hdr["CRPIX1"]) / hdr["CRVAL1"])
            wave_ang = wave * 10
            eflux = np.sqrt(np.abs(flux))
            indices = np.linspace(0, hdr["NAXIS1"], chunk_count + 1, dtype=int)
            for i in range(chunk_count):
                idx0, idx1 = indices[i], indices[i+1]
                spectrum_data[order_index] = {
                    "wavelength": wave_ang[idx0:idx1].tolist(),
                    "flux": flux[idx0:idx1].tolist(),
                    "eflux": eflux[idx0:idx1].tolist(),
                }
                order_index += 1
    process_arm(blue_filename, 30)
    process_arm(red_filename, 31)
    return header, spectrum_data

def read_spectrum_maroonx(filename):
    store = pd.HDFStore(filename, 'r')
    try:
        spec_blue, header_blue = store['spec_blue'], store['header_blue']
        spec_red, header_red = store['spec_red'], store['header_red']
    finally:
        store.close()
    orders_blue = spec_blue.index.levels[1]
    if 91 in orders_blue: orders_blue = orders_blue.drop(91)
    orders_red = spec_red.index.levels[1]
    spectrum_data = {}
    order_idx = 0
    for arm_spec, orders in [(spec_blue, orders_blue), (spec_red, orders_red)]:
        for o in orders:
            wave_nm = arm_spec['wavelengths'][6][o]
            flux = arm_spec['optimal_extraction'][6][o]
            spectrum_data[order_idx] = {
                "wavelength": (np.array(wave_nm) * 10.0).tolist(),
                "flux": flux.tolist(),
                "eflux": np.sqrt(np.abs(flux)).tolist()
            }
            order_idx += 1
    return header_blue, spectrum_data

def read_bias(filename):
    if not filename or not os.path.exists(filename):
        logging.warning(f"Bias file {filename} not found. Assuming zero bias.")
        return {}
    bias = {}
    try:
        df = pd.read_csv(filename, sep='\s+', header=None, comment='#')
        for _, row in df.iterrows():
            try:
                idx = int(row[0])
                bias[idx] = [float(row[1]), float(row[2]), float(row[3])]
            except (ValueError, IndexError):
                continue
    except Exception as e:
        logging.warning(f"Failed to read bias file: {e}. Using zeros.")
    return bias

def write_order_results(order_data, input_filename):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    base = Path(input_filename).stem
    outfile = config.OUTPUT_DIR / f"{base}_orders.txt"
    with open(outfile, "w") as f:
        f.write("# Order | RV (km/s) | RV Error (km/s)\n")
        def _sort_key(k):
            parts = str(k).split("_")
            return tuple(int(x) for x in parts)
        for order in sorted(order_data.keys(), key=_sort_key):
            d = order_data[order]
            rv = f"{d['best_rv']:.8f}" if np.isfinite(d['best_rv']) else "NaN"
            err = f"{d['best_rv_err']:.8f}" if np.isfinite(d['best_rv_err']) else "NaN"
            f.write(f"{order} {rv} {err}\n")

def write_summary(processed_data):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    if not processed_data: return
    first_file = Path(processed_data[0]["file"]).name
    parts = first_file.split("_")
    obj_id = parts[2] if len(parts) > 2 else "summary"
    outfile = config.OUTPUT_DIR / f"{obj_id}_summary.txt"
    existing = {}
    if outfile.exists():
        with open(outfile, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if parts: existing[parts[0]] = line.strip()
    for entry in processed_data:
        line = f"{entry['file']} {entry['mjd']:.8f} {entry['rv']:.8f} {entry['rv_err']:.8f} {entry['rv_rms']:.8f}"
        existing[entry['file']] = line
    with open(outfile, "w") as f:
        f.write("# File Summary\n# Input File | MJD | RV | Err | RMS\n")
        for line in existing.values():
            f.write(line + "\n")
    logging.info(f"Summary written to {outfile}")

def write_star_summary(obj_id, gaia_data, pipeline_results):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    outfile = config.OUTPUT_DIR / f"Gaia_DR3_{obj_id}_summary.txt"
    
    with open(outfile, "w") as f:
        f.write(f"### STAR SUMMARY: {obj_id} ###\n\n")
        
        # 1. Metadata from Gaia (Strict Key: Value format)
        if gaia_data and gaia_data.get('metadata'):
            f.write("[GAIA METADATA]\n")
            m = gaia_data['metadata']
            
            for key, val in m.items():
                if isinstance(val, float):
                    if np.isfinite(val):
                        f.write(f"{key}: {val:.8f}\n")
                    else:
                        f.write(f"{key}: NaN\n")
                else:
                    f.write(f"{key}: {val}\n")
                    
        else:
            f.write("[GAIA METADATA]\nNot Found or Query Failed.\n")

        # 2. External Data
        f.write("\n[EXTERNAL RV DATA]\n")
        f.write("# Telescope | MJD | RV (km/s) | Err (km/s) | Flag/ID\n")
        if gaia_data and gaia_data.get('external_rvs'):
            for ext in gaia_data['external_rvs']:
                f.write(f"{ext['telescope']} {ext['mjd']:.5f} {ext['rv']:.3f} {ext['rv_err']:.3f} {ext['flag']}\n")
        else:
            f.write("# No external data found.\n")

        # 3. Pipeline Data
        f.write("\n[PIPELINE RESULTS]\n")
        f.write("# File | MJD | RV (km/s) | Err (km/s) | RMS | Fallback?\n")
        for res in pipeline_results:
             f.write(f"{res['file']} {res['mjd']:.8f} {res['rv']:.8f} {res['rv_err']:.8f} {res['rv_rms']:.8f} {res['fallback']}\n")
             
    logging.info(f"Detailed star summary written to {outfile}")
