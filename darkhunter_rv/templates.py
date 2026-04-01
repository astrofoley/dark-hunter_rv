# templates.py
import logging
import os
import numpy as np
import astropy.io.fits as fits
from . import config
from . import physics
from . import continuum

MODEL_CACHE = {}

def parse_phoenix_dirname(dirname):
    name = dirname.lower()
    if "phoenixm" in name: return -float(name.split("phoenixm")[1]) / 10.0
    if "phoenixp" in name: return float(name.split("phoenixp")[1]) / 10.0
    return 0.0

def build_template_bank(teff, vsini_proxy, metallicity=0.0, logg=4.5, wave_range=(3300, 9000), air=True):
    global MODEL_CACHE
    templates = {}
    base_dir = config.PHOENIX_BASE_DIR
    
    if vsini_proxy is None: vsini_proxy = 10.0
    vb_values = [max(0, vsini_proxy*0.8), vsini_proxy, vsini_proxy*1.2]
    
    if not base_dir.exists():
        logging.warning("Phoenix dir %s not found.", base_dir)
        return {}

    # Scan directories
    for d in os.listdir(base_dir):
        if not d.lower().startswith("phoenix"): continue
        mval = parse_phoenix_dirname(d)
        if abs(mval - metallicity) > 0.5: continue
        
        path_d = base_dir / d
        if not path_d.is_dir(): continue
        
        for fname in os.listdir(path_d):
            if not fname.endswith(".fits"): continue
            try:
                # Filename format: phoenix_Te_...
                t_file = float(fname.split("_")[1].replace(".fits",""))
                if abs(t_file - teff) > 1000: continue
                
                # Load
                full_path = path_d / fname
                with fits.open(full_path) as hdul:
                    data = hdul[1].data
                    wave = data["WAVELENGTH"]
                    if air: wave = physics.vac_to_air(wave)
                    
                    mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
                    wave = wave[mask]
                    
                    # Columns (logg)
                    for col in hdul[1].columns.names:
                        if not col.startswith("g"): continue
                        g_val = float(col[1:]) / 10.0
                        if abs(g_val - logg) > 0.5: continue
                        
                        flux = data[col][mask]
                        if np.all(flux==0): continue
                        
                        # Generate broadened versions
                        for vb in vb_values:
                            key = (t_file, g_val, mval, vb)
                            if key in MODEL_CACHE:
                                templates[key] = MODEL_CACHE[key]
                            else:
                                f_broad = physics.broaden_spectrum(wave, flux, vb)
                                # Normalize template ONCE here
                                cont = continuum.compute_template_global_continuum(wave, f_broad)
                                # Renormalize local chunks (simplified from original)
                                _, f_norm, _ = continuum.renormalize_local(wave, f_broad, cont, poly_order=2)
                                
                                MODEL_CACHE[key] = (wave, f_norm)
                                templates[key] = (wave, f_norm)
                                
            except Exception as e:
                continue
                
    return templates
