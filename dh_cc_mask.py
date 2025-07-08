# This program will read in a spectrum files, compare to a stellar mask, and determine the radial velocity and uncertainty for those files.

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import savgol_filter, medfilt

from scipy.optimize import curve_fit
from astropy.time import Time
from astropy.stats import sigma_clip
from astropy.table import Table
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import SkyCoord

import glob
import os

from datetime import datetime

from astroquery.gaia import Gaia


def suppress_warnings():
    """Suppress warnings if the --quiet flag is set."""
    warnings.filterwarnings("ignore")


def find_duplicates(filename, last_modified):
    """
    Finds duplicate entries in output folder.
    Parameters:
        filename: Spectrum data file
        last_modified: Last modified date of spectrum data file
    Returns:
        break_flag: True if duplicate, false otherwise
    """
    break_flag = False

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_file = os.path.join(output_dir, f"{base_name}_orders.txt")

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            last_date = f.readline()
            if "# Date last modified:" in last_date:

                last_date = last_date.replace("# Date last modified: ", "")
                last_date = last_date.replace("\n", "")
                last_date = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S.%f")
                print(f"Date last modified: {last_date}")

                if last_date < last_modified:
                    print(f"Recent edits to data. Overwriting...")
                    pass
                else:
                    break_flag = True
            else:
                pass

    return break_flag

def detect_changes(file_path):
    """
    Detects when a file was last modified.
    Parameters:
        file_path: path to spectrum file
    Returns:
        date: Datetime object with date of last modifications
    """
    last_modified = os.path.getmtime(file_path)
    date = datetime.fromtimestamp(last_modified)
    return date


def read_spectrum(filename):
    """
    Reads and stores the spectrum data.
    Returns:
        header: Metadata list
        spectrum data: dictionary in form {'order': {'wavelength':[array], 'flux':[array], 'flux_err':[array]}}
    """
    header = []
    spectrum_data = {}  # Dictionary to store data for each order
    current_order = None  # Track the current order

    if "order" in filename:  # for 1 order spectrums
        filename_list = filename.split("_")
        order_index = filename_list.index("order")
        current_order = filename_list[order_index + 1]
        current_order = int(''.join(num for num in current_order if num.isdigit()))
        spectrum_data[current_order] = {"wavelength": [], "flux": [], "eflux": []}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading and trailing whitespace
            if not line:  # Skip blank lines
                continue
            line = line.rstrip('\n')
            if line.startswith("#"):
                if "Order" in line:  # Detect a new order
                    current_order = int(line.split()[-1])  # Extract the order number
                    spectrum_data[current_order] = {"wavelength": [], "flux": [], "eflux": []}
                else:
                    header.append(line)
            elif current_order is not None:  # Append data to the current order
                values = line.split()  # Try space-separated first
                if len(values) != 2:  # If not exactly two elements, try comma-separated
                    values = line.split(',')
                if len(values) == 2:  # Ensure we have two valid elements
                    spectrum_data[current_order]["wavelength"].append(float(values[0]))
                    spectrum_data[current_order]["flux"].append(float(values[1]))
                    spectrum_data[current_order]["eflux"].append(np.sqrt(np.abs(float(values[1]))))

    return header, spectrum_data


def read_lamost_spectrum(filename):
    """
    Reads and stores the spectrum data from a lamost file.
    Returns:
        header: Metadata list
        spectrum data: dictionary in form {'order': {'wavelength':[array], 'flux':[array], 'flux_err':[array]}}
    """
    header = []
    spectrum_data = {}  # Dictionary to store data for each order
    index = None  # Track table index

    coadd = False
    max_index = []
    early_release = False

    star_data = fits.open(filename, memmap=True)
    if star_data.__contains__("COADD_B"):
        coadd = True
        max_index.append(star_data.index_of("COADD_B") - 1)  # HDUL index 0 is header
    if star_data.__contains__("COADD_R"):
        coadd = True
        max_index.append(star_data.index_of("COADD_R") - 1)
    if star_data.__contains__("COADD"):
        coadd = True
        max_index.append(star_data.index_of("COADD") - 1)
    if star_data.__contains__("Flux"):  # Earlier data releases
        spectrum = Table()
        spectrum['WAVELENGTH'] = [star_data["Flux"].data[2]]
        spectrum['FLUX'] = [star_data["Flux"].data[0]]
        spectrum['FLUX_ERR'] = [star_data["Flux"].data[1]]
        early_release = True

    for i in range(len(star_data)):
        if not early_release:
            spectrum = Table(star_data[i].data)
        if i == 0:
            header = str(star_data[0].header)
            header = ' '.join(header.split())
            header = header.split('/')
        if "FLUX" in spectrum.columns or early_release is True:
            if index is None:
                index = 0  # Tracks index
            else:
                index += 1
            if coadd:
                if index not in max_index:
                    continue  # Only uses coadd spectra, if applicable
            spectrum_data[index] = {"wavelength": [], "flux": [], "eflux": []}
            wavelength = spectrum['WAVELENGTH'].data.tolist()
            wavelength = wavelength[0]
            flux = spectrum['FLUX'].data.tolist()
            flux = flux[0]
            eflux = np.sqrt(np.abs(spectrum['FLUX'].data)).tolist()
            eflux = eflux[0]

            for element in range(len(wavelength)):
                spectrum_data[index]["wavelength"].append(float(wavelength[element]))
                spectrum_data[index]["flux"].append(float(flux[element]))
                spectrum_data[index]["eflux"].append(float(eflux[element]))

    for i in range(len(header)):

        if " RA = " in header[i]:  # Gets RA from header
            ra = header[i]
            ra = ra.split("=")[-1]
            ra = ra.strip()

        if " DEC = " in header[i]:  # Gets DEC from header
            dec = header[i]
            dec = dec.split("=")[-1]
            dec = dec.strip()

        if " OBJNAME = " in header[i]:  # Gets object name from header (not always Gaia designation)
            source = header[i]
            source = source.strip()
            objname = source.split("=")[-1]
            objname = objname.replace("'", "")
            objname = objname.strip()

    try:
        # Uses RA and DEC to search for designation
        coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
        j = Gaia.cone_search_async(coord, radius=u.Quantity(0.0005, u.deg))
        r = j.get_results()
        source = r[0]["designation"]
        designation = source.split()[-1]
    except (NameError, IndexError):
        designation = objname  # Fall back to object name

    return header, spectrum_data, designation


def suborders(suborders_in, old_spectrum_data):  # Suborders
    '''
    Splits spectrum data into suborders
    Parameters:
        suborders_in: number of suborders for each order
        spectrum_data: dictionary of spectrum data
    '''
    spectrum_data = {}  # Empty dictionary to hold data

    for order in old_spectrum_data:

        spectrum_data.update({order: {}})

        old_wavelength = old_spectrum_data[order]["wavelength"]
        wavesplit = np.array_split(old_wavelength, suborders_in)  # Splits the wavelengths into suborders

        old_flux = old_spectrum_data[order]["flux"]
        fluxsplit = np.array_split(old_flux, suborders_in)  # Splits the flux into suborders

        old_eflux = old_spectrum_data[order]["eflux"]
        efluxsplit = np.array_split(old_eflux, suborders_in)  # Splits the flux error into suborders

        for suborder in range(0, suborders_in):
            spectrum_data[order].update({suborder: {}})
            spectrum_data[order][suborder] = {"wavelength": [], "flux": [], "eflux": []}

            for i in range(len(wavesplit[suborder])):
                spectrum_data[order][suborder]["wavelength"].append(float(wavesplit[suborder][i]))
                spectrum_data[order][suborder]["flux"].append(float(fluxsplit[suborder][i]))
                spectrum_data[order][suborder]["eflux"].append(float(efluxsplit[suborder][i]))

    return spectrum_data


def gaia_query(object_id):
    """
    Queries Gaia for object properties, including: RA, DEC, parallax, teff, logg
    Parameters:
        object_id: GAIA ID, as found in spectrum file
    Returns:
        gaia_properties: Dictionary containing properties from Gaia
    """
    # Initial query
    query = f"""SELECT 
    TOP 2000
    source_id, ra, dec, parallax, teff_gspphot, logg_gspphot
    FROM gaiadr3.gaia_source
    WHERE source_id = {object_id}
    """
    job = Gaia.launch_job(query)
    results = job.get_results()

    ra = results[0]['ra']
    dec = results[0]['dec']
    parallax = results[0]['parallax']
    teff = results[0]['teff_gspphot']
    logg = results[0]['logg_gspphot']

    gaia_properties = {"ra": ra, "dec": dec, "parallax": parallax, "teff": teff, "logg": logg}

    return gaia_properties


def get_spectral_type(teff):
    """
    Finds (roughly) the best spectral type from effective temperature, based on limited masks
    If spectral type not included in stellar_masks, defaults to G2
    """

    if 5300 < teff < 6000:
        if 5930 < teff < 6000:
            spectral_type = "G9"
        if 5860 < teff < 5930:
            spectral_type = "G8"
        else:
            spectral_type = "G2"
    elif 6000 < teff < 7300:
        spectral_type = "F9"
    elif 3900 < teff < 5300:
        if 4600 < teff < 5300:
            spectral_type = "K6"
        else:
            spectral_type = "K2"
    elif 2300 < teff < 3900:
        spectral_type = "M2"
    else:
        spectral_type = "G2"

    return spectral_type


def read_bias(filename):
    """
    Reads and stores the bias as an array
    """
    bias = np.zeros((70, 3))

    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()  # Remove leading and trailing whitespace
            if not line:  # Skip blank lines
                continue
            line = line.rstrip('\n')
            values = line.split()
            bias[int(values[0]), :] = [float(values[1]), float(values[2]), float(values[3])]
            # bias[int(values[0]), :] = [float(values[2]), float(values[3]), float(values[4])] # if suborder

    return bias


def read_bias_suborders(filename):
    """
    Reads and stores the bias as a dictionary, with suborders
    Assumes bias already has suborders
    """
    bias = {}

    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()  # Remove leading and trailing whitespace
            if not line:  # Skip blank lines
                continue
            line = line.rstrip('\n')
            values = line.split()
            if len(values) != 5:
                bias = read_bias(filename)  # If bias file does not have suborders, defaults to 0 suborders
                bias_suborder = False
                return bias, bias_suborder
            if int(values[0]) not in bias.keys():
                bias.update({int(values[0]): {}})
            bias[int(values[0])].update({int(values[1]): [float(values[2]), float(values[3]), float(values[4])]})

    bias_suborder = True
    return bias, bias_suborder

def bias_determination(filename, bias_check, suborder_int):
    """
    Determines which bias to use.
    Parameters:
        filename: Name of bias file
        bias_check: Boolean determining whether to use bias
        suborder_int: Number of suborders
    Returns:
        bias: An array (1 suborder) or dictionary with bias data
        suborder_check: Boolean determining if bias file contains suborders
    """
    suborder_check = False
    if not bias_check:
        with open(filename, "r") as file:
            content = file.read()
            if content.__contains__("Suborder"):
                bias, suborder_check = read_bias_suborders(filename)
                if suborder_int == 1:
                    new_bias = np.zeros((70, 3))
                    for order in bias.keys():
                        try:
                            new_bias[order, :] = bias[order][0]
                        except KeyError:
                            for i in range(0, 10):
                                try:
                                    new_bias[order, :] = bias[order][i]
                                    break
                                except KeyError:
                                    new_bias[order, :] = np.array([0, 0, 0])
                    bias = new_bias
                    suborder_check = False  # Fall back to no suborder bias
            else:
                bias = read_bias(filename)

    if bias_check:
        if suborder_int == 1:
            bias = np.zeros((70, 3))  # Empty bias
        else:
            bias = {}  # Empty bias for suborders
            suborder_check = True
            for i in range(70):
                for suborder in range(suborder_int):
                    if i not in bias.keys():
                        bias.update({i: {}})
                    bias[i].update({suborder: [0, 0, 0]})

    return bias, suborder_check


def read_stellar_mask(spectral_type, telescope, version, directory="stellar_masks/", ):
    """
    Reads the stellar mask file for a given spectral type and telescope.
    Defaults to espresso if no telescope given

    Parameters:
        spectral_type (str): The spectral classification (e.g., "G2").
        telescope (str): The telescope name (e.g., "espresso").
        version (str): The version of the stellar mask file (e.g., "v1.0") (for neids)
        directory (str): The path to the directory containing mask files.

    Returns:
        dict: A dictionary with keys "wavelength" and "flux", each containing numpy arrays.

    """
    # Construct the filename
    if telescope == "harps":
        filename = f"{spectral_type}.{telescope}.mas"
    if telescope == "neid":
        if version == "v2":
            filename = f"{spectral_type}.{telescope}.{version}.mas"
        else:
            filename = f"{spectral_type}.{telescope}.v1.mas"
    else:
        telescope = "espresso"
        filename = f"{spectral_type}_{telescope}.txt"

    filepath = os.path.join(directory, filename)

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Stellar mask file '{filename}' not found in {directory}")

    # Load the data (space-delimited)
    data = np.loadtxt(filepath, dtype=np.float64)

    # Extract wavelength and flux
    wavelength, flux = data[:, 0], data[:, 1]

    return {"wavelength": wavelength, "flux": flux}


def outlier_mask(wavelength, flux, window_size=20, hi_sigma_threshold=5, lo_sigma_threshold=20, max_iter=3):
    """
    Improved outlier detection for cosmic rays using a combination of:
    - Median filtering
    - Savitzky-Golay smoothing
    - Derivative-based detection
    """
    flux = np.array(flux, dtype=np.float64)
    mask = np.ones_like(flux, dtype=bool)

    # Apply median filtering for robust trend estimation
    smooth_flux = medfilt(flux, kernel_size=5)

    # Compute Savitzky-Golay smoothed flux for better local fitting
    sg_flux = savgol_filter(flux, window_length=7, polyorder=2, mode='interp')

    # Compute residuals and apply asymmetric sigma-clipping
    residuals = flux - sg_flux
    mad_std = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))

    for _ in range(max_iter):
        hi_mask = residuals > hi_sigma_threshold * mad_std
        lo_mask = residuals < -lo_sigma_threshold * mad_std
        mask[hi_mask | lo_mask] = False  # Mask cosmic rays

    return mask


def absorption_mask(wavelength, flux, mask, window_size=100, drop_threshold=0.5, sn_threshold=10):
    """
    Improved absorption line detection using:
    - Rolling percentile filtering
    - Adaptive S/N thresholding
    - Masking based on local minima
    """
    flux = np.array(flux, dtype=np.float64)
    smooth_flux = medfilt(flux, kernel_size=9)  # Smoother trend estimation

    # Compute a rolling 10th percentile to detect strong absorptions
    rolling_min = np.array([np.percentile(flux[max(0, i - window_size // 2):min(len(flux), i + window_size // 2)], 10)
                            for i in range(len(flux))])

    # Compute S/N estimate
    residuals = np.abs(flux - smooth_flux)
    sn_estimate = smooth_flux / np.maximum(residuals, 1e-6)

    # Identify absorption regions: flux drops significantly below rolling min and S/N is good
    absorption_regions = (flux < rolling_min * drop_threshold) & (sn_estimate > sn_threshold)

    # Update mask
    mask &= ~absorption_regions
    return mask


def fit_continuum(wavelength, flux, eflux, num_knots=10, order=3):
    """
    Improved continuum fitting using:
    - Outlier rejection
    - Adaptive spline fitting
    - Iterative clipping
    Returns:
        norm_wave (array): Filtered and masked wavelength
        norm_flux (array): Filtered and masked flux
        norm_eflux (array): Filtered and masked flux error
    """
    wavelength = np.array(wavelength, dtype=np.float64)
    flux = np.array(flux, dtype=np.float64)

    # Initial outlier and absorption masks
    mask_cr = outlier_mask(wavelength, flux)
    mask_abs = absorption_mask(wavelength, flux, mask_cr)

    # Select clean data points
    good_data = mask_cr & mask_abs

    # Define knot positions adaptively
    knot_positions = np.linspace(wavelength[good_data][0], wavelength[good_data][-1], num_knots)[1:-1]

    # Fit spline to valid points
    spline = LSQUnivariateSpline(wavelength[good_data], flux[good_data], t=knot_positions, k=order)

    # Smooth the final continuum
    continuum = savgol_filter(spline(wavelength), window_length=11, polyorder=2, mode='interp')

    # Mask regions where the spline goes negative or very close to zero
    mask_spline = continuum > 0.1 * np.nanmedian(continuum)

    # Check the S/N
    smooth_flux = medfilt(flux, kernel_size=1001)
    smooth_err = medfilt(np.abs(eflux), kernel_size=1001)
    mask_snr = smooth_flux / smooth_err > 3

    if 10 < 0:
        plt.plot(wavelength, smooth_flux)
        plt.plot(wavelength, smooth_err)
        plt.plot(wavelength, smooth_err * 3)
        plt.show()

    # Identify continuous regions of good (positive) continuum
    bad_indices = np.where((~mask_spline) | (~mask_snr))[0]  # Indices where spline is invalid
    mask = np.ones_like(wavelength, dtype=bool)

    if 10 < 0:
        plt.plot(wavelength, flux)
        plt.plot(wavelength[(~mask_spline) | (~mask_snr)], flux[(~mask_spline) | (~mask_snr)])
        plt.show()

    if len(bad_indices) > 0:
        # Find continuous regions of "bad" points
        breaks = np.where(np.diff(bad_indices) > 1)[0] + 1  # Indices where discontinuities occur
        bad_regions = np.split(bad_indices, breaks)  # Split into separate regions

        # Define good regions as the complement of bad regions
        good_regions = []
        start_idx = 0  # Start index of a potential good region

        for bad_region in bad_regions:
            end_idx = bad_region[0]  # The start of a bad region marks the end of a good one
            if end_idx > start_idx:  # Ensure it's nonempty
                good_regions.append((start_idx, end_idx))
            start_idx = bad_region[-1] + 1  # The end of a bad region marks the start of the next good one

        # Add the final good region if it extends to the end
        if start_idx < len(wavelength):
            good_regions.append((start_idx, len(wavelength)))

        # Select the largest continuous "good" region
        if len(good_regions) > 0:
            best_region = max(good_regions, key=lambda x: x[1] - x[0])  # Largest region by width
            anti_mask_spline = np.zeros_like(wavelength, dtype=bool)
            anti_mask_spline[best_region[0]:best_region[1]] = True  # Keep only this region
        else:
            anti_mask_spline = np.ones_like(wavelength, dtype=bool)  # Default to keeping everything

        # Update mask with both constraints
        mask &= mask_spline
        mask &= anti_mask_spline

    if 10 < 0:
        plt.step(wavelength, flux, color="black", zorder=0)
        plt.step(wavelength[mask_cr], flux[mask_cr], color="green", zorder=1)
        plt.step(wavelength[mask], flux[mask], color="orange", zorder=1)
        plt.step(wavelength, spline(wavelength), color="blue", zorder=2)
        plt.plot(wavelength, continuum, color="red", zorder=3)
        plt.show()

    # Normalize flux
    norm_flux = flux / continuum
    norm_flux[~mask_cr] = 1  # Set masked regions to unity
    norm_eflux = eflux / continuum
    norm_eflux[~mask_cr] = 1e30  # Large uncertainty for masked points

    norm_wave = wavelength[mask]
    norm_flux = norm_flux[mask]
    norm_eflux = norm_eflux[mask]

    return norm_wave, norm_flux, norm_eflux


def extract_mjd_from_header(header):
    """
    Locates the Modified Julian Date
    """
    for line in header:
        if line.startswith("# THEMIDPT"):
            date_str = line.split('=')[1].strip()
            time = Time(date_str, format='isot', scale='utc')
            return time.mjd
        if " MJD = " in line:
            mjd = line.strip()
            mjd = mjd.split("=")[-1]
            mjd = float(mjd)
            return mjd


def cross_correlate_stellar_mask(obs_wavelength, obs_flux, mask_wavelength, mask_strength, max_lag=350, fit_width=50):
    """
    Cross-correlate an observed spectrum with a stellar mask in log-wavelength space,
    and fit a Gaussian to the peak of the cross-correlation function.

    Parameters:
        obs_wavelength (array): Wavelengths of the observed spectrum.
        obs_flux (array): Flux values of the observed spectrum.
        mask_wavelength (array): Wavelengths of the stellar mask (absorption line positions).
        mask_strength (array): Line strengths in the stellar mask.
        max_lag (int): Number of log-wavelength shifts to test.
        fit_width (int): Number of points on either side of the peak for Gaussian fitting.

    Returns:
        best_velocity_shift (float): Best-fit velocity shift (km/s).
        velocity_shifts (array): Velocity shifts corresponding to each log shift.
        cross_corr (array): Cross-correlation function values.
    """
    # Convert to log-wavelength space
    log_obs_wavelength = np.log10(obs_wavelength)
    log_mask_wavelength = np.log10(mask_wavelength)

    # Define log-wavelength shifts to test
    subpixel = 5.0
    median_delta_logw = np.median(np.diff(log_obs_wavelength)) / subpixel  # Approximate step size
    log_shifts = np.linspace(-max_lag * median_delta_logw, max_lag * median_delta_logw, 2 * max_lag + 1)

    cross_corr = np.zeros_like(log_shifts)

    for i, log_shift in enumerate(log_shifts):
        # Shift stellar mask in log-wavelength space
        shifted_mask_wavelength = log_mask_wavelength + log_shift

        # Find nearest observed flux points for the shifted mask
        obs_indices = np.searchsorted(log_obs_wavelength, shifted_mask_wavelength)
        valid_indices = (obs_indices > 0) & (obs_indices < len(log_obs_wavelength))

        if np.sum(valid_indices) < 5:
            continue  # Skip if too few matches

        # Extract matched observed flux values
        matched_flux = obs_flux[obs_indices[valid_indices]]

        # Compute weighted correlation using mask line strengths
        cross_corr[i] = np.sum(matched_flux * mask_strength[valid_indices])

    # Convert log-wavelength shifts to velocity shifts (Doppler formula)
    velocity_shifts = 299792.458 * (10 ** log_shifts - 1)

    # **Identify the peak and fit only a small range around it**
    peak_index = np.argmax(cross_corr)
    fit_indices = np.arange(max(0, peak_index - fit_width), min(len(cross_corr), peak_index + fit_width + 1))

    if len(fit_indices) < 3:  # Ensure we have enough points for fitting
        return velocity_shifts[peak_index], velocity_shifts, cross_corr

    # Define Gaussian function
    def gaussian(x, a, mu, sigma):
        return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    try:
        popt, pcov = curve_fit(
            gaussian, velocity_shifts[fit_indices], cross_corr[fit_indices],
            p0=[np.max(cross_corr), velocity_shifts[peak_index], 10],
            bounds=((0, -np.inf, 0), (np.inf, np.inf, np.inf))
        )
        perr = np.sqrt(np.diag(pcov))  # Standard deviation
        best_velocity_shift = popt[1]  # Gaussian center
        # best_velocity_err = popt[2]
        best_velocity_err = perr[1]
    except (RuntimeError, ValueError):
        best_velocity_shift = velocity_shifts[peak_index]  # Fallback to discrete peak
        best_velocity_err = 1e30

    if 10 < 0:
        print(best_velocity_shift, best_velocity_err, popt[2])
        plt.plot(velocity_shifts, cross_corr)
        plt.plot(velocity_shifts, gaussian(velocity_shifts, popt[0], popt[1], popt[2]))
        plt.show()

    return best_velocity_shift, best_velocity_err, velocity_shifts, cross_corr


def write_order_results(order_data, input_filename, suborders_in, last_modified):
    """
    Writes order-level RV results, including skipped orders as NaN values.

    Parameters:
        order_data (dict): Dictionary where keys are order numbers and values are tuples (rv, rv_err).
        input_filename (str): Name of the input file to generate output file name.
        suborders_in (int): Number of suborders
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Derive output file name
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_file = os.path.join(output_dir, f"{base_name}_orders.txt")

    # Write data
    with open(output_file, "w") as f:

        f.write(f"# Date last modified: {last_modified}\n")

        if suborders_in != 1:
            f.write("# Order | Suborder | RV (km/s) | RV Error (km/s)\n")
        else:
            f.write("# Order | RV (km/s) | RV Error (km/s)\n")

        for order_int, order in sorted(order_data.items()):
            for suborder in sorted(order.keys()):
                if bool(order_data[order_int][suborder]):
                    rv_str = f"{order_data[order_int][suborder]['best_rv']:.8f}" if np.isfinite(
                        order_data[order_int][suborder]['best_rv']) else "NaN"
                    err_str = f"{order_data[order_int][suborder]['best_rv_err']:.8f}" if np.isfinite(
                        order_data[order_int][suborder]["best_rv_err"]) else "NaN"
                    if suborders_in != 1:
                        f.write(f"{order_int} {suborder} {rv_str} {err_str}\n")
                    else:
                        f.write(f"{order_int} {rv_str} {err_str}\n")
                else:
                    continue


def write_summary(processed_data, gaia_properties):
    """
    Writes or updates the summary of processed spectra.

    Parameters:
        processed_data (list): List of dictionaries with processed data.
        gaia_properties (dict): Dictionary of gaia properties.
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Extract object ID from input file name
    # base_name = os.path.splitext(os.path.basename(processed_data["file"]))[0]
    # object_id = base_name.split("_")[2]  # Assumes filename structure
    object_id = gaia_properties["object_id"]

    # Define summary file for the object
    output_file = os.path.join(output_dir, f"{object_id}_summary.txt")

    # Read existing summary if the file exists
    existing_entries = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            lines = f.readlines()
            for line in lines[2:]:  # Skip header
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                filename = parts[0]
                existing_entries[filename] = line.strip()

    # Update entries with new data
    new_entry = f"{processed_data['file']} {processed_data['mjd']:.8f} {processed_data['rv']:.8f} {processed_data['rv_err']:.8f} {processed_data['rv_rms']:.8f}"
    existing_entries[processed_data['file']] = new_entry  # Overwrite if exists

    #    for entry in processed_data:
    #        print(processed_data)
    #        print(processed_data.keys())
    #        print(entry)
    #        print(entry.keys())
    #        new_entry = entry['file'] + f" {entry['mjd']:.8f} {entry['rv']:.8f} {entry['rv_err']:.8f} {entry['rv_rms']:.8f}"
    #        existing_entries[entry['file']] = new_entry  # Overwrite if exists

    # Write updated summary file
    with open(output_file, "w") as f:
        f.write("# File Summary\n")
        f.write(f"# OBJECT ID = {gaia_properties['object_id']}\n")
        f.write(f"# RA = {gaia_properties['ra']} (deg)\n")
        f.write(f"# DEC = {gaia_properties['dec']} (deg)\n")
        f.write(f"# PARALLAX = {gaia_properties['parallax']} (mas)\n")
        f.write(f"# TEFF = {gaia_properties['teff']} (K)\n")
        f.write(f"# LOG(G) = {gaia_properties['logg']} (log(cm.s**-2))\n")
        f.write("# Input File | MJD | RV (km/s) | RV Error (km/s) | RMS (km/s)\n")
        for line in existing_entries.values():
            if line.startswith("#"):
                continue
            else:
                f.write(line + "\n")

    logging.info(f"Summary written to {output_file}")
    return output_file


def old_write_summary(processed_data, output_file):
    """
    Writes a summary of processed spectra to a text file.

    Parameters:
        processed_data (list): List of dictionaries with processed data.
        output_file (str): Path to the output file.
    """
    with open(output_file, "w") as f:
        f.write("# File Summary\n")
        f.write("# Input File | MJD | RV (km/s) | RV Error (km/s) | RMS (km/s)\n")
        for entry in processed_data:
            f.write(
                f"{entry['file']} {entry['mjd']:.5f} {entry['radial_velocity']:.3f} "
                f"{entry['rv_error']:.3f} {entry['rms_orders']:.3f}\n"
            )


def plot_results(processed_data, properties, save=False, show=False, save_path=None):
    """
    Generates plots for processed spectra.

    Parameters:
        processed_data (list): List of dictionaries with processed data.
        save (bool): Whether to save plots to files.
        show (bool): Whether to display plots on-screen.
        save_path (str): Path to save plots to.
        properties (dict): Dictionary of gaia properties.
    """
    plt.figure(figsize=(8, 5))
    star_name = "XXX"

    for entry in processed_data:
        star_name = "Gaia_DR3" + properties['object_id']

        plt.errorbar(
            float(entry['mjd']),
            float(entry['rv']),
            yerr=float(entry['rv_rms']),
            fmt="o"
        )

    plt.xlabel("MJD")
    plt.ylabel("Radial Velocity (km/s)")
    plt.title(f"Radial Velocity for {star_name}")

    if save and save_path is not None:
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def process_spectrum(spectrum_file, spectral_type, telescope, version, sub_order, args=None):
    """Main function to process a spectrum and compute radial velocities."""

    # Bools controlling suborders, orders, biases
    no_bias = False
    order_flag = True
    bias_suborder = False

    modified_date = detect_changes(spectrum_file)

    if args.no_bias:
        no_bias = True

    if not args.overwrite:
        # exits function if the file already exists
        break_flag = find_duplicates(spectrum_file, modified_date)
        if break_flag:
            write_summary_flag = False
            return write_summary_flag, break_flag

    # Read spectrum and stellar mask

    if "Gaia" in spectrum_file:
        # For GAIA data
        header, spectrum_data = read_spectrum(spectrum_file)
        file_name = spectrum_file.split("/")[-1]
        object_id = file_name.split("_")[2]
    elif "med" or "spec" in spectrum_file:
        # for LAMOST data
        header, spectrum_data, object_id = read_lamost_spectrum(spectrum_file)
        no_bias = True
        order_flag = False
    else:
        write_summary_flag = False
        return write_summary_flag, break_flag

    spectrum_data = suborders(sub_order, spectrum_data)
    gaia_properties = gaia_query(object_id)

    gaia_properties.update({"object_id": object_id})
    teff = gaia_properties["teff"]

    if spectral_type is None:
        # Calculates spectral type with TEFF if not given
        spectral_type = get_spectral_type(teff)

    mask_data = read_stellar_mask(spectral_type, telescope, version)

    # Extract MJD
    mjd = extract_mjd_from_header(header)

    mask_wavelength = np.array(mask_data["wavelength"])
    mask_strength = np.array(mask_data["flux"])

    # Store cross-correlation results
    rv_results = {}

    # Order-by-order processing
    order_rvs = []
    order_rv_errs = []

    # List of orders to avoid
    if order_flag:  # No bad orders for fits files
        bad_orders = [0, 1, 2, 53, 57, 58, 59, 60, 63, 64, 65]
    else:
        bad_orders = []
    # bad_orders = [30, 31, 32, 43, 45]
    # 43, 45,  everything <=25

    # Determines bias depending on bias flag and suborder
    bias, bias_suborder = bias_determination("bias_statistics.txt", no_bias, sub_order)

    for order_int, order in spectrum_data.items():
        if order_int not in bad_orders:
            rv_results.update({order_int: {}})
            for suborder, data in order.items():

                if order_int not in rv_results.keys():
                    rv_results.update({order_int: {}})

                rv_results[order_int].update({suborder: {}})

                wavelength = np.array(data["wavelength"])
                flux = np.array(data["flux"])
                eflux = np.array(data["eflux"])

                if (wavelength[-1] > mask_wavelength[0]) & (wavelength[0] < mask_wavelength[-1]):
                    # Clean and normalize spectrum
                    mask_cr = outlier_mask(wavelength, flux)
                    mask_abs = absorption_mask(wavelength, flux, mask_cr)
                    norm_wave, norm_flux, _ = fit_continuum(wavelength, flux, eflux)

                    if np.sum((mask_wavelength < norm_wave[-1]) & (mask_wavelength > norm_wave[0])) > 10:
                        # Cross-correlate with stellar mask
                        if order_int > -1:

                            best_rv, best_rv_err, velocity_shifts, cross_corr = cross_correlate_stellar_mask(
                                norm_wave, 1 - norm_flux, mask_wavelength, mask_strength
                            )

                            if not bias_suborder:
                                # Store results (no suborders)
                                rv_results[order_int][suborder] = {"velocity_shifts": velocity_shifts,
                                                                   "cross_corr": cross_corr,
                                                                   'best_rv': best_rv - bias[order_int, 0],
                                                                   "best_rv_err": np.sqrt(
                                                                       best_rv_err ** 2 + bias[order_int, 1] ** 2 +
                                                                       bias[
                                                                           order_int, 2] ** 2)}
                                order_rvs.append(best_rv - bias[order_int, 0])
                                order_rv_errs.append(
                                    np.sqrt(best_rv_err ** 2 + bias[order_int, 1] ** 2 + bias[order_int, 2] ** 2))
                            else:
                                # Store results (suborders)
                                if order_int in bias.keys():
                                    if suborder in bias[order_int]:
                                        rv_results[order_int][suborder] = {"velocity_shifts": velocity_shifts,
                                                                           "cross_corr": cross_corr,
                                                                           'best_rv': best_rv -
                                                                                      bias[order_int][suborder][0],
                                                                           "best_rv_err": np.sqrt(
                                                                               best_rv_err ** 2 +
                                                                               bias[order_int][suborder][1] ** 2 +
                                                                               bias[order_int][suborder][2] ** 2)}
                                        order_rvs.append(best_rv - bias[order_int][suborder][0])
                                        order_rv_errs.append(np.sqrt(
                                            best_rv_err ** 2 + bias[order_int][suborder][1] ** 2 +
                                            bias[order_int][suborder][2] ** 2))
                                    else:
                                        continue
                                else:
                                    continue

                        if order_int > 100:
                            plt.step(norm_wave, norm_flux, color="black", zorder=0)
                            temp_mask_wavelength = np.zeros(len(mask_wavelength) * 3)
                            temp_mask_strength = np.zeros(len(mask_wavelength) * 3)
                            for i, _ in enumerate(mask_wavelength):
                                temp_mask_wavelength[3 * i:3 * i + 2] = mask_wavelength[i]
                                temp_mask_strength[3 * i] = 1
                                temp_mask_strength[3 * i + 1] = 1 - mask_strength[i]
                                temp_mask_strength[3 * i + 2] = 1

                            plt.step(temp_mask_wavelength, temp_mask_strength, color="green", zorder=1)
                            plt.step(temp_mask_wavelength * (1 + best_rv / 3e5), temp_mask_strength, color="red",
                                     zorder=2)
                            plt.xlim(norm_wave.min(), norm_wave.max())
                            plt.title(f"{order} {best_rv} km/s")
                            plt.show()

                if 'best_rv' not in rv_results[order_int][suborder]:
                    rv_results.pop(order_int, None)

    # Compute overall best velocity and RMS scatter
    order_rvs = np.array(order_rvs)
    order_rv_errs = np.array(order_rv_errs)

    # Write all data to file
    write_order_results(rv_results, spectrum_file, sub_order, modified_date)

    sigma_val = 2.2
    # Sigma_val = 4
    bad_array = sigma_clip(order_rvs, sigma=sigma_val, maxiters=None, masked=True, copy=False)
    bad_index = bad_array.mask

    # Ensure bad_index is the same length as order_rvs
    try:
        if (len(bad_index) == len(order_rvs)) and (np.sum(bad_index) > 0):
            # Remove flagged items from arrays
            order_rvs = order_rvs[~bad_index]
            order_rv_errs = order_rv_errs[~bad_index]  # <- This was incorrectly using `order_rvs`

            # Remove corresponding dictionary entries
            # Convert keys to a sorted list so indexing aligns
            dict_keys_sorted = sorted(rv_results.keys())  # Ensure the keys align with array indices
            keys_to_remove = [key for i, key in enumerate(dict_keys_sorted) if bad_index[i]]

            for key in keys_to_remove:
                rv_results.pop(key, None)  # Remove the key safely
    except TypeError:
        pass

    mean_rv = np.average(order_rvs, weights=1 / order_rv_errs ** 2)
    mean_rv_err = np.sqrt(1 / np.sum(1 / order_rv_errs ** 2))
    rms_rv = np.sum((order_rvs - mean_rv) ** 2 / order_rv_errs ** 2) / np.sum(1 / order_rv_errs ** 2)

    if args.show_plots:
        # Plot radial velocity as function of order
        plt.figure(figsize=(8, 5))
        plt.errorbar(rv_results.keys(), order_rvs, yerr=order_rv_errs, fmt='o', capsize=4, label="Order RVs")
        plt.axhline(mean_rv, color='red', linestyle='--', label=f"Mean RV = {mean_rv:.2f} km/s")
        plt.xlabel("Spectral Order")
        plt.ylabel("Radial Velocity (km/s)")
        plt.title("Radial Velocity vs. Order")
        plt.legend()
        plt.grid()
        plt.show()

    # Print final results
    print(f"Overall Best Radial Velocity: {mean_rv:.2f} km/s")
    print(f"RMS of Orders: {rms_rv:.2f} km/s")

    results = {"file": spectrum_file, "mjd": mjd, "rv": mean_rv, "rv_err": mean_rv_err, "rv_rms": rms_rv}

    return results, gaia_properties


def main():
    parser = argparse.ArgumentParser(description="Process stellar spectra for radial velocity analysis.")
    parser.add_argument(
        "input_file",
        nargs="+",
        help="Input spectrum files or wildcard (e.g., '*.fits')."
    )
    parser.add_argument(
        "--no-bias",
        action="store_true",
        help="Run without bias corrections (usually to measure biases)."
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to files in a specified directory."
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots on the screen."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress warnings and output messages."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Output additional messages and plots."
    )
    parser.add_argument(
        "--output-file",
        default="output_summary.txt",
        help="Path to the output summary file."
    )
    parser.add_argument(
        "--teff",
        default="5800",
        help="Effective Temperature."
    )
    parser.add_argument(
        "--teff_tol",
        default="50",
        help="Effective Temperature."
    )
    parser.add_argument(
        "--suborder",
        default=1,
        help="Suborders."
    )
    parser.add_argument(
        "--spectral_type",
        default=None,
        help="Spectral Type"
    )
    parser.add_argument(
        "--telescope",
        default="espresso",
        help="Spectral Type"
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Telescope version (for NEIDS)"
    )
    parser.add_argument(
        "--args_list",
        default=None,
        help="Include list of args for each file, which can be specified after the argument."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files."
    )

    args = parser.parse_args()

    # Set logging level based on --quiet flag
    if args.quiet:
        suppress_warnings()
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    # Expand input file wildcards
    #    input_files = []
    #    for pattern in args.input_files:
    #        input_files.extend(Path().glob(pattern))
    #    input_files = sorted(set(input_files))  # Remove duplicates

    input_file = args.input_file[0]
    spectral_type = args.spectral_type
    telescope = args.telescope
    version = args.version
    suborder = int(args.suborder)

    if not input_file:
        logging.error("No file found. Please check the input pattern or file paths.")
        return

    logging.info(f"Found {len(input_file)} input files. Processing...")

    stellar_mask_files = glob.glob('stellar_masks/*')

    rv_results, gaia_properties = process_spectrum(input_file, spectral_type, telescope, version, suborder, args=args)

    #    # Process each spectrum file
    #    processed_data = []
    #    for i, file in enumerate(input_files):
    #        rv_results = find_best_standard_star(file, standard_star_files, float(args.teff), args)
    #        processed_data.append(rv_results)

    # Write summary
    if rv_results:
        output_file = write_summary(rv_results, gaia_properties)
    else:
        if gaia_properties:  # Gaia_properties acts as break_flag here
            print("Duplicate found. Overwrite off.")
        else:
            print("Spectrum file type not found.")
        return

    # Generate plots if requested
    if args.save_plots or args.show_plots:
        plot_filename = f"plots/Gaia_DR3_{gaia_properties["object_id"]}_plot.png"
        rv_list = []
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                else:
                    labels = ["file", "mjd", "rv", "rv_err", "rv_rms"]
                    processed_data = line.split()
                    combined = dict(zip(labels, processed_data))
                    rv_list.append(combined)

        plot_results(rv_list, gaia_properties, save=args.save_plots, save_path=plot_filename, show=args.show_plots)


#    # Save results to output file
#    with open(args.output_file, "w") as f:
#        f.write("#MJD,Radial_Velocity,Radial_Velocity_RMS,Correlation_Coefficient,Best_Standard\n")
#        for data in processed_data:
#            f.write(
#                f"{data['mjd']},{data['radial_velocity']},{data['radial_velocity_rms']},{data['correlation_coefficient']},{data['best_standard']}\n"
#            )

if __name__ == "__main__":
    main()
