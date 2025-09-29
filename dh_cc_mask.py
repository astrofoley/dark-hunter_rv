# This program will read in a spectrum files, compare to a stellar mask, and determine the radial velocity and uncertainty for those files.

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from scipy.signal import correlate, savgol_filter, detrend, butter, filtfilt, medfilt

from scipy.optimize import curve_fit
from astropy.time import Time
from astropy.stats import sigma_clip
from scipy.stats import sem
from scipy.ndimage import median_filter

import glob
import os
import astropy.io.fits as fits



def suppress_warnings():
    """Suppress warnings if the --quiet flag is set."""
    warnings.filterwarnings("ignore")



def read_spectrum(filename):
    header = []
    spectrum_data = {}  # Dictionary to store data for each order
    current_order = None  # Track the current order

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


def read_bias(filename):
    bias = np.zeros((70,3))
    current_order = None  # Track the current order

    with open(filename, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()  # Remove leading and trailing whitespace
            if not line:  # Skip blank lines
                continue
            line = line.rstrip('\n')
            values = line.split()
            bias[int(values[0]), :] = [float(values[1]), float(values[2]), float(values[3])]
                
    return bias


def read_stellar_mask(spectral_type="G2", directory="stellar_masks/", telescope="espresso"):
    """
    Reads the stellar mask file for a given spectral type and telescope.

    Parameters:
        stellar_type (str): The spectral classification (e.g., "G2").
        telescope (str): The telescope name (e.g., "espresso").
        directory (str): The path to the directory containing mask files.

    Returns:
        dict: A dictionary with keys "wavelength" and "flux", each containing numpy arrays.
    """
    # Construct the filename
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
    rolling_min = np.array([np.percentile(flux[max(0, i-window_size//2):min(len(flux), i+window_size//2)], 10)
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
        plt.plot(wavelength, smooth_err*3)
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
    for line in header:
        if line.startswith("# THEMIDPT"):
            date_str = line.split('=')[1].strip()
            time = Time(date_str, format='isot', scale='utc')
            return time.mjd

    
    
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
    median_delta_logw = np.median(np.diff(log_obs_wavelength))/subpixel  # Approximate step size
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
    velocity_shifts = 299792.458 * (10**log_shifts - 1)

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
            bounds=((0,-np.inf,0), (np.inf,np.inf,np.inf))
        )
        perr = np.sqrt(np.diag(pcov))  # Standard deiation
        best_velocity_shift = popt[1]  # Gaussian center
        #best_velocity_err = popt[2]
        best_velocity_err = perr[1]
    except RuntimeError:
        best_velocity_shift = velocity_shifts[peak_index]  # Fallback to discrete peak
        best_velocity_err = 1e30

    if 10 < 0:
        print(best_velocity_shift, best_velocity_err, popt[2])
        plt.plot(velocity_shifts, cross_corr)
        plt.plot(velocity_shifts, gaussian(velocity_shifts, popt[0], popt[1], popt[2]))
        plt.show()

    return best_velocity_shift, best_velocity_err, velocity_shifts, cross_corr
    


def write_order_results(order_data, input_filename):
    """
    Writes order-level RV results, including skipped orders as NaN values.
    
    Parameters:
        order_data (dict): Dictionary where keys are order numbers and values are tuples (rv, rv_err).
        input_filename (str): Name of the input file to generate output file name.
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Derive output file name
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_file = os.path.join(output_dir, f"{base_name}_orders.txt")
    
    # Write data
    with open(output_file, "w") as f:
        f.write("# Order | RV (km/s) | RV Error (km/s)\n")
        for order in sorted(order_data.keys()):
            rv_str = f"{order_data[order]['best_rv']:.8f}" if np.isfinite(order_data[order]["best_rv"]) else "NaN"
            err_str = f"{order_data[order]['best_rv_err']:.8f}" if np.isfinite(order_data[order]["best_rv_err"]) else "NaN"
            f.write(f"{order} {rv_str} {err_str}\n")



def write_summary(processed_data):
    """
    Writes or updates the summary of processed spectra.
    
    Parameters:
        processed_data (list): List of dictionaries with processed data.
        summary_file (str): Path to the summary file.
    """
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Extract object ID from input file name
    base_name = os.path.splitext(os.path.basename(processed_data["file"]))[0]
    object_id = base_name.split("_")[2]  # Assumes filename structure

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
        f.write("# Input File | MJD | RV (km/s) | RV Error (km/s) | RMS (km/s)\n")
        for line in existing_entries.values():
            f.write(line + "\n")
            
    logging.info(f"Summary written to {output_file}")


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



def plot_results(processed_data, save=False, show=False, save_path=None):
    """
    Generates plots for processed spectra.
    
    Parameters:
        processed_data (list): List of dictionaries with processed data.
        save_plots (bool): Whether to save plots to files.
        show_plots (bool): Whether to display plots on-screen.
    """
    plt.figure(figsize=(8, 5))
    plt.title(f"Radial Velocity for XXX")

    for entry in processed_data:

        plt.errorbar(
            [entry["mjd"]],
            [entry["radial_velocity"]],
            yerr=[entry["radial_velocity_rms"]],
            fmt="o"
        )

    plt.xlabel("MJD")
    plt.ylabel("Radial Velocity (km/s)")

    if save and save_path != None:
        plot_dir = Path("plots")
        plot_dir.mkdir(exist_ok=True)
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()






def process_spectrum(spectrum_file, spectral_type="G2", args=None):
    """Main function to process a spectrum and compute radial velocities."""

    # Read spectrum and stellar mask
    header, spectrum_data = read_spectrum(spectrum_file)
    mask_data = read_stellar_mask(spectral_type=spectral_type)

    #Extract MJD
    mjd = extract_mjd_from_header(header)

    mask_wavelength = np.array(mask_data["wavelength"])
    mask_strength = np.array(mask_data["flux"])

    # Store cross-correlation results
    rv_results = {}

    # Order-by-order processing
    order_rvs = []
    order_rv_errs = []

    # List of orders to avoid
    bad_orders = [0,1,2,53,57,58,59,60,63,64,65]
    #bad_orders = [30, 31, 32, 43, 45]
    #43, 45,  everything <=25

    bias = np.zeros((70,3))

    if not args.no_bias:
        bias = read_bias("bias_statistics.txt")
        
    for order, data in spectrum_data.items():
        if order not in bad_orders:
#        if order in bad_orders:
            wavelength = np.array(data["wavelength"])
            flux = np.array(data["flux"])
            eflux = np.array(data["eflux"])

            if (wavelength[-1] > mask_wavelength[0]) & (wavelength[0] < mask_wavelength[-1]):
                # Clean and normalize spectrum
                mask_cr = outlier_mask(wavelength, flux)
                mask_abs = absorption_mask(wavelength, flux, mask_cr)
                norm_wave, norm_flux, _ = fit_continuum(wavelength, flux, eflux)

                if (np.sum((mask_wavelength < norm_wave[-1]) & (mask_wavelength > norm_wave[0])) > 10):
                    # Cross-correlate with stellar mask
                    if order > -1:

                        best_rv, best_rv_err, velocity_shifts, cross_corr = cross_correlate_stellar_mask(
                            norm_wave, 1 - norm_flux, mask_wavelength, mask_strength
                        )

                        # Store results
                        rv_results[order] = {"velocity_shifts": velocity_shifts, "cross_corr": cross_corr, "best_rv": best_rv-bias[order,0], "best_rv_err": np.sqrt(best_rv_err**2 + bias[order,1]**2 + bias[order,2]**2)}
                        order_rvs.append(best_rv-bias[order,0])
                        order_rv_errs.append(np.sqrt(best_rv_err**2 + bias[order,1]**2 + bias[order,2]**2))

                    
                    if order > 100:
                        plt.step(norm_wave, norm_flux, color="black", zorder=0)
                        temp_mask_wavelength = np.zeros(len(mask_wavelength)*3)
                        temp_mask_strength = np.zeros(len(mask_wavelength)*3)
                        for i, _ in enumerate(mask_wavelength):
                            temp_mask_wavelength[3*i:3*i+2] = mask_wavelength[i]
                            temp_mask_strength[3*i] = 1
                            temp_mask_strength[3*i+1] = 1-mask_strength[i]
                            temp_mask_strength[3*i+2] = 1

                        plt.step(temp_mask_wavelength, temp_mask_strength, color="green", zorder=1)
                        plt.step(temp_mask_wavelength*(1+best_rv/3e5), temp_mask_strength, color="red", zorder=2)
                        plt.xlim(norm_wave.min(), norm_wave.max())
                        plt.title(f"{order} {best_rv} km/s")
                        plt.show()
                        

    # Compute overall best velocity and RMS scatter
    order_rvs = np.array(order_rvs)
    order_rv_errs = np.array(order_rv_errs)
    
    # Write all data to file
    write_order_results(rv_results, spectrum_file)
    
    med_rv_err = np.median(order_rv_errs)
    bad_index = order_rv_errs > 3*med_rv_err
    
    # Ensure bad_index is the same length as order_rvs
    if len(bad_index) == len(order_rvs) and np.any(bad_index):
        # Remove flagged items from arrays
        order_rvs = order_rvs[~bad_index]
        order_rv_errs = order_rv_errs[~bad_index]  # <- This was incorrectly using `order_rvs`

        # Remove corresponding dictionary entries
        # Convert keys to a sorted list so indexing aligns
        dict_keys_sorted = sorted(rv_results.keys())  # Ensure the keys align with array indices
        keys_to_remove = [key for i, key in enumerate(dict_keys_sorted) if bad_index[i]]

        for key in keys_to_remove:
            rv_results.pop(key, None)  # Remove the key safely

    sigma_val = 2.2
    #sigma_val = 4
    bad_array = sigma_clip(order_rvs, sigma=sigma_val, maxiters=None, masked=True, copy=False)
    bad_index = bad_array.mask

    # Ensure bad_index is the same length as order_rvs
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

    mean_rv = np.average(order_rvs, weights=1/order_rv_errs**2)
    mean_rv_err = np.sqrt(1/np.sum(1/order_rv_errs**2))
    rms_rv = np.sum((order_rvs - mean_rv)**2 / order_rv_errs**2) / np.sum(1/order_rv_errs**2)

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


    return {"file": spectrum_file, "mjd": mjd, "rv": mean_rv, "rv_err": mean_rv_err, "rv_rms": rms_rv}





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

    if not input_file:
        logging.error("No file found. Please check the input pattern or file paths.")
        return

    logging.info(f"Found {len(input_file)} input files. Processing...")


    stellar_mask_files = glob.glob('stellar_masks/*')

    rv_results = process_spectrum(input_file, spectral_type="F9", args=args)

#    # Process each spectrum file
#    processed_data = []
#    for i, file in enumerate(input_files):
#        rv_results = find_best_standard_star(file, standard_star_files, float(args.teff), args)
#        processed_data.append(rv_results)

    # Write summary
    write_summary(rv_results)

    # Generate plots if requested
    if args.save_plots or args.show_plots:
        plot_filename = f"plots/{Path(input_file).stem}_plot.png"
        plot_results(rv_results, save=args.save_plots, save_path=plot_filename, show=args.show_plots)

#    # Save results to output file
#    with open(args.output_file, "w") as f:
#        f.write("#MJD,Radial_Velocity,Radial_Velocity_RMS,Correlation_Coefficient,Best_Standard\n")
#        for data in processed_data:
#            f.write(
#                f"{data['mjd']},{data['radial_velocity']},{data['radial_velocity_rms']},{data['correlation_coefficient']},{data['best_standard']}\n"
#            )

if __name__ == "__main__":
    main()




#TODO:

#CR clean the spectrum before CC?
#Sigma clip order RVs?
#Fit Gaussian width to get error for order - improve this
#Write to file
#Smaller chunks?
