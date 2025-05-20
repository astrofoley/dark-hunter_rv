import numpy as np
from astropy.io import fits
import os
import sys
import argparse
from astropy.stats import sigma_clip
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
import matplotlib.pyplot as plt



def outlier_mask(flux, max_iter=4, hi_sigma_threshold=4, lo_sigma_threshold=30, window_size=20, **kwargs):
    # Convert lists to numpy arrays
    flux = np.array(flux, dtype=np.float64)  # Ensure numerical stability

    mask = np.ones_like(flux, dtype=bool)  # Initialize all values as valid
    
    smooth_flux1 = np.full_like(flux, np.nan, dtype=np.float64)  # Initialize with NaNs
    smooth_flux2 = np.full_like(flux, np.nan, dtype=np.float64)  # Initialize with NaNs
    for i in range(len(flux)):
        # Define original window range
        start1, end1 = max(0, i - window_size // 2), min(len(flux), i + window_size // 2)
        start2, end2 = max(0, int(i - window_size * 5)), min(len(flux), int(i + window_size * 5))

        # Select only valid (masked) flux points within the window
        valid_points1 = mask[start1:end1]
        valid_points2 = mask[start2:end2]
        if np.any(valid_points1):  # Ensure we have valid points
            smooth_flux1[i] = np.median(flux[start1:end1][valid_points1])
        if np.any(valid_points2):  # Ensure we have valid points
            smooth_flux2[i] = np.median(flux[start2:end2][valid_points2])

    hi_clip_mask = sigma_clip(flux, sigma=hi_sigma_threshold, cenfunc='median', stdfunc='mad_std', grow=2, maxiters=max_iter, masked=True).mask & (flux > smooth_flux1)
    lo_clip_mask = sigma_clip(flux, sigma=lo_sigma_threshold, cenfunc='median', stdfunc='mad_std', grow=2, maxiters=max_iter, masked=True).mask & (flux < smooth_flux1) & ((smooth_flux1 > 0.7*smooth_flux2) | (flux < -0.5*np.nanmedian(flux)))
    mask[mask] &= ~hi_clip_mask & ~lo_clip_mask # Update mask in-place

#    x = np.linspace(0, len(flux), len(flux))
#    plt.plot(x, flux, color='dodgerblue')
#    plt.plot(x[~hi_clip_mask], flux[~hi_clip_mask], color='red')
#    plt.plot(x[mask], flux[mask], color='green')
#    plt.show()

    return mask

def absorption_mask(wavelength, flux, mask, max_iter=3, window_size=100, drop_threshold=0.4, sn_threshold=10, **kwargs):
    # Convert lists to numpy arrays
    wavelength = np.array(wavelength)
    flux = np.array(flux, dtype=np.float64)  # Ensure numerical stability

    for _ in range(max_iter):
        # **1. Apply Clipping for Cosmic Rays**
        masked_wavelength = wavelength[mask]
        masked_flux = flux[mask]

        # **2. Compute Smoothed Flux Using a Sliding Median (Respecting Mask)**
        smooth_flux = np.full_like(flux, np.nan, dtype=np.float64)  # Initialize with NaNs
        for i in range(len(flux)):
            # Define original window range
            start, end = max(0, i - window_size // 2), min(len(flux), i + window_size // 2)

            # Select only valid (masked) flux points within the window
            valid_points = mask[start:end]
            if np.any(valid_points):  # Ensure we have valid points
                smooth_flux[i] = np.median(flux[start:end][valid_points])

        # **3. Compute S/N Estimate**
        residuals = np.abs(flux - smooth_flux)
        smooth_residuals = np.full_like(flux, np.nan, dtype=np.float64)
        for i in range(len(flux)):
            start, end = max(0, i - window_size // 2), min(len(flux), i + window_size // 2)
            valid_points = mask[start:end]
            if np.any(valid_points):
                smooth_residuals[i] = np.median(residuals[start:end][valid_points])

        sn_estimate = np.where(smooth_residuals > 0, smooth_flux / smooth_residuals, np.inf)

        # **4. Exclude Low S/N Regions**
        valid_region = (sn_estimate > sn_threshold)
        
        # Check if there are at least 100 valid points above the S/N threshold
        if np.sum(valid_region) >= 100:
            first_valid = np.argmax(valid_region)
            last_valid = len(valid_region) - np.argmax(valid_region[::-1]) - 1

            valid_mask = np.zeros_like(flux, dtype=bool)
            valid_mask[first_valid:last_valid] = True  # Only keep interior region
            
            absorption_regions = (flux < (smooth_flux * drop_threshold)) & valid_mask
            mask &= ~absorption_regions  # Persist across iterations

        else:
            valid_mask = np.ones_like(flux, dtype=bool)  # No S/N threshold applied
    
    return mask

def fit_continuum(wavelength, flux, eflux, num_knots=7, order=2, sigma_clip_threshold=3, max_iter=5, **kwargs):
    """
    Fit the continuum of a spectrum, excluding regions with strong absorption features.
    
    Parameters:
        wavelength (array): Wavelength array.
        flux (array): Flux array.
        num_knots (int): Number of knots for the spline fit (more knots increase flexibility).
        order (int): Order of the spline.
        sigma_clip_threshold (float): Threshold for sigma clipping.
        max_iter (int): Maximum number of sigma-clipping iterations.
    
    Returns:
        continuum (array): Continuum fit to the spectrum.
    """
    # Convert lists to numpy arrays
    wavelength = np.array(wavelength)
    flux = np.array(flux, dtype=np.float64)  # Ensure numerical stability
    eflux = np.array(eflux, dtype=np.float64)  # Ensure numerical stability

    # Mask strong absorption lines
    mask_cr = outlier_mask(flux, max_iter=3)
    temp_mask = np.copy(mask_cr)
    mask_wide = absorption_mask(wavelength, flux, temp_mask, window_size=300, drop_threshold=0.95, max_iter=3, **kwargs)
    mask_medium = absorption_mask(wavelength, flux, temp_mask, window_size=100, drop_threshold=0.9, max_iter=3, **kwargs)
    mask_narrow = absorption_mask(wavelength, flux, temp_mask, window_size=20, drop_threshold=0.6, max_iter=3, **kwargs)
    mask = mask_cr & mask_wide & mask_medium & mask_narrow

    # Define knot positions for the spline fit, excluding endpoints
    knot_positions = np.linspace(wavelength[mask][0], wavelength[mask][-1], num_knots)[1:-1]
    
    # Ensure there are enough unique x-values for the spline
    if len(np.unique(wavelength[mask])) <= len(knot_positions):
        print(len(np.unique(wavelength[mask])), len(knot_positions))
        raise ValueError("Not enough unique wavelength values to satisfy spline conditions.")


    # Fit the spline to the clipped data
    spline = LSQUnivariateSpline(wavelength[mask], flux[mask], t=knot_positions, k=order)
    continuum = spline(wavelength)

    # Mask regions where the spline goes negative or very close to zero
    mask_spline = continuum > 0.15 * np.nanmedian(continuum)

    # Identify continuous regions of good (positive) continuum
    bad_indices = np.where(~mask_spline)[0]  # Indices where spline is invalid

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
        mask_good = mask_cr & mask_spline & anti_mask_spline
        #print(np.sum(mask_cr), np.sum(mask_good), np.sum(mask))
    else:
        mask_good = mask_cr

    if 10 < 0:
    #if kwargs["debug"]:
        # Plot the spectrum compared to what should be clipped
        plt.figure(figsize=(10, 6))
        plt.step(wavelength, flux, color='dodgerblue', label="Original Spectrum")
        plt.step(wavelength[mask_cr], flux[mask_cr], color='black', label="Clipped Spectrum")
        plt.plot(wavelength, continuum, linestyle='--', color='red', label="Fitted Continuum")
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        plt.title("Spectrum with Continuum")
        plt.legend()
        plt.show()
        
    norm_flux = flux / continuum
    norm_flux[~mask_good] = 1
    norm_eflux = eflux / continuum
    norm_eflux[~mask_good] = 1e30
    
    #if 10 < 0:
    if kwargs["debug"]:
        # Plot the spectrum compared to what should be clipped
        plt.figure(figsize=(10, 6))
        plt.step(wavelength, norm_flux, color='dodgerblue', label="Original Normalized Spectrum")
        plt.step(wavelength[mask_good], norm_flux[mask_good], color='black', label="Clipped Normalized Spectrum")
        plt.step(wavelength[mask], norm_flux[mask], color='green', label="Clipped Normalized Spectrum")
        plt.axhline(y=1, linestyle='--', color='red')
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")
        plt.ylim(-10, 30)
        plt.title("Spectrum with Continuum")
        plt.legend()
        plt.show()

    return norm_flux, norm_eflux

#def normalize_spectrum(wavelength, flux, continuum):
#    return flux / continuum

def bin_spectra(wave, flux, eflux, bin_size=2):
    binned_wave = []
    binned_flux = []
    binned_eflux = []
    
    for i in range(0, len(wave) - bin_size + 1, bin_size):
        delta_lambdas = np.diff(wave[i:i + bin_size + 1])
        delta_lambdas = np.append(delta_lambdas, delta_lambdas[-1])  # Handle last bin width
        
        # Compute total flux per bin
        total_fluxes = flux[i:i + bin_size] * delta_lambdas[:bin_size]
        total_flux = np.sum(total_fluxes)
        
        # New bin width
        new_delta_lambda = np.sum(delta_lambdas[:bin_size])
        
        # Flux conserving combination
        new_flux = total_flux / new_delta_lambda
        
        # Error propagation (scaling errors by bin widths)
        new_eflux = np.sqrt(np.sum((eflux[i:i + bin_size] * delta_lambdas[:bin_size]) ** 2)) / new_delta_lambda
        
        # Compute new wavelength as weighted mean
        new_wave = np.sum(wave[i:i + bin_size] * delta_lambdas[:bin_size]) / new_delta_lambda
        
        binned_wave.append(new_wave)
        binned_flux.append(new_flux)
        binned_eflux.append(new_eflux)
    
    return np.array(binned_wave), np.array(binned_flux), np.array(binned_eflux)

def parse_ascii_to_fits(input_file, **kwargs):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    orders = {}
    current_order = None
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('# Wavelength Flux'):
            continue
        
        if line.startswith('# Order'):
            current_order = int(line.split()[-1])
            orders[current_order] = []
            continue
        
        if current_order is not None:
            try:
                wave, flux = map(float, line.split(','))
                eflux = np.sqrt(abs(flux))  # Poisson error
                flux = np.random.normal(loc=0.0, scale=1e-1, size=1)+1
                eflux = 1e10
                orders[current_order].append((wave, flux, eflux))
            except ValueError:
                continue
    
    if kwargs["norm"] | (kwargs["bin_size"] != None):
        for o in range(len(orders)):
            wave = []
            flux = []
            eflux = []
            for i in range(len(orders[o])):
                wave.append(orders[o][i][0])
                flux.append(orders[o][i][1])
                eflux.append(orders[o][i][2])
            if kwargs["norm"]:
                flux, eflux = fit_continuum(wave, flux, eflux, **kwargs)
                if o in [34, 35]:
                    plt.step(wave, flux, color='black')
                    plt.axhline(1, color='red')
                    plt.show()
#                flux = normalize_spectrum(wave, flux, continuum)
            if kwargs["bin_size"] != None:
                wave, flux, eflux = bin_spectra(wave, flux, eflux, bin_size=kwargs["bin_size"])
        
            orders[o] = list(zip(*(wave, flux, eflux)))

    base_name = os.path.splitext(input_file)[0]
    
    for order, data in orders.items():
        data = np.array(data, dtype=[('wave', 'd'), ('flux', 'f4'), ('eflux', 'f4')])
        
        hdu_primary = fits.PrimaryHDU()
        hdu_table = fits.BinTableHDU.from_columns(data)
        
        hdu_table.header['XTENSION'] = 'BINTABLE'
        hdu_table.header['BITPIX'] = 8
        hdu_table.header['NAXIS'] = 2
        hdu_table.header['NAXIS1'] = 16
        hdu_table.header['NAXIS2'] = len(data)
        hdu_table.header['PCOUNT'] = 0
        hdu_table.header['GCOUNT'] = 1
        hdu_table.header['TFIELDS'] = 3
        hdu_table.header['TTYPE1'] = 'wave'
        hdu_table.header['TFORM1'] = 'D'
        hdu_table.header['TTYPE2'] = 'flux'
        hdu_table.header['TFORM2'] = 'E'
        hdu_table.header['TTYPE3'] = 'eflux'
        hdu_table.header['TFORM3'] = 'E'
        
        output_filename = f'{base_name}_order_{order}.fits'
        fits.HDUList([hdu_primary, hdu_table]).writeto(output_filename, overwrite=True)
        print(f'Saved: {output_filename}')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: format_spec.py <input_file>")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input ASCII file")  # Define as a positional argument
    parser.add_argument('--norm', '-n', help='Normalize the spectrum', action='store_true')
    parser.add_argument('--bin_size', '-b', help='Factor for rebinning data', type=int, default=2)
    parser.add_argument('--debug', '-d', help='Debug mode', action='store_true')

    args = parser.parse_args()

    # Call function without passing input_file twice
    parse_ascii_to_fits(args.input_file, **{k: v for k, v in vars(args).items() if k != "input_file"})
