# Import statements
import os
from datetime import datetime
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

from astroquery.gaia import Gaia

import matplotlib as mpl

from scipy.stats import multivariate_normal

import emcee
import corner

import warnings

# Formatting
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'serif'

SMALL_SIZE = 10
MEDIUM_SIZE = 11
BIGGER_SIZE = 12
mpl.rc('font', size=SMALL_SIZE)  # controls default text sizes
mpl.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
mpl.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
mpl.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
mpl.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import sys

sys.path.append("../")
from astrometric_orbit import get_covariance_matrix, find_starting_orientation, calc_ln_prior, calc_ln_P, \
    calc_ln_L_astrometry, calc_Thiele_Innes, calc_rv, calc_ln_L_rv

# Define constants
constG = 6.674e-8
Msun = 1.989e33
Rsun = 6.95e10
AU = 215 * Rsun
pctocm = 3.086e18
secday = 3600 * 24
clight = 2.998e10


def rad_to_deg(x):
    """
    Converts radians to degrees
    Parameters:
        x: In radians
    Returns:
         degrees: In degrees
    """
    degrees = 180 * x / np.pi
    return degrees


def overwrite(source_id, date_modified, flag):
    """
    Overwrite RV fits if the data has been recently modified
    Parameters:
        source_id: Gaia source_ID
        date_modified: The date the input data was last modified
        flag: Boolean, determines if the overwrite argument has been used
    Returns:
        overwrite: Boolean, false if overwrite skipped
    """
    paths_to_output = [f"./output/{source_id}/{source_id}_RV_fit_1.pdf", f"./output/{source_id}/{source_id}_RV_fit_1.png"]

    for path_to_output in paths_to_output:
        if os.path.exists(path_to_output):
            print(f"File last edited: {date_modified.strftime("%Y-%m-%d %H:%M:%S.%f")}")
            dir_modified = os.path.getmtime(path_to_output)
            dir_modified = datetime.fromtimestamp(dir_modified)
            if dir_modified < date_modified or flag is True:
                print("Overwriting...")
                return True
            else:
                print("Skipping overwrite.")
                return False
        else:
            continue

    print("Making output directory...")
    return True


def gaia_query(object):
    """
    Queries Gaia catalogue for object properties
    Parameters:
        object: Gaia source_id
    Returns:
        query_params: Dictionary of Gaia properties
    """
    query = f"""SELECT ra, dec, pmra, pmdec, parallax
            FROM gaiadr3.gaia_source
            WHERE source_id = {object}
            """
    job = Gaia.launch_job(query)
    results = job.get_results()

    query_params = {"ra":results["ra"], "dec":results["dec"], "pmra":results["pmra"], "pmdec":results["pmdec"], "parallax":results["parallax"]}
    return query_params


def orbital_parameters(source_id):
    """
    Gets the orbital parameters for a given source
    Parameters:
        source_id: Gaia source_ID
    Returns:
        sys_orbital: Tuple of orbital parameters from Gaia's Non-single star orbit catalogue
        sys_binary_mass: Tuple in an array from binary masses catalogue
    """
    sys_orbital = None

    try:
        query = f"""SELECT 
            TOP 10
            *
            FROM gaiadr3.nss_two_body_orbit
            WHERE source_id = {source_id}
            """
        job = Gaia.launch_job(query)
        results = job.get_results()
        sys_orbital = results[0]

    except IndexError:
        pass

    query = f"""SELECT *
                FROM gaiadr3.binary_masses
                WHERE source_id = {source_id}
                """
    job = Gaia.launch_job(query)
    sys_binary_mass = job.get_results()

    return sys_orbital, sys_binary_mass


def rv_file(input_file, skip):
    """
    Reads an RV summary file.
    Parameters:
        input_file: Text file with RV summary data
        skip: Boolean, skip RV summary data with one observation
    Returns:
        rv_obs: Array of tuples
    """
    dtype = [('filename', 'U60'), ('t_obs', 'f8'), ('rv', 'f8'), ('rv_err_stat', 'f8'), ('rv_err', 'f8')]
    rv_obs = np.genfromtxt(input_file, comments='#', dtype=dtype)  # Skip header based on number of comments

    rv_obs['t_obs'] -= 57388  # why?
    # To account for unmodeled systematics
    rv_obs['rv_err'] = 0.25  # RMS?

    if rv_obs['rv'].size <= 1 and skip:
        rv_obs = None

    return rv_obs


def gaia_position_params(sys_orbital, bin_mass, inc, Omega, omega, nwalkers, ndim, v_com_val, query_vals):
    """
    Define the initial position from Gaia Data
    Returns:
        p = Array including parameters
    """
    ra = sys_orbital['ra']
    dec = sys_orbital['dec']
    parallax = sys_orbital['parallax']
    pmra = sys_orbital['pmra']
    pmdec = sys_orbital['pmdec']
    period = sys_orbital['period']
    ecc = sys_orbital['eccentricity']
    inc_deg = rad_to_deg(inc)
    omega_deg = rad_to_deg(Omega)
    w_deg = rad_to_deg(omega)
    t_peri = sys_orbital['t_periastron']
    v_com = v_com_val
    m2 = bin_mass[1]  # 1.9
    m1 = bin_mass[0]  # 1.9

    random_dict = {"ra": [0, 360], "dec": [-90, 90], "parallax": [0, 800], "pmra": [-1e3, 1e3], "pmdec": [-1e3, 1e3],
                   "period": [1, 1e4],
                   "ecc": [0, 1], "inc_deg": [0, 360], "omega_deg": [0, 360], "w_deg": [0, 360],
                   "t_peri": [-1000, 3000], "v_com": [-1e3, 1e3],
                   "m2": [0, 20], "m1": [0, 20]}
    param_dict = {"ra": ra, "dec": dec, "parallax": parallax, "pmra": pmra, "pmdec": pmdec, "period": period,
                  "ecc": ecc, "inc_deg": inc_deg, "omega_deg": omega_deg, "w_deg": w_deg, "t_peri": t_peri,
                  "v_com": v_com,
                  "m2": m2, "m1": m1}

    for item in random_dict.keys():
        if np.isnan(param_dict[item]):
            if item in query_vals.keys() and ~np.isnan(query_vals[item]):
                param_dict[item] = query_vals[item]
            else:
                param_dict[item] = np.random.uniform(random_dict[item][0], random_dict[item][1])

    p = np.array([[param_dict["ra"], param_dict["dec"], param_dict["parallax"],
                   param_dict["pmra"], param_dict["pmdec"], param_dict["period"], param_dict["ecc"],
                   param_dict["inc_deg"], param_dict["omega_deg"], param_dict["w_deg"], \
                   param_dict["t_peri"], param_dict["v_com"], param_dict["m2"], param_dict["m1"]]])

    p0 = np.tile(p, (nwalkers, 1))
    p0 += 1.0e-5 * np.random.randn(nwalkers, ndim)

    p0 = np.nan_to_num(p0, nan=1)  # this is a change!

    return p0


def run_sampler(p0, nwalkers, ndim, burn_in, args):
    """
    Runs the MCMC sampler
        p0: Initial positions
        nwalkers: Number of walkers
        ndim: Number of dimensions in the parameter space
        burn_in: Bounds on chain steps
        args: Additional positional arguments for MCMC fit
    Returns:
        sampler: EnsembleSampler object
        chains: EnsembleSampler chains attribute
    """
    sampler = emcee.EnsembleSampler(nwalkers, ndim, calc_ln_P, args=args)

    sampler.run_mcmc(p0, 100)
    sampler.reset()  # Burn in

    sampler.run_mcmc(p0, 30000)

    if burn_in is not None:
        chains = sampler.chain[:, burn_in:, :]
    else:
        chains = sampler.chain[:, :, :]

    return sampler, chains


def compare_fits(chains):
    """
    Creates flatchain from chain
    """
    nwalkers, nsteps, ndim = chains.shape
    flatchain = chains.reshape((nwalkers * nsteps, ndim))
    half_sampler = flatchain[:, 5:]

    return half_sampler, flatchain


def rv_checker(sampler_1, sampler_2, rv_obs, source_id, extension):
    """
    Plots astrometry solutions and RV observations
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for j, i in enumerate(np.random.randint(len(sampler_1.flatchain), size=300)):
        model_times = np.linspace(np.min(rv_obs['t_obs']) - 200, np.max(rv_obs['t_obs']) + 200, 1000)
        model_t_in_yrs = model_times / 365.25 + 2016

        sample_1 = sampler_1.flatchain[i]
        sample_2 = sampler_2.flatchain[i]

        model_rv_1 = calc_rv(sample_1, model_times)  # + np.mean(rv_obs['rv'])
        model_rv_2 = calc_rv(sample_2, model_times)  # + np.mean(rv_obs['rv'])

        ax[0].plot(model_t_in_yrs, model_rv_1, color='c', alpha=0.02, label=None)
        ax[0].plot(model_t_in_yrs, model_rv_2, color='k', alpha=0.02, label=None)

        ax[1].plot((model_t_in_yrs - 2023) * 365.25, model_rv_1, color='c', alpha=0.02, label=None)
        ax[1].plot((model_t_in_yrs - 2023) * 365.25, model_rv_2, color='k', alpha=0.02, label=None)

    t_obs_in_yrs = rv_obs['t_obs'] / 365.25 + 2016
    ax[0].scatter(t_obs_in_yrs, rv_obs['rv'], color='r', zorder=10, s=15)
    ax[1].scatter((t_obs_in_yrs - 2023) * 365.25, rv_obs['rv'], color='r', zorder=10, s=15)

    custom_lines = [Line2D([0], [0], color='c', label='Astrometry only, Solution 1'),
                    Line2D([0], [0], color='k', label='Astrometry only, Solution 2'),
                    Line2D([0], [0], marker='o', color='w', label='APF Observations',
                           markerfacecolor='r', markersize=10)]

    ax[0].legend(handles=custom_lines)

    # ax[1].text()

    ax[0].set_ylim(-45, 45)
    ax[1].set_ylim(-20, 45)
    ax[0].set_xlim(np.min(model_t_in_yrs), np.max(model_t_in_yrs))
    ax[1].set_xlim(630, 800)
    x_ticks = np.array([244, 274, 305, 335, 365.25, 365.25 + 32, 365.25 + 60, 365.25 + 91]) + 365.25
    # x_ticks = np.array([121, 152, 182, 213, 244, 274, 305, 335, 365.25, 365.25+32]) + 365.25
    ax[1].set_xticks(x_ticks)
    # ax[1].set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'])
    ax[1].set_xticklabels(['Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr'])

    ax[0].set_xlabel('Time (year)')
    ax[1].set_xlabel('Time (in 2024)')
    ax[0].set_ylabel('Radial Velocity')

    fig.suptitle(f"Gaia DR3 {source_id} Astrometry Only")

    plt.tight_layout()

    plot_name = f"output/{source_id}/{source_id}_astrometry_solutions"  # Remove existing plots
    name_test = [f"{plot_name}.png", f"{plot_name}.pdf"]

    for name in name_test:
        if os.path.exists(name):
            os.remove(name)

    plt.savefig(f"{plot_name}.{extension}")

    plt.savefig(f"{plot_name}.{extension}")


def chain_plot(samplers, inputs, source_id, extension):
    """
    Plots Markov chains
    Parameters:
        samplers: list of EnsembleSampler objects
        inputs: Bounds on chain steps
        source_id: Gaia source_id
        extension: File extension for saved figure
    """
    fig, ax = plt.subplots(1, len(samplers), figsize=(12, 4))

    for sampler in samplers:
        index = samplers.index(sampler)
        if inputs[index] is not None:
            chains = sampler.chain[:, inputs[index]:, :]
        else:
            chains = sampler.chain[:, :, :]
        for i in range(32):
            ax[index].plot(chains[i, :, -2], color='k', alpha=0.2)
            ax[index].set_title(f"Sampler {index + 1}")

    fig.suptitle(f"Gaia DR3 {source_id} Chains")

    plot_name = f"output/{source_id}/{source_id}_chains"  # Remove existing plots
    name_test = [f"{plot_name}.png", f"{plot_name}.pdf"]

    for name in name_test:
        if os.path.exists(name):
            os.remove(name)

    plt.savefig(f"{plot_name}.{extension}")


def plot_err(x, x_err, ax):
    tmp_x = np.linspace(x - 3 * x_err, x + 3 * x_err, 100)
    y = 1 / np.sqrt(2 * np.pi * x_err ** 2) * np.exp(-(x - tmp_x) ** 2 / (2 * x_err ** 2))
    ax.plot(tmp_x, y, color='C0')


def corner_plots(sys_orbital, half_samplers, flatchains, labels, source_id, extension):
    """
    Plots corner plots
    """
    ranges = None

    fig = corner.corner(half_samplers[0], labels=labels, show_titles=True, plot_datapoints=False,
                        plot_density=False, color='c', range=ranges)
    corner.corner(half_samplers[1], labels=labels, show_titles=True, plot_datapoints=False,
                        plot_density=False, color='k', label_kwargs={'fontsize': 20}, range=ranges,
                        fig=fig)
    if half_samplers[2] is not None:
        corner.corner(half_samplers[2], labels=labels, show_titles=True, plot_datapoints=False,
                      plot_density=False, color='C0', label_kwargs={'fontsize': 20}, range=ranges,
                      fig=fig)
        corner.corner(half_samplers[3], labels=labels, show_titles=True, plot_datapoints=False,
                      plot_density=False, color='y', label_kwargs={'fontsize': 20}, range=ranges,
                      fig=fig)
        fig.axes[4].plot([], [], 'C0', label=r'$\rm Astrometry\ Only\ Solution $ 1', linewidth=2)
        fig.axes[4].plot([], [], 'y', label=r'$\rm Astrometry\ Only\ Solution $ 2', linewidth=2)

    fig.axes[4].plot([], [], 'c', label=r'$\rm Astrometry\ and\ APF\ RVs,\ Solution\ 1$', linewidth=2)
    fig.axes[4].plot([], [], 'k', label=r'$\rm Astrometry\ and\ APF\ RVs,\ Solution\ 2$', linewidth=2)
    fig.axes[4].legend(loc='upper right', frameon=False, fontsize=24)

    # Orbital Period
    for i in 9 * np.arange(9):
        fig.axes[i].set_xlim(875, 1225)

    # Eccentricity
    for i in [9]:
        fig.axes[i].set_ylim(0.0, 0.5)
    for i in 9 * np.arange(8) + 10:
        fig.axes[i].set_xlim(0.0, 0.5)

    # omega
    for i in [27, 28, 29]:
        fig.axes[i].set_ylim(150, 350)
    for i in 9 * np.arange(6) + 30:
        fig.axes[i].set_xlim(150, 350)

    # Omega
    for i in [36, 37, 38, 39]:
        fig.axes[i].set_ylim(60, 325)
    for i in 9 * np.arange(5) + 40:
        fig.axes[i].set_xlim(60, 325)

    # gamma
    for i in np.arange(6) + 54:
        fig.axes[i].set_ylim(-30, 50)
    for i in 9 * np.arange(3) + 60:
        fig.axes[i].set_xlim(-30, 50)

    fig.text(0.72, 0.97, "Posterior Predictive Checking", fontsize=14)

    spec2 = gridspec.GridSpec(ncols=4, nrows=5, figure=fig,
                              height_ratios=[1, 15, 1, 15, 50],
                              width_ratios=[50, 15, 1, 15])
    f2_ax1 = fig.add_subplot(spec2[1, 1])
    f2_ax2 = fig.add_subplot(spec2[1, 3])
    f2_ax3 = fig.add_subplot(spec2[3, 1])
    f2_ax4 = fig.add_subplot(spec2[3, 3])

    f2_ax1.set_xlabel('A (mas)')
    f2_ax2.set_xlabel('B (mas)')
    f2_ax3.set_xlabel('F (mas)')
    f2_ax4.set_xlabel('G (mas)')
    for a in [f2_ax1, f2_ax2, f2_ax3, f2_ax4]:
        a.set_yticklabels([])

    A = []
    B = []
    F = []
    G = []

    for j, i in enumerate(np.random.randint(len(flatchains[0]), size=30000)):
        p_sample = flatchains[0][i]

        A_sample, B_sample, F_sample, G_sample = calc_Thiele_Innes(p_sample)

        A.append(A_sample)
        B.append(B_sample)
        F.append(F_sample)
        G.append(G_sample)

    A = np.array(A)
    B = np.array(B)
    F = np.array(F)
    G = np.array(G)

    f2_ax1.hist(A, bins=50, density=True, alpha=0.5, color='k')
    f2_ax1.hist(A, bins=50, density=True, histtype='step', color='k')
    f2_ax2.hist(B, bins=50, density=True, alpha=0.5, color='k')
    f2_ax2.hist(B, bins=50, density=True, histtype='step', color='k')
    f2_ax3.hist(F, bins=50, density=True, alpha=0.5, color='k')
    f2_ax3.hist(F, bins=50, density=True, histtype='step', color='k')
    f2_ax4.hist(G, bins=50, density=True, alpha=0.5, color='k')
    f2_ax4.hist(G, bins=50, density=True, histtype='step', color='k')

    plot_err(sys_orbital['a_thiele_innes'], sys_orbital['a_thiele_innes_error'], f2_ax1)
    plot_err(sys_orbital['b_thiele_innes'], sys_orbital['b_thiele_innes_error'], f2_ax2)
    plot_err(sys_orbital['f_thiele_innes'], sys_orbital['f_thiele_innes_error'], f2_ax3)
    plot_err(sys_orbital['g_thiele_innes'], sys_orbital['g_thiele_innes_error'], f2_ax4)

    fig.suptitle(f"Gaia DR3 {source_id} Corner Plots")

    plot_name = f"output/{source_id}/{source_id}_corner_plots"  # Remove existing plots
    name_test = [f"{plot_name}.png", f"{plot_name}.pdf"]

    for name in name_test:
        if os.path.exists(name):
            os.remove(name)

    plt.savefig(f"{plot_name}.{extension}")


def rv_plots(flatchains, rv_obs, source_id, sampler_int, astrometry, extension):
    """
    Plots solutions with astrometry and RV measurements
    Parameters:
        flatchains: List storing MCMC solutions
        rv_obs: Array of APF RV measurements
        source_id: Gaia source_id
        sampler_int: Integer representing which solution to plot
        astrometry: Multivariate normal object or None
        extension: File extension for saved figure
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    if astrometry:
        for j, i in enumerate(np.random.randint(len(flatchains[2]), size=300)):
            model_times = np.linspace(np.min(rv_obs['t_obs']) - 200, np.max(rv_obs['t_obs']) + 200, 1000)
            model_t_in_yrs = model_times / 365.25 + 2016

            if sampler_int == 1:
                sample_1 = flatchains[2][i]
            if sampler_int == 2:
                sample_1 = flatchains[3][i]

            model_rv_1 = calc_rv(sample_1, model_times)  # RV from astrometry

            ax[0].plot(model_t_in_yrs, model_rv_1, color='c', alpha=0.02, label=None)

            ax[1].plot((model_t_in_yrs - 2023) * 365.25, model_rv_1, color='c', alpha=0.02, label=None)

    for j, i in enumerate(np.random.randint(len(flatchains[1]), size=300)):
        model_times = np.linspace(np.min(rv_obs['t_obs']) - 200, np.max(rv_obs['t_obs']) + 200, 1000)
        model_t_in_yrs = model_times / 365.25 + 2016

        # sample_1 = sampler_1.flatchain[i]
        if sampler_int == 1:
            sample_2 = flatchains[0][i]
        if sampler_int == 2:
            sample_2 = flatchains[1][i]

        model_rv_2 = calc_rv(sample_2, model_times)  # RV from RV measurements and astrometry

        ax[0].plot(model_t_in_yrs, model_rv_2, color='k', alpha=0.02, label=None)

        ax[1].plot((model_t_in_yrs - 2023) * 365.25, model_rv_2, color='k', alpha=0.02, label=None)

    t_obs_in_yrs = rv_obs['t_obs'] / 365.25 + 2016
    ax[0].scatter(t_obs_in_yrs, rv_obs['rv'], color='r', zorder=10)
    ax[1].scatter((t_obs_in_yrs - 2023) * 365.25, rv_obs['rv'], color='r', zorder=10)

    custom_lines = [Line2D([0], [0], color='c', label=f'Astrometry only, Solution {sampler_int}'),
                    Line2D([0], [0], color='k', label='Astrometry and APF RVs'),
                    Line2D([0], [0], marker='o', color='w', label='APF Observations',
                           markerfacecolor='r', markersize=10)]

    ax[0].legend(handles=custom_lines)


    ax[0].set_ylim(-70, 70)
    ax[1].set_ylim(-70, 70)
    ax[0].set_xlim(np.min(model_t_in_yrs), np.max(model_t_in_yrs))
    ax[1].set_xlim(650, 820)
    x_ticks = np.array(
        [274, 305, 335, 366.25, 366.25 + 31, 366.25 + 59, 366.25 + 90, 366.25 + 121, 366.25 + 152]) + 365.25
    ax[1].set_xticks(x_ticks)
    ax[1].set_xticklabels(['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])

    ax[0].set_xlabel('Time (year)')
    ax[1].set_xlabel('Time (2024-2025)')
    ax[0].set_ylabel('Radial Velocity (km/s)')

    fig.suptitle(f"Gaia DR3 {source_id} RV Fit")

    plot_name = f"output/{source_id}/{source_id}_RV_fit_{sampler_int}"  # Remove existing plots
    name_test = [f"{plot_name }.png", f"{plot_name }.pdf"]

    for name in name_test:
        if os.path.exists(name):
            os.remove(name)

    plt.tight_layout()
    plt.savefig(f"{plot_name }.{extension}")


def skip_astro(sys, final_statement):
    means = [sys['ra'], sys['dec'], sys['parallax'], sys['pmra'], sys['pmdec'],
             sys['a_thiele_innes'], sys['b_thiele_innes'],
             sys['f_thiele_innes'], sys['g_thiele_innes'], sys['eccentricity'],
             sys['period'], sys['t_periastron']]
    astrometry_obs = None
    final_statement = final_statement + "Skipped astrometry. "

    return means, astrometry_obs, final_statement


def run_rv_fit(input, args=None):
    os.makedirs("output", exist_ok=True)

    if args.skip:
        return "File skipped."

    final_statement = ""

    last_modified = os.path.getmtime(input)
    last_modified = datetime.fromtimestamp(last_modified)

    source_id = int(input.split('/')[-1].split('_')[0])

    if args.overwrite:
        run_over = True
    else:
        run_over = False

    if args.save_pdfs:
        extension = "pdf"
    else:
        extension = "png"

    overwrite_flag = overwrite(source_id, last_modified, run_over)

    if overwrite_flag:
        pass
    else:
        return "Duplicate RV fits found. Not overwritten."

    burn_in = int(args.burn_in)

    sys_orbital, sys_binary_mass = orbital_parameters(source_id)

    if sys_orbital is None:
        return "Not in NSS Two Body Orbit catalogue. File skipped."

    rv_obs = rv_file(input, args.skip_ones)
    if rv_obs is None:
        final_statement = final_statement + "One or less RV observations. Skipping RV fit. "

    query_vals = gaia_query(source_id)

    try:
        m1_bin = sys_binary_mass[['m1', 'm1_lower', 'm1_upper']][0][0]
        m2_bin = sys_binary_mass[['m2', 'm2_lower', 'm2_upper']][0][0]
    except IndexError:
        m1_bin = np.random.uniform(1.1, 2)
        m2_bin = np.random.uniform(1.1, 2)

    if m1_bin > 1.1:
        m1_bin = sys_binary_mass[['m1', 'm1_lower', 'm1_upper']][0][2]
    if m2_bin > 1.1:
        m2_bin = sys_binary_mass[['m2', 'm2_lower', 'm2_upper']][0][2]

    if args.mass1:
        m1_bin = float(args.mass1)
    if args.mass2:
        m2_bin = float(args.mass2)

    masses = [m1_bin, m2_bin]

    # Define astrometry
    if not args.skip_astrometry:
        try:
            # Get covariance matrix
            means, cov = get_covariance_matrix(sys_orbital)
            try:
                astrometry_obs = multivariate_normal(mean=means, cov=cov)
            except Exception:
                astrometry_obs = None
                final_statement = final_statement + "Covariance matrix not positive definite. Skipped astrometry. "

        except Exception:
            means, astrometry_obs, final_statement = skip_astro(sys_orbital, final_statement)
    else:
        means, astrometry_obs, final_statement = skip_astro(sys_orbital, final_statement)

    # Define model for m1
    m1_obs = multivariate_normal(mean=masses[0], cov=0.1 ** 2)

    # Set up emcee
    ndim = 14
    nwalkers = int(args.walkers)  # 32
    v_com = 10
    omega_1, Omega_1, omega_2, Omega_2, inc = find_starting_orientation(means)  # Rotation elements

    if astrometry_obs is None and rv_obs is None:
        final_statement = final_statement + "Missing both astrometry and RV measurements. File skipped. "

    # Solution 1
    if astrometry_obs:
        p0_1 = gaia_position_params(sys_orbital, masses, inc, Omega_1, omega_1, nwalkers, ndim, v_com, query_vals)

        sampler_astro_1, chains_astro_1 = run_sampler(p0_1, nwalkers, ndim, burn_in=-10000,
                                                      args=[astrometry_obs, m1_obs])

        # Solution 2
        p0_2 = gaia_position_params(sys_orbital, masses, inc, Omega_2, omega_2, nwalkers, ndim, v_com, query_vals)

        sampler_astro_2, chains_astro_2 = run_sampler(p0_2, nwalkers, ndim, burn_in=-10000,
                                                      args=[astrometry_obs, m1_obs])
        # Compare the fits
        half_sampler_astro_1, flatchain_astro_1 = compare_fits(chains_astro_1)
        half_sampler_astro_2, flatchain_astro_2 = compare_fits(chains_astro_2)

        os.makedirs(f"output/{source_id}", exist_ok=True)
        rv_checker(sampler_astro_1, sampler_astro_2, rv_obs, source_id, extension)

    else:
        half_sampler_astro_1, flatchain_astro_1 = None, None
        half_sampler_astro_2, flatchain_astro_2 = None, None

    labels = [r'$P_{\rm orb}\,\,[\rm days]$', r'$\rm ecc$', r'$\rm inc\,\,[\rm deg]$', r'$\omega\,\,[\rm deg]$',
              r'$\Omega\,\,[\rm deg]$', r'$T_{p}\,\,[\rm days]$', r'$\gamma\,\,[\rm km\,s^{-1}]$',
              r'$M_2\,\,[M_{\odot}]$', r'$M_{\star}\,\,[M_{\odot}]$']

    # APF RVs
    if rv_obs is not None:
        v_com = np.mean(rv_obs['rv'])
        omega_1, Omega_1, omega_2, Omega_2, inc = find_starting_orientation(means)

        # Solution 1
        p0_1 = gaia_position_params(sys_orbital, masses, inc, Omega_1, omega_1, nwalkers, ndim, v_com, query_vals)
        sampler_1, chains_1 = run_sampler(p0_1, nwalkers, ndim, burn_in=burn_in,
                                          args=[astrometry_obs, m1_obs, rv_obs])

        # Solution 2
        p0_2 = gaia_position_params(sys_orbital, masses, inc, Omega_2, omega_2, nwalkers, ndim, v_com, query_vals)
        sampler_2, chains_2 = run_sampler(p0_2, nwalkers, ndim, burn_in=burn_in,
                                          args=[astrometry_obs, m1_obs, rv_obs])

        samplers = [sampler_1, sampler_2]
        c_inputs = [None, 6000]
        os.makedirs(f"output/{source_id}", exist_ok=True)
        chain_plot(samplers, c_inputs, source_id, extension)

        half_sampler_1, flatchain_1 = compare_fits(chains_1)
        half_sampler_2, flatchain_2 = compare_fits(chains_2)
    else:
        half_sampler_1, flatchain_1 = None, None
        half_sampler_2, flatchain_2 = None, None

    half_samplers = [half_sampler_1, half_sampler_2, half_sampler_astro_1, half_sampler_astro_2]
    flatchains = [flatchain_1, flatchain_2, flatchain_astro_1, flatchain_astro_2]

    os.makedirs(f"output/{source_id}", exist_ok=True)

    try:
        corner_plots(sys_orbital, half_samplers, flatchains, labels, source_id, extension)
    except Exception:
        final_statement = final_statement + "Failed to generate corner plots. "

    if rv_obs is not None:
        rv_plots(flatchains, rv_obs, source_id, 1, astrometry_obs, extension)
        rv_plots(flatchains, rv_obs, source_id, 2, astrometry_obs, extension)
        final_statement = final_statement + "RV fits successfully generated."

    return final_statement


def main():
    parser = argparse.ArgumentParser(description="Generate RV fits.")
    parser.add_argument(
        "input_file",
        nargs="+",
        help="Input spectrum files or wildcard (e.g., '*.fits')."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files."
    )
    parser.add_argument(
        "--args_list",
        default=None,
        help="Include list of args for each file, which can be specified after the argument."
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip file."
    )
    parser.add_argument(
        "--walkers",
        default=64,
        help="Choose number of walkers for MCMC."
    )
    parser.add_argument(
        "--mass1",
        default=None,
        help="Custom m1."
    )
    parser.add_argument(
        "--mass2",
        default=None,
        help="Custom m2."
    )
    parser.add_argument(
        "--skip-ones",
        action="store_true",
        help="Skip RV files with only one measurement."
    )
    parser.add_argument(
        "--quiet-runtime",
        action="store_true",
        help="Allow fits with runtime warnings."
    )
    parser.add_argument(
        "--skip-astrometry",
        action="store_true",
        help="Exclude astrometry from fit."
    )
    parser.add_argument(
        "--save-pdfs",
        action="store_true",
        help="Save PDFs instead of PNGs."
    )
    parser.add_argument(
        "--burn-in",
        default=10000,
        help="Set the burn-in."
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Path to RV file directory."
    )

    args = parser.parse_args()

    if not args.quiet_runtime:
        warnings.filterwarnings("error", category=RuntimeWarning)

    input_file = args.input_file[0]

    try:
        final_statement = run_rv_fit(input_file, args=args)
    except Exception as e:
        final_statement = f"Failed to converge. Error: {e}"

    log = f"{input_file}: {final_statement} \n"
    print(log)

    line_found = False

    if os.path.exists("error_log.txt"):
        with open("error_log.txt", "r") as f:
            lines = f.readlines()
    else:
        with open("error_log.txt", "w") as f:
            lines = []
            pass

    with open("error_log.txt", "w") as file:
        for line in lines:
            if input_file not in line:
                file.write(line)
            else:
                file.write(log)
                line_found = True
        if not line_found:
            file.write(f"{input_file}: {final_statement} \n")


if __name__ == "__main__":
    main()
