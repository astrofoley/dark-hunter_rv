"""Fit orbits to Gaia data and RVs.

This program provides all the code necessary to fit orbits to Gaia astrometric
and possibly RV data.
"""

import numpy as np
from scipy.stats import multivariate_normal

from utils import keep_angle_in_range, consistent_angles

# First, load up constants
constG = 6.674e-8
Msun = 1.989e33
Rsun = 6.95e10
AU = 215 * Rsun
secday = 3600 * 24


def calc_E(e, M, n):
    """Calculate the eccentric anomaly.

    Inputs
    ------
    e : float
        Eccentricity
    M : float
        Mean anomaly
    n : float
        Binary's mean motion

    Returns
    -------
    E : float
        Eccentric anomaly

    """
    E = M
    if n > 0:
        E = M + e * np.sin(calc_E(e, M, n - 1))
    return E


def calc_rv(p, t_obs):
    """Calculate the radial velocities at some input times.

    Inputs
    ------
    p : tuple
        Tuple of input parameters
    t_obs : float
        Times to calculate the radial velocities

    Returns
    -------
    RV : float
        Radial velocity at time t=t_obs

    """
    # Unpack parameters
    ra, dec, parallax, pmra, pmdec, period, ecc, inc_deg, omega_deg, w_deg, \
        t_peri, v_com, m2, m1 = p

    # Convert input angles from degrees to radians
    inc = inc_deg * np.pi / 180
    omega = w_deg * np.pi / 180

    # Calculate the mean motion
    n = 2 * np.pi / period

    # Total mass
    M_tot = m1 + m2

    # Kepler's 3rd law
    a = ((period * secday / (2 * np.pi)) ** 2 * constG * M_tot * Msun) ** (1 / 3) / AU

    # Calculate mean anomaly, eccentric anomaly, true anomaly
    M = n * (t_obs - t_peri)
    E_set = calc_E(ecc, M, 20)
    beta = ecc / (1 + np.sqrt(1 - ecc ** 2))
    f_set = E_set + 2 * np.arctan((beta * np.sin(E_set)) / (1 - beta * np.cos(E_set)))

    # Finally, calculate RV terms
    mass_term = m2 / M_tot
    shape_term = 2 * np.pi * a * AU / (period * secday) / np.sqrt(1 - ecc ** 2) / 1e5
    angle_term = np.sin(inc) * (np.cos(omega + f_set) + ecc * np.cos(omega))

    return mass_term * shape_term * angle_term + v_com


def get_covariance_matrix(sys):
    """Calculate the covariance matrix of a particular Gaia binary.

    Inputs
    ------
    sys : numpy structured array
        Array containing astrometric data from Gaia

    Returns
    -------
    means : numpy array
        Array containing the Gaia observation means
    cov : numpy array
        2D array containing the covariance matrix of the Gaia observations

    """
    size = 12

    # From notes:
    # – ra
    # – dec
    # – parallax
    # – pmra
    # – pmdec
    # – a_thiele_innes
    # – b_thiele_innes
    # – f_thiele_innes
    # – g_thiele_innes (not fitted if bit_index = 8179)
    # – eccentricity (not fitted if bit_index = 8179)
    # – period
    # – t_periastron

    means = [sys['ra'], sys['dec'], sys['parallax'], sys['pmra'], sys['pmdec'],
             sys['a_thiele_innes'], sys['b_thiele_innes'],
             sys['f_thiele_innes'], sys['g_thiele_innes'], sys['eccentricity'],
             sys['period'], sys['t_periastron']]

    var_err = [1 / 10 * sys['ra_error'], 1 / 10 * sys['dec_error'],
               sys['parallax_error'], sys['pmra_error'], sys['pmdec_error'],
               sys['a_thiele_innes_error'], sys['b_thiele_innes_error'],
               sys['f_thiele_innes_error'], sys['g_thiele_innes_error'],
               sys['eccentricity_error'], sys['period_error'],
               sys['t_periastron_error']]

    # Turn correlation vector from string into list
    if isinstance(sys['corr_vec'], np.ndarray):
        corr_vec = sys['corr_vec']
    else:
        corr_vec = sys['corr_vec'][1:-1]
        corr_vec = np.array(corr_vec.split(',')).astype(float)

    # Create the correlation matrix
    corr = np.ones((size, size))

    N = 0
    for i in range(size):
        for j in range(size):

            # Only get off-diagonal terms
            if j >= i:
                continue

            corr[i, j] = corr_vec[N] * var_err[i] * var_err[j]
            corr[j, i] = corr_vec[N] * var_err[i] * var_err[j]

            N += 1

    # Now, add diagonal terms
    for i in range(size):
        corr[i, i] = var_err[i] ** 2

    return means, corr


def calc_ln_prior(p):
    """Calculate the prior for a model point p."""
    # Unpack parameters
    ra, dec, parallax, pmra, pmdec, period, ecc, inc_deg, omega_deg, w_deg, \
        t_peri, v_com, m2, m1 = p

    if ra < 0 or ra > 360:
        return -np.inf
    if dec < -90 or dec > 90:
        return -np.inf
    if pmra < -1e3 or pmra > 1e3:
        return -np.inf
    if pmdec < -1e3 or pmdec > 1e3:
        return -np.inf
    if period < 1 or period > 1e4:
        return -np.inf
    if ecc < 0 or ecc > 1:
        return -np.inf
    if inc_deg < 0 or inc_deg > 360:
        return -np.inf
    if omega_deg < 0 or omega_deg > 360:
        return -np.inf
    if w_deg < 0 or w_deg > 360:
        return -np.inf
    if t_peri < -1000 or t_peri > 3000:
        return -np.inf
    if v_com < -1e3 or v_com > 1e3:
        return -np.inf
    if m2 < 0 or m2 > 20:
        return -np.inf
    if m1 < 0 or m1 > 20:
        return -np.inf

    return 0


def calc_Thiele_Innes(p):
    """Calculate the Thiele-Innes parameters."""
    # Unpack parameters
    ra, dec, parallax, pmra, pmdec, period, ecc, inc_deg, omega_deg, w_deg, \
        t_peri, v_com, m2, m1 = p

    # Convert angles from degrees to radians
    inc = inc_deg * np.pi / 180
    Omega = omega_deg * np.pi / 180
    omega = w_deg * np.pi / 180

    # Calculate astrometric separation
    a_au = (((period * secday / (2 * np.pi)) ** 2 * constG * (m1 + m2) * Msun)) ** (1 / 3) / AU
    a_mas = a_au * parallax
    q = m2 / m1
    a0_mas = a_mas * q / (1 + q)

    # Calculate the Thiele-Innes parameters
    A = a0_mas * (np.cos(omega) * np.cos(Omega) -
                  np.sin(omega) * np.sin(Omega) * np.cos(inc))
    B = a0_mas * (np.cos(omega) * np.sin(Omega) +
                  np.sin(omega) * np.cos(Omega) * np.cos(inc))
    F = -a0_mas * (np.sin(omega) * np.cos(Omega) +
                   np.cos(omega) * np.sin(Omega) * np.cos(inc))
    G = -a0_mas * (np.sin(omega) * np.sin(Omega) -
                   np.cos(omega) * np.cos(Omega) * np.cos(inc))

    return A, B, F, G


def calc_ln_L_astrometry(p, astrometry_obs):
    """Calculate the log likelihood for just the astrometry."""
    # Unpack parameters
    ra, dec, parallax, pmra, pmdec, period, ecc, inc_deg, omega_deg, w_deg, \
        t_peri, v_com, m2, m1 = p

    # Calculate Thiele-Innes parameters
    A, B, F, G = calc_Thiele_Innes(p)

    # Create model array
    x_model = np.array([ra,
                        dec,
                        parallax,
                        pmra,
                        pmdec,
                        A, B, F, G,
                        ecc,
                        period,
                        t_peri])

    # Calculate log likelihood from multivariate
    L_astrometry = astrometry_obs.pdf(x_model)
    if L_astrometry <= 0:
        return -np.inf

    return np.log(L_astrometry)


def calc_ln_L_M1(p, m1_obs):
    """Calculate the log likelihood on luminous star mass."""
    # Unpack parameters
    ra, dec, parallax, pmra, pmdec, period, ecc, inc_deg, omega_deg, w_deg, \
        t_peri, v_com, m2, m1 = p

    # Calculate log likelihood
    L_m1 = m1_obs.pdf(m1)
    if L_m1 <= 0:
        return -np.inf

    return np.log(L_m1)


def calc_ln_L_gamma(p, gamma_obs):
    """Calculate the log likelihood on the center of mass velocity."""
    # Unpack parameters
    ra, dec, parallax, pmra, pmdec, period, ecc, inc_deg, omega_deg, w_deg, \
        t_peri, v_com, m2, m1 = p

    # Calculate log likelihood
    L_gamma = gamma_obs.pdf(v_com)
    if L_gamma <= 0:
        return -np.inf

    return np.log(L_gamma)


def calc_ln_L_rv(p, rv_obs):
    """Calculate the log likelihood contribution from RV measurements."""
    # Calculate the model rv's
    rv_model = calc_rv(p, rv_obs['t_obs'])

    # Calculate log likelihood
    ln_L_rv = np.sum(-0.5 * (rv_model - rv_obs['rv']) ** 2 / rv_obs['rv_err'] ** 2)

    if np.isnan(ln_L_rv):
        return -np.inf

    return ln_L_rv


def calc_ln_L(p, astrometry_obs, m1_obs, rv_obs=None):
    """Calculate the total log likelihood."""
    # The astrometric component
    ln_L_astrometry = 0
    if astrometry_obs is not None:
        ln_L_astrometry = calc_ln_L_astrometry(p, astrometry_obs)

    # The M1 component
    ln_L_M1 = calc_ln_L_M1(p, m1_obs)
    if np.isnan(ln_L_M1):
        return -np.inf

    # The radial velocity component
    ln_L_gamma = 0
    ln_L_rv = 0
    if rv_obs is not None:
        ln_L_rv = calc_ln_L_rv(p, rv_obs)
        if np.isnan(ln_L_rv):
            return -np.inf
    else:
        gamma_obs = multivariate_normal(mean=0.0, cov=10.0 ** 2)
        ln_L_gamma = calc_ln_L_gamma(p, gamma_obs)
        if np.isnan(ln_L_gamma):
            return -np.inf

    return ln_L_astrometry + ln_L_M1 + ln_L_gamma + ln_L_rv


def calc_ln_P(p, astrometry_obs, m1_obs, rv_obs=None):
    """Calculate the log posterior."""
    # Calculate the log prior
    ln_prior = calc_ln_prior(p)
    if np.isinf(ln_prior):
        return -np.inf

    # Calculate the log likelihood
    ln_L = calc_ln_L(p, astrometry_obs, m1_obs, rv_obs)
    if np.isinf(ln_L):
        return -np.inf

    return ln_prior + ln_L


def find_starting_orientation(means):
    """Find the starting orientation angles."""
    # First, grab the observed Thieles-Innes parameters
    A = means[5]
    B = means[6]
    F = means[7]
    G = means[8]

    # Calculate the angular orbital separation, a0
    u = (A * A + B * B + F * F + G * G) / 2
    v = A * G - B * F
    a0 = np.sqrt(u + np.sqrt(u * u - v * v))

    # Now solve for omega and Omega
    o_minus_O = np.arctan2((B + F), (G - A))
    o_plus_O = np.arctan2((B - F), (G + A))

    omega = 0.5 * (o_minus_O + o_plus_O)
    Omega = o_plus_O - omega

    inc = np.arccos(((A + G) / a0) / np.cos(omega + Omega) - 1)

    # Check for sign consistency
    if (-B - F) * (np.sin(omega - Omega)) < 0:
        o_minus_O += np.pi
        omega = 0.5 * (o_minus_O + o_plus_O)
        Omega = o_plus_O - omega
        inc = np.arccos(((A + G) / a0) / np.cos(omega + Omega) - 1)

    # Two solutions due to orbital orientation degeneracy
    omega_1 = omega
    omega_2 = omega + np.pi
    Omega_1 = Omega
    Omega_2 = Omega + np.pi

    omega_1 = keep_angle_in_range(omega_1)
    omega_2 = keep_angle_in_range(omega_2)
    Omega_1 = keep_angle_in_range(Omega_1)
    Omega_2 = keep_angle_in_range(Omega_2)

    # Check for consistency of solutions
    if not consistent_angles(a0, A, B, F, G, omega_1, Omega_1, inc):
        raise ValueError("First solution is bad")
    if not consistent_angles(a0, A, B, F, G, omega_2, Omega_2, inc):
        raise ValueError("First solution is bad")

    return omega_1, Omega_1, omega_2, Omega_2, inc
