"""Utility Functions."""

import numpy as np


def keep_angle_in_range(angle):
    """Keep angles within (0, 2 pi)."""
    while (angle < 0):
        angle += 2*np.pi
    while (angle > 2*np.pi):
        angle -= 2*np.pi
    return angle


def consistent_angles(a0, A, B, F, G, omega, Omega, inc):
    """Check to make sure angles are consistent."""
    if (-B-F)*np.sin(omega-Omega) < 0 or (B-F)*np.sin(omega+Omega) < 0:
        return False
    if np.abs((A+G)/a0 - (1+np.cos(inc)) * np.cos(omega+Omega)) > 1e-5:
        return False
    if np.abs((A-G)/a0 - (1-np.cos(inc)) * np.cos(omega-Omega)) > 1e-5:
        return False
    if np.abs((B-F)/a0 - (1+np.cos(inc)) * np.sin(omega+Omega)) > 1e-5:
        return False
    if np.abs((-B-F)/a0 - (1-np.cos(inc)) * np.sin(omega-Omega)) > 1e-5:
        return False

    return True
