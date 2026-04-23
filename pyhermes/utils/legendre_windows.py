"""Legendre multipole window kernels and window-array builders."""

import numpy as np
from numba import njit
from scipy.special import spherical_jn, sph_harm

from pyhermes.utils.legendre_fast import (
    calculate_fast_legendre_window_array_with_lm,
    has_fast_window_function,
)
from pyhermes.utils.special_functions import _phase_from_kR, spherical_harmonic_numba, spherical_jn_numba


def window_function_legendre_reference(ki, kj, kk, R, l, m):
    """Evaluate the Legendre multipole window with NumPy/SciPy for validation."""
    if abs(m) > l:
        return 0.0 + 0.0j

    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0.0:
        if l == 0 and m == 0:
            return 1.0 / np.sqrt(4.0 * np.pi) + 0.0j
        return 0.0 + 0.0j

    theta = np.arccos(np.clip(kk / k, -1.0, 1.0))
    phi = np.arctan2(kj, ki)
    phase = 2.0 * np.pi * k * R
    return spherical_jn(l, phase) * sph_harm(m, l, phi, theta)


@njit
def window_function_legendre_numba(ki, kj, kk, R, l, m):
    """Evaluate j_l(2*pi*|k|*R) * Y_l^m(khat) with Numba helpers."""
    if l < 0:
        return np.nan + 0.0j
    if abs(m) > l:
        return 0.0 + 0.0j

    phase = _phase_from_kR(ki, kj, kk, R)
    jl = spherical_jn_numba(l, phase)
    ylm = spherical_harmonic_numba(l, m, ki, kj, kk)
    return jl * ylm


@njit
def calculate_legendre_window_array_numba(L, DeltaXi, PowerPhi, rescaleR, l, m):
    """Build a complex FFT-space window array for one Legendre (l, m) mode."""
    window_array = np.zeros((L, L, L), dtype=np.complex128)
    for i in range(-L, L):
        pi = PowerPhi[abs(i)]
        for j in range(-L, L):
            pij = pi * PowerPhi[abs(j)]
            for k in range(-L, L):
                window_array[i, j, k] += (
                    pij
                    * PowerPhi[abs(k)]
                    * window_function_legendre_numba(i * DeltaXi, j * DeltaXi, k * DeltaXi, rescaleR, l, m)
                )
    return window_array


def calculate_legendre_window_array(L, DeltaXi, PowerPhi, rescaleR, l, m):
    """Build a Legendre window array, dispatching to generated fast kernels when possible."""
    if has_fast_window_function(l, m):
        return calculate_fast_legendre_window_array_with_lm(L, DeltaXi, PowerPhi, rescaleR, l, m)
    return calculate_legendre_window_array_numba(L, DeltaXi, PowerPhi, rescaleR, l, m)
