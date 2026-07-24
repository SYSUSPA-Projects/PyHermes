"""Legendre multipole window kernels and window-array builders."""

import numpy as np
from numba import njit, prange
from scipy.special import sph_harm_y, spherical_jn

from pyhermes.utils.legendre_fast import (
    calculate_fast_legendre_window_array_with_lm,
    has_fast_window_function,
)
from pyhermes.utils.special_functions import (
    _phase_from_kR,
    spherical_harmonic_numba,
    spherical_jn_numba,
)


def window_function_legendre_reference(ki, kj, kk, R, l, m):
    """
    Evaluate one Legendre multipole window value with NumPy/SciPy.

    This is a reference implementation for validation and diagnostics. It is
    not used by the production window-array builders, which use either Numba
    generic kernels or explicit fast kernels.
    """
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
    return spherical_jn(l, phase) * sph_harm_y(l, m, theta, phi)


@njit
def window_function_legendre_numba(ki, kj, kk, R, l, m):
    """
    Evaluate one generic Legendre multipole window value with Numba helpers.

    The returned value is ``j_l(2*pi*|k|*R) * Y_l^m(khat)``. This generic path
    supports arbitrary valid ``(l, m)`` values, but is slower than the explicit
    kernels in ``legendre_fast`` for the supported low-order modes.
    """
    if l < 0:
        return np.nan + 0.0j
    if abs(m) > l:
        return 0.0 + 0.0j

    phase = _phase_from_kR(ki, kj, kk, R)
    jl = spherical_jn_numba(l, phase)
    ylm = spherical_harmonic_numba(l, m, ki, kj, kk)
    return jl * ylm


@njit(parallel=True)
def calculate_legendre_window_array_numba(L, phi_fourier_power, rescaleR, l, m):
    """
    Build a complex full-FFT Legendre window array with the generic kernel.

    The array has shape ``(L, L, L)`` and is intended for complex full-FFT
    convolution. It intentionally mirrors
    ``calculate_fast_legendre_window_array_numba`` while calling the generic
    ``window_function_legendre_numba`` directly; keeping the kernels separate
    avoids dispatch overhead inside the ``O(L^3)`` loop.
    """
    window_array = np.zeros((L, L, L), dtype=np.complex128)
    inv_L = 1.0 / L
    for x in prange(L):
        i0 = x - L
        i1 = x
        pi0 = phi_fourier_power[L - x]
        pi1 = phi_fourier_power[x]
        ki0 = i0 * inv_L
        ki1 = i1 * inv_L
        for y in range(L):
            j0 = y - L
            j1 = y
            pj0 = phi_fourier_power[L - y]
            pj1 = phi_fourier_power[y]
            kj0 = j0 * inv_L
            kj1 = j1 * inv_L
            for z in range(L):
                k0 = z - L
                k1 = z
                pk0 = phi_fourier_power[L - z]
                pk1 = phi_fourier_power[z]
                kk0 = k0 * inv_L
                kk1 = k1 * inv_L
                window_array[x, y, z] = (
                    pi0 * pj0 * pk0 * window_function_legendre_numba(ki0, kj0, kk0, rescaleR, l, m)
                    + pi0 * pj0 * pk1 * window_function_legendre_numba(ki0, kj0, kk1, rescaleR, l, m)
                    + pi0 * pj1 * pk0 * window_function_legendre_numba(ki0, kj1, kk0, rescaleR, l, m)
                    + pi0 * pj1 * pk1 * window_function_legendre_numba(ki0, kj1, kk1, rescaleR, l, m)
                    + pi1 * pj0 * pk0 * window_function_legendre_numba(ki1, kj0, kk0, rescaleR, l, m)
                    + pi1 * pj0 * pk1 * window_function_legendre_numba(ki1, kj0, kk1, rescaleR, l, m)
                    + pi1 * pj1 * pk0 * window_function_legendre_numba(ki1, kj1, kk0, rescaleR, l, m)
                    + pi1 * pj1 * pk1 * window_function_legendre_numba(ki1, kj1, kk1, rescaleR, l, m)
                )
    return window_array


def calculate_legendre_window_array(L, phi_fourier_power, rescaleR, l, m, use_fast=True):
    """
    Build a production Legendre window array for one ``(l, m)`` mode.

    By default, supported low-order modes are sent to the explicit fast kernels
    in ``legendre_fast``; all other modes use the generic Numba implementation.
    Set ``use_fast=False`` to force the generic path, which is useful for
    validation and backend comparisons.
    """
    if use_fast and has_fast_window_function(l, m):
        return calculate_fast_legendre_window_array_with_lm(L, phi_fourier_power, rescaleR, l, m)
    return calculate_legendre_window_array_numba(L, phi_fourier_power, rescaleR, l, m)
