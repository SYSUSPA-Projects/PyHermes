"""General radial-profile multipole window kernels."""

import math

import numpy as np
from numba import njit, prange

from pyhermes.utils.special_functions import spherical_harmonic_numba, spherical_jn_numba


RADIAL_PROFILE_CODES = {
    "shell": 0,
    "gaussian_shell": 1,
    "thick_shell": 2,
    "sphere": 3,
    "gaussian": 4,
}


def _require_arg(len_args, name, radial_type):
    if name not in len_args:
        raise ValueError(f"radial_multipole profile '{radial_type}' requires len_args['{name}'].")
    return float(len_args[name])


def radial_profile_args(radial_type, len_args):
    """Return a compact numeric argument array for one radial profile."""
    radial_type = str(radial_type).strip().lower()
    if radial_type not in RADIAL_PROFILE_CODES:
        supported = ", ".join(sorted(RADIAL_PROFILE_CODES))
        raise ValueError(f"Unsupported radial multipole profile '{radial_type}'. Supported profiles: {supported}.")

    args = np.zeros(4, dtype=np.float64)
    if radial_type == "shell":
        args[0] = _require_arg(len_args, "R", radial_type)
    elif radial_type == "gaussian_shell":
        args[0] = _require_arg(len_args, "R_shell", radial_type)
        args[1] = _require_arg(len_args, "R_smooth", radial_type)
    elif radial_type == "thick_shell":
        args[0] = _require_arg(len_args, "R", radial_type)
        args[1] = _require_arg(len_args, "delta_R", radial_type)
        if args[1] <= 0.0:
            raise ValueError("radial_multipole profile 'thick_shell' requires delta_R > 0.")
    elif radial_type == "sphere":
        args[0] = _require_arg(len_args, "R", radial_type)
    elif radial_type == "gaussian":
        args[0] = _require_arg(len_args, "R", radial_type)
    return RADIAL_PROFILE_CODES[radial_type], args


@njit
def _sinc(q):
    if q == 0.0:
        return 1.0
    return math.sin(q) / q


@njit
def _sphere_profile(k, radius):
    q = 2.0 * math.pi * k * radius
    if q == 0.0:
        return 1.0
    return 3.0 * (math.sin(q) - q * math.cos(q)) / (q * q * q)


@njit
def _gaussian_shell_profile(k, radius, smooth_radius):
    if k == 0.0:
        return 1.0
    q_shell = 2.0 * math.pi * k * radius
    q_smooth = 2.0 * math.pi * k * smooth_radius
    denom = radius * radius + smooth_radius * smooth_radius
    if denom == 0.0:
        return 1.0
    return (
        (smooth_radius * smooth_radius * math.cos(q_shell) + radius * radius * _sinc(q_shell))
        / denom
        * math.exp(-0.5 * q_smooth * q_smooth)
    )


@njit
def _thick_shell_profile(k, radius, width):
    r_inner = radius - 0.5 * width
    if r_inner < 0.0:
        r_inner = 0.0
    r_outer = radius + 0.5 * width
    denom = r_outer * r_outer * r_outer - r_inner * r_inner * r_inner
    if denom == 0.0:
        return 1.0
    return (
        r_outer * r_outer * r_outer * _sphere_profile(k, r_outer)
        - r_inner * r_inner * r_inner * _sphere_profile(k, r_inner)
    ) / denom


@njit
def _radial_profile(k, radial_profile_code, args, l):
    if radial_profile_code == 0:
        return spherical_jn_numba(l, 2.0 * math.pi * k * args[0])
    if radial_profile_code == 1:
        return _gaussian_shell_profile(k, args[0], args[1])
    if radial_profile_code == 2:
        return _thick_shell_profile(k, args[0], args[1])
    if radial_profile_code == 3:
        return _sphere_profile(k, args[0])
    if radial_profile_code == 4:
        q = 2.0 * math.pi * k * args[0]
        return math.exp(-0.5 * q * q)
    return math.nan


@njit
def window_function_radial_multipole_numba(ki, kj, kk, radial_profile_code, args, l, m):
    """
    Evaluate ``U_l(k) * Y_l^m(khat)`` for one k-vector.

    For ``shell`` the radial response is the exact thin-shell response
    ``j_l(2*pi*k*R)``. Other built-in profiles are interpreted as direct
    k-space radial profiles ``U_l(k)`` and are therefore primarily intended
    for monopole/radial-binning scans.
    """
    if l < 0:
        return math.nan + 0.0j
    if abs(m) > l:
        return 0.0 + 0.0j
    k = math.sqrt(ki * ki + kj * kj + kk * kk)
    radial = _radial_profile(k, radial_profile_code, args, l)
    angular = spherical_harmonic_numba(l, m, ki, kj, kk)
    return radial * angular


@njit(parallel=True)
def calculate_radial_multipole_window_array_numba(L, phi_fourier_power, radial_profile_code, args, l, m):
    """Build a complex full-FFT radial multipole window array."""
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
                    pi0 * pj0 * pk0 * window_function_radial_multipole_numba(ki0, kj0, kk0, radial_profile_code, args, l, m)
                    + pi0 * pj0 * pk1 * window_function_radial_multipole_numba(ki0, kj0, kk1, radial_profile_code, args, l, m)
                    + pi0 * pj1 * pk0 * window_function_radial_multipole_numba(ki0, kj1, kk0, radial_profile_code, args, l, m)
                    + pi0 * pj1 * pk1 * window_function_radial_multipole_numba(ki0, kj1, kk1, radial_profile_code, args, l, m)
                    + pi1 * pj0 * pk0 * window_function_radial_multipole_numba(ki1, kj0, kk0, radial_profile_code, args, l, m)
                    + pi1 * pj0 * pk1 * window_function_radial_multipole_numba(ki1, kj0, kk1, radial_profile_code, args, l, m)
                    + pi1 * pj1 * pk0 * window_function_radial_multipole_numba(ki1, kj1, kk0, radial_profile_code, args, l, m)
                    + pi1 * pj1 * pk1 * window_function_radial_multipole_numba(ki1, kj1, kk1, radial_profile_code, args, l, m)
                )
    return window_array


def calculate_radial_multipole_window_array(L, phi_fourier_power, len_args, radial_type, l, m):
    """Build a complex full-FFT window for a radial profile times ``Y_lm``."""
    radial_profile_code, args = radial_profile_args(radial_type, len_args)
    return calculate_radial_multipole_window_array_numba(
        L,
        phi_fourier_power,
        int(radial_profile_code),
        args,
        int(l),
        int(m),
    )
