"""General radial-profile multipole window kernels."""

import math

import numpy as np
from numba import njit, prange

from pyhermes.utils.radial_profiles import (
    build_radial_multipole_table,
    validate_radial_profile_request,
)
from pyhermes.utils.special_functions import (
    spherical_harmonic_numba,
    spherical_jn_numba,
)

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
    if not np.all(np.isfinite(args)):
        raise ValueError(f"radial_multipole profile '{radial_type}' requires finite length arguments.")
    if args[0] < 0.0:
        raise ValueError(f"radial_multipole profile '{radial_type}' requires a non-negative radial scale.")
    return RADIAL_PROFILE_CODES[radial_type], args


def validate_radial_multipole_profile(radial_type, len_args, l_max, profile_config=None):
    """Validate one radial profile for a requested multipole range."""
    del l_max
    validate_radial_profile_request(radial_type, len_args, profile_config)


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

    This direct path is used for thin shells at all orders and for the exact
    analytic monopoles of the remaining built-in profiles.  Higher orders of
    non-shell profiles use the tabulated Hankel path below.
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


@njit
def _tabulated_radial_value(k, table, k_max):
    if k <= 0.0:
        return table[0]
    if k >= k_max:
        return table[-1]
    index = k * (table.size - 1) / k_max
    lower = int(index)
    fraction = index - lower
    if lower == 0 or lower >= table.size - 2:
        return table[lower] * (1.0 - fraction) + table[lower + 1] * fraction

    # The table is uniform in k.  A local cubic avoids smearing oscillatory
    # U_l(k) profiles without materializing a much larger Fourier kernel.
    p0 = table[lower - 1]
    p1 = table[lower]
    p2 = table[lower + 1]
    p3 = table[lower + 2]
    fraction2 = fraction * fraction
    fraction3 = fraction2 * fraction
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * fraction
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * fraction2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * fraction3
    )


@njit
def window_function_tabulated_radial_multipole_numba(
    ki, kj, kk, table, k_max, grid_to_physical, l, m
):
    if l < 0 or abs(m) > l:
        return 0.0 + 0.0j
    k_grid = math.sqrt(ki * ki + kj * kj + kk * kk)
    radial = _tabulated_radial_value(k_grid * grid_to_physical, table, k_max)
    angular = spherical_harmonic_numba(l, m, ki, kj, kk)
    return radial * angular


@njit(parallel=True)
def calculate_tabulated_radial_multipole_window_array_numba(
    L, phi_fourier_power, table, k_max, grid_to_physical, l, m
):
    """Build a complex full-FFT radial multipole window from a U_l(k) table."""
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
                    pi0 * pj0 * pk0 * window_function_tabulated_radial_multipole_numba(ki0, kj0, kk0, table, k_max, grid_to_physical, l, m)
                    + pi0 * pj0 * pk1 * window_function_tabulated_radial_multipole_numba(ki0, kj0, kk1, table, k_max, grid_to_physical, l, m)
                    + pi0 * pj1 * pk0 * window_function_tabulated_radial_multipole_numba(ki0, kj1, kk0, table, k_max, grid_to_physical, l, m)
                    + pi0 * pj1 * pk1 * window_function_tabulated_radial_multipole_numba(ki0, kj1, kk1, table, k_max, grid_to_physical, l, m)
                    + pi1 * pj0 * pk0 * window_function_tabulated_radial_multipole_numba(ki1, kj0, kk0, table, k_max, grid_to_physical, l, m)
                    + pi1 * pj0 * pk1 * window_function_tabulated_radial_multipole_numba(ki1, kj0, kk1, table, k_max, grid_to_physical, l, m)
                    + pi1 * pj1 * pk0 * window_function_tabulated_radial_multipole_numba(ki1, kj1, kk0, table, k_max, grid_to_physical, l, m)
                    + pi1 * pj1 * pk1 * window_function_tabulated_radial_multipole_numba(ki1, kj1, kk1, table, k_max, grid_to_physical, l, m)
                )
    return window_array


def calculate_radial_multipole_window_array(
    L, phi_fourier_power, len_args, radial_type, l, m, *, box_size, profile_config=None
):
    """Build a complex full-FFT window for ``U_l(k) Y_lm(khat)``."""
    radial_type = str(radial_type).strip().lower()
    validate_radial_multipole_profile(radial_type, len_args, l, profile_config)
    l = int(l)
    m = int(m)

    if radial_type == "shell":
        rescaled_len_args = {
            key: float(value) * float(L) / float(box_size)
            for key, value in len_args.items()
        }
        radial_profile_code, args = radial_profile_args(radial_type, rescaled_len_args)
        return calculate_radial_multipole_window_array_numba(
            L, phi_fourier_power, int(radial_profile_code), args, l, m
        )

    if l == 0 and radial_type in RADIAL_PROFILE_CODES:
        rescaled_len_args = {
            key: float(value) * float(L) / float(box_size)
            for key, value in len_args.items()
        }
        radial_profile_code, args = radial_profile_args(radial_type, rescaled_len_args)
        return calculate_radial_multipole_window_array_numba(
            L, phi_fourier_power, int(radial_profile_code), args, l, m
        )

    if radial_type == "gaussian_shell" and float(len_args["R_smooth"]) == 0.0:
        rescaled_len_args = {"R": float(len_args["R_shell"]) * float(L) / float(box_size)}
        radial_profile_code, args = radial_profile_args("shell", rescaled_len_args)
        return calculate_radial_multipole_window_array_numba(
            L, phi_fourier_power, int(radial_profile_code), args, l, m
        )

    grid_to_physical = float(L) / float(box_size)
    k_max = math.sqrt(3.0) * grid_to_physical
    table_k_max, table = build_radial_multipole_table(
        radial_type, len_args, l, k_max, profile_config=profile_config
    )
    return calculate_tabulated_radial_multipole_window_array_numba(
        L,
        phi_fourier_power,
        table,
        table_k_max,
        grid_to_physical,
        l,
        m,
    )
