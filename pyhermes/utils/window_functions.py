"""Built-in Fourier-space window transfer kernels.

The functions in this module return the kernels multiplied onto FFT
coefficients by ``WindowFunc``. They use a non-unitary Fourier convention in
cycle-frequency coordinates,

    T_W(k_cyc) = int d^d x W(x) exp(-2*pi*i*k_cyc dot x).

Equivalently, with angular wavenumber K = 2*pi*k_cyc, this is the
non-unitary transform W_hat(K) = int d^d x W(x) exp(-i*K dot x).

These kernels are not unitary Fourier transforms. For a unit-integral
smoothing window the transfer kernel satisfies T_W(0) = 1, while a unitary
Fourier transform would carry extra factors such as (2*pi)^(-d/2). Inside
``WindowFunc`` the k components are cycles per grid unit, and length arguments
are rescaled to grid units before the kernels are evaluated.
"""

import numpy as np
from numba import njit

from pyhermes.param.logbase import setup_logger
from pyhermes.utils import func_util
from pyhermes.utils.func_util import get_fname_info
from pyhermes.utils.special_functions import jn_numba

SPEED_OF_LIGHT_KM_S = 299792.458

# Isotropic windows.
@njit
def window_function_sphere_numba(ki, kj, kk, R):
    """
    Spherical top-hat window in k-space.

    Let k = sqrt(ki^2 + kj^2 + kk^2) and q = 2*pi*k*R.
    W(k; R) = 3 * (sin(q) - q*cos(q)) / q^3, with W(0; R) = 1.
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    q = 2 * np.pi * k * R
    if q == 0.0:
        return 1
    result = 3 * (np.sin(q) - q * np.cos(q)) / q**3
    return result


@njit
def window_function_gauss_numba(ki, kj, kk, R):
    """
    Gaussian window in k-space.

    Let k = sqrt(ki^2 + kj^2 + kk^2) and q = 2*pi*k*R.
    W(k; R) = exp(-q^2 / 2).
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    q = 2 * np.pi * k * R
    result = np.exp(-(q**2) / 2)
    return result


@njit
def window_function_shell_numba(ki, kj, kk, R):
    """
    Thin spherical-shell window in k-space.

    Let k = sqrt(ki^2 + kj^2 + kk^2) and q = 2*pi*k*R.
    W(k; R) = sin(q) / q, with W(0; R) = 1.
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    q = 2 * np.pi * k * R
    if q == 0.0:
        return 1
    result = np.sin(q) / q
    return result

@njit
def window_function_gauss_shell_numba(ki, kj, kk, R_shell, R_smooth):
    """
    Gaussian-damped shell-like window in k-space.

    ``R_shell`` sets the shell-like oscillation scale, and ``R_smooth`` sets
    the Gaussian damping scale.

    Let k = sqrt(ki^2 + kj^2 + kk^2), q_shell = 2*pi*k*R_shell,
    and q_smooth = 2*pi*k*R_smooth.
    W(k; R_shell, R_smooth) =
        ((R_smooth^2*cos(q_shell) + R_shell^2*sin(q_shell)/q_shell)
        / (R_shell^2 + R_smooth^2)) * exp(-q_smooth^2 / 2),
    with W(0; R_shell, R_smooth) = 1.
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 1
    q_shell = 2 * np.pi * k * R_shell
    q_smooth = 2 * np.pi * k * R_smooth
    denom = R_shell * R_shell + R_smooth * R_smooth
    if denom == 0.0:
        return 1
    if q_shell == 0.0:
        shell_sinc = 1.0
    else:
        shell_sinc = np.sin(q_shell) / q_shell
    result = (
        (R_smooth * R_smooth * np.cos(q_shell) + R_shell * R_shell * shell_sinc) 
        / denom * np.exp(-(q_smooth**2) / 2)
    )
    return result


# Axis-aligned windows.
@njit
def window_function_cubic_numba(ki, kj, kk, Lx, Ly, Lz):
    """
    Axis-aligned rectangular top-hat window in k-space.

    With the PyHermes Fourier convention exp(-2*pi*i*k*x), the one-dimensional
    factor is sin(pi*k_i*L_i)/(pi*k_i*L_i), with value 1 when pi*k_i*L_i = 0.
    """
    qx = np.pi * ki * Lx
    qy = np.pi * kj * Ly
    qz = np.pi * kk * Lz

    if qx == 0.0:
        part_x = 1.0
    else:
        part_x = np.sin(qx) / qx

    if qy == 0.0:
        part_y = 1.0
    else:
        part_y = np.sin(qy) / qy

    if qz == 0.0:
        part_z = 1.0
    else:
        part_z = np.sin(qz) / qz

    return part_x * part_y * part_z


# Line-of-sight windows.
@njit
def window_function_ring_numba(ki, kj, kk, R, H, nx=0.0, ny=0.0, nz=1.0):
    """
    Thin ring-pair window in k-space with a configurable line of sight.

    ``(nx, ny, nz)`` defaults to the z direction, is normalized internally, and
    should be passed via ``los_args``. ``R`` and ``H`` are lengths and should be
    passed via ``len_args``.
    """
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0.0:
        return np.nan
    nx = nx / norm
    ny = ny / norm
    nz = nz / norm

    k_parallel = ki * nx + kj * ny + kk * nz
    k2 = ki * ki + kj * kj + kk * kk
    k_perp2 = k2 - k_parallel * k_parallel

    if k_perp2 < 0.0:
        k_perp2 = 0.0

    k_perp = np.sqrt(k_perp2)

    q_perp = 2.0 * np.pi * k_perp * R
    q_parallel = 2.0 * np.pi * k_parallel * H

    return jn_numba(0, q_perp) * np.cos(q_parallel)


@njit
def window_function_disk_numba(ki, kj, kk, R, H, nx=0.0, ny=0.0, nz=1.0):
    """
    Thin disk-pair window in k-space with a configurable line of sight.

    ``(nx, ny, nz)`` defaults to the z direction, is normalized internally, and
    should be passed via ``los_args``. ``R`` and ``H`` are lengths and should be
    passed via ``len_args``.

    Let k_parallel = k dot n, k_perp = sqrt(|k|^2 - k_parallel^2),
    q_perp = 2*pi*k_perp*R, and q_parallel = 2*pi*k_parallel*H.
    W(k; R, H) = (2*J1(q_perp)/q_perp) * cos(q_parallel),
    with the perpendicular factor set to 1 when q_perp = 0.
    """
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0.0:
        return np.nan
    nx = nx / norm
    ny = ny / norm
    nz = nz / norm

    k_parallel = ki * nx + kj * ny + kk * nz
    k2 = ki * ki + kj * kj + kk * kk
    k_perp2 = k2 - k_parallel * k_parallel

    if k_perp2 < 0.0:
        k_perp2 = 0.0

    k_perp = np.sqrt(k_perp2)
    q_perp = 2.0 * np.pi * k_perp * R
    q_parallel = 2.0 * np.pi * k_parallel * H

    part_parallel = np.cos(q_parallel)

    if q_perp == 0.0:
        part_perp = 1.0
    else:
        part_perp = 2.0 * jn_numba(1, q_perp) / q_perp
    return part_perp * part_parallel


@njit
def window_function_cylshell_numba(ki, kj, kk, R, H, nx=0.0, ny=0.0, nz=1.0):
    """
    Thin cylindrical-shell window in k-space with a configurable line of sight.

    ``(nx, ny, nz)`` defaults to the z direction, is normalized internally, and
    should be passed via ``los_args``. ``R`` and ``H`` are lengths and should be
    passed via ``len_args``. Here ``H`` is the half-height of the cylinder, so
    the real-space support is ``|z| <= H``.

    Let k_parallel = k dot n, k_perp = sqrt(|k|^2 - k_parallel^2),
    q_perp = 2*pi*k_perp*R, and q_parallel = 2*pi*k_parallel*H.
    W(k; R, H) = J0(q_perp) * (sin(q_parallel)/q_parallel),
    with the parallel factor set to 1 when q_parallel = 0.
    """
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0.0:
        return np.nan
    nx = nx / norm
    ny = ny / norm
    nz = nz / norm

    k_parallel = ki * nx + kj * ny + kk * nz
    k2 = ki * ki + kj * kj + kk * kk
    k_perp2 = k2 - k_parallel * k_parallel

    if k_perp2 < 0.0:
        k_perp2 = 0.0

    k_perp = np.sqrt(k_perp2)
    q_perp = 2.0 * np.pi * k_perp * R
    q_parallel = 2.0 * np.pi * k_parallel * H

    if q_parallel == 0.0:
        part_parallel = 1.0
    else:
        part_parallel = np.sin(q_parallel) / q_parallel

    return jn_numba(0, q_perp) * part_parallel


@njit
def window_function_cylinder_numba(ki, kj, kk, R, H, nx=0.0, ny=0.0, nz=1.0):
    """
    Cylindrical top-hat window in k-space with a configurable line of sight.

    ``(nx, ny, nz)`` defaults to the z direction, is normalized internally, and
    should be passed via ``los_args``. ``R`` and ``H`` are lengths and should be
    passed via ``len_args``. Here ``H`` is the half-height of the cylinder, so
    the real-space support is ``|z| <= H``.

    Let k_parallel = k dot n, k_perp = sqrt(|k|^2 - k_parallel^2),
    q_perp = 2*pi*k_perp*R, and q_parallel = 2*pi*k_parallel*H.
    W(k; R, H) = (2*J1(q_perp)/q_perp) * (sin(q_parallel)/q_parallel),
    with the corresponding factors set to 1 when q_perp = 0 or q_parallel = 0.
    """
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0.0:
        return np.nan
    nx = nx / norm
    ny = ny / norm
    nz = nz / norm

    k_parallel = ki * nx + kj * ny + kk * nz
    k2 = ki * ki + kj * kj + kk * kk
    k_perp2 = k2 - k_parallel * k_parallel

    if k_perp2 < 0.0:
        k_perp2 = 0.0

    k_perp = np.sqrt(k_perp2)
    q_perp = 2.0 * np.pi * k_perp * R
    q_parallel = 2.0 * np.pi * k_parallel * H

    if q_parallel == 0.0:
        part_parallel = 1.0
    else:
        part_parallel = np.sin(q_parallel) / q_parallel

    if q_perp == 0.0:
        part_perp = 1.0
    else:
        part_perp = 2.0 * jn_numba(1, q_perp) / q_perp
    return part_perp * part_parallel


# High-pass and wavelet-like filters.
@njit
def _cosine_wavelet_g_numba(q):
    q_abs = np.abs(q)
    if q_abs == 0.0:
        return 0.0
    if q_abs < 1.0e-4:
        return q_abs**4 / 3.0
    if q_abs > 50.0:
        return 0.0
    return q_abs * (q_abs * np.cosh(q_abs) - np.sinh(q_abs)) * np.exp(-0.5 * q_abs * q_abs)


@njit
def window_function_cw_numba(ki, kj, kk, R):
    """
    One-dimensional cosine wavelet transfer kernel.

    This is the non-unitary transfer-kernel counterpart of the CW row in the
    paper's window table, evaluated as an isotropic radial response in
    |k|. Let k = sqrt(ki^2 + kj^2 + kk^2), q = 2*pi*k*R, and
    C_CW = 2*sqrt(2) / (sqrt(1 + 5*e) * pi^(1/4)).
    W(k; R) = sqrt(2*pi) * C_CW * R^(1/2) * G_CW(q), where
    G_CW(q) = q * (q*cosh(q) - sinh(q)) * exp(-q^2 / 2).
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    q = 2 * np.pi * k * R
    c_cw = 2 * np.sqrt(2.0) / (np.sqrt(1.0 + 5.0 * np.e) * np.pi ** 0.25)
    norm = np.sqrt(2.0 * np.pi) * c_cw * R ** 0.5
    return norm * _cosine_wavelet_g_numba(q)


@njit
def window_function_cws_numba(ki, kj, kk, R):
    """
    Spherical cosine wavelet transfer kernel.

    This is the three-dimensional spherical version of the cosine wavelet.
    Let k = sqrt(ki^2 + kj^2 + kk^2), q = 2*pi*k*R, and
    C_CWS = 2*sqrt(2) / (sqrt(9 + 55*e) * pi^(3/4)).
    W(k; R) = (2*pi)^(3/2) * C_CWS * R^(3/2) * G_CW(q).
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    q = 2 * np.pi * k * R
    c_cws = 2 * np.sqrt(2.0) / (np.sqrt(9.0 + 55.0 * np.e) * np.pi ** 0.75)
    norm = (2.0 * np.pi) ** 1.5 * c_cws * R ** 1.5
    return norm * _cosine_wavelet_g_numba(q)


@njit
def window_function_gdw_numba(ki, kj, kk, R):
    """
    Gaussian-derivative wavelet window in k-space.

    Let k = sqrt(ki^2 + kj^2 + kk^2), q = 2*pi*k*R, and
    A(R) = 2^(5/2) * pi^(3/4) / sqrt(15) * R^(3/2).
    W(k; R) = A(R) * q^2 * exp(-q^2 / 2).
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    q = 2 * np.pi * k * R
    norm = 2 ** (5 / 2) * np.pi ** (3 / 4) / np.sqrt(15) * R ** (3 / 2)
    result = norm * q**2 * np.exp(-(q**2) / 2)
    return result


# Field-derivative operator windows.
@njit
def window_function_directional_derivative_numba(ki, kj, kk, nx=0.0, ny=0.0, nz=1.0):
    """
    Directional derivative window in k-space.

    ``(nx, ny, nz)`` sets the derivative direction and is normalized
    internally.

    Let k_parallel = k dot n_hat.
    W(k; n_hat) = 2*pi*i*k_parallel.

    The derivative is with respect to the same coordinate system used by the
    Fourier variables passed to the window function. In ``WindowFunc`` this is
    the grid coordinate system.
    """
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0.0:
        return np.nan + 0.0j
    nx = nx / norm
    ny = ny / norm
    nz = nz / norm

    k_parallel = ki * nx + kj * ny + kk * nz
    return 0.0 + 1.0j * 2.0 * np.pi * k_parallel


@njit
def window_function_laplacian_numba(ki, kj, kk):
    """
    Laplacian operator window in k-space.

    With the inverse-FFT basis exp(2*pi*i*k*x),
    W(k) = -(2*pi)^2 * |k|^2, corresponding to the operator nabla^2.
    """
    k2 = ki * ki + kj * kj + kk * kk
    return -((2.0 * np.pi) ** 2) * k2


@njit
def window_function_inverse_laplacian_numba(ki, kj, kk):
    """
    Inverse-Laplacian operator window in k-space.

    With the inverse-FFT basis exp(2*pi*i*k*x),
    W(k) = -1 / ((2*pi)^2 * |k|^2), corresponding to nabla^{-2}.
    The zero mode is set to zero, fixing the mean of the resulting field.
    """
    k2 = ki * ki + kj * kj + kk * kk
    if k2 == 0.0:
        return 0.0
    return -1.0 / (((2.0 * np.pi) ** 2) * k2)


@njit
def window_function_gravitational_potential_numba(ki, kj, kk, omegam, H0, a):
    """
    Poisson gravitational-potential window for Phi/c^2.

    For a density contrast delta, the comoving Poisson equation is

        nabla^2 (Phi/c^2) = (3/2) * Omega_m * (H0/c)^2 * delta / a.

    This window returns

        W(k) = -[(3/2) * Omega_m * (H0/c)^2 / a] / ((2*pi)^2 * |k|^2),

    with W(0) = 0. ``H0`` should be supplied in km/s per coordinate unit
    inverse. ``WindowFunc`` accepts H0 in km/s/(Mpc/h) when box_size is in
    Mpc/h, and rescales it internally to the grid-coordinate system used by
    the kernel builder.
    """
    if a <= 0.0:
        return np.nan
    prefactor = 1.5 * omegam * (H0 / SPEED_OF_LIGHT_KM_S) ** 2 / a
    return prefactor * window_function_inverse_laplacian_numba(ki, kj, kk)


WINDOW_TYPE_DICT = {
    "sphere": window_function_sphere_numba,
    "gaussian": window_function_gauss_numba,
    "shell": window_function_shell_numba,
    "gaussian_shell": window_function_gauss_shell_numba,
    "cubic": window_function_cubic_numba,
    "ring": window_function_ring_numba,
    "disk": window_function_disk_numba,
    "cylshell": window_function_cylshell_numba,
    "cylinder": window_function_cylinder_numba,
    "cw": window_function_cw_numba,
    "cws": window_function_cws_numba,
    "gdw": window_function_gdw_numba,
    "directional_derivative": window_function_directional_derivative_numba,
    "laplacian": window_function_laplacian_numba,
    "inverse_laplacian": window_function_inverse_laplacian_numba,
    "gravitational_potential": window_function_gravitational_potential_numba,
}


def set_window_function(w_type, verbose=True):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    if w_type in WINDOW_TYPE_DICT:
        verbose and logger.info(f"Using window function: {w_type}")
        read_function = WINDOW_TYPE_DICT[w_type]
        return read_function

    supported_w_type = ", ".join(WINDOW_TYPE_DICT.keys())
    logger.error(f"Unsupported input window function type: {w_type}")
    logger.error(f"Supported types: {supported_w_type}")
    logger.error("Please see the document for details")
    func_util.safe_exit(1)
