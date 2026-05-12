import numpy as np
from numba import njit

from pyhermes.param.logbase import setup_logger
from pyhermes.utils import func_util
from pyhermes.utils.func_util import get_fname_info
from pyhermes.utils.special_functions import jn_numba

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
def window_function_Tshell_numba(ki, kj, kk, R_in, R_out):
    """
    Finite-thickness spherical shell window in k-space.

    ``R_in`` and ``R_out`` are the inner and outer shell radii.

    Let k = sqrt(ki^2 + kj^2 + kk^2), q_in = 2*pi*k*R_in,
    and q_out = 2*pi*k*R_out.
    W(k; R_in, R_out) =
        3 * (sin(q_out) - q_out*cos(q_out) - sin(q_in) + q_in*cos(q_in))
        / (q_out^3 - q_in^3),
    with W(0; R_in, R_out) = 1.
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 1
    q_in = 2 * np.pi * k * R_in
    q_out = 2 * np.pi * k * R_out
    denom = q_out**3 - q_in**3
    if denom == 0.0:
        if q_out == 0.0:
            return 1
        return np.sin(q_out) / q_out
    result = 3 * (np.sin(q_out) - np.sin(q_in) - q_out * np.cos(q_out) + q_in * np.cos(q_in)) / denom
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


# Anisotropic windows.
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
def window_function_cylinder_numba(ki, kj, kk, R, H, nx=0.0, ny=0.0, nz=1.0):
    """
    Cylindrical top-hat window in k-space with a configurable line of sight.

    ``(nx, ny, nz)`` defaults to the z direction, is normalized internally, and
    should be passed via ``los_args``. ``R`` and ``H`` are lengths and should be
    passed via ``len_args``.

    Let k_parallel = k dot n, k_perp = sqrt(|k|^2 - k_parallel^2),
    q_perp = 2*pi*k_perp*R, and q_parallel = 2*pi*k_parallel*H/2.
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
    q_parallel = np.pi * k_parallel * H

    if q_parallel == 0.0:
        part_parallel = 1.0
    else:
        part_parallel = np.sin(q_parallel) / q_parallel

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


# Special-purpose windows.
@njit
def window_function_gauss_direvative_wavalet_numba(ki, kj, kk, R):
    """
    Gaussian-derivative wavelet window in k-space.

    Let k = sqrt(ki^2 + kj^2 + kk^2), q = 2*pi*k*R, and
    A(R) = 2^(7/4) / sqrt(15) * (2*pi)^(3/4) * R^(3/2).
    W(k; R) = A(R) * q^2 * exp(-q^2 / 2).
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    q = 2 * np.pi * k * R
    norm = 2 ** (7 / 4) / np.sqrt(15) * (2 * np.pi) ** (3 / 4) * R ** (3 / 2)
    result = norm * q**2 * np.exp(-(q**2) / 2)
    return result


WINDOW_TYPE_DICT = {
    "sphere": window_function_sphere_numba,
    "gaussian": window_function_gauss_numba,
    "shell": window_function_shell_numba,
    "Tshell": window_function_Tshell_numba,
    "gaussian_shell": window_function_gauss_shell_numba,
    "ring": window_function_ring_numba,
    "disk": window_function_disk_numba,
    "cylinder": window_function_cylinder_numba,
    "cylshell": window_function_cylshell_numba,
    "gaussian_direvative_wavalet": window_function_gauss_direvative_wavalet_numba,
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
