import numpy as np
from numba import njit

from pyhermes.param.logbase import setup_logger
from pyhermes.utils import func_util
from pyhermes.utils.func_util import get_fname_info
from pyhermes.utils.special_functions import jn_numba


@njit
def window_function_shell_numba(ki, kj, kk, R):
    """
    Thin spherical-shell window in k-space.

    Let k = sqrt(ki^2 + kj^2 + kk^2) and q = 2*pi*k*R.
    W(k; R) = sin(q) / q, with W(0; R) = 1.
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 1
    Phase = 2 * np.pi * k * R
    result = np.sin(Phase) / Phase
    return result


@njit
def window_function_sphere_numba(ki, kj, kk, R):
    """
    Spherical top-hat window in k-space.

    Let k = sqrt(ki^2 + kj^2 + kk^2) and q = 2*pi*k*R.
    W(k; R) = 3 * (sin(q) - q*cos(q)) / q^3, with W(0; R) = 1.
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 1
    Phase = 2 * np.pi * k * R
    result = 3 * (np.sin(Phase) - Phase * np.cos(Phase)) / Phase**3
    return result


@njit
def window_function_gauss_numba(ki, kj, kk, R):
    """
    Gaussian window in k-space.

    Let k = sqrt(ki^2 + kj^2 + kk^2) and q = 2*pi*k*R.
    W(k; R) = exp(-q^2 / 2).
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    Phase = 2 * np.pi * k * R
    result = np.exp(-(Phase**2) / 2)
    return result


@njit
def window_function_gauss_shell_numba(ki, kj, kk, R1, R2):
    """
    Gaussian-damped shell-like window in k-space.

    Let k = sqrt(ki^2 + kj^2 + kk^2), q1 = 2*pi*k*R1,
    and q2 = 2*pi*k*R2.
    W(k; R1, R2) =
        ((R2^2*cos(q1) + R1^2*sin(q1)/q1) / (R1^2 + R2^2))
        * exp(-q2^2 / 2),
    with W(0; R1, R2) = 1.
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 1
    Phase1 = 2 * np.pi * k * R1
    Phase2 = 2 * np.pi * k * R2
    result = (
        (R2 * R2 * np.cos(Phase1) + R1 * R1 * np.sin(Phase1) / Phase1) / (R1 * R1 + R2 * R2)
        * np.exp(-(Phase2**2) / 2)
    )
    return result


@njit
def window_function_Tshell_numba(ki, kj, kk, R1, R2):
    """
    Finite-thickness spherical shell window in k-space.

    Let k = sqrt(ki^2 + kj^2 + kk^2), q1 = 2*pi*k*R1,
    and q2 = 2*pi*k*R2.
    W(k; R1, R2) =
        3 * (sin(q2) - q2*cos(q2) - sin(q1) + q1*cos(q1))
        / (q2^3 - q1^3),
    with W(0; R1, R2) = 1.
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 1
    Phase1 = 2 * np.pi * k * R1
    Phase2 = 2 * np.pi * k * R2
    result = (
        3
        * (np.sin(Phase2) - Phase2 * np.cos(Phase2) - np.sin(Phase1) + Phase1 * np.cos(Phase1))
        / (Phase2**3 - Phase1**3)
    )
    return result


@njit
def window_function_gauss_direvative_wavalet_numba(ki, kj, kk, R):
    """
    Gaussian-derivative wavelet window in k-space.

    Let k = sqrt(ki^2 + kj^2 + kk^2), q = 2*pi*k*R, and
    A(R) = 2^(7/4) / sqrt(15) * (2*pi)^(3/4) * R^(3/2).
    W(k; R) = A(R) * q^2 * exp(-q^2 / 2).
    """
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    Phase = 2 * np.pi * k * R
    norm = 2 ** (7 / 4) / np.sqrt(15) * (2 * np.pi) ** (3 / 4) * R ** (3 / 2)
    result = norm * Phase**2 * np.exp(-(Phase**2) / 2)
    return result


@njit
def window_function_cylinder_numba(ki, kj, kk, R, h):
    """
    Cylindrical top-hat window in k-space.

    Let k_perp = sqrt(ki^2 + kj^2), q_perp = 2*pi*k_perp*R,
    and q_z = 2*pi*kk*h/2.
    W(k; R, h) = (2*J1(q_perp)/q_perp) * (sin(q_z)/q_z),
    with the corresponding factors set to 1 when q_perp = 0 or q_z = 0.
    """
    k1 = np.sqrt(ki**2 + kj**2)
    if kk == 0:
        part1 = 1
    else:
        part1 = np.sin(2 * np.pi * kk * h / 2) / (2 * np.pi * kk * h / 2)

    if k1 == 0:
        sum_val = 1
    else:
        sum_val = 2 * jn_numba(1, 2 * np.pi * k1 * R) / (2 * np.pi * k1 * R)
    return sum_val * part1


@njit
def window_function_ring_numba(ki, kj, kk, R, H, nx=0.0, ny=0.0, nz=1.0):
    """
    Thin ring-pair window in k-space with a configurable line of sight.

    ``(nx, ny, nz)`` defaults to the z direction, is normalized internally, and
    should be passed via
    ``other_args`` because it is dimensionless. ``R`` and ``H`` are lengths and
    should be passed via ``len_args``.
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


def set_window_function(w_type, verbose=True):
    w_type_dict = {
        "shell": window_function_shell_numba,
        "sphere": window_function_sphere_numba,
        "gaussian": window_function_gauss_numba,
        "gaussian_shell": window_function_gauss_shell_numba,
        "Tshell": window_function_Tshell_numba,
        "gaussian_direvative_wavalet": window_function_gauss_direvative_wavalet_numba,
        "cylinder": window_function_cylinder_numba,
        "ring": window_function_ring_numba,
    }
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    if w_type in w_type_dict:
        verbose and logger.info(f"Using window function: {w_type}")
        read_function = w_type_dict[w_type]
        return read_function

    supported_w_type = ", ".join(w_type_dict.keys())
    logger.error(f"Unsupported input window function type: {w_type}")
    logger.error(f"Supported types: {supported_w_type}")
    logger.error("Please see the document for details")
    func_util.safe_exit(1)
