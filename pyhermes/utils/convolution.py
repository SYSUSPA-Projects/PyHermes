"""FFT convolution helpers and generic window-array construction."""

import inspect

import numpy as np
from numba import njit, prange
from scipy.fft import fftn, ifftn, irfftn, rfftn

from pyhermes.param.logbase import setup_logger
from pyhermes.utils import func_util
from pyhermes.utils.func_util import get_fname_info


@njit(parallel=True)
def calculate_real_window_octant_array_numba(L, bandwidth, phi_fourier_power, window_function_numba, *args):
    """Evaluate a real-space window lookup array from a k-space window kernel."""
    WindowArray = np.zeros((L + 1, L + 1, L + 1))
    inv_L = 1.0 / L
    for i in prange(L + 1):
        for j in range(L + 1):
            for k in range(L + 1):
                temp = 0.0
                for ii in range(bandwidth):
                    for jj in range(bandwidth):
                        for kk in range(bandwidth):
                            temp += (
                                phi_fourier_power[ii * L + i]
                                * phi_fourier_power[jj * L + j]
                                * phi_fourier_power[kk * L + k]
                                * window_function_numba(
                                    (ii * L + i) * inv_L,
                                    (jj * L + j) * inv_L,
                                    (kk * L + k) * inv_L,
                                    *args
                                )
                            )
                WindowArray[i, j, k] = temp
    return WindowArray


def _ordered_window_args(window_function_numba, kwargs):
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    sig = inspect.signature(window_function_numba)
    params = sig.parameters
    expected_args = list(params.keys())[3:]
    provided_args = kwargs.keys()
    missing_args = [
        arg for arg in expected_args
        if arg not in provided_args and params[arg].default is inspect.Parameter.empty
    ]
    if missing_args:
        source_code = inspect.getsource(window_function_numba)
        logger.error("\n" + source_code)
        logger.error(f"Missing keyword arguments: {missing_args}")
        logger.error("Please see the document for details")
        func_util.safe_exit(1)
    ordered_args = [
        kwargs[arg] if arg in kwargs else params[arg].default
        for arg in expected_args
    ]
    return ordered_args


def build_real_window_octant_array(L, bandwidth, phi_fourier_power, window_function_numba, **kwargs):
    """Build an octant-symmetric real-window lookup array from keyword arguments."""
    ordered_args = _ordered_window_args(window_function_numba, kwargs)
    return calculate_real_window_octant_array_numba(
        L, bandwidth, phi_fourier_power, window_function_numba, *ordered_args
    )


@njit(parallel=True)
def calculate_real_window_rfft_kernel_numba(L, bandwidth, phi_fourier_power, window_function_numba, *args):
    """Evaluate a real rFFT-space window kernel without assuming octant symmetry."""
    w = np.zeros((L, L, L // 2 + 1))
    inv_L = 1.0 / L
    for x in prange(L):
        x_mirror = L - x
        for y in range(L):
            y_mirror = L - y
            for z in range(L // 2 + 1):
                z_mirror = L - z
                temp = 0.0
                for ii in range(bandwidth):
                    x_pos_idx = ii * L + x
                    x_neg_idx = ii * L + x_mirror
                    x_pos = x_pos_idx * inv_L
                    x_neg = -x_neg_idx * inv_L
                    for jj in range(bandwidth):
                        y_pos_idx = jj * L + y
                        y_neg_idx = jj * L + y_mirror
                        y_pos = y_pos_idx * inv_L
                        y_neg = -y_neg_idx * inv_L
                        for kk in range(bandwidth):
                            z_pos_idx = kk * L + z
                            z_neg_idx = kk * L + z_mirror
                            z_pos = z_pos_idx * inv_L
                            z_neg = -z_neg_idx * inv_L

                            phi_x_pos = phi_fourier_power[x_pos_idx]
                            phi_x_neg = phi_fourier_power[x_neg_idx]
                            phi_y_pos = phi_fourier_power[y_pos_idx]
                            phi_y_neg = phi_fourier_power[y_neg_idx]
                            phi_z_pos = phi_fourier_power[z_pos_idx]
                            phi_z_neg = phi_fourier_power[z_neg_idx]

                            temp += (
                                phi_x_pos
                                * phi_y_pos
                                * phi_z_pos
                                * window_function_numba(x_pos, y_pos, z_pos, *args)
                            )
                            temp += (
                                phi_x_neg
                                * phi_y_pos
                                * phi_z_pos
                                * window_function_numba(x_neg, y_pos, z_pos, *args)
                            )
                            temp += (
                                phi_x_pos
                                * phi_y_neg
                                * phi_z_pos
                                * window_function_numba(x_pos, y_neg, z_pos, *args)
                            )
                            temp += (
                                phi_x_pos
                                * phi_y_pos
                                * phi_z_neg
                                * window_function_numba(x_pos, y_pos, z_neg, *args)
                            )
                            temp += (
                                phi_x_neg
                                * phi_y_neg
                                * phi_z_pos
                                * window_function_numba(x_neg, y_neg, z_pos, *args)
                            )
                            temp += (
                                phi_x_neg
                                * phi_y_pos
                                * phi_z_neg
                                * window_function_numba(x_neg, y_pos, z_neg, *args)
                            )
                            temp += (
                                phi_x_pos
                                * phi_y_neg
                                * phi_z_neg
                                * window_function_numba(x_pos, y_neg, z_neg, *args)
                            )
                            temp += (
                                phi_x_neg
                                * phi_y_neg
                                * phi_z_neg
                                * window_function_numba(x_neg, y_neg, z_neg, *args)
                            )
                w[x, y, z] = temp
    return w


def build_real_window_rfft_kernel(L, bandwidth, phi_fourier_power, window_function_numba, **kwargs):
    """Build a real rFFT-space kernel without assuming per-axis mirror symmetry."""
    ordered_args = _ordered_window_args(window_function_numba, kwargs)
    return calculate_real_window_rfft_kernel_numba(
        L, bandwidth, phi_fourier_power, window_function_numba, *ordered_args
    )


@njit(parallel=True)
def fold_octant_window_to_rfft_kernel(WindowArray):
    """Fold an octant-symmetric window array into an rFFT-compatible kernel."""
    L = WindowArray.shape[0] - 1
    w = np.zeros((L, L, L // 2 + 1))
    for x in prange(L):
        for y in range(L):
            for z in range(L // 2 + 1):
                w[x, y, z] = (
                    WindowArray[x, y, z]
                    + WindowArray[L - x, y, z]
                    + WindowArray[x, L - y, z]
                    + WindowArray[x, y, L - z]
                    + WindowArray[L - x, L - y, z]
                    + WindowArray[L - x, y, L - z]
                    + WindowArray[x, L - y, L - z]
                    + WindowArray[L - x, L - y, L - z]
                )
    return w


def specialized_convolution_3d(s, w, threads):
    """Convolve a real 3D field with an rFFT-space kernel."""
    sc = rfftn(s, workers=threads)
    sc *= w
    return irfftn(sc, workers=threads)


def specialized_convolution_3d_complex(s, w, threads):
    """Convolve a complex 3D field with a full FFT-space kernel."""
    sc = fftn(s, workers=threads)
    sc *= w
    return ifftn(sc, workers=threads)
