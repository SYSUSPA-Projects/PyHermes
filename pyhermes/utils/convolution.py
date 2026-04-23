"""FFT convolution helpers and generic window-array construction."""

import inspect

import numpy as np
from numba import njit, prange
from scipy.fft import fftn, ifftn, irfftn, rfftn

from pyhermes.param.logbase import setup_logger
from pyhermes.utils import func_util
from pyhermes.utils.func_util import get_fname_info


@njit(parallel=True)
def calculate_real_window_octant_array_numba(L, bandwidth, PowerPhi, window_function_numba, *args):
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
                                PowerPhi[ii * L + i]
                                * PowerPhi[jj * L + j]
                                * PowerPhi[kk * L + k]
                                * window_function_numba(
                                    (ii * L + i) * inv_L,
                                    (jj * L + j) * inv_L,
                                    (kk * L + k) * inv_L,
                                    *args
                                )
                            )
                WindowArray[i, j, k] = temp
    return WindowArray


def call_calculate_window_array(L, bandwidth, PowerPhi, window_function_numba, **kwargs):
    """Call ``calculate_real_window_octant_array_numba`` with keyword arguments in kernel-signature order."""
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    sig = inspect.signature(window_function_numba)
    params = sig.parameters
    expected_args = list(params.keys())[3:]
    provided_args = kwargs.keys()
    missing_args = [arg for arg in expected_args if arg not in provided_args]
    if missing_args:
        source_code = inspect.getsource(window_function_numba)
        logger.error("\n" + source_code)
        logger.error(f"Missing keyword arguments: {missing_args}")
        logger.error("Please see the document for details")
        func_util.safe_exit(1)
    ordered_args = [kwargs[arg] for arg in expected_args if arg in kwargs]
    return calculate_real_window_octant_array_numba(
        L, bandwidth, PowerPhi, window_function_numba, *ordered_args
    )


@njit(parallel=True)
def calculate_w_numba(WindowArray):
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
