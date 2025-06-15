import os
import pickle

import numpy as np
import pywt
from numba import njit, prange
from scipy.fft import fftn, ifftn

from legendre_func import *


def spectrum_vectorized(v, k0, k1, N_k):
    N_x = v.shape[0]
    x0 = 0
    x1 = v.shape[0] / SampRate

    Delta_x = (x1 - x0) / N_x
    Delta_k = (k1 - k0) / N_k

    x = np.arange(N_x) * Delta_x
    k = np.arange(N_k + 1) * Delta_k

    # Create 2D grids for x and k
    x_grid, k_grid = np.meshgrid(x, k)

    # Calculate the real and imaginary parts of the spectrum
    s_real = np.sum(v * Delta_x * np.cos(-2 * np.pi * k_grid * x_grid), axis=1)
    s_imag = np.sum(v * Delta_x * np.sin(-2 * np.pi * k_grid * x_grid), axis=1)

    # Interleave the real and imaginary parts
    s = np.empty((N_k + 1) * 2, dtype=np.double)
    s[::2] = s_real
    s[1::2] = s_imag

    return s


def power_spectrum(v, k0, k1, N_k):
    s = spectrum_vectorized(v, k0, k1, N_k)
    p = np.zeros(N_k + 1, dtype=np.double)
    for i in range(N_k + 1):
        p[i] = s[2 * i] ** 2 + s[2 * i + 1] ** 2
    return p


@njit
def window_function_shell_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 1
    # Use np.where to handle the k == 0 case
    Phase = 2 * np.pi * k * R
    result = np.sin(Phase) / Phase
    return result


@njit
def window_function_sphere_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        # return 4 * np.pi * R**3 / 3
        return 1
    # Use np.where to handle the k == 0 case
    Phase = 2 * np.pi * k * R
    # result = (np.sin(Phase) - Phase * np.cos(Phase)) / (2 * np.pi**2 * k**3)
    result = 3 * (np.sin(Phase) - Phase * np.cos(Phase)) / Phase**3
    return result


@njit(parallel=True)
def calculate_window_array_numba(L, bandwidth, DeltaXi, PowerPhi, rescaleR, window_function_numba):
    WindowArray = np.zeros((L + 1, L + 1, L + 1), dtype=np.complex128)
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
                                    (ii * L + i) * DeltaXi,
                                    (jj * L + j) * DeltaXi,
                                    (kk * L + k) * DeltaXi,
                                    rescaleR,
                                )
                            )
                WindowArray[i, j, k] = temp
    return WindowArray


@njit
def calculate_window_array_numba_full(L, DeltaXi, PowerPhi, rescaleR, window_function_numba):
    WindowArray = np.zeros((L, L, L), dtype=np.complex128)
    for i in range(-L, L):
        for j in range(-L, L):
            for k in range(-L, L):
                temp = (
                    PowerPhi[np.abs(i)]
                    * PowerPhi[np.abs(j)]
                    * PowerPhi[np.abs(k)]
                    * window_function_numba(i * DeltaXi, j * DeltaXi, k * DeltaXi, rescaleR)
                )
                WindowArray[i, j, k] += temp
    return WindowArray


@njit
def calculate_window_array_numba_half(L, DeltaXi, PowerPhi, rescaleR, window_function_numba):
    WindowArray = np.zeros((L, L, L), dtype=np.complex128)
    for i in range(-L // 2, L // 2):
        for j in range(-L // 2, L // 2):
            for k in range(-L // 2, L // 2):
                temp = (
                    PowerPhi[np.abs(i)]
                    * PowerPhi[np.abs(j)]
                    * PowerPhi[np.abs(k)]
                    * window_function_numba(i * DeltaXi, j * DeltaXi, k * DeltaXi, rescaleR)
                )
                WindowArray[i, j, k] += temp
    return WindowArray


@njit(parallel=True)
def calculate_w_numba(WindowArray):
    L = WindowArray.shape[0] - 1
    w = np.zeros((L, L, L), dtype=np.complex128)
    for x in prange(L):
        for y in prange(L):
            for z in prange(L):
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


# def specialized_convolution_3d(s, w, threads):
#     sc = rfftn(s, workers=threads)
#     sc *= w
#     result_convol3d = irfftn(sc, workers=threads)
#     return result_convol3d


def specialized_convolution_3d(s, w, threads):
    sc = fftn(s, workers=threads)
    sc *= w
    result_convol3d = ifftn(sc, workers=threads)
    return result_convol3d


def calculate_window_array_with_lm(L, DeltaXi, PowerPhi, rescaleR, l, m):
    window_function_mapping = {
        (0, 0): window_function_legendre_l0_m0_numba,
        (1, 0): window_function_legendre_l1_m0_numba,
        (1, -1): window_function_legendre_l1_m_minus1_numba,
        (1, 1): window_function_legendre_l1_m1_numba,
        (2, 0): window_function_legendre_l2_m0_numba,
        (2, -1): window_function_legendre_l2_m_minus1_numba,
        (2, 1): window_function_legendre_l2_m1_numba,
        (2, -2): window_function_legendre_l2_m_minus2_numba,
        (2, 2): window_function_legendre_l2_m2_numba,
        (3, 0): window_function_legendre_l3_m0_numba,
        (3, -1): window_function_legendre_l3_m_minus1_numba,
        (3, 1): window_function_legendre_l3_m1_numba,
        (3, -2): window_function_legendre_l3_m_minus2_numba,
        (3, 2): window_function_legendre_l3_m2_numba,
        (3, -3): window_function_legendre_l3_m_minus3_numba,
        (3, 3): window_function_legendre_l3_m3_numba,
        (4, 0): window_function_legendre_l4_m0_numba,
        (4, -1): window_function_legendre_l4_m_minus1_numba,
        (4, 1): window_function_legendre_l4_m1_numba,
        (4, -2): window_function_legendre_l4_m_minus2_numba,
        (4, 2): window_function_legendre_l4_m2_numba,
        (4, -3): window_function_legendre_l4_m_minus3_numba,
        (4, 3): window_function_legendre_l4_m3_numba,
        (4, -4): window_function_legendre_l4_m_minus4_numba,
        (4, 4): window_function_legendre_l4_m4_numba,
        (5, 0): window_function_legendre_l5_m0_numba,
        (5, -1): window_function_legendre_l5_m_minus1_numba,
        (5, 1): window_function_legendre_l5_m1_numba,
        (5, -2): window_function_legendre_l5_m_minus2_numba,
        (5, 2): window_function_legendre_l5_m2_numba,
        (5, -3): window_function_legendre_l5_m_minus3_numba,
        (5, 3): window_function_legendre_l5_m3_numba,
        (5, -4): window_function_legendre_l5_m_minus4_numba,
        (5, 4): window_function_legendre_l5_m4_numba,
        (5, -5): window_function_legendre_l5_m_minus5_numba,
        (5, 5): window_function_legendre_l5_m5_numba,
        (6, 0): window_function_legendre_l6_m0_numba,
        (6, -1): window_function_legendre_l6_m_minus1_numba,
        (6, 1): window_function_legendre_l6_m1_numba,
        (6, -2): window_function_legendre_l6_m_minus2_numba,
        (6, 2): window_function_legendre_l6_m2_numba,
        (6, -3): window_function_legendre_l6_m_minus3_numba,
        (6, 3): window_function_legendre_l6_m3_numba,
        (6, -4): window_function_legendre_l6_m_minus4_numba,
        (6, 4): window_function_legendre_l6_m4_numba,
        (6, -5): window_function_legendre_l6_m_minus5_numba,
        (6, 5): window_function_legendre_l6_m5_numba,
        (6, -6): window_function_legendre_l6_m_minus6_numba,
        (6, 6): window_function_legendre_l6_m6_numba,
        (7, 0): window_function_legendre_l7_m0_numba,
        (7, -1): window_function_legendre_l7_m_minus1_numba,
        (7, 1): window_function_legendre_l7_m1_numba,
        (7, -2): window_function_legendre_l7_m_minus2_numba,
        (7, 2): window_function_legendre_l7_m2_numba,
        (7, -3): window_function_legendre_l7_m_minus3_numba,
        (7, 3): window_function_legendre_l7_m3_numba,
        (7, -4): window_function_legendre_l7_m_minus4_numba,
        (7, 4): window_function_legendre_l7_m4_numba,
        (7, -5): window_function_legendre_l7_m_minus5_numba,
        (7, 5): window_function_legendre_l7_m5_numba,
        (7, -6): window_function_legendre_l7_m_minus6_numba,
        (7, 6): window_function_legendre_l7_m6_numba,
        (7, -7): window_function_legendre_l7_m_minus7_numba,
        (7, 7): window_function_legendre_l7_m7_numba,
    }

    if (l, m) not in window_function_mapping:
        raise ValueError(f"Unsupported combination of l={l} and m={m}. Only l=0,1,2,3 and valid m are supported.")

    window_function = window_function_mapping[(l, m)]

    return calculate_window_array_numba_full(L, DeltaXi, PowerPhi, rescaleR, window_function)
    # return calculate_window_array_numba_half(L, DeltaXi, PowerPhi, rescaleR, window_function)


if __name__ == "__main__":
    J = 8
    L = 1 << J

    SampRate = 1024
    SimBoxL = 1000

    Radius = 5
    wavelet = pywt.Wavelet("db2")
    phi, psi, x = wavelet.wavefun(level=10)
    phi_data = phi[:-1]
    work_dir = "/home/tristan/graduate/association_hermes/mdpl/data/r" + str(Radius)
    with open(work_dir + "/deltac_" + str(L) + "_005_r" + str(Radius) + "_pywt.pk", "rb") as f:
        deltac = pickle.load(f)
    R_vector = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120]
    for R1 in R_vector:
        file_dir = work_dir + "/R" + str(R1)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        for l in range(8):
            for m in range(-l, l + 1):
                print("working on l = ", l, " m = ", m)
                bandwidth = 1
                DeltaXi = 1.0 / (L)
                rescaleR = R1 * L / SimBoxL

                PowerPhi = power_spectrum(phi_data, 0, bandwidth, L * bandwidth)

                window_array = calculate_window_array_with_lm(L, DeltaXi, PowerPhi, rescaleR, l, m)

                # print("deltac shape: ", deltac.shape)
                # print("window_array shape: ", window_array.shape)
                shell_c = specialized_convolution_3d(deltac, window_array, 5)
                if m >= 0:
                    file_path = os.path.join(
                        file_dir,
                        "deltac_"
                        + str(L)
                        + "_005_r"
                        + str(Radius)
                        + "_R"
                        + str(R1)
                        + "_l"
                        + str(l)
                        + "_m"
                        + str(m)
                        + "_pywt.pk",
                    )
                    with open(
                        file_path,
                        "wb",
                    ) as f:
                        pickle.dump(shell_c, f)
                else:
                    file_path = os.path.join(
                        file_dir,
                        "deltac_"
                        + str(L)
                        + "_005_r"
                        + str(Radius)
                        + "_R"
                        + str(R1)
                        + "_l"
                        + str(l)
                        + "_m_minus"
                        + str(-m)
                        + "_pywt.pk",
                    )
                    with open(
                        file_path,
                        "wb",
                    ) as f:
                        pickle.dump(shell_c, f)
