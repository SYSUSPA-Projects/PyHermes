"""Wavelet scaling-function grids, particle projection, and spectra."""

import math

import numpy as np
import pywt
from numba import njit


def do_wavelet(mode="db2", level=10):
    """Return sampled scaling-function values for a PyWavelets wavelet."""
    wavelet = pywt.Wavelet(mode)
    _phi, _, _ = wavelet.wavefun(level=level)
    return _phi[:-1]


def random_points_box(N, box_size, ndim=3, rng=None, seed=None):
    """Draw uniformly distributed random points inside a periodic box."""
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    return rng.uniform(0.0, box_size, size=(N, ndim))


@njit
def _scaling_function_numba_impl(p, w, phi_array, output_x, base_x, x_offset, phi_resolution, J, box_size):
    L = 1 << J
    phi_support = phi_array.shape[0] // phi_resolution
    scale_factor = L / box_size
    step = np.arange(phi_support, dtype=np.int32) * phi_resolution
    scale_p = p[:, :3] * scale_factor
    p_coarse = np.floor(scale_p).astype(np.int32)
    p_finer = ((scale_p - p_coarse) * phi_resolution).astype(np.int32)
    total = p.shape[0]
    s = np.zeros((output_x, L, L), dtype=np.float64)
    w = w.astype(np.float64)
    for num in range(total):
        ww = w[num]
        cx, cy, cz = p_coarse[num, 0], p_coarse[num, 1], p_coarse[num, 2]
        fx, fy, fz = p_finer[num, 0], p_finer[num, 1], p_finer[num, 2]
        for i in range(phi_support):
            phix = ww * phi_array[fx + step[i]]
            x_local = (cx - i) - base_x + x_offset
            for j in range(phi_support):
                phixy = phix * phi_array[fy + step[j]]
                for k in range(phi_support):
                    s[x_local, cy - j, cz - k] += phixy * phi_array[fz + step[k]]
    return s


@njit
def scaling_function_numba(p, w, phi_array, phi_resolution=1024, J=8, box_size=1000.0):
    """Project weighted particles onto the full 3D scaling-function grid."""
    L = 1 << J
    return _scaling_function_numba_impl(
        p, w, phi_array, L, 0, 0, phi_resolution, J, box_size
    )


@njit
def scaling_function_numba_part(part, p, w, phi_array, core_width, phi_resolution=1024, J=8, box_size=1000.0):
    """Project weighted particles onto one x-slab plus scaling-function padding."""
    phi_support = phi_array.shape[0] // phi_resolution
    sew_width = phi_support - 1
    expand_x = core_width + 2 * sew_width
    base = part * core_width
    return _scaling_function_numba_impl(
        p, w, phi_array, expand_x, base, sew_width, phi_resolution, J, box_size
    )


def power_spectrum(v, k0, k1, N_k, phi_resolution):
    """Return the squared magnitude of the sampled 1D Fourier spectrum."""
    s = spectrum_vectorized(v, k0, k1, N_k, phi_resolution)
    p = np.zeros(N_k + 1, dtype=np.double)
    for i in range(N_k + 1):
        p[i] = s[2 * i] ** 2 + s[2 * i + 1] ** 2
    return p


def spectrum_vectorized(v, k0, k1, N_k, phi_resolution):
    """Compute interleaved real/imaginary Fourier samples of a 1D vector."""
    N_x = v.shape[0]
    x0 = 0
    x1 = v.shape[0] / phi_resolution
    Delta_x = (x1 - x0) / N_x
    Delta_k = (k1 - k0) / N_k
    x = np.arange(N_x) * Delta_x
    k = np.arange(N_k + 1) * Delta_k
    x_grid, k_grid = np.meshgrid(x, k)
    s_real = np.sum(v * Delta_x * np.cos(-2 * np.pi * k_grid * x_grid), axis=1)
    s_imag = np.sum(v * Delta_x * np.sin(-2 * np.pi * k_grid * x_grid), axis=1)
    s = np.empty((N_k + 1) * 2, dtype=np.double)
    s[::2] = s_real
    s[1::2] = s_imag
    return s


@njit
def phi_at_pos_numba(pos, phi_array, scale_factor, phi_resolution, phi_support):
    """Evaluate local scaling-function stencil values around each position."""
    step = np.arange(phi_support) * phi_resolution
    scale_pos = pos * scale_factor
    pos_coarse = np.floor(scale_pos).astype(np.int32)
    pos_finer = ((scale_pos - pos_coarse) * phi_resolution).astype(np.int32)
    total = scale_pos.shape[0]
    phi_local = np.zeros((total, phi_support, phi_support, phi_support), dtype=np.float64)
    for num in range(total):
        fx, fy, fz = pos_finer[num]
        for i in range(phi_support):
            phix = phi_array[fx + step[i]]
            for j in range(phi_support):
                phixy = phix * phi_array[fy + step[j]]
                for k in range(phi_support):
                    phi_local[num, -i, -j, -k] = phixy * phi_array[fz + step[k]]
    return pos_coarse, phi_local


@njit
def n_at_pos_numba(n_output, pos_scaled, epsilon, phi_array, L, phi_resolution, phi_support,
                   dx=0.0, dy=0.0, dz=0.0):
    """Evaluate normalized ``n(x)`` on scaled grid positions, with optional offsets."""
    Lmask = L - 1
    n = pos_scaled.shape[0]

    for idx in range(n):
        sx = pos_scaled[idx, 0] + dx
        sy = pos_scaled[idx, 1] + dy
        sz = pos_scaled[idx, 2] + dz

        xc = int(math.floor(sx))
        yc = int(math.floor(sy))
        zc = int(math.floor(sz))

        xf = int((sx - xc) * phi_resolution)
        yf = int((sy - yc) * phi_resolution)
        zf = int((sz - zc) * phi_resolution)

        acc = 0.0
        for i in range(phi_support):
            xi = (xc - i) & Lmask
            phix = phi_array[xf + i * phi_resolution]
            for j in range(phi_support):
                yi = (yc - j) & Lmask
                phiy = phi_array[yf + j * phi_resolution]
                for k in range(phi_support):
                    zi = (zc - k) & Lmask
                    phiz = phi_array[zf + k * phi_resolution]
                    acc += epsilon[xi, yi, zi] * phix * phiy * phiz

        n_output[idx] = acc
