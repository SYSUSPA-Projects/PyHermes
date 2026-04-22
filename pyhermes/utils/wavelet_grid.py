"""Wavelet scaling-function grids, particle projection, and spectra."""

import math

import numpy as np
import pywt
from numba import jit, njit


def do_wavelet(mode="db2", level=10):
    """Return sampled scaling-function values for a PyWavelets wavelet."""
    wavelet = pywt.Wavelet(mode)
    _phi, _, _ = wavelet.wavefun(level=level)
    return _phi[:-1]


def random_points_box(N, SimBoxL, ndim=3, rng=None, seed=None):
    """Draw uniformly distributed random points inside a periodic box."""
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    return rng.uniform(0.0, SimBoxL, size=(N, ndim))


@jit(nopython=True)
def scaling_function_numba(p, w, phi_data, SampRate=1024, J=8, SimBoxL=1000.0):
    """Project weighted particles onto the full 3D scaling-function grid."""
    L = 1 << J
    PhiSupport = phi_data.shape[0] // SampRate
    ScaleFactor = L / SimBoxL
    step = np.arange(PhiSupport, dtype=np.int32) * SampRate
    scale_p = p[:, :3] * ScaleFactor
    p_coarse = np.floor(scale_p).astype(np.int32)
    p_finer = ((scale_p - p_coarse) * SampRate).astype(np.int32)
    total = p.shape[0]
    s = np.zeros((L, L, L), dtype=np.float64)
    w = w.astype(np.float64)
    for num in range(total):
        ww = w[num]
        cx, cy, cz = p_coarse[num, 0], p_coarse[num, 1], p_coarse[num, 2]
        fx, fy, fz = p_finer[num, 0], p_finer[num, 1], p_finer[num, 2]
        for i in range(PhiSupport):
            phix = ww * phi_data[fx + step[i]]
            for j in range(PhiSupport):
                phixy = phix * phi_data[fy + step[j]]
                for k in range(PhiSupport):
                    s[cx - i, cy - j, cz - k] += phixy * phi_data[fz + step[k]]
    return s


@jit(nopython=True)
def int_data(data, ScaleFactor):
    """Return coarse-grid x-cell indices for positions scaled by ScaleFactor."""
    num = data.shape[0]
    out = np.empty(num, dtype=np.int32)
    for i in range(num):
        out[i] = int(np.floor(data[i, 0] * ScaleFactor))
    return out


@jit(nopython=True)
def bit(array, J, size_bit):
    """Keep the leading size_bit bits of integer grid coordinates at level J."""
    num = array.shape[0]
    result = np.empty(num, dtype=np.int32)
    shift = J - size_bit
    for i in range(num):
        result[i] = array[i] >> shift
    return result


@jit(nopython=True)
def scaling_function_numba_part(part, p, w, phi_data, core_width, SampRate=1024, J=8, SimBoxL=1000.0):
    """Project weighted particles onto one x-slab plus scaling-function padding."""
    L = 1 << J
    PhiSupport = phi_data.shape[0] // SampRate
    sew_width = PhiSupport - 1
    ScaleFactor = L / SimBoxL
    step = np.arange(PhiSupport, dtype=np.int32) * SampRate
    scale_p = p[:, :3] * ScaleFactor
    p_coarse = np.floor(scale_p).astype(np.int32)
    p_finer = ((scale_p - p_coarse) * SampRate).astype(np.int32)
    total = p.shape[0]
    expand_x = core_width + 2 * sew_width
    s = np.zeros((expand_x, L, L), dtype=np.float64)
    w = w.astype(np.float64)
    base = part * core_width
    for num in range(total):
        ww = w[num]
        cx, cy, cz = p_coarse[num, 0], p_coarse[num, 1], p_coarse[num, 2]
        fx, fy, fz = p_finer[num, 0], p_finer[num, 1], p_finer[num, 2]
        for i in range(PhiSupport):
            phix = ww * phi_data[fx + step[i]]
            x_local = (cx - i) - base + sew_width
            for j in range(PhiSupport):
                phixy = phix * phi_data[fy + step[j]]
                y = cy - j
                for k in range(PhiSupport):
                    s[x_local, y, cz - k] += phixy * phi_data[fz + step[k]]
    return s


def partition_data_single(origin_data, shrink_data, part):
    """Select rows from origin_data whose partition labels equal part."""
    return origin_data[shrink_data == part]


def power_spectrum(v, k0, k1, N_k, SampRate):
    """Return the squared magnitude of the sampled 1D Fourier spectrum."""
    s = spectrum_vectorized(v, k0, k1, N_k, SampRate)
    p = np.zeros(N_k + 1, dtype=np.double)
    for i in range(N_k + 1):
        p[i] = s[2 * i] ** 2 + s[2 * i + 1] ** 2
    return p


def spectrum_vectorized(v, k0, k1, N_k, SampRate):
    """Compute interleaved real/imaginary Fourier samples of a 1D vector."""
    N_x = v.shape[0]
    x0 = 0
    x1 = v.shape[0] / SampRate
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


@jit(nopython=True)
def phi_at_pos_numba(pos, phi_data, ScaleFactor, SampRate, PhiSupport):
    """Evaluate local scaling-function stencil values around each position."""
    step = np.arange(PhiSupport) * SampRate
    scale_pos = pos * ScaleFactor
    pos_coarse = np.floor(scale_pos).astype(np.int32)
    pos_finer = ((scale_pos - pos_coarse) * SampRate).astype(np.int32)
    total = scale_pos.shape[0]
    phi_local = np.zeros((total, PhiSupport, PhiSupport, PhiSupport), dtype=np.float64)
    for num in range(total):
        fx, fy, fz = pos_finer[num]
        for i in range(PhiSupport):
            phix = phi_data[fx + step[i]]
            for j in range(PhiSupport):
                phixy = phix * phi_data[fy + step[j]]
                for k in range(PhiSupport):
                    phi_local[num, -i, -j, -k] = phixy * phi_data[fz + step[k]]
    return pos_coarse, phi_local


@njit
def n_at_pos_numba(n_output, pos_scaled, epsilon, phi_data, L, SampRate, PhiSupport,
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

        xf = int((sx - xc) * SampRate)
        yf = int((sy - yc) * SampRate)
        zf = int((sz - zc) * SampRate)

        acc = 0.0
        for i in range(PhiSupport):
            xi = (xc - i) & Lmask
            phix = phi_data[xf + i * SampRate]
            for j in range(PhiSupport):
                yi = (yc - j) & Lmask
                phiy = phi_data[yf + j * SampRate]
                for k in range(PhiSupport):
                    zi = (zc - k) & Lmask
                    phiz = phi_data[zf + k * SampRate]
                    acc += epsilon[xi, yi, zi] * phix * phiy * phiz

        n_output[idx] = acc
