import inspect
import math
import time
import warnings
from pathlib import Path

import pywt
import numpy as np
from scipy.fft import rfftn, irfftn, fftn, ifftn
from scipy.special import spherical_jn, sph_harm
from numba import cuda, int16, jit, njit, prange
from numba.core.errors import NumbaExperimentalFeatureWarning

from pyhermes.param.logbase import setup_logger
from pyhermes.utils import func_util
from pyhermes.utils.func_util import get_fname_info


_NUMBA_CONFIGURED = False
_NUMBA_THREADS = None


def configure(threads=1):
    """
    Configure Numba threads ONCE for this process.
    Call this before any @njit(parallel=True) function runs.
    """
    global _NUMBA_CONFIGURED, _NUMBA_THREADS
    if _NUMBA_CONFIGURED:
        return
    from numba import set_num_threads, get_num_threads
    set_num_threads(max(1, int(threads)))
    _NUMBA_THREADS = int(get_num_threads())
    _NUMBA_CONFIGURED = True


def do_wavelet(mode="db2", level=10):
    wavelet = pywt.Wavelet(mode)
    _phi, _, _ = wavelet.wavefun(level=level)
    phi_data = _phi[:-1]
    return phi_data


def random_points_box(N, SimBoxL, ndim=3, rng=None, seed=None):
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    return rng.uniform(0.0, SimBoxL, size=(N, ndim))


# Suppress NumbaExperimentalFeatureWarning
warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

# ---------------------------------------------------------------
# ------------- ↓ Numerical function for convols ↓ --------------
# ---------------------------------------------------------------


### ↓ Window functions ↓ ###
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


@njit
def window_function_gauss_numba(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    # Use np.where to handle the k == 0 case
    Phase = 2 * np.pi * k * R
    result = np.exp(-(Phase**2) / 2)
    return result


@njit
def window_function_gauss_shell(ki, kj, kk, R1, R2):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0:
        return 1
    Phase1 = 2 * np.pi * k * R1
    Phase2 = 2 * np.pi * k * R2
    result = (
        (R2 * R2 * np.cos(Phase1) + R1 * R1 * np.sin(Phase1) / Phase1) / (R1 * R1 + R2 * R2) * np.exp(-(Phase2**2) / 2)
    )
    return result


@njit
def window_function_Tshell(ki, kj, kk, R1, R2):
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
def window_function_gauss_direvative_wavalet(ki, kj, kk, R):
    k = np.sqrt(ki**2 + kj**2 + kk**2)
    Phase = 2 * np.pi * k * R
    norm = 2 ** (7 / 4) / np.sqrt(15) * (2 * np.pi) ** (3 / 4) * R ** (3 / 2)
    result = norm * Phase**2 * np.exp(-(Phase**2) / 2)
    return result


def window_function_cylinder(ki, kj, kk, R, h):
    k1 = np.sqrt(ki**2 + kj**2)
    if kk == 0:
        part1 = 1
    else:
        part1 = np.sin(2 * np.pi * kk * h / 2) / (2 * np.pi * kk * h / 2)

    if k1 == 0:
        sum_val = 1
    else:
        sum_val = 2 * jn(1, 2 * np.pi * k1 * R) / (2 * np.pi * k1 * R)
    return (sum_val * part1) * np.pi * h * R**2
### ↑ Window functions ↑ ###


def set_window_function(w_type, verbose=True):
    w_type_dict = {
        "shell": window_function_shell_numba,
        "sphere": window_function_sphere_numba,
        "gaussian": window_function_gauss_numba,
        "gaussian_shell": window_function_gauss_shell,
        "Tshell": window_function_Tshell,
        "gaussian_direvative_wavalet": window_function_gauss_direvative_wavalet,
        "cylinder": window_function_cylinder,
    }
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    if w_type in w_type_dict:
        verbose and logger.info(f"Using window function: {w_type}")
        read_function = w_type_dict[w_type]
        return read_function
    else:
        supported_w_type = ", ".join(w_type_dict.keys())
        logger.error(f"Unsupported input window function type: {w_type}")
        logger.error(f"Supported types: {supported_w_type}")
        logger.error("Please see the document for details")
        func_util.safe_exit(1)


@njit(parallel=True)
def calculate_window_array_numba(L, bandwidth, DeltaXi, PowerPhi, window_function_numba, *args):
    WindowArray = np.zeros((L + 1, L + 1, L + 1))
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
                                    (ii * L + i) * DeltaXi, (jj * L + j) * DeltaXi, (kk * L + k) * DeltaXi, *args
                                )
                            )
                WindowArray[i, j, k] = temp
    return WindowArray


def call_calculate_window_array(L, bandwidth, DeltaXi, PowerPhi, window_function_numba, **kwargs):
    """
    Helper function to dynamically call calculate_window_array_numba with the appropriate window function and parameters,
    handling both positional and keyword arguments.
    """
    _mod_name, _func_name = get_fname_info()
    logger = setup_logger(_mod_name, _func_name)
    # window_function = globals()[window_type]
    sig = inspect.signature(window_function_numba)
    params = sig.parameters  # Parameters of the window function
    expected_args = list(params.keys())[3:]  # Exclude 'ki', 'kj', 'kk' which are handled separately
    # If kwargs are provided, disregard args and validate kwargs
    provided_args = kwargs.keys()
    missing_args = [arg for arg in expected_args if arg not in provided_args]
    if missing_args:
        # raise ValueError(f"Missing keyword arguments: {missing_args}")
        source_code = inspect.getsource(window_function_numba)
        # logger.error(black.format_str(source_code, mode=black.Mode()))
        logger.error("\n" + source_code)
        logger.error(f"Missing keyword arguments: {missing_args}")
        logger.error(f"Please see the document for details")
        func_util.safe_exit(1)
    # Construct args from kwargs in the order expected by the window function
    ordered_args = [kwargs[arg] for arg in expected_args if arg in kwargs]
    # Call the core function with dynamically built arguments list
    return calculate_window_array_numba(L, bandwidth, DeltaXi, PowerPhi, window_function_numba, *ordered_args)


@njit(parallel=True)
def calculate_w_numba(WindowArray):
    L = WindowArray.shape[0] - 1
    w = np.zeros((L, L, L // 2 + 1))
    for x in prange(L):
        for y in prange(L):
            for z in prange(L // 2 + 1):
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


@jit(nopython=True)
def scaling_function_numba(p, w, phi_data, SampRate=1024, J=8, SimBoxL=1000.0):
    L = 1 << J
    PhiSupport = phi_data.shape[0] // SampRate
    ScaleFactor = L / SimBoxL
    step = np.arange(PhiSupport, dtype=np.int32) * SampRate
    scale_p  = p[:, :3] * ScaleFactor
    p_coarse = np.floor(scale_p).astype(np.int32)
    p_finer  = ((scale_p - p_coarse) * SampRate).astype(np.int32)
    total = p.shape[0]
    s = np.zeros((L, L, L), dtype=np.float64)
    w = w.astype(np.float64)
    for num in range(total):
        ww = w[num]  # <-- weight for this particle
        cx, cy, cz = p_coarse[num, 0], p_coarse[num, 1], p_coarse[num, 2]
        fx, fy, fz = p_finer[num, 0],  p_finer[num, 1],  p_finer[num, 2]
        for i in range(PhiSupport):
            phix = ww * phi_data[fx + step[i]]  
            for j in range(PhiSupport):
                phixy = phix * phi_data[fy + step[j]]
                for k in range(PhiSupport):
                    s[cx - i, cy - j, cz - k] += phixy * phi_data[fz + step[k]]
    return s


@jit(nopython=True)
def int_data(data, ScaleFactor):
    num = data.shape[0]
    out = np.empty(num, dtype=np.int32)
    for i in range(num):
        out[i] = int(np.floor(data[i, 0] * ScaleFactor))
    return out


## bit opereator in numba
@jit(nopython=True)
def bit(array, J, size_bit):
    num = array.shape[0]
    result = np.empty(num, dtype=np.int32)
    shift = J - size_bit
    for i in range(num):
        result[i] = array[i] >> shift
    return result


@jit(nopython=True)
def scaling_function_numba_part(part, p, w, phi_data, core_width, SampRate=1024, J=8, SimBoxL=1000.0):
    L = 1 << J
    PhiSupport = phi_data.shape[0] // SampRate
    sew_width  = PhiSupport - 1
    ScaleFactor = L / SimBoxL
    step = np.arange(PhiSupport, dtype=np.int32) * SampRate
    scale_p  = p[:, :3] * ScaleFactor
    p_coarse = np.floor(scale_p).astype(np.int32)
    p_finer  = ((scale_p - p_coarse) * SampRate).astype(np.int32)
    total = p.shape[0]
    expand_x = core_width + 2 * sew_width
    s = np.zeros((expand_x, L, L), dtype=np.float64)
    w = w.astype(np.float64)
    base = part * core_width
    for num in range(total):
        ww = w[num]  # <-- weight for this particle
        cx, cy, cz = p_coarse[num, 0], p_coarse[num, 1], p_coarse[num, 2]
        fx, fy, fz = p_finer[num, 0],  p_finer[num, 1],  p_finer[num, 2]
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
    dm_part = origin_data[shrink_data == part]
    return dm_part


def specialized_convolution_3d(s, w, threads):
    # Run FFt in multi-thread manner
    sc = rfftn(s, workers= threads)
    sc *= w
    result_convol3d = irfftn(sc, workers = threads)
    return result_convol3d


def specialized_convolution_3d_complex(s, w, threads):
    sc = fftn(s, workers=threads)
    sc *= w
    result_convol3d = ifftn(sc, workers=threads)
    return result_convol3d


def power_spectrum(v, k0, k1, N_k, SampRate):
    s = spectrum_vectorized(v, k0, k1, N_k, SampRate)
    p = np.zeros(N_k + 1, dtype=np.double)
    for i in range(N_k + 1):
        p[i] = s[2*i] ** 2 + s[2*i+1] ** 2
    return p


def spectrum_vectorized(v, k0, k1, N_k, SampRate):
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


# ----------------------------------------------------------------
# ------------- ↓ Numerical function for counting ↓ --------------
# ----------------------------------------------------------------

@jit(nopython=True)
def phi_at_pos_numba(pos, phi_data, ScaleFactor, SampRate, PhiSupport):
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
    """
    Evaluate normalized n(x) on a set of SCALED (GRID) positions.

    Inputs:
      pos_scaled : (N,3) positions in GRID units [0, L)
      dx,dy,dz   : optional GRID offsets added to every position
      epsilon    : (L,L,L) field coefficients (normalized convention)
      phi_data   : scaling-function samples
      L          : grid size (assumed power of two; periodic wrapping via & (L-1))
      SampRate   : fine sampling per cell
      PhiSupport : support size in coarse-cell units (e.g. 3)

    Output:
      n_output   : (N,) filled in-place
    """
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

# ------------------------------------------------------------
# ------------- ↓ Numerical function for 2pcf ↓ --------------
# ------------------------------------------------------------



# ------------------------------------------------------------
# ------------- ↓ Numerical function for 3pcf ↓ --------------
# ------------------------------------------------------------

def third_side(r1, r2, theta):
    """Return r23 from (r1, r2, theta) using the law of cosines."""
    return np.sqrt(r1**2 + r2**2 - 2.0 * r1 * r2 * np.cos(theta))


@njit
def _k_norm(ki, kj, kk):
    return math.sqrt(ki * ki + kj * kj + kk * kk)


@njit
def _phase_from_kR(ki, kj, kk, R):
    return 2.0 * math.pi * _k_norm(ki, kj, kk) * R


@njit
def _angles_from_k(ki, kj, kk):
    k = _k_norm(ki, kj, kk)
    if k == 0.0:
        return 0.0, 0.0, 0.0

    mu = kk / k
    if mu > 1.0:
        mu = 1.0
    elif mu < -1.0:
        mu = -1.0

    theta = math.acos(mu)
    phi = math.atan2(kj, ki)
    return k, theta, phi


@njit
def _factorial_small(n):
    out = 1.0
    for i in range(2, n + 1):
        out *= i
    return out


@njit
def spherical_jn_numba(l, x):
    if l < 0 or l > 7:
        return np.nan

    if x == 0.0:
        if l == 0:
            return 1.0
        return 0.0

    sx = math.sin(x)
    cx = math.cos(x)
    j0 = sx / x
    if l == 0:
        return j0

    j1 = sx / (x * x) - cx / x
    if l == 1:
        return j1

    jm1 = j0
    jcur = j1
    for ell in range(1, l):
        jnext = ((2.0 * ell + 1.0) / x) * jcur - jm1
        jm1 = jcur
        jcur = jnext

    return jcur


@njit
def assoc_legendre_numba(l, m, x):
    if l < 0 or l > 7:
        return np.nan
    if m < 0 or m > l:
        return np.nan

    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt(max(0.0, 1.0 - x * x))
        fact = 1.0
        for _ in range(m):
            pmm *= -fact * somx2
            fact += 2.0

    if l == m:
        return pmm

    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1

    pll_minus2 = pmm
    pll_minus1 = pmmp1
    pll = 0.0
    for ell in range(m + 2, l + 1):
        pll = ((2.0 * ell - 1.0) * x * pll_minus1 - (ell + m - 1.0) * pll_minus2) / (ell - m)
        pll_minus2 = pll_minus1
        pll_minus1 = pll

    return pll


@njit
def spherical_harmonic_numba(l, m, ki, kj, kk):
    if l < 0 or l > 7:
        return np.nan + 0.0j
    if abs(m) > l:
        return 0.0 + 0.0j

    k, theta, phi = _angles_from_k(ki, kj, kk)
    if k == 0.0:
        if l == 0 and m == 0:
            return 1.0 / math.sqrt(4.0 * math.pi) + 0.0j
        return 0.0 + 0.0j

    x = math.cos(theta)
    if m >= 0:
        P = assoc_legendre_numba(l, m, x)
        norm = math.sqrt(
            (2.0 * l + 1.0) / (4.0 * math.pi) *
            _factorial_small(l - m) / _factorial_small(l + m)
        )
        phase = complex(math.cos(m * phi), math.sin(m * phi))
        return norm * P * phase

    mp = -m
    ypos = spherical_harmonic_numba(l, mp, ki, kj, kk)
    sign = -1.0 if (mp % 2 == 1) else 1.0
    return sign * np.conj(ypos)


@njit
def window_function_legendre_numba(ki, kj, kk, R, l, m):
    if l < 0 or l > 7:
        return np.nan + 0.0j
    if abs(m) > l:
        return 0.0 + 0.0j

    phase = _phase_from_kR(ki, kj, kk, R)
    jl = spherical_jn_numba(l, phase)
    ylm = spherical_harmonic_numba(l, m, ki, kj, kk)
    return jl * ylm


def window_function_legendre(ki, kj, kk, R, l, m, use_fast=True):
    if use_fast and l <= 7:
        return window_function_legendre_numba(ki, kj, kk, R, l, m)

    if abs(m) > l:
        return 0.0 + 0.0j

    k = np.sqrt(ki**2 + kj**2 + kk**2)
    if k == 0.0:
        if l == 0 and m == 0:
            return 1.0 / np.sqrt(4.0 * np.pi) + 0.0j
        return 0.0 + 0.0j

    theta = np.arccos(np.clip(kk / k, -1.0, 1.0))
    phi = np.arctan2(kj, ki)
    phase = 2.0 * np.pi * k * R
    return spherical_jn(l, phase) * sph_harm(m, l, phi, theta)


@njit
def calculate_window_array_legendre_numba(L, DeltaXi, PowerPhi, rescaleR, l, m):
    window_array = np.zeros((L, L, L), dtype=np.complex128)
    for i in range(-L, L):
        ii = i % L
        pi = PowerPhi[abs(i)]
        for j in range(-L, L):
            jj = j % L
            pij = pi * PowerPhi[abs(j)]
            for k in range(-L, L):
                kk = k % L
                window_array[ii, jj, kk] += (
                    pij
                    * PowerPhi[abs(k)]
                    * window_function_legendre_numba(i * DeltaXi, j * DeltaXi, k * DeltaXi, rescaleR, l, m)
                )
    return window_array


def calculate_window_array_legendre(L, DeltaXi, PowerPhi, rescaleR, l, m):
    return calculate_window_array_legendre_numba(L, DeltaXi, PowerPhi, rescaleR, l, m)


def cal_gamma(phi_data, PhiSupport, SampRate):
    gamma = np.zeros((PhiSupport, PhiSupport))
    for l1 in range(PhiSupport):
        for l2 in range(PhiSupport):
            rolled_phi1 = np.roll(phi_data, l1 * SampRate)
            rolled_phi2 = np.roll(phi_data, l2 * SampRate)
            gamma[l1, l2] = np.sum(phi_data * rolled_phi1 * rolled_phi2) / SampRate
    return gamma


@cuda.jit
def compute_3d_result_gpu(data, data_R1, data_R2, Gamma, result, L, PhiSupport):
    lx, ly, lz = cuda.grid(3)
    if lx < L and ly < L and lz < L:
        sum_over_l1 = 0.0 + 0.0j
        for l1x in range(PhiSupport):
            index_l1x = (lx - l1x) % L
            for l1y in range(PhiSupport):
                index_l1y = (ly - l1y) % L
                for l1z in range(PhiSupport):
                    index_l1z = (lz - l1z) % L
                    sum_over_l2 = 0.0 + 0.0j
                    for l2x in range(PhiSupport):
                        index_l2x = (lx - l2x) % L
                        res_y = 0.0 + 0.0j
                        for l2y in range(PhiSupport):
                            index_l2y = (ly - l2y) % L
                            res_z = 0.0 + 0.0j
                            for l2z in range(PhiSupport):
                                index_l2z = (lz - l2z) % L
                                res_z += Gamma[l1z, l2z] * data_R2[index_l2x, index_l2y, index_l2z]
                            res_y += Gamma[l1y, l2y] * res_z
                        sum_over_l2 += Gamma[l1x, l2x] * res_y
                    sum_over_l1 += data_R1[index_l1x, index_l1y, index_l1z] * sum_over_l2
        result[lx, ly, lz] = data[lx, ly, lz] * sum_over_l1


def combine_multipole_m_terms(m_values, l):
    coeff = complex(m_values[0])
    for m in range(1, l + 1):
        coeff += ((-1) ** m) * complex(m_values[m])
        coeff += ((-1) ** (-m)) * np.conj(complex(m_values[m]))
    coeff *= (-1) ** l
    return coeff.real


def _cache_file_path(cache_dir, radius, l, m):
    sign = "m" if m >= 0 else "m_minus"
    suffix = f"{m}" if m >= 0 else f"{-m}"
    return Path(cache_dir) / f"R{radius:g}_l{l}_{sign}{suffix}.npy"


def _stream_convolution_fields(field, radius, l, threads, cache_multipole_fields=False, cache_dir=""):
    delta_xi = 1.0 / field.L
    power_phi = power_spectrum(field.phi_data, 0, field.bandwidth, field.L * field.bandwidth, field.SampRate)
    rescaleR = radius * field.ScaleFactor
    m_fields = []
    for m in range(-l, l + 1):
        cached = None
        cache_path = None
        if cache_multipole_fields and cache_dir:
            cache_path = _cache_file_path(cache_dir, radius, l, m)
            if cache_path.exists():
                cached = np.load(cache_path)
        if cached is None:
            window_array = calculate_window_array_legendre(field.L, delta_xi, power_phi, rescaleR, l, m)
            cached = specialized_convolution_3d_complex(field.epsilon, window_array, threads=threads)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, cached)
        m_fields.append(np.ascontiguousarray(cached, dtype=np.complex128))
    return m_fields


def calc_DDD_multipole(
    deltaD1, deltaD2, deltaD3,
    r1, r2, l_max,
    gpu_device_id=0,
    cache_multipole_fields=False,
    cache_dir="",
    threads=1,
    progress_callback=None,
    m_progress_callback=None,
):
    if not cuda.is_available():
        raise RuntimeError("CUDA is required for Corr_3PCF_Multipole, but no CUDA device is available.")

    if l_max < 0:
        raise ValueError("l_max must be non-negative.")
    if l_max > 7:
        raise ValueError("The current fast multipole implementation supports l_max <= 7.")

    cuda.select_device(int(gpu_device_id))
    gamma = np.ascontiguousarray(cal_gamma(deltaD1.phi_data, deltaD1.PhiSupport, deltaD1.SampRate), dtype=np.float64)
    gamma_gpu = cuda.to_device(gamma)
    data_gpu = cuda.to_device(np.ascontiguousarray(deltaD1.epsilon, dtype=np.float64))
    result_gpu = cuda.device_array(deltaD1.epsilon.shape, dtype=np.complex128)

    l_values = np.arange(l_max + 1, dtype=np.int32)
    ddd_l = np.empty(l_max + 1, dtype=np.float64)
    total_m_tasks = (l_max + 1) * (l_max + 2) // 2
    completed_m_tasks = 0

    rho = 1.0 / deltaD1.V
    rho3 = rho ** 3
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        (deltaD1.L + threads_per_block[0] - 1) // threads_per_block[0],
        (deltaD1.L + threads_per_block[1] - 1) // threads_per_block[1],
        (deltaD1.L + threads_per_block[2] - 1) // threads_per_block[2],
    )

    for l in range(l_max + 1):
        t_l_start = time.perf_counter()
        fields_r1 = _stream_convolution_fields(
            deltaD2, r1, l, threads=threads,
            cache_multipole_fields=cache_multipole_fields,
            cache_dir=cache_dir,
        )
        fields_r2 = _stream_convolution_fields(
            deltaD3, r2, l, threads=threads,
            cache_multipole_fields=cache_multipole_fields,
            cache_dir=cache_dir,
        )
        m_values = np.empty(l + 1, dtype=np.complex128)
        for m in range(0, l + 1):
            t_m_start = time.perf_counter()
            idx_r1 = m + l
            idx_r2 = l if m == 0 else l - m
            data_r1_gpu = cuda.to_device(fields_r1[idx_r1])
            data_r2_gpu = cuda.to_device(fields_r2[idx_r2])
            compute_3d_result_gpu[blocks_per_grid, threads_per_block](
                data_gpu, data_r1_gpu, data_r2_gpu, gamma_gpu, result_gpu, deltaD1.L, deltaD1.PhiSupport
            )
            cuda.synchronize()
            result = result_gpu.copy_to_host()
            m_values[m] = (4.0 * np.pi) * np.sum(result)
            del data_r1_gpu
            del data_r2_gpu
            if m_progress_callback is not None:
                completed_m_tasks += 1
                m_progress_callback(
                    l=l,
                    l_max=l_max,
                    m=m,
                    m_max=l,
                    value=m_values[m],
                    elapsed_sec=time.perf_counter() - t_m_start,
                    completed_m_tasks=completed_m_tasks,
                    total_m_tasks=total_m_tasks,
                )
        ddd_l[l] = combine_multipole_m_terms(m_values, l)
        del fields_r1
        del fields_r2
        if progress_callback is not None:
            progress_callback(
                l=l,
                l_max=l_max,
                ddd_l=float(ddd_l[l]),
                zeta_l=float(ddd_l[l] / rho3),
                elapsed_sec=time.perf_counter() - t_l_start,
                completed_m_tasks=completed_m_tasks,
                total_m_tasks=total_m_tasks,
            )

    return l_values, ddd_l


@njit
def generate_triangle_offsets(R1, R2, theta, phi, costheta1, alpha):
    """
    Generate offsets (x2,y2,z2) and (x3,y3,z3) that form a triangle with:
      |r1| = R1, |r2| = R2, angle(r1,r2) = theta

    All lengths here are in SCALED (GRID) units.
    """
    sintheta1 = math.sqrt(1.0 - costheta1 * costheta1)
    dx1 = sintheta1 * math.cos(phi)
    dy1 = sintheta1 * math.sin(phi)
    dz1 = costheta1

    x2 = R1 * dx1
    y2 = R1 * dy1
    z2 = R1 * dz1

    if abs(dx1) > 1e-10 or abs(dy1) > 1e-10:
        ox1, oy1, oz1 = -dy1, dx1, 0.0
    else:
        ox1, oy1, oz1 = 0.0, -dz1, dy1

    norm = math.sqrt(ox1 * ox1 + oy1 * oy1 + oz1 * oz1)
    ox1 /= norm
    oy1 /= norm
    oz1 /= norm

    ox2 = dy1 * oz1 - dz1 * oy1
    oy2 = dz1 * ox1 - dx1 * oz1
    oz2 = dx1 * oy1 - dy1 * ox1

    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)

    dx2 = math.cos(theta) * dx1 + math.sin(theta) * (cos_alpha * ox1 + sin_alpha * ox2)
    dy2 = math.cos(theta) * dy1 + math.sin(theta) * (cos_alpha * oy1 + sin_alpha * oy2)
    dz2 = math.cos(theta) * dz1 + math.sin(theta) * (cos_alpha * oz1 + sin_alpha * oz2)

    x3 = R2 * dx2
    y3 = R2 * dy2
    z3 = R2 * dz2

    return x2, y2, z2, x3, y3, z3


@njit
def calc_DDD_mc_pos_center_fast(
    R1_scaled, R2_scaled, theta,
    centers_scaled, n_rot,
    R, epsilon2, epsilon3,
    phi_data, L, SampRate, PhiSupport,
    seed_base_rot=-1, theta_index=-1
):
    
    """
    Method 1 (fast for small samples; centers = particle positions)

    Estimates DDD = < D1 * D2 * D3 > by:
      - sampling triangle orientations (n_rot)
      - evaluating D2(x+r1) and D3(x+r2) on centers_scaled
      - multiplying by constant R to account for the (implicit) D1 term
        under the normalized-density convention (R = 1/V_scaled).

    Best use case:
      - Halo / galaxy catalogs with relatively small N (e.g. ~1e5–1e6),
        where using real object centers is faster than using many random centers.

    Notes:
      - centers_scaled must be in SCALED (GRID) coordinates [0, L).
      - Returns a scalar DDD (normalized-grid convention).
    """
    npos = centers_scaled.shape[0]
    two_pi = 2.0 * math.pi
    n2 = np.empty(npos, dtype=np.float64)
    n3 = np.empty(npos, dtype=np.float64)
    total_sum = 0.0
    for irot in range(n_rot):
        if seed_base_rot >= 0:
            seed_rot = seed_base_rot + irot
            if theta_index >= 0:
                seed_rot += theta_index * 1000003
            np.random.seed(seed_rot)
        a, b, c = np.random.rand(3)
        phi = two_pi * a
        costheta1 = 2.0 * b - 1.0
        alpha = two_pi * c
        x2, y2, z2, x3, y3, z3 = generate_triangle_offsets(R1_scaled, R2_scaled, theta, phi, costheta1, alpha)
        n_at_pos_numba(n2, centers_scaled, epsilon2, phi_data, L, SampRate, PhiSupport, x2, y2, z2)
        n_at_pos_numba(n3, centers_scaled, epsilon3, phi_data, L, SampRate, PhiSupport, x3, y3, z3)
        s = 0.0
        for i in range(npos):
            s += n2[i] * n3[i]
        total_sum += s
    return R * total_sum / (n_rot * npos)


@njit
def calc_DDD_mc_random_center(
    R1_scaled, R2_scaled, theta,
    centers_scaled, n_rot,
    epsilon1, epsilon2, epsilon3,
    phi_data, L, SampRate, PhiSupport,
    seed_base_rot=-1, theta_index=-1
):
    """
    Method 2 (recommended default; centers = uniform random points)

    Estimates DDD = < D1 * D2 * D3 > by:
      - sampling triangle orientations (n_rot)
      - averaging over centers_scaled which must be uniformly distributed
        in the periodic box (SCALED/GRID coordinates).

    Best use case:
      - Very large underlying particle sets (e.g. dark matter),
        where volume-averaged 3PCF is desired and random-center sampling
        provides the cleanest estimator in a periodic box.

    Notes:
      - centers_scaled must be in SCALED (GRID) coordinates [0, L).
      - Returns a scalar DDD (normalized-grid convention).
    """

    npos = centers_scaled.shape[0]
    two_pi = 2.0 * math.pi

    n1 = np.empty(npos, dtype=np.float64)
    n2 = np.empty(npos, dtype=np.float64)
    n3 = np.empty(npos, dtype=np.float64)

    n_at_pos_numba(n1, centers_scaled, epsilon1, phi_data, L, SampRate, PhiSupport)

    total_sum = 0.0

    for irot in range(n_rot):
        if seed_base_rot >= 0:
            seed_rot = seed_base_rot + irot
            if theta_index >= 0:
                seed_rot += theta_index * 1000003
            np.random.seed(seed_rot)
        a, b, c = np.random.rand(3)
        phi = two_pi * a
        costheta1 = 2.0 * b - 1.0
        alpha = two_pi * c

        x2, y2, z2, x3, y3, z3 = generate_triangle_offsets(R1_scaled, R2_scaled, theta, phi, costheta1, alpha)

        n_at_pos_numba(n2, centers_scaled, epsilon2, phi_data, L, SampRate, PhiSupport, x2, y2, z2)
        n_at_pos_numba(n3, centers_scaled, epsilon3, phi_data, L, SampRate, PhiSupport, x3, y3, z3)

        s = 0.0
        for i in range(npos):
            s += n1[i] * n2[i] * n3[i]
        total_sum += s

    return total_sum / (n_rot * npos)


@njit
def calc_DDD_RDD_mc_pos_center_legacy(
    R1_scaled, R2_scaled, theta,
    pos_scaled, rand_scaled, n_rot,
    R, epsilon2, epsilon3,
    phi_data, L, SampRate, PhiSupport,
    seed=-1
):
    """
    Method 3 (legacy estimator; centers = particle positions + matched random centers)

    Computes the control-variate quantity:
      DDD_RDD = < D1*D2*D3 > - < R1*D2*D3 >
    by:
      - using pos_scaled as object centers
      - using rand_scaled as uniform random centers (same length as pos_scaled)
      - evaluating both (D2, D3) on pos_scaled and on rand_scaled each rotation.

    Best use case:
      - Keep for backward compatibility / cross-checks with older PyHermes workflow.

    Cost:
      - About ~2x of Method 1 for the same n_rot and sample size,
        because D2/D3 are evaluated on BOTH pos_scaled and rand_scaled.

    Notes:
      - pos_scaled and rand_scaled must be in SCALED (GRID) coordinates [0, L).
      - Returns a scalar DDD_RDD (normalized-grid convention).
    """
    if seed >= 0:
        np.random.seed(seed)
    npos = pos_scaled.shape[0]
    two_pi = 2.0 * math.pi
    n2 = np.empty(npos, dtype=np.float64)
    n3 = np.empty(npos, dtype=np.float64)
    n20 = np.empty(npos, dtype=np.float64)
    n30 = np.empty(npos, dtype=np.float64)
    total_sum = 0.0
    for _ in range(n_rot):
        a, b, c = np.random.rand(3)
        phi = two_pi * a
        costheta1 = 2.0 * b - 1.0
        alpha = two_pi * c
        x2, y2, z2, x3, y3, z3 = generate_triangle_offsets(R1_scaled, R2_scaled, theta, phi, costheta1, alpha)
        n_at_pos_numba(n2, pos_scaled, epsilon2, phi_data, L, SampRate, PhiSupport, x2, y2, z2)
        n_at_pos_numba(n3, pos_scaled, epsilon3, phi_data, L, SampRate, PhiSupport, x3, y3, z3)
        n_at_pos_numba(n20, rand_scaled, epsilon2, phi_data, L, SampRate, PhiSupport, x2, y2, z2)
        n_at_pos_numba(n30, rand_scaled, epsilon3, phi_data, L, SampRate, PhiSupport, x3, y3, z3)
        s = 0.0
        for i in range(npos):
            s += n2[i] * n3[i] - n20[i] * n30[i]
        total_sum += s
    return R * total_sum / (n_rot * npos)

# def cal_gamma(phi_data, PhiSupport, SampRate):
#     Gamma = np.zeros((PhiSupport, PhiSupport))
#     for l1 in range(PhiSupport):
#         for l2 in range(PhiSupport):
#             rolled_phi1 = np.roll(phi_data, l1 * SampRate)
#             rolled_phi2 = np.roll(phi_data, l2 * SampRate)
#             Gamma[l1, l2] = np.sum(phi_data * rolled_phi1 * rolled_phi2) / SampRate
#     return Gamma

# @cuda.jit
# def compute_3d_result_gpu(data, data_R1, data_R2, Gamma, result, L, PhiSupport):
#     lx, ly, lz = cuda.grid(3)
#     if lx < L and ly < L and lz < L:
#         sum_over_l1 = 0
#         for l1x in range(PhiSupport):
#             index_l1x = (lx - l1x) % L
#             for l1y in range(PhiSupport):
#                 index_l1y = (ly - l1y) % L
#                 for l1z in range(PhiSupport):
#                     index_l1z = (lz - l1z) % L
#                     sum_over_l2 = 0
#                     for l2x in range(PhiSupport):
#                         index_l2x = (lx - l2x) % L
#                         res_y = 0
#                         for l2y in range(PhiSupport):
#                             index_l2y = (ly - l2y) % L
#                             res_z = 0
#                             for l2z in range(PhiSupport):
#                                 index_l2z = (lz - l2z) % L
#                                 res_z += Gamma[l1z, l2z] * data_R2[index_l2x, index_l2y, index_l2z]
#                             res_y += Gamma[l1y, l2y] * res_z
#                         sum_over_l2 += Gamma[l1x, l2x] * res_y
#                     sum_over_l1 += data_R1[index_l1x, index_l1y, index_l1z] * sum_over_l2
#         result[lx, ly, lz] = data[lx, ly, lz] * sum_over_l1

# def cal_coefficients(data, l):
#     sum_res = 0
#     for m in range(1, l + 1):
#         sum_res += data[m] * (-1) ** m + np.conjugate(data[m]) * (-1) ** (-m)
#     sum_res += data[0]
#     return sum_res
