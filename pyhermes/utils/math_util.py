import inspect
import math
import time
import warnings
from pathlib import Path

import pywt
import numpy as np
import numba
from scipy.fft import rfftn, irfftn, fftn, ifftn
from scipy.special import spherical_jn, sph_harm, gammaln
from numba import cuda, jit, njit, prange
from numba.core.errors import NumbaExperimentalFeatureWarning

from pyhermes.param.logbase import setup_logger
from pyhermes.utils import func_util
from pyhermes.utils.func_util import get_fname_info
from pyhermes.utils.legendre_fast import (
    calculate_window_array_with_lm_fast,
    has_fast_window_function,
    window_function_legendre_fast,
)


_NUMBA_CONFIGURED = False
_NUMBA_THREADS = None


def configure(threads=1):
    """
    Configure Numba threads for this process.
    Re-applying with the same value is a no-op; changing the value updates
    the current runtime setting.
    """
    global _NUMBA_CONFIGURED, _NUMBA_THREADS
    requested_threads = max(1, int(threads))
    if _NUMBA_CONFIGURED and _NUMBA_THREADS == requested_threads:
        return
    from numba import set_num_threads, get_num_threads
    set_num_threads(requested_threads)
    _NUMBA_THREADS = int(get_num_threads())
    _NUMBA_CONFIGURED = True


def do_wavelet(mode="db2", level=10):
    """Return sampled scaling-function values for a PyWavelets wavelet."""
    wavelet = pywt.Wavelet(mode)
    _phi, _, _ = wavelet.wavefun(level=level)
    phi_data = _phi[:-1]
    return phi_data


def random_points_box(N, SimBoxL, ndim=3, rng=None, seed=None):
    """Draw uniformly distributed random points inside a periodic box."""
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    return rng.uniform(0.0, SimBoxL, size=(N, ndim))


# Suppress NumbaExperimentalFeatureWarning
warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

# ---------------------------------------------------------------
# ------------- ↓ Numerical function for convols ↓ --------------
# ---------------------------------------------------------------


@njit(parallel=True)
def calculate_window_array_numba(L, bandwidth, DeltaXi, PowerPhi, window_function_numba, *args):
    """Evaluate a real-space window lookup array from a k-space window kernel."""
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
        source_code = inspect.getsource(window_function_numba)
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
    """Fold an octant-symmetric window array into an rFFT-compatible kernel."""
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
    """Project weighted particles onto the full 3D scaling-function grid."""
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
    """Return coarse-grid x-cell indices for positions scaled by ScaleFactor."""
    num = data.shape[0]
    out = np.empty(num, dtype=np.int32)
    for i in range(num):
        out[i] = int(np.floor(data[i, 0] * ScaleFactor))
    return out


## bit opereator in numba
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
    """Select rows from origin_data whose partition labels equal part."""
    dm_part = origin_data[shrink_data == part]
    return dm_part


def specialized_convolution_3d(s, w, threads):
    """Convolve a real 3D field with an rFFT-space kernel."""
    # Run FFt in multi-thread manner
    sc = rfftn(s, workers= threads)
    sc *= w
    result_convol3d = irfftn(sc, workers = threads)
    return result_convol3d


def specialized_convolution_3d_complex(s, w, threads):
    """Convolve a complex 3D field with a full FFT-space kernel."""
    sc = fftn(s, workers=threads)
    sc *= w
    result_convol3d = ifftn(sc, workers=threads)
    return result_convol3d


def power_spectrum(v, k0, k1, N_k, SampRate):
    """Return the squared magnitude of the sampled 1D Fourier spectrum."""
    s = spectrum_vectorized(v, k0, k1, N_k, SampRate)
    p = np.zeros(N_k + 1, dtype=np.double)
    for i in range(N_k + 1):
        p[i] = s[2*i] ** 2 + s[2*i+1] ** 2
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

# ---------------------------------------------------------------------------
# -------- ↓ Numerical function for 3pcf: zeta(r1, r2, theta)↓ --------------
# ---------------------------------------------------------------------------


def third_side(r1, r2, theta):
    """Return r23 from (r1, r2, theta) using the law of cosines."""
    return np.sqrt(r1**2 + r2**2 - 2.0 * r1 * r2 * np.cos(theta))


def third_side_from_mu(r1, r2, mu):
    """Return r23 from (r1, r2, mu) with mu = cos(theta)."""
    return np.sqrt(np.clip(r1**2 + r2**2 - 2.0 * r1 * r2 * mu, 0.0, None))


@njit
def generate_triangle_offsets(R1, R2, mu, phi, costheta1, alpha):
    """
    Generate offsets (x2,y2,z2) and (x3,y3,z3) that form a triangle with:
      |r1| = R1, |r2| = R2, cos(angle(r1,r2)) = mu

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
    cos_theta = mu
    sin_theta_sq = 1.0 - mu * mu
    if sin_theta_sq < 0.0:
        sin_theta_sq = 0.0
    sin_theta = math.sqrt(sin_theta_sq)

    dx2 = cos_theta * dx1 + sin_theta * (cos_alpha * ox1 + sin_alpha * ox2)
    dy2 = cos_theta * dy1 + sin_theta * (cos_alpha * oy1 + sin_alpha * oy2)
    dz2 = cos_theta * dz1 + sin_theta * (cos_alpha * oz1 + sin_alpha * oz2)

    x3 = R2 * dx2
    y3 = R2 * dy2
    z3 = R2 * dz2

    return x2, y2, z2, x3, y3, z3


@njit
def estimate_triplet_product_particle_centers(
    R1_scaled, R2_scaled, mu,
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
        x2, y2, z2, x3, y3, z3 = generate_triangle_offsets(R1_scaled, R2_scaled, mu, phi, costheta1, alpha)
        n_at_pos_numba(n2, centers_scaled, epsilon2, phi_data, L, SampRate, PhiSupport, x2, y2, z2)
        n_at_pos_numba(n3, centers_scaled, epsilon3, phi_data, L, SampRate, PhiSupport, x3, y3, z3)
        s = 0.0
        for i in range(npos):
            s += n2[i] * n3[i]
        total_sum += s
    return R * total_sum / (n_rot * npos)


@njit
def estimate_triplet_product_box_random_centers(
    R1_scaled, R2_scaled, mu,
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

        x2, y2, z2, x3, y3, z3 = generate_triangle_offsets(R1_scaled, R2_scaled, mu, phi, costheta1, alpha)

        n_at_pos_numba(n2, centers_scaled, epsilon2, phi_data, L, SampRate, PhiSupport, x2, y2, z2)
        n_at_pos_numba(n3, centers_scaled, epsilon3, phi_data, L, SampRate, PhiSupport, x3, y3, z3)

        s = 0.0
        for i in range(npos):
            s += n1[i] * n2[i] * n3[i]
        total_sum += s

    return total_sum / (n_rot * npos)


@njit
def estimate_triplet_contrast_particle_centers_legacy(
    R1_scaled, R2_scaled, mu,
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
        x2, y2, z2, x3, y3, z3 = generate_triangle_offsets(R1_scaled, R2_scaled, mu, phi, costheta1, alpha)
        n_at_pos_numba(n2, pos_scaled, epsilon2, phi_data, L, SampRate, PhiSupport, x2, y2, z2)
        n_at_pos_numba(n3, pos_scaled, epsilon3, phi_data, L, SampRate, PhiSupport, x3, y3, z3)
        n_at_pos_numba(n20, rand_scaled, epsilon2, phi_data, L, SampRate, PhiSupport, x2, y2, z2)
        n_at_pos_numba(n30, rand_scaled, epsilon3, phi_data, L, SampRate, PhiSupport, x3, y3, z3)
        s = 0.0
        for i in range(npos):
            s += n2[i] * n3[i] - n20[i] * n30[i]
        total_sum += s
    return R * total_sum / (n_rot * npos)

# ---------------------------------------------------------------------------
# ------- ↓ Numerical function for 3pcf multipole: zeta_l(r1, r2) ↓ ---------
# ---------------------------------------------------------------------------


@njit
def _k_norm(ki, kj, kk):
    """Return the Euclidean norm of a 3D k-vector."""
    return math.sqrt(ki * ki + kj * kj + kk * kk)


@njit
def _phase_from_kR(ki, kj, kk, R):
    """Return the Fourier phase 2*pi*|k|*R."""
    return 2.0 * math.pi * _k_norm(ki, kj, kk) * R


@njit
def _angles_from_k(ki, kj, kk):
    """Return |k|, polar angle theta, and azimuth phi for a k-vector."""
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
    """Compute n! as float for small nonnegative integer n."""
    out = 1.0
    for i in range(2, n + 1):
        out *= i
    return out


@njit
def spherical_jn_numba(l, x):
    """Compute spherical Bessel j_l(x) with stable small-x and high-l handling."""
    if l < 0:
        return np.nan

    if x == 0.0:
        if l == 0:
            return 1.0
        return 0.0

    ax = abs(x)
    if ax <= 5.0e-2:
        if l == 0:
            x2 = x * x
            return 1.0 - x2 / 6.0 + x2 * x2 / 120.0
        denom = 1.0
        for n in range(1, l + 1):
            denom *= 2.0 * n + 1.0
        x2 = x * x
        return (x ** l / denom) * (
            1.0
            - x2 / (2.0 * (2.0 * l + 3.0))
            + x2 * x2 / (8.0 * (2.0 * l + 3.0) * (2.0 * l + 5.0))
        )

    sx = math.sin(x)
    cx = math.cos(x)
    j0 = sx / x
    if l == 0:
        return j0

    j1 = sx / (x * x) - cx / x
    if l == 1:
        return j1

    if ax < l + 1.0:
        n_stop = l + 40
        jp1 = 0.0
        jcur = 1.0
        j_target = 0.0
        j_down_0 = 0.0
        j_down_1 = 0.0
        # Upward recurrence is unstable here; Miller downward recurrence is
        # normalized against stable closed forms for j0 or j1 after the sweep.
        for n in range(n_stop, 0, -1):
            jm1 = ((2.0 * n + 1.0) / x) * jcur - jp1
            if n - 1 == l:
                j_target = jm1
            if n - 1 == 1:
                j_down_1 = jm1
            if n - 1 == 0:
                j_down_0 = jm1
            jp1 = jcur
            jcur = jm1
            if abs(jcur) > 1.0e200:
                jp1 *= 1.0e-200
                jcur *= 1.0e-200
                j_target *= 1.0e-200
                j_down_0 *= 1.0e-200
                j_down_1 *= 1.0e-200
            elif abs(jcur) < 1.0e-200 and jcur != 0.0:
                jp1 *= 1.0e200
                jcur *= 1.0e200
                j_target *= 1.0e200
                j_down_0 *= 1.0e200
                j_down_1 *= 1.0e200

        if abs(j_down_0) >= abs(j_down_1) or j_down_1 == 0.0:
            return j_target * (j0 / j_down_0)
        return j_target * (j1 / j_down_1)

    jm1 = j0
    jcur = j1
    for ell in range(1, l):
        jnext = ((2.0 * ell + 1.0) / x) * jcur - jm1
        jm1 = jcur
        jcur = jnext

    return jcur


@njit
def assoc_legendre_numba(l, m, x):
    """Compute the associated Legendre function P_l^m(x) for nonnegative m."""
    if l < 0:
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
    """Evaluate complex spherical harmonic Y_l^m in the direction of k."""
    if l < 0:
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
    """Evaluate j_l(2*pi*|k|*R) * Y_l^m(khat) with Numba helpers."""
    if l < 0:
        return np.nan + 0.0j
    if abs(m) > l:
        return 0.0 + 0.0j

    phase = _phase_from_kR(ki, kj, kk, R)
    jl = spherical_jn_numba(l, phase)
    ylm = spherical_harmonic_numba(l, m, ki, kj, kk)
    return jl * ylm


def window_function_legendre(ki, kj, kk, R, l, m, use_fast=True):
    """Evaluate the Legendre multipole window, using fast kernels when available."""
    if use_fast:
        if has_fast_window_function(l, m):
            return window_function_legendre_fast(ki, kj, kk, R, l, m)
        else:
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
    """Build a complex FFT-space window array for one Legendre (l, m) mode."""
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
    """Build a Legendre window array, dispatching to generated fast kernels when possible."""
    if has_fast_window_function(l, m):
        return calculate_window_array_with_lm_fast(L, DeltaXi, PowerPhi, rescaleR, l, m)
    return calculate_window_array_legendre_numba(L, DeltaXi, PowerPhi, rescaleR, l, m)


def cal_gamma(phi_data, PhiSupport, SampRate):
    """Compute scaling-function triple-overlap weights for GPU summation."""
    gamma = np.zeros((PhiSupport, PhiSupport))
    for l1 in range(PhiSupport):
        for l2 in range(PhiSupport):
            rolled_phi1 = np.roll(phi_data, l1 * SampRate)
            rolled_phi2 = np.roll(phi_data, l2 * SampRate)
            gamma[l1, l2] = np.sum(phi_data * rolled_phi1 * rolled_phi2) / SampRate
    return gamma


@cuda.jit
def compute_3d_result_gpu(data, data_R1, data_R2, Gamma, result, L, PhiSupport):
    """Multiply the center field by two convolved fields using Gamma overlaps."""
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


REDUCE_THREADS = 256


@cuda.jit
def reduce_complex_sum_kernel(data, partial_real, partial_imag, n):
    """Reduce a complex device array into per-block real and imaginary sums."""
    shared_real = cuda.shared.array(shape=REDUCE_THREADS, dtype=numba.float64)
    shared_imag = cuda.shared.array(shape=REDUCE_THREADS, dtype=numba.float64)

    tid = cuda.threadIdx.x
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    local_real = 0.0
    local_imag = 0.0
    while idx < n:
        value = data[idx]
        local_real += value.real
        local_imag += value.imag
        idx += stride

    shared_real[tid] = local_real
    shared_imag[tid] = local_imag
    cuda.syncthreads()

    offset = cuda.blockDim.x // 2
    while offset > 0:
        if tid < offset:
            shared_real[tid] += shared_real[tid + offset]
            shared_imag[tid] += shared_imag[tid + offset]
        cuda.syncthreads()
        offset //= 2

    if tid == 0:
        partial_real[cuda.blockIdx.x] = shared_real[0]
        partial_imag[cuda.blockIdx.x] = shared_imag[0]


def combine_multipole_m_terms(m_values, l):
    """Combine nonnegative m summands into one real multipole coefficient."""
    coeff = complex(m_values[0])
    for m in range(1, l + 1):
        coeff += ((-1) ** m) * complex(m_values[m])
        coeff += ((-1) ** (-m)) * np.conj(complex(m_values[m]))
    coeff *= (-1) ** l
    return coeff.real


def _cache_file_path(cache_dir, radius, l, m):
    """Return the cache path for one radius and Legendre (l, m) field."""
    sign = "m" if m >= 0 else "m_minus"
    suffix = f"{m}" if m >= 0 else f"{-m}"
    return Path(cache_dir) / f"R{radius:g}_l{l}_{sign}{suffix}.npy"


def _prepare_legendre_convolution_context(field):
    """Precompute shared wavelet-spectrum inputs for Legendre convolutions."""
    return {
        "delta_xi": 1.0 / field.L,
        "power_phi": power_spectrum(field.phi_data, 0, field.bandwidth, field.L * field.bandwidth, field.SampRate),
    }


def _stream_convolution_fields(
    field,
    radius,
    l,
    threads,
    m_values=None,
    cache_multipole_fields=False,
    cache_dir="",
    conv_context=None,
):
    """Generate or load convolved fields for selected m values at one radius."""
    if conv_context is None:
        conv_context = _prepare_legendre_convolution_context(field)
    delta_xi = conv_context["delta_xi"]
    power_phi = conv_context["power_phi"]
    rescaleR = radius * field.ScaleFactor
    if m_values is None:
        m_values = range(-l, l + 1)
    m_fields = []
    for m in m_values:
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


def _prepare_multipole_gpu_context(field1, gpu_device_id=0):
    """Allocate reusable CUDA state for multipole m-term summation."""
    if not cuda.is_available():
        raise RuntimeError("CUDA is required for Corr_3PCF_Multipole, but no CUDA device is available.")

    cuda.select_device(int(gpu_device_id))
    gamma = np.ascontiguousarray(cal_gamma(field1.phi_data, field1.PhiSupport, field1.SampRate), dtype=np.float64)
    gamma_gpu = cuda.to_device(gamma)
    data_gpu = cuda.to_device(np.ascontiguousarray(field1.epsilon, dtype=np.float64))
    result_gpu = cuda.device_array(field1.epsilon.shape, dtype=np.complex128)
    n_result = field1.epsilon.size
    result_gpu_flat = result_gpu.reshape(n_result)
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        (field1.L + threads_per_block[0] - 1) // threads_per_block[0],
        (field1.L + threads_per_block[1] - 1) // threads_per_block[1],
        (field1.L + threads_per_block[2] - 1) // threads_per_block[2],
    )
    reduce_blocks = min(1024, (n_result + REDUCE_THREADS - 1) // REDUCE_THREADS)
    partial_real_gpu = cuda.device_array(reduce_blocks, dtype=np.float64)
    partial_imag_gpu = cuda.device_array(reduce_blocks, dtype=np.float64)
    return {
        "gamma_gpu": gamma_gpu,
        "data_gpu": data_gpu,
        "result_gpu": result_gpu,
        "result_gpu_flat": result_gpu_flat,
        "n_result": n_result,
        "threads_per_block": threads_per_block,
        "blocks_per_grid": blocks_per_grid,
        "reduce_blocks": reduce_blocks,
        "partial_real_gpu": partial_real_gpu,
        "partial_imag_gpu": partial_imag_gpu,
        "L": field1.L,
        "PhiSupport": field1.PhiSupport,
    }


def compute_multipole_m_summand(field_r1_m, field_r2_m, gpu_context):
    """Compute one complex m summand and return timing diagnostics."""
    t_h2d_start = time.perf_counter()
    data_r1_gpu = cuda.to_device(np.ascontiguousarray(field_r1_m, dtype=np.complex128))
    data_r2_gpu = cuda.to_device(np.ascontiguousarray(field_r2_m, dtype=np.complex128))
    h2d_elapsed = time.perf_counter() - t_h2d_start

    t_kernel_start = time.perf_counter()
    compute_3d_result_gpu[gpu_context["blocks_per_grid"], gpu_context["threads_per_block"]](
        gpu_context["data_gpu"],
        data_r1_gpu,
        data_r2_gpu,
        gpu_context["gamma_gpu"],
        gpu_context["result_gpu"],
        gpu_context["L"],
        gpu_context["PhiSupport"],
    )
    cuda.synchronize()
    kernel_elapsed = time.perf_counter() - t_kernel_start

    t_reduce_start = time.perf_counter()
    reduce_complex_sum_kernel[gpu_context["reduce_blocks"], REDUCE_THREADS](
        gpu_context["result_gpu_flat"],
        gpu_context["partial_real_gpu"],
        gpu_context["partial_imag_gpu"],
        gpu_context["n_result"],
    )
    cuda.synchronize()
    reduce_elapsed = time.perf_counter() - t_reduce_start

    t_d2h_start = time.perf_counter()
    partial_real = gpu_context["partial_real_gpu"].copy_to_host()
    partial_imag = gpu_context["partial_imag_gpu"].copy_to_host()
    d2h_elapsed = time.perf_counter() - t_d2h_start

    del data_r1_gpu
    del data_r2_gpu

    value = (4.0 * np.pi) * complex(np.sum(partial_real), np.sum(partial_imag)) / gpu_context["n_result"]
    return value, {
        "h2d_elapsed_sec": h2d_elapsed,
        "kernel_elapsed_sec": kernel_elapsed,
        "reduce_elapsed_sec": reduce_elapsed,
        "d2h_elapsed_sec": d2h_elapsed,
    }


def calc_DDD_multipole(
    deltaD1, deltaD2, deltaD3,
    r1, r2, l_min, l_max,
    gpu_device_id=0,
    cache_multipole_fields=False,
    cache_dir="",
    threads=1,
    progress_callback=None,
    m_progress_callback=None,
):
    """Compute DDD Legendre multipoles over l_min..l_max for three fields."""
    if l_min < 0:
        raise ValueError("l_min must be non-negative.")
    if l_max < 0:
        raise ValueError("l_max must be non-negative.")
    if l_min > l_max:
        raise ValueError("l_min must be less than or equal to l_max.")
    gpu_context = _prepare_multipole_gpu_context(deltaD1, gpu_device_id=gpu_device_id)
    conv_context_r1 = _prepare_legendre_convolution_context(deltaD2)
    conv_context_r2 = _prepare_legendre_convolution_context(deltaD3)

    l_values = np.arange(l_min, l_max + 1, dtype=np.int32)
    ddd_l = np.empty(l_values.size, dtype=np.float64)
    total_m_tasks = sum(l + 1 for l in range(l_min, l_max + 1))
    completed_m_tasks = 0
    total_conv_elapsed = 0.0
    total_sum_elapsed = 0.0
    total_sum_h2d_elapsed = 0.0
    total_sum_kernel_elapsed = 0.0
    total_sum_d2h_elapsed = 0.0
    total_sum_reduce_elapsed = 0.0
    total_sum_callback_elapsed = 0.0

    rho = 1.0 / deltaD1.V
    rho3 = rho ** 3
    for l_idx, l in enumerate(range(l_min, l_max + 1)):
        t_l_start = time.perf_counter()
        conv_elapsed = 0.0
        m_values = np.empty(l + 1, dtype=np.complex128)
        sum_elapsed = 0.0
        for m in range(0, l + 1):
            t_m_start = time.perf_counter()
            t_conv_m_start = time.perf_counter()
            field_r1_m = _stream_convolution_fields(
                deltaD2, r1, l, threads=threads,
                m_values=[m],
                cache_multipole_fields=cache_multipole_fields,
                cache_dir=cache_dir,
                conv_context=conv_context_r1,
            )[0]
            field_r2_m = _stream_convolution_fields(
                deltaD3, r2, l, threads=threads,
                m_values=[-m],
                cache_multipole_fields=cache_multipole_fields,
                cache_dir=cache_dir,
                conv_context=conv_context_r2,
            )[0]
            conv_m_elapsed = time.perf_counter() - t_conv_m_start
            conv_elapsed += conv_m_elapsed
            total_conv_elapsed += conv_m_elapsed
            t_sum_m_start = time.perf_counter()
            m_values[m], timing = compute_multipole_m_summand(field_r1_m, field_r2_m, gpu_context)
            total_sum_h2d_elapsed += timing["h2d_elapsed_sec"]
            total_sum_kernel_elapsed += timing["kernel_elapsed_sec"]
            total_sum_reduce_elapsed += timing["reduce_elapsed_sec"]
            total_sum_d2h_elapsed += timing["d2h_elapsed_sec"]
            sum_elapsed += time.perf_counter() - t_sum_m_start
            completed_m_tasks += 1
            if m_progress_callback is not None:
                t_callback_start = time.perf_counter()
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
                total_sum_callback_elapsed += time.perf_counter() - t_callback_start
        ddd_l[l_idx] = combine_multipole_m_terms(m_values, l)
        total_sum_elapsed += sum_elapsed
        if progress_callback is not None:
            progress_callback(
                l=l,
                l_max=l_max,
                ddd_l=float(ddd_l[l_idx]),
                zeta_l=float(ddd_l[l_idx] / rho3),
                elapsed_sec=time.perf_counter() - t_l_start,
                conv_elapsed_sec=conv_elapsed,
                sum_elapsed_sec=sum_elapsed,
                completed_m_tasks=completed_m_tasks,
                total_m_tasks=total_m_tasks,
            )

    timing_info = {
        "conv_elapsed_sec": total_conv_elapsed,
        "sum_elapsed_sec": total_sum_elapsed,
        "sum_h2d_elapsed_sec": total_sum_h2d_elapsed,
        "sum_kernel_elapsed_sec": total_sum_kernel_elapsed,
        "sum_d2h_elapsed_sec": total_sum_d2h_elapsed,
        "sum_reduce_elapsed_sec": total_sum_reduce_elapsed,
        "sum_callback_elapsed_sec": total_sum_callback_elapsed,
    }
    return l_values, ddd_l, timing_info


def legendre_triple_coeff(l: int, lp: int, k: int) -> float:
    """
    Return the Legendre triple-product coupling coefficient

        G_{lp,l,k} = (2k+1)/2 * ∫_{-1}^1 P_lp(mu) P_l(mu) P_k(mu) dmu

    using the closed-form expression valid for Legendre polynomials.

    Notes
    -----
    Nonzero only if:
      - triangle condition: |l-lp| <= k <= l+lp
      - parity condition: l + lp + k is even
    """
    s = l + lp + k

    # selection rules
    if (k > l + lp) or (k < abs(l - lp)) or (s % 2 != 0):
        return 0.0

    g = s // 2

    # closed-form log expression
    log_val = (
        gammaln(s - 2 * l + 1)
        + gammaln(s - 2 * lp + 1)
        + gammaln(s - 2 * k + 1)
        - gammaln(s + 2)
        + 2 * (
            gammaln(g + 1)
            - gammaln(g - l + 1)
            - gammaln(g - lp + 1)
            - gammaln(g - k + 1)
        )
    )

    # this equals the Legendre triple-product integral coefficient
    return float((2 * k + 1) * np.exp(log_val))


def build_mixing_matrix(B: np.ndarray, lmax: int) -> np.ndarray:
    """
    Return the mixing matrix M defined by

        A_k = sum_l M[k, l] * C_l

    for the ratio problem

        A(mu) = B(mu) * C(mu).

    Parameters
    ----------
    B : ndarray
        Multipoles B_l of the denominator, with length >= lmax + 1.
    lmax : int
        Maximum multipole order.

    Returns
    -------
    ndarray
        Mixing matrix M of shape (lmax + 1, lmax + 1).

    Notes
    -----
    The matrix is built by separating the monopole term B_0 from the
    higher-order multipoles, i.e.

        B(mu) = B_0 [1 + sum_{l>0} (B_l / B_0) P_l(mu)].

    The normalized matrix is first constructed using B_l / B_0, and the
    overall factor B_0 is restored at the end.

    In the limit B_l = 0 for l > 0, the matrix reduces to

        M = B_0 I.
    """
    B = np.asarray(B, dtype=np.float64)
    B0 = B[0]

    if B0 == 0:
        raise ValueError("B[0] is zero.")

    f = B / B0
    M = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)

    # k = 0 row
    for l in range(lmax + 1):
        for lp in range(1, lmax + 1):
            if l == lp:
                M[0, l] += f[lp] / (2 * l + 1)

    # k >= 1 rows
    for k in range(1, lmax + 1):
        for l in range(lmax + 1):
            for lp in range(1, lmax + 1):
                M[k, l] += legendre_triple_coeff(l, lp, k) * f[lp]

    # add the identity contribution in the normalized convention
    M += np.eye(lmax + 1)

    # restore the overall monopole factor
    M *= B0
    return M


def solve_multipoles_from_ratio(A: np.ndarray, B: np.ndarray, lmax: int, rcond_warning: float = 1e12):
    """
    Return the multipoles C_l of the ratio defined by

        A(mu) = B(mu) * C(mu).

    Parameters
    ----------
    A : ndarray
        Multipoles A_l of the numerator, with length >= lmax + 1.
    B : ndarray
        Multipoles B_l of the denominator, with length >= lmax + 1.
    lmax : int
        Maximum multipole order.

    Returns
    -------
    C : ndarray
        Multipoles C_l of the ratio.
    M : ndarray
        Mixing matrix of shape (lmax + 1, lmax + 1).
    condM : float
        Condition number of M.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if A.shape[0] < lmax + 1 or B.shape[0] < lmax + 1:
        raise ValueError("A and B must have length at least lmax+1.")

    A_cut = A[:lmax + 1]
    B_cut = B[:lmax + 1]

    M = build_mixing_matrix(B_cut, lmax)
    
    condM = np.linalg.cond(M)
    if condM > rcond_warning:
        print(f"[warning] mixing matrix is ill-conditioned: cond(M) = {condM:.3e}")

    C = np.linalg.solve(M, A_cut)

    return C, M, condM
