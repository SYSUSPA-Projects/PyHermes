import inspect
import math
import warnings

import pywt
import numpy as np
from scipy.fft import rfftn, irfftn
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

def cal_gamma(phi_data, PhiSupport, SampRate):
    Gamma = np.zeros((PhiSupport, PhiSupport))
    for l1 in range(PhiSupport):
        for l2 in range(PhiSupport):
            rolled_phi1 = np.roll(phi_data, l1 * SampRate)
            rolled_phi2 = np.roll(phi_data, l2 * SampRate)
            Gamma[l1, l2] = np.sum(phi_data * rolled_phi1 * rolled_phi2) / SampRate
    return Gamma

@cuda.jit
def compute_3d_result_gpu(data, data_R1, data_R2, Gamma, result, L, PhiSupport):
    lx, ly, lz = cuda.grid(3)
    if lx < L and ly < L and lz < L:
        sum_over_l1 = 0
        for l1x in range(PhiSupport):
            index_l1x = (lx - l1x) % L
            for l1y in range(PhiSupport):
                index_l1y = (ly - l1y) % L
                for l1z in range(PhiSupport):
                    index_l1z = (lz - l1z) % L
                    sum_over_l2 = 0
                    for l2x in range(PhiSupport):
                        index_l2x = (lx - l2x) % L
                        res_y = 0
                        for l2y in range(PhiSupport):
                            index_l2y = (ly - l2y) % L
                            res_z = 0
                            for l2z in range(PhiSupport):
                                index_l2z = (lz - l2z) % L
                                res_z += Gamma[l1z, l2z] * data_R2[index_l2x, index_l2y, index_l2z]
                            res_y += Gamma[l1y, l2y] * res_z
                        sum_over_l2 += Gamma[l1x, l2x] * res_y
                    sum_over_l1 += data_R1[index_l1x, index_l1y, index_l1z] * sum_over_l2
        result[lx, ly, lz] = data[lx, ly, lz] * sum_over_l1

def cal_coefficients(data, l):
    sum_res = 0
    for m in range(1, l + 1):
        sum_res += data[m] * (-1) ** m + np.conjugate(data[m]) * (-1) ** (-m)
    sum_res += data[0]
    return sum_res

def do_wavelet(mode="db2", level=10):
    wavelet = pywt.Wavelet(mode)
    _phi, _, _ = wavelet.wavefun(level=level)
    phi_data = _phi[:-1]
    return phi_data

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
        logger.error(f"Please see the document for details")
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
def result_interpret4(convol3d, p_coarse, p_finer, phi_data, J, SampRate=1024):
    L = 1 << J
    PhiStart = 0
    PhiEnd = phi_data.shape[0] // SampRate
    PhiSupport = PhiEnd - PhiStart
    step = np.arange(PhiSupport) * SampRate
    total = p_coarse.shape[0]
    total_sum = 0
    for num in range(total):
        pp_coarse0, pp_coarse1, pp_coarse2 = p_coarse[num]
        pp_finer0, pp_finer1, pp_finer2 = p_finer[num]
        res = 0
        for i in range(PhiSupport):
            x1 = (pp_coarse0 - i) & (L - 1)
            phi1 = phi_data[int(pp_finer0) + step[i]]
            for j in range(PhiSupport):
                y1 = (pp_coarse1 - j) & (L - 1)
                res2 = 0
                for k in range(PhiSupport):
                    res2 += convol3d[x1, y1, ((pp_coarse2 - k) & (L - 1))] * phi_data[int(pp_finer2) + step[k]]
                res += res2 * phi1 * phi_data[int(pp_finer1) + step[j]]
        total_sum += res
    return total_sum


@jit(nopython=True)
def result_interpret5(convol3d, size, phi_data, J, SampRate=1024):
    L = 1 << J
    total = 0
    for _ in range(size):
        scale_p = np.random.rand(3) * L
        p_coarse = np.floor(scale_p).astype(np.int32)
        p_finer = (scale_p - p_coarse) * SampRate
        total += result_interpret6(convol3d, p_coarse, p_finer, phi_data, J, SampRate)
    return total


@jit(nopython=True)
def result_interpret6(convol3d, p_coarse, p_finer, phi_data, J, SampRate=1024):
    PhiStart = 0
    PhiEnd = phi_data.shape[0] // SampRate
    PhiSupport = PhiEnd - PhiStart
    step = np.arange(PhiSupport) * SampRate
    pp_coarse0, pp_coarse1, pp_coarse2 = p_coarse
    pp_finer0, pp_finer1, pp_finer2 = p_finer
    res = 0
    for i in range(PhiSupport):
        x1 = pp_coarse0 - i
        phi1 = phi_data[int(pp_finer0) + step[i]]
        for j in range(PhiSupport):
            y1 = pp_coarse1 - j
            res2 = 0
            for k in range(PhiSupport):
                res2 += convol3d[x1, y1, pp_coarse2 - k] * phi_data[int(pp_finer2) + step[k]]
            res += res2 * phi1 * phi_data[int(pp_finer1) + step[j]]
    return res


@jit(nopython=True)
def result_interpret2(convol3d, size, phi_data, J, SampRate=1024):
    L = 1 << J
    PhiStart = 0
    PhiEnd = phi_data.shape[0] // SampRate
    PhiSupport = PhiEnd - PhiStart
    step = np.arange(PhiSupport) * SampRate
    scale_p = np.random.rand(size, 3) * L
    p_coarse = np.floor(scale_p).astype(np.int32)
    p_finer = (scale_p - p_coarse) * SampRate
    total = size
    result = np.zeros(total)
    for num in range(total):
        pp_coarse0, pp_coarse1, pp_coarse2 = p_coarse[num]
        pp_finer0, pp_finer1, pp_finer2 = p_finer[num]
        res = 0
        for i in range(PhiSupport):
            x1 = (pp_coarse0 - i) & (L - 1)
            phi1 = phi_data[int(pp_finer0) + step[i]]
            for j in range(PhiSupport):
                y1 = (pp_coarse1 - j) & (L - 1)
                res2 = 0
                for k in range(PhiSupport):
                    res2 += convol3d[x1, y1, ((pp_coarse2 - k) & (L - 1))] * phi_data[int(pp_finer2) + step[k]]
                res += res2 * phi1 * phi_data[int(pp_finer1) + step[j]]
        result[num] = res
    return result


def result_interpret3(convol3d, p_input, phi_data, J, SampRate=1024):
    scale_p = p_input
    p_coarse = np.floor(scale_p).astype(np.int32)
    p_finer = (scale_p - p_coarse) * SampRate
    return result_interpret4(convol3d, p_coarse, p_finer, phi_data, J, SampRate)


# ------------------------------------------------------------
# ------------- ↓ Numerical function for 2pcf ↓ --------------
# ------------------------------------------------------------


# ------------------------------------------------------------
# ------------- ↓ Numerical function for 3pcf ↓ --------------
# ------------------------------------------------------------


@njit
def generate_points_device(R1, R2, theta):
    phi = 2 * math.pi * np.random.uniform(0, 1)
    costheta = np.random.uniform(0, 1) - 1
    sintheta = math.sqrt(1 - costheta**2)
    dx1 = sintheta * math.cos(phi)
    dy1 = sintheta * math.sin(phi)
    dz1 = costheta
    x2 = R1 * dx1
    y2 = R1 * dy1
    z2 = R1 * dz1
    if abs(dx1) > 1e-10 or abs(dy1) > 1e-10:
        ox1, oy1, oz1 = -dy1, dx1, 0
    else:
        ox1, oy1, oz1 = 0, -dz1, dy1
    norm = math.sqrt(ox1**2 + oy1**2 + oz1**2)
    ox1 /= norm
    oy1 /= norm
    oz1 /= norm
    ox2 = dy1 * oz1 - dz1 * oy1
    oy2 = dz1 * ox1 - dx1 * oz1
    oz2 = dx1 * oy1 - dy1 * ox1
    alpha = 2 * math.pi * np.random.uniform(0, 1)
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
def result_3pcf_cpu_location(data_array, phidata, step, part_data, random_data, Nrotation, R1, R2, theta):
    L = 1 << 8
    SimBoxL = 1000
    ScaleFactor = L / SimBoxL
    GridSize = L - 1
    PhiStart = 0
    PhiEnd = 3
    PhiSupport = PhiEnd - PhiStart
    particle_num = part_data.shape[0]
    results = np.zeros(particle_num)
    for idx in range(particle_num):
        res = 0
        scalepx = part_data[idx, 0] * ScaleFactor
        scalepy = part_data[idx, 1] * ScaleFactor
        scalepz = part_data[idx, 2] * ScaleFactor
        scalepx0 = random_data[idx, 0] * ScaleFactor
        scalepy0 = random_data[idx, 1] * ScaleFactor
        scalepz0 = random_data[idx, 2] * ScaleFactor
        for loop in range(Nrotation):
            x2, y2, z2, x3, y3, z3 = generate_points_device(R1, R2, theta)
            third_value = 0
            second_value = 0
            inputx = scalepx + x2
            inputy = scalepy + y2
            inputz = scalepz + z2
            pp_coarse0 = int16(math.floor(inputx))
            pp_coarse1 = int16(math.floor(inputy))
            pp_coarse2 = int16(math.floor(inputz))
            pp_finer0 = int16((inputx - pp_coarse0) * 1024)
            pp_finer1 = int16((inputy - pp_coarse1) * 1024)
            pp_finer2 = int16((inputz - pp_coarse2) * 1024)
            for i in range(PhiSupport):
                for j in range(PhiSupport):
                    for k in range(PhiSupport):
                        second_value += (
                            data_array[
                                (pp_coarse0 - i) & GridSize, (pp_coarse1 - j) & GridSize, (pp_coarse2 - k) & GridSize
                            ]
                            * phidata[pp_finer0 + step[i]]
                            * phidata[pp_finer1 + step[j]]
                            * phidata[pp_finer2 + step[k]]
                        )
            inputx = scalepx + x3
            inputy = scalepy + y3
            inputz = scalepz + z3
            pp_coarse0 = int16(math.floor(inputx))
            pp_coarse1 = int16(math.floor(inputy))
            pp_coarse2 = int16(math.floor(inputz))
            pp_finer0 = int16((inputx - pp_coarse0) * 1024)
            pp_finer1 = int16((inputy - pp_coarse1) * 1024)
            pp_finer2 = int16((inputz - pp_coarse2) * 1024)
            for i in range(PhiSupport):
                for j in range(PhiSupport):
                    for k in range(PhiSupport):
                        third_value += (
                            data_array[
                                (pp_coarse0 - i) & GridSize, (pp_coarse1 - j) & GridSize, (pp_coarse2 - k) & GridSize
                            ]
                            * phidata[pp_finer0 + step[i]]
                            * phidata[pp_finer1 + step[j]]
                            * phidata[pp_finer2 + step[k]]
                        )
            third_value0 = 0
            second_value0 = 0
            inputx0 = scalepx0 + x2
            inputy0 = scalepy0 + y2
            inputz0 = scalepz0 + z2
            pp_coarse0 = int16(math.floor(inputx0))
            pp_coarse1 = int16(math.floor(inputy0))
            pp_coarse2 = int16(math.floor(inputz0))
            pp_finer0 = int16((inputx0 - pp_coarse0) * 1024)
            pp_finer1 = int16((inputy0 - pp_coarse1) * 1024)
            pp_finer2 = int16((inputz0 - pp_coarse2) * 1024)
            for i in range(PhiSupport):
                for j in range(PhiSupport):
                    for k in range(PhiSupport):
                        second_value0 += (
                            data_array[
                                (pp_coarse0 - i) & GridSize, (pp_coarse1 - j) & GridSize, (pp_coarse2 - k) & GridSize
                            ]
                            * phidata[pp_finer0 + step[i]]
                            * phidata[pp_finer1 + step[j]]
                            * phidata[pp_finer2 + step[k]]
                        )
            inputx0 = scalepx0 + x3
            inputy0 = scalepy0 + y3
            inputz0 = scalepz0 + z3
            pp_coarse0 = int16(math.floor(inputx0))
            pp_coarse1 = int16(math.floor(inputy0))
            pp_coarse2 = int16(math.floor(inputz0))
            pp_finer0 = int16((inputx0 - pp_coarse0) * 1024)
            pp_finer1 = int16((inputy0 - pp_coarse1) * 1024)
            pp_finer2 = int16((inputz0 - pp_coarse2) * 1024)
            for i in range(PhiSupport):
                for j in range(PhiSupport):
                    for k in range(PhiSupport):
                        third_value0 += (
                            data_array[
                                (pp_coarse0 - i) & GridSize, (pp_coarse1 - j) & GridSize, (pp_coarse2 - k) & GridSize
                            ]
                            * phidata[pp_finer0 + step[i]]
                            * phidata[pp_finer1 + step[j]]
                            * phidata[pp_finer2 + step[k]]
                        )
            res += second_value * third_value - second_value0 * third_value0
        results[idx] = res
    return results
