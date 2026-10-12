"""3PCF multipole convolution, caching, and CUDA summation helpers."""

import copy
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numba
import numpy as np
from numba import cuda

from pyhermes.io.window import WindowFunc
from pyhermes.utils.wavelet_grid import fourier_power_spectrum
from pyhermes.utils.window_params import serialize_window_params

MULTIPOLE_FIELD_CACHE_SCHEMA_VERSION = 2


def cal_gamma(phi_array, phi_support, phi_resolution):
    """Integrate all signed triple overlaps, with zero outside basis support.

    Entry [ia, ib] uses a=ia-(s-1), b=ib-(s-1) and integrates
    phi(t)*phi(t+a)*phi(t+b), matching coefficient indices l-a, l-b.
    Periodic wrapping belongs to the grid coefficients, not this local basis.
    """
    phi = np.asarray(phi_array, dtype=np.float64)
    if phi.ndim != 1 or phi.size != phi_support * phi_resolution:
        raise ValueError("phi_array must contain support*resolution samples.")
    offsets = range(1 - phi_support, phi_support)
    shifted = []
    for offset in offsets:
        value = np.zeros_like(phi)
        step = offset * phi_resolution
        if step >= 0:
            value[:phi.size-step] = phi[step:]
        else:
            value[-step:] = phi[:phi.size+step]
        shifted.append(value)
    gamma = np.empty((len(shifted), len(shifted)), dtype=np.float64)
    for ia, left in enumerate(shifted):
        for ib, right in enumerate(shifted):
            gamma[ia, ib] = np.sum(phi * left * right) / phi_resolution
    return gamma


@cuda.jit
def compute_3d_result_gpu(data, data_R1, data_R2, Gamma, result, L, phi_support):
    """Multiply the center field by two convolved fields using Gamma overlaps."""
    lx, ly, lz = cuda.grid(3)
    if lx < L and ly < L and lz < L:
        sum_over_l1 = 0.0 + 0.0j
        for l1x in range(2 * phi_support - 1):
            index_l1x = (lx - l1x + phi_support - 1) % L
            for l1y in range(2 * phi_support - 1):
                index_l1y = (ly - l1y + phi_support - 1) % L
                for l1z in range(2 * phi_support - 1):
                    index_l1z = (lz - l1z + phi_support - 1) % L
                    sum_over_l2 = 0.0 + 0.0j
                    for l2x in range(2 * phi_support - 1):
                        index_l2x = (lx - l2x + phi_support - 1) % L
                        res_y = 0.0 + 0.0j
                        for l2y in range(2 * phi_support - 1):
                            index_l2y = (ly - l2y + phi_support - 1) % L
                            res_z = 0.0 + 0.0j
                            for l2z in range(2 * phi_support - 1):
                                index_l2z = (lz - l2z + phi_support - 1) % L
                                res_z += Gamma[l1z, l2z] * data_R2[index_l2x, index_l2y, index_l2z]
                            res_y += Gamma[l1y, l2y] * res_z
                        sum_over_l2 += Gamma[l1x, l2x] * res_y
                    sum_over_l1 += data_R1[index_l1x, index_l1y, index_l1z] * sum_over_l2
        result[lx, ly, lz] = data[lx, ly, lz] * sum_over_l1


@cuda.jit
def _gamma_filter_axis_gpu(source, gamma, row, target, axis, L, origin):
    idx = cuda.grid(1)
    if idx < L * L * L:
        x = idx // (L * L)
        y = (idx // L) % L
        z = idx % L
        value = 0.0 + 0.0j
        for ib in range(gamma.shape[1]):
            weight = gamma[row, ib]
            if weight != 0.0:
                offset = ib - origin
                if axis == 0:
                    value += weight * source[(x - offset) % L, y, z]
                else:
                    value += weight * source[x, (y - offset) % L, z]
        target[x, y, z] = value


@cuda.jit
def _gamma_contract_z_gpu(center, first, filtered_xy, gamma, ax, ay,
                          result, L, origin, overwrite):
    idx = cuda.grid(1)
    if idx < L * L * L:
        x = idx // (L * L)
        y = (idx // L) % L
        z = idx % L
        fx = (x - ax) % L
        fy = (y - ay) % L
        value = 0.0 + 0.0j
        for ia in range(gamma.shape[0]):
            fz = (z - ia + origin) % L
            second = 0.0 + 0.0j
            for ib in range(gamma.shape[1]):
                second += gamma[ia, ib] * filtered_xy[x, y, (z - ib + origin) % L]
            value += first[fx, fy, fz] * second
        weighted = center[x, y, z] * value
        if overwrite:
            result[x, y, z] = weighted
        else:
            result[x, y, z] += weighted


def _launch_separable_gpu(context, first, second):
    """Complete signed stencil, reassociated exactly as the CPU path.

    Independent output cells avoid atomics. The default CUDA stream orders
    filtering and accumulation; two device work arrays are reused per m.
    A flattened grid makes adjacent threads access contiguous z coefficients.
    """
    gamma = context["gamma_gpu"]
    length = context["L"]
    origin = context["phi_support"] - 1
    blocks = context["separable_blocks_per_grid"]
    threads = context["separable_threads_per_block"]
    for ia in range(gamma.shape[0]):
        _gamma_filter_axis_gpu[blocks, threads](second, gamma, ia,
            context["work_x_gpu"], 0, length, origin)
        for ja in range(gamma.shape[0]):
            _gamma_filter_axis_gpu[blocks, threads](context["work_x_gpu"], gamma, ja,
                context["work_xy_gpu"], 1, length, origin)
            _gamma_contract_z_gpu[blocks, threads](context["data_gpu"], first,
                context["work_xy_gpu"], gamma, ia-origin, ja-origin,
                context["result_gpu"], length, origin, ia == 0 and ja == 0)


REDUCE_THREADS = 256


def normalize_gpu_threads_per_block(gpu_threads_per_block):
    """Validate the legacy CUDA triple; its product sets the production 1D block size."""
    if gpu_threads_per_block is None:
        gpu_threads_per_block = (8, 8, 8)
    if isinstance(gpu_threads_per_block, str):
        stripped = gpu_threads_per_block.strip().strip("[]()")
        gpu_threads_per_block = [item.strip() for item in stripped.split(",") if item.strip()]
    if not isinstance(gpu_threads_per_block, (list, tuple)) or len(gpu_threads_per_block) != 3:
        raise ValueError("gpu_threads_per_block must be a list or tuple with three positive integers.")

    threads = []
    for value in gpu_threads_per_block:
        if isinstance(value, bool):
            raise ValueError("gpu_threads_per_block values must be integers, not booleans.")
        if isinstance(value, (float, np.floating)) and not float(value).is_integer():
            raise ValueError("gpu_threads_per_block values must be integers.")
        threads.append(int(value))
    threads_per_block = tuple(threads)
    if any(value <= 0 for value in threads_per_block):
        raise ValueError("gpu_threads_per_block values must be positive integers.")
    if np.prod(threads_per_block) > 1024:
        raise ValueError("gpu_threads_per_block product must not exceed 1024 CUDA threads per block.")
    return threads_per_block


def normalize_summation_backend(summation_backend):
    """Validate the backend used for the SFC triple-product summation."""
    if summation_backend is None:
        summation_backend = "gpu"
    backend = str(summation_backend).strip().lower()
    if backend not in {"gpu", "cpu"}:
        raise ValueError("summation_backend must be either 'gpu' or 'cpu'.")
    return backend


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


"""Exact separable evaluation of the signed Gamma contraction.

For each first-arm x/y offset, filter the second arm along x then y;
reuse those two arrays when contracting all z-offset pairs. This is an
algebraic reassociation of the complete tensor stencil, without truncation.
"""

@numba.njit(parallel=True, cache=True)
def _gamma_filter_axis_cpu(source, weights, target, axis, L, origin):
    for idx in numba.prange(L * L * L):
        x = idx // (L * L)
        y = (idx // L) % L
        z = idx % L
        value = 0.0 + 0.0j
        for ib in range(len(weights)):
            weight = weights[ib]
            if weight != 0.0:
                offset = ib - origin
                if axis == 0:
                    value += weight * source[(x - offset) % L, y, z]
                else:
                    value += weight * source[x, (y - offset) % L, z]
        target[x, y, z] = value


@numba.njit(parallel=True, cache=True)
def _gamma_contract_z_cpu(center, first, filtered_xy, gamma, ax, ay, L, origin):
    total_real = 0.0
    total_imag = 0.0
    for idx in numba.prange(L * L * L):
        x = idx // (L * L)
        y = (idx // L) % L
        z = idx % L
        fx = (x - ax) % L
        fy = (y - ay) % L
        value = 0.0 + 0.0j
        for ia in range(gamma.shape[0]):
            fz = (z - ia + origin) % L
            second = 0.0 + 0.0j
            for ib in range(gamma.shape[1]):
                weight = gamma[ia, ib]
                second += weight * filtered_xy[x, y, (z - ib + origin) % L]
            value += first[fx, fy, fz] * second
        weighted = center[x, y, z] * value
        total_real += weighted.real
        total_imag += weighted.imag
    return total_real + 1j * total_imag


def _compute_3d_separable_cpu(data, data_R1, data_R2, Gamma, L, phi_support, work_x, work_xy):
    origin = phi_support - 1
    total = 0.0 + 0.0j
    for ia in range(Gamma.shape[0]):
        _gamma_filter_axis_cpu(data_R2, Gamma[ia], work_x, 0, L, origin)
        for ja in range(Gamma.shape[0]):
            _gamma_filter_axis_cpu(work_x, Gamma[ja], work_xy, 1, L, origin)
            total += _gamma_contract_z_cpu(data, data_R1, work_xy, Gamma,
                                          ia-origin, ja-origin, L, origin)
    return total


def compute_3d_summand_cpu(data, data_R1, data_R2, Gamma, L, phi_support):
    """Evaluate the complete stencil; pipeline contexts reuse the work arrays."""
    work_x = np.empty(data_R2.shape, dtype=np.complex128)
    work_xy = np.empty_like(work_x)
    return _compute_3d_separable_cpu(data, data_R1, data_R2, Gamma, L, phi_support, work_x, work_xy)


def combine_multipole_m_terms(m_values, l):
    """Combine nonnegative m summands into one real multipole coefficient."""
    coeff = complex(m_values[0])
    for m in range(1, l + 1):
        coeff += ((-1) ** m) * complex(m_values[m])
        coeff += ((-1) ** (-m)) * np.conj(complex(m_values[m]))
    coeff *= (-1) ** l
    return coeff.real


def _array_digest(array):
    """Return a stable content digest for one contiguous numeric array."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha1()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _field_cache_signature(field):
    """Return a cached identity for a convolved SFC field and its basis."""
    cached = getattr(field, "_multipole_cache_signature", None)
    if cached is not None:
        return cached

    metadata = {
        key: getattr(field, key, None)
        for key in ("L", "box_size", "phi_resolution", "wavelet_mode", "wavelet_level", "phi_support")
    }
    metadata["sfc_info"] = serialize_window_params(getattr(field, "sfc_info", {}))
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha1(payload.encode("utf-8"))
    digest.update(_array_digest(field.epsilon).encode("utf-8"))
    digest.update(_array_digest(field.phi_array).encode("utf-8"))
    cached = digest.hexdigest()
    field._multipole_cache_signature = cached
    return cached


def _cache_file_path(cache_dir, field, binning_window, l, m, cache_namespace=""):
    """Return a cache path keyed by the source field and multipole operator."""
    sign = "m" if m >= 0 else "m_minus"
    suffix = f"{m}" if m >= 0 else f"{-m}"
    payload = {
        "schema": MULTIPOLE_FIELD_CACHE_SCHEMA_VERSION,
        "field": _field_cache_signature(field),
        "binning_window": serialize_window_params(binning_window),
        "l": int(l),
        "m": int(m),
    }
    payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    namespace = str(cache_namespace).strip()
    prefix = f"{namespace}_" if namespace else ""
    return Path(cache_dir) / f"{prefix}win_{digest}_l{l}_{sign}{suffix}.npy"


def _multipole_window_params(binning_window, l, m):
    """Convert an edge-binning window spec into a WindowFunc multipole spec."""
    if not isinstance(binning_window, dict):
        raise TypeError("binning_window must be a dictionary.")
    radial_type = str(binning_window.get("type", "")).strip().lower()
    if not radial_type:
        raise ValueError("binning_window must define a non-empty 'type'.")
    len_args = binning_window.get("len_args", {})
    if len_args is None:
        len_args = {}
    return {
        "type": "radial_multipole",
        "len_args": dict(len_args),
        "other_args": {
            "radial_type": radial_type,
            "l": int(l),
            "m": int(m),
            "profile_config": copy.deepcopy(binning_window.get("other_args", {})),
        },
        "kernel_mode": "complex_full_fft",
    }


def _prepare_legendre_convolution_context(field):
    """
    Precompute shared wavelet-spectrum inputs for Legendre convolutions.

    Legendre multipole windows currently use only the base Fourier band,
    equivalent to ``bandwidth=1`` in the real-window builder. Support for
    higher-band Legendre window construction may be added later.
    """
    return {
        "phi_fourier_power": fourier_power_spectrum(field.phi_array, 0, 1, field.L, field.phi_resolution),
    }


def _stream_convolution_fields(
    field,
    binning_window,
    l,
    threads,
    m_values=None,
    cache_multipole_fields=False,
    cache_dir="",
    conv_context=None,
    cache_namespace="",
):
    """Generate or load convolved fields for selected m values at one edge window."""
    if conv_context is None:
        conv_context = _prepare_legendre_convolution_context(field)
    phi_fourier_power = conv_context["phi_fourier_power"]
    if m_values is None:
        m_values = range(-l, l + 1)
    m_fields = []
    for m in m_values:
        cached = None
        cache_path = None
        if cache_multipole_fields and cache_dir:
            cache_path = _cache_file_path(
                cache_dir, field, binning_window, l, m, cache_namespace=cache_namespace
            )
            if cache_path.exists():
                try:
                    cached = np.load(cache_path)
                except (OSError, ValueError):
                    cached = None
                if cached is not None and cached.shape != field.epsilon.shape:
                    cached = None
        if cached is None:
            window = WindowFunc(
                _multipole_window_params(binning_window, l, m),
                field.sfc_info,
                bandwidth=1,
                threads=threads,
                phi_array=field.phi_array,
                phi_fourier_power=phi_fourier_power,
            )
            cached = (field @ window).epsilon
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = cache_path.with_name(
                    f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npy"
                )
                try:
                    np.save(temp_path, cached)
                    os.replace(temp_path, cache_path)
                finally:
                    if temp_path.exists():
                        temp_path.unlink()
        m_fields.append(np.ascontiguousarray(cached, dtype=np.complex128))
    return m_fields


def _prepare_multipole_gpu_context(field1, gpu_device_id=0, gpu_threads_per_block=(8, 8, 8)):
    """Allocate reusable CUDA state for multipole m-term summation."""
    if not cuda.is_available():
        raise RuntimeError("CUDA is required for Corr_3PCF_Multipole, but no CUDA device is available.")

    cuda.select_device(int(gpu_device_id))
    threads_per_block = normalize_gpu_threads_per_block(gpu_threads_per_block)
    gamma = np.ascontiguousarray(cal_gamma(field1.phi_array, field1.phi_support, field1.phi_resolution), dtype=np.float64)
    gamma_gpu = cuda.to_device(gamma)
    data_gpu = cuda.to_device(np.ascontiguousarray(field1.epsilon, dtype=np.float64))
    result_gpu = cuda.device_array(field1.epsilon.shape, dtype=np.complex128)
    n_result = field1.epsilon.size
    result_gpu_flat = result_gpu.reshape(n_result)
    blocks_per_grid = (
        (field1.L + threads_per_block[0] - 1) // threads_per_block[0],
        (field1.L + threads_per_block[1] - 1) // threads_per_block[1],
        (field1.L + threads_per_block[2] - 1) // threads_per_block[2],
    )
    reduce_blocks = min(1024, (n_result + REDUCE_THREADS - 1) // REDUCE_THREADS)
    partial_real_gpu = cuda.device_array(reduce_blocks, dtype=np.float64)
    partial_imag_gpu = cuda.device_array(reduce_blocks, dtype=np.float64)
    return {
        "backend": "gpu",
        "gpu_algorithm": "exact_separable_signed",
        "work_x_gpu": cuda.device_array(field1.epsilon.shape, dtype=np.complex128),
        "work_xy_gpu": cuda.device_array(field1.epsilon.shape, dtype=np.complex128),
        "separable_threads_per_block": int(np.prod(threads_per_block)),
        "separable_blocks_per_grid": (n_result + int(np.prod(threads_per_block)) - 1) // int(np.prod(threads_per_block)),
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
        "phi_support": field1.phi_support,
    }


def _prepare_multipole_cpu_context(field1):
    """Allocate reusable CPU state for multipole m-term summation."""
    gamma = np.ascontiguousarray(cal_gamma(field1.phi_array, field1.phi_support, field1.phi_resolution), dtype=np.float64)
    data = np.ascontiguousarray(field1.epsilon, dtype=np.float64)
    return {
        "backend": "cpu",
        "gamma": gamma,
        "data": data,
        "n_result": field1.epsilon.size,
        "L": field1.L,
        "phi_support": field1.phi_support,
        "work_x": np.empty(data.shape, dtype=np.complex128),
        "work_xy": np.empty(data.shape, dtype=np.complex128),
    }


def _prepare_multipole_sum_context(
    field1,
    summation_backend="gpu",
    gpu_device_id=0,
    gpu_threads_per_block=(8, 8, 8),
):
    """Allocate reusable state for the selected multipole summation backend."""
    backend = normalize_summation_backend(summation_backend)
    if backend == "gpu":
        return _prepare_multipole_gpu_context(
            field1,
            gpu_device_id=gpu_device_id,
            gpu_threads_per_block=gpu_threads_per_block,
        )
    return _prepare_multipole_cpu_context(field1)


def compute_multipole_m_summand(field_r1_m, field_r2_m, sum_context):
    """Compute one complex m summand and return timing diagnostics."""
    backend = sum_context.get("backend", "gpu")
    if backend == "cpu":
        t_kernel_start = time.perf_counter()
        raw_sum = _compute_3d_separable_cpu(
            sum_context["data"],
            np.ascontiguousarray(field_r1_m, dtype=np.complex128),
            np.ascontiguousarray(field_r2_m, dtype=np.complex128),
            sum_context["gamma"],
            sum_context["L"],
            sum_context["phi_support"],
            sum_context["work_x"],
            sum_context["work_xy"],
        )
        kernel_elapsed = time.perf_counter() - t_kernel_start
        value = (4.0 * np.pi) * raw_sum / sum_context["n_result"]
        return value, {
            "h2d_elapsed_sec": 0.0,
            "kernel_elapsed_sec": kernel_elapsed,
            "reduce_elapsed_sec": 0.0,
            "d2h_elapsed_sec": 0.0,
        }

    t_h2d_start = time.perf_counter()
    data_r1_gpu = cuda.to_device(np.ascontiguousarray(field_r1_m, dtype=np.complex128))
    data_r2_gpu = cuda.to_device(np.ascontiguousarray(field_r2_m, dtype=np.complex128))
    h2d_elapsed = time.perf_counter() - t_h2d_start

    t_kernel_start = time.perf_counter()
    _launch_separable_gpu(sum_context, data_r1_gpu, data_r2_gpu)
    cuda.synchronize()
    kernel_elapsed = time.perf_counter() - t_kernel_start

    t_reduce_start = time.perf_counter()
    reduce_complex_sum_kernel[sum_context["reduce_blocks"], REDUCE_THREADS](
        sum_context["result_gpu_flat"],
        sum_context["partial_real_gpu"],
        sum_context["partial_imag_gpu"],
        sum_context["n_result"],
    )
    cuda.synchronize()
    reduce_elapsed = time.perf_counter() - t_reduce_start

    t_d2h_start = time.perf_counter()
    partial_real = sum_context["partial_real_gpu"].copy_to_host()
    partial_imag = sum_context["partial_imag_gpu"].copy_to_host()
    d2h_elapsed = time.perf_counter() - t_d2h_start

    del data_r1_gpu
    del data_r2_gpu

    value = (4.0 * np.pi) * complex(np.sum(partial_real), np.sum(partial_imag)) / sum_context["n_result"]
    return value, {
        "h2d_elapsed_sec": h2d_elapsed,
        "kernel_elapsed_sec": kernel_elapsed,
        "reduce_elapsed_sec": reduce_elapsed,
        "d2h_elapsed_sec": d2h_elapsed,
    }


def calc_DDD_multipole(
    deltaD1, deltaD2, deltaD3,
    binning_window12, binning_window13, l_min, l_max,
    summation_backend="gpu",
    gpu_device_id=0,
    gpu_threads_per_block=(8, 8, 8),
    cache_multipole_fields=False,
    cache_dir="",
    cache_namespace="",
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
    sum_context = _prepare_multipole_sum_context(
        deltaD1,
        summation_backend=summation_backend,
        gpu_device_id=gpu_device_id,
        gpu_threads_per_block=gpu_threads_per_block,
    )
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

    rho = deltaD1.field_mean_density(value_unit="grid") if hasattr(deltaD1, "field_mean_density") else None
    if rho is None or np.isclose(rho, 0.0):
        rho = 1.0 / deltaD1.V
    rho3 = rho ** 3
    for l_idx, l in enumerate(range(l_min, l_max + 1)):
        t_l_start = time.perf_counter()
        conv_elapsed = 0.0
        m_values = np.empty(l + 1, dtype=np.complex128)
        sum_elapsed = 0.0
        side12_namespace = f"{cache_namespace}_side12" if cache_namespace else "side12"
        side13_namespace = f"{cache_namespace}_side13" if cache_namespace else "side13"
        for m in range(0, l + 1):
            t_m_start = time.perf_counter()
            t_conv_m_start = time.perf_counter()
            field_r1_m = _stream_convolution_fields(
                deltaD2, binning_window12, l, threads=threads,
                m_values=[m],
                cache_multipole_fields=cache_multipole_fields,
                cache_dir=cache_dir,
                conv_context=conv_context_r1,
                cache_namespace=side12_namespace,
            )[0]
            field_r2_m = _stream_convolution_fields(
                deltaD3, binning_window13, l, threads=threads,
                m_values=[-m],
                cache_multipole_fields=cache_multipole_fields,
                cache_dir=cache_dir,
                conv_context=conv_context_r2,
                cache_namespace=side13_namespace,
            )[0]
            conv_m_elapsed = time.perf_counter() - t_conv_m_start
            conv_elapsed += conv_m_elapsed
            total_conv_elapsed += conv_m_elapsed
            t_sum_m_start = time.perf_counter()
            m_values[m], timing = compute_multipole_m_summand(field_r1_m, field_r2_m, sum_context)
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
