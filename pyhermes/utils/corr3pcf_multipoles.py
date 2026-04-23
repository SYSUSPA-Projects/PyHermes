"""3PCF multipole convolution, caching, and CUDA summation helpers."""

import time
from pathlib import Path

import numba
import numpy as np
from numba import cuda

from pyhermes.utils.convolution import specialized_convolution_3d_complex
from pyhermes.utils.legendre_windows import calculate_legendre_window_array
from pyhermes.utils.wavelet_grid import power_spectrum


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
    """
    Precompute shared wavelet-spectrum inputs for Legendre convolutions.

    Legendre multipole windows currently use only the base Fourier band,
    equivalent to ``bandwidth=1`` in the real-window builder. Support for
    higher-band Legendre window construction may be added later.
    """
    return {
        "delta_xi": 1.0 / field.L,
        "power_phi": power_spectrum(field.phi_data, 0, 1, field.L, field.SampRate),
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
            window_array = calculate_legendre_window_array(field.L, delta_xi, power_phi, rescaleR, l, m)
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
