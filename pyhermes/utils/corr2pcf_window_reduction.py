"""Opt-in scalar window evaluation with the original octant folding weights.

This applies the transpose of the existing eight-term fold to pair power,
then evaluates every original octant/alias term. No scaling factor is moved
out of an alias sum and no full three-dimensional window is allocated.
"""
import time

import numpy as np
from numba import get_num_threads, njit, prange

from pyhermes.utils.convolution import _ordered_window_args
from pyhermes.utils.window_functions import window_function_sphere_numba
from pyhermes.utils.mpi_util import MPI


@njit(parallel=True, cache=True)
def octant_pair_integral(power, bandwidth, phi, function, zero_dc, *args):
    L = power.shape[0]
    partials = np.zeros(L + 1, dtype=np.float64)
    inv_L = 1.0 / L
    for i in prange(L + 1):
        x = i % L
        reflected_x = (L - i) % L
        for j in range(L + 1):
            y = j % L
            reflected_y = (L - j) % L
            value = 0.0
            for k in range(L + 1):
                # Each octant entry appears once or twice along x/y, and
                # at Nyquist twice along z. Retain those boundary multiplicities.
                if zero_dc and x == 0 and y == 0 and k % L == 0:
                    continue
                z = min(k, L - k)
                weight = power[x, y, z]
                if 0 < i < L:
                    weight += power[reflected_x, y, z]
                if 0 < j < L:
                    weight += power[x, reflected_y, z]
                    if 0 < i < L:
                        weight += power[reflected_x, reflected_y, z]
                if 0 < k < L:
                    weight *= 2.0
                temp = 0.0
                for ii in range(bandwidth):
                    for jj in range(bandwidth):
                        for kk in range(bandwidth):
                            a, b, c = ii*L+i, jj*L+j, kk*L+k
                            temp += (phi[a] * phi[b] * phi[c]
                                     * function(a*inv_L, b*inv_L, c*inv_L, *args))
                value += weight * temp
            partials[i] += value
    cells = float(L * L * L)
    return np.sum(partials) / (cells * cells)



@njit(parallel=True, cache=True)
def octant_ball_integrals_batch(power, bandwidth, phi, radii):
    """Accumulate several balls in one traversal; preserve each original alias.

    Only per-alias coordinates/scaling products and the transposed-fold power
    weight are reused across radii. There is no factorization across aliases.
    Working memory is O(L * nradii), not a grid for every radius.
    """
    L = power.shape[0]
    count = len(radii)
    result = np.zeros(count, dtype=np.float64)
    if count == 0:
        return result
    partials = np.zeros((L + 1, count), dtype=np.float64)
    inv_L = 1.0 / L
    for i in prange(L + 1):
        x, reflected_x = i % L, (L-i) % L
        accumulated = np.zeros(count, dtype=np.float64)
        terms = np.empty(count, dtype=np.float64)
        for j in range(L + 1):
            y, reflected_y = j % L, (L-j) % L
            for k in range(L + 1):
                z = min(k, L-k)
                weight = power[x, y, z]
                if 0 < i < L:
                    weight += power[reflected_x, y, z]
                if 0 < j < L:
                    weight += power[x, reflected_y, z]
                    if 0 < i < L:
                        weight += power[reflected_x, reflected_y, z]
                if 0 < k < L:
                    weight *= 2.0
                for edge in range(count):
                    terms[edge] = 0.0
                for ii in range(bandwidth):
                    for jj in range(bandwidth):
                        for kk in range(bandwidth):
                            a, b, c = ii*L+i, jj*L+j, kk*L+k
                            ka, kb, kc = a*inv_L, b*inv_L, c*inv_L
                            frequency = np.sqrt(ka**2 + kb**2 + kc**2)
                            q_base = 2*np.pi*frequency
                            alias_scaling = phi[a]*phi[b]*phi[c]
                            for edge in range(count):
                                q = q_base*radii[edge]
                                window = 1.0 if q == 0.0 else 3*(np.sin(q)-q*np.cos(q))/q**3
                                terms[edge] += alias_scaling*window
                for edge in range(count):
                    accumulated[edge] += weight*terms[edge]
        for edge in range(count):
            partials[i, edge] = accumulated[edge]
    cells = float(L*L*L)
    for edge in range(count):
        total = 0.0
        for i in range(L+1):
            total += partials[i, edge]
        result[edge] = total/(cells*cells)
    return result



# A conservative per-rank scratch bound, including thread-private bins, MPI
# buffers, compact indices/weights and evaluation scratch. No persistent cache.
RADIAL_SCRATCH_LIMIT_BYTES = 256 * 1024**2


def radial_scratch_bytes(L, bandwidth, threads):
    return (3 * (int(L) * int(bandwidth))**2 + 1) * 8 * (int(threads) + 6)


@njit(parallel=True, cache=True)
def _octant_radial_power(power, bandwidth, phi, rank, size, threads):
    """Group the original weighted alias terms by exact integer squared norm.

    Each term retains phi[a]*phi[b]*phi[c] BEFORE grouping. The MPI ranks
    partition i planes and Numba threads own disjoint histogram rows. Neither
    folding multiplicities nor any alias is removed. Signed cross-power works.
    """
    L = power.shape[0]
    bins = 3 * (bandwidth * L)**2 + 1
    rows = np.zeros((threads, bins), dtype=np.float64)
    for thread in prange(threads):
        for plane in range(thread, (L + size - rank) // size, threads):
            i = rank + plane * size
            x, reflected_x = i % L, (L-i) % L
            for j in range(L + 1):
                y, reflected_y = j % L, (L-j) % L
                for k in range(L + 1):
                    z = min(k, L-k)
                    weight = power[x, y, z]
                    if 0 < i < L:
                        weight += power[reflected_x, y, z]
                    if 0 < j < L:
                        weight += power[x, reflected_y, z]
                        if 0 < i < L:
                            weight += power[reflected_x, reflected_y, z]
                    if 0 < k < L:
                        weight *= 2.0
                    for ii in range(bandwidth):
                        for jj in range(bandwidth):
                            for kk in range(bandwidth):
                                a, b, c = ii*L+i, jj*L+j, kk*L+k
                                squared_norm = a*a + b*b + c*c
                                rows[thread, squared_norm] += weight*phi[a]*phi[b]*phi[c]
    result = np.zeros(bins, dtype=np.float64)
    for squared_norm in prange(bins):
        value = 0.0
        for thread in range(rows.shape[0]):
            value += rows[thread, squared_norm]
        result[squared_norm] = value
    return result


def octant_radial_power(power, bandwidth, phi, rank=0, size=1):
    # Explicit thread count keeps the compiled kernel disk-cacheable. Each
    # parallel work item owns its histogram even if the runtime reschedules it.
    return _octant_radial_power(power, bandwidth, phi, rank, size, get_num_threads())


@njit(parallel=True, cache=True)
def radial_ball_integrals(squared_norms, weights, L, radii):
    """Exact radial classes, without interpolation, cutoffs or fastmath."""
    result = np.zeros(len(radii), dtype=np.float64)
    cells = float(L*L*L)
    for edge in prange(len(radii)):
        value = 0.0
        for index in range(len(squared_norms)):
            q = (2*np.pi/L)*np.sqrt(float(squared_norms[index]))*radii[edge]
            window = 1.0 if q == 0.0 else 3*(np.sin(q)-q*np.cos(q))/q**3
            value += weights[index]*window
        result[edge] = value/(cells*cells)
    return result


def supports_octant(window):
    return (getattr(window, 'w_kernel', None) is None
            and not getattr(window, 'is_composite_window', False)
            and not window._requires_full_rfft_kernel()
            and not window._requires_complex_rfft_kernel()
            and not window._requires_complex_full_fft_kernel()
            and callable(window.func))


def shell_bounds(window):
    # A custom callable is authoritative, including one named binned_shell.
    if not supports_octant(window) or window.has_custom_func:
        return None
    if window.type == 'binned_shell':
        inner, outer = (window.rescale_len_args[k] for k in ('R_in', 'R_out'))
    elif window.type == 'thick_shell':
        radius, width = (window.rescale_len_args[k] for k in ('R', 'delta_R'))
        inner, outer = max(0., radius-width/2), radius+width/2
    else:
        return None
    if not (np.isfinite(inner) and np.isfinite(outer) and 0 <= inner < outer):
        return None
    return float(inner), float(outer)


class ScalarWindowEvaluator:
    """Single-run cache of scalar ball integrals, never full window/results across runs."""

    def __init__(self, reducer, batch_edges=False, radial_edges=False):
        self.reducer = reducer
        self.batch_edges = batch_edges
        self.radial_edges = radial_edges
        self.shell_values = {}
        self.metrics = dict(window_fused_seconds=0., window_fused_count=0,
                            shell_edge_compute_seconds=0., shell_edge_collective_seconds=0.,
                            shell_edge_count=0, shell_result_count=0,
                            shell_edge_sweeps=0, shell_edge_max_batch=0,
                            shell_edges_active=False, radial_edges_active=False,
                            radial_group_seconds=0., radial_evaluate_seconds=0.,
                            radial_group_count=0, radial_occupied_bins=0, radial_scratch_bound_bytes=0)

    def prepare_shell_edges(self, windows, comm):
        started = time.perf_counter()
        bounds = [shell_bounds(window) for window in windows]
        eligible = bool(windows) and all(pair is not None for pair in bounds)
        if not all(comm.allgather(eligible)):
            self.metrics['shell_edges_fallback'] = 'unsupported window; use ordinary fused/materialized path'
            return
        reference = windows[0]
        eligible = all(window.L == reference.L and window.bandwidth == reference.bandwidth
                       and np.array_equal(window.phi_fourier_power, reference.phi_fourier_power)
                       for window in windows)
        radii = sorted({radius for pair in bounds for radius in pair})
        powers = list(self.reducer.powers.items())
        descriptor = (reference.L, reference.bandwidth, radii, len(powers))
        descriptors = comm.allgather(descriptor)
        if not all(comm.allgather(eligible)) or any(item != descriptor for item in descriptors):
            self.metrics['shell_edges_fallback'] = 'different basis/edge/pair descriptors between ranks'
            return
        self.metrics['shell_edge_collective_seconds'] += time.perf_counter()-started
        local = []
        started = time.perf_counter()
        local_radii = np.asarray(radii[comm.Get_rank()::comm.Get_size()], dtype=np.float64)
        radial = self.radial_edges
        if radial:
            scratch = radial_scratch_bytes(reference.L, reference.bandwidth, get_num_threads())
            radial = all(comm.allgather(scratch <= RADIAL_SCRATCH_LIMIT_BYTES))
            self.metrics['radial_scratch_bound_bytes'] = scratch
            if not radial:
                self.metrics['radial_edges_fallback'] = 'scratch bound exceeded; use batch edges'
        if radial:
            for key, power in powers:
                stamp = time.perf_counter()
                histogram = octant_radial_power(power, reference.bandwidth,
                    reference.phi_fourier_power, comm.Get_rank(), comm.Get_size())
                self.metrics['radial_group_seconds'] += time.perf_counter()-stamp
                self.metrics['radial_group_count'] += 1
                self.metrics['shell_edge_sweeps'] += 1
                stamp = time.perf_counter()
                global_histogram = np.empty_like(histogram)
                comm.Allreduce(histogram, global_histogram, op=MPI.SUM)
                self.metrics['shell_edge_collective_seconds'] += time.perf_counter()-stamp
                stamp = time.perf_counter()
                squared_norms = np.flatnonzero(global_histogram).astype(np.int64)
                weights = global_histogram[squared_norms]
                self.metrics['radial_occupied_bins'] += len(squared_norms)
                values = radial_ball_integrals(squared_norms, weights, reference.L, local_radii)
                self.metrics['radial_evaluate_seconds'] += time.perf_counter()-stamp
                for radius, value in zip(local_radii, values):
                    self.shell_values[key, float(radius)] = float(value)
                del histogram, global_histogram, squared_norms, weights
            local = [(float(radius), [self.shell_values[key, float(radius)] for key, _ in powers])
                     for radius in local_radii]
            self.metrics['shell_edge_max_batch'] = len(local_radii)
            self.metrics['radial_edges_active'] = True
        elif self.batch_edges or self.radial_edges:
            # Bound scratch size for serial runs or unusually many edges.
            for begin in range(0, len(local_radii), 8):
                block = local_radii[begin:begin+8]
                values = [octant_ball_integrals_batch(power, reference.bandwidth,
                               reference.phi_fourier_power, block) for _, power in powers]
                self.metrics['shell_edge_sweeps'] += len(powers)
                self.metrics['shell_edge_max_batch'] = max(self.metrics['shell_edge_max_batch'],len(block))
                for offset, radius in enumerate(block):
                    local.append((float(radius), [float(value[offset]) for value in values]))
        else:
            for radius in local_radii:
                values = [float(octant_pair_integral(power, reference.bandwidth,
                                reference.phi_fourier_power, window_function_sphere_numba, False, radius))
                          for _, power in powers]
                local.append((float(radius), values))
                self.metrics['shell_edge_sweeps'] += len(powers)
                self.metrics['shell_edge_max_batch'] = 1
        self.metrics['shell_edge_compute_seconds'] = (
            self.metrics['radial_group_seconds'] + self.metrics['radial_evaluate_seconds']
            if radial else time.perf_counter()-started)
        self.metrics['shell_edge_count'] = len(local)*len(powers)
        started = time.perf_counter()
        gathered = comm.allgather(local)
        for rows in gathered:
            for radius, values in rows:
                for (key, _), value in zip(powers, values):
                    self.shell_values[key, radius] = value
        self.metrics['shell_edge_collective_seconds'] += time.perf_counter()-started
        self.metrics['shell_edges_active'] = True

    def reduce(self, first, second, window):
        key = self.reducer._pair_key(first, second)
        power = self.reducer.powers.get(key)
        if (power is None or not supports_octant(window)
                or power.shape != (window.L, window.L, window.L//2+1)):
            return None
        bounds = shell_bounds(window)
        if bounds is not None and self.metrics['shell_edges_active']:
            inner, outer = bounds
            if (key, inner) in self.shell_values and (key, outer) in self.shell_values:
                self.metrics['shell_result_count'] += 1
                return ((outer**3*self.shell_values[key, outer]
                         - inner**3*self.shell_values[key, inner]) / (outer**3-inner**3))
        args = _ordered_window_args(window.func, window.window_args)
        zero_dc = not window.has_custom_func and window.type == 'inverse_laplacian'
        started = time.perf_counter()
        value = octant_pair_integral(power, window.bandwidth, window.phi_fourier_power,
                                    window.func, zero_dc, *args)
        self.metrics['window_fused_seconds'] += time.perf_counter()-started
        self.metrics['window_fused_count'] += 1
        return float(value)
