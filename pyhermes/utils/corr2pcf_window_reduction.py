"""Opt-in scalar window evaluation with the original octant folding weights.

This applies the transpose of the existing eight-term fold to pair power,
then evaluates every original octant/alias term. No scaling factor is moved
out of an alias sum and no full three-dimensional window is allocated.
"""
import time

import numpy as np
from numba import njit, prange

from pyhermes.utils.convolution import _ordered_window_args
from pyhermes.utils.window_functions import window_function_sphere_numba


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

    def __init__(self, reducer):
        self.reducer = reducer
        self.shell_values = {}
        self.metrics = dict(window_fused_seconds=0., window_fused_count=0,
                            shell_edge_compute_seconds=0., shell_edge_collective_seconds=0.,
                            shell_edge_count=0, shell_result_count=0,
                            shell_edges_active=False)

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
        for index in range(comm.Get_rank(), len(radii), comm.Get_size()):
            radius = radii[index]
            values = [float(octant_pair_integral(power, reference.bandwidth,
                            reference.phi_fourier_power, window_function_sphere_numba, False, radius))
                      for _, power in powers]
            local.append((radius, values))
        self.metrics['shell_edge_compute_seconds'] = time.perf_counter()-started
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
