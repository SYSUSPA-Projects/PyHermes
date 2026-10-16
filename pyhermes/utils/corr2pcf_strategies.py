"""Structure-driven 2PCF evaluation; no changes to alias/scaling mathematics."""

import time
from collections import OrderedDict

import numpy as np
from numba import get_num_threads, njit, prange

from pyhermes.utils import corr2pcf_window_reduction as legacy
from pyhermes.utils.convolution import _ordered_window_args
from pyhermes.utils.mpi_util import MPI
from pyhermes.utils.window_structure import describe_window, exact_grid_axis

FACTOR_CACHE_BYTES = 16 * 1024**2
FACTOR_SCRATCH_LIMIT_BYTES = 64 * 1024**2


@njit(parallel=True, cache=True)
def radial_function_integral(norms, weights, L, function, args):
    partials = np.zeros(len(norms), dtype=np.float64)
    for index in prange(len(norms)):
        frequency = np.sqrt(float(norms[index])) / L
        partials[index] = weights[index] * function(frequency, 0.0, 0.0, *args)
    cells = float(L * L * L)
    return np.sum(partials) / (cells * cells)


@njit(parallel=True, cache=True)
def transverse_table(L, bandwidth, scale, function):
    count = bandwidth * L + 1
    values = np.empty((count, count), dtype=np.float64)
    inv_L = 1.0 / L
    for a in prange(count):
        for b in range(count):
            frequency = np.sqrt((a * inv_L) ** 2 + (b * inv_L) ** 2)
            values[a, b] = function(2 * np.pi * frequency * scale)
    return values


@njit(cache=True)
def longitudinal_table(L, bandwidth, scale, function):
    values = np.empty(bandwidth * L + 1, dtype=np.float64)
    inv_L = 1.0 / L
    for c in range(len(values)):
        values[c] = function(2 * np.pi * (c * inv_L) * scale)
    return values


@njit(parallel=True, cache=True)
def octant_factor_integral(power, bandwidth, phi, transverse, longitudinal, axis):
    L = power.shape[0]
    partials = np.zeros(L + 1, dtype=np.float64)
    for i in prange(L + 1):
        for j in range(L + 1):
            value = 0.0
            for k in range(L + 1):
                weight = legacy.octant_fold_weight(power, i, j, k)
                temp = 0.0
                for ii in range(bandwidth):
                    for jj in range(bandwidth):
                        for kk in range(bandwidth):
                            a, b, c = ii * L + i, jj * L + j, kk * L + k
                            if axis == 2:
                                u, v, w = a, b, c
                            elif axis == 1:
                                u, v, w = a, c, b
                            else:
                                u, v, w = b, c, a
                            # Preserve each alias's phi product inside the sum,
                            # and preserve the original factor multiplication order.
                            temp += (
                                phi[a]
                                * phi[b]
                                * phi[c]
                                * (transverse[u, v] * longitudinal[w])
                            )
                value += weight * temp
            partials[i] += value
    cells = float(L * L * L)
    return np.sum(partials) / (cells * cells)


class AutoWindowEvaluator(legacy.ScalarWindowEvaluator):
    """A bounded per-run planner; custom functions never inferred from names."""

    def __init__(self, reducer):
        super().__init__(reducer, radial_edges=True)
        self.factor_cache = OrderedDict()
        self.factor_cache_bytes = 0
        self.radial_results = {}
        self.metrics.update(
            auto_plan="lazy",
            window_strategy_counts={},
            generic_radial_evaluate_seconds=0.0,
            factor_table_seconds=0.0,
            factor_reduce_seconds=0.0,
            factor_cache_peak_bytes=0,
            factor_table_build_count=0,
            factor_cache_hits=0,
        )

    @staticmethod
    def needs_radial_plan(window):
        return (
            window is not None
            and legacy.supports_octant(window)
            and describe_window(window).kind == "radial"
        )

    @staticmethod
    def signature(window):
        args = _ordered_window_args(window.func, window.window_args)
        try:
            valid = all(
                np.isscalar(value) and np.isrealobj(value) and np.isfinite(value)
                for value in args
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            return None
        return (
            window.func,
            tuple(args),
            describe_window(window).zero_dc,
            tuple(type(value) for value in args),
        )

    def _count(self, name):
        counts = self.metrics["window_strategy_counts"]
        counts[name] = counts.get(name, 0) + 1

    def prepare_auto(self, windows, comm):
        """All histogram/collective/evaluation work stays inside main-loop time."""
        if not windows or not all(self.needs_radial_plan(w) for w in windows):
            self.metrics["auto_plan"] = "general_or_mixed"
            return
        if all(legacy.shell_bounds(w) is not None for w in windows):
            self.prepare_shell_edges(windows, comm)
            self.metrics["auto_plan"] = (
                "radial_shell_edges"
                if self.metrics["radial_edges_active"]
                else "shell_edges_batch"
            )
            return
        reference = windows[0]
        signatures = [self.signature(w) for w in windows]
        eligible = (
            len(windows) >= max(2, comm.Get_size())
            and all(s is not None for s in signatures)
            and all(
                w.L == reference.L
                and w.bandwidth == reference.bandwidth
                and np.array_equal(w.phi_fourier_power, reference.phi_fourier_power)
                for w in windows
            )
        )
        scratch = legacy.radial_scratch_bytes(
            reference.L, reference.bandwidth, get_num_threads()
        )
        self.metrics["radial_scratch_bound_bytes"] = scratch
        eligible = eligible and scratch <= legacy.RADIAL_SCRATCH_LIMIT_BYTES
        descriptor = [
            (
                w.type,
                getattr(w.func, "__module__", ""),
                getattr(w.func, "__name__", ""),
                (
                    (
                        s[1],
                        s[2],
                        tuple(t.__module__ + "." + t.__qualname__ for t in s[3]),
                    )
                    if s
                    else None
                ),
            )
            for w, s in zip(windows, signatures)
        ]
        descriptors = comm.allgather(
            (reference.L, reference.bandwidth, descriptor, len(self.reducer.powers))
        )
        if not all(comm.allgather(eligible)) or any(
            d != descriptors[0] for d in descriptors
        ):
            self.metrics["auto_plan"] = "fused_radial_guard"
            return
        queries = list(dict.fromkeys(signatures))
        powers = list(self.reducer.powers.items())
        zero_modes = sorted({s[2] for s in queries})
        # Process one field pair/zero-mode convention at a time. Keeping all
        # compact histograms at once would multiply the advertised scratch bound.
        local = []
        for pair_index, (_, power) in enumerate(powers):
            for zero_dc in zero_modes:
                started = time.perf_counter()
                histogram = legacy.octant_radial_power(
                    power,
                    reference.bandwidth,
                    reference.phi_fourier_power,
                    comm.Get_rank(),
                    comm.Get_size(),
                    zero_dc=zero_dc,
                )
                self.metrics["radial_group_seconds"] += time.perf_counter() - started
                self.metrics["radial_group_count"] += 1
                started = time.perf_counter()
                combined = np.empty_like(histogram)
                comm.Allreduce(histogram, combined, op=MPI.SUM)
                self.metrics["shell_edge_collective_seconds"] += (
                    time.perf_counter() - started
                )
                norms = np.flatnonzero(combined).astype(np.int64)
                weights = combined[norms]
                del histogram, combined
                self.metrics["radial_occupied_bins"] = max(
                    self.metrics["radial_occupied_bins"], len(norms)
                )
                started = time.perf_counter()
                for index in range(comm.Get_rank(), len(queries), comm.Get_size()):
                    function, args, query_zero_dc, _ = queries[index]
                    if query_zero_dc == zero_dc:
                        value = float(
                            radial_function_integral(
                                norms, weights, reference.L, function, args
                            )
                        )
                        local.append((pair_index, index, value))
                self.metrics["generic_radial_evaluate_seconds"] += (
                    time.perf_counter() - started
                )
                del norms, weights
        started = time.perf_counter()
        gathered = comm.allgather(local)
        self.metrics["shell_edge_collective_seconds"] += time.perf_counter() - started
        for rows in gathered:
            for pair_index, index, value in rows:
                pair = powers[pair_index][0]
                self.radial_results[pair, queries[index]] = value
        self.metrics["auto_plan"] = "radial"

    def _table(self, key, builder):
        if key in self.factor_cache:
            value = self.factor_cache.pop(key)
            self.factor_cache[key] = value
            self.metrics["factor_cache_hits"] += 1
            return value
        started = time.perf_counter()
        value = builder()
        self.metrics["factor_table_seconds"] += time.perf_counter() - started
        self.metrics["factor_table_build_count"] += 1
        if value.nbytes <= FACTOR_CACHE_BYTES:
            while (
                self.factor_cache
                and self.factor_cache_bytes + value.nbytes > FACTOR_CACHE_BYTES
            ):
                _, old = self.factor_cache.popitem(last=False)
                self.factor_cache_bytes -= old.nbytes
            self.factor_cache[key] = value
            self.factor_cache_bytes += value.nbytes
            self.metrics["factor_cache_peak_bytes"] = max(
                self.metrics["factor_cache_peak_bytes"], self.factor_cache_bytes
            )
        return value

    def reduce(self, first, second, window):
        if not legacy.supports_octant(window):
            self._count("materialized")
            return None
        pair = self.reducer._pair_key(first, second)
        power = self.reducer.powers.get(pair)
        if power is None or power.shape != (window.L, window.L, window.L // 2 + 1):
            self._count("materialized")
            return None
        structure = describe_window(window)
        signature = self.signature(window) if structure.kind == "radial" else None
        if signature is not None and (pair, signature) in self.radial_results:
            self._count("radial")
            return self.radial_results[pair, signature]
        if structure.ball_bounds is not None and self.metrics["shell_edges_active"]:
            self._count("radial_shell_edges")
            return super().reduce(first, second, window)
        axis = exact_grid_axis(window) if structure.kind == "separable_axial" else None
        count = window.bandwidth * window.L + 1
        if (
            axis is not None
            and (count * count + count) * 8 <= FACTOR_SCRATCH_LIMIT_BYTES
        ):
            radius, height = (float(window.rescale_len_args[k]) for k in ("R", "H"))
            transverse = self._table(
                (
                    "transverse",
                    window.L,
                    window.bandwidth,
                    structure.transverse,
                    radius,
                ),
                lambda: transverse_table(
                    window.L, window.bandwidth, radius, structure.transverse
                ),
            )
            longitudinal = self._table(
                (
                    "longitudinal",
                    window.L,
                    window.bandwidth,
                    structure.longitudinal,
                    height,
                ),
                lambda: longitudinal_table(
                    window.L, window.bandwidth, height, structure.longitudinal
                ),
            )
            started = time.perf_counter()
            result = octant_factor_integral(
                power,
                window.bandwidth,
                window.phi_fourier_power,
                transverse,
                longitudinal,
                axis,
            )
            self.metrics["factor_reduce_seconds"] += time.perf_counter() - started
            self._count("separable_axial")
            return float(result)
        self._count("fused")
        return super().reduce(first, second, window)
