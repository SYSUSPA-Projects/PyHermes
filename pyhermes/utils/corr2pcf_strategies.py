"""Structure-driven 2PCF evaluation; no changes to alias/scaling mathematics."""

import time
from collections import OrderedDict

import numpy as np
from numba import get_num_threads, njit, prange

from pyhermes.utils import corr2pcf_window_reduction as legacy
from pyhermes.utils.convolution import _ordered_window_args
from pyhermes.utils.window_structure import describe_window, exact_grid_axis

FACTOR_CACHE_BYTES = 16 * 1024**2
FACTOR_SCRATCH_LIMIT_BYTES = 64 * 1024**2


# Dynamic dispatcher arguments cannot safely persist weak references in Numba
# cache indices. Specializations are reused in-process; warmup is explicit.
@njit(parallel=True)
def radial_function_integral(norms, weights, L, function, args):
    partials = np.zeros(len(norms), dtype=np.float64)
    for index in prange(len(norms)):
        frequency = np.sqrt(float(norms[index])) / L
        partials[index] = weights[index] * function(frequency, 0.0, 0.0, *args)
    cells = float(L * L * L)
    return np.sum(partials) / (cells * cells)


@njit(parallel=True)
def _transverse_table(L, bandwidth, args, function):
    count = bandwidth * L + 1
    values = np.empty((count, count), dtype=np.float64)
    inv_L = 1.0 / L
    for a in prange(count):
        for b in range(count):
            frequency = np.sqrt((a * inv_L) ** 2 + (b * inv_L) ** 2)
            values[a, b] = function(frequency, *args)
    return values


@njit
def _longitudinal_table(L, bandwidth, args, function):
    values = np.empty(bandwidth * L + 1, dtype=np.float64)
    inv_L = 1.0 / L
    for c in range(len(values)):
        values[c] = function(c * inv_L, *args)
    return values


def transverse_table(L, bandwidth, lengths, function):
    args = lengths if isinstance(lengths, tuple) else (lengths,)
    return _transverse_table(L, bandwidth, args, function)


def longitudinal_table(L, bandwidth, lengths, function):
    args = lengths if isinstance(lengths, tuple) else (lengths,)
    return _longitudinal_table(L, bandwidth, args, function)


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


@njit(parallel=True, cache=True)
def octant_factor_integrals_batch(
    power, bandwidth, phi, transverse, longitudinal, axis
):
    """Share fold/scaling work across a bounded batch; never across aliases."""
    L = power.shape[0]
    count = transverse.shape[0]
    partials = np.zeros((L + 1, count), dtype=np.float64)
    for i in prange(L + 1):
        accumulated = np.zeros(count, dtype=np.float64)
        terms = np.empty(count, dtype=np.float64)
        for j in range(L + 1):
            for k in range(L + 1):
                weight = legacy.octant_fold_weight(power, i, j, k)
                for query in range(count):
                    terms[query] = 0.0
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
                            alias_scaling = phi[a] * phi[b] * phi[c]
                            for query in range(count):
                                terms[query] += alias_scaling * (
                                    transverse[query, u, v] * longitudinal[query, w]
                                )
                for query in range(count):
                    accumulated[query] += weight * terms[query]
        for query in range(count):
            partials[i, query] = accumulated[query]
    result = np.zeros(count, dtype=np.float64)
    cells = float(L * L * L)
    for query in range(count):
        for i in range(L + 1):
            result[query] += partials[i, query]
        result[query] /= cells * cells
    return result


class _FactorTableCache:
    """Per-run bounded factor ownership, with the existing diagnostic metrics."""

    def __init__(self, metrics):
        self._values = OrderedDict()
        self._bytes = 0
        self.metrics = metrics

    def get(self, key, builder):
        if key in self._values:
            value = self._values.pop(key)
            self._values[key] = value
            self.metrics["factor_cache_hits"] += 1
            return value
        started = time.perf_counter()
        value = builder()
        self.metrics["factor_table_seconds"] += time.perf_counter() - started
        self.metrics["factor_table_build_count"] += 1
        if value.nbytes <= FACTOR_CACHE_BYTES:
            while self._values and self._bytes + value.nbytes > FACTOR_CACHE_BYTES:
                _, old = self._values.popitem(last=False)
                self._bytes -= old.nbytes
            self._values[key] = value
            self._bytes += value.nbytes
            self.metrics["factor_cache_peak_bytes"] = max(
                self.metrics["factor_cache_peak_bytes"], self._bytes
            )
        return value


class AutoWindowEvaluator(legacy.ScalarWindowEvaluator):
    """A bounded per-run planner; custom functions never inferred from names."""

    def __init__(self, reducer):
        super().__init__(reducer, radial_edges=True)
        self.radial_results = {}
        self.factor_results = {}
        self.metrics.update(
            auto_plan="lazy",
            window_strategy_counts={},
            generic_radial_evaluate_seconds=0.0,
            factor_table_seconds=0.0,
            factor_reduce_seconds=0.0,
            factor_cache_peak_bytes=0,
            factor_table_build_count=0,
            factor_cache_hits=0,
            factor_batch_count=0,
            factor_batch_max_size=0,
        )
        self._factor_cache = _FactorTableCache(self.metrics)

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
        queries = self._radial_queries(windows, comm)
        if queries is not None:
            self._prepare_radial_results(windows[0], queries, comm)

    @staticmethod
    def _query_descriptor(window, signature):
        args = None
        if signature is not None:
            args = (
                signature[1], signature[2],
                tuple(t.__module__ + "." + t.__qualname__ for t in signature[3]),
            )
        return (
            window.type, getattr(window.func, "__module__", ""),
            getattr(window.func, "__name__", ""), args,
        )

    def _radial_queries(self, windows, comm):
        """Validate shared queries and resources before any grouping collective."""
        reference = windows[0]
        signatures = [self.signature(w) for w in windows]
        eligible = (
            len(windows) >= max(2, comm.Get_size())
            and all(s is not None for s in signatures)
            and legacy.same_window_basis(windows)
        )
        scratch = legacy.radial_scratch_bytes(
            reference.L, reference.bandwidth, get_num_threads()
        )
        self.metrics["radial_scratch_bound_bytes"] = scratch
        eligible = eligible and scratch <= legacy.RADIAL_SCRATCH_LIMIT_BYTES
        descriptor = (
            reference.L, reference.bandwidth,
            [self._query_descriptor(w, s) for w, s in zip(windows, signatures)],
            len(self.reducer.powers),
        )
        if not legacy.matching_rank_descriptors(comm, descriptor, eligible):
            self.metrics["auto_plan"] = "fused_radial_guard"
            return None
        return list(dict.fromkeys(signatures))

    def _prepare_radial_results(self, reference, queries, comm):
        """Evaluate compatible radial queries with bounded per-pair scratch."""
        powers = list(self.reducer.powers.items())
        zero_modes = sorted({s[2] for s in queries})
        # Process one field pair/zero-mode convention at a time. Keeping all
        # compact histograms at once would multiply the advertised scratch bound.
        local = []
        for pair_index, (_, power) in enumerate(powers):
            for zero_dc in zero_modes:
                combined = self._group_radial_power(
                    power, reference, comm, zero_dc=zero_dc
                )
                norms = np.flatnonzero(combined).astype(np.int64)
                weights = combined[norms]
                del combined
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

    @staticmethod
    def _factor_axis(window):
        if not legacy.supports_octant(window):
            return None
        structure = describe_window(window)
        if structure.kind == "separable_cartesian":
            return 2
        return exact_grid_axis(window) if structure.kind == "separable_axial" else None

    @classmethod
    def needs_factor_plan(cls, window):
        return window is not None and cls._factor_axis(window) is not None

    def prepare_local(self, windows):
        """Plan local separable batches without MPI collectives or full grids."""
        plan = self._factor_plan(windows)
        if plan is None:
            return
        axis, unique, batch_size = plan
        powers = list(self.reducer.powers.items())
        for begin in range(0, len(unique), batch_size):
            self._prepare_factor_batch(
                unique[begin:begin + batch_size], powers, windows[0], axis
            )

    def _factor_plan(self, windows):
        """Select compatible queries while preserving the existing scratch guard."""
        if len(windows) < 4:
            return
        axis = self._factor_axis(windows[0])
        reference = windows[0]
        if (axis is None or not legacy.same_window_basis(windows)
                or any(self._factor_axis(w) != axis for w in windows)):
            return
        count = reference.L * reference.bandwidth + 1
        # Original tables + stacked batch arrays fit the same scratch guard.
        bytes_per_query = (count * count + count) * 8
        batch_size = min(8, FACTOR_SCRATCH_LIMIT_BYTES // (2 * bytes_per_query))
        if batch_size < 2:
            return
        signatures = [self.signature(w) for w in windows]
        if any(key is None for key in signatures):
            return
        unique = list(dict(zip(signatures, windows)).items())
        powers = list(self.reducer.powers.items())
        if any(
            p.shape != (reference.L, reference.L, reference.L // 2 + 1)
            for _, p in powers
        ):
            return
        return axis, unique, batch_size

    def _prepare_factor_batch(self, block, powers, reference, axis):
        factors = [self._factor_tables(w, describe_window(w)) for _, w in block]
        started = time.perf_counter()
        transverse = np.stack([f[0] for f in factors])
        longitudinal = np.stack([f[1] for f in factors])
        for pair, power in powers:
            values = octant_factor_integrals_batch(
                power, reference.bandwidth, reference.phi_fourier_power,
                transverse, longitudinal, axis,
            )
            for (key, _), value in zip(block, values):
                self.factor_results[pair, key] = float(value)
        self.metrics["factor_reduce_seconds"] += time.perf_counter() - started
        self.metrics["factor_batch_count"] += len(powers)
        self.metrics["factor_batch_max_size"] = max(
            self.metrics["factor_batch_max_size"], len(block)
        )

    def _longitudinal(self, window, function, length):
        return self._factor_cache.get(
            ("longitudinal", window.L, window.bandwidth, function, length),
            lambda: longitudinal_table(window.L, window.bandwidth, length, function),
        )

    def _factor_tables(self, window, structure):
        if structure.kind == "separable_cartesian":
            x, y, z = (
                float(window.rescale_len_args[key]) / 2 for key in ("Lx", "Ly", "Lz")
            )
            # A rectangular window is three sinc factors. Its xy outer product
            # uses the same 2D x 1D reduction and bounded cache as axial windows.
            first = self._longitudinal(window, structure.transverse, x)
            second = self._longitudinal(window, structure.transverse, y)
            transverse = self._factor_cache.get(
                (
                    "cartesian_xy",
                    window.L,
                    window.bandwidth,
                    structure.transverse,
                    x,
                    y,
                ),
                lambda: np.multiply.outer(first, second),
            )
            return transverse, self._longitudinal(window, structure.longitudinal, z)
        lengths = tuple(
            float(window.rescale_len_args[key]) for key in structure.transverse_args
        )
        height = float(window.rescale_len_args["H"])
        transverse = self._factor_cache.get(
            ("transverse", window.L, window.bandwidth, structure.transverse, lengths),
            lambda: transverse_table(
                window.L, window.bandwidth, lengths, structure.transverse
            ),
        )
        return transverse, self._longitudinal(window, structure.longitudinal, height)

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
        signature = self.signature(window) if structure.kind != "general" else None
        if signature is not None and (pair, signature) in self.factor_results:
            self._count(structure.kind + "_batch")
            return self.factor_results[pair, signature]
        if signature is not None and (pair, signature) in self.radial_results:
            self._count("radial")
            return self.radial_results[pair, signature]
        if structure.ball_bounds is not None and self.metrics["shell_edges_active"]:
            self._count("radial_shell_edges")
            return super().reduce(first, second, window)
        axis = self._factor_axis(window)
        count = window.bandwidth * window.L + 1
        if (
            axis is not None
            and (count * count + count) * 8 <= FACTOR_SCRATCH_LIMIT_BYTES
        ):
            transverse, longitudinal = self._factor_tables(window, structure)
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
            self._count(structure.kind)
            return float(result)
        self._count("fused")
        return super().reduce(first, second, window)
