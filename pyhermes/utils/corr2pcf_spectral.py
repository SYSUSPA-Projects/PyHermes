"""Experimental scalar 2PCF reduction using unchanged, folded window kernels."""

import time

import numpy as np
from numba import njit, prange
from scipy.fft import rfftn


@njit(parallel=True, cache=True)
def _real_pair_power(first, second):
    result = np.empty(first.shape, dtype=np.float64)
    for i in prange(first.shape[0]):
        for j in range(first.shape[1]):
            for k in range(first.shape[2]):
                a, b = first[i, j, k], second[i, j, k]
                result[i, j, k] = a.real * b.real + a.imag * b.imag
    return result


@njit(parallel=True, cache=True)
def _half_spectrum_inner_product(power, kernel, last_axis_length):
    partials = np.zeros(power.shape[0], dtype=np.float64)
    for i in prange(power.shape[0]):
        value = 0.0
        for j in range(power.shape[1]):
            for k in range(power.shape[2]):
                boundary = k == 0 or (last_axis_length % 2 == 0 and k == last_axis_length // 2)
                weight = 1.0 if boundary else 2.0
                value += weight * power[i, j, k] * kernel[i, j, k]
        partials[i] = value
    cells = power.shape[0] * power.shape[1] * last_axis_length
    return np.sum(partials) / (float(cells) * float(cells))


class SpectralPairReducer:
    """Cache real pair powers only for one run of immutable prepared fields.

    Complex kernels deliberately use the existing convolution fallback.
    Scaling factors and frequency folding remain inside the original kernels.
    """

    def __init__(self, threads=1):
        self.threads = threads
        self.powers = {}
        self.fields = {}
        self.metrics = dict(forward_fft_count=0, inverse_fft_count=0,
                            source_fft_seconds=0.0, pair_power_seconds=0.0,
                            spectral_reduce_count=0, spectral_reduce_seconds=0.0)

    @staticmethod
    def _real_field(field):
        values = getattr(field, 'epsilon', None)
        return isinstance(values, np.ndarray) and values.ndim == 3 and not np.iscomplexobj(values)

    @staticmethod
    def _pair_key(first, second):
        return tuple(sorted((id(first), id(second))))

    def prepare(self, pairs):
        spectra = {}
        for first, second in pairs:
            if not self._real_field(first) or not self._real_field(second):
                continue
            if first.epsilon.shape != second.epsilon.shape:
                continue
            key = self._pair_key(first, second)
            if key in self.powers:
                continue
            for field in (first, second):
                identity = id(field)
                self.fields[identity] = field
                if identity not in spectra:
                    started = time.perf_counter()
                    spectra[identity] = rfftn(field.epsilon, workers=self.threads)
                    self.metrics['source_fft_seconds'] += time.perf_counter() - started
                    self.metrics['forward_fft_count'] += 1
            started = time.perf_counter()
            self.powers[key] = _real_pair_power(spectra[id(first)], spectra[id(second)])
            self.metrics['pair_power_seconds'] += time.perf_counter() - started
        # The real pair powers are sufficient; do not retain complex FFT copies.
        spectra.clear()

    def reduce(self, first, second, kernel):
        key = self._pair_key(first, second)
        power = self.powers.get(key)
        if power is None or np.iscomplexobj(kernel) or kernel.shape != power.shape:
            return None
        started = time.perf_counter()
        value = _half_spectrum_inner_product(power, kernel, first.epsilon.shape[-1])
        self.metrics['spectral_reduce_seconds'] += time.perf_counter() - started
        self.metrics['spectral_reduce_count'] += 1
        return float(value)
