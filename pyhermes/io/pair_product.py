"""Local windowed pair products, reusing exact 2PCF strategies."""

import copy

import numpy as np

from pyhermes.utils.corr2pcf_spectral import SpectralPairReducer
from pyhermes.utils.corr2pcf_strategies import AutoWindowEvaluator
from pyhermes.utils.mpi_util import MPI
from pyhermes.utils.runtime import configure

from .sfc_field import SFCField
from .window import WindowFunc


class PairProduct:
    """A local reusable representation of ``mean((first @ W) * second)``.

    Real source fields must have matching SFC geometry. Their FFTs/pair power
    are prepared once; ``evaluate(windows)`` also shares structural window
    plans. This returns a coefficient-array mean, not an automatically
    RR-normalized xi. No MPI world collectives are performed.

    Keep input arrays and metadata unchanged while the object is alive.
    Array replacement is detected; in-place edits cannot be detected cheaply.
    Use the context manager or ``close()`` to release retained powers/fields.
    Real symmetric kernels use the real cross-power. Complex-response windows
    retain the full convolution, including the phase information it requires.
    """

    def __init__(self, first, second=None, *, threads=None):
        second = first if second is None else second
        for value in (first, second):
            if not isinstance(value, SFCField):
                raise TypeError("PairProduct inputs must be SFCField objects.")
            if not isinstance(value.epsilon, np.ndarray) or value.epsilon.ndim != 3:
                raise ValueError("PairProduct requires populated three-dimensional fields.")
            if np.iscomplexobj(value.epsilon):
                raise TypeError("PairProduct currently accepts real source fields only.")
        self._geometry = {key: first.sfc_info[key] for key in SFCField._REQUIRED_ARGV}
        if any(second.sfc_info.get(key) != value for key, value in self._geometry.items()):
            raise ValueError("PairProduct inputs must have matching SFC geometry and basis.")
        length = 1 << int(self._geometry['J'])
        if first.epsilon.shape != (length,) * 3 or second.epsilon.shape != first.epsilon.shape:
            raise ValueError("PairProduct input shapes must match their SFC grid resolution.")
        self.threads = max(1, int(threads if threads is not None else first.threads or 1))
        configure(threads=self.threads)
        self._first, self._second = first, second
        self._arrays = (first.epsilon, second.epsilon)
        self._reducer = SpectralPairReducer(threads=self.threads)
        self._reducer.prepare([(first, second)])
        self._metrics = dict(evaluated_windows=0, fallback_convolution_count=0)
        self._closed = False

    @property
    def metrics(self):
        """Diagnostic counters/times; values are copies and remain after close."""
        result = dict(self._metrics)
        if self._reducer is not None:
            result.update(self._reducer.metrics)
        return copy.deepcopy(result)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("PairProduct is closed; create a new pair for another run.")
        if (self._first.epsilon is not self._arrays[0]
                or self._second.epsilon is not self._arrays[1]):
            raise RuntimeError("A source array was replaced; create a new PairProduct.")
        for field in (self._first, self._second):
            if any(field.sfc_info.get(key) != value for key, value in self._geometry.items()):
                raise RuntimeError("Source geometry changed; create a new PairProduct.")

    def _check_window(self, window):
        if not isinstance(window, WindowFunc):
            raise TypeError("PairProduct evaluation requires WindowFunc objects.")
        if (window.L != 1 << int(self._geometry['J'])
                or any(getattr(window, key, None) != value
                       for key, value in self._geometry.items() if key != 'J')):
            raise ValueError("Window and pair must have matching SFC geometry and basis.")

    def evaluate(self, windows):
        """Return values in input order, planning the complete local window list.

        Duplicate and mixed windows are allowed. Unsupported/complex responses
        use ``mean((first @ window) * second)`` with the original operand order.
        Full kernels built only for a fallback query are released afterwards;
        pre-existing caller-materialized kernels are retained unchanged.
        """
        self._check_open()
        windows = list(windows)
        for window in windows:
            self._check_window(window)
        if not windows:
            return np.empty(0, dtype=np.float64)
        configure(threads=self.threads)
        evaluator = AutoWindowEvaluator(self._reducer)
        evaluator.prepare_auto(windows, getattr(MPI, 'COMM_SELF', MPI.COMM_WORLD))
        evaluator.prepare_local(windows)
        values = []
        try:
            for window in windows:
                value = evaluator.reduce(self._first, self._second, window)
                if value is None:
                    had_kernel = window.w_kernel is not None
                    try:
                        kernel = window.as_array()
                        value = self._reducer.reduce(self._first, self._second, kernel)
                        if value is None:
                            # Real cross-power is insufficient for a complex
                            # response. Retain its phase through the old path.
                            convolved = self._first @ window
                            value = np.mean(convolved.epsilon * self._second.epsilon).item()
                            self._metrics['fallback_convolution_count'] += 1
                    finally:
                        if not had_kernel:
                            window.w_kernel = None
                values.append(value)
                self._metrics['evaluated_windows'] += 1
        finally:
            self._metrics['last_window_plan'] = copy.deepcopy(evaluator.metrics)
        return np.asarray(values)

    def close(self):
        """Release powers and source references; safe to call more than once."""
        if not self._closed:
            self._metrics.update(self._reducer.metrics)
            self._reducer.powers.clear()
            self._reducer.fields.clear()
            self._reducer = None
            self._first = self._second = self._arrays = None
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
