"""Exact-stencil and FFT-snapshot checks for the isolated optimization."""

from types import SimpleNamespace

import numba
import numpy as np
import pytest

from pyhermes.io import SFCField, WindowFunc
from pyhermes.utils import corr3pcf_multipoles as multipoles
from pyhermes.utils.wavelet_grid import sample_scaling_function


@pytest.fixture(autouse=True)
def _small_thread_budget():
    previous = numba.get_num_threads()
    numba.set_num_threads(min(previous, 2))
    yield
    numba.set_num_threads(previous)


def _field(complex_source=False):
    field = SFCField(threads=2)
    phi = sample_scaling_function("db2", 3)
    field.sfc_info = {
        "J": 3, "L": 8, "V": 8**3, "box_size": 64.0,
        "scale_factor": 8 / 64.0, "phi_resolution": 8,
        "wavelet_mode": "db2", "wavelet_level": 3, "phi_array": phi,
        "phi_support": 3,
    }
    field.format_sfc_params()
    rng = np.random.default_rng(546)
    field.epsilon = rng.normal(size=(8, 8, 8))
    if complex_source:
        field.epsilon = field.epsilon + 0.3j * rng.normal(size=field.epsilon.shape)
    return field


@pytest.mark.parametrize("complex_source", [False, True])
def test_fft_snapshot_matches_generic_convolution_across_modes(monkeypatch, complex_source):
    field = _field(complex_source)
    before = field.epsilon.copy()
    original_fft = multipoles.fftn
    calls = []

    def counted_fft(*args, **kwargs):
        calls.append(1)
        return original_fft(*args, **kwargs)

    monkeypatch.setattr(multipoles, "fftn", counted_fft)
    context = multipoles._prepare_legendre_convolution_context(field)
    snapshot = None
    for radius, ell, m in [(8.0, 0, 0), (8.0, 2, 1), (16.0, 3, -2), (16.0, 7, 3)]:
        window = {"type": "shell", "len_args": {"R": radius}}
        expected = (field @ WindowFunc(
            multipoles._multipole_window_params(window, ell, m), field.sfc_info,
            bandwidth=1, threads=2, phi_array=field.phi_array,
            phi_fourier_power=context["phi_fourier_power"],
        )).epsilon
        actual = multipoles._stream_convolution_fields(
            field, window, ell, threads=2, m_values=[m], conv_context=context,
        )[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        if snapshot is None:
            snapshot = context["source_fft"].copy()
        np.testing.assert_array_equal(context["source_fft"], snapshot)
    assert len(calls) == 1
    np.testing.assert_array_equal(field.epsilon, before)


def test_replacing_source_and_starting_new_context_do_not_use_stale_fft(monkeypatch):
    field = _field()
    original_fft = multipoles.fftn
    calls = []

    def counted_fft(*args, **kwargs):
        calls.append(1)
        return original_fft(*args, **kwargs)

    monkeypatch.setattr(multipoles, "fftn", counted_fft)
    window = {"type": "shell", "len_args": {"R": 8.0}}
    context = multipoles._prepare_legendre_convolution_context(field)
    first = multipoles._stream_convolution_fields(field, window, 1, 2, [1], conv_context=context)[0]
    field.epsilon = field.epsilon * 2
    second = multipoles._stream_convolution_fields(field, window, 1, 2, [1], conv_context=context)[0]
    np.testing.assert_allclose(second, 2 * first, rtol=2e-13, atol=2e-13)
    field.epsilon *= 3
    fresh = multipoles._prepare_legendre_convolution_context(field)
    third = multipoles._stream_convolution_fields(field, window, 1, 2, [1], conv_context=fresh)[0]
    np.testing.assert_allclose(third, 6 * first, rtol=2e-13, atol=2e-13)
    assert len(calls) == 3


def test_disk_field_cache_hit_does_not_build_source_fft(tmp_path, monkeypatch):
    field = _field()
    window = {"type": "shell", "len_args": {"R": 8.0}}
    first = multipoles._stream_convolution_fields(
        field, window, 2, 2, [1], cache_multipole_fields=True, cache_dir=str(tmp_path),
    )[0]

    def unexpected_fft(*args, **kwargs):
        raise AssertionError("Cached fields should not need a forward FFT")

    monkeypatch.setattr(multipoles, "fftn", unexpected_fft)
    second = multipoles._stream_convolution_fields(
        field, window, 2, 2, [1], cache_multipole_fields=True, cache_dir=str(tmp_path),
    )[0]
    np.testing.assert_array_equal(first, second)


def test_gamma_stencil_retains_tiny_signed_weights_and_wrapped_shifts():
    gamma = np.zeros((9, 9))
    gamma[1, 7] = -1e-300
    gamma[4, 4] = 0.7
    shifts, columns, weights, counts = multipoles._prepare_gamma_stencil(gamma, 2)
    rebuilt = np.zeros_like(gamma)
    for row, count in enumerate(counts):
        rebuilt[row, columns[row, :count]] = weights[row, :count]
    np.testing.assert_array_equal(rebuilt, gamma)
    np.testing.assert_array_equal(shifts, (np.arange(2)[None, :] - np.arange(9)[:, None] + 4) % 2)


@pytest.mark.parametrize("wavelet", ["db2", "db3"])
def test_context_reuses_stencil_and_workspace_with_dense_complex_fields(wavelet):
    phi = sample_scaling_function(wavelet, 3)
    rng = np.random.default_rng(235)
    center = SimpleNamespace(
        epsilon=rng.normal(size=(4, 4, 4)), phi_array=phi,
        phi_resolution=8, phi_support=len(phi) // 8, L=4,
    )
    context = multipoles._prepare_multipole_cpu_context(center)
    ids = [id(context[key]) for key in ["gamma_stencil", "work_x", "work_xy"]]
    for _ in range(2):
        first, second = rng.normal(size=(2, 4, 4, 4)) + 0.2j * rng.normal(size=(2, 4, 4, 4))
        value, _ = multipoles.compute_multipole_m_summand(first, second, context)
        expected = multipoles.compute_3d_summand_cpu(
            center.epsilon, first, second, context["gamma"], 4, center.phi_support,
        ) * 4 * np.pi / 4**3
        np.testing.assert_allclose(value, expected, rtol=2e-13, atol=2e-13)
        assert ids == [id(context[key]) for key in ["gamma_stencil", "work_x", "work_xy"]]
