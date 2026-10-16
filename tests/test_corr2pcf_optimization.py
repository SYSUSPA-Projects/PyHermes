"""Numerical equivalence and lifecycle tests for the opt-in 2PCF paths."""

import copy
from types import SimpleNamespace

import numba
import numpy as np
import pytest
from scipy.fft import irfftn, rfftn

from pyhermes.io import SFCField
from pyhermes.theory.corr2pcf import Corr_2PCF
from pyhermes.utils.corr2pcf_spectral import SpectralPairReducer


@pytest.fixture(autouse=True)
def thread_budget():
    previous = numba.get_num_threads()
    numba.set_num_threads(2)
    yield
    numba.set_num_threads(previous)


def field(seed=1, dtype=np.float64):
    value = SFCField(threads=2)
    value.sfc_info = dict(J=3, L=8, V=512, box_size=64., scale_factor=.125,
                          phi_resolution=64, wavelet_mode='db2', wavelet_level=6,
                          field_kind='catalog_field', weight_normalization='catalog',
                          field_integral=1., raw_field_weighted_sum=100.,
                          catalog_weight_sum=100., catalog_weight_sq_sum=100., particle_count=100)
    epsilon = np.random.default_rng(seed).uniform(.1, 1., (8, 8, 8))
    value.epsilon = (epsilon / epsilon.sum()).astype(dtype)
    value.format_sfc_params()
    return value


def task(reduction='convolution', reuse=False, **options):
    params = dict(threads=2, random='uniform', sampling={'s': [4., 8., 12.]},
                  binning_window='thick_shell', pair_reduction=reduction,
                  reuse_input_fields=reuse, products=['xi'], fout_path='')
    params['binning_window'] = dict(type='thick_shell', len_args={'R': None, 'delta_R': 4.},
                                    mapping='s_to_R', kernel_mode='octant')
    params.update(options)
    return Corr_2PCF({'Corr_2PCF': params})


@pytest.mark.parametrize('shape', [(4, 6, 8), (5, 3, 5)])
def test_spectral_matches_direct_circular_bilinear_integral(shape):
    rng = np.random.default_rng(19)
    a, b = rng.normal(size=(2, *shape))
    first, second = SimpleNamespace(epsilon=a), SimpleNamespace(epsilon=b)
    reducer = SpectralPairReducer(threads=2)
    reducer.prepare([(first, second)])
    # An independently constructed symmetric spatial kernel, including odd grids.
    kernel = rng.normal(size=shape)
    reflected = kernel.copy()
    for axis, size in enumerate(shape):
        reflected = np.take(reflected, (-np.arange(size)) % size, axis=axis)
    kernel = (kernel + reflected) / 2
    spectrum = rfftn(kernel).real
    direct = np.zeros(shape)
    for index in np.ndindex(shape):
        direct += kernel[index] * np.roll(a, index, axis=(0, 1, 2))
    expected = np.mean(direct * b)
    np.testing.assert_allclose(reducer.reduce(first, second, spectrum), expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('mode', ['dc', 'nyquist', 'interior'])
def test_dc_nyquist_and_interior_half_spectrum_weights(mode):
    coordinates = np.indices((4, 4, 8))[2]
    if mode == 'dc':
        a = np.full((4, 4, 8), 3.)
    elif mode == 'nyquist':
        a = (-1.) ** coordinates
    else:
        a = np.cos(2*np.pi*coordinates/8)
    value = SimpleNamespace(epsilon=a)
    reducer = SpectralPairReducer(threads=2)
    reducer.prepare([(value, value)])
    kernel = np.ones((4, 4, 5))
    np.testing.assert_allclose(reducer.reduce(value, value, kernel), np.mean(a*a), rtol=1e-13)
    assert reducer.metrics['forward_fft_count'] == 1


def test_spectral_reuses_inputs_and_does_not_modify_cached_power():
    a, b = field(), field(2)
    reducer = SpectralPairReducer(threads=2)
    reducer.prepare([(a, a), (a, b), (b, b), (b, a)])
    assert reducer.metrics['forward_fft_count'] == 2
    before = {key: value.copy() for key, value in reducer.powers.items()}
    kernel = np.ones((8, 8, 5))
    for scale in [1., -2., .4]:
        actual = reducer.reduce(a, b, kernel*scale)
        np.testing.assert_allclose(actual, np.mean(a.epsilon*b.epsilon)*scale, rtol=1e-12)
    for key, value in before.items():
        np.testing.assert_array_equal(reducer.powers[key], value)
    assert reducer.reduce(a, b, kernel.astype(complex)) is None


@pytest.mark.parametrize('reduction,reuse', [('convolution', True), ('spectral', False), ('spectral', True)])
@pytest.mark.parametrize('cross,nonuniform', [(False, False), (True, False), (True, True)])
def test_all_products_match_legacy_and_preserve_inputs(reduction, reuse, cross, nonuniform):
    a, b = field(), field(2)
    if not cross:
        b = a
    random1, random2 = (field(3), field(4)) if nonuniform else ('uniform', 'uniform')
    products = ['dd', 'dr', 'rd', 'delta_dd', 'rr', 'xi']
    before = a.epsilon.copy(), b.epsilon.copy(), copy.deepcopy(a.sfc_info)
    results = []
    for method, shared in [('convolution', False), (reduction, reuse)]:
        current = task(method, shared, products=products)
        current.prepare_input_fields(sfc_field1=a, sfc_field2=b, random1=random1, random2=random2)
        results.append(current.run(save_result=False))
    for name in products:
        np.testing.assert_allclose(getattr(results[0], name), getattr(results[1], name), rtol=1e-8, atol=1e-11)
    np.testing.assert_array_equal(a.epsilon, before[0])
    np.testing.assert_array_equal(b.epsilon, before[1])
    assert a.sfc_info == before[2]
    metrics = current.runtime_metrics
    assert metrics['compute_including_spectral_setup_seconds'] >= metrics['main_loop_seconds']


def test_shared_preparation_and_fft_counts_for_uniform_autocorrelation():
    a = field()
    current = task('spectral', True)
    current.prepare_input_fields(sfc_field1=a, sfc_field2=a)
    assert current.sfc_field1 is current.sfc_field2
    current.run(save_result=False)
    counts = current.runtime_metrics['per_rank'][0]
    assert counts['forward_fft_count'] == 1
    assert counts['inverse_fft_count'] == 0
    assert counts['convolution_count'] == 0
    assert counts['spectral_reduce_count'] == 3
    assert current._spectral_reducer is None


def test_different_vertex_windows_are_not_shared():
    a = field()
    reference = task()
    candidate = task('spectral', True)
    results = []
    for current in [reference, candidate]:
        current.prepare_input_fields(sfc_field1=a, sfc_field2=a,
                                     window1={'type': 'sphere', 'len_args': {'R': 3.}},
                                     window2={'type': 'sphere', 'len_args': {'R': 5.}})
        assert current.sfc_field1 is not current.sfc_field2
        results.append(current.run(save_result=False).xi)
    np.testing.assert_allclose(*results, rtol=1e-8, atol=1e-11)


@pytest.mark.parametrize('normalization', ['raw', 'unit'])
def test_normalization_conversion_is_not_skipped(normalization):
    a = field()
    before = a.epsilon.copy()
    results = []
    for method, reuse in [('convolution', False), ('spectral', True)]:
        current = task(method, reuse, weight_normalization=normalization)
        current.prepare_input_fields(sfc_field1=a, sfc_field2=a)
        results.append(current.run(save_result=False).xi)
    np.testing.assert_allclose(*results, rtol=1e-8, atol=1e-11)
    np.testing.assert_array_equal(a.epsilon, before)
    assert a.weight_normalization == 'catalog'


def test_repeated_run_refreshes_power_after_input_changes():
    a = field()
    current = task('spectral', True)
    values = []
    for shift in [0., .0001]:
        a.epsilon += shift * np.cos(2*np.pi*np.indices(a.epsilon.shape)[0]/8)
        current.prepare_input_fields(sfc_field1=a, sfc_field2=a, random1="uniform", random2="uniform")
        values.append(current.run(save_result=False).xi.copy())
    assert not np.allclose(values[0], values[1], rtol=1e-10, atol=1e-12)
    reference = task()
    reference.prepare_input_fields(sfc_field1=a, sfc_field2=a)
    np.testing.assert_allclose(values[1], reference.run(save_result=False).xi, rtol=1e-8, atol=1e-11)


def test_shared_float32_source_is_not_replaced_or_modified_by_broadcast():
    a = field(dtype=np.float32)
    epsilon = a.epsilon
    before = epsilon.copy()
    current = task('spectral', True)
    current.prepare_input_fields(sfc_field1=a, sfc_field2=a)
    current.run(save_result=False)
    assert a.epsilon is epsilon
    np.testing.assert_array_equal(a.epsilon, before)


def test_unsupported_complex_kernel_uses_legacy_fallback(monkeypatch):
    a = field()
    current = task('spectral', True, products=['dd'])
    current.prepare_input_fields(sfc_field1=a, sfc_field2=a)
    kernel = np.ones((8, 8, 5), dtype=complex) * (1+0.2j)
    window = SimpleNamespace(as_array=lambda: kernel)
    monkeypatch.setattr(current, '_build_binning_window_for_sample', lambda *args: window)
    expected = float(np.mean(irfftn(rfftn(a.epsilon)*kernel, s=a.epsilon.shape)*a.epsilon))
    result = current.run(save_result=False)
    np.testing.assert_allclose(result.dd, expected * current._pair_product_physical_scale(), rtol=1e-12)
    counts = current.runtime_metrics['per_rank'][0]
    assert counts['spectral_reduce_count'] == 0
    assert counts['convolution_count'] == 3


def test_legacy_is_the_default_and_invalid_reduction_rejected():
    current = Corr_2PCF()
    assert current.pair_reduction == 'convolution'
    assert current.reuse_input_fields is False
    with pytest.raises(ValueError, match='pair_reduction'):
        task('unknown')._sync_runtime_options()
