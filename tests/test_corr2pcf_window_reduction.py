"""Independent materialized-kernel, field-product and lifecycle checks."""
import numpy as np
import pytest
from numba import njit

from pyhermes.io import WindowFunc
from pyhermes.utils.convolution import calculate_real_window_octant_array_numba, fold_octant_window_to_rfft_kernel
from pyhermes.utils.corr2pcf_spectral import _half_spectrum_inner_product
from pyhermes.utils.corr2pcf_window_reduction import octant_pair_integral
from pyhermes.utils.window_functions import window_function_sphere_numba
from test_corr2pcf_optimization import field, task, thread_budget


@njit
def asymmetric_octant(a, b, c, scale):
    return 1 + scale*a*a + 2*b*b + 3*c*c


@pytest.mark.parametrize('bandwidth', [1, 2])
@pytest.mark.parametrize('zero_dc', [False, True])
def test_transposed_fold_preserves_every_alias_and_boundary(bandwidth, zero_dc):
    L = 8
    rng = np.random.default_rng(783)
    # Deliberately not symmetric under x/y reflection; expose lost fold terms.
    power = rng.normal(size=(L, L, L//2+1))
    phi = rng.uniform(.2, 2., bandwidth*L+1)
    octant = calculate_real_window_octant_array_numba(L, bandwidth, phi, asymmetric_octant, .7)
    kernel = fold_octant_window_to_rfft_kernel(octant)
    if zero_dc:
        kernel[0, 0, 0] = 0.
    expected = _half_spectrum_inner_product(power, kernel, L)
    actual = octant_pair_integral(power, bandwidth, phi, asymmetric_octant, zero_dc, .7)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize('mode', ['fused', 'shell_edges'])
@pytest.mark.parametrize('nonuniform', [False, True])
def test_all_products_cross_fields_match_materialized(mode, nonuniform):
    a, b = field(), field(2)
    random1, random2 = (field(3), field(4)) if nonuniform else ('uniform', 'uniform')
    products = ['dd', 'dr', 'rd', 'delta_dd', 'rr', 'xi']
    outputs = []
    for path in ['materialized', mode]:
        current = task('spectral', True, spectral_window=path, products=products)
        current.prepare_input_fields(sfc_field1=a, sfc_field2=b, random1=random1, random2=random2)
        outputs.append(current.run(save_result=False))
    for name in products:
        np.testing.assert_allclose(getattr(outputs[0], name), getattr(outputs[1], name), rtol=1e-8, atol=1e-11)
    metrics = current.runtime_metrics['per_rank'][0]
    assert current._window_evaluator is None and current._window_basis_cache == {}
    if mode == 'shell_edges':
        assert metrics['shell_edges_active']
        assert metrics['shell_edge_count'] == 4 * (5 if nonuniform else 2)
    else:
        assert metrics['window_fused_count'] > 0
    assert metrics['spectral_reduce_count'] == 0


def explicit_edges(sample, template):
    edges = [0., 4., 10., 17.]
    index = int(sample['s'])
    return dict(template, len_args={'R_in': edges[index], 'R_out': edges[index+1]})


@pytest.mark.parametrize('path', ['fused', 'shell_edges'])
def test_explicit_shared_edges_match_legacy_and_count_unique_edges(path):
    a = field()
    options = dict(binning_window=dict(type='binned_shell', len_args=['R_in', 'R_out'],
                                      mapping=explicit_edges, kernel_mode='octant'), sampling={'s': [0., 1., 2.]})
    outputs = []
    for reduction, window in [('convolution', 'materialized'), ('spectral', path)]:
        current = task(reduction, True, spectral_window=window, **options)
        current.prepare_input_fields(sfc_field1=a, sfc_field2=a)
        outputs.append(current.run(save_result=False).xi.copy())
    np.testing.assert_allclose(*outputs, rtol=1e-8, atol=1e-11)
    if path == 'shell_edges':
        metrics = current.runtime_metrics['per_rank'][0]
        assert metrics['shell_edge_count'] == 4
        assert metrics['shell_result_count'] == 3


@pytest.mark.parametrize('kind,extra', [('inverse_laplacian', {}), ('gaussian', {'R': 3.})])
def test_fused_builtin_windows_preserve_zero_mode(kind, extra):
    a = field()
    outputs = []
    window = dict(type=kind, len_args=extra, mapping='s_to_R', kernel_mode='octant')
    for path in ['materialized', 'fused']:
        current = task('spectral', True, spectral_window=path, binning_window=window, products=['dd'])
        current.prepare_input_fields(sfc_field1=a, sfc_field2=a)
        outputs.append(current.run(save_result=False).dd.copy())
    np.testing.assert_allclose(*outputs, rtol=1e-10, atol=1e-11)


def test_custom_callable_is_not_assumed_to_be_a_shell():
    a = field()
    window = dict(type='binned_shell', func=window_function_sphere_numba,
                  len_args={'R': 3.}, mapping='s_to_R', kernel_mode='octant')
    current = task('spectral', True, spectral_window='shell_edges', binning_window=window)
    current.prepare_input_fields(sfc_field1=a, sfc_field2=a)
    current.run(save_result=False)
    metric = current.runtime_metrics['per_rank'][0]
    assert not metric['shell_edges_active']
    assert metric['window_fused_count'] == 3


def test_directional_and_disk_cache_fall_back(tmp_path):
    a = field()
    for path, cache in [('fused', False), ('shell_edges', True)]:
        window = dict(type='directional_derivative', len_args={'R': 0., 'H': 0.}, mapping='s_to_R', kernel_mode='complex_rfft')
        results=[]
        for mode in ['materialized', path]:
            current = task('spectral', True, spectral_window=mode, products=['dd'], binning_window=window,
                           binning_window_cache=cache, binning_window_cache_dir=str(tmp_path))
            current.prepare_input_fields(sfc_field1=a, sfc_field2=a)
            results.append(current.run(save_result=False).dd.copy())
        np.testing.assert_allclose(*results, rtol=1e-10, atol=1e-11)
        assert current.runtime_metrics['per_rank'][0]['convolution_count'] == 3


def test_invalid_shell_edges_rejected():
    a = field()
    with pytest.raises(ValueError, match='R_in'):
        WindowFunc(dict(type='binned_shell', len_args={'R_in': 4., 'R_out': 2.}), a.sfc_info)


def test_second_run_refreshes_scalar_cache():
    a = field()
    values=[]
    current=task('spectral', True, spectral_window='shell_edges')
    for amount in [0., .0002]:
        a.epsilon += amount*np.cos(2*np.pi*np.indices(a.epsilon.shape)[0]/8)
        current.prepare_input_fields(sfc_field1=a, sfc_field2=a, random1='uniform', random2='uniform')
        values.append(current.run(save_result=False).xi.copy())
    assert not np.allclose(*values, rtol=1e-10, atol=1e-12)
    old=task()
    old.prepare_input_fields(sfc_field1=a, sfc_field2=a)
    np.testing.assert_allclose(values[-1], old.run(save_result=False).xi, rtol=1e-8, atol=1e-11)
