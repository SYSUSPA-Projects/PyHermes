"""Windowed pairing: numerical equivalence, phase safety and owned lifetime."""

import numba
import numpy as np
import pytest

from pyhermes.io import PairProduct, SFCField, WindowFunc


@pytest.fixture(autouse=True)
def thread_budget():
    previous = numba.get_num_threads()
    numba.set_num_threads(2)
    yield
    numba.set_num_threads(previous)


def field(seed=1):
    result = SFCField(threads=2)
    result.sfc_info = dict(J=3, L=8, V=64.**3, box_size=64., scale_factor=.125,
                           phi_resolution=64, wavelet_mode='db2', wavelet_level=6,
                           field_kind='derived_field', weight_normalization='catalog',
                           field_integral=1.)
    result.epsilon = np.random.default_rng(seed).normal(size=(8, 8, 8))
    result.format_sfc_params()
    return result


def window(source, name='shell', **lengths):
    return WindowFunc(dict(type=name, len_args=lengths), source.sfc_info, threads=2)


def direct(first, second, win):
    return ((first @ win) * second).as_array().mean()


@pytest.mark.parametrize('kind,lengths', [
    ('shell', dict(R=6.)), ('thick_shell', dict(R=6., delta_R=2.)),
    ('gaussian_shell', dict(R_shell=6., R_smooth=2.)), ('sphere', dict(R=3.)),
    ('ring', dict(R=4., H=2.)), ('gaussian_ring', dict(R=4., H=2., sigma_perp=1.)),
    ('disk', dict(R=4., H=2.)), ('cylshell', dict(R=4., H=2.)),
    ('cylinder', dict(R=4., H=2.)), ('cubic', dict(Lx=3., Ly=4., Lz=5.)),
])
def test_same_and_cross_fields_match_existing_expression(kind, lengths):
    first = field()
    for second in (first, field(8)):
        original = first.epsilon.copy(), second.epsilon.copy()
        expected = direct(first, second, window(first, kind, **lengths))
        with first.pair_with(second) as pair:
            actual = pair @ window(first, kind, **lengths)
            assert pair.metrics['forward_fft_count'] == (1 if second is first else 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(first.epsilon, original[0])
        np.testing.assert_array_equal(second.epsilon, original[1])


def test_batch_plan_duplicates_order_and_signed_cross_power():
    first, second = field(), field(2)
    radii = [0., 8., 3., 8., 6.]
    expected = [direct(first, second, window(first, R=r)) for r in radii]
    with first.pair_with(second) as pair:
        actual = pair.evaluate(window(first, R=r) for r in radii)
        assert pair.metrics['last_window_plan']['auto_plan'] == 'radial'
        assert pair.metrics['forward_fft_count'] == 2
        assert pair.metrics['fallback_convolution_count'] == 0
        repeated = pair.evaluate([window(first, R=0.)])
        assert pair.metrics['forward_fft_count'] == 2
        assert pair.evaluate([]).shape == (0,)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(repeated[0], expected[0], rtol=1e-10, atol=1e-12)


def test_materialized_real_window_keeps_caller_kernel():
    first, second = field(), field(3)
    win = window(first, R=4.)
    kernel = win.as_array()
    expected = direct(first, second, win)
    with PairProduct(first, second) as pair:
        np.testing.assert_allclose(pair @ win, expected, rtol=1e-11, atol=1e-12)
    assert win.w_kernel is kernel


def test_complex_response_retains_cross_phase_and_orientation():
    first, second = field(), field(2)
    phase = 2*np.pi*np.indices((8, 8, 8))[0]/8
    first.epsilon = np.cos(phase)
    second.epsilon = np.sin(phase)
    win = window(first, R=0.)
    # Exact one-cell translation: an independent spatial-roll reference.
    kernel = np.broadcast_to(np.exp(-2j*np.pi*np.fft.fftfreq(8))[:, None, None], (8, 8, 5)).copy()
    win.w_kernel = kernel
    win.is_composite_window = True
    expected = np.mean(np.roll(first.epsilon, 1, axis=0)*second.epsilon)
    with first.pair_with(second) as pair:
        np.testing.assert_allclose(pair @ win, expected, rtol=1e-13, atol=1e-13)
        assert abs(expected) > .3
        assert pair.metrics['fallback_convolution_count'] == 1
    with second.pair_with(first) as pair:
        np.testing.assert_allclose(pair @ win, -expected, rtol=1e-13, atol=1e-13)
    assert win.w_kernel is kernel


def test_builtin_directional_derivative_uses_phase_preserving_fallback():
    first, second = field(), field(2)
    params = dict(type='directional_derivative', len_args={}, los_args=[1., 0., 0.])
    expected = direct(first, second, WindowFunc(params, first.sfc_info, threads=2))
    win = WindowFunc(params, first.sfc_info, threads=2)
    with first.pair_with(second) as pair:
        np.testing.assert_allclose(pair @ win, expected, rtol=1e-11, atol=1e-12)
        assert pair.metrics['fallback_convolution_count'] == 1
    assert win.w_kernel is None


def test_context_lifetime_and_replaced_arrays():
    first = field()
    pair = first.pair_with()
    first.epsilon = first.epsilon.copy()
    with pytest.raises(RuntimeError, match='replaced'):
        pair @ window(first, R=2.)
    pair.close()
    pair.close()
    assert pair._reducer is None and pair._first is None
    assert pair.metrics['forward_fft_count'] == 1
    with pytest.raises(RuntimeError, match='closed'):
        pair.evaluate([])
    with pytest.raises(ZeroDivisionError):
        with first.pair_with() as other:
            raise ZeroDivisionError
    assert other._reducer is None


def test_validation_and_mixed_window_list():
    first, second = field(), field(2)
    second.sfc_info['box_size'] = 32.
    with pytest.raises(ValueError, match='geometry'):
        first.pair_with(second)
    second = field(2)
    second.epsilon = second.epsilon.astype(complex)
    with pytest.raises(TypeError, match='real source'):
        first.pair_with(second)
    with pytest.raises(TypeError, match='SFCField'):
        first.pair_with(1.)
    with first.pair_with() as pair:
        with pytest.raises(TypeError):
            pair @ 1.
        with pytest.raises(TypeError, match='WindowFunc'):
            pair.evaluate([1.])
        mismatched = window(first, R=3.)
        mismatched.box_size = 12.
        with pytest.raises(ValueError, match='geometry'):
            pair @ mismatched
        windows = [window(first, R=3.), window(first, 'ring', R=2., H=1.)]
        expected = [direct(first, first, w) for w in windows]
        # Fresh windows to exercise fused fallback rather than cached kernels.
        actual = pair.evaluate([window(first, R=3.), window(first, 'ring', R=2., H=1.)])
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
