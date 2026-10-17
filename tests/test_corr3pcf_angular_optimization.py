"""Angular-estimator equivalence, local pair folds and product-cache lifetimes."""

import numba
import numpy as np
import pytest
from test_corr2pcf_optimization import field

from pyhermes.theory.corr2pcf import (
    compute_pair_product_at_sample,
    compute_pair_products_at_samples,
)
from pyhermes.theory.corr3pcf import Corr_3PCF
from pyhermes.utils import corr2pcf_spectral
from pyhermes.utils.corr3pcf_kernels import estimate_triplet_product_box_random_centers
from pyhermes.utils.sampling import random_box_positions
from pyhermes.utils.wavelet_grid import sample_scaling_function


@pytest.fixture(autouse=True)
def thread_budget():
    previous = numba.get_num_threads()
    numba.set_num_threads(2)
    yield
    numba.set_num_threads(previous)


def angular_field(seed=1):
    value = field(seed)
    value.sfc_info.update(phi_array=sample_scaling_function("db2", 6), phi_support=3)
    value.format_sfc_params()
    return value


class ConvolutionPairs:
    """Reference: retain the previous per-radius convolution for pair terms."""

    def _compute_pair_stats_series(self, first, second, random1, random2, radii):
        radii_arr = np.atleast_1d(np.asarray(radii, dtype=float))
        delta1 = self._compute_delta_field(first, random1)
        delta2 = self._compute_delta_field(second, random2)
        rr = np.array([self.calc_pair_product(float(r), random1, random2) for r in radii_arr])
        dd = np.array([self.calc_pair_product(float(r), delta1, delta2) for r in radii_arr])
        values = dict(rr=rr, delta_dd=dd, xi=dd / rr)
        return {key: float(value[0]) for key, value in values.items()} if np.ndim(radii) == 0 else values


def angular_task(cls=Corr_3PCF, center="box_random", nonuniform=False, smoothed=False, products=None):
    fields = [angular_field(seed) for seed in [1, 2, 3]]
    randoms = [angular_field(seed) for seed in [4, 5, 6]] if nonuniform else ["uniform"] * 3
    options = dict(sfc_field1=fields[0], sfc_field2=fields[1], sfc_field3=fields[2],
                   random1=randoms[0], random2=randoms[1], random3=randoms[2],
                   center=center, n_box_centers=12, n_rot=2, base_seed=13,
                   r12=4., r13=8., theta=[.2, .9, 1.7, 2.8],
                   products=products or ["ddd", "Q"], threads=2, fout_path="")
    if smoothed:
        options.update(window2=dict(type="sphere", len_args={"R": 1.}),
                       window3=dict(type="gaussian", len_args={"R": 2.}))
    if center == "particle":
        options.update(particle_pos1=np.array([[1., 2., 3.], [12., 24., 48.], [61., 62., 63.]]),
                       particle_weight1=np.array([1., 3., 2.]))
        if nonuniform:
            options.update(random_pos1=np.array([[2., 4., 6.], [8., 16., 32.]]),
                           random_weight1=np.array([2., 5.]))
    # Runtime SFC objects contain communicators; pass them through the preparation
    # API rather than the constructor's collectively pickled configuration.
    inputs = {name: options.pop(name) for name in ["sfc_field1", "sfc_field2", "sfc_field3", "random1", "random2", "random3"]}
    task = Corr_3PCF({"Corr_3PCF": options})
    task.prepare_input_fields(**inputs)
    if cls is ConvolutionPairs:
        task._compute_pair_stats_series = ConvolutionPairs._compute_pair_stats_series.__get__(task, Corr_3PCF)
    return task


@pytest.mark.parametrize("window", [None, dict(type="thick_shell", len_args={"R": None, "delta_R": 2.}, mapping="s_to_R"), dict(type="gaussian_shell", len_args={"R_shell": None, "R_smooth": 2.}, mapping=lambda sample, template: dict(template, len_args={"R_shell": sample["s"], "R_smooth": 2.}))])
def test_local_pair_series_matches_original_convolution(window):
    first, second = angular_field(), angular_field(2)
    samples = [{"s": r} for r in [0., 4., 8., 4.]]
    expected = [compute_pair_product_at_sample(sample, first, second, window) for sample in samples]
    actual = compute_pair_products_at_samples(samples, first, second, window)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-18)


def test_local_pair_series_tilted_window_fallback():
    first, second = angular_field(), angular_field(2)
    window = dict(type="ring", mapping="smu_to_RH", los_args={"nx": 1., "ny": 2., "nz": 3.})
    samples = [{"s": r, "mu": .3} for r in [0., 4., 8., 4.]]
    expected = [compute_pair_product_at_sample(sample, first, second, window) for sample in samples]
    np.testing.assert_allclose(compute_pair_products_at_samples(samples, first, second, window), expected, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("same", [True, False])
def test_local_pair_series_fft_reuse_and_input_ownership(monkeypatch, same):
    first = angular_field()
    second = first if same else angular_field(2)
    before = [value.epsilon.copy() for value in [first, second]]
    original = corr2pcf_spectral.rfftn
    calls = []

    def tracked(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(corr2pcf_spectral, "rfftn", tracked)
    samples = [{"s": r} for r in [0., 4., 8., 4.]]
    compute_pair_products_at_samples(samples, first, second)
    assert len(calls) == (1 if same else 2)
    first.epsilon *= .9
    expected = [compute_pair_product_at_sample(sample, first, second) for sample in samples]
    actual = compute_pair_products_at_samples(samples, first, second)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18)
    if not same:
        np.testing.assert_array_equal(second.epsilon, before[1])
    assert len(calls) == (2 if same else 4)


def test_local_pair_series_uniform_and_empty():
    first = angular_field()
    np.testing.assert_array_equal(compute_pair_products_at_samples([], first), [])
    np.testing.assert_array_equal(compute_pair_products_at_samples([{"s": 0.}, {"s": 4.}], .25, first), np.full(2, .25 * first.field_mean_density(value_unit="grid")))


@pytest.mark.parametrize("center", ["box_random", "particle"])
@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("smoothed", [False, True])
def test_complete_angular_outputs_against_convolution_pairs(center, nonuniform, smoothed):
    reference = angular_task(ConvolutionPairs, center, nonuniform, smoothed).run(save_result=False)
    optimized = angular_task(Corr_3PCF, center, nonuniform, smoothed).run(save_result=False)
    for product in ["ddd", "rrr", "delta_ddd", "d_delta_dd", "r_delta_dd", "xi12", "xi13", "xi23", "zeta", "zeta_H", "Q"]:
        expected = getattr(reference, product, None)
        if expected is not None:
            np.testing.assert_allclose(getattr(optimized, product), expected, rtol=3e-10, atol=1e-22)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_box_contrast_cache_matches_direct_sampled_estimator(monkeypatch, nonuniform):
    task = angular_task(nonuniform=nonuniform, products=["delta_ddd"])
    calls = []
    original = task._compute_delta_field

    def tracked(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(task, "_compute_delta_field", tracked)
    result = task.run(save_result=False)
    assert len(calls) == 3
    assert task._box_delta_fields is None and task.meta_sfc is None
    fields = [getattr(task, f"sfc_field{i}") for i in [1, 2, 3]]
    deltas = [original(value, getattr(task, f"random{i}")) for i, value in enumerate(fields, 1)]
    positions = random_box_positions(count=task.n_box_centers, box_size=fields[0].L, seed=task.base_seed + 1000003)
    expected = [estimate_triplet_product_box_random_centers(task.r12_scaled, task.r13_scaled, mu, positions, task.n_rot, *(value.epsilon for value in deltas), fields[0].phi_array, fields[0].L, fields[0].phi_resolution, fields[0].phi_support, task.base_seed + 1, index) for index, mu in enumerate(task.mu_arr)]
    np.testing.assert_allclose(result.delta_ddd, np.array(expected) * task._triplet_product_physical_scale(), rtol=2e-12, atol=1e-25)
    first_result = result.delta_ddd.copy()
    task.run(save_result=False)
    assert len(calls) == 6 and task._box_delta_fields is None
    np.testing.assert_array_equal(task.corr3pcf_data.delta_ddd, first_result)


def test_product_cache_is_released_after_failure(monkeypatch):
    task = angular_task(products=["delta_ddd"])

    def fail(*args, **kwargs):
        assert task._box_delta_fields is not None
        raise RuntimeError("injected kernel failure")

    monkeypatch.setattr(task, "_compute_random_center_mu", fail)
    with pytest.raises(SystemExit):
        task.run(save_result=False)
    assert task._box_delta_fields is None and task._particle_delta_fields is None


def test_zero_pair_normalizer_keeps_previous_error():
    task = angular_task(products=["delta_ddd"])
    with pytest.raises(ZeroDivisionError):
        task._compute_pair_stats_series(task.sfc_field1, task.sfc_field2, 0., 1., [4.])


def test_fallback_series_keeps_only_one_materialized_kernel(monkeypatch):
    import weakref

    from pyhermes.io import WindowFunc

    original = WindowFunc.as_array
    seen = []

    def tracked(window):
        result = original(window)
        seen.append(weakref.ref(result))
        assert sum(ref() is not None for ref in seen) <= 1
        return result

    monkeypatch.setattr(WindowFunc, "as_array", tracked)
    samples = [{"s": r, "mu": .3} for r in [0., 4., 8., 12.]]
    window = dict(type="ring", mapping="smu_to_RH", los_args={"nx": 1., "ny": 2., "nz": 3.})
    compute_pair_products_at_samples(samples, angular_field(), angular_field(2), window)
    assert len(seen) == len(samples) and not any(ref() is not None for ref in seen)
