"""Independent materialized folds and end-to-end checks for automatic strategies."""

import numpy as np
import pytest
from numba import njit
from test_corr2pcf_optimization import field, task
from test_corr2pcf_optimization import thread_budget as thread_budget

from pyhermes.utils import corr2pcf_strategies as strategies
from pyhermes.utils.convolution import (
    calculate_real_window_octant_array_numba,
    fold_octant_window_to_rfft_kernel,
)
from pyhermes.utils.corr2pcf_spectral import _half_spectrum_inner_product
from pyhermes.utils.corr2pcf_window_reduction import octant_radial_power
from pyhermes.utils.window_functions import WINDOW_TYPE_DICT
from pyhermes.utils.window_structure import BUILTIN_STRUCTURES

RADIAL_CASES = [
    ("sphere", (1.3,)),
    ("shell", (1.3,)),
    ("gaussian", (0.7,)),
    ("gaussian_shell", (1.3, 0.7)),
    ("thick_shell", (1.3, 0.4)),
    ("binned_shell", (0.8, 1.7)),
    ("cw", (0.7,)),
    ("cws", (0.7,)),
    ("gdw", (0.7,)),
    ("laplacian", ()),
    ("inverse_laplacian", ()),
]


@pytest.mark.parametrize("bandwidth", [1, 2])
@pytest.mark.parametrize("kind,args", RADIAL_CASES)
def test_radial_grouping_matches_independent_materialized_fold(kind, args, bandwidth):
    L = 8
    rng = np.random.default_rng(812)
    power = rng.normal(size=(L, L, L // 2 + 1))
    phi = rng.uniform(0.1, 2.0, bandwidth * L + 1)
    function = WINDOW_TYPE_DICT[kind]
    kernel = fold_octant_window_to_rfft_kernel(
        calculate_real_window_octant_array_numba(L, bandwidth, phi, function, *args)
    )
    zero_dc = kind == "inverse_laplacian"
    if zero_dc:
        kernel[0, 0, 0] = 0.0
    expected = _half_spectrum_inner_product(power, kernel, L)
    histogram = sum(
        octant_radial_power(power, bandwidth, phi, rank, 12, zero_dc)
        for rank in range(12)
    )
    norms = np.flatnonzero(histogram)
    actual = strategies.radial_function_integral(
        norms, histogram[norms], L, function, args
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=2e-13)


@pytest.mark.parametrize("kind", ["ring", "disk", "cylshell", "cylinder"])
@pytest.mark.parametrize("axis,sign", [(0, 1), (1, 1), (2, 1), (2, -1)])
@pytest.mark.parametrize("bandwidth", [1, 2])
def test_separable_factors_match_materialized_alias_fold(kind, axis, sign, bandwidth):
    L = 8
    rng = np.random.default_rng(140)
    power = rng.normal(size=(L, L, L // 2 + 1))
    phi = rng.uniform(0.1, 2.0, bandwidth * L + 1)
    los = [0.0, 0.0, 0.0]
    los[axis] = sign * 2.0
    structure = BUILTIN_STRUCTURES[kind]
    transverse = strategies.transverse_table(L, bandwidth, 1.7, structure.transverse)
    longitudinal = strategies.longitudinal_table(
        L, bandwidth, 0.9, structure.longitudinal
    )
    kernel = fold_octant_window_to_rfft_kernel(
        calculate_real_window_octant_array_numba(
            L, bandwidth, phi, WINDOW_TYPE_DICT[kind], 1.7, 0.9, *los
        )
    )
    expected = _half_spectrum_inner_product(power, kernel, L)
    actual = strategies.octant_factor_integral(
        power, bandwidth, phi, transverse, longitudinal, axis
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=2e-13)


def fixed_window(sample, template):
    return template


def compare_pipeline(window, sampling=None, nonuniform=False):
    a, b = field(), field(2)
    random1, random2 = (field(3), field(4)) if nonuniform else ("uniform", "uniform")
    products = ["dd", "dr", "rd", "delta_dd", "rr", "xi"]
    outputs = []
    for mode in ["materialized", "auto"]:
        current = task(
            "spectral",
            True,
            spectral_window=mode,
            binning_window=window,
            sampling=sampling or {"s": [4.0, 8.0, 12.0, 16.0]},
            products=products,
        )
        current.prepare_input_fields(
            sfc_field1=a, sfc_field2=b, random1=random1, random2=random2
        )
        outputs.append(current.run(save_result=False))
    for name in products:
        np.testing.assert_allclose(
            getattr(outputs[0], name), getattr(outputs[1], name), rtol=1e-8, atol=1e-11
        )
    assert current._window_evaluator is None
    return current.runtime_metrics["per_rank"][0]


@pytest.mark.parametrize("kind,args", RADIAL_CASES)
@pytest.mark.parametrize("nonuniform", [False, True])
def test_radial_planner_all_products(kind, args, nonuniform):
    import inspect

    names = list(inspect.signature(WINDOW_TYPE_DICT[kind]).parameters)[3:]
    window = dict(
        type=kind,
        len_args=dict(zip(names, args)),
        kernel_mode="octant",
        mapping=fixed_window,
    )
    # Fixed radii still exercise duplicate-query elimination and collective planning.
    metrics = compare_pipeline(window, nonuniform=nonuniform)
    assert metrics["auto_plan"] in ("radial", "radial_shell_edges")
    assert sum(metrics["window_strategy_counts"].values()) > 0


@pytest.mark.parametrize("kind", ["ring", "disk", "cylshell", "cylinder"])
def test_axial_pipeline_all_products_and_tilt_fallback(kind):
    window = dict(
        type=kind,
        len_args={"R": None, "H": None},
        mapping="smu_to_RH",
        kernel_mode="auto",
    )
    sampling = {"s": [4.0, 8.0, 12.0], "mu": [0.0, 0.5, 1.0]}
    metrics = compare_pipeline(window, sampling, True)
    assert metrics["window_strategy_counts"]["separable_axial"] > 0
    window = dict(window, los_args={"nx": 1.0, "ny": 2.0, "nz": 3.0})
    metrics = compare_pipeline(window, sampling, True)
    assert metrics["window_strategy_counts"] == {"materialized": 45}


@njit
def custom_anisotropic(ki, kj, kk, R):
    return 1 + R * (ki * ki + 2 * kj * kj + 3 * kk * kk)


@njit
def custom_radial(ki, kj, kk, R):
    return np.exp(-R * (ki * ki + kj * kj + kk * kk))


def test_custom_authoritative_and_explicit_radial_contract():
    window = dict(
        type="shell",
        func=custom_anisotropic,
        len_args={"R": None},
        mapping="s_to_R",
        kernel_mode="octant",
    )
    assert compare_pipeline(window)["window_strategy_counts"] == {"fused": 8}
    window = dict(window, func=custom_radial, structure="radial")
    assert compare_pipeline(window)["auto_plan"] == "radial"


def test_memory_guards_and_factor_cache_lifecycle(monkeypatch):
    monkeypatch.setattr(strategies.legacy, "RADIAL_SCRATCH_LIMIT_BYTES", 1)
    metrics = compare_pipeline(
        dict(type="shell", len_args={"R": None}, mapping="s_to_R")
    )
    assert metrics["auto_plan"] == "fused_radial_guard"
    monkeypatch.setattr(strategies, "FACTOR_CACHE_BYTES", 2048)
    metrics = compare_pipeline(
        dict(type="ring", len_args={"R": None, "H": None}, mapping="smu_to_RH"),
        {"s": [4.0, 8.0, 12.0], "mu": [0.0, 0.5, 1.0]},
    )
    assert 0 < metrics["factor_cache_peak_bytes"] <= 2048
    monkeypatch.setattr(strategies, "FACTOR_SCRATCH_LIMIT_BYTES", 1)
    metrics = compare_pipeline(
        dict(type="ring", len_args={"R": None, "H": None}, mapping="smu_to_RH"),
        {"s": [4.0, 8.0], "mu": [0.0, 0.5]},
    )
    assert set(metrics["window_strategy_counts"]) == {"fused"}
