"""Independent readout, background subtraction and full-task particle checks."""

from types import SimpleNamespace

import numba
import numpy as np
import pytest

from pyhermes.io import SFCField
from pyhermes.theory import Corr_3PCF_Multipole
from pyhermes.utils import corr3pcf_multipoles as multipoles
from pyhermes.utils.corr3pcf_particle import (
    bilinear_product_sum,
    compute_particle_summand,
    prepare_particle_catalog,
)
from pyhermes.utils.wavelet_grid import (
    project_scaling_grid_numba,
    sample_scaling_function,
)


@pytest.fixture(autouse=True)
def _small_thread_budget():
    previous = numba.get_num_threads()
    numba.set_num_threads(min(previous, 2))
    yield
    numba.set_num_threads(previous)


def _direct_readout(values, positions, phi, resolution):
    """Explicit periodized basis matrix, independent of production stencils."""
    length = values.shape[0]
    support = len(phi) // resolution
    sampled = np.floor(np.remainder(positions, length) * resolution).astype(int)
    basis = np.zeros((len(positions), 3, length))
    for axis in range(3):
        for cell in range(length):
            for image in range(-support, support + 1):
                index = sampled[:, axis] - (cell + image * length) * resolution
                valid = (index >= 0) & (index < len(phi))
                basis[valid, axis, cell] += phi[index[valid]]
    return np.einsum("pi,pj,pk,ijk->p", basis[:, 0], basis[:, 1], basis[:, 2], values)


def _catalog(positions, weights, phi, resolution, length):
    return SimpleNamespace(
        field_kind="catalog_field", phi_array=phi, phi_resolution=resolution,
        phi_support=len(phi) // resolution, L=length, scale_factor=1.0,
        get_particle_data=lambda: {"pos": positions, "projection_weight": weights},
    )


@pytest.mark.parametrize("wavelet", ["db2", "db3"])
def test_complex_particle_readout_matches_periodized_basis(wavelet):
    phi = sample_scaling_function(wavelet, 3)
    rng = np.random.default_rng(1264)
    positions = np.vstack([rng.uniform(-8, 12, (13, 3)), [0, 4, -4], [3.999, 0.001, 8]])
    weights = rng.normal(size=len(positions))
    arms = rng.normal(size=(2, 4, 4, 4)) + 0.3j * rng.normal(size=(2, 4, 4, 4))
    component = prepare_particle_catalog(_catalog(positions, weights, phi, 8, 4))
    spec = {"grid_size": 4, "components": [(component, 1.0)], "uniform_density": 0.0}
    expected = np.sum(weights * _direct_readout(arms[0], positions, phi, 8)
                      * _direct_readout(arms[1], positions, phi, 8))
    np.testing.assert_allclose(compute_particle_summand(*arms, spec), expected, rtol=2e-13, atol=2e-13)


def test_signed_random_and_uniform_subtraction_match_independent_integral():
    phi = sample_scaling_function("db2", 3)
    rng = np.random.default_rng(83)
    arms = rng.normal(size=(2, 4, 4, 4)) + 0.4j * rng.normal(size=(2, 4, 4, 4))
    components = []
    expected = 0.0j
    for n, factor in [(7, 1.0), (11, -0.8)]:
        positions = rng.uniform(0, 4, (n, 3))
        weights = rng.uniform(0.2, 1.0, n)
        components.append((prepare_particle_catalog(_catalog(positions, weights, phi, 8, 4)), factor))
        expected += factor * np.sum(weights * _direct_readout(arms[0], positions, phi, 8)
                                    * _direct_readout(arms[1], positions, phi, 8))
    # Physical-space quadrature of the two reconstructed arms checks the
    # orthonormal shortcut, including its non-conjugated complex convention.
    grid = np.stack(np.meshgrid(*[np.arange(32) / 8] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    uniform = np.mean(_direct_readout(arms[0], grid, phi, 8)
                      * _direct_readout(arms[1], grid, phi, 8)) * 4**3
    np.testing.assert_allclose(bilinear_product_sum(*arms), uniform, rtol=2e-13, atol=2e-13)
    spec = {"grid_size": 4, "components": components, "uniform_density": -0.3}
    np.testing.assert_allclose(compute_particle_summand(*arms, spec), expected - 0.3 * uniform,
                               rtol=2e-13, atol=2e-13)


def test_particle_and_gamma_are_different_finite_grid_discretizations():
    phi = sample_scaling_function("db2", 3)
    positions = np.array([[0.25, 1.375, 2.75], [3.625, 0.5, 1.125]])
    weights = np.array([0.3, 0.7])
    rng = np.random.default_rng(73)
    arms = rng.normal(size=(2, 4, 4, 4)).astype(complex)
    center = project_scaling_grid_numba(positions, weights, phi, 8, 2, 4.0)
    gamma = multipoles.cal_gamma(phi, 3, 8)
    grid_value = multipoles.compute_3d_summand_cpu(center, *arms, gamma, 4, 3)
    component = prepare_particle_catalog(_catalog(positions, weights, phi, 8, 4))
    spec = {"grid_size": 4, "components": [(component, 1.0)], "uniform_density": 0.0}
    particle_value = compute_particle_summand(*arms, spec)
    assert abs(particle_value - grid_value) > 1e-3


def make_field(tmp_path, seed=120, J=3):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0, 16, (12, 3))
    catalog_weights = rng.uniform(0.5, 1.5, 12)
    marks = rng.uniform(0.6, 1.8, 12)
    path = tmp_path / f"particles_{seed}.npz"
    np.savez(path, pos=positions, catalog_weight=catalog_weights, field_value=marks)
    length = 2**J
    phi = sample_scaling_function("db2", 3)
    field = SFCField(threads=2)
    field.sfc_info = {
        "J": J, "L": length, "V": length**3, "box_size": 16.0, "scale_factor": length / 16,
        "wavelet_mode": "db2", "wavelet_level": 3, "phi_resolution": 8,
        "phi_array": phi, "phi_support": 3, "field_kind": "catalog_field",
        "particle_data_path": str(path), "particle_data_retrievable": True, "particle_count": 12,
        "catalog_weight_sum": float(catalog_weights.sum()),
        "catalog_weight_sq_sum": float(np.sum(catalog_weights**2)),
        "raw_field_weighted_sum": float(np.sum(catalog_weights * marks)),
        "field_integral": float(np.sum(catalog_weights * marks)), "weight_normalization": "raw",
    }
    field.format_sfc_params()
    field.epsilon = project_scaling_grid_numba(positions, catalog_weights * marks, phi, 8, J, 16.0)
    field.sfc_info["sfc_field_path"] = str(tmp_path / f"field_{seed}.pkl")
    field.save_sfc_field(field.sfc_info["sfc_field_path"])
    return field


def task_params(field, **kwargs):
    params = {
        "sfc_field": field.sfc_info["sfc_field_path"], "random": "uniform", "sampling": {"r12": 2.0, "r13": 4.0},
        "binning_window12": {"type": "shell", "mapping": {"R": "r12"}},
        "binning_window13": {"type": "shell", "mapping": {"R": "r13"}},
        "summation_backend": "cpu", "summation_method": "particle", "threads": 2,
        "l_max": 2, "radial_profile_diagnostics": False,
    }
    params.update(kwargs)
    if isinstance(params["random"], SFCField):
        params["random"] = params["random"].sfc_info["sfc_field_path"]
    return {"Corr_3PCF_Multipole": params}


@pytest.mark.parametrize("normalization", ["raw", "catalog", "field", "unit"])
def test_full_task_preserves_projection_weights_background_and_physical_units(tmp_path, normalization):
    field = make_field(tmp_path)
    task = Corr_3PCF_Multipole(task_params(field, weight_normalization=normalization))
    task.prepare_input_fields()
    center = task.sfc_field1.get_particle_data()
    positions = center["pos"] * field.scale_factor
    weights = center["projection_weight"]
    first, second = task._prepare_product_fields("delta_ddd_l")[1:]
    expected_raw = []
    for ell in range(3):
        values = []
        for m in range(ell + 1):
            a = multipoles._stream_convolution_fields(first, task.binning_window12[0], ell, 2, m_values=[m])[0]
            b = multipoles._stream_convolution_fields(second, task.binning_window13[0], ell, 2, m_values=[-m])[0]
            direct = np.sum(weights * _direct_readout(a, positions, field.phi_array, 8)
                            * _direct_readout(b, positions, field.phi_array, 8))
            direct -= task.random1 * np.sum(a * b)
            values.append(4 * np.pi * direct / field.V)
        combined = (-1)**ell * (values[0].real + 2 * sum((-1)**m * values[m].real for m in range(1, ell + 1)))
        expected_raw.append(combined)
    result = task.run(save_result=False)
    expected_raw = np.asarray(expected_raw)
    physical_scale = field.scale_factor**9
    np.testing.assert_allclose(result.delta_ddd_l[0], expected_raw * physical_scale, rtol=3e-12, atol=1e-14)
    expected_zeta = expected_raw / (task.random1 * task.random2 * task.random3)
    np.testing.assert_allclose(result.zeta_l[0], expected_zeta, rtol=3e-12, atol=1e-12)
    assert result.task_params["summation_method"] == "particle"


def test_full_task_catalog_random_and_sample_mode(tmp_path):
    field = make_field(tmp_path)
    random = make_field(tmp_path, seed=321)
    serial = Corr_3PCF_Multipole(task_params(field, random=random)).run(save_result=False)
    sampled = Corr_3PCF_Multipole(task_params(field, random=random, execution_mode="sample_mpi")).run(save_result=False)
    np.testing.assert_allclose(serial.zeta_l, sampled.zeta_l, rtol=1e-12, atol=1e-12)


def test_particle_mode_rejects_centre_window_and_cuda(tmp_path):
    field = make_field(tmp_path)
    with pytest.raises(ValueError, match="summation_backend='cpu'"):
        Corr_3PCF_Multipole(task_params(field, summation_backend="gpu"))
    task = Corr_3PCF_Multipole(task_params(field, window1={"type": "sphere", "len_args": {"R": 1.0}}))
    with pytest.raises(ValueError, match="unfiltered vertex 1"):
        task.prepare_input_fields()
    with pytest.raises(ValueError, match="retrievable catalog"):
        prepare_particle_catalog(field - 1.0)


def test_empty_centres_and_shape_validation():
    phi = sample_scaling_function("db2", 3)
    component = prepare_particle_catalog(_catalog(np.empty((0, 3)), np.empty(0), phi, 8, 4))
    spec = {"grid_size": 4, "components": [(component, 1.0)], "uniform_density": 0.0}
    arms = np.ones((2, 4, 4, 4), complex)
    assert compute_particle_summand(*arms, spec) == 0.0
    with pytest.raises(ValueError, match="matching cubic"):
        compute_particle_summand(arms[0, :2], arms[1], spec)
