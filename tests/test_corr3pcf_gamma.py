"""Compare the production contraction with independent physical-space integrals."""

import itertools
from types import SimpleNamespace

import numba
import numpy as np
import pytest
from numba import cuda

from pyhermes.utils import corr3pcf_multipoles as multipoles
from pyhermes.utils.wavelet_grid import sample_scaling_function


def _physical_fields(coefficients, phi, resolution):
    """Periodize sampled basis functions, without connection coefficients."""
    length = coefficients[0].shape[0]
    support = len(phi) // resolution
    positions = np.arange(length * resolution)
    basis = np.zeros((len(positions), length))
    for cell in range(length):
        for image in range(-support, support + 1):
            index = positions - (cell + image * length) * resolution
            valid = (index >= 0) & (index < len(phi))
            basis[valid, cell] += phi[index[valid]]
    return [
        np.einsum("xi,yj,zk,ijk->xyz", basis, basis, basis, values, optimize=True)
        for values in coefficients
    ]


@pytest.fixture(autouse=True)
def _small_thread_budget():
    previous = numba.get_num_threads()
    numba.set_num_threads(min(previous, 2))
    yield
    numba.set_num_threads(previous)


@pytest.mark.parametrize("wavelet", ["db2", "db3"])
def test_dense_complex_contraction_matches_direct_integral(wavelet):
    resolution = 8
    phi = sample_scaling_function(wavelet, 3)
    support = len(phi) // resolution
    length = 4  # db3 also exercises basis support longer than a periodic side.
    rng = np.random.default_rng(946)
    fields = [rng.normal(size=(length,) * 3)]
    fields += [rng.normal(size=(length,) * 3) + 0.2j * rng.normal(size=(length,) * 3) for _ in range(2)]
    direct = np.mean(np.prod(_physical_fields(fields, phi, resolution), axis=0))
    gamma = multipoles.cal_gamma(phi, support, resolution)
    value = multipoles.compute_3d_summand_cpu(*fields, gamma, length, support) / length**3
    np.testing.assert_allclose(value, direct, rtol=2e-12, atol=2e-13)


def test_real_triple_integral_is_symmetric_in_all_three_fields():
    phi = sample_scaling_function("db2", 3)
    gamma = multipoles.cal_gamma(phi, 3, 8)
    fields = np.random.default_rng(174).normal(size=(3, 4, 4, 4))
    direct = np.mean(np.prod(_physical_fields(fields, phi, 8), axis=0))
    for order in itertools.permutations(range(3)):
        center, first, second = [fields[i] for i in order]
        value = multipoles.compute_3d_summand_cpu(
            center, first.astype(complex), second.astype(complex), gamma, 4, 3
        ) / 4**3
        np.testing.assert_allclose(value, direct, rtol=2e-12, atol=2e-13)


def test_uniform_center_recovers_orthonormal_bilinear_integral():
    phi = sample_scaling_function("db2", 10)
    gamma = multipoles.cal_gamma(phi, 3, 1024)
    rng = np.random.default_rng(209)
    first, second = rng.normal(size=(2, 4, 4, 4)) + 1j * rng.normal(size=(2, 4, 4, 4))
    value = multipoles.compute_3d_summand_cpu(np.ones((4,) * 3), first, second, gamma, 4, 3) / 4**3
    np.testing.assert_allclose(value, np.mean(first * second), rtol=2e-12, atol=2e-13)


def test_overlap_table_rejects_truncated_basis_samples():
    with pytest.raises(ValueError, match="support\\*resolution"):
        multipoles.cal_gamma(np.ones(23), 3, 8)


@pytest.mark.skipif(not cuda.is_available(), reason="Requires a CUDA device or NUMBA_ENABLE_CUDASIM=1")
def test_cuda_contraction_matches_direct_complex_integral():
    phi = sample_scaling_function("db2", 3)
    gamma = multipoles.cal_gamma(phi, 3, 8)
    rng = np.random.default_rng(71)
    center = rng.normal(size=(4,) * 3)
    first, second = rng.normal(size=(2, 4, 4, 4)) + 0.1j * rng.normal(size=(2, 4, 4, 4))
    direct = np.mean(np.prod(_physical_fields([center, first, second], phi, 8), axis=0))
    result = cuda.device_array((4,) * 3, dtype=np.complex128)
    multipoles.compute_3d_result_gpu[(2, 2, 2), (2, 2, 2)](
        cuda.to_device(center), cuda.to_device(first), cuda.to_device(second),
        cuda.to_device(gamma), result, 4, 3,
    )
    np.testing.assert_allclose(np.mean(result.copy_to_host()), direct, rtol=2e-12, atol=2e-13)


@pytest.mark.skipif(not cuda.is_available(), reason="Requires a CUDA device or NUMBA_ENABLE_CUDASIM=1")
@pytest.mark.parametrize("wavelet", ["db2", "db3"])
def test_production_cuda_contraction_matches_integral_and_reuses_workspace(wavelet):
    phi = sample_scaling_function(wavelet, 3)
    support = len(phi) // 8
    rng = np.random.default_rng(743)
    center = rng.normal(size=(4,) * 3)
    arms = rng.normal(size=(2, 4, 4, 4)) + 0.2j * rng.normal(size=(2, 4, 4, 4))
    field = SimpleNamespace(epsilon=center, phi_array=phi, phi_support=support, phi_resolution=8, L=4)
    context = multipoles._prepare_multipole_gpu_context(field, gpu_threads_per_block=(2, 2, 2))
    assert context["gpu_algorithm"] == "exact_separable_signed"
    for first, second in [arms, [-arms[1], arms[0]]]:
        direct = np.mean(np.prod(_physical_fields([center, first, second], phi, 8), axis=0))
        value, _ = multipoles.compute_multipole_m_summand(first, second, context)
        np.testing.assert_allclose(value / (4 * np.pi), direct, rtol=2e-12, atol=2e-13)
