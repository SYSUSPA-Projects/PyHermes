"""Run with mpiexec -n 2 python -m pytest tests/test_corr2pcf_mpi.py."""

import numpy as np
import pytest

from pyhermes.io import SFCField
from pyhermes.theory.corr2pcf import Corr_2PCF
from pyhermes.utils.mpi_util import MPI

COMM = MPI.COMM_WORLD
pytestmark = pytest.mark.skipif(
    COMM.Get_size() < 2, reason="requires a real multi-rank MPI run"
)


def field(seed):
    if COMM.Get_rank() != 0:
        return None
    result = SFCField(threads=1)
    result.sfc_info = dict(
        J=3,
        L=8,
        V=512,
        box_size=64.0,
        scale_factor=0.125,
        phi_resolution=64,
        wavelet_mode="db2",
        wavelet_level=6,
        field_kind="catalog_field",
        weight_normalization="catalog",
        field_integral=1.0,
        raw_field_weighted_sum=100.0,
        catalog_weight_sum=100.0,
        catalog_weight_sq_sum=100.0,
        particle_count=100,
    )
    values = np.random.default_rng(seed).uniform(0.1, 1.0, (8, 8, 8))
    result.epsilon = values / values.sum()
    result.format_sfc_params()
    return result


@pytest.mark.parametrize("nonuniform", [False, True])
def test_multi_rank_products_and_global_fft_counts(nonuniform):
    first = field(701)
    second = field(702) if nonuniform else first
    random1, random2 = (
        (field(703), field(704)) if nonuniform else ("uniform", "uniform")
    )
    results = []
    for method, reuse, window_mode, fft_mode in [
        ("convolution", False, "materialized", "per_rank"),
        ("convolution", True, "materialized", "per_rank"),
        ("spectral", False, "materialized", "per_rank"),
        ("spectral", True, "materialized", "per_rank"),
        ("spectral", True, "fused", "per_rank"),
        ("spectral", True, "shell_edges", "per_rank"),
        ("spectral", True, "shell_edges_batch", "per_rank"),
        ("spectral", True, "shell_edges_batch", "shared"),
        ("spectral", True, "shell_edges_radial", "per_rank"),
        ("spectral", True, "shell_edges_radial", "shared"),
    ]:
        params = dict(
            threads=1,
            random="uniform",
            sampling={"s": [4.0, 8.0, 12.0]},
            binning_window=dict(
                type="thick_shell",
                len_args={"R": None, "delta_R": 4.0},
                mapping="s_to_R",
                kernel_mode="octant",
            ),
            pair_reduction=method,
            reuse_input_fields=reuse,
            spectral_window=window_mode,
            spectral_fft=fft_mode,
            products=["dd", "dr", "rd", "delta_dd", "rr", "xi"],
        )
        current = Corr_2PCF({"Corr_2PCF": params})
        current.prepare_input_fields(
            sfc_field1=first, sfc_field2=second, random1=random1, random2=random2
        )
        output = current.run(save_result=False)
        error = None
        if COMM.Get_rank() == 0:
            try:
                values = {
                    name: getattr(output, name).copy() for name in params["products"]
                }
                if results:
                    for name, value in values.items():
                        np.testing.assert_allclose(
                            value, results[0][name], rtol=1e-8, atol=1e-11
                        )
                results.append(values)
                metrics = current.runtime_metrics["per_rank"]
                if method == "spectral":
                    assert sum(row["legacy_inverse_fft_count"] for row in metrics) == 0
                    if not nonuniform and reuse:
                        # dd and delta_dd are two different fields, each reused across bins.
                        assert sum(row["forward_fft_count"] for row in metrics) == 2 * (
                            1 if fft_mode == "shared" else COMM.Get_size()
                        )
            except (AssertionError, ValueError) as exc:
                error = repr(exc)
        error = COMM.bcast(error, root=0)
        assert error is None, error


@pytest.mark.parametrize(
    "path", ["fused", "shell_edges", "shell_edges_batch", "shell_edges_radial"]
)
def test_window_reduction_more_ranks_than_bins(path):
    from test_corr2pcf_window_reduction import explicit_edges

    first, second = field(750), field(751)
    saved = None
    for mode in ["materialized", path]:
        params = dict(
            threads=1,
            random="uniform",
            sampling={"s": [0.0]},
            binning_window=dict(
                type="binned_shell",
                len_args=["R_in", "R_out"],
                mapping=explicit_edges,
                kernel_mode="octant",
            ),
            pair_reduction="spectral",
            spectral_window=mode,
            reuse_input_fields=True,
            products=["dd", "delta_dd", "rr", "xi"],
        )
        current = Corr_2PCF({"Corr_2PCF": params})
        current.prepare_input_fields(sfc_field1=first, sfc_field2=second)
        output = current.run(save_result=False)
        error = None
        if COMM.Get_rank() == 0:
            try:
                if saved is None:
                    saved = output.xi.copy()
                else:
                    np.testing.assert_allclose(saved, output.xi, rtol=1e-8, atol=1e-11)
                    rows = current.runtime_metrics["per_rank"]
                    if path.startswith("shell_edges"):
                        assert (
                            sum(row["shell_edge_count"] for row in rows) == 4
                        )  # two edges x two powers
                        assert sum(row["shell_result_count"] for row in rows) == 2
            except (AssertionError, ValueError) as exc:
                error = repr(exc)
        assert COMM.bcast(error, root=0) is None


@pytest.mark.parametrize("fft_mode", ["per_rank", "shared"])
def test_empty_edge_rank_and_batch_sweep_counts(fft_mode):
    from test_corr2pcf_window_reduction import explicit_edges

    first = field(761)
    current = Corr_2PCF(
        {
            "Corr_2PCF": dict(
                threads=1,
                random="uniform",
                sampling={"s": [0.0]},
                binning_window=dict(
                    type="binned_shell",
                    len_args=["R_in", "R_out"],
                    mapping=explicit_edges,
                    kernel_mode="octant",
                ),
                pair_reduction="spectral",
                reuse_input_fields=True,
                spectral_window="shell_edges_batch",
                spectral_fft=fft_mode,
                products=["xi"],
            )
        }
    )
    current.prepare_input_fields(sfc_field1=first, sfc_field2=first)
    current.run(save_result=False)
    error = None
    if COMM.Get_rank() == 0:
        try:
            rows = current.runtime_metrics["per_rank"]
            assert sum(row["shell_edge_count"] for row in rows) == 2
            assert sum(row["shell_edge_sweeps"] for row in rows) == min(
                2, COMM.Get_size()
            )
            assert sum(row["forward_fft_count"] for row in rows) == (
                1 if fft_mode == "shared" else COMM.Get_size()
            )
            for row in rows:
                if row["shell_edge_count"] == 0:
                    assert row["shell_edge_sweeps"] == 0
        except (AssertionError, ValueError) as exc:
            error = repr(exc)
    assert COMM.bcast(error, root=0) is None


@pytest.mark.parametrize(
    "kind", ["shell", "gaussian_shell", "thick_shell", "ring", "inverse_laplacian"]
)
@pytest.mark.parametrize("count", [1, 6])
def test_automatic_window_strategy_real_mpi(kind, count):
    from test_corr2pcf_strategies import fixed_window

    first, second = field(865), field(866)
    random1, random2 = field(867), field(868)
    specifications = {
        "shell": dict(type="shell", len_args={"R": None}, mapping="s_to_R"),
        "gaussian_shell": dict(
            type="gaussian_shell",
            len_args={"R_shell": 5.0, "R_smooth": 2.0},
            mapping=fixed_window,
        ),
        "thick_shell": dict(
            type="thick_shell", len_args={"R": None, "delta_R": 4.0}, mapping="s_to_R"
        ),
        "ring": dict(type="ring", len_args={"R": None, "H": None}, mapping="smu_to_RH"),
        "inverse_laplacian": dict(
            type="inverse_laplacian", len_args={}, mapping=fixed_window
        ),
    }
    saved = None
    for mode in ["materialized", "auto"]:
        current = Corr_2PCF(
            {
                "Corr_2PCF": dict(
                    threads=1,
                    random="uniform",
                    sampling=(
                        {"s": list(np.linspace(4.0, 24.0, count)), "mu": [0.5]}
                        if kind == "ring"
                        else {"s": list(np.linspace(4.0, 24.0, count))}
                    ),
                    binning_window=specifications[kind],
                    pair_reduction="spectral",
                    reuse_input_fields=True,
                    spectral_window=mode,
                    spectral_fft="shared",
                    products=["dd", "dr", "rd", "delta_dd", "rr", "xi"],
                )
            }
        )
        current.prepare_input_fields(
            sfc_field1=first, sfc_field2=second, random1=random1, random2=random2
        )
        output = current.run(save_result=False)
        error = None
        if COMM.Get_rank() == 0:
            try:
                values = {
                    name: getattr(output, name).copy() for name in current.products
                }
                if saved is None:
                    saved = values
                else:
                    for name in values:
                        np.testing.assert_allclose(
                            values[name], saved[name], rtol=1e-8, atol=1e-11
                        )
                    rows = current.runtime_metrics["per_rank"]
                    assert sum(row["forward_fft_count"] for row in rows) == 6
                    assert sum(row["legacy_inverse_fft_count"] for row in rows) == 0
                    if (
                        kind in ("shell", "gaussian_shell", "inverse_laplacian")
                        and count >= COMM.Get_size()
                    ):
                        assert all(row["auto_plan"] == "radial" for row in rows)
            except (AssertionError, ValueError) as exc:
                error = repr(exc)
        assert COMM.bcast(error, root=0) is None, error
