import numpy as np
import pytest
from numba import njit

import pyhermes
from pyhermes.utils.mpi_util import MPI


@njit
def _custom_thick_shell_window(ki, kj, kk, R_in, R_out):
    k_squared = ki**2 + kj**2 + kk**2
    return np.exp(-k_squared * R_in * R_out)


def test_package_version_is_exposed():
    assert pyhermes.__version__


def test_public_theory_package_imports_without_mpi():
    import pyhermes.theory  # noqa: F401


def test_fake_mpi_single_process_collectives():
    comm = MPI.COMM_WORLD

    assert comm.Get_rank() == 0
    assert comm.Get_size() == 1
    assert comm.bcast({"value": 1}, root=0) == {"value": 1}
    assert comm.reduce(3, op=MPI.SUM, root=0) == 3
    assert comm.allreduce(3, op=MPI.MAX) == 3
    assert comm.allgather(3) == [3]
    assert comm.Split_type(MPI.COMM_TYPE_SHARED) is comm
    assert comm.Split(0) is comm
    assert comm.Split(MPI.UNDEFINED) is MPI.COMM_NULL


def test_fake_mpi_buffer_collectives():
    comm = MPI.COMM_WORLD

    send = np.arange(4, dtype=np.float64)
    reduced = np.empty_like(send)
    scattered = np.empty_like(send)

    comm.Allreduce(send, reduced, op=MPI.SUM)
    comm.Scatterv([send, [4], [0], MPI.DOUBLE], scattered, root=0)

    np.testing.assert_array_equal(reduced, send)
    np.testing.assert_array_equal(scattered, send)


def test_package_data_is_installed():
    from pathlib import Path
    import pyhermes.base
    import pyhermes.theory
    import pyhermes.utils

    assert (Path(pyhermes.base.__file__).parent / "default_params.json").is_file()
    assert (Path(pyhermes.theory.__file__).parent / "default_params.json").is_file()
    assert (Path(pyhermes.utils.__file__).parent / "plot_styles.json").is_file()


def test_safe_exit_uses_unified_fake_mpi():
    from pyhermes.utils.func_util import safe_exit

    with pytest.raises(SystemExit) as exc_info:
        safe_exit(7)

    assert exc_info.value.code == 7


def test_custom_window_function_overrides_matching_builtin_type():
    from pyhermes.io import WindowFunc
    from pyhermes.utils.window_params import normalize_binning_window_template

    custom_params = {
        "type": "thick_shell",
        "func": _custom_thick_shell_window,
        "len_args": {"R_in": 8.0, "R_out": 12.0},
    }
    normalized = normalize_binning_window_template(custom_params)

    assert normalized["len_args"] == {"R_in": 8.0, "R_out": 12.0}
    assert normalized["los_args"] == {}
    assert normalized["kernel_mode"] == "full_rfft"

    custom_params["kernel_mode"] = "octant"
    window = WindowFunc(
        custom_params,
        {
            "J": 3,
            "box_size": 64.0,
            "phi_resolution": 1024,
            "wavelet_mode": "db2",
            "wavelet_level": 10,
        },
    )

    assert window.type == "thick_shell"
    assert window.has_custom_func
    assert window.func is _custom_thick_shell_window
    assert window.len_args == {"R_in": 8.0, "R_out": 12.0}
    assert np.all(np.isfinite(window.as_array()))


def test_builtin_thick_shell_keeps_builtin_argument_validation():
    from pyhermes.io import WindowFunc

    sfc_params = {
        "J": 3,
        "box_size": 64.0,
        "phi_resolution": 1024,
        "wavelet_mode": "db2",
        "wavelet_level": 10,
    }
    window = WindowFunc(
        {
            "type": "thick_shell",
            "len_args": {"R": 10.0, "delta_R": 4.0},
        },
        sfc_params,
    )

    assert not window.has_custom_func
    assert np.all(np.isfinite(window.as_array()))
    with pytest.raises(ValueError, match="requires finite len_args"):
        WindowFunc(
            {
                "type": "thick_shell",
                "len_args": {"R_in": 8.0, "R_out": 12.0},
            },
            sfc_params,
        )
