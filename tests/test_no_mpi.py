import numpy as np
import pytest

import pyhermes
from pyhermes.utils.mpi_util import MPI


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
