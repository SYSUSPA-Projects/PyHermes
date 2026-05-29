"""
Example: run 2PCF with a custom cylinder-surface plus disk-cap window.

This script uses ``examples/configs/param_2pcf_smu_disk.yaml`` as a base
configuration and replaces only the pair window. Run it from ``examples``:

    python scripts/run_2pcf_cylsurf.py
"""


from numba import njit
from pyhermes.param.parambase import read_param
from pyhermes.theory.corr2pcf import Corr_2PCF
from pyhermes.utils.mpi_util import MPI
from pyhermes.utils.window_functions import (
    window_function_cylshell_numba,
    window_function_disk_numba,
)


@njit
def window_function_cylsurf_numba(ki, kj, kk, R, H, nx=0.0, ny=0.0, nz=1.0):
    denom = 2.0 * H + R
    if denom == 0.0:
        return 1.0

    win_cylshell = window_function_cylshell_numba(ki, kj, kk, R, H, nx, ny, nz)
    win_disk = window_function_disk_numba(ki, kj, kk, R, H, nx, ny, nz)
    return (2.0 * H * win_cylshell + R * win_disk) / denom


CYLSURF_PAIR_WINDOW = {
    "type": "cylsurf",
    "func": window_function_cylsurf_numba,
    "len_args": ["R", "H"],
    "los_args": {"nx": 0.0, "ny": 0.0, "nz": 1.0},
    "other_args": {},
    "mapping": "smu_to_RH",
    "kernel_mode": "octant",
}

params = read_param(config_path="./configs/param_2pcf_smu_disk.yaml")
if MPI.COMM_WORLD.Get_rank() == 0:
    params["Corr_2PCF"]["pair_window"] = CYLSURF_PAIR_WINDOW
    params["Corr_2PCF"]["fout_path"] = "./output/quijote8000_snap004_rsd_2pcf_smu_cylsurf.pkl"

task = Corr_2PCF(param_task=params)
task.run(overwrite=True)
