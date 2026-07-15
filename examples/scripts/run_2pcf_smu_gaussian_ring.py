"""Run the s-mu 2PCF with a custom Gaussian-ring binning window."""

import sys

import numpy as np
from numba import njit

from pyhermes.param.parambase import read_param
from pyhermes.theory.corr2pcf import Corr_2PCF
from pyhermes.utils.mpi_util import MPI
from pyhermes.utils.special_functions import jn_numba


@njit
def window_function_gaussian_ring_numba(ki, kj, kk, R, H, sigma_perp):
    k_perp = np.sqrt(ki * ki + kj * kj)
    q_ring = 2.0 * np.pi * k_perp * R
    q_smooth = 2.0 * np.pi * k_perp * sigma_perp
    q_parallel = 2.0 * np.pi * kk * H
    return (
        jn_numba(0, q_ring)
        * np.exp(-0.5 * q_smooth * q_smooth)
        * np.cos(q_parallel)
    )


config_path = sys.argv[1] if len(sys.argv) > 1 else "./configs/param_2pcf_smu.yaml"
sigma_perp = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
sigma_label = f"{sigma_perp:g}".replace(".", "p")

gaussian_ring_window = {
    "type": "gaussian_ring",
    "func": window_function_gaussian_ring_numba,
    "len_args": {"R": None, "H": None, "sigma_perp": sigma_perp},
    "mapping": "smu_to_RH",
    "kernel_mode": "octant",
}

params = read_param(config_path=config_path)
if MPI.COMM_WORLD.Get_rank() == 0:
    params["Corr_2PCF"]["binning_window"] = gaussian_ring_window
    params["Corr_2PCF"]["fout_path"] = (
        f"./output/quijote8000_snap004_rsd_2pcf_smu_gaussian_ring_sigma{sigma_label}.pkl"
    )

task = Corr_2PCF(param_task=params)
task.run(overwrite=True)
