"""Run the s-mu 2PCF with the built-in Gaussian-ring binning window."""

import sys

from pyhermes.param.parambase import read_param
from pyhermes.theory.corr2pcf import Corr_2PCF
from pyhermes.utils.mpi_util import MPI

config_path = sys.argv[1] if len(sys.argv) > 1 else "./configs/param_2pcf_smu.yaml"
sigma_perp = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
sigma_label = f"{sigma_perp:g}".replace(".", "p")

gaussian_ring_window = {
    "type": "gaussian_ring",
    "len_args": {"R": None, "H": None, "sigma_perp": sigma_perp},
    "mapping": "smu_to_RH",
    "kernel_mode": "octant",
}

params = read_param(config_path=config_path)
if MPI.COMM_WORLD.Get_rank() == 0:
    params["Corr_2PCF"]["binning_window"] = gaussian_ring_window
    params["Corr_2PCF"]["fout_path"] = (
        f"./output/quijote8000_snap004_rsd_2pcf_smu_gaussian_ring_sigma{sigma_label}_optimized_20261017.pkl"
    )

task = Corr_2PCF(param_task=params)
task.run(overwrite=True)
