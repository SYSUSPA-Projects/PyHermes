"""
Run the s-mu 2PCF with a Gaussian-blurred ring binning window.

The custom binning window is assembled from two projected factors:

    J0(2*pi*k_perp*R) * exp[-(2*pi*k_perp*sigma_perp)^2/2]
    cos(2*pi*k_parallel*H)

The first factor is a Gaussian-blurred transverse ring; the second fixes the
line-of-sight displacement. The product uses WindowFunc.__mul__, matching the
current projected-operator composition semantics in PyHermes.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from numba import njit

from pyhermes.io import Corr2PCFData, WindowFunc
from pyhermes.param.parambase import read_param
from pyhermes.theory.corr2pcf import Corr_2PCF, build_binning_window_params_for_sample
from pyhermes.utils.mpi_util import MPI
from pyhermes.utils.special_functions import jn_numba


DEFAULT_CONFIG = "./configs/param_2pcf_smu.yaml"
DEFAULT_SIGMA_PERP = 5.0


def output_path_for_sigma(sigma_perp: float) -> str:
    sigma_label = f"{sigma_perp:g}".replace(".", "p")
    return f"./output/quijote8000_snap004_rsd_2pcf_smu_gaussian_ring_sigma{sigma_label}.pkl"


@njit
def window_function_gaussian_ring_perp_z_numba(ki, kj, kk, R, sigma_perp):
    k_perp = np.sqrt(ki * ki + kj * kj)
    q_ring = 2.0 * np.pi * k_perp * R
    q_smooth = 2.0 * np.pi * k_perp * sigma_perp
    return jn_numba(0, q_ring) * np.exp(-0.5 * q_smooth * q_smooth)


@njit
def window_function_ring_parallel_z_numba(ki, kj, kk, H):
    q_parallel = 2.0 * np.pi * kk * H
    return np.cos(q_parallel)


class Corr2PCFGaussianRing(Corr_2PCF):
    """Corr_2PCF variant using a Gaussian-blurred transverse ring window."""

    __module__ = Corr_2PCF.__module__

    def __init__(self, param_task=None, sigma_perp=DEFAULT_SIGMA_PERP):
        if param_task is None:
            param_task = {"Corr_2PCF": {}}
        self.task_name = "Corr_2PCF"
        self.sigma_perp = float(sigma_perp)
        super(Corr_2PCF, self).__init__(param_task=param_task)
        self.format_params()
        self._fields_prepared = False

    def _build_binning_window_for_sample(self, sample, reference_field):
        ring_params = build_binning_window_params_for_sample(sample, self.binning_window)
        radius = float(ring_params["len_args"]["R"])
        half_height = float(ring_params["len_args"]["H"])

        perp_window = WindowFunc(
            {
                "type": "gaussian_ring_perp_z",
                "func": window_function_gaussian_ring_perp_z_numba,
                "len_args": {"R": radius, "sigma_perp": self.sigma_perp},
                "kernel_mode": "octant",
            },
            reference_field.sfc_info,
            threads=self.threads,
        )
        parallel_window = WindowFunc(
            {
                "type": "ring_parallel_z",
                "func": window_function_ring_parallel_z_numba,
                "len_args": {"H": half_height},
                "kernel_mode": "octant",
            },
            reference_field.sfc_info,
            threads=self.threads,
        )
        return perp_window * parallel_window


def print_baseline_comparison(result_data, baseline_path, result_path) -> None:
    baseline = Path(baseline_path)
    if not baseline.exists() or result_data.xi is None:
        print(f"Baseline comparison skipped; missing baseline xi file: {baseline}")
        print(f"Gaussian-ring output: {result_path}")
        return

    baseline_data = Corr2PCFData(data_path=str(baseline), threads=1)
    if baseline_data.xi is None or baseline_data.xi.shape != result_data.xi.shape:
        print(f"Baseline comparison skipped; incompatible xi shape in {baseline}")
        print(f"Gaussian-ring output: {result_path}")
        return

    diff = result_data.xi - baseline_data.xi
    rel_l2 = np.linalg.norm(diff.ravel()) / max(
        np.linalg.norm(baseline_data.xi.ravel()),
        np.finfo(np.float64).eps,
    )
    print("Baseline comparison against built-in ring result:")
    print(f"  baseline: {baseline}")
    print(f"  gaussian: {result_path}")
    print(f"  max_abs : {np.max(np.abs(diff)):.6e}")
    print(f"  rel_l2  : {rel_l2:.6e}")


def load_gaussian_ring_params(config_path, output_path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    params = read_param(config_path=config_path)
    baseline_output = ""
    error = None

    if rank == 0:
        try:
            params = copy.deepcopy(params)
            if "Corr_2PCF" not in params:
                raise KeyError("Expected a Corr_2PCF section in the input config.")
            task_params = params["Corr_2PCF"]
            baseline_output = task_params.get("fout_path", "")
            task_params["binning_window"] = "ring"
            task_params["fout_path"] = output_path
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(error)

    params = comm.bcast(params, root=0)
    baseline_output = comm.bcast(baseline_output, root=0)
    return params, baseline_output


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    sigma_perp = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SIGMA_PERP
    output_path = output_path_for_sigma(sigma_perp)
    params, baseline_output = load_gaussian_ring_params(config_path, output_path)

    corr2pcf = Corr2PCFGaussianRing(param_task=params, sigma_perp=sigma_perp)
    result_data = corr2pcf.run(overwrite=True)
    if corr2pcf.rank == 0:
        print(f"Gaussian-ring sigma_perp = {sigma_perp:g} Mpc/h")
        print_baseline_comparison(result_data, baseline_output, output_path)


if __name__ == "__main__":
    main()
