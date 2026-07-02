"""
Run the s-mu 2PCF with a ring binning window composed from projected factors.

This script mirrors configs/param_2pcf_smu.yaml, but replaces each built-in
ring binning window with the projected-kernel product of two custom windows:

    J0(2*pi*k_perp*R) * cos(2*pi*k_parallel*H)

The product is formed with WindowFunc.__mul__, so it matches the current
projected-operator composition semantics rather than a raw transfer product
projected once.
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
DEFAULT_OUTPUT = "./output/quijote8000_snap004_rsd_2pcf_smu_projected_ring_product.pkl"


@njit
def window_function_ring_perp_z_numba(ki, kj, kk, R):
    q_perp = 2.0 * np.pi * np.sqrt(ki * ki + kj * kj) * R
    return jn_numba(0, q_perp)


@njit
def window_function_ring_parallel_z_numba(ki, kj, kk, H):
    q_parallel = 2.0 * np.pi * kk * H
    return np.cos(q_parallel)


class Corr2PCFProjectedRingProduct(Corr_2PCF):
    """Corr_2PCF variant using W_ring = W_perp * W_parallel for each sample."""

    __module__ = Corr_2PCF.__module__

    def __init__(self, param_task=None):
        if param_task is None:
            param_task = {"Corr_2PCF": {}}
        self.task_name = "Corr_2PCF"
        super(Corr_2PCF, self).__init__(param_task=param_task)
        self.format_params()
        self._fields_prepared = False

    def _build_binning_window_for_sample(self, sample, reference_field):
        ring_params = build_binning_window_params_for_sample(sample, self.binning_window)
        radius = float(ring_params["len_args"]["R"])
        half_height = float(ring_params["len_args"]["H"])

        perp_window = WindowFunc(
            {
                "type": "ring_perp_z",
                "func": window_function_ring_perp_z_numba,
                "len_args": {"R": radius},
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
        print(f"Projected-ring-product output: {result_path}")
        return

    baseline_data = Corr2PCFData(data_path=str(baseline), threads=1)
    if baseline_data.xi is None or baseline_data.xi.shape != result_data.xi.shape:
        print(f"Baseline comparison skipped; incompatible xi shape in {baseline}")
        print(f"Projected-ring-product output: {result_path}")
        return

    diff = result_data.xi - baseline_data.xi
    rel_l2 = np.linalg.norm(diff.ravel()) / max(
        np.linalg.norm(baseline_data.xi.ravel()),
        np.finfo(np.float64).eps,
    )
    print("Baseline comparison against built-in ring result:")
    print(f"  baseline: {baseline}")
    print(f"  product : {result_path}")
    print(f"  max_abs : {np.max(np.abs(diff)):.6e}")
    print(f"  rel_l2  : {rel_l2:.6e}")


def load_projected_ring_product_params(config_path):
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
            task_params["fout_path"] = DEFAULT_OUTPUT
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
    params, baseline_output = load_projected_ring_product_params(config_path)

    corr2pcf = Corr2PCFProjectedRingProduct(param_task=params)
    result_data = corr2pcf.run(overwrite=True)
    if corr2pcf.rank == 0:
        print_baseline_comparison(result_data, baseline_output, DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
