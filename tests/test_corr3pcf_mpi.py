"""Execute with mpiexec -n 2; compare actual multipole estimator layouts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from pyhermes.base import SFCProjection
from pyhermes.theory import Corr_3PCF_Multipole
from pyhermes.utils.mpi_util import MPI


def compare_layouts(output):
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        raise ValueError("The estimator smoke test requires two MPI ranks.")
    output = Path(output).resolve()
    if comm.Get_rank() == 0:
        output.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    SFCProjection({"SFCProjection": {
        "particle_pos": np.random.default_rng(902).uniform(0, 32, (64, 3)).astype(np.float32),
        "box_size": 32., "J": 3, "threads": 1,
        "wavelet_mode": "db2", "wavelet_level": 3, "phi_resolution": 8,
        "save_particle_data": True, "particle_data_path": str(output / "particles.npz"),
        "fout_path": str(output / "field.pkl"),
    }}).run()
    comm.Barrier()
    comparisons = []
    for method in ["gamma", "particle"]:
        reference = None
        for mode in ["serial", "pair_mpi", "sample_mpi"]:
            result = Corr_3PCF_Multipole({"Corr_3PCF_Multipole": {
                "sfc_field": str(output / "field.pkl"), "random": "uniform", "threads": 1,
                "summation_method": method, "summation_backend": "cpu", "execution_mode": mode,
                "window2": {"type": "sphere", "len_args": {"R": 1.}},
                "window3": {"type": "sphere", "len_args": {"R": 1.}},
                "binning_window12": {"type": "shell", "mapping": {"R": "r12"}},
                "binning_window13": {"type": "shell", "mapping": {"R": "r13"}},
                "sampling": {"mode": "paired", "r12": [4., 6.], "r13": [8., 10.]},
                "l_max": 2,
            }}).run(save_result=False)
            if comm.Get_rank() == 0:
                assert result.zeta_l.shape == (2, 3) and np.isfinite(result.zeta_l).all()
                if reference is None:
                    reference = result.zeta_l.copy()
                np.testing.assert_allclose(result.zeta_l, reference, rtol=1e-10, atol=1e-12)
                comparisons.append({"method": method, "mode": mode,
                                    "max_absolute_difference": float(np.max(abs(result.zeta_l-reference)))})
            comm.Barrier()
    if comm.Get_rank() == 0:
        (output / "verification.json").write_text(json.dumps({
            "status": "verified", "ranks": 2, "comparisons": comparisons,
        }, indent=2) + "\n")


def test_mpi_estimator_runner_requires_two_ranks():
    if MPI.COMM_WORLD.Get_size() != 1:
        pytest.skip("Use the CLI entry point when launched under MPI.")
    with pytest.raises(ValueError, match="two MPI ranks"):
        compare_layouts(".")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        compare_layouts(args.output_dir)
    except BaseException:
        if MPI.COMM_WORLD.Get_size() > 1:
            MPI.COMM_WORLD.Abort(1)
        raise
