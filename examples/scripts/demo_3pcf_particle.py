"""Small CPU-only particle-multipole example with no catalogue download."""

import argparse
import os
from pathlib import Path

import numpy as np


def run_demo(output_dir, threads=2):
    if "OMPI_COMM_WORLD_SIZE" not in os.environ:
        os.environ.setdefault("OMPI_MCA_btl", "self")
    from pyhermes.base import SFCProjection
    from pyhermes.theory import Corr_3PCF_Multipole
    from pyhermes.utils.mpi_util import MPI

    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError("This projection demo is serial; use the YAML driver for MPI measurements.")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "field.pkl").exists() or (output / "multipoles.pkl").exists():
        raise FileExistsError("Choose a fresh output directory for the demo.")
    field = SFCProjection({"SFCProjection": {
        "particle_pos": np.random.default_rng(123).uniform(0, 64, (512, 3)).astype(np.float32),
        "box_size": 64.0, "J": 4, "threads": threads,
        "wavelet_mode": "db2", "wavelet_level": 3, "phi_resolution": 8,
        "weight_normalization": "catalog", "save_particle_data": True,
        "particle_data_path": str(output / "particles.npz"),
        "fout_path": str(output / "field.pkl"),
    }}).run()
    params = {"Corr_3PCF_Multipole": {
        "sfc_field": str(output / "field.pkl"), "random": "uniform",
        "weight_normalization": "catalog", "summation_method": "particle",
        "summation_backend": "cpu", "execution_mode": "serial", "threads": threads,
        "window2": {"type": "sphere", "len_args": {"R": 2.0}},
        "window3": {"type": "sphere", "len_args": {"R": 2.0}},
        "binning_window12": {"type": "shell", "mapping": {"R": "r12"}},
        "binning_window13": {"type": "shell", "mapping": {"R": "r13"}},
        "sampling": {"mode": "paired", "r12": [8.0], "r13": [16.0]},
        "l_max": 2, "products": "zeta_l", "fout_path": str(output / "multipoles.pkl"),
    }}
    result = Corr_3PCF_Multipole(params).run()
    assert field is not None and result.zeta_l.shape == (1, 3)
    assert np.isfinite(result.zeta_l).all()
    np.savez(output / "multipoles.npz", ell=result.l, r12=result.r12,
             r13=result.r13, zeta_l=result.zeta_l)
    print("ell:", result.l)
    print("zeta_l:", result.zeta_l)
    print("Saved to:", output)
    print("Synthetic low-resolution demonstration; not a cosmological accuracy reference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("./output/particle_demo"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run_demo(args.output_dir, args.threads)


if __name__ == "__main__":
    main()
