"""
Prepare the SFCField products used by the PyHermes examples.

Run without MPI:

    python examples/scripts/prepare_sfc_fields.py

The script resolves paths relative to the examples directory, so it works from
the repository root or from examples/.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXAMPLES_DIR / "data"
OUTPUT_DIR = EXAMPLES_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the example SFCField files required by later PyHermes tutorials."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of CPU threads used by each SFCProjection task. Default: 8.",
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=10_000_000,
        help="Number of uniform random points used for random_sfc.pkl. Default: 10000000.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild files even when the target output already exists.",
    )
    return parser.parse_args()


def configure_serial_mpi_environment() -> None:
    """Keep serial runs from opening unnecessary Open MPI network transports."""
    if "OMPI_COMM_WORLD_SIZE" not in os.environ:
        os.environ.setdefault("OMPI_MCA_btl", "self")


def print_step(message: str) -> None:
    print(f"\n== {message} ==")


def fof_input(fields=None) -> dict:
    from pyhermes.param.parambase import read_param

    params = read_param(config_path="./configs/param_sfc_projection.yaml")
    source = params["SFCProjection"]["fin"]
    reader_params = source["reader_params"]
    reader_params["redshift"] = 0.0
    if fields is not None:
        reader_params["fields"] = fields
    return source


def read_fof_catalog() -> dict:
    from pyhermes.io import read_particle_data

    source = fof_input(
        fields={
            "vel": "vel",
            "vx": "vel_x",
            "vy": "vel_y",
            "vz": "vel_z",
            "mass": "mass",
            "npart": "npart",
        }
    )
    return read_particle_data(
        source["path"],
        data_format=source["format"],
        download=source["download"],
        **source["reader_params"],
    )


def run_if_needed(label: str, output_path: Path, overwrite: bool, build_task) -> None:
    print_step(label)
    if output_path.exists() and not overwrite:
        print(f"Output already exists: {output_path.relative_to(EXAMPLES_DIR)}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task = build_task()
    task.run(overwrite=True)


def base_sfc_task(threads: int, fout_path: str):
    from pyhermes.base.sfc_projection import SFCProjection

    task = SFCProjection()
    task.box_size = 1000
    task.J = 8
    task.wavelet_mode = "db2"
    task.wavelet_level = 10
    task.phi_resolution = 1024
    task.threads = threads
    task.fout_path = fout_path
    return task


def build_standard_field(threads: int):
    from pyhermes.base.sfc_projection import SFCProjection
    from pyhermes.param.parambase import read_param

    params = read_param(config_path="./configs/param_sfc_projection.yaml")
    task = SFCProjection(param_task=params)
    task.threads = threads
    return task


def build_j9_field(threads: int):
    task = base_sfc_task(threads, "./output/quijote8000_snap004_sfc_J9.pkl")
    task.fin = fof_input()
    task.J = 9
    return task


def build_mass_weighted_field(threads: int):
    task = base_sfc_task(threads, "./output/quijote8000_snap004_sfc_massweight.pkl")
    task.fin = fof_input(fields={"mass": "mass"})
    task.fin["field_value_key"] = "mass"
    return task


def build_redshift_space_field(
    pos: np.ndarray,
    threads: int,
    diag: bool = False,
    field_value: np.ndarray | None = None,
):
    suffix = "rsd_diag" if diag else "rsd"
    weight_suffix = "_massweight" if field_value is not None else ""
    task = base_sfc_task(
        threads,
        f"./output/quijote8000_snap004_{suffix}_sfc{weight_suffix}.pkl",
    )
    task.particle_pos = pos
    if field_value is not None:
        task.field_value = np.asarray(field_value, dtype=np.float32)
    task.save_particle_data = True
    task.particle_data_path = (
        f"./data/quijote_halos/8000/groups_004/group_tab_004.pos.{suffix}{weight_suffix}.npz"
    )
    return task


def build_random_field(random_count: int, threads: int):
    from pyhermes.utils.sampling import random_box_positions

    random_pos = random_box_positions(count=random_count, box_size=1000, seed=42).astype(
        np.float32, copy=False
    )
    task = base_sfc_task(threads, "./output/random_sfc.pkl")
    task.particle_pos = random_pos
    task.save_particle_data = True
    task.particle_data_path = "./data/random_1e7.npz"
    return task


def main() -> None:
    args = parse_args()
    configure_serial_mpi_environment()
    from pyhermes.utils.mpi_util import MPI
    from pyhermes.utils.redshift_space import (
        hubble_at_redshift,
        redshift_space_positions,
    )

    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("prepare_sfc_fields.py is a serial helper; run it without mpirun.")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1.")
    if args.random_count < 1:
        raise ValueError("--random-count must be >= 1.")

    os.chdir(EXAMPLES_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {EXAMPLES_DIR}")

    print_step("Loading the public FoF catalogue")
    fof_data = read_fof_catalog()

    run_if_needed(
        "Building real-space SFCField",
        OUTPUT_DIR / "quijote8000_snap004_sfc.pkl",
        args.overwrite,
        lambda: build_standard_field(args.threads),
    )
    run_if_needed(
        "Building J=9 SFCField",
        OUTPUT_DIR / "quijote8000_snap004_sfc_J9.pkl",
        args.overwrite,
        lambda: build_j9_field(args.threads),
    )
    run_if_needed(
        "Building mass-weighted SFCField",
        OUTPUT_DIR / "quijote8000_snap004_sfc_massweight.pkl",
        args.overwrite,
        lambda: build_mass_weighted_field(args.threads),
    )

    print_step("Preparing redshift-space particle positions")
    hubble_parameter = hubble_at_redshift(redshift=0)
    pos_z = redshift_space_positions(
        fof_data["pos"], fof_data["vel"], box_size=1000, hubble=hubble_parameter, redshift=0, los="z"
    )
    pos_diag = redshift_space_positions(
        fof_data["pos"],
        fof_data["vel"],
        box_size=1000,
        hubble=hubble_parameter,
        redshift=0,
        los=[1, 1, 1],
    )

    run_if_needed(
        "Building redshift-space SFCField with z-axis LOS",
        OUTPUT_DIR / "quijote8000_snap004_rsd_sfc.pkl",
        args.overwrite,
        lambda: build_redshift_space_field(pos_z, args.threads, diag=False),
    )
    run_if_needed(
        "Building mass-weighted redshift-space SFCField with z-axis LOS",
        OUTPUT_DIR / "quijote8000_snap004_rsd_sfc_massweight.pkl",
        args.overwrite,
        lambda: build_redshift_space_field(
            pos_z,
            args.threads,
            diag=False,
            field_value=fof_data["mass"],
        ),
    )
    run_if_needed(
        "Building redshift-space SFCField with diagonal LOS",
        OUTPUT_DIR / "quijote8000_snap004_rsd_diag_sfc.pkl",
        args.overwrite,
        lambda: build_redshift_space_field(pos_diag, args.threads, diag=True),
    )
    run_if_needed(
        "Building matching random SFCField",
        OUTPUT_DIR / "random_sfc.pkl",
        args.overwrite,
        lambda: build_random_field(args.random_count, args.threads),
    )

    print_step("Done")
    print("Generated products are available under examples/output/.")


if __name__ == "__main__":
    main()
