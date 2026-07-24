"""Build the example J=8 SFC field from a local Quijote DM snapshot."""

import argparse
import os
from pathlib import Path

from pyhermes.base.sfc_projection import SFCProjection

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = EXAMPLES_DIR / "output/quijote_fiducial_8000_snap004_dm_sfc_J8.pkl"
DEFAULT_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a J=8 SFC field from a local Gadget HDF5 Quijote "
            "dark-matter snapshot."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "snapshot_base",
        type=Path,
        help=(
            "Snapshot base path understood by the Gadget HDF5 reader, for example "
            "/data/snapdir_004/snap_004."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output SFCField pickle path.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="CPU threads used by SFCProjection.",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    output_path = args.output.expanduser().resolve()

    task = SFCProjection()
    task.fin = {
        "path": args.snapshot_base.expanduser(),
        "format": "gadget_hdf5",
        "reader_params": {"ptype": 1, "position_scale": 1.0e-3},
    }
    task.box_size = 1000
    task.J = 8
    task.wavelet_mode = "db2"
    task.wavelet_level = 10
    task.phi_resolution = 1024
    task.threads = args.threads
    task.fout_path = str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    task.run(overwrite=True)


if __name__ == "__main__":
    main()
