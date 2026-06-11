"""
Build a PyHermes ConvolsData field from a Quijote dark-matter snapshot.

Typical server run from the PyHermes repository root:

    python examples/scripts/build_quijote_dm_convols.py

The default input is the Quijote fiducial realization 8000, snapshot 004
dark-matter particle snapshot, and the default output is the file expected by
the comparison notebook:

    examples/output/quijote_fiducial_8000_snap004_dm_sfc_J8.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_BASE = Path(
    "/Raid6/1/xutp/Quijote/Snapshots/fiducial/8000/snapdir_004/snap_004"
)
DEFAULT_OUTPUT = REPO_ROOT / "examples/output/quijote_fiducial_8000_snap004_dm_sfc_J8.pkl"


def configure_serial_mpi_environment() -> None:
    """Keep serial runs from opening unnecessary Open MPI network transports."""
    if "OMPI_COMM_WORLD_SIZE" not in os.environ:
        os.environ.setdefault("OMPI_MCA_btl", "self")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return number


def parse_args() -> argparse.Namespace:
    default_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    parser = argparse.ArgumentParser(
        description="Project Quijote dark-matter particles to a saved PyHermes ConvolsData field."
    )
    parser.add_argument(
        "--snapshot-base",
        type=Path,
        default=DEFAULT_SNAPSHOT_BASE,
        help=(
            "Base Gadget snapshot path. For split files, pass the common base, "
            "for example /.../snapdir_004/snap_004. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output ConvolsData .pkl path. Default: %(default)s",
    )
    parser.add_argument("--j", type=positive_int, default=8, help="PyHermes grid level J. Default: 8.")
    parser.add_argument("--box-size", type=float, default=1000.0, help="Box size in Mpc/h. Default: 1000.")
    parser.add_argument("--ptype", type=int, default=1, help="Gadget particle type. Default: 1 for DM.")
    parser.add_argument(
        "--threads",
        type=positive_int,
        default=max(1, default_threads),
        help="CPU threads for PyHermes/Numba. Default: SLURM_CPUS_PER_TASK or 8.",
    )
    parser.add_argument("--wavelet-mode", default="db2", help="Wavelet family. Default: db2.")
    parser.add_argument("--wavelet-level", type=positive_int, default=10, help="Wavelet level. Default: 10.")
    parser.add_argument("--phi-resolution", type=positive_int, default=1024, help="Phi lookup resolution.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild even if the output file already exists.",
    )
    return parser.parse_args()


def split_snapshot_files(snapshot_base: Path) -> list[Path]:
    if snapshot_base.exists():
        return [snapshot_base]

    def sort_key(path: Path):
        suffix = path.name.rsplit(".", 1)[-1]
        return int(suffix) if suffix.isdigit() else suffix

    return sorted(snapshot_base.parent.glob(snapshot_base.name + ".*"), key=sort_key)


def resolve_output_path(output_path: Path) -> Path:
    if output_path.is_absolute():
        return output_path
    return (REPO_ROOT / output_path).resolve()


def summarize_field(label: str, field) -> None:
    import numpy as np

    epsilon = field.epsilon
    print(
        f"{label}: shape={epsilon.shape}, dtype={epsilon.dtype}, "
        f"min={np.min(epsilon):.6e}, max={np.max(epsilon):.6e}, "
        f"mean={np.mean(epsilon):.6e}, sum={np.sum(epsilon, dtype=np.float64):.6e}"
    )
    print(
        "metadata:",
        f"J={field.J}",
        f"L={field.L}",
        f"box_size={field.box_size}",
        f"particle_count={field.particle_count}",
        f"weight_normalization={field.weight_normalization}",
    )


def main() -> None:
    configure_serial_mpi_environment()
    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
    os.environ["NUMBA_NUM_THREADS"] = str(args.threads)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from pyhermes.base.convols import Convols
    from pyhermes.io.convols import ConvolsData
    from pyhermes.utils.mpi_util import MPI

    output_path = resolve_output_path(args.output)
    snapshot_files = split_snapshot_files(args.snapshot_base)

    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("This helper is intended for a single Slurm task; run without mpirun/srun -n > 1.")
    if not snapshot_files:
        raise FileNotFoundError(
            f"Could not find '{args.snapshot_base}' or split files like '{args.snapshot_base}.0'."
        )
    if args.box_size <= 0:
        raise ValueError("--box-size must be positive.")

    os.chdir(REPO_ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("PyHermes root:", REPO_ROOT)
    print("Snapshot base:", args.snapshot_base)
    print("Snapshot files:", len(snapshot_files))
    print("First snapshot file:", snapshot_files[0])
    print("Output:", output_path)
    print("Threads:", args.threads)
    print("Grid:", f"J={args.j}", f"L={1 << args.j}", f"dx={args.box_size / (1 << args.j):.6g} Mpc/h")

    if output_path.exists() and not args.overwrite:
        print("Output already exists; skipping rebuild. Use --overwrite to rebuild.")
        existing = ConvolsData(data_path=str(output_path), threads=args.threads)
        summarize_field("existing ConvolsData", existing)
        return

    task_params = {
        "fin": {
            "path": str(args.snapshot_base),
            "format": "gadget",
            "reader_params": {"ptype": args.ptype},
            "catalog_weight_key": None,
            "field_value_key": None,
        },
        "weight_normalization": "catalog",
        "box_size": float(args.box_size),
        "J": int(args.j),
        "wavelet_mode": args.wavelet_mode,
        "wavelet_level": int(args.wavelet_level),
        "phi_resolution": int(args.phi_resolution),
        "threads": int(args.threads),
        "fout_path": str(output_path),
    }
    task = Convols({"Convols": task_params})
    field = task.run(save_result=True, overwrite=args.overwrite)
    summarize_field("built ConvolsData", field)
    print("Done:", output_path)


if __name__ == "__main__":
    main()
