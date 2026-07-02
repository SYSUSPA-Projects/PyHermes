"""
Benchmark isotropic 2PCF measurements with PyHermes, pycorr, and TreeCorr.

The script mirrors examples/notebooks/benchmark_2pcf_isotropic_compare.ipynb
but is easier to submit on a server.

Typical CPU run:

    python examples/scripts/benchmark_2pcf_isotropic_compare.py --threads 8

Optional TreeCorr MPI-only run:

    mpirun -np 4 python examples/scripts/benchmark_2pcf_isotropic_compare.py \\
        --treecorr-mpi --treecorr-npatch 64 --threads 1

The default outputs are written to examples/output and examples/figs.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT_DIR / "examples"
DATA_DIR = EXAMPLES_DIR / "data"
OUTPUT_DIR = EXAMPLES_DIR / "output"
FIGS_DIR = EXAMPLES_DIR / "figs"


def configure_serial_mpi_environment() -> None:
    """Keep serial runs from opening unnecessary Open MPI network transports."""
    if "OMPI_COMM_WORLD_SIZE" not in os.environ:
        os.environ.setdefault("OMPI_MCA_btl", "self")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare isotropic 2PCF runtime and curves for PyHermes, pycorr, and TreeCorr."
    )
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for each code. Default: 8.")
    parser.add_argument("--box-size", type=float, default=1000.0, help="Periodic box size. Default: 1000.")
    parser.add_argument("--bin-min", type=float, default=5.0, help="Minimum pair-count bin edge. Default: 5.")
    parser.add_argument("--bin-max", type=float, default=150.0, help="Maximum pair-count bin edge. Default: 150.")
    parser.add_argument("--n-bins", type=int, default=31, help="Number of pycorr/TreeCorr bins. Default: 31.")
    parser.add_argument("--n-max", type=int, default=None, help="Optional random subsample size for smoke tests.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed used for subsampling/randoms. Default: 12345.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DATA_DIR / "quijote_halos/8000/groups_004/group_tab_004.pos.npz",
        help="Input .npz catalogue with a 'pos' array.",
    )
    parser.add_argument(
        "--sfc-field",
        type=Path,
        default=OUTPUT_DIR / "quijote8000_snap004_sfc.pkl",
        help="PyHermes SFCField file used for the warm run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for benchmark JSON/CSV/NPZ/PyHermes outputs.",
    )
    parser.add_argument(
        "--figs-dir",
        type=Path,
        default=FIGS_DIR,
        help="Directory for the comparison figure.",
    )
    parser.add_argument("--skip-pyhermes", action="store_true", help="Skip the PyHermes warm 2PCF benchmark.")
    parser.add_argument("--run-pyhermes-cold", action="store_true", help="Also time catalogue-to-field projection.")
    parser.add_argument("--skip-pycorr", action="store_true", help="Skip pycorr.")
    parser.add_argument("--skip-treecorr", action="store_true", help="Skip TreeCorr.")
    parser.add_argument(
        "--treecorr-rr",
        action="store_true",
        help="Run explicit TreeCorr RR and calculate xi. Default uses analytic periodic RR from DD.",
    )
    parser.add_argument(
        "--treecorr-random-factor",
        type=float,
        default=1.0,
        help="Random catalogue size divided by data size for --treecorr-rr. Default: 1.",
    )
    parser.add_argument(
        "--treecorr-mpi",
        action="store_true",
        help="Run TreeCorr with mpi4py comm. Intended for mpirun; skips PyHermes and pycorr.",
    )
    parser.add_argument(
        "--treecorr-npatch",
        type=int,
        default=0,
        help="Number of TreeCorr patches. Required for useful --treecorr-mpi scaling.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write the s^2 xi(s) comparison figure.",
    )
    return parser.parse_args()


def optional_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, exc


def maxrss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / 1024**2
    return value / 1024


def benchmark(label: str, func, **meta):
    print(f"\n--- {label} ---", flush=True)
    start_mem = maxrss_mb()
    t0 = time.perf_counter()
    try:
        value = func()
        status = "ok"
        error = ""
    except Exception as exc:
        value = None
        status = "failed"
        error = repr(exc)
        print("FAILED:", error, flush=True)
    elapsed = time.perf_counter() - t0
    end_mem = maxrss_mb()
    row = {
        "label": label,
        "status": status,
        "wall_time_s": elapsed,
        "maxrss_mb_after": end_mem,
        "maxrss_mb_delta": max(0.0, end_mem - start_mem),
        "error": error,
        **meta,
    }
    print({key: row[key] for key in ("status", "wall_time_s", "maxrss_mb_after", "maxrss_mb_delta")}, flush=True)
    return row, value


def save_rows(rows: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2))
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")


def load_positions(path: Path, n_max: int | None, seed: int) -> tuple[np.ndarray, bool]:
    data = np.load(path)
    pos = np.asarray(data["pos"], dtype=np.float64)
    subset_used = False
    if n_max is not None and n_max < len(pos):
        rng = np.random.default_rng(seed)
        index = rng.choice(len(pos), size=int(n_max), replace=False)
        pos = pos[index]
        subset_used = True
    return np.ascontiguousarray(pos), subset_used


def run_pyhermes_projection(pos: np.ndarray, args: argparse.Namespace, output_path: Path):
    from pyhermes.base.sfc_projection import SFCProjection

    params = {
        "SFCProjection": {
            "fin": {
                "path": "./data/quijote_halos/8000",
                "format": "fof",
                "reader_params": {"snapnum": 4},
            },
            "box_size": args.box_size,
            "J": 8,
            "wavelet_mode": "db2",
            "wavelet_level": 10,
            "phi_resolution": 1024,
            "threads": args.threads,
            "weight_normalization": "catalog",
            "save_particle_data": False,
            "fout_path": str(output_path),
        }
    }
    task = SFCProjection(params)
    task.particle_pos = np.asarray(pos, dtype=np.float32)
    return task.run(save_result=True, overwrite=True)


def run_pyhermes_2pcf(args: argparse.Namespace, sfc_path: Path, output_path: Path, s_samples: np.ndarray):
    from pyhermes.theory.corr2pcf import Corr_2PCF

    params = {
        "Corr_2PCF": {
            "sfc_field": str(sfc_path),
            "random": "uniform",
            "binning_window": "shell",
            "sampling": {
                "s": {
                    "min": float(s_samples[0]),
                    "max": float(s_samples[-1]),
                    "n": int(len(s_samples)),
                }
            },
            "products": ["xi"],
            "threads": args.threads,
            "weight_normalization": "catalog",
            "fout_path": str(output_path),
        }
    }
    return Corr_2PCF(params).run(save_result=True, overwrite=True)


def run_pycorr_2pcf(pos: np.ndarray, args: argparse.Namespace, edges: np.ndarray):
    from pycorr import TwoPointCorrelationFunction

    return TwoPointCorrelationFunction(
        "s",
        edges,
        data_positions1=pos,
        boxsize=args.box_size,
        position_type="pos",
        engine="corrfunc",
        nthreads=args.threads,
    )


def make_treecorr_catalog(pos: np.ndarray, npatch: int, seed: int):
    import treecorr

    kwargs = {}
    if npatch and npatch > 1:
        kwargs["npatch"] = int(npatch)
        kwargs["rng"] = np.random.RandomState(seed)
    return treecorr.Catalog(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], **kwargs)


def run_treecorr_dd(pos: np.ndarray, args: argparse.Namespace, edges: np.ndarray, comm=None):
    import treecorr

    cat = make_treecorr_catalog(pos, args.treecorr_npatch, args.seed)
    nn = treecorr.NNCorrelation(
        min_sep=float(edges[0]),
        max_sep=float(edges[-1]),
        nbins=int(len(edges) - 1),
        bin_type="Linear",
        num_threads=args.threads,
        period=args.box_size,
    )
    nn.process(cat, metric="Periodic", comm=comm)
    return nn


def run_treecorr_xi_with_rr(pos: np.ndarray, args: argparse.Namespace, edges: np.ndarray, comm=None):
    import treecorr

    rng = np.random.default_rng(args.seed)
    n_random = int(round(args.treecorr_random_factor * len(pos)))
    random_pos = rng.random((n_random, 3)) * args.box_size
    data_cat = make_treecorr_catalog(pos, args.treecorr_npatch, args.seed)
    random_cat = make_treecorr_catalog(random_pos, args.treecorr_npatch, args.seed + 1)
    kwargs = dict(
        min_sep=float(edges[0]),
        max_sep=float(edges[-1]),
        nbins=int(len(edges) - 1),
        bin_type="Linear",
        num_threads=args.threads,
        period=args.box_size,
    )
    dd = treecorr.NNCorrelation(**kwargs)
    rr = treecorr.NNCorrelation(**kwargs)
    dd.process(data_cat, metric="Periodic", comm=comm)
    rr.process(random_cat, metric="Periodic", comm=comm)
    xi, varxi = dd.calculateXi(rr=rr)
    return {"dd": dd, "rr": rr, "xi": xi, "varxi": varxi, "n_random": n_random}


def pyhermes_curve(value, output_path: Path):
    if value is None and output_path.exists():
        from pyhermes.io.corr2pcf import Corr2PCFData

        value = Corr2PCFData(data_path=str(output_path), threads=1)
    if value is None or getattr(value, "xi", None) is None:
        return None
    return np.asarray(value.s, dtype=float), np.asarray(value.xi, dtype=float)


def pycorr_curve(value):
    if value is None or getattr(value, "corr", None) is None:
        return None
    sep = getattr(value, "sep", None)
    if sep is None:
        return None
    return np.asarray(sep, dtype=float), np.asarray(value.corr, dtype=float)


def treecorr_curve(value, pos_count: int, box_size: float):
    if value is None:
        return None
    if isinstance(value, dict):
        dd = value["dd"]
        sep = np.asarray(getattr(dd, "meanr", None), dtype=float)
        if sep.size == 0 or not np.any(sep > 0):
            sep = np.asarray(dd.rnom, dtype=float)
        return sep, np.asarray(value["xi"], dtype=float)

    nn = value
    if getattr(nn, "npairs", None) is None:
        return None
    left = np.asarray(nn.left_edges, dtype=float)
    right = np.asarray(nn.right_edges, dtype=float)
    sep = np.asarray(getattr(nn, "meanr", None), dtype=float)
    if sep.size == 0 or not np.any(sep > 0):
        sep = np.asarray(nn.rnom, dtype=float)
    shell_volume = 4.0 * np.pi / 3.0 * (right**3 - left**3)
    rr_pairs = 0.5 * pos_count * (pos_count - 1) * shell_volume / box_size**3
    xi = np.asarray(nn.npairs, dtype=float) / rr_pairs - 1.0
    return sep, xi


def save_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path) -> None:
    payload = {}
    for name, (x_values, xi_values) in curves.items():
        key = name.replace(" ", "_").replace("/", "_")
        payload[f"{key}_s"] = x_values
        payload[f"{key}_xi"] = xi_values
        payload[f"{key}_s2xi"] = x_values**2 * xi_values
    np.savez(output_path, **payload)
    print(f"saved: {output_path}")


def plot_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for name, (x_values, xi_values) in curves.items():
        ax.plot(x_values, x_values**2 * xi_values, marker="o", markersize=3.5, linewidth=1.4, label=name)
    ax.axhline(0.0, color="0.7", linewidth=0.8)
    ax.set_xlabel(r"$s\ [h^{-1}\mathrm{Mpc}]$")
    ax.set_ylabel(r"$s^2\xi(s)\ [(h^{-1}\mathrm{Mpc})^2]$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {fig_path}")


def main() -> None:
    configure_serial_mpi_environment()
    args = parse_args()
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    os.chdir(EXAMPLES_DIR)

    comm = None
    rank = 0
    size = 1
    if args.treecorr_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        args.skip_pyhermes = True
        args.skip_pycorr = True
        args.skip_treecorr = False
        if args.treecorr_npatch <= 1 and rank == 0:
            print("WARNING: --treecorr-mpi is most useful with --treecorr-npatch > 1.")

    def rprint(*items) -> None:
        if rank == 0:
            print(*items, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figs_dir.mkdir(parents=True, exist_ok=True)

    edges = np.linspace(args.bin_min, args.bin_max, args.n_bins + 1)
    s_samples = 0.5 * (edges[:-1] + edges[1:])

    rprint("ROOT =", ROOT_DIR)
    rprint("EXAMPLES =", EXAMPLES_DIR)
    rprint("catalog =", args.catalog)
    rprint("threads =", args.threads)
    rprint("bin range =", (float(edges[0]), float(edges[-1])), "n_bins =", args.n_bins)
    if args.treecorr_mpi:
        rprint("TreeCorr MPI ranks =", size, "npatch =", args.treecorr_npatch)

    pos, subset_used = load_positions(args.catalog, args.n_max, args.seed)
    rprint("N_data =", len(pos))

    rows: list[dict] = []
    outputs = {}

    pyhermes_output_path = args.output_dir / "benchmark_pyhermes_2pcf_isotropic.pkl"
    active_sfc_path = args.sfc_field
    if subset_used:
        # Subsample benchmarks need their own field so all codes see the same data.
        active_sfc_path = args.output_dir / f"benchmark_pyhermes_sfc_N{len(pos)}.pkl"
        pyhermes_output_path = args.output_dir / f"benchmark_pyhermes_2pcf_isotropic_N{len(pos)}.pkl"
        args.run_pyhermes_cold = True

    if not args.skip_pyhermes:
        if args.run_pyhermes_cold or not active_sfc_path.exists():
            row, value = benchmark(
                "pyhermes_projection_cold",
                lambda: run_pyhermes_projection(pos, args, active_sfc_path),
                code="pyhermes",
                task="projection",
                n_data=len(pos),
                n_samples=0,
                threads=args.threads,
                box_size=args.box_size,
            )
            rows.append(row)
            outputs["pyhermes_projection"] = value

        row, value = benchmark(
            "pyhermes_warm_xi_s",
            lambda: run_pyhermes_2pcf(args, active_sfc_path, pyhermes_output_path, s_samples),
            code="pyhermes",
            task="xi_s",
            n_data=len(pos),
            n_samples=len(s_samples),
            threads=args.threads,
            box_size=args.box_size,
        )
        rows.append(row)
        outputs["pyhermes_xi_s"] = value

    if not args.skip_pycorr:
        pycorr, pycorr_error = optional_import("pycorr")
        if pycorr is None:
            rows.append(
                {
                    "label": "pycorr_xi_s",
                    "status": "missing",
                    "wall_time_s": np.nan,
                    "maxrss_mb_after": np.nan,
                    "maxrss_mb_delta": np.nan,
                    "error": repr(pycorr_error),
                    "code": "pycorr",
                    "task": "xi_s",
                    "n_data": len(pos),
                    "n_samples": len(edges) - 1,
                    "threads": args.threads,
                    "box_size": args.box_size,
                }
            )
        else:
            rprint("pycorr version:", getattr(pycorr, "__version__", "unknown"))
            row, value = benchmark(
                "pycorr_xi_s",
                lambda: run_pycorr_2pcf(pos, args, edges),
                code="pycorr",
                task="xi_s",
                n_data=len(pos),
                n_samples=len(edges) - 1,
                threads=args.threads,
                box_size=args.box_size,
            )
            rows.append(row)
            outputs["pycorr_xi_s"] = value

    if not args.skip_treecorr:
        treecorr, treecorr_error = optional_import("treecorr")
        if treecorr is None:
            rows.append(
                {
                    "label": "treecorr_dd_xi_s",
                    "status": "missing",
                    "wall_time_s": np.nan,
                    "maxrss_mb_after": np.nan,
                    "maxrss_mb_delta": np.nan,
                    "error": repr(treecorr_error),
                    "code": "treecorr",
                    "task": "dd_or_xi_s",
                    "n_data": len(pos),
                    "n_samples": len(edges) - 1,
                    "threads": args.threads,
                    "box_size": args.box_size,
                    "mpi_ranks": size,
                }
            )
        else:
            rprint("TreeCorr version:", getattr(treecorr, "__version__", "unknown"))
            if args.treecorr_rr:
                row, value = benchmark(
                    "treecorr_xi_s_with_rr",
                    lambda: run_treecorr_xi_with_rr(pos, args, edges, comm=comm),
                    code="treecorr",
                    task="xi_s_with_explicit_rr",
                    n_data=len(pos),
                    n_random=int(round(args.treecorr_random_factor * len(pos))),
                    n_samples=len(edges) - 1,
                    threads=args.threads,
                    box_size=args.box_size,
                    mpi_ranks=size,
                    npatch=args.treecorr_npatch,
                )
                rows.append(row)
                outputs["treecorr_xi_with_rr"] = value
            else:
                row, value = benchmark(
                    "treecorr_dd_xi_s",
                    lambda: run_treecorr_dd(pos, args, edges, comm=comm),
                    code="treecorr",
                    task="DD_s",
                    n_data=len(pos),
                    n_samples=len(edges) - 1,
                    threads=args.threads,
                    box_size=args.box_size,
                    mpi_ranks=size,
                    npatch=args.treecorr_npatch,
                )
                rows.append(row)
                outputs["treecorr_dd"] = value

    if rank != 0:
        return

    result_json = args.output_dir / "benchmark_2pcf_isotropic_compare.json"
    result_csv = args.output_dir / "benchmark_2pcf_isotropic_compare.csv"
    save_rows(rows, result_json, result_csv)

    curves = {
        "PyHermes shell": pyhermes_curve(outputs.get("pyhermes_xi_s"), pyhermes_output_path),
        "pycorr bins": pycorr_curve(outputs.get("pycorr_xi_s")),
        "TreeCorr DD/RR": treecorr_curve(
            outputs.get("treecorr_xi_with_rr") or outputs.get("treecorr_dd"),
            len(pos),
            args.box_size,
        ),
    }
    curves = {name: curve for name, curve in curves.items() if curve is not None}
    if curves:
        curve_path = args.output_dir / "benchmark_2pcf_isotropic_s2xi_curves.npz"
        save_curves(curves, curve_path)
        if not args.no_plot:
            fig_path = args.figs_dir / "benchmark_2pcf_isotropic_s2xi_curves.png"
            plot_curves(curves, fig_path)
    else:
        print("No successful curve outputs were available for plotting.")


if __name__ == "__main__":
    main()
