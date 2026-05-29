"""
Two-stage Corr_3PCF run with explicit random-field normalization.

The script is self-contained and does not read a YAML config. Rank 0 reads the
data/random ConvolsData objects, attaches heavy arrays to each Corr_3PCF task,
and lets Corr_3PCF broadcast only the fields needed by each product.

Stage 1 computes random-center ``r_delta_dd`` through the existing box-random
``ddd`` product by setting leg 1 to the unit-weight random field and legs 2/3
to the already normalized contrast field ``d-r``.
Stage 2 computes particle-center ``d_delta_dd`` and the reduced denominator
``zeta_H``. Rank 0 then combines both stages and saves the final result.
"""

import gc
import os

import numpy as np

from pyhermes.io import ConvolsData
from pyhermes.theory.corr3pcf import Corr_3PCF
from pyhermes.utils.mpi_util import MPI


DATA_PATH = "./output/quijote8000_snap004_sfc.pkl"
RANDOM_PATH = "./output/random_sfc.pkl"
OUTPUT_PATH = "./output/quijote8000_snap004_3pcf_pcenter_with_random.pkl"

WINDOW = {"type": "sphere", "len_args": {"R": 5}}
R12 = 20.0
R13 = 40.0
THETA = np.linspace(0.0, np.pi, 20)
BASE_SEED = 42


def corr3pcf_params(**updates):
    """Build lightweight task parameters shared by all MPI ranks."""
    params = {
        "r12": R12,
        "r13": R13,
        "weight_normalization": "catalog",
        "angle_param": "theta",
        "theta": THETA.tolist(),
        "base_seed": BASE_SEED,
        "threads": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "fout_path": "",
    }
    params.update(updates)
    return {"Corr_3PCF": params}


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = ConvolsData(data_path=DATA_PATH)
    random = ConvolsData(data_path=RANDOM_PATH)
    data_stat = data
    random_stat = random
else:
    data = None
    random = None
    data_stat = None
    random_stat = None


# ---------------------------------------------------------------------------
# Stage 1: random-center r_delta_dd and rrr.
# ---------------------------------------------------------------------------

random_center_task = Corr_3PCF(
    param_task=corr3pcf_params(
        n_rot=200,
        center="box_random",
        n_box_centers=8_000_000,
        products=["ddd", "rrr"],
        window2=WINDOW,
        window3=WINDOW,
    )
)

if rank == 0:
    delta_field = data_stat - random_stat
    random_center_task.convols_data1 = random_stat.copy()
    random_center_task.convols_data2 = delta_field
    random_center_task.convols_data3 = delta_field.copy()
    random_center_task.random1 = random_stat.copy()
    random_center_task.random2 = random_stat.copy()
    random_center_task.random3 = random_stat.copy()

random_center_result = random_center_task.run(save_result=False)

if rank == 0:
    r_delta_dd = np.array(random_center_result.ddd, copy=True)
    rrr = np.array(random_center_result.rrr, copy=True)
else:
    r_delta_dd = None
    rrr = None

del random_center_result, random_center_task
if rank == 0:
    del delta_field
gc.collect()
comm.Barrier()


# ---------------------------------------------------------------------------
# Stage 2: particle-center d_delta_dd and zeta_H.
# ---------------------------------------------------------------------------

particle_center_task = Corr_3PCF(
    param_task=corr3pcf_params(
        n_rot=1000,
        center="particle",
        products=["d_delta_dd", "zeta_H"],
        window2=WINDOW,
        window3=WINDOW,
    )
)

if rank == 0:
    particle_center_task.convols_data1 = data.copy()
    particle_center_task.convols_data2 = data.copy()
    particle_center_task.convols_data3 = data.copy()
    particle_center_task.random1 = random.copy()
    particle_center_task.random2 = random.copy()
    particle_center_task.random3 = random.copy()

particle_center_result = particle_center_task.run(save_result=False)

if rank == 0:
    particle_center_result.r_delta_dd = r_delta_dd
    particle_center_result.rrr = rrr
    particle_center_result.delta_ddd = particle_center_result.d_delta_dd - particle_center_result.r_delta_dd
    particle_center_result.zeta = particle_center_result.delta_ddd / particle_center_result.rrr
    particle_center_result.Q = particle_center_result.zeta / particle_center_result.zeta_H

    particle_center_result.corr3pcf_info["products"] = [
        "d_delta_dd",
        "r_delta_dd",
        "rrr",
        "delta_ddd",
        "zeta",
        "zeta_H",
        "Q",
    ]
    particle_center_result.corr3pcf_info["two_stage"] = {
        "r_delta_dd_rrr": {
            "center": "box_random",
            "n_rot": 200,
            "n_box_centers": 8_000_000,
            "r_delta_dd_source": "box_random ddd with leg1=r and legs2/3=d-r",
        },
        "d_delta_dd_zeta_H": {
            "center": "particle",
            "n_rot": 1000,
        },
    }
    particle_center_result.save_corr3pcf(OUTPUT_PATH, overwrite=True)

comm.Barrier()
