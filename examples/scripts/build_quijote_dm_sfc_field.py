"""Build the example J=8 SFC field from Quijote dark-matter particles."""

import os
from pathlib import Path

from pyhermes.base.sfc_projection import SFCProjection


SNAPSHOT_BASE = Path(
    "/Raid6/1/xutp/Quijote/Snapshots/fiducial/8000/snapdir_004/snap_004"
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = EXAMPLES_DIR / "output/quijote_fiducial_8000_snap004_dm_sfc_J8.pkl"
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))


task = SFCProjection()
task.fin = {
    "path": SNAPSHOT_BASE,
    "format": "gadget_hdf5",
    "reader_params": {"ptype": 1, "position_scale": 1.0e-3},
}
task.box_size = 1000
task.J = 8
task.wavelet_mode = "db2"
task.wavelet_level = 10
task.phi_resolution = 1024
task.threads = THREADS
task.fout_path = str(OUTPUT_PATH)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
task.run(overwrite=True)
