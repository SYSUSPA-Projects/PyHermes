#!/bin/bash
#SBATCH --job-name=pyhermes_3pcf_mp_smoke
#SBATCH --partition=gpu4090
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=./output/slurm_3pcf_multipole_smoke_%j.log

set -euo pipefail

source /software/modules/init/profile.sh

# If your environment is conda-based, replace "pyhermes" with the actual env name.
source /software/miniconda3/etc/profile.d/conda.sh
conda activate pyhermes

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export PYTHONUNBUFFERED=1

cd /home/${USER}/PyHermes/examples

nvidia-smi
python run_3pcf_multipole.py ./configs/param_3pcf_multipole_smoke.yaml
