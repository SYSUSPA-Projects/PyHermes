#!/bin/bash
#SBATCH --job-name=fig13-2pcf-opt
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=fig13_2pcf_opt_%j.log
set -euo pipefail
# Submit from the prepared experiment directory, or set FIG13_OPT_EXPERIMENT.
cd "${FIG13_OPT_EXPERIMENT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export FIG13_OPT_OUTPUT="${FIG13_OPT_OUTPUT:-$PWD/output/fig13_2pcf_optimization}"
exec "${PYTHON_BIN:-python}" campaign.py
