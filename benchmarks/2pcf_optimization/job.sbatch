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
#SBATCH --output=/home/xutp/pyhermes_projects/isotropic_3pcf_benchmark/logs/fig13_2pcf_opt_%j.log
set -euo pipefail
export PATH=/usr/local/slurm/bin:/home/xutp/anaconda3/bin:$PATH
export FIG13_OPT_OUTPUT=/home/xutp/pyhermes_projects/isotropic_3pcf_benchmark/output/fig13_2pcf_optimization_20261016_attempt2
cd /home/xutp/pyhermes_projects/isotropic_3pcf_benchmark/experiments/fig13_2pcf_optimization_20261016_attempt2
exec /home/xutp/anaconda3/bin/python campaign.py
