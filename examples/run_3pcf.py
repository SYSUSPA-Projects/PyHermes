"""
Example: run 3PCF with PyHermes

Purpose
-------
This script computes the 3-point correlation function (3PCF) using PyHermes.

It reads the task configuration from a YAML file, builds the corresponding
`Corr_3PCF` task, and runs it.

Input / output
--------------
The configuration file used here is:
    ./configs/param_3pcf.yaml

Usage
-----
1. Edit the YAML config file if needed:
       ./configs/param_3pcf.yaml

2. Run:
       python run_3pcf.py

MPI usage
---------
To run with MPI, use:
       mpirun -np <NPROC> python run_3pcf.py

For example:
       mpirun -np 8 python run_3pcf.py

Notes
-----
- `read_param(...)` is only executed on rank 0 in PyHermes' parameter workflow.
  If you want to modify parameters directly inside this Python script when using MPI,
  do it only on rank 0.

- The commented block below is an example of this pattern:
      if MPI.COMM_WORLD.Get_rank() == 0:
          corr3pcf_params['Corr_3PCF']['center'] = 'particle'
          corr3pcf_params['Corr_3PCF']['fout_path'] = "./output/quijote_3pcf_part.pkl"

  This means:
  * only rank 0 changes the parameter dictionary
  * the updated parameters will then be used consistently in the MPI run

- The `center` option controls how the 3PCF estimator samples the center positions:
  * `center = "random"`   : use uniformly distributed random centers
  * `center = "particle"` : use particle positions as centers
"""

from pyhermes.theory.corr3pcf import Corr_3PCF
from pyhermes.param.parambase import read_param

corr3pcf_params = read_param(config_path='./configs/param_3pcf.yaml')

# Example: modify parameters only on rank 0 when running with MPI
# from pyhermes.utils.mpi_util import MPI
# if MPI.COMM_WORLD.Get_rank() == 0:
#     corr3pcf_params['Corr_3PCF']['center'] = 'particle'
#     corr3pcf_params['Corr_3PCF']['fout_path'] = "./output/quijote_3pcf_part.pkl"

corr3pcf = Corr_3PCF(param_task=corr3pcf_params)
corr3pcf.run(overwrite=True)
