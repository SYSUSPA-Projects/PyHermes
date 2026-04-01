"""
Example: run 2PCF with PyHermes

Purpose
-------
This script computes the 2-point correlation function (2PCF) using PyHermes.

It reads the task configuration from a YAML file, builds the corresponding
`Corr_2PCF` task, and runs it.

Input / output
--------------
The configuration file used here is:
    ./param_quijote_2pcf.yaml

Usage
-----
1. Edit the YAML config file if needed:
       ./param_quijote_2pcf.yaml

2. Run:
       python run_2pcf.py

MPI usage
---------
To run with MPI, use:
       mpirun -np <NPROC> python run_2pcf.py

For example:
       mpirun -np 8 python run_2pcf.py

Notes
-----
- `read_param(...)` is only executed on rank 0 in PyHermes' parameter workflow.
  If you want to modify parameters directly inside this Python script when using MPI,
  do it only on rank 0.

- The commented block below is an example of this pattern:
      if MPI.COMM_WORLD.Get_rank() == 0:
          corr2pcf_params['Corr_2PCF']['n_r'] = 20
          corr2pcf_params['Corr_2PCF']['fout_path'] = "./quijote_J8_2pcf_num20.pkl"

  This means:
  * only rank 0 changes the parameter dictionary
  * the updated parameters will then be used consistently in the MPI run
"""

from pyhermes.theory.corr2pcf import Corr_2PCF
from pyhermes.param.parambase import read_param

corr2pcf_params = read_param(config_path='./param_quijote_2pcf.yaml')

# Example: modify parameters only on rank 0 when running with MPI
# from pyhermes.utils.mpi_util import MPI
# if MPI.COMM_WORLD.Get_rank() == 0:
#     corr2pcf_params['Corr_2PCF']['n_r'] = 20
#     corr2pcf_params['Corr_2PCF']['fout_path'] = "./quijote_J8_2pcf_num20.pkl"

corr2pcf = Corr_2PCF(param_task=corr2pcf_params)
corr2pcf.run(overwrite=True)
