"""
Example: generate multiresolution coefficients (epsilon) with PyHermes

Purpose
-------
This script performs the most fundamental PyHermes task:
it reads a particle-position field and computes the multiresolution-space
coefficients `epsilon`.

The output `epsilon` field is the starting point for later analyses, including:
- window-function convolution
- random-point counting / PDF estimation
- 2-point correlation function (2PCF)
- 3-point correlation function (3PCF)

Input / output
--------------
The configuration file used here is:
    ./param_quijote_convols.yaml

In the current example:
- input particle catalog:
    https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin
- input format:
    generic_pos
- output coefficient file:
    ./quijote_J8.pkl

Usage
-----
1. Edit the YAML config file if needed:
       ./param_quijote_convols.yaml

2. Run:
       python run_convols.py

MPI usage
---------
To run with MPI, use:
       mpirun -np <NPROC> python run_convols.py

For example:
       mpirun -np 8 python run_convols.py

Notes
-----
- `read_param(...)` is only executed on rank 0 in PyHermes' parameter workflow.
  If you want to modify parameters directly inside this Python script when using MPI,
  do it only on rank 0.

- The output file (here `quijote_J8.pkl`) can be reused by later PyHermes tasks
  such as window convolution, counting/PDF, 2PCF, and 3PCF.
"""

from pyhermes.base.convols import Convols
from pyhermes.param.parambase import read_param

convols_params = read_param(config_path='./param_quijote_convols.yaml')
convols = Convols(param_task=convols_params)
convols.run(overwrite=True)
