"""
Example: run counting / random-point sampling with PyHermes

Purpose
-------
This script evaluates the field on a large set of random points and saves
the resulting counting data.

Starting from a precomputed multiresolution coefficient file (`epsilon`),
PyHermes first applies the specified window function and then samples the
smoothed field at many random positions. This is typically used for:
- estimating the one-point PDF of the field
- studying the distribution of local densities
- preparing intermediate data products for later statistical analysis

Input / output
--------------
The configuration file used here is:
    ./configs/param_counting.yaml

In the current example:
- number of random points:
    N_randoms = 10000000
- input coefficient file:
    ./output/quijote_sfc.pkl
- output counting file:
    ./output/quijote_counting_sph20.pkl

Usage
-----
1. Make sure the input coefficient file already exists:
       ./output/quijote_sfc.pkl

2. Edit the YAML config file if needed:
       ./configs/param_counting.yaml

3. Run:
       python run_counting.py

MPI usage
---------
To run with MPI, use:
       mpirun -np <NPROC> python run_counting.py

For example:
       mpirun -np 8 python run_counting.py

Notes
-----
- `read_param(...)` is only executed on rank 0 in PyHermes' parameter workflow.
  If you want to modify parameters directly inside this Python script when using MPI,
  do it only on rank 0.

- The output file (here `quijote_J8_counting_sph20.pkl`) can be used for
  PDF analysis or other statistics based on random-point sampling of the field.
"""

from pyhermes.theory.counting import Counting
from pyhermes.param.parambase import read_param

counting_params = read_param(config_path='./configs/param_counting.yaml')

# Example: modify parameters only on rank 0 when running with MPI
# from pyhermes.utils.mpi_util import MPI
# if MPI.COMM_WORLD.Get_rank() == 0:
#     counting_params['Counting']['window'] = {}
#     counting_params['Counting']['fout_path'] = "./output/quijote_counting.pkl"

counting = Counting(param_task=counting_params)
counting.run(overwrite=True)
