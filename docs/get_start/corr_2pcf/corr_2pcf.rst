Corr_2PCF
=========

``Corr_2PCF`` measures the two-point correlation function from a saved
multiresolution field.

Example configuration
---------------------

The repository includes ``examples/configs/param_2pcf.yaml``:

.. code-block:: yaml

   Corr_2PCF:
      convols_data_path: "./output/quijote_sfc.pkl"
      fout_path: "./output/quijote_2pcf.pkl"
      r_min: 1.0
      r_max: 150.0
      n_r: 30
      field_mode: "delta"

Minimal Python driver
---------------------

.. code-block:: python

   from pyhermes.theory.corr2pcf import Corr_2PCF
   from pyhermes.param.parambase import read_param

   corr2pcf_params = read_param(config_path="./configs/param_2pcf.yaml")
   corr2pcf = Corr_2PCF(param_task=corr2pcf_params)
   corr2pcf.run(overwrite=True)

If you want to modify parameters directly inside the Python driver when using
MPI, do it only on rank 0. For example:

.. code-block:: python

   from pyhermes.utils.mpi_util import MPI

   if MPI.COMM_WORLD.Get_rank() == 0:
       corr2pcf_params["Corr_2PCF"]["n_r"] = 20
       corr2pcf_params["Corr_2PCF"]["fout_path"] = "./output/quijote_2pcf_num20.pkl"

Run it
------

From the ``examples`` directory:

.. code-block:: bash

   python run_2pcf.py

Or with MPI:

.. code-block:: bash

   mpirun -np 8 python run_2pcf.py

Output
------

The example writes:

.. code-block:: text

   ./output/quijote_2pcf.pkl

Key parameters
--------------

- ``convols_data_path``: path to the saved multiresolution field
- ``fout_path``: output path for the 2PCF result
- ``r_min``: minimum separation
- ``r_max``: maximum separation
- ``n_r``: number of radial bins
- ``field_mode``: choose ``"delta"`` to compute with ``D-R`` and save ``delta_dd``, or ``"raw"`` to compute with ``D`` and save ``dd``
- ``threads``: number of threads per MPI rank

Notes
-----

You can also provide explicit window definitions through ``window``,
``window1``, or ``window2`` if you need custom smoothing behavior. For most
basic runs, the radial settings shown above are the main controls.
