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

Minimal Python driver
---------------------

.. code-block:: python

   from pyhermes.theory.corr2pcf import Corr_2PCF
   from pyhermes.param.parambase import read_param

   corr2pcf_params = read_param(config_path="./configs/param_2pcf.yaml")
   corr2pcf = Corr_2PCF(param_task=corr2pcf_params)
   corr2pcf.run(overwrite=True)

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
- ``threads``: number of threads per MPI rank

Notes
-----

You can also provide explicit window definitions through ``window``,
``window1``, or ``window2`` if you need custom smoothing behavior. For most
basic runs, the radial settings shown above are the main controls.
