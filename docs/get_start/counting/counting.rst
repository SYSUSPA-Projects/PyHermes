Counting
========

``Counting`` evaluates the field on many random positions. This is useful for
sampling local densities, estimating one-point distributions, and generating
downstream summary statistics after a smoothing step.

Example configuration
---------------------

The repository includes ``examples/configs/param_counting.yaml``:

.. code-block:: yaml

   Counting:
      N_randoms: 10000000
      convols_data_path: "./output/quijote_sfc.pkl"
      fout_path: "./output/quijote_counting_sph20.pkl"
      window:
         type: sphere
         len_args:
            R: 20

Minimal Python driver
---------------------

.. code-block:: python

   from pyhermes.theory.counting import Counting
   from pyhermes.param.parambase import read_param

   counting_params = read_param(config_path="./configs/param_counting.yaml")
   counting = Counting(param_task=counting_params)
   counting.run(overwrite=True)

Run it
------

From the ``examples`` directory:

.. code-block:: bash

   python run_counting.py

Or with MPI:

.. code-block:: bash

   mpirun -np 8 python run_counting.py

Output
------

The example writes:

.. code-block:: text

   ./output/quijote_counting_sph20.pkl

Key parameters
--------------

- ``N_randoms``: number of random sampling points
- ``convols_data_path``: path to the ``Convols`` output file
- ``fout_path``: output path for counting results
- ``window.type``: smoothing window type
- ``window.len_args``: window scale parameters, for example ``R`` for a sphere
- ``threads``: number of threads per MPI rank

Notes
-----

``Counting`` requires a previously generated ``Convols`` field. Run
:doc:`../convols/convols` first if the input file does not yet exist.
