Corr_3PCF
=========

``Corr_3PCF`` measures the three-point correlation function for triangles
defined by two side lengths and an angular grid.

Example configuration
---------------------

The repository includes ``examples/configs/param_3pcf.yaml``:

.. code-block:: yaml

   Corr_3PCF:
      convols_data_path: "./output/quijote_sfc.pkl"
      fout_path: "./output/quijote_3pcf_rand1e7.pkl"
      window2:
         type: "sphere"
         len_args:
            R: 5
      window3:
         type: "sphere"
         len_args:
            R: 5
      r12: 20.0
      r13: 40.0
      n_theta: 20
      n_rot: 20
      center: "random"
      n_rand: 10000000
      base_seed: 42

Minimal Python driver
---------------------

.. code-block:: python

   from pyhermes.theory.corr3pcf import Corr_3PCF
   from pyhermes.param.parambase import read_param

   corr3pcf_params = read_param(config_path="./configs/param_3pcf.yaml")
   corr3pcf = Corr_3PCF(param_task=corr3pcf_params)
   corr3pcf.run(overwrite=True)

Run it
------

From the ``examples`` directory:

.. code-block:: bash

   python run_3pcf.py

Or with MPI:

.. code-block:: bash

   mpirun -np 8 python run_3pcf.py

Output
------

The example writes:

.. code-block:: text

   ./output/quijote_3pcf_rand1e7.pkl

Key parameters
--------------

- ``convols_data_path``: path to the saved multiresolution field
- ``fout_path``: output path for the 3PCF result
- ``r12`` and ``r13``: the two side lengths defining the triangle family
- ``n_theta``: number of angular bins
- ``n_rot``: number of rotations used in the estimator
- ``center``: center sampling mode, usually ``random`` or ``particle``
- ``n_rand``: number of random centers when ``center`` is ``random``
- ``base_seed``: random seed for reproducibility
- ``window2`` and ``window3``: smoothing windows applied to the two legs

Notes
-----

When ``center = "particle"``, PyHermes uses particle positions as triangle
centers. When ``center = "random"``, it samples centers uniformly in the box.
