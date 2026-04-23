Corr_3PCF
=========

``Corr_3PCF`` measures the three-point correlation function for triangles
defined by two side lengths ``r12`` and ``r13`` and an angular grid in
``theta``.

The field preparation step and the center-sampling step are intentionally split:

- ``prepare_input_fields()`` prepares the three legs of the correlation
- ``run()`` handles center generation or particle-center sampling and performs
  the actual estimator

Workflow Ladder
---------------

- **Workflow A. Command-Line Driver**
- **Workflow B. Config-Driven Python API**
- **Workflow C. Task Object with Attribute Overrides**
- **Workflow D. Manual Input Objects and Custom Preparation**
- **Workflow E. Low-Level Building Blocks**

Workflow A. Command-Line Driver
-------------------------------

Use the shipped config:

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
      center: "box_random"
      field_mode: "delta"
      n_box_centers: 10000000
      base_seed: 42

Then run:

.. code-block:: bash

   python run_3pcf.py

or with MPI:

.. code-block:: bash

   mpirun -np 4 python run_3pcf.py ./configs/param_3pcf.yaml

Workflow B. Config-Driven Python API
------------------------------------

.. code-block:: python

   from pyhermes.param.parambase import read_param
   from pyhermes.theory.corr3pcf import Corr_3PCF

   params = read_param("./configs/param_3pcf.yaml")
   corr3pcf_task = Corr_3PCF(param_task=params)
   corr3pcf = corr3pcf_task.run(overwrite=True)

Workflow C. Task Object with Attribute Overrides
------------------------------------------------

.. code-block:: python

   from pyhermes.theory.corr3pcf import Corr_3PCF

   corr3pcf_task = Corr_3PCF()
   corr3pcf_task.threads = 8
   corr3pcf_task.center = "particle"
   corr3pcf_task.n_theta = 20
   corr3pcf_task.n_rot = 20
   corr3pcf_task.prepare_input_fields()
   corr3pcf = corr3pcf_task.run(save_result=False)

Workflow D. Manual Input Objects and Custom Preparation
-------------------------------------------------------

This layer gives explicit control over the three legs:

.. code-block:: python

   from pyhermes.io import ConvolsData, WindowFunc
   from pyhermes.theory.corr3pcf import Corr_3PCF

   D = ConvolsData(data_path="./output/quijote_sfc.pkl")
   win_params = {"type": "sphere", "len_args": {"R": 5}}
   filter_sph5 = WindowFunc(win_params, D.convols_info)

   corr3pcf_task = Corr_3PCF()
   corr3pcf_task.threads = 8
   corr3pcf_task.center = "particle"
   corr3pcf_task.convols_data1 = D.copy()
   corr3pcf_task.convols_data2 = D.copy()
   corr3pcf_task.convols_data3 = D.copy()
   corr3pcf_task.window2 = filter_sph5
   corr3pcf_task.window3 = filter_sph5
   corr3pcf_task.prepare_input_fields()
   corr3pcf = corr3pcf_task.run(save_result=False)

Workflow E. Low-Level Building Blocks
-------------------------------------

At the lowest level, you can work directly with prepared fields, explicit
center positions, and the low-level Monte Carlo kernels in
``pyhermes.utils.corr3pcf_kernels``. This is the most flexible route, but it
assumes you want to manage normalization, windows, and estimator bookkeeping
yourself.

Output
------

The standard output is:

.. code-block:: text

   ./output/quijote_3pcf_rand1e7.pkl

Key parameters
--------------

- ``convols_data_path``:
  shared fallback input field path
- ``convols_data1_path`` / ``convols_data2_path`` / ``convols_data3_path``:
  optional leg-specific field paths
- ``window``, ``window1``, ``window2``, ``window3``:
  optional smoothing windows for the three legs
- ``r12`` and ``r13``:
  triangle side lengths
- ``n_theta``:
  number of angular bins
- ``n_rot``:
  number of rotations used by the estimator
- ``center``:
  ``"box_random"`` or ``"particle"``
- ``field_mode``:
  ``"raw"`` or ``"delta"``
- ``n_box_centers``:
  number of random centers when ``center="box_random"``
- ``base_seed``:
  random seed

Notes
-----

- ``prepare_input_fields()`` prepares the three fields and checks that they are
  compatible.
- ``run()`` keeps the center handling in the runtime stage, because center
  generation depends strongly on MPI rank count and execution mode.
- When using MPI, modify config values in Python only on rank 0.
