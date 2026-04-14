Corr_3PCF_Multipole
===================

``Corr_3PCF_Multipole`` measures the multipole coefficients
``zeta_l(r12, r13)`` from one or more prepared fields.

As in ``Corr_3PCF``, field preparation and the main estimator are separated:

- ``prepare_input_fields()`` prepares the three input legs
- ``run()`` executes either the serial or grouped MPI multipole workflow

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

   Corr_3PCF_Multipole:
      convols_data_path: "./output/quijote_sfc.pkl"
      fout_path: "./output/quijote_3pcf_multipole.pkl"
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
      l_min: 0
      l_max: 4
      gpu_device_id: 0
      field_mode: "delta"
      execution_mode: "serial"
      threads: 1

Then run:

.. code-block:: bash

   python run_3pcf_multipole.py

Workflow B. Config-Driven Python API
------------------------------------

.. code-block:: python

   from pyhermes.param.parambase import read_param
   from pyhermes.theory.corr3pcf_multipole import Corr_3PCF_Multipole

   params = read_param("./configs/param_3pcf_multipole.yaml")
   task = Corr_3PCF_Multipole(param_task=params)
   result = task.run(overwrite=True)

Workflow C. Task Object with Attribute Overrides
------------------------------------------------

.. code-block:: python

   from pyhermes.theory.corr3pcf_multipole import Corr_3PCF_Multipole

   task = Corr_3PCF_Multipole()
   task.execution_mode = "pair_mpi"
   task.l_max = 7
   task.threads = 1
   task.prepare_input_fields()
   result = task.run(save_result=False)

Workflow D. Manual Input Objects and Custom Preparation
-------------------------------------------------------

This layer gives you explicit control over the three fields and optional
smoothing windows:

.. code-block:: python

   from pyhermes.io import ConvolsData, WindowFunc
   from pyhermes.theory.corr3pcf_multipole import Corr_3PCF_Multipole

   D = ConvolsData(data_path="./output/quijote_sfc.pkl")
   win_params = {"type": "sphere", "len_args": {"R": 5}}
   filter_sph5 = WindowFunc(win_params, D.convols_info)

   task = Corr_3PCF_Multipole()
   task.execution_mode = "pair_mpi"
   task.convols_data1 = D.copy()
   task.convols_data2 = D.copy()
   task.convols_data3 = D.copy()
   task.window2 = filter_sph5
   task.window3 = filter_sph5
   task.prepare_input_fields()
   result = task.run(save_result=False)

Workflow E. Low-Level Building Blocks
-------------------------------------

At the lowest level, you can work directly with the streamed Legendre-window
convolutions and GPU-side summation helpers used inside the official task.
This offers maximum flexibility, but it also requires the strongest familiarity
with the estimator and the normalization conventions.

Output
------

The standard output is:

.. code-block:: text

   ./output/quijote_3pcf_multipole.pkl

Key parameters
--------------

- ``convols_data_path``:
  shared fallback input field path
- ``convols_data1_path`` / ``convols_data2_path`` / ``convols_data3_path``:
  optional leg-specific field paths
- ``window``, ``window1``, ``window2``, ``window3``:
  optional smoothing windows for the three legs
- ``r12`` and ``r13``:
  side lengths of the multipole family
- ``l_min`` and ``l_max``:
  multipole range
- ``gpu_device_id``:
  CUDA device index
- ``field_mode``:
  ``"raw"`` or ``"delta"``
- ``execution_mode``:
  ``"serial"`` or ``"pair_mpi"``
- ``cache_multipole_fields`` and ``cache_dir``:
  optional intermediate cache

Notes
-----

- ``prepare_input_fields()`` prepares the three legs and checks compatibility.
- ``run()`` executes the actual multipole estimator.
- In the current implementation, ``execution_mode="pair_mpi"`` is the preferred
  high-performance mode on a single GPU node with an even number of MPI ranks.
