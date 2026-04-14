Corr_2PCF
=========

``Corr_2PCF`` measures the two-point correlation function from one or two
prepared fields.

The current interface supports both traditional shell-based 2PCF measurements
and generalized pair statistics through ``pair_window``, which defines the
kernel used inside the pair-correlation measurement itself.

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

   Corr_2PCF:
      convols_data_path: "./output/quijote_sfc.pkl"
      fout_path: "./output/quijote_2pcf.pkl"
      r_min: 1.0
      r_max: 150.0
      n_r: 30
      field_mode: "delta"
      pair_window:
         type: "shell"
         len_args:
            R: null
         other_args: {}

Then run:

.. code-block:: bash

   python run_2pcf.py

or with MPI:

.. code-block:: bash

   mpirun -np 4 python run_2pcf.py ./configs/param_2pcf.yaml

Workflow B. Config-Driven Python API
------------------------------------

.. code-block:: python

   from pyhermes.param.parambase import read_param
   from pyhermes.theory.corr2pcf import Corr_2PCF

   params = read_param("./configs/param_2pcf.yaml")
   corr2pcf_task = Corr_2PCF(param_task=params)
   corr2pcf = corr2pcf_task.run(overwrite=True)

Workflow C. Task Object with Attribute Overrides
------------------------------------------------

.. code-block:: python

   from pyhermes.theory.corr2pcf import Corr_2PCF

   corr2pcf_task = Corr_2PCF()
   corr2pcf_task.threads = 8
   corr2pcf_task.n_r = 40
   corr2pcf_task.r_max = 200.0
   corr2pcf_task.prepare_input_fields()
   corr2pcf = corr2pcf_task.run(save_result=False)

Workflow D. Manual Input Objects and Custom Preparation
-------------------------------------------------------

This layer is useful when you want explicit control over the two legs and their
windows:

.. code-block:: python

   from numba import njit
   from pyhermes.io import ConvolsData, WindowFunc
   from pyhermes.theory.corr2pcf import Corr_2PCF

   D = ConvolsData(data_path="./output/quijote_sfc.pkl")
   win_params = {"type": "sphere", "len_args": {"R": 20}}
   filter_sph20 = WindowFunc(win_params, D.convols_info)

   @njit
   def window_function_cosine_numba(ki, kj, kk, R):
       k = (ki**2 + kj**2 + kk**2) ** 0.5
       return np.cos(2 * np.pi * k * R)

   pair_win_params = {"func": window_function_cosine_numba, "len_args": {"R": None}}

   corr2pcf_task = Corr_2PCF()
   corr2pcf_task.threads = 8
   corr2pcf_task.n_r = 40
   corr2pcf_task.convols_data1 = D.copy()
   corr2pcf_task.convols_data2 = D.copy()
   corr2pcf_task.window1 = filter_sph20
   corr2pcf_task.window2 = filter_sph20
   corr2pcf_task.pair_window = pair_win_params
   corr2pcf_task.prepare_input_fields()
   corr2pcf = corr2pcf_task.run(save_result=False)

Workflow E. Low-Level Building Blocks
-------------------------------------

At the lowest level, you can work directly with ``ConvolsData``, ``WindowFunc``,
and ``calc_DD_mean_r`` to build custom pair statistics. This is the most
flexible route, but it requires that you manage field preparation and
normalization explicitly.

Output
------

The standard output is:

.. code-block:: text

   ./output/quijote_2pcf.pkl

Key parameters
--------------

- ``convols_data_path``:
  shared fallback input field path
- ``convols_data1_path`` and ``convols_data2_path``:
  optional leg-specific input paths
- ``r_min``, ``r_max``, ``n_r``:
  radial sampling controls
- ``field_mode``:
  ``"delta"`` for ``(D-R)(D-R)/RR`` and ``"raw"`` for ``DD/RR - 1``
- ``window``, ``window1``, ``window2``:
  optional smoothing windows for the input fields
- ``pair_window``:
  pair-correlation kernel template; by default this is the standard shell
  window with runtime radius injection
- ``threads``:
  CPU threads per MPI rank

Notes
-----

- ``prepare_input_fields()`` handles field loading, compatibility checking, and
  optional smoothing.
- ``run()`` evaluates the pair statistic across the requested radial grid.
- When using MPI, modify config values in Python only on rank 0.
