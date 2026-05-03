Corr_2PCF
=========

``Corr_2PCF`` measures two-point pair statistics from one or two prepared
fields. The interface is organized around two dictionaries:

- ``pair_window`` defines the Fourier-space pair kernel.
- ``sampling`` defines the sampling coordinates consumed by the pair-window
  mapping.

There is no task-level ``mode`` or task-level ``los``. The sampling coordinate
names determine the output shape, and line-of-sight information belongs inside
``pair_window.los_args``.

Pair Window
-----------

``pair_window`` may be a dictionary or one of the built-in strings
``"shell"``, ``"ring"``, ``"disk"``, or ``"cylinder"``.

A pair-window dictionary has these fields:

.. code-block:: yaml

   pair_window:
      type: "disk"
      len_args: ["R", "H"]
      los_args: [0, 0, 1]
      other_args: {}
      mapping: "smu_to_RH"
      kernel_mode: "auto"

``len_args`` contains length-like window-function arguments. Entries may be a
list of names or a dictionary; ``null`` values are filled by ``mapping`` at
runtime. ``los_args`` contains the line-of-sight vector ``nx``, ``ny``, and
``nz`` as either a three-element array or dictionary. ``other_args`` is reserved
for any window-function arguments other than ``kx``, ``ky``, ``kz``,
``len_args``, and ``los_args``.

Only ``null`` length arguments are overwritten by the mapping. Numeric
``len_args`` values are treated as user-fixed window parameters. For example,
with ``mapping: "smu_to_RH"``, ``len_args: {R: 20.0, H: null}`` keeps
``R = 20.0`` for every sampling point and fills only ``H = s * mu``. If both
``R`` and ``H`` are numeric, the same pair window is evaluated at every
``sampling`` grid point.

Built-in LOS-aware pair windows, ``ring``, ``disk``, and ``cylinder``, default
to ``los_args: [0, 0, 1]`` when ``los_args`` is omitted. ``shell`` does not
receive LOS arguments.

Built-in pair-window dictionaries may omit fields that can be inferred from
``type``. For example, ``type: "ring"`` defaults to ``len_args: ["R", "H"]``
and ``mapping: "smu_to_RH"``. Built-in string shortcuts expand the same way:

.. code-block:: yaml

   pair_window: "shell"     # xi(s): type=shell, mapping=s_to_R
   # pair_window: "disk"    # xi(s, mu): type=disk, mapping=smu_to_RH, los_args=[0, 0, 1]

Sampling And Mapping
--------------------

``sampling`` is a dictionary keyed by coordinate name. Each coordinate accepts
either a config dictionary with ``min``, ``max``, and ``n``, or an explicit
1D array.

.. code-block:: yaml

   sampling:
      s:
         min: 1.0
         max: 150.0
         n: 20
      mu:
         min: 0.0
         max: 1.0
         n: 20

The built-in mappings are:

- ``s_to_R``: requires ``sampling.s`` and fills ``len_args.R = s``.
- ``smu_to_RH``: requires ``sampling.s`` and ``sampling.mu``; fills
  ``R = s * sqrt(1 - mu^2)`` and ``H = s * mu``.
- ``rppi_to_RH``: requires ``sampling.rp`` and ``sampling.pi``; fills
  ``R = rp`` and ``H = pi``.

The keys in ``sampling`` must match the mapping exactly. For ``xi(s)``, use a
shell pair window with ``mapping: "s_to_R"`` and provide only ``sampling.s``.
For ``xi(s, mu)``, use a LOS-aware pair window with ``mapping: "smu_to_RH"``
and provide ``sampling.s`` and ``sampling.mu``. For ``xi(rp, pi)``, use
``mapping: "rppi_to_RH"`` and provide ``sampling.rp`` and ``sampling.pi``.
``s``, ``rp``, and ``pi`` are treated as non-negative separations and may
include zero.

Kernel Mode
-----------

``pair_window.kernel_mode`` controls how the window kernel is built:

- ``full_rfft`` builds the full real-FFT kernel.
- ``octant`` uses symmetry folding.
- ``auto`` uses folding only when the LOS is aligned with a coordinate axis.

Built-in ``ring``, ``disk``, and ``cylinder`` windows default to ``auto``.
``shell`` defaults to ``octant``. Custom windows default to ``full_rfft`` unless
``kernel_mode`` is specified.

Use ``octant`` only when the k-space window is invariant under independent sign
flips of ``kx``, ``ky``, and ``kz``. Oblique LOS windows usually require
``full_rfft``.

Workflow A. Command-Line Driver
-------------------------------

Use the shipped config:

.. code-block:: yaml

   Corr_2PCF:
      convols_data: "./output/quijote_sfc.pkl"
      random: "uniform"
      fout_path: "./output/quijote_2pcf.pkl"
      pair_window:
         type: "shell"
         len_args: ["R"]
         los_args: {}
         other_args: {}
         mapping: "s_to_R"
         kernel_mode: "octant"
      sampling:
         s:
            min: 1.0
            max: 150.0
            n: 30
      products: "xi"

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

Workflow C. Task Object Overrides
---------------------------------

.. code-block:: python

   from pyhermes.theory.corr2pcf import Corr_2PCF

   corr2pcf_task = Corr_2PCF()
   corr2pcf_task.threads = 8
   corr2pcf_task.sampling = {
       "s": {"min": 1.0, "max": 200.0, "n": 40},
   }
   corr2pcf_task.prepare_input_fields()
   corr2pcf = corr2pcf_task.run(save_result=False)

Workflow D. Manual Input Objects
--------------------------------

.. code-block:: python

   import numpy as np
   from numba import njit
   from pyhermes.io import ConvolsData, WindowFunc
   from pyhermes.theory.corr2pcf import Corr_2PCF

   D = ConvolsData(data_path="./output/quijote_sfc.pkl")
   filter_sph20 = WindowFunc({"type": "sphere", "len_args": {"R": 20}}, D.convols_info)

   @njit
   def window_function_cosine_numba(ki, kj, kk, R):
       k = (ki**2 + kj**2 + kk**2) ** 0.5
       return np.cos(2 * np.pi * k * R)

   pair_win_params = {
       "func": window_function_cosine_numba,
       "len_args": {"R": None},
       "los_args": {},
       "other_args": {},
       "mapping": "s_to_R",
   }

   corr2pcf_task = Corr_2PCF()
   corr2pcf_task.threads = 8
   corr2pcf_task.sampling = {"s": {"min": 1.0, "max": 150.0, "n": 40}}
   corr2pcf_task.convols_data1 = D.copy()
   corr2pcf_task.convols_data2 = D.copy()
   corr2pcf_task.window1 = filter_sph20
   corr2pcf_task.window2 = filter_sph20
   corr2pcf_task.pair_window = pair_win_params
   corr2pcf_task.prepare_input_fields()
   corr2pcf = corr2pcf_task.run(save_result=False)

Low-Level Building Block
------------------------

At the lowest level, call ``compute_pair_product_at_sample`` directly with a
sample dictionary:

.. code-block:: python

   from pyhermes.theory.corr2pcf import compute_pair_product_at_sample

   value = compute_pair_product_at_sample(
       {"s": 20.0, "mu": 0.5},
       field1,
       field2,
       pair_window="disk",
   )

Output
------

The standard saved output is:

.. code-block:: text

   ./output/quijote_2pcf.pkl

Key Parameters
--------------

- ``convols_data``, ``convols_data1``, ``convols_data2``: input field paths or
  prepared ``ConvolsData`` objects.
- ``random``, ``random1``, ``random2``: random field inputs; use ``"uniform"``
  for a uniform random density shortcut.
- ``window``, ``window1``, ``window2``: optional smoothing windows for input
  fields.
- ``pair_window``: pair-correlation kernel template dictionary or built-in
  string.
- ``sampling``: coordinate dictionary consumed by ``pair_window.mapping``.
  Saved ``Corr2PCFData`` objects expose coordinates through this dictionary,
  for example ``corr2pcf.sampling["s"]``, ``corr2pcf.sampling["mu"]``,
  ``corr2pcf.sampling["rp"]``, or ``corr2pcf.sampling["pi"]``.
- ``products``: one or more products from ``dd``, ``dr``, ``rd``,
  ``delta_dd``, ``rr``, and ``xi``.
- ``threads``: CPU threads per MPI rank.

Notes
-----

- ``prepare_input_fields()`` handles field loading, compatibility checking, and
  optional smoothing.
- ``run()`` evaluates the requested products across the sampling grid.
- When using MPI, modify config values in Python only on rank 0.
