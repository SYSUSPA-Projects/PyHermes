Corr_2PCF
=========

``Corr_2PCF`` measures the two-point correlation function from one or two
prepared fields.

The current interface supports both traditional shell-based 2PCF measurements
and generalized pair statistics through ``pair_window``, which defines the
kernel used inside the pair-correlation measurement itself.

``mode: "s"`` measures the isotropic ``xi(s)`` using a shell pair window by
default. ``mode: "smu"`` measures ``xi(s, mu)`` using a ring pair window by
default, where ``mu`` is the cosine of the angle between the pair separation
and the line-of-sight direction. For the default ring window, ``mu`` only needs
to cover ``0 <= mu <= 1`` because the kernel is symmetric under
``mu -> -mu``. In PyHermes these coordinates are sampling points; finite bin
widths are not represented explicitly by the current pair-window interface.

The pair window uses a ``mapping`` field to convert sampling coordinates into
window length arguments. ``mode: "s"`` defaults to ``mapping: "s_to_R"``,
which sets ``R = s``. ``mode: "smu"`` defaults to ``mapping: "smu_to_RH"``,
which sets ``R = s sqrt(1 - mu^2)`` and ``H = s mu``. Built-in mappings are
mode-specific: ``s_to_R`` is only valid with ``mode: "s"``, and
``smu_to_RH`` is only valid with ``mode: "smu"``. Custom pair windows may use
either built-in mapping or provide a callable mapping in Python.

When a pair-window dictionary is passed to ``Corr_2PCF``, ``None`` can be used
as a runtime placeholder for arguments filled by the mapping. For
``mapping: "smu_to_RH"``, ``R`` and ``H`` are filled from ``s`` and ``mu`` only
when their ``len_args`` entries are ``None``; fixed numeric values are left
unchanged. If ``other_args`` explicitly contains ``nx``, ``ny``, or ``nz`` with
value ``None``, those entries are filled from the task ``los`` vector. A
dictionary used to construct a ``WindowFunc`` directly should instead provide
fixed numeric values for all window-function arguments.

Built-in anisotropic pair windows include ``ring``, ``disk``, and ``cylinder``.
They use ``R`` and ``H`` as length arguments and optional ``nx``, ``ny``, and
``nz`` entries in ``other_args`` for the line-of-sight direction.

``pair_window.kernel_mode`` controls how the window kernel is built. Use
``full_rfft`` for the full real-FFT kernel, ``octant`` for symmetry folding, or
``auto`` to fold only when the LOS is aligned with a coordinate axis. Built-in
``ring``, ``disk``, and ``cylinder`` windows default to ``auto``. Custom
windows default to ``full_rfft`` unless ``kernel_mode`` is specified.

Use ``octant`` only when the k-space window is invariant under independent
sign flips of ``kx``, ``ky``, and ``kz``:
``W(kx, ky, kz) = W(-kx, ky, kz) = W(kx, -ky, kz) = W(kx, ky, -kz)``.
Isotropic windows satisfy this automatically. Axis-aligned LOS windows may
also satisfy it when their parallel dependence is even, such as ``cos`` or
``sin(q)/q`` factors. Oblique LOS windows usually do not satisfy this symmetry,
so ``full_rfft`` is the safer choice for custom anisotropic windows. The
criterion is tied to the FFT grid coordinates rather than visual geometric
symmetry: a ring with ``los: [1, 1, 0]`` can be symmetric in a rotated
coordinate system while still failing the independent ``kx`` and ``ky`` sign
flip tests required by ``octant``.

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
      mode: "s"
      s:
         s_min: 1.0
         s_max: 150.0
         n_s: 30
      pair_window:
         type: "shell"
         len_args:
            R: null
         other_args: {}
         mapping: "s_to_R"

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
   corr2pcf_task.s = {"s_min": 1.0, "s_max": 200.0, "n_s": 40}
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

   pair_win_params = {
       "func": window_function_cosine_numba,
       "len_args": {"R": None},
       "mapping": "s_to_R",
   }

   corr2pcf_task = Corr_2PCF()
   corr2pcf_task.threads = 8
   corr2pcf_task.s = {"s_min": 1.0, "s_max": 150.0, "n_s": 40}
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
and ``compute_pair_product_at_smu`` to build custom pair statistics. This is the most
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
- ``mode``:
  ``"s"`` for isotropic ``xi(s)`` or ``"smu"`` for ``xi(s, mu)``
- ``s``:
  separation sampling controls, either a dict with ``s_min``, ``s_max``, and
  ``n_s`` or an explicit 1D sampling array
- ``mu``:
  angular sampling controls used only in ``mode: "smu"``
- ``los``:
  line-of-sight direction for ``mode: "smu"``; use ``"x"``, ``"y"``,
  ``"z"``, or a length-3 vector
- ``window``, ``window1``, ``window2``:
  optional smoothing windows for the input fields
- ``pair_window``:
  pair-correlation kernel template; by default this is a shell window for
  ``mode: "s"`` and a ring window for ``mode: "smu"``
- ``threads``:
  CPU threads per MPI rank

Notes
-----

- ``prepare_input_fields()`` handles field loading, compatibility checking, and
  optional smoothing.
- ``run()`` evaluates the pair statistic across the requested sampling grid.
- When using MPI, modify config values in Python only on rank 0.
