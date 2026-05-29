Window
======

``window.ipynb`` sits between field construction and the measurement notebooks.
After ``convols.ipynb`` has produced reusable ``ConvolsData`` files, this
notebook shows how PyHermes lets you manipulate fields and Fourier-space
windows directly.

What this notebook covers
-------------------------

The notebook is organized around four practical ideas:

1. ``ConvolsData`` arithmetic for derived fields
2. ``WindowFunc`` construction and application through ``D @ W``
3. ``WindowFunc`` arithmetic for composite smoothing filters
4. built-in and custom windows, including appendix k-space sketches for the
   built-in window family

This notebook focuses on ordinary field filters and window arithmetic. The same
``WindowFunc`` abstraction is reused later for 2PCF ``pair_window`` objects,
3PCF multipole filters, and field-derivative windows, but those task-specific
roles are introduced in the later notebooks and in :doc:`../../windows`.

Inputs and outputs
------------------

The examples read the field products created by ``convols.ipynb`` or
``examples/scripts/prepare_convols_data.py``:

.. code-block:: text

   examples/output/quijote8000_snap004_sfc.pkl
   examples/output/random_sfc.pkl

No new production output is required. Most cells build objects in memory so that
the algebra is visible.

``ConvolsData`` arithmetic
--------------------------

``ConvolsData`` arithmetic acts on the field coefficients stored in
``epsilon`` and returns a new ``ConvolsData`` object:

.. code-block:: python

   from pyhermes.io import ConvolsData

   D = ConvolsData(data_path="./output/quijote8000_snap004_sfc.pkl", threads=8)
   R = ConvolsData(data_path="./output/random_sfc.pkl", threads=8)
   rho = D.field_mean_density(value_unit="grid")

   delta_from_random = D - R
   delta_from_uniform = (D - rho) / rho
   mean_field = (D + R) / 2.0

Supported field operations include addition, subtraction, and multiplication.
Supported scalar operations include addition, subtraction, multiplication, and
division. Ordinary ``D`` and ``R`` fields already use catalogue-normalized
weights by default. Any arithmetic or window convolution returns a derived
field: the grid values are kept, while catalogue metadata such as
``catalog_weight_sum``, ``raw_field_weighted_sum`` and
``weight_normalization`` are cleared. The derived field still records
``field_integral=sum(epsilon)``, so ``field_mean_density(value_unit="grid")`` and
``field_mean_density(value_unit="physical")`` remain available. A correlation task uses
``weight_normalization: field`` only when a positive marked field should be
reduced to unit integral; signed physical fields usually use
``weight_normalization: raw``.

``WindowFunc`` construction
---------------------------

A smoothing window is described by a compact dictionary and the ``convols_info``
of the field it will act on:

.. code-block:: python

   from pyhermes.io import WindowFunc

   win_params = {"type": "sphere", "len_args": {"R": 20.0}}
   W_sphere20 = WindowFunc(win_params, D.convols_info, threads=8)

   D_smooth = D @ W_sphere20

The kernel is built lazily when the window is used by ``D @ W`` or when
``W.as_array()`` is called.

``WindowFunc`` arithmetic
-------------------------

``WindowFunc`` arithmetic acts on ``w_kernel``. Composite windows are
materialized: the resulting object stores the combined kernel directly.

.. code-block:: python

   W_mix = 0.7 * W_sphere20 + 0.3 * W_gaussian8
   D_mix = D @ W_mix

Supported operations are ``W1 + W2``, ``W1 - W2``, ``a * W``, ``W * a``,
``W / a``, and ``-W``. Both windows in a binary operation must be built for the
same grid and wavelet setup.

Built-in window examples
------------------------

The built-in window dictionaries used in this notebook follow the same compact
shape:

.. code-block:: python

   {"type": "sphere", "len_args": {"R": 20.0}}
   {"type": "gaussian", "len_args": {"R": 8.0}}
   {"type": "shell", "len_args": {"R": 20.0}}
   {"type": "gaussian_shell", "len_args": {"R_shell": 20.0, "R_smooth": 5.0}}
   {"type": "cubic", "len_args": {"Lx": 20.0, "Ly": 20.0, "Lz": 20.0}}
   {"type": "ring", "len_args": {"R": 20.0, "H": 10.0}}
   {"type": "disk", "len_args": {"R": 20.0, "H": 10.0}}
   {"type": "cylshell", "len_args": {"R": 20.0, "H": 10.0}}
   {"type": "cylinder", "len_args": {"R": 20.0, "H": 10.0}}
   {"type": "cw", "len_args": {"R": 8.0}}
   {"type": "cws", "len_args": {"R": 8.0}}
   {"type": "gdw", "len_args": {"R": 8.0}}
   {"type": "directional_derivative", "los_args": {"nx": 0.0, "ny": 0.0, "nz": 1.0}}
   {"type": "laplacian"}

For line-of-sight windows, add ``los_args`` when the default z-axis line of
sight is not the intended direction:

.. code-block:: python

   {
       "type": "cylshell",
       "len_args": {"R": 20.0, "H": 10.0},
       "los_args": {"nx": 1.0, "ny": 1.0, "nz": 1.0},
       "kernel_mode": "full_rfft",
   }

Here ``H`` is a distance along the line of sight. For ``cylshell`` and
``cylinder``, it is the half-height of the finite cylinder.

The appendix cells plot the Fourier-space responses of all fourteen built-in
windows. Those plots are useful when choosing whether a window acts as a
low-pass smoother, a pair-geometry selector, a high-pass filter, or a
differential operator.

Custom and composite windows
----------------------------

Custom windows are Numba functions for the Fourier-space kernel. Length
parameters belong in ``len_args`` so PyHermes can rescale them consistently with
the field grid:

.. code-block:: python

   import numpy as np
   from numba import njit

   @njit
   def window_function_cos_shell_numba(ki, kj, kk, R, amplitude=1.0):
       k = (ki * ki + kj * kj + kk * kk) ** 0.5
       q = 2.0 * np.pi * k * R
       return amplitude if q == 0.0 else amplitude * np.sin(q) / q

   W_cos_shell = WindowFunc(
       {
           "type": "cos_shell",
           "func": window_function_cos_shell_numba,
           "len_args": {"R": 20.0},
           "other_args": {"amplitude": 1.0},
       },
       D.convols_info,
       threads=8,
   )

The notebook also shows how to combine existing windows into new definitions,
for example a finite-thickness spherical shell from two spheres, or a cylinder
surface from ``cylshell`` and ``disk``. The important rule is to combine
normalized kernels with the correct volume, area, or line-of-sight weights when
the result should remain normalized.

What to carry forward
---------------------

Use this notebook when you want to reason about what a window does before using
it in a task. ``Counting`` uses the same ordinary smoothing-window role.
``Corr_2PCF`` additionally uses windows as ``pair_window`` objects, where the
window selects a separation bin instead of smoothing the input field.
``weighted_fields.ipynb`` uses the same composition rules with derivative
windows to compute gradients, divergence, and curl.
