Field and window operations
===========================

``SFCField`` stores a represented field. ``WindowFunc`` stores an operator
compatible with that representation. Their small algebra is the low-level
language from which PyHermes tasks are assembled.

Load a field
------------

.. code-block:: python

   from pyhermes.io import SFCField, WindowFunc

   D = SFCField(
       data_path="./output/quijote8000_snap004_sfc.pkl",
       threads=8,
   )

Field arithmetic
----------------

Compatible fields support coefficient-wise arithmetic:

.. code-block:: python

   delta = D - D.field_mean_density(value_unit="grid")
   cross_sum = field_a + field_b
   local_product = field_a * field_b
   rescaled = 2.0 * field_a

``+`` and ``-`` combine fields. ``*`` between two fields forms their local
coefficient-space product; ``*`` with a scalar rescales a field. The result is
another ``SFCField`` marked as a derived field.

Apply a window
--------------

.. code-block:: python

   smooth = WindowFunc(
       {"type": "gaussian", "len_args": {"R": 10.0}},
       D.sfc_info,
       threads=8,
   )
   delta_smooth = delta @ smooth

``@`` is reserved for field--window convolution. The kernel is built lazily
the first time ``smooth.as_array()`` or ``delta @ smooth`` is called, then
cached on the object.

The two multiplications are deliberately different:

.. code-block:: python

   filtered = delta @ smooth       # convolution in real space
   quadratic = filtered * filtered # local field product

This distinction mirrors the Hermes equations: first construct
window-filtered fields, then multiply and average them.

Inspect values
--------------

``as_array()`` returns the represented coefficient array:

.. code-block:: python

   epsilon = delta_smooth.as_array()
   mean_product = quadratic.as_array().mean()

To evaluate the reconstructed continuous field at positions, use:

.. code-block:: python

   values = delta_smooth.field_density_at_pos(
       positions,
       value_unit="physical",
   )

Compatibility
-------------

Fields and windows must agree in ``J``, ``box_size``, ``wavelet_mode``,
``wavelet_level``, and ``phi_resolution``. Always construct a window from the
``sfc_info`` of the field it will act on. A visually similar array from another
grid is not interchangeable.

Window arithmetic
-----------------

Window addition, subtraction, and scalar arithmetic combine materialised
projected kernels. For example, a projected finite shell can be explored as a
volume-weighted difference of two top-hats:

.. code-block:: python

   R, width = 40.0, 6.0
   r_in, r_out = R - width / 2.0, R + width / 2.0

   W_in = WindowFunc(
       {"type": "sphere", "len_args": {"R": r_in}}, D.sfc_info
   )
   W_out = WindowFunc(
       {"type": "sphere", "len_args": {"R": r_out}}, D.sfc_info
   )
   W_shell_projected = (
       r_out**3 * W_out - r_in**3 * W_in
   ) / (r_out**3 - r_in**3)

The built-in ``thick_shell`` is preferable for production because its complete
continuous Fourier expression is projected once.

Likewise, ``W1 * W2`` is a pointwise product of two already projected kernels.
It is useful for experiments with separable factors, but it is not identical
to projecting the bare continuous product once. See :doc:`../../windows` for
the full numerical note.

A custom window
---------------

Custom ordinary kernels use the same interface:

.. code-block:: python

   import numpy as np
   from numba import njit

   @njit
   def transverse_gaussian(ki, kj, kk, R):
       q2 = (2.0 * np.pi * R) ** 2 * (ki * ki + kj * kj)
       return np.exp(-0.5 * q2)

   W_perp = WindowFunc(
       {"func": transverse_gaussian, "len_args": {"R": 5.0}},
       D.sfc_info,
       threads=8,
   )

``ki``, ``kj``, and ``kk`` are cycle frequencies in grid coordinates.
Physical lengths belong in ``len_args`` and are rescaled automatically.
If both ``type`` and ``func`` are supplied, the callable is authoritative:
``type`` remains a descriptive name, but it does not activate same-named
built-in argument defaults or validation.

Use the task layer when appropriate
-----------------------------------

Direct operations are ideal for understanding an estimator, constructing a
new physical field, or prototyping a custom statistic. For production
Counting, 2PCF, and 3PCF runs, task classes also manage data/random fields,
sampling grids, MPI distribution, dependencies among output products, and
metadata. Low-level and high-level paths use the same field/window machinery.

Public tutorial
---------------

``examples/notebooks/window.ipynb`` demonstrates field arithmetic, ordinary
windows, custom kernels, projected-kernel composition, smoothing versus
binning roles, and Fourier-space sketches of the built-in library.
