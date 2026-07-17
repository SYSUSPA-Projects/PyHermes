Calculation parameters
======================

This page collects the parameters that change the mathematical measurement.
File readers and output objects are covered in :doc:`../io/io`; resource
choices are covered in :doc:`../perform/perform`.

Common window schema
--------------------

.. code-block:: yaml

   window:
      type: "gaussian"
      len_args:
         R: 10.0
      los_args: {}
      other_args: {}
      kernel_mode: "auto"

``len_args`` contains physical length scales, ``los_args`` contains an optional
direction, and ``other_args`` contains non-length options. ``kernel_mode`` is
needed only when overriding kernel construction. See :doc:`../../windows` for
the built-in types and their exact arguments.

SFCProjection
-------------

``J`` and ``box_size`` set the grid geometry. ``wavelet_mode``,
``wavelet_level``, and ``phi_resolution`` define the MRA basis. The input
amplitude is ``catalog_weight * field_value`` and ``weight_normalization`` is
one of ``catalog``, ``raw``, ``field``, or the construction-time alias ``unit``.

All fields combined by arithmetic or window convolution must share compatible
box geometry, ``J``, and basis metadata.

Counting
--------

``random_count`` positions are drawn from the periodic box using ``seed``.
``window`` optionally filters the input field before evaluating it. The result
stores positions and sampled values; it does not histogram them, leaving PDF
binning to the analysis code.

Corr_2PCF
---------

``sfc_field`` and ``random`` are shared fallbacks for both vertices. The numbered
forms override individual vertices. ``random: uniform`` is an analytic constant
field.

``window1`` and ``window2`` smooth or otherwise transform the input vertices.
``binning_window`` defines the separation measurement. These are distinct
roles: a Gaussian vertex window changes the fields being correlated; a
``gaussian_shell`` binning window changes how separations are averaged.

Built-in sampling mappings are:

.. list-table:: 2PCF sampling mappings
   :header-rows: 1
   :widths: 28 27 45

   * - Mapping
     - Coordinates
     - Runtime window arguments
   * - ``s_to_R``
     - ``s``
     - ``R=s`` for isotropic radial windows.
   * - ``smu_to_RH``
     - ``s``, ``mu``
     - ``R=s sqrt(1-mu^2)``, ``H=s mu``.
   * - ``rppi_to_RH``
     - ``rp``, ``pi``
     - ``R=rp``, ``H=pi``.

Set ``products`` to any subset of ``dd``, ``dr``, ``rd``, ``rr``, and ``xi``;
dependencies are expanded automatically. ``memory_strategy=speed`` retains
reusable fields, while ``memory`` reduces simultaneous field residency.

Corr_3PCF
---------

``r12`` and ``r13`` fix two triangle sides. ``angle_param`` chooses ``theta``
or ``mu`` and the matching block supplies a range or explicit array.
``n_rot`` controls the Monte Carlo orientation average.

``center=particle`` samples the first vertex from particle metadata and applies
only ``window2`` and ``window3`` to displaced vertices. ``center=box_random`` uses
``n_box_centers`` uniform centres and permits windows on all three vertices.

Products are ``ddd``, ``rrr``, ``delta_ddd``, ``xi12``, ``xi13``, ``xi23``,
``zeta``, ``zeta_H``, and ``Q``. Particle-centred runs additionally expose
``d_delta_dd`` and ``r_delta_dd``.

Corr_3PCF_Multipole
-------------------

The two radial windows are mandatory templates:

.. code-block:: yaml

   binning_window12:
      type: "shell"
      len_args: {}
      other_args: {}
      mapping: {R: "r12"}
   binning_window13:
      type: "thick_shell"
      len_args: {delta_R: 6.0}
      other_args: {}
      mapping: {R: "r13"}

``sampling`` defines the values named by those mappings. Scalar, list,
``values``, ``min/max/n``, and ``start/stop/step`` forms are accepted.
``mode=grid`` forms a Cartesian product; ``mode=paired`` zips arrays and
broadcasts scalars.

``l_min`` and ``l_max`` select multipole orders. The products are ``ddd_l``,
``rrr_l``, ``delta_ddd_l``, and ``zeta_l``. ``window1`` through ``window3`` are
optional vertex smoothing operators and are separate from the two radial
binning profiles.

Radial-profile parameters and custom-profile interfaces are documented in
:doc:`../../get_start/corr_3pcf_multipole/corr_3pcf_multipole`.
