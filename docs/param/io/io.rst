Input and Output (IO)
=====================

This section describes how PyHermes reads particle data and how task outputs are
stored on disk.

Particle input formats
----------------------

PyHermes dispatches particle readers based on ``fin.format``. Supported values
are:

- ``bin``: raw binary table with configurable column mappings
- ``npz``: NumPy ``.npz`` particle dataset
- ``gadget``: legacy Gadget binary snapshots
- ``gadget_hdf5``: single or split Gadget HDF5 snapshots
- ``gadget-fof``
- ``fof``: Quijote/Pylians-style FoF ``group_tab`` halo catalogs

Binary table format
-------------------

``bin`` reads a raw binary table with ``reader_params``:

.. code-block:: yaml

   fin:
      path: "./data/halo.bin"
      format: "bin"
      reader_params:
         dtype: "float32"
         ncols: 7
         pos_cols: [0, 1, 2]
         fields:
            vel_x: 3
            vel_y: 4
            vel_z: 5
            mass: 6
      catalog_weight_key: null
      field_value_key: "mass"

Scalar field mappings return one-dimensional arrays. List mappings return
two-dimensional arrays. ``catalog_weight_key`` selects observational or
selection weights :math:`w_g`, while ``field_value_key`` selects the measured
per-object quantity :math:`x`; both must refer to one-dimensional fields. Use
``null`` for unit values.

NPZ format
----------

``npz`` reads an existing NumPy particle dataset:

.. code-block:: yaml

   fin:
      path: "./data/quijote_particles.npz"
      format: "npz"
      reader_params:
         pos_key: "pos"
         fields:
            completeness: "weight"
      catalog_weight_key: "completeness"
      field_value_key: null

Gadget HDF5 format
------------------

``gadget_hdf5`` accepts either one HDF5 file or the common base path of a
split snapshot. For example, ``snap_004`` resolves files named
``snap_004.0.hdf5``, ``snap_004.1.hdf5``, and so on. Positions are read by
default; velocity and mass arrays are optional to avoid unnecessary memory use.

.. code-block:: yaml

   fin:
      path: "/path/to/snapdir_004/snap_004"
      format: "gadget_hdf5"
      reader_params:
         ptype: 1
         position_scale: 1.0e-3  # Quijote kpc/h -> Mpc/h
         load_velocity: false
         load_mass: false
      catalog_weight_key: null
      field_value_key: null

Set ``load_velocity: true`` to expose ``vel``, ``vel_x``, ``vel_y``, and
``vel_z``. Set ``load_mass: true`` to read ``Masses`` or the corresponding
constant value from the Header ``MassTable``. The optional ``velocity_scale``
and ``mass_scale`` parameters provide explicit unit conversion.

FoF format
----------

``fof`` reads local Quijote/Pylians-style ``group_tab`` halo catalogs directly.
By default it returns ``pos``, ``mass``, ``vel``, ``vel_x``, ``vel_y``,
``vel_z``, ``npart``, and ``group_offset``. Use ``fields`` to select or rename
optional fields; ``pos`` and ``size`` are always retained.

.. code-block:: yaml

   fin:
      path: "./tests/data/halos/8000"
      format: "fof"
      reader_params:
         snapnum: 4
         redshift: 0.0
         fields:
            mass: "mass"
            vel_x: "vel_x"
      catalog_weight_key: null
      field_value_key: "vel_x"

The projected coefficient field uses
:math:`w_g x / Z`, where ``weight_normalization`` chooses
``Z = catalog_weight_sum`` (``catalog``), ``Z = 1`` (``raw``), or
``Z = raw_field_weighted_sum`` (``field``). ``unit`` is accepted here as an
alias for ``field`` when constructing a catalog field. Keeping catalogue
weights and physical field values separate means that the usual ``catalog``
convention is insensitive to an arbitrary global rescaling of the catalogue
weight, while PyHermes still retains ``catalog_weight_sum``,
``catalog_weight_sq_sum``, ``raw_field_weighted_sum`` and ``field_integral``.
For example, use
``field_value_key: "mass"`` for a mass-valued field and
``field_value_key: "vel_x"`` for a signed velocity-weighted field.
After field arithmetic or window convolution the result is a derived field:
``weight_normalization`` becomes ``None`` and ``field_integral`` is the direct
sum of the derived ``epsilon`` grid.

Task outputs
------------

PyHermes writes serialized task outputs as pickle-based ``.pkl`` files. Common
examples are:

- ``SFCField``
- ``CountingData``
- ``Corr2PCFData``
- ``Corr3PCFData``

Most workflows reuse the ``SFCProjection`` output file as the main upstream input for
later tasks.

Relevant fields
---------------

- ``fin.path``: local path of the particle catalog
- ``fin.format``: declared input format; when omitted or ``null``, PyHermes
  infers simple file formats from the suffix, such as ``.bin`` or ``.npz``
- ``fin.reader_params``: format-specific reader options
- ``fin.catalog_weight_key``: optional one-dimensional selection-weight selector; ``null`` means unit catalogue weights
- ``fin.field_value_key``: optional one-dimensional physical-field selector; ``null`` means unit field values
- ``fout_path``: output file path for the current task
