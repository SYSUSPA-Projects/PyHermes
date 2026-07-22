Input and output
================

PyHermes has two I/O layers. Particle readers turn catalogue files into a
common ``{"pos": (N, 3), ...}`` structure. Task-specific data classes read and
write reusable MRA fields and statistical products.

Particle input
--------------

``SFCProjection.fin`` selects a reader:

.. code-block:: yaml

   fin:
      path: "./data/catalog.bin"
      format: "bin"
      reader_params: {}
      catalog_weight_key: null
      field_value_key: null

``catalog_weight_key`` selects completeness or selection weights.
``field_value_key`` selects a per-particle mark or physical quantity. Both must
resolve to one-dimensional arrays with one entry per particle. ``null`` means
unit values.

Remote files and caching
------------------------

``fin.path`` may be a local path or an HTTP(S) URL. Remote inputs are resolved
to a local file before the format-specific reader runs:

.. code-block:: yaml

   fin:
      path: "https://data.example.org/haloes.npz"
      format: "npz"
      download:
         cache_path: "./data/haloes.npz"
         sha256: "0123456789abcdef..."
         timeout: 60

``cache_path`` gives one exact destination. Alternatively, ``cache_dir`` puts
the file in that directory under a deterministic URL-derived name. With
neither option, PyHermes uses ``$PYHERMES_CACHE_DIR``, then
``$XDG_CACHE_HOME/pyhermes``, and finally ``~/.cache/pyhermes``.

An existing cache file is reused after optional SHA256 verification. New data
are downloaded to a unique partial file and atomically renamed only after a
successful transfer and checksum. Under MPI, ``SFCProjection`` performs this
work only on rank 0 before distributing particle slabs. Remote input currently
targets single files such as NPZ or BIN; directory catalogues and split Gadget
snapshots should remain on local or shared storage.

The tracked
``examples/data/quijote_halos_8000_snap004_schema.yaml`` records the source,
arrays, units, object count, byte size, URL, and checksum of the Quick Start
catalogue. The data stay outside Git; their scientific and binary contract
does not.

Supported formats
-----------------

.. list-table:: Built-in particle readers
   :header-rows: 1
   :widths: 22 78

   * - ``format``
     - Reader
   * - ``bin``
     - Raw binary table with configurable dtype and column selectors.
   * - ``npz``
     - NumPy archive with a position key and optional field-key mapping.
   * - ``gadget``
     - Legacy Gadget-format snapshot, including split files.
   * - ``gadget_hdf5``
     - Single or split Gadget HDF5 snapshot with explicit unit scales.
   * - ``gadget-fof``
     - Legacy Gadget FoF catalogue without SUBFIND.
   * - ``fof``
     - Quijote/Pylians-style ``group_tab`` halo catalogue.

The local or URL-path suffix can infer ``bin``, ``npz``, ``gadget``, or ``fof``
when ``format`` is omitted. Directory and HDF5 base paths should set it
explicitly.

Raw binary tables
-----------------

.. code-block:: yaml

   fin:
      path: "./data/haloes.bin"
      format: "bin"
      reader_params:
         dtype: "float32"
         ncols: 7
         pos_cols: [0, 1, 2]
         fields:
            vx: 3
            vy: 4
            vz: 5
            mass: 6
      catalog_weight_key: null
      field_value_key: "mass"

A scalar selector returns one column; a list selector returns multiple columns.
Only scalar fields can be selected as catalogue weights or projected field
values.

NPZ archives
------------

.. code-block:: yaml

   fin:
      path: "./data/haloes.npz"
      format: "npz"
      reader_params:
         pos_key: "position"
         fields:
            completeness: "weight"
            vx: "velocity_x"
      catalog_weight_key: "completeness"
      field_value_key: "vx"

When ``fields`` is omitted, every array except ``pos_key`` is exposed under its
original name.

Gadget snapshots
----------------

The HDF5 reader resolves either one file or a base path such as ``snap_004``
with siblings ``snap_004.0.hdf5``, ``snap_004.1.hdf5``, and so on.

.. code-block:: yaml

   fin:
      path: "/path/to/snapdir_004/snap_004"
      format: "gadget_hdf5"
      reader_params:
         ptype: 1
         position_scale: 1.0e-3
         velocity_scale: 1.0
         mass_scale: 1.0
         load_velocity: false
         load_mass: false

For Quijote coordinates stored in :math:`\mathrm{kpc}/h`,
``position_scale=1e-3`` converts to :math:`\mathrm{Mpc}/h`. Velocity and mass
are not loaded unless requested, which avoids large unused arrays. ``h5py`` is
required; compressed snapshots may also require ``hdf5plugin``.

Quijote FoF catalogues
----------------------

.. code-block:: yaml

   fin:
      path: "./data/quijote_halos/8000"
      format: "fof"
      reader_params:
         snapnum: 4
         redshift: 0.0
         fields:
            mass: "mass"
            vx: "vel_x"
            vy: "vel_y"
            vz: "vel_z"
      catalog_weight_key: null
      field_value_key: null

The reader exposes ``pos``, ``mass``, ``vel``, component velocities, ``npart``,
and ``group_offset`` before optional selection. It converts Quijote positions
to :math:`\mathrm{Mpc}/h`, masses to :math:`M_\odot/h`, and applies the reader's
redshift velocity convention.

In-memory input
---------------

File input can be replaced by arrays:

.. code-block:: python

   task = SFCProjection()
   task.particle_pos = pos
   task.catalog_weight = completeness
   task.field_value = mass
   task.weight_normalization = "raw"
   field = task.run(save_result=False)

Positions must have shape ``(N, 3)``. Scalar weights or field values are
broadcast; arrays must be finite and have shape ``(N,)``.

Saved data classes
------------------

.. list-table:: Task output objects
   :header-rows: 1
   :widths: 28 30 42

   * - Task
     - Reader class
     - Main arrays
   * - ``SFCProjection``
     - ``SFCField``
     - ``epsilon`` and ``sfc_info``.
   * - ``Counting``
     - ``CountingData``
     - sampled positions and ``nx`` values.
   * - ``Corr_2PCF``
     - ``Corr2PCFData``
     - sampling coordinates, count products, and ``xi``.
   * - ``Corr_3PCF``
     - ``Corr3PCFData``
     - angles, side geometry, triplet products, ``zeta``, and ``Q``.
   * - ``Corr_3PCF_Multipole``
     - ``Corr3PCFMultipoleData``
     - samples, radial windows, ``l``, count multipoles, and ``zeta_l``.

Read products through those classes:

.. code-block:: python

   from pyhermes.io import SFCField, Corr2PCFData, Corr3PCFMultipoleData

   field = SFCField(data_path="./output/halo_sfc.pkl", threads=8)
   corr = Corr2PCFData(data_path="./output/halo_2pcf.pkl")
   multipoles = Corr3PCFMultipoleData(
       data_path="./output/halo_3pcf_multipoles.pkl"
   )

The current files contain NumPy-framed, pickle-serialized Python metadata.
They are trusted local products, not a safe interchange format for untrusted
files. Treat the data-class interface as public and the internal dictionary
layout as an implementation detail.

PyHermes converts path-like metadata to strings when writing new products.
The loader also translates the private ``pathlib._local`` path classes emitted
by Python 3.13, so trusted products can move between Python 3.12 and 3.13
without requiring users to rewrite ``task.fin``. This targeted compatibility
does not make arbitrary pickle objects portable or safe to load.

Particle companions
-------------------

``save_particle_data: true`` writes positions, catalogue weights, and field
values to ``particle_data_path`` as NPZ. When the path is empty, it is derived
from ``fout_path``. This companion is required when a later particle-centred
task cannot recover the original catalogue from ``fin``. Derived fields made by
arithmetic or convolution do not preserve a unique particle catalogue; pass
explicit centre arrays when using them as a particle-centred first vertex.

Output policy
-------------

Only rank 0 writes task products. Parent directories are created as needed.
Existing files are protected unless ``overwrite=True`` is passed to ``run``.
Store large generated products outside the tracked ``examples/`` source files;
the repository's scripts and notebooks use ``examples/output/`` for this
purpose.
