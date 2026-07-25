Particle input and conversion
=============================

PyHermes separates catalogue I/O from field projection. Every built-in
particle reader returns the same small contract: ``pos`` with shape ``(N, 3)``,
``size``, and any requested one-dimensional particle fields. Once a catalogue
has this form, the same arrays can be passed to ``SFCProjection`` regardless of
the original file format.

The :doc:`executable Particle I/O notebook </notebooks/particle_io>` downloads
the original single-file Quijote ``group_tab`` catalogue, reads it directly,
and converts all 406,728 haloes to equivalent NPZ and BIN files. The complete
notebook remains suitable for a local first run.

Read a group_tab file directly
------------------------------

The public FoF file reports ``Nfiles = 1`` in its header, so it contains the
complete catalogue. It can therefore be downloaded, cached, and read without
constructing a directory tree or unpacking an archive:

.. code-block:: python

   from pyhermes.io import read_particle_data

   particles = read_particle_data(
       "https://pyhermes.astroslacker.com/downloads/group_tab_004.0",
       data_format="fof",
       download={
           "cache_path": "./data/quijote_halos/8000/groups_004/group_tab_004.0",
           "sha256": "4a1c6ca4f6747a70e9e552685226ecf5d678c6c97551e2caa7cc3883502eac85",
       },
       redshift=0.0,
       fields={
           "vx": "vel_x",
           "vy": "vel_y",
           "vz": "vel_z",
           "mass": "mass",
           "npart": "npart",
       },
   )

For a split catalogue, pass its local ``.0`` file and keep every numerically
suffixed sibling beside it. The reader obtains ``Nfiles`` from the header and
loads the remaining pieces. Catalogue-root input with ``snapnum`` remains
available for existing workflows.

Reuse the Quick Start input
---------------------------

``read_particle_data`` accepts local paths and HTTP(S) URLs. The Quick Start
YAML records the same FoF URL, exact cache destination, and optional SHA256
digest used above:

.. code-block:: python

   from pyhermes.io import read_particle_data
   from pyhermes.param.parambase import read_param

   config = read_param("./configs/param_sfc_projection.yaml")
   fin = config["SFCProjection"]["fin"]

   particles = read_particle_data(
       fin["path"],
       data_format=fin["format"],
       download=fin["download"],
   )

The first call downloads and verifies the ``group_tab`` file. A valid file at
``download.cache_path`` is reused on later calls, so the example remains
offline-friendly after its first run. ``sha256`` is recommended for a public
scientific dataset but is not required for a user's own local file.

Make a portable NPZ catalogue
-----------------------------

NPZ is a convenient interchange format for a compact user catalogue. It is
constructed locally here to demonstrate the NPZ reader; PyHermes does not
distribute a second copy of the example catalogue in this format. Keep
positions in one ``(N, 3)`` array and give each reusable scalar or vector
component a descriptive key:

.. code-block:: python

   import numpy as np

   np.savez_compressed(
       "./data/my_halo_catalogue.npz",
       pos=pos.astype("float32"),
       vel_x=vel_x.astype("float32"),
       vel_y=vel_y.astype("float32"),
       vel_z=vel_z.astype("float32"),
       mass=mass.astype("float32"),
   )

   particles = read_particle_data(
       "./data/my_halo_catalogue.npz",
       data_format="npz",
       fields={"mass": "mass", "vz": "vel_z"},
   )

When ``fields`` is omitted, all arrays other than the position key are
exposed. The conversion cells in
:doc:`particle_io.ipynb </notebooks/particle_io>` make the local format
contract explicit: array names, dtypes, shapes, and physical units are shown
beside the code that creates them.

Describe a raw BIN table where it is used
-----------------------------------------

A headerless binary table does not describe its own column layout. Keep that
mapping in the reader call or the task YAML rather than hiding it in a
catalogue-specific helper file:

.. code-block:: python

   particles = read_particle_data(
       "./data/catalogue.bin",
       data_format="bin",
       dtype="float32",
       ncols=7,
       pos_cols=(0, 1, 2),
       fields={
           "vel_x": 3,
           "vel_y": 4,
           "vel_z": 5,
           "mass": 6,
       },
   )

This makes the otherwise implicit binary contract visible next to the code
that depends on it.

Native simulation readers
-------------------------

For production data, PyHermes also reads legacy Gadget snapshots, Gadget
HDF5 snapshots, Gadget FoF catalogues, and Quijote/Pylians ``group_tab``
files or directories. These readers preserve format-specific unit controls
while returning the same shared particle dictionary. See
:doc:`../param/io/io` for their complete parameters.

Source catalogue versus projection companion
---------------------------------------------

The source catalogue may contain velocities, mass, particle counts, and other
scientific columns, whether it is FoF, NPZ, BIN, or another supported format.
By contrast, ``SFCProjection.save_particle_data`` writes a compact companion
containing only ``pos``, ``catalog_weight``, and ``field_value``. The companion
records exactly what entered one projection and supports particle-centred
estimators; it is not intended to replace the richer source catalogue.

With the reader boundary understood, continue to
:doc:`sfc_projection/sfc_projection` to build number, weighted, and physical
``SFCField`` objects.
