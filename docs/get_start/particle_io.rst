Particle input and conversion
=============================

PyHermes separates catalogue I/O from field projection. Every built-in
particle reader returns the same small contract: ``pos`` with shape ``(N, 3)``,
``size``, and any requested one-dimensional particle fields. Once a catalogue
has this form, the same arrays can be passed to ``SFCProjection`` regardless of
the original file format.

The executable companion is ``examples/notebooks/particle_io.ipynb``. It
downloads the compact Quijote FoF example, reads the original catalogue, and
converts all 406,728 haloes to equivalent NPZ and BIN files. The complete
notebook remains suitable for a local first run.

Read a URL-backed NPZ catalogue
-------------------------------

``read_particle_data`` accepts local paths and HTTP(S) URLs. The Quick Start
YAML supplies a stable URL, an exact cache destination, and an optional SHA256
digest:

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

The first call downloads and verifies the file. A valid file at
``download.cache_path`` is reused on later calls, so the example remains
offline-friendly after its first run. ``sha256`` is recommended for a public
scientific dataset but is not required for a user's own local file.

Make a portable NPZ catalogue
-----------------------------

NPZ is the recommended interchange format for a compact tutorial catalogue.
Keep positions in one ``(N, 3)`` array and give each reusable scalar or vector
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
exposed. The conversion cells in ``particle_io.ipynb`` define the public
catalogue contract directly: array names, dtypes, shapes, and physical units
are shown beside the code that creates them. The Quick Start YAML records the
stable download URL and checksum.

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
directories. These readers preserve format-specific unit controls while
returning the same shared particle dictionary. See :doc:`../param/io/io` for
their complete parameters.

Source catalogue versus projection companion
---------------------------------------------

The source NPZ may contain velocities, mass, particle counts, and other
scientific columns. By contrast, ``SFCProjection.save_particle_data`` writes a
compact companion containing only ``pos``, ``catalog_weight``, and
``field_value``. The companion records exactly what entered one projection and
supports particle-centred estimators; it is not intended to replace the richer
source catalogue.

With the reader boundary understood, continue to
:doc:`sfc_projection/sfc_projection` to build number, weighted, and physical
``SFCField`` objects.
