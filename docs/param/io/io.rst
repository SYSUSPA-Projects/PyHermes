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
- ``gadget``
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
      weight_key: "mass"

Scalar field mappings return one-dimensional arrays. List mappings return
two-dimensional arrays. ``weight_key`` must refer to a one-dimensional field;
use ``null`` for unit weights.

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
            weight: "weight"
      weight_key: "weight"

Task outputs
------------

PyHermes writes serialized task outputs as pickle-based ``.pkl`` files. Common
examples are:

- ``ConvolsData``
- ``CountingData``
- ``Corr2PCFData``
- ``Corr3PCFData``

Most workflows reuse the ``Convols`` output file as the main upstream input for
later tasks.

Relevant fields
---------------

- ``fin.path``: local path of the particle catalog
- ``fin.format``: declared input format
- ``fin.reader_params``: format-specific reader options
- ``fin.weight_key``: optional one-dimensional weight selector; ``null`` means unit weights
- ``fout_path``: output file path for the current task
