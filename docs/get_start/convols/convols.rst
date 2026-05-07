Convols
=======

``Convols`` is the entry point that converts a particle catalog into the
multiresolution coefficient field used by later PyHermes analyses.

In practice, this is the task that turns a raw particle sample into a reusable
``ConvolsData`` object. Later stages such as ``Counting``, ``Corr_2PCF``,
``Corr_3PCF``, and ``Corr_3PCF_Multipole`` all build on top of this field.

What it does
------------

- reads particle positions from a local file
- computes the scaling-coefficient field
- saves a reusable ``.pkl`` product for later tasks

Workflow Ladder
---------------

PyHermes supports several interface layers for the same computation. For
``Convols``, they can be thought of as a workflow ladder:

- **Workflow A. Command-Line Driver**:
  best for quick starts, batch runs, and Slurm jobs
- **Workflow B. Config-Driven Python API**:
  best when you want notebook control but still keep YAML as the main source of truth
- **Workflow C. Task Object with Attribute Overrides**:
  best when you want to tweak a few parameters interactively
- **Workflow D. Manual Input Objects and Custom Preparation**:
  best when particle data are already prepared in Python
- **Workflow E. Low-Level Building Blocks**:
  best when you want to work directly with the numerical ingredients

Workflow A. Command-Line Driver
-------------------------------

Use the shipped example config:

.. code-block:: yaml

   Convols:
      J: 8
      fin:
         path: "./data/quijote10000.bin"
         format: "bin"
         reader_params:
            dtype: "float32"
            ncols: 3
            pos_cols: [0, 1, 2]
            fields: {}
         weight_key: null
      fout_path: "./output/quijote_sfc.pkl"
      save_particle_data: false
      particle_data_path: ""
      phi_resolution: 1024
      box_size: 1000
      wavelet_mode: "db2"
      wavelet_level: 10

Then run:

.. code-block:: bash

   python run_convols.py

or with an explicit config path:

.. code-block:: bash

   python run_convols.py ./configs/param_convols.yaml

Workflow B. Config-Driven Python API
------------------------------------

This is the most direct Python equivalent of the command-line run:

.. code-block:: python

   from pyhermes.base.convols import Convols
   from pyhermes.param.parambase import read_param

   convols_params = read_param("./configs/param_convols.yaml")
   convols_task = Convols(param_task=convols_params)
   convols = convols_task.run(overwrite=True)

If you modify config values in an MPI workflow, do it only on rank 0.

Workflow C. Task Object with Attribute Overrides
------------------------------------------------

This layer is useful when the defaults are almost right and you only want to
adjust a few runtime attributes:

.. code-block:: python

   from pyhermes.base.convols import Convols

   convols_task = Convols()
   convols_task.threads = 8
   convols_task.fin = {
       "path": "./quijote10000.bin",
       "format": "bin",
       "reader_params": {
           "dtype": "float32",
           "ncols": 3,
           "pos_cols": [0, 1, 2],
           "fields": {},
       },
       "weight_key": None,
   }
   convols_task.prepare_input_fields()
   convols = convols_task.run(save_result=False)

Workflow D. Manual Input Objects and Custom Preparation
-------------------------------------------------------

If the particle positions are already available in memory, you can inject them
directly:

.. code-block:: python

   from pyhermes.base.convols import Convols

   convols_task = Convols()
   convols_task.particle_pos = p_pos
   # If particle_weight is omitted, unit weights are assumed.
   convols_task.prepare_input_fields()
   convols = convols_task.run(save_result=False)

This is the natural interface when the particle catalog is produced by another
piece of Python code and you do not want to write it to disk first.

Workflow E. Low-Level Building Blocks
-------------------------------------

At the lowest level, you can work directly with the numerical helpers used by
``Convols`` internally, for example by constructing the scaling field from
particle positions and wavelet data yourself. This gives maximum flexibility,
but it requires a clear understanding of the internal normalization and data
layout. Most users should start from Workflows A to D.

Output
------

The standard output is a serialized coefficient field such as:

.. code-block:: text

   ./output/quijote_sfc.pkl

That file becomes the standard input for ``Counting``, ``Corr_2PCF``,
``Corr_3PCF``, and ``Corr_3PCF_Multipole``.

Key parameters
--------------

- ``J``:
  multiresolution level
- ``fin.path``:
  local particle file path
- ``fin.format``:
  particle format such as ``bin``, ``npz``, ``gadget``, ``gadget-fof``, or ``fof``
- ``fin.reader_params``:
  format-specific reader options, such as binary column mappings
- ``fin.weight_key``:
  one-dimensional particle weight field name, or ``null`` for unit weights
- ``fout_path``:
  output path for the serialized coefficient field
- ``save_particle_data``:
  whether to save particle positions and weights to a companion ``.npz`` file
- ``particle_data_path``:
  optional companion particle-data path; if empty, PyHermes derives it from ``fout_path``
- ``phi_resolution``:
  number of samples used to tabulate ``phi``, the wavelet scaling function
- ``box_size``:
  simulation box side length
- ``wavelet_mode`` and ``wavelet_level``:
  wavelet settings

Notes
-----

- ``prepare_input_fields()`` prepares particle input and runtime metadata.
- ``run()`` performs the actual field construction.
- PyHermes does not download particle catalogs. Download or generate the input
  file before running ``Convols``.
