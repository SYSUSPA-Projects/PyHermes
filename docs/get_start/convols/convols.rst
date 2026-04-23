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
- optionally downloads the particle file first if ``fin.url`` is given
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
         url: "https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin"
         format: "generic_pos"
         weight_key: "no_weight"
      fout_path: "./output/quijote_sfc.pkl"
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
       "format": "generic_pos",
       "weight_key": "no_weight",
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
- ``fin.url``:
  optional download URL; if non-empty, PyHermes downloads to ``fin.path`` first
- ``fin.format``:
  particle format such as ``generic_pos``
- ``fin.weight_key``:
  particle weight field name, or ``no_weight``
- ``fout_path``:
  output path for the serialized coefficient field
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
- If the directory in ``fin.path`` does not exist, it is created automatically
  before download.
