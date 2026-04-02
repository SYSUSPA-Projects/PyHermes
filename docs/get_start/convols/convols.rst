Convols
=======

``Convols`` is the entry task that converts a particle catalog into the
multiresolution coefficient field used by later PyHermes analyses.

What it does
------------

- reads particle positions from a local path or URL
- supports formats such as ``generic_pos`` and ``gadget``
- builds the internal field representation
- writes a reusable ``.pkl`` output file

Example configuration
---------------------

The repository includes ``examples/configs/param_convols.yaml`` with the
following structure:

.. code-block:: yaml

   Convols:
      J: 8
      fin:
         path: "https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin"
         format: "generic_pos"
      fout_path: "./output/quijote_sfc.pkl"
      SampRate: 1024
      SimBoxL: 1000
      wavelet_mode: "db2"
      wavelet_level: 10
      bandwidth: 1

Minimal Python driver
---------------------

.. code-block:: python

   from pyhermes.base.convols import Convols
   from pyhermes.param.parambase import read_param

   convols_params = read_param(config_path="./configs/param_convols.yaml")
   convols = Convols(param_task=convols_params)
   convols.run(overwrite=True)

Run it
------

From the ``examples`` directory:

.. code-block:: bash

   python run_convols.py

Or with MPI:

.. code-block:: bash

   mpirun -np 8 python run_convols.py

Output
------

The example writes a coefficient field to:

.. code-block:: text

   ./output/quijote_sfc.pkl

That file becomes the standard input for ``Counting``, ``Corr_2PCF``, and
``Corr_3PCF``.

Key parameters
--------------

- ``J``: multiresolution level
- ``fin.path``: input particle catalog path or URL
- ``fin.format``: particle format such as ``generic_pos``
- ``fout_path``: output path for the serialized coefficient field
- ``SampRate``: sampling rate used in the field construction
- ``SimBoxL``: simulation box side length
- ``wavelet_mode`` and ``wavelet_level``: wavelet settings
- ``bandwidth``: bandwidth control used by the algorithm

See :doc:`../../param/param` for a more complete parameter reference.
