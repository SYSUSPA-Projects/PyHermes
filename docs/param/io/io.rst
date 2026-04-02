Input and Output (IO)
=====================

This section describes how PyHermes reads particle data and how task outputs are
stored on disk.

Particle input formats
----------------------

PyHermes currently dispatches particle readers based on ``fin.format``.
Supported values in the current codebase are:

- ``generic_pos``
- ``generic_pos_weight``
- ``gadget``
- ``gadget-fof``

Generic binary position format
------------------------------

``generic_pos`` expects a raw binary file of ``float32`` values that can be
reshaped into ``(-1, 3)``:

.. code-block:: python

   import numpy as np

   pos = np.fromfile("data.bin", dtype=np.float32).reshape(-1, 3)

Each row is interpreted as ``(x, y, z)``.

Generic binary position + weight format
---------------------------------------

``generic_pos_weight`` expects ``float32`` values reshaped into ``(-1, 4)``,
where the first three columns are positions and the fourth column is a weight.

Remote input files
------------------

If ``fin.path`` is an ``http://`` or ``https://`` URL, PyHermes downloads the
file automatically before reading it. The downloaded file is saved using the
remote filename in the current working directory.

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

- ``fin.path``: local path or URL of the particle catalog
- ``fin.format``: declared input format
- ``fin.weight_key``: optional weight selector field for supported formats
- ``fout_path``: output file path for the current task
