Counting and one-point PDFs
===========================

``Counting`` filters an ``SFCField`` with one optional window and evaluates the
continuous result at uniformly sampled box positions. The returned values can
be used for count-in-cell PDFs, moments, environmental marks, and filtered-field
diagnostics.

The :doc:`complete Counting notebook </notebooks/counting>` compares the task
and low-level interfaces and visualises the resulting one-point PDFs.

Minimal YAML
------------

.. code-block:: yaml

   Counting:
      sfc_field: ./output/quijote8000_snap004_sfc.pkl
      random_count: 10000000
      seed: 42
      weight_normalization: catalog
      window:
         type: sphere
         len_args: {R: 20.0}
      threads: 8
      fout_path: ./output/quijote8000_snap004_counting_sph20.pkl

Run it from ``examples/``:

.. code-block:: bash

   python scripts/run_counting.py configs/param_counting.yaml

The result
----------

.. code-block:: python

   import numpy as np
   from pyhermes.io import CountingData

   counts = CountingData(
       data_path="./output/quijote8000_snap004_counting_sph20.pkl"
   )
   print(counts.nx.shape)
   print(np.mean(counts.nx), np.std(counts.nx))

``CountingData.nx`` contains physical-unit field densities at the sampled
positions. ``counting_info`` records the seed, sample count, window, input
field, and normalisation. The random positions themselves are deterministic
from those settings and are not stored separately.

Python task interface
---------------------

.. code-block:: python

   from pyhermes.theory.counting import Counting

   task = Counting()
   task.sfc_field = "./output/quijote8000_snap004_sfc.pkl"
   task.random_count = 300000
   task.seed = 7
   task.window = {"type": "gaussian", "len_args": {"R": 10.0}}
   task.threads = 8
   task.fout_path = "./output/counting_gaussian_R10.pkl"
   result = task.run()

Low-level equivalent
--------------------

When positions are scientifically meaningful rather than random, evaluate the
field directly:

.. code-block:: python

   from pyhermes.io import SFCField, WindowFunc

   D = SFCField(data_path="./output/quijote8000_snap004_sfc.pkl", threads=8)
   W = WindowFunc(
       {"type": "sphere", "len_args": {"R": 20.0}},
       D.sfc_info,
       threads=8,
   )
   local_density = (D @ W).field_density_at_pos(
       halo_positions,
       value_unit="physical",
   )

This is the route used to evaluate a local environment at each halo before
constructing a marked field.

Choose the window deliberately
------------------------------

``sphere``
   Literal count-in-sphere or top-hat-smoothed density.

``gaussian``
   Smooth low-pass field without a hard boundary.

``cubic`` or ``cylinder``
   Anisotropic finite-volume average.

``cw``, ``cws``, ``gdw``
   Zero-mean localised fluctuation rather than a count density.

For a sphere of radius :math:`R`, a raw number field can be converted to an
expected count by multiplying density by :math:`4\pi R^3/3`. With catalogue
normalisation, remember that the field integrates to one; use its metadata and
mean density rather than assuming a raw particle count.

MPI distribution
----------------

The filtered field is broadcast to all ranks. Each rank samples a disjoint
deterministic subset of positions and rank 0 gathers the values. ``random_count``
is padded internally only when needed for an even rank distribution; the
returned array is trimmed to the requested size.

Public tutorial
---------------

The :doc:`rendered notebook </notebooks/counting>` compares task and low-level interfaces,
constructs one-point PDFs, and contrasts low-pass and high-pass responses.
