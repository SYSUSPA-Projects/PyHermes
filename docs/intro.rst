Introduction
============

PyHermes is designed for large-scale structure workflows that start from
particle positions and end with correlation statistics. Its core idea is to
convert particle data into a multiresolution field representation and then
reuse that representation for multiple downstream measurements.

What PyHermes provides
----------------------

- A ``Convols`` task that reads particle positions and builds the
  multiresolution coefficient field.
- A ``Counting`` task that samples the field on random points, optionally after
  smoothing with a window.
- A ``Corr_2PCF`` task for two-point correlation measurements over a radial grid.
- A ``Corr_3PCF`` task for three-point correlation measurements with configurable
  triangle geometry and center sampling.
- A ``Corr_3PCF_Multipole`` task for multipole measurements of the three-point
  correlation function at fixed ``(r1, r2)``.
- Native parameter-file support for both YAML and JSON5.
- Optional MPI acceleration through ``mpi4py``.

Typical workflow
----------------

The usual PyHermes workflow is:

1. Read particle positions from disk or from a URL-supported example dataset.
2. Build the multiresolution coefficient field with ``Convols``.
3. Reuse that saved field for one or more downstream tasks:

   - ``Counting`` for random-point sampling and one-point statistics.
   - ``Corr_2PCF`` for two-point correlation measurements.
   - ``Corr_3PCF`` for three-point correlation measurements.
   - ``Corr_3PCF_Multipole`` for multipole moments of the 3PCF.

4. Save task outputs as pickle-based PyHermes data products.

Why the multiresolution field matters
-------------------------------------

The multiresolution field is the shared intermediate product in PyHermes. Once
it is written to disk, later tasks do not need to reread and repartition the
original particle catalog. That makes it much easier to iterate on smoothing
choices, radial ranges, random sampling density, and correlation settings.

Supported input formats
-----------------------

PyHermes currently supports the following particle input formats:

- ``generic_pos``: raw binary file of ``float32`` values reshaped to ``(-1, 3)``
- ``generic_pos_weight``: raw binary file of ``float32`` values reshaped to
  ``(-1, 4)``, where the fourth column is a particle weight
- ``gadget``
- ``gadget-fof``

For details on formats and parameter fields, see :doc:`param/param`.
