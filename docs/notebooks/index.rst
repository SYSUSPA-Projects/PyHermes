Notebook tutorials
==================

These pages render the notebooks tracked in ``examples/notebooks`` directly.
They show the saved outputs committed with each tutorial, while documentation
builds never rerun the scientific calculations.

Use the prose guides for a compact explanation of each workflow and the
notebooks for complete, executable code. The source notebooks are also
available in the `GitHub examples directory
<https://github.com/SYSUSPA-Projects/PyHermes/tree/main/examples/notebooks>`_.

.. toctree::
   :maxdepth: 1

   quick_start
   particle_io
   sfc_projection
   window
   physical_fields
   counting
   corr2pcf
   corr3pcf

.. note::

   Saved three-point multipole outputs in the older ``corr3pcf`` notebook
   predate the 1.2 Gamma correction. Use the corrected task-guide figures and
   :doc:`../benchmark` for current validation. Standard direct-angular 3PCF
   outputs are not affected by the Gamma correction.
