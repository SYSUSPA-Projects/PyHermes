Example catalogue
=================

The public tutorials are deliberately small enough to read from top to bottom.
They are not a dump of internal tests.

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Notebook
     - Main objects
     - Scientific focus
   * - :doc:`quick_start.ipynb </notebooks/quick_start>`
     - ``SFCProjection``, ``Corr_2PCF``
     - URL catalogue, saved base field, isotropic 2PCF, and result plot
   * - :doc:`particle_io.ipynb </notebooks/particle_io>`
     - ``read_particle_data``
     - FoF input, NPZ conversion, raw BIN layouts, and native readers
   * - :doc:`sfc_projection.ipynb </notebooks/sfc_projection>`
     - ``SFCProjection``, ``SFCField``
     - Field values, normalisation, resolution, redshift space, and metadata
   * - :doc:`window.ipynb </notebooks/window>`
     - ``SFCField``, ``WindowFunc``
     - Field algebra, kernels, custom windows, and composition caveats
   * - :doc:`physical_fields.ipynb </notebooks/physical_fields>`
     - weighted ``SFCField`` objects, operator windows
     - Velocity derivatives, momentum density, potential, and acceleration
   * - :doc:`counting.ipynb </notebooks/counting>`
     - ``Counting``, ``CountingData``
     - One-point samples, PDFs, and smoothing response
   * - :doc:`corr2pcf.ipynb </notebooks/corr2pcf>`
     - ``Corr_2PCF``, ``Corr2PCFData``
     - Isotropic and redshift-space two-point statistics
   * - :doc:`corr3pcf.ipynb </notebooks/corr3pcf>`
     - ``Corr_3PCF``, ``Corr_3PCF_Multipole``
     - Monte Carlo 3PCF, multipoles, and consistency checks

Each tutorial names the script and YAML file that generate any heavy saved
product it reads. Start with the notebook to understand a workflow; move to the
script and configuration when the calculation belongs on a server.
