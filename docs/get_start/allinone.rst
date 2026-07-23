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
   * - ``quick_start.ipynb``
     - ``SFCProjection``, ``Corr_2PCF``
     - URL catalogue, saved base field, isotropic 2PCF, and result plot
   * - ``particle_io.ipynb``
     - ``read_particle_data``
     - URL caching, NPZ conversion, raw BIN layouts, and native readers
   * - ``sfc_projection.ipynb``
     - ``SFCProjection``, ``SFCField``
     - Field values, normalisation, resolution, redshift space, and metadata
   * - ``window.ipynb``
     - ``SFCField``, ``WindowFunc``
     - Field algebra, kernels, custom windows, and composition caveats
   * - ``physical_fields.ipynb``
     - weighted ``SFCField`` objects, operator windows
     - Velocity derivatives, momentum density, potential, and acceleration
   * - ``counting.ipynb``
     - ``Counting``, ``CountingData``
     - One-point samples, PDFs, and smoothing response
   * - ``corr2pcf.ipynb``
     - ``Corr_2PCF``, ``Corr2PCFData``
     - Isotropic and redshift-space two-point statistics
   * - ``corr3pcf.ipynb``
     - ``Corr_3PCF``, ``Corr_3PCF_Multipole``
     - Monte Carlo 3PCF, multipoles, and consistency checks

Each tutorial names the script and YAML file that generate any heavy saved
product it reads. Start with the notebook to understand a workflow; move to the
script and configuration when the calculation belongs on a server.
