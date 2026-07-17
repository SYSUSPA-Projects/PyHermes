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
     - ``SFCProjection``, ``WindowFunc``
     - Smallest end-to-end field and 2PCF calculation
   * - ``sfc_projection.ipynb``
     - ``SFCProjection``, ``SFCField``
     - Readers, field resolution, weights, and field reconstruction
   * - ``window.ipynb``
     - ``SFCField``, ``WindowFunc``
     - Field algebra, kernels, custom windows, and composition caveats
   * - ``counting.ipynb``
     - ``Counting``, ``CountingData``
     - One-point samples, PDFs, and smoothing response
   * - ``corr2pcf.ipynb``
     - ``Corr_2PCF``, ``Corr2PCFData``
     - Isotropic and redshift-space two-point statistics
   * - ``corr3pcf.ipynb``
     - ``Corr_3PCF``, ``Corr_3PCF_Multipole``
     - Monte Carlo 3PCF, multipoles, and consistency checks
   * - ``weighted_fields.ipynb``
     - weighted ``SFCField`` objects, operator windows
     - Velocity derivatives, potential, and acceleration

Each tutorial names the script and YAML file that generate any heavy saved
product it reads. Start with the notebook to understand a workflow; move to the
script and configuration when the calculation belongs on a server.
