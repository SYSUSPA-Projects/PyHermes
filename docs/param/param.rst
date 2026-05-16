Parameters
==========

This section is a reference appendix for PyHermes configuration files. The
fastest way to learn the parameter structure is still to read the example YAML
files under ``examples/configs/`` alongside the tutorial notebooks.

Supported top-level task sections include:

- ``Convols``
- ``Counting``
- ``Corr_2PCF``
- ``Corr_3PCF``
- ``Corr_3PCF_Multipole``

Each task starts from a default parameter dictionary bundled with the package.
User YAML or JSON5 files only need to override the keys they actually want to
change.

.. toctree::
   :maxdepth: 2
   :caption: Reference

   defaults/defaults
   io/io
   perform/perform
   cal/cal

Practical advice
----------------

For day-to-day use:

- use ``examples/configs/`` as the most concrete starting point
- use the notebook pages in :doc:`../get_start/get_start` to understand which
  parameters matter for each task
- come back to this section when you need field-by-field details

Minimal example
---------------

.. code-block:: yaml

   Convols:
      fin:
         path: "./data.bin"
         format: "bin"
         reader_params:
            dtype: "float32"
            ncols: 3
            pos_cols: [0, 1, 2]
            fields: {}
         weight_key: null
      fout_path: "./output/convols.pkl"

   Corr_2PCF:
      convols_data: "./output/convols.pkl"
      random: "uniform"
      fout_path: "./output/corr2pcf.pkl"
      pair_window: "shell"
      sampling:
         s:
            min: 1.0
            max: 150.0
            n: 30
