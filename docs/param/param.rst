Parameters
==========

This section summarizes the configuration structure used by PyHermes. Parameter
files can be written in YAML or JSON5, and each top-level task name matches the
corresponding Python class.

Supported top-level sections include:

- ``Convols``
- ``Counting``
- ``Corr_2PCF``
- ``Corr_3PCF``

Each task has a default parameter dictionary bundled with the package. User
configuration values override only the keys you specify, while unspecified
fields fall back to those defaults.

.. toctree::
   :maxdepth: 2
   :caption: Reference
   
   io/io
   perform/perform
   cal/cal

Minimal example
---------------

.. code-block:: yaml

   Convols:
      fin:
         path: "./data.bin"
         format: "generic_pos"
      fout_path: "./output/convols.pkl"

   Corr_2PCF:
      convols_data_path: "./output/convols.pkl"
      fout_path: "./output/corr2pcf.pkl"
      r_min: 1.0
      r_max: 150.0
      n_r: 30
