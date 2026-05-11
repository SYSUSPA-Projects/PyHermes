Getting Started
===============

This section mirrors the five notebooks in ``examples/notebooks/``. The goal
is not to duplicate every cell, but to explain what each notebook is for, what
inputs it expects, and which outputs are lightweight enough to create inside
the notebook versus better produced by a separate script run.

Recommended order
-----------------

1. ``quick_start.ipynb`` for the smallest conceptual example
2. ``convols.ipynb`` for data preparation and field construction
3. ``counting.ipynb`` for one-point statistics
4. ``corr2pcf.ipynb`` for real-space and redshift-space 2PCF
5. ``corr3pcf.ipynb`` for 3PCF, low-level reconstruction, and multipoles

How the examples are organized
------------------------------

The notebooks, scripts, and configs are designed to use the same relative
paths from the ``examples/`` directory.

- notebooks live in ``examples/notebooks/``
- drivers live in ``examples/scripts/``
- YAML configs live in ``examples/configs/``

Each notebook switches its working directory to ``examples/`` near the top, so
the same script and config paths work in both notebook and command-line usage.


The repository does not commit the example data or the full set of outputs.

- ``examples/data/`` is populated locally. ``convols.ipynb`` includes the data
  preparation steps used by the rest of the tutorial chain.
- ``examples/output/`` is populated locally. Small products are produced during
  notebook execution.
- heavier 2PCF and 3PCF products are referenced in ``corr2pcf.ipynb`` and
  ``corr3pcf.ipynb`` together with the exact script and YAML file needed to
  reproduce them on your own machine or server

Notebook Guide
--------------

.. toctree::
   :maxdepth: 1

   quick_start
   convols/convols
   counting/counting
   corr_2pcf/corr_2pcf
   corr_3pcf/corr_3pcf
