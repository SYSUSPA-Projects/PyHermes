Convols
=======

``convols.ipynb`` is the upstream notebook for the whole example chain. It is
where particle catalogs become ``ConvolsData`` fields that later notebooks
reuse.

What this notebook covers
-------------------------

The notebook is organized in five steps:

1. read particle data through different reader paths
2. build ``ConvolsData`` through the standard driver, config-driven API, and
   task-object overrides
3. construct a matching random field
4. reload data and random fields from disk
5. build ``delta`` and apply a window

It also includes a redshift-space preparation section, so this is the right
place to understand how the real-space and redshift-space example fields are
created.

Why this notebook matters
-------------------------

Every later notebook assumes that one or more field files already exist. In the
tracked example workflow, ``convols.ipynb`` is the notebook that creates them.

Tracked inputs and local products
---------------------------------

Tracked in the repository:

- ``examples/notebooks/convols.ipynb``
- ``examples/scripts/run_convols.py``
- ``examples/configs/param_convols.yaml``
- ``examples/data/.gitkeep``

Generated locally while following the notebook:

- the downloaded and unpacked example halo catalog under ``examples/data/``
- the compact FoF-derived binary table documented by
  ``examples/data/quijote_halos/quijote_halo_bin_schema.yaml``
- the main field products in ``examples/output/``, such as:

  - ``quijote8000_snap004_sfc.pkl``
  - ``quijote8000_snap004_rsd_sfc.pkl``
  - ``quijote8000_snap004_rsd_diag_sfc.pkl``
  - ``random_sfc.pkl``

Recommended usage modes
-----------------------

Use the notebook when you want to:

- inspect supported particle readers
- prepare the tracked example data for the first time
- compare real-space and redshift-space field construction
- work interactively with manual particle arrays

Use the driver script when you want a clean batch run:

.. code-block:: bash

   cd examples
   python ./scripts/run_convols.py ./configs/param_convols.yaml

The notebook shows the same task through the config-driven API and through
manual object setup, so it doubles as both a tutorial and a reference for
interactive usage.

Key idea
--------

``Convols`` builds a weighted multiresolution field and stores it in a reusable
format. Once that field exists, downstream tasks no longer need to reread and
repartition the original particle catalog.
