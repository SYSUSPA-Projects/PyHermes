Convols
=======

``convols.ipynb`` is the upstream notebook for the whole example chain. It is
where particle catalogs become ``ConvolsData`` fields that later notebooks
reuse. If you only want to prepare those files without opening a notebook, run
``examples/scripts/prepare_convols_data.py`` from the repository root.

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
tracked example workflow, ``convols.ipynb`` shows the construction step by
step, while ``examples/scripts/prepare_convols_data.py`` provides the direct
command-line route.

Tracked inputs and local products
---------------------------------

Tracked in the repository:

- ``examples/notebooks/convols.ipynb``
- ``examples/scripts/prepare_convols_data.py``
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

Minimal YAML Shape
------------------

The standard field-construction config starts from an input catalog, defines the
multiresolution grid, and writes a reusable ``ConvolsData`` object:

.. code-block:: yaml

   Convols:
      fin:
         path: "./data/quijote_halos/8000"
         format: "fof"
         reader_params:
            snapnum: 4
      box_size: 1000
      J: 8
      wavelet_mode: "db2"
      wavelet_level: 10
      phi_resolution: 1024
      threads: 2
      save_particle_data: True
      particle_data_path: "./data/quijote_halos/8000/groups_004/group_tab_004.pos.npz"
      fout_path: "./output/quijote8000_snap004_sfc.pkl"

The notebook shows this shape before the command-line run so that the YAML file
and driver script can be read together.

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

Mathematical idea
-----------------

The input catalog is a weighted point process,

.. math::

   n(\mathbf{x}) =
   \sum_i w_i\,\delta_{\rm D}^{(3)}(\mathbf{x}-\mathbf{x}_i).

``Convols`` projects it onto scaling-function coefficients,

.. math::

   n_j(\mathbf{x}) =
   \sum_\ell \epsilon_{j\ell}\phi_{j\ell}(\mathbf{x}),
   \qquad
   \epsilon_{j\ell} =
   \sum_i w_i\phi_{j\ell}(\mathbf{x}_i).

The saved ``ConvolsData`` object stores these normalized coefficients and the
metadata needed to apply later windows. The redshift-space cells first map
positions along a chosen line of sight,

.. math::

   \mathbf{s}
   =
   \mathbf{x}
   +
   {(\mathbf{v}\cdot\widehat{\mathbf{n}})(1+z)\over H(z)}
   \widehat{\mathbf{n}},

with periodic wrapping in the simulation box, and then build the same
coefficient field from :math:`\mathbf{s}`.
