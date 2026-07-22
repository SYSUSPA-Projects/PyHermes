Quick start
===========

This is the canonical first PyHermes calculation. Starting with no local
catalogue, it downloads one compact Quijote halo file, constructs a reusable
``SFCField``, measures an isotropic 2PCF, saves both products, and plots the
result. The matching notebook is ``examples/notebooks/quick_start.ipynb``;
it reads the same YAML files and writes the same outputs shown here.

Run from ``examples/``
----------------------

The paths in the public configurations are relative to that directory:

When working from a repository clone, install that checkout once so the
commands and notebook import the code beside them rather than another
PyHermes installation:

.. code-block:: bash

   python -m pip install -e ".[plot]"

.. code-block:: bash

   cd examples

Project the catalogue
---------------------

``configs/param_sfc_projection.yaml`` contains the remote input and its local
cache policy:

.. code-block:: yaml

   SFCProjection:
      fin:
         path: https://pyhermes.astroslacker.com/downloads/quijote_halos_8000_snap004.npz
         format: npz
         download:
            cache_path: ./data/quijote_halos_8000_snap004.npz
            sha256: b2b4b8c2fb91fa857e21b43d943cc32a2c423ee7cf5d7f13dede608264b08ef6
         catalog_weight_key: null
         field_value_key: null
      box_size: 1000.0
      J: 8
      wavelet_mode: db2
      wavelet_level: 10
      phi_resolution: 1024
      weight_normalization: catalog
      threads: 2
      save_particle_data: true
      particle_data_path: ./output/quijote8000_snap004_particles.npz
      fout_path: ./output/quijote8000_snap004_sfc.pkl

Run the standard driver:

.. code-block:: bash

   python scripts/run_sfc_projection.py configs/param_sfc_projection.yaml

On the first run PyHermes downloads about 9 MB, verifies its SHA256 digest,
and caches it at ``data/quijote_halos_8000_snap004.npz``. Later runs use that
file directly. ``SFCProjection`` writes the reusable field and a compact
particle companion for particle-centred tasks.

Measure the 2PCF
----------------

The second configuration consumes exactly that field:

.. code-block:: yaml

   Corr_2PCF:
      sfc_field: ./output/quijote8000_snap004_sfc.pkl
      random: uniform
      binning_window: shell
      sampling:
         s: {min: 0.0, max: 150.0, n: 31}
      products: [dd, dr, rd, xi]
      threads: 2
      fout_path: ./output/quijote8000_snap004_2pcf.pkl

.. code-block:: bash

   python scripts/run_2pcf.py configs/param_2pcf.yaml

``random: uniform`` is an analytic constant reference field, so this basic
run has no stochastic random catalogue or hidden random seed. The vertices
receive no additional smoothing; ``shell`` supplies only the sampled radial
binning operator.

Load and plot
-------------

Task outputs are read through their public data classes:

.. code-block:: python

   import matplotlib.pyplot as plt
   from pyhermes.io import Corr2PCFData

   corr = Corr2PCFData(
       data_path="./output/quijote8000_snap004_2pcf.pkl"
   )

   fig, ax = plt.subplots(figsize=(7.0, 4.6))
   ax.plot(corr.s, corr.s**2 * corr.xi, color="#1f77b4", lw=2.0)
   ax.axhline(0.0, color="0.35", lw=0.8)
   ax.set(
       xlabel=r"$s\ [h^{-1}\mathrm{Mpc}]$",
       ylabel=r"$s^2\xi(s)$",
   )
   ax.grid(alpha=0.25)
   fig.tight_layout()
   plt.show()

.. figure:: ../_static/results/quick_start_2pcf.png
   :width: 70%
   :align: center

   The isotropic 2PCF produced by the Quick Start YAML and notebook.

The notebook executes these same three stages through the Python task API.
There is deliberately no second set of separations, smoothing parameters, or
output names to reconcile.

Where next?
-----------

The generated ``output/quijote8000_snap004_sfc.pkl`` is the common starting
field for the next tutorials:

1. :doc:`particle_io` explains URL caching, catalogue conversion, and the
   common particle-reader contract.
2. :doc:`sfc_projection/sfc_projection` explains weights, normalisation,
   resolution, redshift-space coordinates, and field metadata.
3. :doc:`window/window` introduces the ``SFCField @ WindowFunc`` language.
4. :doc:`counting/counting` samples the same field for one-point statistics.
5. :doc:`corr_2pcf/corr_2pcf` extends this run to smoothing, alternate bins,
   cross-correlations, and redshift-space anisotropy.
6. :doc:`corr_3pcf/corr_3pcf` and
   :doc:`corr_3pcf_multipole/corr_3pcf_multipole` reuse the field for
   three-point statistics.

Examples that require J=9, redshift-space catalogues, a dense sampled random
field, or a dark-matter snapshot are identified as advanced data products in
their own tutorials. They are not prerequisites for this first run.
