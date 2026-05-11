Benchmark
=========

This page summarizes the benchmark logs collected under ``examples/logs``.
The raw logs are useful for development, but this page records the extracted
numbers so users can quickly understand the expected scale of the example
workflows.

Reading The Tables
------------------

All rows use the Quijote halo example unless noted otherwise:

- approximately ``4.07e5`` halos
- ``box_size = 1000``
- ``wavelet_mode = db2`` and ``wavelet_level = 10``
- ``phi_resolution = 1024``
- ``J = 8`` for the main field, corresponding to a ``256^3`` coefficient grid

The ``main`` column reports the core measurement loop when the log exposes it.
The ``task`` column includes setup, post-processing, and output writing. For
MPI runs, the resource label is written as ``MPI ranks x threads per rank``.
The logs do not record the CPU model. The multipole logs do record an
``NVIDIA GeForce RTX 4090`` with CUDA 12.4.

2PCF
----

The 2PCF chart groups runs by the actual performance distinction rather than
by every cosmetic option in the YAML file. Sphere smoothing, ring/cylinder
shape choices, and disk-like pair-window variants have the same basic compute
scale, so they are treated as one timing family. In the current cleaned log
set, the pooled axis-LOS uniform group contains ring, cylinder, and ``sph5``
runs.

.. image:: _static/benchmark_2pcf.png
   :alt: 2PCF benchmark chart
   :class: benchmark-figure

Label notes: ``smu`` means the ``(s, mu)`` redshift-space coordinate grid;
``rppi`` means the ``(rp, pi)`` coordinate grid; ``axis LOS`` uses a box-axis
line of sight; ``diag LOS`` uses the diagonal line of sight ``[1, 1, 1]``;
``uniform shortcut`` uses the analytic uniform random field; ``explicit
random`` reads and convolves a saved random ``ConvolsData`` field. Error bars
are sample standard deviations across logs in the same group.

.. list-table::
   :header-rows: 1
   :widths: 26 22 12 12 12 28

   * - Group
     - What is averaged
     - Resources
     - Main [s]
     - Task [s]
     - Source logs
   * - ``smu`` axis LOS, uniform shortcut
     - ring, cylinder, and ``sph5`` runs
     - ``16 x 8``
     - ``75.55 +/- 1.47``
     - ``77.30 +/- 2.44``
     - ``slurm_2pcf_smu_89383.log``, ``slurm_2pcf_smu_cylinder_89381.log``, ``slurm_2pcf_smu_sph5_89382.log``
   * - ``smu`` diagonal LOS, uniform shortcut
     - no smoothing and ``sph5`` runs
     - ``16 x 8``
     - ``135.78 +/- 0.20``
     - ``138.24 +/- 2.48``
     - ``slurm_2pcf_smu_diag_89614.log``, ``slurm_2pcf_smu_sph5_diag_89615.log``
   * - ``smu`` axis LOS, explicit random
     - saved random field
     - ``16 x 8``
     - ``112.57``
     - ``117.74``
     - ``slurm_2pcf_smu_sph5_with_random_89384.log``
   * - ``rppi`` axis LOS, explicit random
     - saved random field
     - ``16 x 8``
     - ``102.18``
     - ``109.74``
     - ``slurm_2pcf_rppi_sph5_with_random_89379.log``

The diagonal LOS is slower than the axis-aligned LOS because it falls back to
the more general line-of-sight-aware kernel path. Explicit random fields are
also slower than the uniform shortcut. The ``rppi`` explicit-random run is
slightly cheaper than the ``smu`` explicit-random run here because the sampled
grid is smaller.

Standard 3PCF
-------------

These runs correspond to the standard and low-level ``Q`` reconstruction
sections in ``corr3pcf.ipynb``. They use ``r12 = 20``, ``r13 = 40``, sphere
smoothing with ``R = 5`` on the second and third legs, and ``20`` angular
samples. The parallel layout differs between rows, so the chart labels include
the resource layout directly.

.. image:: _static/benchmark_3pcf.png
   :alt: Standard 3PCF benchmark chart
   :class: benchmark-figure

Label notes: ``pcenter`` means particle centers; ``rcenter`` means box-random
Monte Carlo centers; ``theta`` and ``mu`` are the angular coordinate used by
the run; ``nrot`` is the number of rotations per angular bin; ``[16x8]`` means
16 MPI ranks with 8 threads per rank. ``two-stage explicit random`` is the
helper workflow that first computes box-random normalization with an explicit
random field and then computes the particle-center data-minus-random part.

.. list-table::
   :header-rows: 1
   :widths: 26 22 14 13 12 12 12 22

   * - Config
     - Estimator
     - Centers
     - Resources
     - Main [s]
     - 3PCF [s]
     - Task [s]
     - Source log
   * - ``param_3pcf_pcenter_nrot200.yaml``
     - particle center, ``theta``
     - ``406728``
     - ``32 x 1``
     - ``54.44``
     - ``78.56``
     - ``85.17``
     - ``slurm_3pcf_pcenter_nrot200_89352.log``
   * - ``param_3pcf_pcenter_nrot500.yaml``
     - particle center, ``theta``
     - ``406728``
     - ``32 x 1``
     - ``237.64``
     - ``263.04``
     - ``276.04``
     - ``slurm_3pcf_pcenter_nrot500_89353.log``
   * - ``param_3pcf_pcenter_nrot1000.yaml``
     - particle center, ``theta``
     - ``406728``
     - ``16 x 8``
     - ``77.12``
     - ``85.98``
     - ``90.22``
     - ``slurm_3pcf_pcenter_nrot1000_89354.log``
   * - ``param_3pcf_pcenter_nrot1000_mu.yaml``
     - particle center, ``mu``
     - ``406728``
     - ``16 x 8``
     - ``109.67``
     - ``118.35``
     - ``122.70``
     - ``slurm_3pcf_pcenter_nrot1000_mu_89628.log``
   * - ``param_3pcf_pcenter_nrot2000.yaml``
     - particle center, ``theta``
     - ``406728``
     - ``8 x 4``
     - ``508.21``
     - ``519.38``
     - ``523.88``
     - ``slurm_3pcf_pcenter_nrot2000_89358.log``
   * - ``param_3pcf_rcenter_with_random.yaml``
     - box-random center, explicit random
     - ``8e6``
     - ``16 x 8``
     - ``590.06``
     - ``606.06``
     - ``611.13``
     - ``slurm_3pcf_rcenter_with_random_89357.log``
   * - ``run_3pcf_pcenter_with_random.py``
     - two-stage explicit random
     - ``8e6 + 406728``
     - ``16 x 8``
     - ``595.34 + 77.70``
     - ``621.88 + 105.04``
     - ``734.67``
     - ``slurm_3pcf_pcenter_with_random_89364.log``
   * - ``param_3pcf_rcenter_nrot200.yaml``
     - box-random center, uniform shortcut
     - ``8e6``
     - ``32 x 1``
     - ``1742.98``
     - ``1772.22``
     - ``1785.23``
     - ``slurm_3pcf_rcenter_nrot200_89356.log``

The particle-center estimator is much cheaper because it samples existing halo
centers. Box-random center runs sample millions of Monte Carlo centers and are
dominated by the center loop. Because these rows use different parallel
layouts, they are best read as representative example timings rather than a
clean scaling law.

3PCF Multipoles
---------------

The multipole examples use ``pair_mpi`` execution and the RTX 4090 recorded in
the logs. The non-``full`` configs request ``zeta_l`` with a uniform-random
shortcut. The ``full`` configs use an explicit random field and request the
heavier raw products needed for the full normalization path.

.. image:: _static/benchmark_3pcf_multipole.png
   :alt: 3PCF multipole benchmark chart
   :class: benchmark-figure

Label notes: ``J8`` and ``J9`` are the multiresolution grid levels; ``lmax`` is
the maximum multipole order; ``shortcut`` means the uniform random contribution
uses the analytic shortcut; ``full`` means the run computes explicit
``ddd_l``, ``delta_ddd_l``, and ``rrr_l`` products with a saved random field;
``[36x1]`` means 36 MPI ranks with 1 thread per rank.

.. list-table::
   :header-rows: 1
   :widths: 24 10 10 15 16 12 12 22

   * - Config
     - ``J``
     - ``lmax``
     - Resources
     - Heavy product [s]
     - Multipole [s]
     - Task [s]
     - Source log
   * - ``param_3pcf_multipole_lmax7.yaml``
     - ``8``
     - ``7``
     - ``36 x 1``
     - ``37.86``
     - ``38.18``
     - ``49.23``
     - ``slurm_3pcf_multipole_lmax7_mpi_89343.log``
   * - ``param_3pcf_multipole_lmax10.yaml``
     - ``8``
     - ``10``
     - ``36 x 1``
     - ``106.13``
     - ``106.14``
     - ``113.13``
     - ``slurm_3pcf_multipole_lmax10_mpi_89347.log``
   * - ``param_3pcf_multipole_lmax14.yaml``
     - ``8``
     - ``14``
     - ``36 x 1``
     - ``217.31``
     - ``217.33``
     - ``224.33``
     - ``slurm_3pcf_multipole_lmax14_mpi_89346.log``
   * - ``param_3pcf_multipole_lmax20.yaml``
     - ``8``
     - ``20``
     - ``30 x 4``
     - ``199.11``
     - ``199.18``
     - ``204.05``
     - ``slurm_3pcf_multipole_lmax20_mpi_89349.log``
   * - ``param_3pcf_multipole_lmax7_full.yaml``
     - ``8``
     - ``7``
     - ``36 x 1``
     - ``37.21 + 33.46 + 33.92``
     - ``104.63``
     - ``115.66``
     - ``slurm_3pcf_multipole_lmax7_full_mpi_89344.log``
   * - ``param_3pcf_multipole_lmax10_full.yaml``
     - ``8``
     - ``10``
     - ``36 x 1``
     - ``105.42 + 98.84 + 98.61``
     - ``302.90``
     - ``313.44``
     - ``slurm_3pcf_multipole_lmax10_full_mpi_89348.log``
   * - ``param_3pcf_multipole_lmax14_full.yaml``
     - ``8``
     - ``14``
     - ``30 x 4``
     - ``93.34 + 84.89 + 84.60``
     - ``262.89``
     - ``268.93``
     - ``slurm_3pcf_multipole_lmax14_full_mpi_89345.log``
   * - ``param_3pcf_multipole_J9_lmax7.yaml``
     - ``9``
     - ``7``
     - ``12 x 8``
     - ``222.20``
     - ``222.22``
     - ``238.83``
     - ``slurm_3pcf_multipole_J9_lmax7_mpi_89350.log``
   * - ``param_3pcf_multipole_J9_lmax14.yaml``
     - ``9``
     - ``14``
     - ``12 x 8``
     - ``912.81``
     - ``912.83``
     - ``923.03``
     - ``slurm_3pcf_multipole_J9_lmax14_mpi_89351.log``

The cost increases with angular basis size and field resolution, but the rows
do not isolate one variable at a time because the parallel layouts differ.
The ``J=9`` runs are substantially heavier because the coefficient grid doubles
per dimension.

Recommended Additional Logs
---------------------------

The current benchmark is useful, but these extra logs would make the comparison
cleaner:

- Run ``param_3pcf_pcenter_nrot200.yaml``, ``param_3pcf_pcenter_nrot500.yaml``, ``param_3pcf_pcenter_nrot1000.yaml``, and ``param_3pcf_pcenter_nrot2000.yaml`` with the same resources, preferably ``16 x 8``. This would make the ``n_rot`` scaling much clearer.
- Run ``param_3pcf_rcenter_nrot200.yaml`` and ``param_3pcf_rcenter_with_random.yaml`` with the same resources. This would isolate the cost of the uniform shortcut versus explicit random fields for box-random centers.
- Run ``param_3pcf_multipole_lmax7.yaml``, ``lmax10``, ``lmax14``, and ``lmax20`` under one fixed multipole layout, either all ``36 x 1`` or all ``30 x 4``. This would make the ``lmax`` trend honest.
- Run the matching ``full`` multipole configs under the same layout as the shortcut configs. This would isolate the extra cost of explicit ``ddd_l`` and ``rrr_l`` products.
- Add one ``rppi`` run with the uniform shortcut and, if diagonal LOS is important, one explicit-random diagonal ``smu`` run. This would complete the 2PCF separation between coordinate choice, LOS choice, and random-field choice.
- For publication-quality benchmark claims, keep three repeats per representative config and report mean plus standard deviation.
