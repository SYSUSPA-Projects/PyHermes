Bundled defaults
================

Defaults live beside the task implementations:

- ``pyhermes/base/default_params.json`` for ``SFCProjection``
- ``pyhermes/theory/default_params.json`` for the statistical tasks

They are JSON5 files used internally by ``read_param``. Do not edit them for an
individual analysis; override values in a user YAML file instead.

SFCProjection
-------------

.. list-table:: Field-construction defaults
   :header-rows: 1
   :widths: 28 20 52

   * - Key
     - Default
     - Purpose
   * - ``box_size``
     - ``1000``
     - Periodic-box side length in the coordinate units of ``particle_pos``.
   * - ``J``
     - ``8``
     - Dilation level, with :math:`L=2^J` cells per axis.
   * - ``wavelet_mode``
     - ``db2``
     - PyWavelets scaling-function family.
   * - ``wavelet_level``
     - ``10``
     - Refinement depth used to tabulate the scaling function.
   * - ``phi_resolution``
     - ``1024``
     - Sampling resolution of the tabulated scaling function.
   * - ``weight_normalization``
     - ``catalog``
     - Normalization of ``catalog_weight * field_value``.
   * - ``save_particle_data``
     - ``false``
     - Save positions and projected particle weights for particle-centred
       estimators.
   * - ``threads``
     - ``1``
     - Numba threads per MPI rank.

``particle_pos``, ``catalog_weight``, and ``field_value`` default to ``null``;
file input then comes from ``fin``. The default reader is a three-column
``float32`` raw binary table. ``fout_path`` and ``particle_data_path`` are empty
until supplied by the user.

Counting
--------

``random_count=1000000`` and ``seed=42`` define the sampling positions. The
default ``window.type`` is empty, so no optional smoothing is applied.
``weight_normalization=catalog`` converts catalogue fields to their standard
normalized amplitude before sampling.

Corr_2PCF
---------

The shared field and random inputs are empty by default. The default estimator
uses:

.. code-block:: yaml

   weight_normalization: "catalog"
   binning_window:
      type: "shell"
      len_args: ["R"]
      mapping: "s_to_R"
   sampling:
      s: {min: 1.0, max: 150.0, n: 30}
   products: "xi"
   memory_strategy: "speed"
   binning_window_cache: false
   threads: 1

The list form ``len_args: ["R"]`` is normalized internally into a runtime
placeholder. User configurations are clearer when written explicitly as
``len_args: {R: null}``.

Corr_3PCF
---------

The default direct-angular triangle has ``r12=20``, ``r13=40``, 20 angular
samples, 100 rotations, and ``center=box_random`` with five million centres.
``angle_param=theta``, ``base_seed=42``, and ``products=Q``. Shared and
leg-specific fields, randoms, and windows remain empty until configured.

Corr_3PCF_Multipole
-------------------

Multipole radii have no top-level defaults. Both edge windows and ``sampling``
must be supplied. The execution defaults are:

.. code-block:: yaml

   l_min: 0
   l_max: 4
   products: "zeta_l"
   execution_mode: "serial"
   summation_backend: "gpu"
   gpu_device_id: 0
   gpu_threads_per_block: [8, 8, 8]
   sample_mpi:
      ranks_per_sample: 1
      gpu_device_ids: []
   cache_multipole_fields: false
   verbose_m_progress: false
   verbose_profile: false
   radial_profile_diagnostics: true
   radial_profile_diagnostic_tolerance: 1.0e-5
   radial_profile_diagnostic_probes: 33
   zeta_condition_warning: 1.0e12
   threads: 1

The default GPU backend requires a usable CUDA device. Set
``summation_backend: cpu`` explicitly on CPU-only systems.

Resolution and memory
---------------------

The deceptively small ``J`` parameter controls a three-dimensional array:

.. math::

   N_{\rm MRA}=L^3=2^{3J}.

Increasing ``J`` by one multiplies the coefficient count by eight. Treat a
change from :math:`J=8` to :math:`J=9` as a new resource regime, not a cosmetic
resolution tweak.
