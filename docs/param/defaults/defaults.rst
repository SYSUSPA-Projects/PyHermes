Default Parameter Guide
=======================

PyHermes ships task-level default parameter dictionaries in:

- ``pyhermes/base/default_params.json``
- ``pyhermes/theory/default_params.json``

These files define the fallback values used when a field is not specified in
your YAML or JSON5 configuration. In practice, most users only override a small
subset of these keys.

This page explains the meaning of the most important default parameters used by
the main public tasks.

Convols
-------

The ``Convols`` defaults come from ``pyhermes/base/default_params.json``.

Core field-construction parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``J``:
  multiresolution level. The field size is ``L = 2^J``.
- ``phi_resolution``:
  number of samples used to tabulate ``phi``, the wavelet scaling function.
- ``box_size``:
  physical side length of the simulation box.
- ``wavelet_mode``:
  wavelet family name, such as ``"db2"``.
- ``wavelet_level``:
  level used in the wavelet decomposition.
- ``threads``:
  CPU threads per MPI rank.

Particle input parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``particle_pos``:
  optional in-memory particle position array. If provided, it can be used
  instead of reading particle positions from file.
- ``particle_weight``:
  optional in-memory particle weights. If omitted, unit weights are assumed.
- ``save_particle_data``:
  whether to save particle positions and weights to a companion ``.npz`` file.
- ``particle_data_path``:
  optional companion particle-data path. If empty, PyHermes derives it from
  ``fout_path``.
- ``fin.path``:
  local path to the particle catalog.
- ``fin.url``:
  optional remote URL. If this is non-empty, PyHermes downloads the file and
  stores it at ``fin.path`` before reading it.
- ``fin.format``:
  particle file format, for example ``generic_pos``.
- ``fin.weight_key``:
  particle weight field name, or ``unit`` if unweighted counts are desired.

Output
^^^^^^

- ``fout_path``:
  path where the serialized ``ConvolsData`` result is written.

Counting
--------

The ``Counting`` defaults come from ``pyhermes/theory/default_params.json``.

- ``seed``:
  random seed for the sampled positions.
- ``N_randoms``:
  number of random positions to evaluate.
- ``convols_data_path``:
  fallback path to the input ``ConvolsData`` file.
- ``window``:
  optional smoothing window applied before counting.
- ``threads``:
  CPU threads per MPI rank.
- ``fout_path``:
  output path for the serialized counting result.

Corr_2PCF
---------

- ``convols_data_path``:
  shared fallback input field path.
- ``convols_data1_path`` and ``convols_data2_path``:
  optional leg-specific field paths. If empty, they fall back to
  ``convols_data_path``.
- ``window``:
  shared fallback smoothing window for the two legs.
- ``window1`` and ``window2``:
  optional leg-specific smoothing windows.
- ``pair_window``:
  kernel template used in the pair-correlation measurement itself. The default
  is a shell window with ``mapping: s_to_R``. Built-in mappings are
  ``s_to_R``, ``smu_to_RH``, and ``rppi_to_RH``; the ``sampling`` keys must
  match the selected mapping exactly. Only ``None`` values in ``len_args`` are
  filled at runtime by the mapping; fixed numeric values are left unchanged.
  LOS information belongs in ``pair_window.los_args``.
- ``pair_window.kernel_mode``:
  kernel construction strategy. ``full_rfft`` evaluates the full real-FFT
  kernel and is the default for custom windows. ``octant`` uses symmetry
  folding. ``auto`` folds only for coordinate-axis LOS directions and otherwise
  uses ``full_rfft``; built-in ``ring``, ``disk``, and ``cylinder`` pair
  windows default to ``auto``. ``octant`` is safe only when
  ``W(kx, ky, kz)`` is unchanged by independent sign flips of ``kx``, ``ky``,
  and ``kz``. Isotropic windows satisfy this automatically; oblique LOS
  anisotropic windows generally require ``full_rfft``. This is a symmetry test
  in the FFT grid coordinates, not merely a visual shape-symmetry test.
- ``sampling``:
  coordinate specification for the output grid. Supported built-in mapping
  coordinate sets are ``s``, ``s`` and ``mu``, or ``rp`` and ``pi``.
- ``pair_window.los_args``:
  line-of-sight direction for LOS-aware pair windows, expressed as
  ``[nx, ny, nz]`` or a dictionary with ``nx``, ``ny``, and ``nz``.
- ``threads``:
  CPU threads per MPI rank.
- ``memory_strategy``:
  ``speed`` keeps all required fields resident and reuses each pair window
  across products at a sampling point. ``memory`` computes product groups in
  sequence to reduce peak memory, at the cost of rebuilding pair windows.
- ``pair_window_cache``:
  optional disk cache for pair-window kernels, useful with
  ``memory_strategy: memory`` when repeated product groups would otherwise
  rebuild the same kernels. Disabled by default.
- ``pair_window_cache_dir``:
  directory used when ``pair_window_cache`` is enabled. If empty, PyHermes
  derives a cache directory from ``fout_path``.
- ``fout_path``:
  output path for the 2PCF result.

Corr_3PCF
---------

- ``convols_data_path``:
  shared fallback input field path.
- ``convols_data1_path``, ``convols_data2_path``, ``convols_data3_path``:
  optional leg-specific input paths.
- ``window``:
  shared fallback smoothing window.
- ``window1``, ``window2``, ``window3``:
  optional leg-specific smoothing windows.
- ``r12`` and ``r13``:
  triangle side lengths used to define the 3PCF family.
- ``n_theta``:
  number of angular bins.
- ``n_rot``:
  number of rotations used in the Monte Carlo estimator.
- ``center``:
  center sampling mode, usually ``random`` or ``particle``.
- ``field_mode``:
  either ``delta`` or ``raw``.
- ``n_rand``:
  total number of random centers when ``center="random"``.
- ``base_seed``:
  random seed controlling reproducibility.
- ``threads``:
  CPU threads per MPI rank.
- ``fout_path``:
  output path for the 3PCF result.

Corr_3PCF_Multipole
-------------------

- ``convols_data_path``:
  shared fallback input field path.
- ``convols_data1_path``, ``convols_data2_path``, ``convols_data3_path``:
  optional leg-specific input paths.
- ``window``:
  shared fallback smoothing window.
- ``window1``, ``window2``, ``window3``:
  optional leg-specific smoothing windows.
- ``r12`` and ``r13``:
  side lengths for the multipole family.
- ``l_min`` and ``l_max``:
  minimum and maximum multipole order.
- ``gpu_device_id``:
  CUDA device index used by the GPU summation stage.
- ``field_mode``:
  either ``delta`` or ``raw``.
- ``execution_mode``:
  ``serial`` or ``pair_mpi``.
- ``cache_multipole_fields``:
  whether to write intermediate multipole convolution fields to disk.
- ``cache_dir``:
  directory used for the optional intermediate cache.
- ``verbose_m_progress``:
  whether to print detailed progress and timing information for each multipole.
- ``threads``:
  CPU threads per MPI rank.
- ``fout_path``:
  output path for the multipole result.

Notes
-----

- Empty path fields usually mean “use the shared fallback path instead”.
- Window dictionaries follow the common PyHermes structure:

  .. code-block:: yaml

     window:
        type: "sphere"
        len_args:
           R: 20
        other_args: {}

  For finite shell windows, use descriptive length names:
  ``Tshell`` expects ``R_in`` and ``R_out``; ``gaussian_shell`` expects
  ``R_shell`` and ``R_smooth``.

- In most workflows, you only need to override a small subset of the defaults.
  The rest can safely remain untouched.
