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

SFCProjection
-------

The ``SFCProjection`` defaults come from ``pyhermes/base/default_params.json``.

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
- ``catalog_weight``:
  optional in-memory observational or selection weights. If omitted, unit
  catalogue weights are assumed.
- ``field_value``:
  optional in-memory per-object physical values, such as mass or one velocity
  component. If omitted, unit values are assumed.
- ``weight_normalization``:
  projection convention for ``catalog_weight * field_value``. ``catalog`` is
  the default and divides by the catalogue-weight sum. ``raw`` keeps the raw
  weighted field. ``field`` divides by the raw weighted field sum. ``unit`` is
  accepted as a construction-time alias for ``field``.
- ``save_particle_data``:
  whether to save particle positions and weights to a companion ``.npz`` file.
- ``particle_data_path``:
  optional companion particle-data path. If empty, PyHermes derives it from
  ``fout_path``.
- ``fin.path``:
  local path to the particle catalog.
- ``fin.format``:
  particle file format, for example ``bin``.
- ``fin.reader_params``:
  format-specific reader options.
- ``fin.catalog_weight_key``:
  one-dimensional catalogue-weight field name, or ``null`` for unit weights.
- ``fin.field_value_key``:
  one-dimensional physical-field value name, or ``null`` for unit values.

Output
^^^^^^

- ``fout_path``:
  path where the serialized ``SFCField`` result is written.

Counting
--------

The ``Counting`` defaults come from ``pyhermes/theory/default_params.json``.

- ``seed``:
  random seed for the sampled positions.
- ``N_randoms``:
  number of random positions to evaluate.
- ``sfc_field_path``:
  fallback path to the input ``SFCField`` file.
- ``window``:
  optional smoothing window applied before counting.
- ``weight_normalization``:
  normalization applied before the optional counting window. ``raw``,
  ``catalog`` and ``field`` apply to catalog fields; ``unit`` rescales either
  catalog or derived fields to unit field integral.
- ``threads``:
  CPU threads per MPI rank.
- ``fout_path``:
  output path for the serialized counting result.

Corr_2PCF
---------

- ``sfc_field_path``:
  shared fallback input field path.
- ``sfc_field1_path`` and ``sfc_field2_path``:
  optional leg-specific field paths. If empty, they fall back to
  ``sfc_field_path``.
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
- ``weight_normalization``:
  input-weight convention for catalog fields. ``catalog`` (default) divides by
  ``catalog_weight_sum``. ``raw`` keeps the raw weighted amplitude.
  ``field`` divides by ``raw_field_weighted_sum`` for positive marked fields.
  ``unit`` rescales either catalog or derived fields to unit field integral.
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

- ``sfc_field_path``:
  shared fallback input field path.
- ``sfc_field1_path``, ``sfc_field2_path``, ``sfc_field3_path``:
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
  center sampling mode, usually ``box_random`` or ``particle``.
- ``field_mode``:
  either ``delta`` or ``raw``.
- ``n_box_centers``:
  total number of Monte Carlo centers when ``center="box_random"``.
- ``base_seed``:
  random seed controlling reproducibility.
- ``threads``:
  CPU threads per MPI rank.
- ``weight_normalization``:
  same input-weight choices as ``Corr_2PCF``. With particle centers,
  automatically recovered center marks follow this choice; explicitly supplied
  ``particle_pos1`` and ``particle_weight1`` must be provided together and are
  used as given.
- ``fout_path``:
  output path for the 3PCF result.

Corr_3PCF_Multipole
-------------------

- ``sfc_field_path``:
  shared fallback input field path.
- ``sfc_field1_path``, ``sfc_field2_path``, ``sfc_field3_path``:
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
- ``weight_normalization``:
  same input-weight choices as ``Corr_2PCF``.
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

  Use descriptive length names for custom and compound windows. For example,
  a custom finite shell can use ``R_in`` and ``R_out``; ``gaussian_shell`` uses
  ``R_shell`` and ``R_smooth``; ``cubic`` uses ``Lx``, ``Ly``, and ``Lz``.

- In most workflows, you only need to override a small subset of the defaults.
  The rest can safely remain untouched.
