Calculation
============

This section summarizes the main task-specific calculation parameters.

Convols
-------

- ``J``: multiresolution level used for the field representation
- ``phi_resolution``: number of samples used to tabulate ``phi``, the wavelet scaling function
- ``box_size``: simulation box size
- ``wavelet_mode``: wavelet family
- ``wavelet_level``: wavelet decomposition depth
- ``save_particle_data``: save particle positions and weights to a companion ``.npz`` file
- ``particle_data_path``: optional companion particle-data path; derived from ``fout_path`` when empty

Counting
--------

- ``N_randoms``: number of random points
- ``window.type``: window shape used before sampling
- ``window.len_args``: scale parameters for the window
- ``seed``: random seed

Window Parameter Names
----------------------

Window dictionaries pass ``len_args`` directly to the selected window function.
Common radial windows use the following length parameters:

- ``sphere``, ``gaussian``, ``shell``: ``R``
- ``gaussian_shell``: ``R_shell`` for the shell-like oscillation scale and
  ``R_smooth`` for the Gaussian damping scale
- ``cubic``: ``Lx``, ``Ly``, and ``Lz`` for the three axis-aligned side lengths
- ``ring``, ``disk``, and ``cylinder``: ``R`` and ``H``; optional
  line-of-sight components ``nx``, ``ny``, and ``nz`` belong in
  ``los_args``

Corr_2PCF
---------

- ``sampling``: coordinate dictionary for the requested output grid. Supported
  coordinate sets are ``s`` for ``xi(s)``, ``s`` and ``mu`` for
  ``xi(s, mu)``, or ``rp`` and ``pi`` for ``xi(rp, pi)``.
- ``window``, ``window1``, ``window2``: optional window definitions for custom
  smoothing behavior
- ``pair_window.mapping``: coordinate mapping from sampling variables to
  pair-window length arguments. ``s_to_R`` maps ``R=s``; ``smu_to_RH`` maps
  ``R=s sqrt(1-mu^2)``, ``H=s mu``; and ``rppi_to_RH`` maps ``R=rp``,
  ``H=pi``. In ``Corr_2PCF`` pair windows, ``None`` marks a runtime
  placeholder; fixed numeric values in ``len_args`` are left unchanged. LOS
  information belongs in ``pair_window.los_args``.
- ``pair_window.kernel_mode``: kernel construction strategy. ``full_rfft``
  evaluates the full real-FFT kernel and is the default for custom windows.
  ``octant`` uses symmetry folding and is appropriate only for windows with the
  required octant symmetries. ``auto`` uses folding for coordinate-axis LOS
  directions and full real-FFT otherwise; built-in ``ring``, ``disk``, and
  ``cylinder`` pair windows default to ``auto``.
  ``octant`` is mathematically safe only when the k-space window is invariant
  under independent sign flips of all three components:
  ``W(kx, ky, kz) = W(-kx, ky, kz) = W(kx, -ky, kz) = W(kx, ky, -kz)``.
  Isotropic windows satisfy this condition. Axis-aligned LOS windows can also
  satisfy it if the parallel dependence is even, for example through ``cos`` or
  ``sin(q)/q``. Oblique LOS windows, for example
  ``los_args: [1, 1, 1]``, generally do not satisfy the condition and should
  use ``full_rfft`` unless the user has proven the required symmetry. This
  criterion is about symmetry with respect to the FFT grid coordinates, not
  just the apparent geometric symmetry of the window: for example, a ring with
  ``los_args: [1, 1, 0]`` may look symmetric in a rotated coordinate system,
  but it is not invariant under independent ``kx`` and ``ky`` sign flips in
  the original grid coordinates.
- ``memory_strategy``: ``speed`` keeps more fields in memory to reuse pair
  windows across products; ``memory`` computes product groups sequentially to
  reduce peak memory
- ``pair_window_cache``: optional disk cache for pair-window kernels, most
  useful with ``memory_strategy: memory``
- ``pair_window_cache_dir``: directory for cached pair-window kernels

Corr_3PCF
---------

- ``r12``: first side length
- ``r13``: second side length
- ``n_theta``: number of angular bins
- ``n_rot``: number of rotations
- ``center``: center sampling strategy, usually ``box_random`` or ``particle``
- ``n_box_centers``: number of Monte Carlo centers for ``center: "box_random"``
- ``base_seed``: seed controlling random center generation
- ``window2`` and ``window3``: window definitions for the two displaced legs
- ``window1``: optional center-leg window, available when ``center: "box_random"``
