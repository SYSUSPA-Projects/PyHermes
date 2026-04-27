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
- ``Tshell``: ``R_in`` and ``R_out`` for the inner and outer shell radii
- ``gaussian_shell``: ``R_shell`` for the shell-like oscillation scale and
  ``R_smooth`` for the Gaussian damping scale
- ``ring`` and ``cylinder``: ``R`` and ``H``; optional line-of-sight components
  ``nx``, ``ny``, and ``nz`` belong in ``other_args``

Corr_2PCF
---------

- ``s_min``: minimum separation
- ``s_max``: maximum separation
- ``n_s``: number of sampled separations
- ``mode``: ``s`` for ``xi(s)`` or ``smu`` for ``xi(s, mu)``
- ``window``, ``window1``, ``window2``: optional window definitions for custom
  smoothing behavior
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
- ``center``: center sampling strategy, usually ``random`` or ``particle``
- ``n_rand``: number of random centers
- ``base_seed``: seed controlling random center generation
- ``window2`` and ``window3``: window definitions for the two legs
