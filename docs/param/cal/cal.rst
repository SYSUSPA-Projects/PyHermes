Calculation
============

This section summarizes the main task-specific calculation parameters.

Convols
-------

- ``J``: multiresolution level used for the field representation
- ``SampRate``: sampling rate
- ``SimBoxL``: simulation box size
- ``wavelet_mode``: wavelet family
- ``wavelet_level``: wavelet decomposition depth

Counting
--------

- ``N_randoms``: number of random points
- ``window.type``: window shape used before sampling
- ``window.len_args``: scale parameters for the window
- ``seed``: random seed

Corr_2PCF
---------

- ``r_min``: minimum separation
- ``r_max``: maximum separation
- ``n_r``: number of radial bins
- ``window``, ``window1``, ``window2``: optional window definitions for custom
  smoothing behavior

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
