Release notes and migration
===========================

Version 1.2.0 (2026-10-20)
--------------------------

Version 1.2.0 was released on 2026-10-20. This page collects the changes since 1.1.0,
including the 2PCF optimizations, angular-3PCF reuse, Gamma correction and
particle multipoles. See :doc:`install` for installation options. Keep the
installed code and examples on the same version or commit, and retain the
commit for reproducible source measurements.

2PCF: one optional execution strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended opt-in configuration is ``pair_reduction: spectral``,
``reuse_input_fields: true``, ``spectral_window: auto`` and
``spectral_fft: shared``, with ``memory_strategy: speed``.
``auto`` shares exact radial grouping for built-in isotropic windows and
bounded factor tables/batches for grid-axis axial windows and Cartesian cubic
windows. Finite shells retain common-boundary evaluation. Unsupported windows
keep their applicable fallback; custom callables remain authoritative.

Defaults remain convolution, independent preparation, materialized windows and
per-rank FFT preparation. Constructor/run interfaces and result shapes are
unchanged. Every alias keeps its scaling weight within the original fold;
there is no interpolation or new scientific approximation. These optimizations
do not themselves require recomputing an existing valid 2PCF result. Choose
timing boundaries deliberately: the main loop excludes the initial FFT/power
setup, while preprojected-field task total includes input IO and preparation.
See :doc:`get_start/corr_2pcf/optimization` for options, memory limits and
verified measurements.

2PCF: direct field-pair evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SFCField.pair_with(other)`` prepares a reusable local pair of real fields on
the same MRA grid and basis, and
``pair.evaluate(windows)`` evaluates its windowed coefficient-array means in
input order. It shares the existing spectral reducer and automatic window
strategies. The caller supplies estimator normalization; the uniform-reference
correlation divides a contrast-field pair mean by ``rho**2``. Use a ``with``
block to release the retained powers, and keep input arrays unchanged until
the context closes. Complex-response windows preserve phase through the
existing convolution fallback.

This local interface adds no task setting and changes no ``Corr_2PCF`` default,
estimator or result format. It performs no MPI world collectives; distributed
sampling and shared MPI FFT preparation remain responsibilities of the task.
See :ref:`corr2pcf-local-pair-evaluation` and Section 2 of the
:doc:`2PCF notebook <notebooks/corr2pcf>` for usage and timing boundaries.

Standard angular 3PCF: internal reuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Corr_3PCF`` reuses box-random contrast fields within one product and the exact
local 2PCF backend for lower-order pair series. It adds no user option and
preserves centres, per-rank seeds, rotation counts, interpolation and weights.
Root-only pair work uses local communication. Existing valid angular results
remain usable; finite Monte Carlo samples can change when the MPI rank count
changes, independently of this optimization.

Triplet-only DDD and Q are different workloads: Q additionally needs connected
products and pair postprocessing. Long Monte Carlo loops can gain little from
faster preparation/pair work. See :doc:`get_start/corr_3pcf/corr_3pcf` and
:doc:`benchmark` for measured timing and memory boundaries.

Examples and retained results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 2PCF notebook computes isotropic, full 46-by-51 ring and Gaussian-ring
curves locally with one process and eight CPU threads, using the projected
example fields. It demonstrates both direct convolution and local Parseval
evaluation. The tracked 2PCF YAML files opt into automatic spectral strategies.

The 3PCF notebook runs particle multipoles directly on the example catalogue,
compares saved Gamma multipoles, and includes CPU/GPU performance plotting.
Examples use the shared task drivers and YAML configurations; redundant
standalone demos and plotting scripts have been removed. Saved ``zeta_l``
reconstructs the angular curve without an additional ``2*ell+1`` multiplier.
See :doc:`benchmark` for measurements and timing/memory definitions.

Migration from 1.1.0
--------------------

Gamma correction
~~~~~~~~~~~~~~~~

Recompute affected Gamma three-point multipole measurements from 1.1.0 using
1.2.0. The corrected implementation zero-extends compact basis samples and
retains the complete positive/negative displacement overlap table. Existing
SFC catalogue projections can be reused; this numerical correction does not
change 2PCFs or the direct-angular Monte Carlo 3PCF estimator.

Optimization and particle input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CPU Gamma contraction reuses exact signed stencils and workspace; harmonic
arms reuse a source FFT. Retained spectra increase memory, and source arrays
must remain immutable during their context lifetime. Timing improvements are
workload-specific.

The experimental particle option uses the existing multipole task with
``summation_method: particle`` and ``summation_backend: cpu``. It requires
an unfiltered first vertex and retrievable particle data. Endpoint smoothing
remains available. Three-vertex smoothing requires Gamma summation. Defaults
remain ``gamma/gpu`` for compatibility; CPU-only users must select ``cpu``.

Saved ``zeta_l`` reconstructs the angular curve as the sum of its coefficients
times Legendre polynomials, with no additional ``2*ell+1`` multiplier.

See the :doc:`multipole task guide <get_start/corr_3pcf_multipole/corr_3pcf_multipole>`
for runnable examples. Documentation builds render saved notebook outputs;
they do not execute heavy scientific jobs.
