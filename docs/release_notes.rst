Release notes and migration
===========================

Unreleased source updates
-------------------------

The changes below are integrated into source ``main`` after the 1.2.0 changes
described later on this page. They are tracked under ``Unreleased`` in the
repository changelog; this documentation does not announce a new PyPI release.
Use an installation of the same checkout as its examples; see
:doc:`install`. The existing package-version string alone does not identify a
development commit, so retain the source commit for reproducible measurements.

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

The tracked 2PCF YAML files opt into automatic spectral strategies and use
``_optimized_20261017`` output names. The three refreshed CPU Gamma examples use
``_gamma_corrected_20261017`` names. The 2PCF and 3PCF notebook cells that load
these results have been executed and their figures checked; old output files
are retained. The Quick Start and task guides read output paths from those
same configurations.

The retained direct-angular curves do not record the exact coefficient-array
input SHA used by the refreshed multipoles. Their overlay is therefore a
diagnostic, not strict identical-input validation. Other unrecomputed multipole
and ensemble figures remain explicitly historical. Details and observed
timings are in :download:`the verified example refresh
</examples_refresh_results_20261017.md>`.

Version 1.2.0 migration
-----------------------

Gamma correction
~~~~~~~~~~~~~~~~

Recompute affected Gamma three-point multipole measurements from 1.1 using
1.2. The corrected implementation zero-extends compact basis samples and
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
