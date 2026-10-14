Version 1.2 migration
======================

Gamma correction
----------------

Recompute affected Gamma three-point multipole measurements from 1.1 using
1.2. The corrected implementation zero-extends compact basis samples and
retains the complete positive/negative displacement overlap table. Existing
SFC catalogue projections can be reused; the correction does not change
2PCFs or the direct-angular Monte Carlo 3PCF estimator.

Optimization and particle input
-------------------------------

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

Historical outputs
------------------

The two paper multipole figures and current runtime figure use corrected
archived results. Older multipole sections of the public 3PCF notebook and
other unrecomputed convergence plots are labelled historical and must not be
used as validation of 1.2. They are retained as examples of analysis layout.
No heavy scientific jobs execute during documentation builds.

See the :doc:`multipole task guide <get_start/corr_3pcf_multipole/corr_3pcf_multipole>`
for runnable examples and :doc:`benchmark` for measurement boundaries.
