# Software optimization review, 2026-10-17

This review follows the exact 2PCF window-family work. It distinguishes implemented, numerically verified changes from opportunities that still require a controlled end-to-end profile. The status below includes the subsequent angular-3PCF integration; its Monte Carlo estimator is unchanged.

## Implemented 2PCF changes

The optional spectral path applies Parseval's identity to field products, reuses prepared inputs/source FFTs within a run, and retains all alias-specific scaling/folding terms. Automatic mathematical-structure metadata now selects exact radial frequency grouping or shared separable factor tables. Thin shells and Gaussian shells use the same radial backend; finite shells preserve their existing shared-boundary method. Ring, Gaussian-ring, disk, cylindrical shell/cylinder and Cartesian cubic windows share bounded factor reduction. Small query batches share traversal and each alias's scaling product without moving scaling outside the alias sum.

The controlled 85-run/31-gate report records input hashes, real physical-core bindings, phase timings, startup/warmup, numerical differences and independent ring batch RSS. Existing default computation, window definitions and public result shapes remain unchanged. Explicit custom callables remain authoritative. Unsupported/tilted/complex cases retain existing safe paths. Tests include independent materialized alias folds and end-to-end uniform/nonuniform products, not only internal implementation comparisons.

The next 2PCF priorities are usability and broader workload evidence rather than another mathematical rewrite: retain the single `spectral_window="auto"` selection, document timing boundaries, and profile short cold-start workloads separately. Dynamic Numba dispatcher cache indices have unsafe weak-reference lifetimes; disabling only those disk caches preserves correctness but can make process startup noticeable. A cacheable built-in kernel representation could be investigated later with cold-start profiles and custom-callable compatibility tests. It has not been implemented here. Scaling factorization across aliases, approximate radial interpolation and arbitrary-LOS snapping are not proposed.

## Standard angular 3PCF: implemented work and remaining opportunities

The initial review identified contrast reuse, lower-order pair-series reuse and
prepared-vertex sharing. The first two were implemented in 73df42f, received a
fallback-memory guard in 8f3106f, and were integrated into main by 1e224ab.
See [the implementation and measurements](3pcf_angular_optimization_results_20261017.md)
for local controlled profiles, catalogue DDD/Q timings and numerical checks.

1. **Implemented: box-random contrasts once per product.** The three D-R fields
   for `delta_ddd` are prepared before the angle loop and released after that
   product, including on exceptions. Particle-centre contrast handling remains
   unchanged. Centres, rotations, seeds, weights and interpolation are retained.
2. **Implemented: common local pair series.** Lower-order pairs reuse exact
   spectral powers and automatic window plans through
   `compute_pair_products_at_samples` in `theory/corr2pcf.py`. Root-only planning
   uses `COMM_SELF`; unsupported full kernels are released after each radius.
   This does not construct a collective `Corr_2PCF` task on root alone.
3. **Not implemented: shared identical prepared vertices and broadcasts.**
   Existing path resolution already caches repeated file paths. Further sharing
   would concern normalized/smoothed arrays and product broadcasts, rather than
   repeated file reading. Identity-and-window-aware sharing would need mutation,
   ownership, heterogeneous-window, cross-field and repeated-run checks.
4. **Not implemented: central interpolation reuse across angles.** The large
   catalogue DDD/Q tasks are dominated by sampled interpolation; their Q pair
   postprocessing takes about 1.8 seconds. Profile a representative case before
   changing the kernel and preserve finite samples, weights and normalization.

The historical preparation-only microprofile used the actual J=8 real-space
example field, one thread, uniform random and twenty angles. Rebuilding three
contrasts twenty times took 1.086-1.104 s; preparing them once took
0.048-0.055 s. It supported the mechanism of the now-completed first change,
not an end-to-end server speedup estimate. Its three samples remain in
`analysis/nonmultipole_contrast_microprofile.json`.

Finite Monte Carlo rotations and centre sampling define the estimator. Changing
seeds/rotation counts, substituting analytic identities for sampled terms,
truncating interpolation stencils or applying fast-math would require separate
scientific validation; none is part of the completed optimization.

## Multipoles and whole-software priorities

The current CPU/GPU Gamma contraction retains signed overlap coefficients and separable workspace reuse. Source FFT snapshots avoid repeated forward transforms but keep full complex grids resident; more MPI ranks can therefore increase total memory substantially. Layout choice should use measured task time and per-case RSS under the actual physical-core/NUMA allocation. A larger rank count is not automatically better. The CPU example refresh uses fewer ranks for J=9, explicitly records process-only THP and NUMA policies, and does not compare those timings directly with a different historical node/layout. No user GPU job is queried or changed.

Input provenance is a scientific prerequisite for result reuse. The current local example fields have different hashes from the prior performance-test fields. The historical full-ring example output is byte-identical to the reference validated with the old benchmark field, so its roughly 3.25e-7 difference from the new example calculation must be separated from same-input implementation equivalence. Exact current-input CPU multipoles are therefore refreshed rather than populated with coefficients from a different field.

Further implementation should begin with a measured bottleneck, such as cold
startup, sampled interpolation or prepared-vertex memory. The contrast cache
and common local pair helper are complete. Keep reusing shared window metadata,
contraction helpers and input audits; do not add one public switch or reduction
kernel per window. No additional production matrix is required for this review.

## Benchmark observation overhead

The example refresh records parent RSS sampling work separately. For the first
J=8 multipole example it spent 222.277 seconds in observations during a
455.978-second task, with 2318 samples. The configured 0.1-second sleep therefore
gave a mean cadence of about 0.197 seconds, including observation work. This
is concurrent parent wall time, not a measured CPU cost or a subtractable
component of scientific wall time. No completed result is rerun or adjusted
to conceal this cost. The example timings describe the observed execution.

A future benchmark-harness improvement could track the small launcher/worker
process tree directly rather than repeatedly scanning host process metadata.
It needs its own overhead and RSS-coverage check. The completed frozen
experiment remains unchanged. This is separate from the scientific estimator
optimizations and from whole-job Slurm batch-step MaxRSS.
