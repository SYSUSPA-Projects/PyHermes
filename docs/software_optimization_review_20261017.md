# Software optimization review, 2026-10-17

This review follows the exact 2PCF window-family work. It distinguishes implemented, numerically verified changes from opportunities that still require a controlled end-to-end profile. No standard angular-3PCF scientific algorithm is changed by this review.

## Implemented 2PCF changes

The optional spectral path applies Parseval's identity to field products, reuses prepared inputs/source FFTs within a run, and retains all alias-specific scaling/folding terms. Automatic mathematical-structure metadata now selects exact radial frequency grouping or shared separable factor tables. Thin shells and Gaussian shells use the same radial backend; finite shells preserve their existing shared-boundary method. Ring, Gaussian-ring, disk, cylindrical shell/cylinder and Cartesian cubic windows share bounded factor reduction. Small query batches share traversal and each alias's scaling product without moving scaling outside the alias sum.

The controlled 85-run/31-gate report records input hashes, real physical-core bindings, phase timings, startup/warmup, numerical differences and independent ring batch RSS. Existing default computation, window definitions and public result shapes remain unchanged. Explicit custom callables remain authoritative. Unsupported/tilted/complex cases retain existing safe paths. Tests include independent materialized alias folds and end-to-end uniform/nonuniform products, not only internal implementation comparisons.

The next 2PCF priorities are usability and broader workload evidence rather than another mathematical rewrite: retain the single `spectral_window="auto"` selection, document timing boundaries, and profile short cold-start workloads separately. Dynamic Numba dispatcher cache indices have unsafe weak-reference lifetimes; disabling only those disk caches preserves correctness but can make process startup noticeable. A cacheable built-in kernel representation could be investigated later with cold-start profiles and custom-callable compatibility tests. It has not been implemented here. Scaling factorization across aliases, approximate radial interpolation and arbitrary-LOS snapping are not proposed.

## Standard angular 3PCF

The opportunities below describe the initial main-branch review. The later experimental branch implements the first two while retaining the Monte Carlo estimator; see [the implementation and controlled local profiles](3pcf_angular_optimization_results_20261017.md). Vertex/broadcast sharing remains a separate future step.

Three concrete opportunities appear in `Corr_3PCF`:

1. **Prepare box-random contrast fields once per product.** `_compute_random_center_mu` currently constructs all three D-R fields for every angle of `delta_ddd`. These fields do not depend on angle. The particle-center branch already prepares its two contrast arms outside the angle loop. A product-local cache, cleared at `_release_product_runtime` and rebuilt on every run, would remove repeated full-grid allocation/subtraction without changing centers, rotations, seeds, interpolation or normalization. This is a small, readable future change, rather than a claim of a large estimator speedup.
2. **Reuse the low-order pair-statistic backend.** `_compute_pair_stats_series` already prepares two contrast fields once, but calls the convolution-based pair estimator separately at each radius. The exact spectral/radial helpers could compute these terms from shared power and window metadata. This must use a genuinely local helper on the root: constructing a collective `Corr_2PCF` task inside root-only postprocessing would deadlock. Measure the existing `2PCF timing` breakdown first; the benefit depends on how much of Q comes from this postprocessing versus Monte Carlo interpolation.
3. **Share identical prepared vertices and broadcasts within a product.** The path resolver already caches repeated file paths, so this is not a proposal to remove nonexistent repeated file reading. Normalization/smoothing still creates separate vertex arrays, and product runtime broadcasts vertices individually. Exact identity-and-window-aware sharing could reduce memory and communication. It needs mutation/aliasing, heterogeneous-window, cross-field and repeated-run tests. The memory-saving execution plan must remain available.

A local preparation-only profile used the actual J=8 real-space example field, one thread, uniform random and twenty angles. Rebuilding three contrast fields twenty times took 1.086–1.104 s; preparing them once took 0.048–0.055 s. This is a Mac microprofile of allocation/subtraction, with no Monte Carlo, FFT or Slurm workload. It supports the mechanism of opportunity (1), not a server-wide or end-to-end speedup estimate. A long Monte Carlo task may gain little from that approximately one-second local preparation saving. All three samples are retained in `analysis/nonmultipole_contrast_microprofile.json`.

Finite Monte Carlo rotations and particle-center sampling define the estimator. Replacing finite sampled terms by analytic identities, changing seeds/rotation counts, truncating interpolation stencils, or applying fast-math without independent numerical checks would change the measurement. These are outside this optimization review.

## Multipoles and whole-software priorities

The current CPU/GPU Gamma contraction retains signed overlap coefficients and separable workspace reuse. Source FFT snapshots avoid repeated forward transforms but keep full complex grids resident; more MPI ranks can therefore increase total memory substantially. Layout choice should use measured task time and per-case RSS under the actual physical-core/NUMA allocation. A larger rank count is not automatically better. The CPU example refresh uses fewer ranks for J=9, explicitly records process-only THP and NUMA policies, and does not compare those timings directly with a different historical node/layout. No user GPU job is queried or changed.

Input provenance is a scientific prerequisite for result reuse. The current local example fields have different hashes from the prior performance-test fields. The historical full-ring example output is byte-identical to the reference validated with the old benchmark field, so its roughly 3.25e-7 difference from the new example calculation must be separated from same-input implementation equivalence. Exact current-input CPU multipoles are therefore refreshed rather than populated with coefficients from a different field.

A sensible next implementation sequence is to profile the existing 3PCF phase logs, make the small product-local contrast cache change if worthwhile, then introduce one shared local pair-reduction helper if pair postprocessing is material. Vertex sharing follows only with ownership/lifetime tests. Broader work should continue to reuse the common window metadata, contraction helpers and source/input audit machinery rather than adding a new public flag or a dedicated reduction kernel for every window. No further production matrix is submitted merely for this review.

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
