# Changelog

## 1.2.0 - 2026-10-20

Released on 2026-10-20. Changes since 1.1.0 are listed below.

### Gamma multipole correction and migration

- Fix the Gamma table for isotropic three-point multipoles. Compact basis
  samples are zero outside their support, with the full positive and negative
  displacement range. Periodic wrapping applies to coefficient grids, not
  local basis samples. Previous cyclic basis shifts and incomplete
  displacement ranges could change multipoles.
- Recompute affected Gamma multipole measurements from 1.1.0. Existing SFC
  catalogue projections can be reused. Two-point and direct-angular Monte
  Carlo estimators are unchanged by this numerical correction.
- Validate the corrected contraction against independent periodized-basis
  integrals, complex fields, symmetries, uniform backgrounds and CPU/CUDA
  comparisons.

### Optional 2PCF spectral computation

- Add Parseval-based pair reduction, compatible input preparation reuse,
  fused folded-window evaluation and shared finite-shell boundaries.
- Add batched boundary evaluation, root-computed MPI spectra and exact integer
  radial grouping. Preserve all alias-specific scaling weights and folding
  multiplicities; no interpolation or frequency cutoff is introduced.
- Add `spectral_window="auto"` to share exact radial grouping for built-in
  isotropic windows, common finite-shell boundaries, and bounded separable
  factors for grid-axis ring/Gaussian-ring, disk, cylindrical and Cartesian
  cubic windows. Preserve custom callables and applicable fallback paths.
- Share factor tables and batches of up to eight compatible queries within
  per-run cache and scratch-memory limits.
- Preserve task defaults: `pair_reduction="convolution"`,
  `reuse_input_fields=False`, `spectral_window="materialized"` and
  `spectral_fft="per_rank"`. The optimized example configurations opt in to
  the spectral path; existing valid 2PCF results do not require recomputation.

### Direct field-pair evaluation

- Add `SFCField.pair_with(other)` and `PairProduct.evaluate(windows)` for
  local averages of real fields on the same MRA grid and basis, with shared
  spectral and window preparation.
  Return coefficient-array means in input order; estimator normalization
  remains the caller's responsibility.
- Reuse the task's exact spectral/window backends without MPI world
  collectives. Preserve complex-response phase through the convolution
  fallback. Use a context manager to release retained powers and keep source
  arrays unchanged while the pair is alive.

### Standard angular 3PCF

- Prepare box-random D-R fields once per product/rank and release them on
  completion or failure. Reuse the exact local 2PCF backend for lower-order
  pair series and `Q` postprocessing; root-only planning uses `COMM_SELF`.
- Keep Monte Carlo kernels, centres, seeds, rotations, weights and
  interpolation unchanged. Add no user setting or result-format change.
- Validate weighted/nonuniform inputs, MPI layouts, repeated runs, fallback
  memory and cleanup. Report local profiles and catalogue measurements with
  their timing boundaries, rather than claiming a universal speedup.

### Multipole performance and particle summation

- Reuse CPU Gamma indices, exactly nonzero signed coefficients and work
  arrays without tolerance or rank truncation. Reuse source FFTs across
  harmonic arms; retained spectra trade memory for computation and require
  immutable source arrays during their context lifetime.
- Preserve source precision. Use the ordinary inverse-FFT path for
  single-precision spectra to match generic convolution across SciPy
  versions; retain the double-precision in-place path.
- Add experimental `summation_method="particle"` with
  `summation_backend="cpu"` to `Corr_3PCF_Multipole`, supporting serial,
  pair-MPI and sample-MPI execution through the existing result interface.
- Require an unfiltered first vertex and a retrievable particle catalogue.
  Preserve effective projection weights, marks, data-minus-random subtraction
  and catalogue or uniform-random centres. Endpoint smoothing is supported;
  particle CUDA and filtered-centre particle summation are rejected.
- Preserve multipole defaults (`gamma`, `gpu`). Particle and Gamma methods
  have different finite-J discretizations; equality at fixed J is not an
  absolute-accuracy criterion.
- Support directory and plain-filename multipole output paths, and restore
  `task_params` metadata when loading saved results.

### Examples, documentation and packaging

- Explain convolution and Parseval reconstruction in the 2PCF notebook,
  including normalization, task options and preparation/main-loop timing.
  Compute full 46-by-51 ring and Gaussian-ring curves directly in the notebook
  with one process and eight CPU threads.
- Add a runnable catalogue particle-multipole example and CPU/GPU performance
  plotting to the 3PCF notebook, using the shared task driver and particle YAML
  configuration. Remove redundant standalone demo and plotting scripts.
- Refresh Gamma example outputs and figures, and document the convention
  `zeta(theta) = sum_l zeta_l P_l(cos(theta))` without another `(2*l+1)` factor.
- Publish window-family and angular DDD/Q measurements with sample counts,
  main-loop/task totals and distinct Slurm MaxRSS/process-tree RSS definitions.
- Keep notebook displays portable and omit machine-specific paths. Keep local
  validation and experiment assets outside the tracked repository and built
  distributions. CI retains static checks, package imports, MPI smoke checks,
  distribution builds and documentation builds.
