# Changelog

## Unreleased

### Optional 2PCF spectral computation

- Add `SFCField.pair_with(other)` and `PairProduct.evaluate(windows)` for
  direct local field-pair averages with shared spectral/window preparation,
  explicit estimator normalization and context-managed cleanup. Retain complex
  cross-field phase through the convolution fallback when required.
- Add Parseval-based real-power reduction, compatible input preparation reuse,
  fused folded-window evaluation and shared finite-shell boundaries.
- Add batched boundary evaluation, root-computed MPI spectra and exact integer
  radial grouping. Preserve all alias-specific scaling weights and folding
  multiplicities; no interpolation or frequency cutoff is introduced.
- Keep existing convolution defaults and unsupported-window fallbacks. Document
  configuration, memory guards, numerical validation and total/main-loop timing.
- Add independent numerical checks and real 2PCF MPI checks to CI.
- Add `spectral_window="auto"` to select exact radial grouping for built-in
  isotropic windows, shared finite-shell boundaries, and bounded separable
  factors for grid-axis ring/Gaussian-ring, disk, cylindrical and Cartesian
  cubic windows. Unsupported or tilted windows keep their applicable paths.
- Share factor tables and batches of up to eight queries across compatible
  windows, with per-run cache/scratch limits. Preserve custom callables and
  every alias-specific scaling term inside the fold.

### Standard angular 3PCF

- Prepare box-random D-R fields once per product/rank and release them on
  completion or failure. Reuse the exact local 2PCF backend for lower-order
  pair series and `Q` postprocessing; root-only planning uses `COMM_SELF`.
- Keep Monte Carlo kernels, centres, seeds, rotations, weights and
  interpolation unchanged. Add no user setting or result-format change.
- Cover weighted/nonuniform inputs, MPI, repeated runs, fallback memory and
  cleanup. Separate local controlled profiles from catalogue measurements
  without a full-size baseline speedup claim.

### Examples and documentation

- Explain convolution and Parseval reconstruction in the 2PCF notebook,
  including normalization, correspondence to task options and fresh-window
  timing. Document local pair evaluation alongside the full task workflow.
- Refresh four 2PCF and three corrected CPU Gamma examples under versioned
  output names and execute affected notebook plots. Retain historical files
  and document the input-provenance boundary of angular overlays.
- Publish matched 46-by-51 axial-window timings and shell-matched cubic
  sampling, including main-loop/task totals and per-sample costs.
- Align Quick Start and task guides with the tracked YAML files. The
  benchmark overview distinguishes current pure-DDD measurements from
  historical Q timings and keeps Slurm MaxRSS separate from process-tree RSS.

## 1.2.0

### Numerical correction

- Fix the Gamma table for isotropic three-point multipoles. Compact basis
  samples are zero outside their support, with the full positive and negative
  displacement range. Periodic wrapping applies to coefficient grids, not
  local basis samples. Previous cyclic basis shifts and incomplete
  displacement ranges could change multipoles.
- Recompute affected Gamma measurements from 1.1. Existing SFC catalogue
  projections can be reused. Two-point and direct-angular Monte Carlo
  estimators are unchanged by this correction.
- Add independent periodized-basis integral checks for complex fields,
  symmetry, uniform backgrounds, and CPU/CUDA contractions.

### Performance

- Reuse CPU Gamma indices, exactly nonzero signed coefficients, and work
  arrays; no tolerance or rank truncation changes the integral.
- Reuse a source FFT within a multipole convolution context, preserving
  source precision. Replacing the coefficient array invalidates its spectrum;
  a context assumes its source array is immutable until discarded.
- Use the ordinary inverse-FFT path for single-precision spectra so it matches
  generic convolution across SciPy versions; double-precision production
  retains its existing in-place path. Regression tolerances remain unchanged.
- Retained spectra increase memory. Document existing three-repeat timings
  and their allocation and memory boundaries.

### Experimental particle summation

- Add `summation_method="particle"` with `summation_backend="cpu"` to the
  existing `Corr_3PCF_Multipole` task, with serial, pair MPI, and sample MPI.
- Require an unfiltered first vertex and a retrievable particle catalogue.
  Endpoint windows remain available. Particle CUDA and filtered-centre
  particle summation are not implemented and are rejected explicitly.
- Retain effective projection weights, marks, data-minus-random subtraction,
  and explicit catalogue or uniform random centres.
- Particle and Gamma summation have different finite-J discretizations;
  agreement at fixed J is not an absolute-accuracy test.

### User interface and documentation

- Add a CPU-only synthetic example and particle YAML configuration;
  document particle companions, output shape, normalization, and MPI launch.
- Preserve defaults (`gamma`, `gpu`) and existing saved-result containers.
- Support directory and plain-filename multipole output paths; restore
  `task_params` metadata when loading saved results.
- Correct the documentation convention: saved `zeta_l` reconstructs the
  angular function as `sum_l zeta_l P_l(mu)`, without another `(2*l+1)` factor.
- Replace the two paper multipole figures with corrected archived results;
  mark unrecomputed historical figures and notebook sections explicitly.
