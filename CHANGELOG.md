# Changelog

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
