# PyHermes Package Code Map

This file is a developer-facing map of the `pyhermes/` package. It is meant to
answer a practical question: "Where is the code for this feature?"

For user-facing tutorials and mathematical background, start from the repository
`README.md` and the documentation under `docs/`. This file focuses on source
layout and implementation responsibilities.

## Big Picture

PyHermes is organized around one main data flow:

```text
particle catalog
  -> SFCProjection task
  -> SFCField field
  -> WindowFunc convolution
  -> Counting / Corr_2PCF / Corr_3PCF / multipoles / weighted-field analyses
```

The package directories roughly follow this split:

| Directory | Main role |
| --- | --- |
| `base/` | Low-level task implementations, especially field construction. |
| `io/` | User-facing data containers, file IO, readers, and window objects. |
| `theory/` | Scientific task drivers and estimator logic. |
| `param/` | Parameter loading, default handling, and logging helpers. |
| `pipeline/` | Shared task base class and pipeline exceptions. |
| `utils/` | Numerical kernels and helper routines used by the higher-level layers. |

The most common public entry points are imported from `pyhermes.io` and
`pyhermes.theory`:

```python
from pyhermes.io import SFCField, WindowFunc, read_particle_data
from pyhermes.theory import Counting, Corr_2PCF, Corr_3PCF, Corr_3PCF_Multipole
```

## Where To Find Common Features

| If you are looking for... | Start here |
| --- | --- |
| Building `SFCField` from particles | `base/sfc_projection.py` |
| Loading or manipulating saved fields | `io/sfc_field.py`, `io/base.py` |
| Constructing and applying windows | `io/window.py`, `utils/convolution.py` |
| Built-in window formulas | `utils/window_functions.py` |
| Window parameter normalization | `utils/window_params.py` |
| Particle readers and field selection | `io/readers.py` |
| Redshift-space position mapping | `utils/redshift_space.py` |
| Scaling-function projection/interpolation | `utils/wavelet_grid.py` |
| Counting task logic | `theory/counting.py`, `io/counting.py` |
| 2PCF task logic and binning-window mappings | `theory/corr2pcf.py`, `io/corr2pcf.py` |
| Direct field-pair evaluation and owned spectral lifetime | `io/sfc_field.py` (`pair_with`), `io/pair_product.py` (`PairProduct.evaluate`) |
| 3PCF task logic | `theory/corr3pcf.py`, `utils/corr3pcf_kernels.py`, `io/corr3pcf.py` |
| 3PCF multipoles | `theory/corr3pcf_multipole.py`, `utils/corr3pcf_multipoles.py` |
| Legendre/spherical-harmonic window kernels | `utils/legendre_windows.py`, `utils/legendre_fast.py`, `utils/special_functions.py` |
| 2D 2PCF plotting helpers | `utils/plot.py`, `utils/plot_styles.json` |
| MPI helpers | `utils/mpi_util.py` |
| Task parameters and YAML loading | `param/parambase.py`, `base/default_params.json`, `theory/default_params.json` |

## Shared 2PCF execution and angular-3PCF reuse

The public configuration is documented in
[`docs/get_start/corr_2pcf/optimization.rst`](../docs/get_start/corr_2pcf/optimization.rst).
The opt-in `spectral_window="auto"` delegates to common structural planners;
do not add a reduction loop or public flag for every window family.

| Responsibility | Implementation |
| --- | --- |
| Task preparation, broadcasts, sample distribution and timing | `theory/corr2pcf.py` |
| Local public pair evaluation, validation and cleanup | `io/pair_product.py` (`PairProduct`) |
| Mathematical structure and built-in/custom eligibility | `utils/window_structure.py` |
| Real input FFTs, pair powers and half-spectrum normalization | `utils/corr2pcf_spectral.py` (`SpectralPairReducer`) |
| Generic exact alias reduction, common shell boundaries and radial planning | `utils/corr2pcf_window_reduction.py` (`ScalarWindowEvaluator`) |
| Automatic radial/separable planning, bounded factor cache and batches | `utils/corr2pcf_strategies.py` (`AutoWindowEvaluator`) |
| Angle-independent box-random contrast lifetime and Q assembly | `theory/corr3pcf.py` |

`compute_pair_products_at_samples` in `theory/corr2pcf.py` is the local-series
bridge used by angular 3PCF pair postprocessing. It shares the same reducer and
window plans, uses `COMM_SELF` for root-only work and releases full fallback
kernels per radius. It must not construct a world-collective task on root alone.
These helpers are internal implementation points, not additional user task APIs.

For user-controlled local series, `SFCField.pair_with(other)` returns a
`PairProduct` whose `evaluate(windows)` shares those same reducers and window
plans. Its context owns the pair power and performs no world collectives.
It returns coefficient-array means; estimator normalization stays with the
caller. Keep the distributed task orchestration in `Corr_2PCF`.

Power, basis and factor caches belong to one run or local pair series. Angular
contrasts belong to one product and are cleared on completion or failure.
Caller-owned fields must not be mutated concurrently. Keep cache ownership,
resource guards and every alias-specific scaling weight explicit when extending
these helpers. The `memory` execution strategy and unsupported-window fallback
remain available; historical explicit window strategies are retained for checks.

## Multipole contraction and CUDA layout

The three-point contraction uses the full signed triple-overlap table. Basis
samples are zero outside their compact support; coefficient grids are periodic.
CPU and GPU implementations reuse intermediate arrays in an exact separable
reassociation of the same contraction. This does not remove finite-grid projection
or window errors.

For compatibility, `gpu_threads_per_block` remains a three-integer setting. The
production separable CUDA kernels use its **product** as the size of a flattened
one-dimensional block: the default `[8, 8, 8]` means 512 threads per block, not a
three-dimensional production launch. The retained dense reference kernel uses
the original three-dimensional shape. The product must not exceed 1024.

The separable GPU context reuses two complex128 work grids in addition to the
existing device arrays. Their combined size is `32 * (2**J)**3` bytes, or 512 MiB
at J=8. Convolution and MPI field exchange still run on the CPU; selecting the
GPU backend accelerates the final contraction and does not move the whole
pipeline to the device. CPU/GPU numerical agreement and independent physical
integrals are covered by `tests/test_corr3pcf_gamma.py`; CUDA tests require a real
device or an explicitly enabled simulator.

### Exact Gamma and convolution optimization

The CPU implementation precomputes periodic coefficient indices and evaluates
Gamma rows using every exactly nonzero signed
coefficient. It applies no tolerance, rank truncation, or change to the
three-field integral. CPU work arrays and stencil indices are reused within
each summation context; the existing GPU contraction is unchanged.

Multipole convolution contexts lazily retain a source FFT across modes.
Each context belongs to one product invocation and one immutable coefficient
array; do not mutate those coefficients in place until the context is discarded.
Replacing the source array invalidates its spectrum. Disk-cached convolved
fields do not require building a source FFT. A retained complex128 spectrum
adds 256 MiB per source context at J=8 (two in serial mode, one per active
convolution rank in pair-MPI mode). Numerical agreement and whole-pipeline
runtime/memory measurements determine whether this tradeoff is useful.

### Experimental particle-centred multipoles

`Corr_3PCF_Multipole` also accepts `summation_method="particle"` with
`summation_backend="cpu"`. The default method remains `"gamma"`. The particle
method reconstructs each harmonic arm separately at catalogue positions,
multiplies the two values, then sums with the centre's effective projection
weights. It supports the existing serial, pair-MPI and sample-MPI layouts.
Centre stencils are prepared once and reused across harmonic modes and radii;
the particle context does not allocate Gamma contraction work grids.

This first version requires an unfiltered centre: omit `window` and `window1`,
and set endpoint smoothing with `window2`/`window3`. The central catalogue must
be retrievable through `SFCField.get_particle_data()` (a saved particle dataset
or a valid original input path). A derived centre field cannot replace that
catalogue. CUDA particle summation and an adjoint centre-window convolution are
not implemented. Invalid combinations raise errors rather than silently
changing methods or ignoring a requested window.

For `delta_ddd_l` and `zeta_l`, data-minus-random subtraction is retained. A
uniform random centre contributes the orthonormal two-field integral; an
explicit random catalogue uses its own particle stencils. Raw, catalogue,
field and unit normalization follow the existing task rules, including marks.
The result metadata records `summation_method`.

Particle and Gamma methods share a continuum target, but need not agree at
fixed J: particle readout evaluates `A_j(x_i) B_j(x_i)`, whereas the Gamma
centre contraction corresponds to `P_j(A_j B_j)(x_i)`. Validate convergence and
independent reference errors; do not treat disagreement at fixed J as an
automatic regression. FFT/window costs remain; documented production timings
are workload-specific. See the user multipole guide,
`examples/notebooks/corr3pcf.ipynb`, and
`examples/configs/param_3pcf_multipole_particle.yaml` for runnable examples.
Independent periodized-basis readout,
complex phases, random subtraction, normalization and task integration are
covered by `tests/test_corr3pcf_particle.py`.

## Directory Details

### `base/`

Low-level task code that constructs the field representation.

- `sfc_projection.py`: implements the `SFCProjection` task. This is where particle
  positions and weights are projected into scaling-function coefficients.
- `default_params.json`: defaults for the field-construction layer.

### `io/`

Data containers, readers, and user-facing field/window objects.

- `base.py`: defines `HermesData`, the base class for saved result containers.
- `sfc_projection.py`: defines `SFCField`, the reusable field object used by later
  tasks.
- `window.py`: defines `WindowFunc` and the `SFCField @ WindowFunc`
  convolution interface.
- `readers.py`: particle catalog readers for `bin`, `npz`, `gadget`,
  `gadget-fof`, and `fof`, plus weight resolution helpers.
- `counting.py`, `corr2pcf.py`, `corr3pcf.py`, `corr3pcf_multipole.py`: saved
  result containers for each task family.
- `funcs.py`: small file-loading and file-writing helpers used by IO classes.

### `theory/`

Task drivers and estimator logic. These classes usually combine parameters,
`SFCField`, `WindowFunc`, random fields, and numerical kernels.

- `counting.py`: `Counting` task; samples a field at random positions.
- `corr2pcf.py`: `Corr_2PCF` task; sampling grids, binning-window mappings,
  prepared-input reuse, shared exact pair-series helper, and product assembly.
- `corr3pcf.py`: `Corr_3PCF` task; standard 3PCF products for particle-center
  and box-random-center modes, product-local contrasts and local pair reuse.
- `corr3pcf_multipole.py`: `Corr_3PCF_Multipole` task driver.
- `default_params.json`: defaults for theory-level tasks.

### `param/`

Parameter and logging helpers.

- `parambase.py`: YAML/JSON parameter loading and the `ParamBase` helper class.
- `logbase.py`: logger setup.

### `pipeline/`

Shared task infrastructure.

- `pipeline.py`: defines `TaskBase`, the base class used by task drivers.
- `custom_exceptions.py`: pipeline and parameter exceptions.

### `utils/`

Numerical kernels and implementation helpers. This directory is intentionally
more internal than `io/` and `theory/`, but it contains many important pieces.

#### Window and convolution utilities

- `window_functions.py`: built-in Fourier-space window functions, including
  smoothing windows, binning windows, field-derivative windows, and the
  Gaussian-derivative wavelet.
- `window_params.py`: normalization and serialization of window dictionaries,
  length arguments, LOS arguments, binning-window defaults, and kernel modes.
- `convolution.py`: real and complex window-kernel construction plus the
  specialized 3D convolution routines used by `SFCField @ WindowFunc`.

#### Wavelet/grid utilities

- `wavelet_grid.py`: scaling-function sampling, particle projection onto the
  grid, Fourier power of the scaling function, and interpolation of a grid field
  at arbitrary positions.

#### Correlation and multipole kernels

- `corr3pcf_kernels.py`: Numba kernels for standard 3PCF triplet products.
- `corr3pcf_multipoles.py`: 3PCF multipole convolution, streaming, GPU context,
  and `m`-term combination helpers.
- `legendre_windows.py`: general Legendre/spherical-harmonic window construction
  used by 3PCF multipoles.
- `legendre_fast.py`: generated or specialized fast low-order Legendre window
  functions.
- `special_functions.py`: spherical Bessel functions, associated Legendre
  functions, spherical harmonics, and multipole mixing helpers.

#### Runtime, sampling, and plotting helpers

- `sampling.py`: random-box and regular-grid positions.
- `redshift_space.py`: Hubble-factor helper and redshift-space coordinate
  mapping.
- `mpi_util.py`: MPI-related utilities.
- `runtime.py`: runtime thread configuration.
- `plot.py`: 2D 2PCF plotting helpers for `(s, mu)` and `(rp, pi)` products.
- `plot_styles.json`: default plotting style values.
- `func_util.py`: miscellaneous helpers such as notebook detection, split-file
  discovery, window-action descriptions, and `SFCField` compatibility checks.

## Maintenance Notes

- Prefer adding user-facing objects to `io/` or `theory/`; keep `utils/` for
  numerical kernels and internal helpers.
- If a new feature adds a new window formula, update both
  `utils/window_functions.py` and the window documentation.
- If a new feature adds a new window dictionary shape, update
  `utils/window_params.py`.
- If a helper in `utils/` becomes a stable public concept, consider re-exporting
  it through a clearer module rather than asking users to import from deep
  utility paths.
- Before moving files, check existing import paths in examples, tests, and docs.
  A code map like this is cheap; package-level refactors are not.
