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
  -> Convols task
  -> ConvolsData field
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
from pyhermes.io import ConvolsData, WindowFunc, read_particle_data
from pyhermes.theory import Counting, Corr_2PCF, Corr_3PCF, Corr_3PCF_Multipole
```

## Where To Find Common Features

| If you are looking for... | Start here |
| --- | --- |
| Building `ConvolsData` from particles | `base/convols.py` |
| Loading or manipulating saved fields | `io/convols.py`, `io/base.py` |
| Constructing and applying windows | `io/window.py`, `utils/convolution.py` |
| Built-in window formulas | `utils/window_functions.py` |
| Window parameter normalization | `utils/window_params.py` |
| Particle readers and field selection | `io/readers.py` |
| Redshift-space position mapping | `utils/redshift_space.py` |
| Scaling-function projection/interpolation | `utils/wavelet_grid.py` |
| Counting task logic | `theory/counting.py`, `io/counting.py` |
| 2PCF task logic and pair-window mappings | `theory/corr2pcf.py`, `io/corr2pcf.py` |
| 3PCF task logic | `theory/corr3pcf.py`, `utils/corr3pcf_kernels.py`, `io/corr3pcf.py` |
| 3PCF multipoles | `theory/corr3pcf_multipole.py`, `utils/corr3pcf_multipoles.py` |
| Legendre/spherical-harmonic window kernels | `utils/legendre_windows.py`, `utils/legendre_fast.py`, `utils/special_functions.py` |
| 2D 2PCF plotting helpers | `utils/plot.py`, `utils/plot_styles.json` |
| MPI helpers | `utils/mpi_util.py` |
| Task parameters and YAML loading | `param/parambase.py`, `base/default_params.json`, `theory/default_params.json` |

## Directory Details

### `base/`

Low-level task code that constructs the field representation.

- `convols.py`: implements the `Convols` task. This is where particle
  positions and weights are projected into scaling-function coefficients.
- `default_params.json`: defaults for the field-construction layer.

### `io/`

Data containers, readers, and user-facing field/window objects.

- `base.py`: defines `HermesData`, the base class for saved result containers.
- `convols.py`: defines `ConvolsData`, the reusable field object used by later
  tasks.
- `window.py`: defines `WindowFunc` and the `ConvolsData @ WindowFunc`
  convolution interface.
- `readers.py`: particle catalog readers for `bin`, `npz`, `gadget`,
  `gadget-fof`, and `fof`, plus weight resolution helpers.
- `counting.py`, `corr2pcf.py`, `corr3pcf.py`, `corr3pcf_multipole.py`: saved
  result containers for each task family.
- `funcs.py`: small file-loading and file-writing helpers used by IO classes.

### `theory/`

Task drivers and estimator logic. These classes usually combine parameters,
`ConvolsData`, `WindowFunc`, random fields, and numerical kernels.

- `counting.py`: `Counting` task; samples a field at random positions.
- `corr2pcf.py`: `Corr_2PCF` task; sampling grids, pair-window mappings,
  pair-product calculations, and product assembly.
- `corr3pcf.py`: `Corr_3PCF` task; standard 3PCF products for particle-center
  and box-random-center modes.
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
  smoothing windows, pair windows, field-derivative windows, and the
  Gaussian-derivative wavelet.
- `window_params.py`: normalization and serialization of window dictionaries,
  length arguments, LOS arguments, pair-window defaults, and kernel modes.
- `convolution.py`: real and complex window-kernel construction plus the
  specialized 3D convolution routines used by `ConvolsData @ WindowFunc`.

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

- `sampling.py`: random points in a periodic box.
- `redshift_space.py`: Hubble-factor helper and redshift-space coordinate
  mapping.
- `mpi_util.py`: MPI-related utilities.
- `runtime.py`: runtime thread configuration.
- `plot.py`: 2D 2PCF plotting helpers for `(s, mu)` and `(rp, pi)` products.
- `plot_styles.json`: default plotting style values.
- `func_util.py`: miscellaneous helpers such as notebook detection, split-file
  discovery, window-action descriptions, and `ConvolsData` compatibility checks.

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
