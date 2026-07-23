<p align="center">
  <img src="https://pyhermes.astroslacker.com/_static/pyhermes_logo_transparent.png" alt="PyHermes logo" width="220">
</p>

# PyHermes

**PyHermes** is the Python implementation of **Hermes** (HypER-speed
MultirEsolution cosmic Statistics), an in situ framework for measuring cosmic
statistics with reusable multiresolution fields and window operators.

A catalogue is projected once into an `SFCField`. Smoothing, geometric binning,
multipole decomposition, differentiation, and inverse-Laplacian operations are
then expressed through `WindowFunc` objects. The same field can therefore feed
counting, 2PCF, conventional 3PCF, 3PCF multipoles, marked statistics, and
derived physical-field calculations without returning to particle-level tuple
counting for every configuration.

- **Documentation:** [pyhermes.astroslacker.com](https://pyhermes.astroslacker.com)
- **Source:** [SYSUSPA-Projects/PyHermes](https://github.com/SYSUSPA-Projects/PyHermes)
- **Tutorials:** [`examples/notebooks/`](https://github.com/SYSUSPA-Projects/PyHermes/tree/main/examples/notebooks)
- **Runnable configurations:** [`examples/configs/`](https://github.com/SYSUSPA-Projects/PyHermes/tree/main/examples/configs) and
  [`examples/scripts/`](https://github.com/SYSUSPA-Projects/PyHermes/tree/main/examples/scripts)

![Hermes field-window workflow](https://pyhermes.astroslacker.com/_static/paper/PyHermes-Workflow.png)

## What It Covers

- catalogue-to-field projection with configurable compactly supported scaling
  functions and resolution `J`;
- built-in and user-defined smoothing, binning, multipole, and operator windows;
- one-point counting and field sampling;
- isotropic and anisotropic 2PCF measurements;
- Monte Carlo and spherical-harmonic 3PCF estimators;
- MPI/thread parallelism and CPU or CUDA contraction for 3PCF multipoles;
- weighted fields, velocity derivatives, Poisson potential, acceleration, and
  density-dependent marks.

## Installation

For notebooks, development, and other single-process work, install PyHermes
from PyPI:

```bash
python -m pip install pyhermes-cosmo
```

This installs PyHermes and its regular Python dependencies automatically. MPI
and CUDA are optional, so neither is required for the default installation.

The Python import name remains unchanged:

```python
import pyhermes
```

For a ready-to-use MPICH environment on Linux or macOS:

```bash
conda create -n pyhermes -c conda-forge python=3.12 mpi4py mpich pip
conda activate pyhermes
python -m pip install pyhermes-cosmo
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"
```

Here conda provides a mutually compatible MPI runtime and Python binding, while
pip installs PyHermes and the remaining Python dependencies. PyHermes does not
yet require a separate conda package for this workflow.

Without `mpi4py`, PyHermes automatically uses its single-process MPI fallback.
Users of an existing cluster MPI should follow the
[installation guide](https://pyhermes.astroslacker.com/install.html) for
the matching `mpi4py` and GPU setup.

## Smallest Workflow

The tracked Quick Start configurations use paths relative to `examples/`:

```bash
cd examples
python scripts/run_sfc_projection.py configs/param_sfc_projection.yaml
python scripts/run_2pcf.py configs/param_2pcf.yaml
```

The first command downloads and caches a compact Quijote halo catalogue from
the URL in the YAML, verifies its SHA256 digest, and writes the base
`SFCField`. The second command consumes that exact field and writes an
isotropic `Corr2PCFData` result. The matching `quick_start.ipynb` executes the
same configs and plotting code rather than maintaining a parallel example.

The dedicated `particle_io.ipynb` shows how URL caching, NPZ conversion, raw
BIN layouts, and native simulation readers feed the same projection API.
Advanced examples that require J=9, weighted or redshift-space fields, or an
explicit sampled random field use `scripts/prepare_sfc_fields.py`. The optional
dark-matter snapshot builder accepts the local Gadget HDF5 snapshot prefix
explicitly; no cluster-specific path is embedded in the code:

```bash
python scripts/build_quijote_dm_sfc_field.py /path/to/snapdir_004/snap_004
```

```python
from pyhermes.base.sfc_projection import SFCProjection
from pyhermes.io import WindowFunc
from pyhermes.param.parambase import read_param

params = read_param("./configs/param_sfc_projection.yaml")
field = SFCProjection(params).run()

gaussian = WindowFunc(
    {"type": "gaussian", "len_args": {"R": 10.0}},
    field.sfc_info,
    threads=8,
)
smoothed_field = field @ gaussian
```

This is the core language of PyHermes: **field @ window**. Statistical tasks
build the required window families and normalizations around the same objects.

## Start With The Notebooks

The recommended route through
[`examples/notebooks/`](https://github.com/SYSUSPA-Projects/PyHermes/tree/main/examples/notebooks)
is:

1. [`quick_start.ipynb`](https://github.com/SYSUSPA-Projects/PyHermes/blob/main/examples/notebooks/quick_start.ipynb)
2. [`particle_io.ipynb`](https://github.com/SYSUSPA-Projects/PyHermes/blob/main/examples/notebooks/particle_io.ipynb)
3. [`sfc_projection.ipynb`](https://github.com/SYSUSPA-Projects/PyHermes/blob/main/examples/notebooks/sfc_projection.ipynb)
4. [`window.ipynb`](https://github.com/SYSUSPA-Projects/PyHermes/blob/main/examples/notebooks/window.ipynb)
5. [`physical_fields.ipynb`](https://github.com/SYSUSPA-Projects/PyHermes/blob/main/examples/notebooks/physical_fields.ipynb)
6. [`counting.ipynb`](https://github.com/SYSUSPA-Projects/PyHermes/blob/main/examples/notebooks/counting.ipynb)
7. [`corr2pcf.ipynb`](https://github.com/SYSUSPA-Projects/PyHermes/blob/main/examples/notebooks/corr2pcf.ipynb)
8. [`corr3pcf.ipynb`](https://github.com/SYSUSPA-Projects/PyHermes/blob/main/examples/notebooks/corr3pcf.ipynb)

The first four notebooks establish the common catalogue-to-field-to-window
workflow. Continue with physical fields, one-point Counting, or the
2PCF-to-3PCF statistics path according to the calculation you need.

Generated catalogues and estimator products are intentionally not committed.
The notebooks state which lightweight cells run locally and which script/YAML
pairs are intended for a workstation or cluster.

## Documentation

The full guide at [pyhermes.astroslacker.com](https://pyhermes.astroslacker.com)
follows the terminology and estimator definitions of the Hermes paper. It
covers the mathematical construction, current APIs, window catalogue,
parameter mappings, numerical validation, and performance interpretation.

To build the documentation locally:

```bash
python -m pip install ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

## Citing PyHermes

If PyHermes contributes to a publication, please cite the software and, once
available, the accompanying Hermes/PyHermes paper. Citation metadata is
provided in [`CITATION.cff`](https://github.com/SYSUSPA-Projects/PyHermes/blob/main/CITATION.cff),
with BibTeX examples in the
[citation guide](https://pyhermes.astroslacker.com/citing.html).

The current citation author list is Long-Long Feng, Tengpeng Xu, Tian-Cheng
Luan and collaborators. The manuscript entry remains clearly
marked as a placeholder until its final title, journal, and identifier are
available.
