# PyHermes

PyHermes is a Python package for high-performance cosmic statistics on large particle datasets. It provides a workflow that starts from particle positions, builds a multiresolution field representation, and then evaluates statistics such as window convolutions, random-point counting, the 2-point correlation function (2PCF), and the 3-point correlation function (3PCF).

Project links:

- Official documentation: [PyHermes Official Docs](https://pyhermes.astroslacker.com)
- Source code: [PyHermes on GitHub](https://github.com/PyHermes/PyHermes)

## What PyHermes Can Do

PyHermes currently provides task-level entry points for:

- building multiresolution coefficient fields from particle positions
- applying spherical or shell-like window convolutions
- counting / sampling the field on random points
- computing the 2-point correlation function
- computing the 3-point correlation function
- running in single-process mode by default, with optional MPI acceleration through `mpi4py`

## Installation

PyHermes supports normal Python installation with `pip`. If `mpi4py` is not installed, PyHermes falls back to a single-process compatibility wrapper. Install `mpi4py` only if you want MPI-based parallel execution.

### Method 1 - create a conda environment, then install PyHermes

This is the easiest way to get a clean environment:

```bash
conda create -n pyhermes python=3.10
conda activate pyhermes
pip install -r requirements.txt
pip install .
```

If you want MPI support, install `mpi4py` after your MPI runtime is available:

```bash
pip install mpi4py
```

### Method 2 - install from PyPI

If you are using a published PyPI release of PyHermes, install it with:

```bash
pip install pyhermes
```

Optional MPI support:

```bash
pip install mpi4py
```

For MPI installation details, see [mpi4py on PyPI](https://pypi.org/project/mpi4py/).

### Method 3 - install from source

1. Clone the repository:

   ```bash
   git clone https://github.com/PyHermes/PyHermes.git
   cd PyHermes
   ```

2. Install dependencies and the package:

   ```bash
   pip install -r requirements.txt
   pip install .
   ```

3. Optional: install `mpi4py` for parallel runs.

## Quick Start

### Workflow overview

The basic PyHermes workflow is:

1. build the multiresolution coefficient field from particle positions
2. construct the fluctuation field relative to a uniform random field
3. smooth the field with a spherical window
4. measure the 2PCF by shell convolution and spatial averaging

### From particle positions to a smoothed 2PCF

```python
import numpy as np
import matplotlib.pyplot as plt

from pyhermes.base.convols import Convols
from pyhermes.io import WindowFunc

# Step 1: build the multiresolution coefficient field from particle positions
task_params = {"Convols": {"fin": {"path": "./quijote10000.bin"}}}
D = Convols(task_params).run()

# Step 2: construct the fluctuation field delta = D - R
R = 1 / D.V          # uniform random field in the same normalized convention
RR = R ** 2          # <RR>, used to normalize the 2PCF estimator
deltaD = D - R       # fluctuation field

# Step 3: smooth the fluctuation field with a spherical window
win_params = {"type": "sphere", "len_args": {"R": 5}}
win_filter = WindowFunc(win_params, D.convols_info)
deltaD_w = deltaD @ win_filter

# Step 4: measure xi(r) by shell convolution + spatial averaging
r_arr = np.linspace(1, 150, 30)
xi_arr = np.zeros_like(r_arr)

for i, r in enumerate(r_arr):
    win_params = {"type": "shell", "len_args": {"R": r}}
    win_shell = WindowFunc(win_params, D.convols_info)
    deltaDD_w = deltaD_w @ win_shell * deltaD_w
    xi_arr[i] = deltaDD_w.as_array().mean() / RR

plt.plot(r_arr, xi_arr * r_arr**2)
plt.xlabel(r"$r$")
plt.ylabel(r"$r^2 \xi(r)$")
plt.show()
```

Notes:

- `D` is the normalized density field in PyHermes' multiresolution representation.
- `R = 1 / D.V` is the corresponding uniform random field.
- `deltaD = D - R` is the fluctuation field.
- `WindowFunc` constructs spherical, shell, and related window functions.
- The `@` operator applies a window convolution to a field.
- The product of two fields followed by `.as_array().mean()` gives the spatial average used in correlation estimators.

## Example Scripts

The [`examples`](./examples) directory contains runnable task-level examples:

- [`examples/run_convols.py`](./examples/run_convols.py): build the multiresolution coefficient field from particle positions
- [`examples/run_counting.py`](./examples/run_counting.py): sample the smoothed field on many random points
- [`examples/run_2pcf.py`](./examples/run_2pcf.py): compute the 2PCF
- [`examples/run_3pcf.py`](./examples/run_3pcf.py): compute the 3PCF
- [`examples/quick_start.ipynb`](./examples/quick_start.ipynb): notebook-based quick start
- [`examples/full_example.ipynb`](./examples/full_example.ipynb): end-to-end notebook example

Example configuration files are stored in [`examples/configs`](./examples/configs):

- `param_convols.yaml`
- `param_counting.yaml`
- `param_2pcf.yaml`
- `param_3pcf.yaml`

## Running the Examples

After installation, you can enter the example directory and run any task directly:

```bash
cd examples
python run_convols.py
python run_counting.py
python run_2pcf.py
python run_3pcf.py
```

To run with MPI:

```bash
mpirun -np 8 python run_convols.py
mpirun -np 8 python run_counting.py
mpirun -np 8 python run_2pcf.py
mpirun -np 8 python run_3pcf.py
```

If `mpi4py` is not installed, PyHermes still works in single-process mode.

## Input and Output

The provided example configs show the typical workflow:

- `Convols` reads particle positions and produces a coefficient file such as `./output/quijote_sfc.pkl`
- `Counting`, `Corr_2PCF`, and `Corr_3PCF` reuse that coefficient file as input
- task parameters are stored in YAML and loaded with `read_param(...)`

The default example dataset in `param_convols.yaml` points to a hosted binary particle catalog:

- `https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin`

## Package Requirements

Core dependencies listed in this repository include:

- `numpy`
- `scipy`
- `numba`
- `pyyaml`
- `json5`
- `requests`
- `PyWavelets`
- `rich`

`mpi4py` is optional and only needed for MPI-based parallel execution.

## Documentation

For more detailed guides and API-oriented usage, see:

- [PyHermes Official Docs](https://pyhermes.astroslacker.com)
- [`docs/`](./docs) in this repository

## License

PyHermes is released under the MIT License.
