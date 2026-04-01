# PyHermes


Please see details in [PyHermes Official Docs](https://pyhermes.astroslacker.com)


## Installation

### Method1 - from conda (recommended)
<font color='orange'>TODO</font>

### Method2 - from pypi
Users need to install mpi4py mannually.

Visit [mpi4py](https://pypi.org/project/mpi4py/) for installation details.

<font color='orange'>TODO</font>

### Method3 - from source
1. Download the source: git clone this repository,
   ```bash
   git clone https://github.com/PyHermes/PyHermes.git
   ```
2. Install: change your current workdir to PyHermes, then type
   ```bash
   cd PyHermes
   pip install .
   ```
3. Enjoy!

## Quick start: from particle positions to a smoothed 2PCF

The example below shows the basic PyHermes workflow:

1.	build the multiresolution coefficient field from particle positions
2.	construct the fluctuation field relative to a uniform random field
3.	smooth the field with a spherical window
4.	measure the two-point correlation function by shell convolution and spatial averaging

```python
import numpy as np
import matplotlib.pyplot as plt

from pyhermes.base.convols import Convols
from pyhermes.io import WindowFunc

# Step 1: build the multiresolution coefficient field from particle positions
task_params = {'Convols': {'fin': {'path': './quijote10000.bin'}}}
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
# DD(r) = <n(x)n(x+r)>; n(x+r) = n_r(x) = n(x) @ W_shell(r)
# xi = (D-R)(D-R)/RR
r_arr = np.linspace(1, 150, 30)
xi_arr = np.zeros_like(r_arr)

for i, r in enumerate(r_arr):
    win_params = {"type": "shell", "len_args": {"R": r}}
    win_shell = WindowFunc(win_params, D.convols_info)

    # n(x+r) is obtained by convolving the field with a shell window of radius r
    deltaDD_w = deltaD_w @ win_shell * deltaD_w

    # spatial mean + RR normalization gives xi(r)
    xi_arr[i] = deltaDD_w.as_array().mean() / RR

plt.plot(r_arr, xi_arr * r_arr**2)
plt.xlabel(r"$r$")
plt.ylabel(r"$r^2 \xi(r)$")
plt.show()
```

### Notes
- `D` is the normalized density field in PyHermes’ multiresolution representation.
- `R = 1 / D.V` is the corresponding uniform random field.
- `deltaD = D - R` is the fluctuation field.
- `WindowFunc` is used to construct spherical, shell, and other window functions.
- The operator `@` applies a window convolution to a field.
- The product of two fields followed by `.as_array().mean()` gives the spatial average needed for correlation-function estimators.
