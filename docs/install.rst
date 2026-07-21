Installation
============

PyPI installation
-----------------

For notebooks, development, and other single-process work, install from PyPI:

.. code-block:: bash

   python -m pip install pyhermes-cosmo

The distribution name is ``pyhermes-cosmo`` while the Python import name is
``pyhermes``. Pip installs the regular Python dependencies automatically. MPI
and CUDA are optional and are not needed for this default installation.

Choose the route that matches the machine:

* **Local or notebook use:** install only ``pyhermes-cosmo`` with pip. The
  single-process compatibility layer is selected automatically.
* **A ready-to-use workstation MPI:** let conda-forge provide ``mpi4py`` and
  MPICH, then install PyHermes with pip in the same environment.
* **An existing cluster MPI:** load the site's MPI module first, then install
  the ``mpi`` optional dependency so that every rank uses the same MPI stack.

Verify the installation
-----------------------

.. code-block:: bash

   python -c "import pyhermes; print(pyhermes.__version__)"

The core package should import without MPI or CUDA. A minimal object check is:

.. code-block:: python

   from pyhermes.base.sfc_projection import SFCProjection
   from pyhermes.io import SFCField, WindowFunc
   from pyhermes.theory.corr2pcf import Corr_2PCF

MPI support
-----------

MPI is optional for local work and required only for distributed runs. When
``mpi4py`` is unavailable, PyHermes uses its single-process compatibility
layer automatically, so users do not need to install an MPI implementation
just to import or use the package locally.

For a ready-to-use MPICH environment on Linux or macOS:

.. code-block:: bash

   conda create -n pyhermes -c conda-forge python=3.12 mpi4py mpich pip
   conda activate pyhermes
   python -m pip install pyhermes-cosmo
   mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"

This deliberately combines the two package managers: conda-forge supplies a
matched MPI runtime and ``mpi4py`` build, while pip installs PyHermes and its
remaining Python dependencies. A separate PyHermes conda package is not
required for this setup.

For a lightweight pip-only MPICH environment on a supported Linux or macOS
workstation, the equivalent setup is:

.. code-block:: bash

   python -m pip install "pyhermes-cosmo[mpi]" mpich
   mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"

The conda and pip MPI packages prioritize portability. On an HPC system,
prefer the MPI implementation supplied and tuned by the site administrator.

Users of an existing cluster MPI should load that implementation first and
then install the optional Python binding:

.. code-block:: bash

   module load openmpi
   python -m pip install "pyhermes-cosmo[mpi]"

Use one consistent MPI stack. Mixing a system ``mpirun`` with an ``mpi4py``
wheel linked against a different implementation is a common source of startup
failures.

Development installation
------------------------

From a repository clone:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[test,docs]"

CUDA support
------------

CUDA is optional and is currently used by the ``Corr_3PCF_Multipole``
summation backend. Field projection and FFT window convolution remain CPU-side
operations. A CUDA-capable NVIDIA GPU, a working driver, and Numba CUDA support
are required when ``summation_backend: gpu`` is selected.

CPU-only systems should set:

.. code-block:: yaml

   Corr_3PCF_Multipole:
      summation_backend: cpu

Example data
------------

The repository does not commit the Quijote catalogue or generated products.
Prepare the public example inputs from the repository root with:

.. code-block:: bash

   python examples/scripts/prepare_sfc_fields.py

Alternatively, execute the download and preparation cells in
``examples/notebooks/sfc_projection.ipynb``. Later notebooks reuse the fields
written to ``examples/output/``.

The public halo catalogue is sufficient for the tutorials. If you also have a
local Quijote Gadget HDF5 dark-matter snapshot, build the optional DM field by
passing its snapshot prefix explicitly:

.. code-block:: bash

   python examples/scripts/build_quijote_dm_sfc_field.py \
      /path/to/snapdir_004/snap_004

Use ``--output`` and ``--threads`` to override the documented defaults. No
private cluster path is embedded in the script.

Build the documentation
-----------------------

Install the documentation dependencies and build with warnings treated as
errors:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -W -b html docs docs/_build/html

The generated site starts at ``docs/_build/html/index.html``.

Troubleshooting
---------------

**An import fails inside SciPy**
   Confirm that the active interpreter and the environment receiving ``pip``
   packages are the same. Reinstall NumPy, SciPy, and Numba together rather
   than mixing packages from several environments.

**An MPI job starts too many threads**
   Match the YAML ``threads`` value, ``OMP_NUM_THREADS``, and the scheduler's
   ``--cpus-per-task``. Their product with the MPI rank count must fit the
   allocation.

**A high-``J`` job runs out of memory**
   Increasing ``J`` by one multiplies the number of three-dimensional field
   coefficients by eight. Prefer fewer MPI ranks and more threads per rank
   when rank-local copies dominate memory.
