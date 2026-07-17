Installation
============

Local installation
------------------

A fresh environment is recommended because NumPy, SciPy, Numba, and
PyWavelets need mutually compatible builds. From a repository clone:

.. code-block:: bash

   conda create -n pyhermes python=3.12
   conda activate pyhermes
   pip install -r requirements.txt
   pip install -e .

The editable install is convenient while following the notebooks. Use
``pip install .`` for an ordinary non-editable installation.

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

MPI is optional for local notebook work and required for distributed runs. An
MPI implementation must be available before installing ``mpi4py``:

.. code-block:: bash

   pip install mpi4py
   mpirun -np 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank)"

Use one consistent MPI stack. Mixing a system ``mpirun`` with an ``mpi4py``
wheel linked against a different implementation is a common source of startup
failures.

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

Build the documentation
-----------------------

Install the documentation dependencies and build with warnings treated as
errors:

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme sphinx-copybutton
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
