Installation
============

PyHermes can run in a normal Python environment. MPI support is optional.
If ``mpi4py`` is not installed, PyHermes falls back to a single-process
compatibility wrapper.

Recommended environment
-----------------------

Using a fresh conda environment is the easiest way to avoid dependency
conflicts:

.. code-block:: bash

   conda create -n pyhermes python=3.10
   conda activate pyhermes

Install from source
-------------------

Clone the repository and install the package:

.. code-block:: bash

   git clone https://github.com/PyHermes/PyHermes.git
   cd PyHermes
   pip install -r requirements.txt
   pip install .

Install from PyPI
-----------------

If you are using a published release, install with:

.. code-block:: bash

   pip install pyhermes

Optional MPI support
--------------------

If you want parallel execution with MPI, install ``mpi4py`` after your MPI
runtime is available:

.. code-block:: bash

   pip install mpi4py

See `mpi4py on PyPI <https://pypi.org/project/mpi4py/>`_ for platform-specific
installation notes.

Verify the installation
-----------------------

You can verify that the package imports correctly with:

.. code-block:: bash

   python -c "import pyhermes; print(pyhermes.__version__)"

Next step
---------

Once installation is complete, continue with :doc:`get_start/get_start`.
