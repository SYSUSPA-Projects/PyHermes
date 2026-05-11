Installation
============

PyHermes runs in a normal Python environment. MPI support is optional.

Recommended setup
-----------------

Using a fresh conda environment is the simplest way to avoid dependency
conflicts:

.. code-block:: bash

   conda create -n pyhermes python=3.10
   conda activate pyhermes

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/PyHermes/PyHermes.git
   cd PyHermes
   pip install -r requirements.txt
   pip install .

The ``fof`` reader used by the Quijote halo examples imports ``readfof``.
This module is provided by ``Pylians``, so installing ``requirements.txt`` is
enough for the standard examples. On clusters, ``Pylians`` may need a working
C/C++ compiler and OpenMP runtime during installation.

Install from PyPI
-----------------

If you are using a published release:

.. code-block:: bash

   pip install pyhermes

Optional MPI support
--------------------

If you want multi-process execution, install ``mpi4py`` after your MPI runtime
is available:

.. code-block:: bash

   pip install mpi4py

See `mpi4py on PyPI <https://pypi.org/project/mpi4py/>`_ for platform-specific
installation notes.

Verify the install
------------------

.. code-block:: bash

   python -c "import pyhermes; print(pyhermes.__version__)"

What to open next
-----------------

After installation, start with :doc:`get_start/convols/convols` to prepare the
main example data in ``examples/data/``. Once that data exists, open
:doc:`get_start/quick_start` for the smallest end-to-end calculation.
