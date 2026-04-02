Performance
===========

PyHermes supports both single-process runs and MPI-based parallel execution.

MPI
---

If ``mpi4py`` is installed and you launch a task with ``mpirun``, PyHermes uses
multiple MPI ranks:

.. code-block:: bash

   mpirun -np 8 python run_2pcf.py

If ``mpi4py`` is not installed, PyHermes falls back to a single-process wrapper
so the same Python code can still run.

Threads
-------

Most task sections also accept:

- ``threads``: number of threads per MPI rank

When ``threads`` is set, PyHermes configures the numerical backend accordingly.
Use this to control per-rank CPU parallelism in addition to MPI rank count.

Practical guidance
------------------

- Start with single-process runs to validate configuration files.
- Add MPI only after the basic workflow succeeds.
- Increase ``threads`` carefully to avoid oversubscribing CPU resources when
  also using many MPI ranks.
