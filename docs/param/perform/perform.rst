Parallel execution and HPC
==========================

PyHermes combines MPI ranks with Numba threads. The two levels serve different
purposes: MPI distributes independent fields, samples, or modes; ``threads``
accelerates the numerical work local to each rank.

Launch pattern
--------------

.. code-block:: bash

   cd examples
   export OMP_NUM_THREADS=4
   mpirun -np 8 --map-by ppr:8:node:pe=4 --bind-to core \
       python scripts/run_2pcf.py configs/param_2pcf.yaml

Set the task's ``threads`` to the same value as the CPU allocation per rank.
Avoid nested oversubscription: ``ranks * threads`` should not exceed the cores
allocated by the scheduler.

How tasks distribute work
-------------------------

.. list-table:: Parallel work by task
   :header-rows: 1
   :widths: 25 75

   * - Task
     - Main distribution
   * - ``SFCProjection``
     - Particles are divided across ranks and reduced into the shared MRA
       coefficient field.
   * - ``Counting``
     - Random evaluation positions are divided among ranks.
   * - ``Corr_2PCF``
     - Separation samples and product work are distributed across ranks.
   * - ``Corr_3PCF``
     - Centre and rotation work is distributed; rank 0 assembles outputs.
   * - ``Corr_3PCF_Multipole`` ``pair_mpi``
     - Even rank pairs divide multipole-field convolution work for each sample.
   * - ``Corr_3PCF_Multipole`` ``sample_mpi``
     - Radial samples are assigned round-robin, one rank per sample in the
       current implementation.

Multipole CPU and GPU work
--------------------------

The 3PCF multipole convolution is CPU/MPI work for both backends.
``summation_backend`` changes the final filtered-field contraction only:

.. code-block:: yaml

   execution_mode: "pair_mpi"
   summation_backend: "gpu"   # or "cpu"
   gpu_device_id: 0
   gpu_threads_per_block: [8, 8, 8]

For ``sample_mpi`` on a CPU node, use approximately one rank per concurrently
active sample and assign several threads to each rank. On a single-GPU node,
many simultaneous GPU ranks usually contend for the same device; ``pair_mpi``
or a smaller number of sample ranks is often easier to tune.

Memory planning
---------------

The base coefficient count is :math:`2^{3J}`. A one-level increase in ``J``
multiplies it by eight, and temporary real/complex FFT arrays multiply the
resident bytes further. MPI ranks may also hold rank-local field buffers.

Practical rules:

- Validate a configuration at :math:`J=7` or :math:`J=8` before a large run.
- At high ``J``, prefer fewer ranks and more threads when rank-local field
  replication dominates memory.
- Use ``memory_strategy: memory`` for a many-product 2PCF when peak memory is
  more important than rebuilding work.
- Enable field or kernel caches only when repeated computation outweighs I/O;
  place caches on fast local storage when available.
- Request memory from measured ``MaxRSS`` plus headroom, not from the serialized
  output size.

Input I/O
---------

In multipole ``sample_mpi`` mode, rank 0 reads a file-backed ``SFCField`` and
broadcasts it. This reduces metadata and read contention on shared storage.
For large production arrays, storage location can still affect startup and
cache performance; copy hot inputs to node-local scratch if the cluster makes
that workflow available.

Slurm template
--------------

.. code-block:: bash

   #!/bin/bash
   #SBATCH --nodes=1
   #SBATCH --ntasks=24
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=36G

   export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
   export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

   srun --cpu-bind=cores python tests/scripts/run_3pcf_multipole.py \
       --config tests/configs/param_3pcf_multipole_lmax7.yaml \
       --summation-backend cpu

Scheduler syntax and MPI integration vary by site. If ``mpirun`` reports that a
rank spans CPU packages, adjust the scheduler's task placement or mapper rather
than enabling overload. A running process with poor binding can be slower than
a smaller, correctly placed job.

Measurement and reproducibility
-------------------------------

Use the task logs for phase timings and Slurm accounting for whole-job memory:

.. code-block:: bash

   sacct -j JOBID --units=G \
       -o JobID,JobName%24,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,AveRSS

Keep ``base_seed`` or ``seed`` fixed when comparing algorithms. Repeat timing
runs under comparable node load; convolution-heavy workloads are sensitive to
memory bandwidth and CPU placement even when the estimator output is identical.
