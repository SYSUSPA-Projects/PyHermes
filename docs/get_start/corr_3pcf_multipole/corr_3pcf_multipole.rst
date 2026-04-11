Corr_3PCF_Multipole
===================

``Corr_3PCF_Multipole`` measures ``zeta_l(r1, r2)`` from a saved multiresolution
field using streamed CPU-side convolutions and CUDA summation.

Example configuration
---------------------

The repository includes ``examples/configs/param_3pcf_multipole.yaml``:

.. code-block:: yaml

   Corr_3PCF_Multipole:
      convols_data_path: "./output/quijote_sfc.pkl"
      fout_path: "./output/quijote_3pcf_multipole.pkl"
      window2:
         type: "sphere"
         len_args:
            R: 5
      window3:
         type: "sphere"
         len_args:
            R: 5
      r1: 20.0
      r2: 40.0
      l_min: 0
      l_max: 4
      gpu_device_id: 0
      field_mode: "delta"
      execution_mode: "serial"
      threads: 1

Minimal Python driver
---------------------

.. code-block:: python

   from pyhermes.theory.corr3pcf_multipole import Corr_3PCF_Multipole
   from pyhermes.param.parambase import read_param

   params = read_param(config_path="./configs/param_3pcf_multipole.yaml")
   corr3pcf_multipole = Corr_3PCF_Multipole(param_task=params)
   corr3pcf_multipole.run(overwrite=True)

Run it
------

From the ``examples`` directory:

.. code-block:: bash

   python run_3pcf_multipole.py

Key parameters
--------------

- ``r1`` and ``r2``: the two side lengths defining the multipole family
- ``l_min`` and ``l_max``: minimum and maximum multipole order to compute
- ``gpu_device_id``: CUDA device index used for the summation stage
- ``field_mode``: choose ``"raw"`` to save ``ddd_l`` from ``<DDD>`` or ``"delta"`` to save ``delta_ddd_l`` and ``zeta_l``
- ``execution_mode``: use ``"serial"`` for the default single-rank workflow or ``"pair_mpi"`` for grouped MPI execution; with one rank it falls back to serial, and with MPI it expects an even number of ranks so that the `(r1,+m)` and `(r2,-m)` convolution legs can be processed in parallel batches before rank 0 performs the CUDA summation
- ``threads``: CPU threads used by the convolution stage; the current default is ``1``, and recent tests showed little difference between ``1`` and ``8`` for the present implementation
- ``cache_multipole_fields`` and ``cache_dir``: optional disk cache for intermediate convolution fields

Notes
-----

This task computes ``zeta_l(r1, r2) = <(D-R)(D-R)(D-R)>_l / <RRR>`` with
``R = 1 / V`` and ``<RRR> = R^3``. The convolution stage runs on CPU while the
local summation stage requires CUDA.

For the low-order range ``l <= 7``, the implementation uses explicit fast-path
Legendre window functions adapted from the validated legacy workflow. For
higher multipoles it automatically falls back to the generic recursive window
construction, which is supported but typically slower. Only the subset of
``m`` values actually used by the final contractions is convolved, and the CUDA
summation stage performs a device-side reduction before transferring the result
back to host memory. In practice this combination substantially reduces first-run
time compared with the original legacy scripts while preserving numerical
agreement.

For the current single-node GPU environment, the recommended high-performance
workflow is ``execution_mode: "pair_mpi"`` with an even number of MPI ranks and
``threads: 1``. In this mode the two convolution legs needed for each ``m`` are
processed in grouped MPI batches on the CPU side, while rank 0 performs the
CUDA summation. This mode substantially outperformed the single-rank
``serial`` workflow in the current benchmark setup.
