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
- ``cache_multipole_fields`` and ``cache_dir``: optional disk cache for intermediate convolution fields

Notes
-----

This task computes ``zeta_l(r1, r2) = <(D-R)(D-R)(D-R)>_l / <RRR>`` with
``R = 1 / V`` and ``<RRR> = R^3``. The convolution stage runs on CPU while the
local summation stage requires CUDA.

For the currently supported low-order range ``l <= 7``, the implementation uses
explicit fast-path Legendre window functions adapted from the validated legacy
workflow. Only the subset of ``m`` values actually used by the final
contractions is convolved, and the CUDA summation stage performs a device-side
reduction before transferring the result back to host memory. In practice this
combination substantially reduces first-run time compared with the original
legacy scripts while preserving numerical agreement.
