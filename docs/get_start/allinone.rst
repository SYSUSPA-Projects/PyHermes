Run all in one
===============

PyHermes tasks can be chained in a single Python script. This is convenient
when you want to keep intermediate objects in memory instead of reloading them
from disk between stages.

This page is best thought of as a cross-task version of **Workflow B / C**:
you still use the standard task objects, but you orchestrate several of them in
one Python driver.

Single parameter file
---------------------

You can place multiple task sections into one YAML or JSON5 file. For example:

.. code-block:: yaml

   Convols:
      J: 8
      fin:
         path: "https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin"
         format: "generic_pos"
      fout_path: "./output/quijote_sfc.pkl"
      phi_resolution: 1024
      box_size: 1000
      wavelet_mode: "db2"
      wavelet_level: 10

   Counting:
      N_randoms: 1000000
      convols_data_path: "./output/quijote_sfc.pkl"
      fout_path: "./output/quijote_counting.pkl"
      window:
         type: "sphere"
         len_args:
            R: 20

   Corr_2PCF:
      convols_data_path: "./output/quijote_sfc.pkl"
      fout_path: "./output/quijote_2pcf.pkl"
      r_min: 1.0
      r_max: 150.0
      n_r: 30

   Corr_3PCF:
      convols_data_path: "./output/quijote_sfc.pkl"
      fout_path: "./output/quijote_3pcf.pkl"
      r12: 20.0
      r13: 40.0
      n_theta: 20
      n_rot: 20
      center: "random"
      n_rand: 1000000
      base_seed: 42

   Corr_3PCF_Multipole:
      convols_data_path: "./output/quijote_sfc.pkl"
      fout_path: "./output/quijote_3pcf_multipole.pkl"
      r12: 20.0
      r13: 40.0
      l_max: 4
      gpu_device_id: 0

Driver script
-------------

.. code-block:: python

   from pyhermes.base.convols import Convols
   from pyhermes.theory.counting import Counting
   from pyhermes.theory.corr2pcf import Corr_2PCF
   from pyhermes.theory.corr3pcf import Corr_3PCF
   from pyhermes.theory.corr3pcf_multipole import Corr_3PCF_Multipole
   from pyhermes.param.parambase import read_param

   params = read_param(config_path="./param_multi.yaml")

   convols = Convols(param_task=params)
   convols.run(overwrite=True)

   counting = Counting(param_task=params)
   counting.run(overwrite=True)

   corr2pcf = Corr_2PCF(param_task=params)
   corr2pcf.run(overwrite=True)

   corr3pcf = Corr_3PCF(param_task=params)
   corr3pcf.run(overwrite=True)

   corr3pcf_multipole = Corr_3PCF_Multipole(param_task=params)
   corr3pcf_multipole.run(overwrite=True)

When to use this pattern
------------------------

Use the all-in-one style when:

- you want one config file for a pipeline run
- you prefer orchestrating tasks from a Python script
- you are building your own wrapper workflow around PyHermes

If you prefer simpler, task-specific examples, start with the dedicated pages
for :doc:`convols/convols`, :doc:`counting/counting`, :doc:`corr_2pcf/corr_2pcf`,
 :doc:`corr_3pcf/corr_3pcf`, and :doc:`corr_3pcf_multipole/corr_3pcf_multipole`.
