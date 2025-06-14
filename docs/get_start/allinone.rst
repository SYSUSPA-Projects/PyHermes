Run all in one
===============


Of course users can run all code in just one python script. 

For example to calculate 3PCF, we create parameter file `param_multi.json`, which contains all the parameters to calculate:

.. code-block:: json

   {
      "Convols": {
         "J": 9,
         "fin": {
            "path": "https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin",
            "format": "generic",
         },
         "fout_path": "./multi_test.npy",
         "SampRate": 1024,
         "SimBoxL": 1000,
         "window": {
            "type": "shell",
            "R": 5,
         },
         "wavelet_mode": "db2",
         "wavelet_level": 10,
         "bandwidth": 1,
         "threads": 2
      },
      "Corr_2PCF": {
         "deltac_in_path": "", //not necessary, cause will load from argv
         "fout_path": "./multi_test_2pcf.txt",
         "threads": 20,
         "R1": 1.0,
         "R2": 150.0,
         "xi_num": 200
      },
      "Corr_3PCF": {
         "fin": {
            "path": "", //not necessary, cause will load from argv
            "format": "" //not necessary, cause will load from argv
         },
         "deltac_in_path": "", //not necessary, cause will load from argv
         "corr2pcf_in_path" : "", //not necessary, cause will load from argv
         "fout_path": "./multi_test_3pcf.txt",
         "NStheta": 20,
         "R1": 20.0,
         "R2": 40.0,
         "rot_num": 3
      }
   }

Not a fan of *JSON*? No worries! ``PyHermes`` happily supports both *JSON* and *YAML* for your parameter files. Pick your favorite and get started :)

For instance, `param_multi.yaml`:

.. code:: YAML

   Convols:
      J: 9
      fin:
         path: "https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin"
         format: "generic"
      fout_path: "./multi_test.npy"
      SampRate: 1024
      SimBoxL: 1000
      window:
         type: "shell"
         R: 5
      wavelet_mode: "db2"
      wavelet_level: 10
      bandwidth: 1
      threads: 2

   Corr_2PCF:
      deltac_in_path: "" # not necessary, cause will load from argv
      fout_path: "./multi_test_2pcf.txt"
      threads: 20
      R1: 1.0
      R2: 150.0
      xi_num: 200

   Corr_3PCF:
      fin:
         path: "" # not necessary, cause will load from argv
         format: "" # not necessary, cause will load from argv
      deltac_in_path: "" # not necessary, cause will load from argv
      corr2pcf_in_path: "" # not necessary, cause will load from argv
      fout_path: "./multi_test_3pcf.txt"
      NStheta: 20
      R1: 20.0
      R2: 40.0
      rot_num: 3

this is completely equivalent to the JSON format file `param_multi.json` above!

Now lets create the python script, `run_multi.py`:

.. code:: python

   from pyhermes.base.convols import Convols
   from pyhermes.theory.corr2pcf import Corr_2PCF
   from pyhermes.theory.corr3pcf import Corr_3PCF
   from pyhermes.param.parambase import read_param
   
   # Read all parameters
   #  Both JSON (.json) and YAML (.yaml) formats are supported!
   param_input = read_param(config_path='./param_multi.json')
   # Calculate deltac, then return deltac and particle data p_dm
   deltac, p_dm = Convols(param_task=param_input).run(return_pData=True)
   # Calculate 2pcf, using the deltac from the result of above step
   corr2pcf = Corr_2PCF(param_task=param_input).run(deltac=deltac)
   # Calculate 3pcf, using the deltac from the results of above steps
   Corr_3PCF(param_task=param_input).run(deltac=deltac, corr2pcf=corr2pcf, p_dm=p_dm)

just run it in your terminal, like

.. prompt:: bash $pyhermes_user@toymachine, auto

   $pyhermes_user@toymachine mpirun -n 8 python run_multi.py
    15:00:14 - INFO - pyhermes.param.parambase:JsonBase - Reading configure file: './param_multi.json'
    15:00:14 - INFO - pyhermes.param.parambase:JsonBase - Set default parameters of module <base> ...
    15:00:14 - INFO - pyhermes.param.parambase:JsonBase - Default 'Convols.fin.path' from './data.bin' to 'https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin'
    15:00:14 - INFO - pyhermes.param.parambase:JsonBase - Default 'Convols.fout_path' from 'empty' to './multi_test.npy'
    15:00:14 - INFO - pyhermes.param.parambase:JsonBase - Adding customizable window arg: 'Convols.window.R' as '5'
    15:00:14 - INFO - pyhermes.param.parambase:JsonBase - Default 'Convols.threads' from '1' to '2'

    15:00:14 - INFO - pyhermes.pipeline.pipeline:Convols - The task will run on 8 MPI ranks
    15:00:14 - INFO - pyhermes.io.funcs:read_particle_data - Selected input particle format: generic
    15:00:15 - INFO - pyhermes.io.funcs:dl_rich_pbar - File 'quijote10000.bin' already exists. Skipping download.
    15:00:15 - INFO - pyhermes.io.funcs:read_generic - Reading paricle data from ---> quijote10000.bin <---
    15:00:15 - INFO - pyhermes.pipeline.pipeline:Convols - Start partition ... 
    15:00:15 - INFO - pyhermes.pipeline.pipeline:Convols - The time for partition data: 0.0083 sec
    15:00:15 - INFO - pyhermes.pipeline.pipeline:Convols - Start to calculate scaling coefficient... 
    15:00:18 - INFO - pyhermes.pipeline.pipeline:Convols - The time for scaling function: 3.2623 sec
    15:00:18 - INFO - pyhermes.utils.math_util:set_window_function - Using window function: shell
    15:00:21 - INFO - pyhermes.pipeline.pipeline:Convols - Start to calculte FFT
    15:00:24 - INFO - pyhermes.pipeline.pipeline:Convols - The time for FFT: 2.8320 sec
    15:00:24 - INFO - pyhermes.io.base:ConvolsData - Writing data to ---> ./multi_test.npy <---

    15:00:26 - INFO - pyhermes.pipeline.pipeline:Convols - The time for task: 11.2947 sec
    15:00:26 - INFO - pyhermes.pipeline.pipeline:Convols - Bye.


    15:00:26 - INFO - pyhermes.param.parambase:JsonBase - Set default parameters of module <theory> ...
    15:00:26 - INFO - pyhermes.param.parambase:JsonBase - Default 'Corr_2PCF.fout_path' from 'empty' to './multi_test_2pcf.txt'
    15:00:26 - INFO - pyhermes.param.parambase:JsonBase - Default 'Corr_2PCF.threads' from '1' to '20'
    15:00:26 - INFO - pyhermes.param.parambase:JsonBase - Default 'Corr_2PCF.xi_num' from '150' to '200'

    15:00:26 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF - The task will run on 8 MPI ranks
    15:00:26 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF - Loading DeltaC from argument 'deltac'
    15:00:27 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF - Start to calculate 2PCF ...
    15:00:29 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF -  Progress:   0.00%
    15:00:38 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF -  Progress:  14.00%
    15:00:42 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF -  Progress:  22.00%
    15:00:46 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF -  Progress:  30.00%
    15:00:53 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF -  Progress:  40.00%
    15:00:59 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF -  Progress:  53.00%
    15:01:03 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF -  Progress:  61.00%
    15:01:10 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF -  Progress:  72.00%
    15:01:14 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF -  Progress:  80.00%
    15:01:21 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF -  Progress:  93.00%
    15:01:23 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF -  Progress: 100.00%
    15:01:23 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF - The time for 2PCF: 56.0502 sec
    15:01:23 - INFO - pyhermes.io.base:Corr2PCFData - Writing data to ---> ./multi_test_2pcf.txt <---

    15:01:23 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF - The time for task: 56.7735 sec
    15:01:23 - INFO - pyhermes.pipeline.pipeline:Corr_2PCF - Bye.


    15:01:23 - INFO - pyhermes.param.parambase:JsonBase - Set default parameters of module <theory> ...
    15:01:23 - INFO - pyhermes.param.parambase:JsonBase - Default 'Corr_3PCF.fin.path' from './data.bin' to ''
    15:01:23 - INFO - pyhermes.param.parambase:JsonBase - Default 'Corr_3PCF.fin.format' from 'generic' to ''
    15:01:23 - INFO - pyhermes.param.parambase:JsonBase - Default 'Corr_3PCF.fout_path' from 'empty' to './multi_test_3pcf.txt'
    15:01:23 - INFO - pyhermes.param.parambase:JsonBase - Default 'Corr_3PCF.rot_num' from '10000' to '3'

    15:01:23 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF - The task will run on 8 MPI ranks
    15:01:23 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF - Loading Corr2pcf from argument 'corr2pcf'
    15:01:23 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF - Hi tristan, now you already have the corr2pcf info!
    15:01:23 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF - Loading DeltaC from argument 'deltac'
    15:01:23 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF - Loading Particle data from argument 'p_dm'
    15:01:23 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF - Start to calculate 3PCF ...
    15:01:25 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF -  Progress:   0.00%
    15:01:25 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF -  Progress:  16.67%
    15:01:25 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF -  Progress:  19.05%
    15:01:26 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF -  Progress:  28.57%
    15:01:26 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF -  Progress:  38.10%
    15:01:26 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF -  Progress:  47.62%
    15:01:27 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF -  Progress:  57.14%
    15:01:27 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF -  Progress:  66.67%
    15:01:27 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF -  Progress:  76.19%
    15:01:28 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF -  Progress:  85.71%
    15:01:28 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF -  Progress: 100.00%
    15:01:28 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF - Finished in 4.4 sec
    15:01:28 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF - The time for 3PCF: 4.4160 sec
    15:01:28 - INFO - pyhermes.io.base:Corr3PCFData - Writing data to ---> ./multi_test_3pcf.txt <---

    15:01:28 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF - The time for task: 4.9321 sec
    15:01:28 - INFO - pyhermes.pipeline.pipeline:Corr_3PCF - Bye.
