Convolution
===========


This is the base step to touch the whole truth.

We provide example file in the format of generic, 

.. _download_example:

Download the file here: :download:`quijote10000.bin <_static/quijote10000.bin>`

Ok, first you need parameter file, in the format of ``JSON``. See example below, we create parameter file named `param_convols.json`:


.. code-block:: json

    {
        "Convols": {
            "J": 9,
            "fin": {
                "path": "https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin",
                "format": "generic"
            },
            "fout_dir": "./",
            "SampRate": 1024,
            "SimBoxL": 1000,
            "window_type": "shell",
            "wavelet_mode": "db2",
            "wavelet_level": 10,
            "Radius": 5,
            "bandwidth": 1,
            "threads": 2
        }
    }

for the defination of these parameters, please see :ref:.

Then we can create python script here, i.g., `run_pyhermes_convols.py`

.. code:: python

    from pyhermes.base.convols import Convols
    from pyhermes.param.parambase import read_param

    param_input = read_param()
    Convols(param_task=param_input).run()

then run the command in terminal:


.. prompt:: bash $user@linux, auto

   $user@linux mpirun -n 8 python run_pyhermes_convols.py -c param_convols_mpi.json
    17:43:39 - INFO - pyhermes.param.parambase:JsonBase - Reading configure file: 'param_convols_mpi.json'
    17:43:39 - INFO - pyhermes.param.parambase:JsonBase - Set default parameters of module <base> ...
    17:43:39 - INFO - pyhermes.param.parambase:JsonBase - Default 'Convols.fin.path' from './data.bin' to 'https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin'
    17:43:39 - INFO - pyhermes.param.parambase:JsonBase - Default 'Convols.fout_dir' from 'empty' to './'
    17:43:39 - INFO - pyhermes.param.parambase:JsonBase - Default 'Convols.window_type' from 'sphere' to 'shell'
    17:43:39 - INFO - pyhermes.param.parambase:JsonBase - Default 'Convols.threads' from '5' to '2'
    17:43:39 - INFO - pyhermes.pipeline.pipeline:Convols - The task will run on 8 MPI ranks
    17:43:39 - INFO - pyhermes.io.funcs:read_particle_data - Selected input particle format: generic
    17:43:40 - INFO - pyhermes.io.funcs:dl_rich_pbar - Downloading file from 'https://pyhermes.astroslacker.com/_downloads/906e0695649e3634a5fe8081b9ab2086/quijote10000.bin'
    Downloading quijote10000.bin... ━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 2.4 MB/s 0:00:00
    17:43:43 - INFO - pyhermes.io.funcs:read_generic - Reading paricle data from ---> quijote10000.bin <---
    17:43:43 - INFO - pyhermes.pipeline.pipeline:Convols - Start partition ...
    17:43:43 - INFO - pyhermes.pipeline.pipeline:Convols - The time for partition data: 0.0094 sec
    17:43:43 - INFO - pyhermes.pipeline.pipeline:Convols - Start to calculate scaling coefficient...
    17:43:47 - INFO - pyhermes.pipeline.pipeline:Convols - The time for scaling function: 3.1818 sec
    17:43:47 - INFO - pyhermes.utils.math_util:set_window_function - Using window function: shell
    17:43:49 - INFO - pyhermes.pipeline.pipeline:Convols - Start to calculte FFT
    17:43:52 - INFO - pyhermes.pipeline.pipeline:Convols - The time for FFT: 2.8357 sec
    17:43:52 - INFO - pyhermes.io.base:ColvolsData - Writing data to ---> ./convols_L512_r5_pywt.npy <---

    17:43:59 - INFO - pyhermes.pipeline.pipeline:Convols - The time for task: 19.9119 sec
    17:43:59 - INFO - pyhermes.pipeline.pipeline:Convols - Bye.

This will first download the online example `quijote`:ref: simulation data in format of ``generic``, saved to disk of current workdir named *quijote10000.bin*, then use 8 MPI ranks to run the task of ``Convols``.
