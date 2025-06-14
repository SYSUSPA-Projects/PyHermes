Counting
========


Counting should be displayed here.


Users should prepare the parameter file in the format of ``JSON``. See example below, we create parameter file named `param_counting.json`:



.. code-block:: json

    {
    "Counting": {
        "n_tasks": 100,
        "deltac_in_path": "./convols_L512_r5_pywt.npy",
        "fout_dir": "./"
      }
    }

for the defination of these parameters, please see :ref:.

Then we create python script here, i.g., `run_pyhermes_counting.py`


.. code:: python

    from pyhermes.theory.counting import Counting
    from pyhermes.param.parambase import read_param

    param_input = read_param()
    Counting(param_task=param_input).run()

then run the command in terminal: