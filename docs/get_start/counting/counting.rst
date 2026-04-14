Counting
========

``Counting`` evaluates the field at many random positions. This is useful for
sampling local densities, estimating one-point distributions, and inspecting how
a previously prepared field behaves under additional smoothing.

Workflow Ladder
---------------

As with the other PyHermes tasks, ``Counting`` can be used at several levels:

- **Workflow A. Command-Line Driver**
- **Workflow B. Config-Driven Python API**
- **Workflow C. Task Object with Attribute Overrides**
- **Workflow D. Manual Input Objects and Custom Preparation**
- **Workflow E. Low-Level Building Blocks**

Workflow A. Command-Line Driver
-------------------------------

Use the shipped example config:

.. code-block:: yaml

   Counting:
      N_randoms: 10000000
      convols_data_path: "./output/quijote_sfc.pkl"
      fout_path: "./output/quijote_counting_sph20.pkl"
      window:
         type: sphere
         len_args:
            R: 20

Then run:

.. code-block:: bash

   python run_counting.py

Workflow B. Config-Driven Python API
------------------------------------

.. code-block:: python

   from pyhermes.theory.counting import Counting
   from pyhermes.param.parambase import read_param

   counting_params = read_param("./configs/param_counting.yaml")
   counting_task = Counting(param_task=counting_params)
   counting = counting_task.run(overwrite=True)

If you modify config values in an MPI workflow, do it only on rank 0.

Workflow C. Task Object with Attribute Overrides
------------------------------------------------

.. code-block:: python

   from pyhermes.theory.counting import Counting

   counting_task = Counting()
   counting_task.N_randoms = 1_000_000
   counting_task.threads = 8
   counting_task.convols_data_path = "./output/quijote_sfc.pkl"
   counting_task.prepare_input_fields()
   counting = counting_task.run(save_result=False)

Workflow D. Manual Input Objects and Custom Preparation
-------------------------------------------------------

If you already have a prepared field or a custom smoothing window, inject them
directly:

.. code-block:: python

   from pyhermes.io import ConvolsData, WindowFunc
   from pyhermes.theory.counting import Counting

   D = ConvolsData(data_path="./output/quijote_sfc.pkl")
   win_params = {"type": "sphere", "len_args": {"R": 20}}
   filter_sph20 = WindowFunc(win_params, D.convols_info)

   counting_task = Counting()
   counting_task.N_randoms = 1_000_000
   counting_task.convols_data = D.copy()
   counting_task.window = filter_sph20
   counting_task.prepare_input_fields()
   counting = counting_task.run(save_result=False)

Workflow E. Low-Level Building Blocks
-------------------------------------

At the lowest level, you can generate the random positions yourself and call
``n_at_pos`` directly on a prepared field. This is the most flexible option,
but it assumes you want to manage the sampling and post-processing manually.

Output
------

The example writes:

.. code-block:: text

   ./output/quijote_counting_sph20.pkl

Key parameters
--------------

- ``N_randoms``:
  number of random sampling points
- ``convols_data_path``:
  path to the ``Convols`` output file
- ``window``:
  optional smoothing window applied before counting
- ``threads``:
  CPU threads per MPI rank
- ``fout_path``:
  output path for the counting result

Notes
-----

- ``prepare_input_fields()`` prepares the counting field.
- ``run()`` generates random positions, evaluates the field, and gathers the result.
- ``Counting`` requires a previously generated ``Convols`` field unless you pass a
  prepared ``ConvolsData`` object directly.
