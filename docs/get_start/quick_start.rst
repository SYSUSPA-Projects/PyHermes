Quick start
===========

This page runs one complete field-to-2PCF workflow. The matching interactive
version is ``examples/notebooks/quick_start.ipynb``.

Prepare the example field
-------------------------

From the repository root:

.. code-block:: bash

   python examples/scripts/prepare_sfc_fields.py

The preparation script downloads or reads the Quijote halo example and writes
``examples/output/quijote8000_snap004_sfc.pkl``.

Run a task from YAML
--------------------

.. code-block:: bash

   cd examples
   python scripts/run_2pcf.py configs/param_2pcf.yaml

The task writes a ``Corr2PCFData`` product. Load it without knowing its internal
serialisation format:

.. code-block:: python

   from pyhermes.io import Corr2PCFData

   corr = Corr2PCFData(
       data_path="./output/quijote8000_snap004_2pcf.pkl"
   )
   print(corr.s)
   print(corr.xi)

Run and override from Python
----------------------------

YAML is useful for reproducible runs, while task attributes are convenient for
small experiments:

.. code-block:: python

   import numpy as np

   from pyhermes.param.parambase import read_param
   from pyhermes.theory.corr2pcf import Corr_2PCF

   params = read_param(config_path="./configs/param_2pcf.yaml")
   task = Corr_2PCF(params)
   task.sampling = {"s": np.linspace(0.0, 180.0, 46)}
   task.products = ["xi"]
   task.fout_path = "./output/quick_start_2pcf.pkl"
   corr = task.run()

See the operation directly
--------------------------

The high-level task is assembled from the same lower-level field and window
objects exposed to users:

.. code-block:: python

   import numpy as np

   from pyhermes.io import SFCField, WindowFunc

   D = SFCField(data_path="./output/quijote8000_snap004_sfc.pkl", threads=8)
   rho = D.field_mean_density(value_unit="grid")
   delta = D - rho

   smooth = WindowFunc(
       {"type": "sphere", "len_args": {"R": 5.0}},
       D.sfc_info,
       threads=8,
   )
   delta_s = delta @ smooth

   radii = np.linspace(0.0, 150.0, 31)
   xi = np.empty_like(radii)
   for i, radius in enumerate(radii):
       shell = WindowFunc(
           {"type": "shell", "len_args": {"R": radius}},
           D.sfc_info,
           threads=8,
       )
       xi[i] = ((delta_s @ shell) * delta_s).as_array().mean() / rho**2

``@`` applies a window by FFT convolution; ``*`` multiplies two fields
coefficient by coefficient. The task API handles data/random bookkeeping,
normalisation, sampling, MPI distribution, and output metadata around these
same operations.

Where next?
-----------

- :doc:`sfc_projection/sfc_projection` explains what is stored in ``D``.
- :doc:`window/window` explains the two operators used above.
- :doc:`corr_2pcf/corr_2pcf` covers random fields, anisotropic bins, and result
  products.
