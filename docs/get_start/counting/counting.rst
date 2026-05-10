Counting
========

``counting.ipynb`` is the one-point companion to ``convols.ipynb``. It starts
from a saved ``ConvolsData`` field, optionally smooths it, evaluates it at
many random positions, and studies the resulting distribution.

What this notebook covers
-------------------------

The notebook walks through:

1. the standard ``Counting`` driver
2. the config-driven Python API
3. task-object overrides
4. manual preparation of ``ConvolsData`` and ``WindowFunc``
5. direct low-level sampling of the smoothed field

The last section is useful because it makes the estimator interpretation
explicit: ``Counting`` is fundamentally a random-position probe of a field.

Inputs and outputs
------------------

Inputs are produced by ``convols.ipynb`` and live locally in
``examples/output/``. The main tracked files involved in this stage are:

- ``examples/notebooks/counting.ipynb``
- ``examples/scripts/run_counting.py``
- ``examples/configs/param_counting.yaml``

The counting result itself is lightweight and is expected to be generated
locally during notebook execution or by the driver script:

.. code-block:: bash

   cd examples
   python ./scripts/run_counting.py ./configs/param_counting.yaml

What you should learn here
--------------------------

This notebook is where the field representation starts to feel concrete. It
shows how smoothing radius changes the sampled distribution and how the saved
``CountingData`` result relates to direct field evaluation.

In other words, if ``Convols`` explains how PyHermes stores the field,
``Counting`` explains how PyHermes reads values back out of it.
