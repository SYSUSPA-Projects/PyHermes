Corr_3PCF
=========

``corr3pcf.ipynb`` is the most advanced tutorial notebook in the repository.
It combines three related topics in one place:

1. standard reduced 3PCF measurements
2. low-level reconstruction of ``Q`` from saved components
3. 3PCF multipole examples

Notebook structure
------------------

The advanced multipole material is the last section of ``corr3pcf.ipynb``
because it builds naturally on the same prepared fields and triangle
configuration ideas as the standard 3PCF sections.

What this notebook covers
-------------------------

The main progression is:

1. run the standard task with particle centers and box-random centers
2. compare saved outputs and task-level API variants
3. reconstruct ``Q`` from lower-level ingredients to make the estimator more
   transparent
4. inspect multipole truncation choices and field-resolution effects

This is the notebook to read when you want both the practical workflow and the
estimator logic.

Heavy outputs and external runs
-------------------------------

Many of the comparison figures in this notebook rely on heavier saved products.
Those outputs are not committed to the repository. Instead, the notebook marks
the exact script and YAML file needed to generate them locally.

Tracked files involved here include:

- ``examples/notebooks/corr3pcf.ipynb``
- ``examples/scripts/run_3pcf.py``
- ``examples/scripts/run_3pcf_multipole.py``
- the ``param_3pcf_*.yaml`` and ``param_3pcf_multipole_*.yaml`` files under
  ``examples/configs/``

Typical commands look like:

.. code-block:: bash

   cd examples
   mpirun -np 4 python ./scripts/run_3pcf.py ./configs/param_3pcf_pcenter_nrot20.yaml
   mpirun -np 4 python ./scripts/run_3pcf.py ./configs/param_3pcf_rcenter_nrot20.yaml

How to read the notebook
------------------------

If you are new to PyHermes 3PCF, focus first on the standard workflow and the
diagnostic plots.

Then read the low-level ``Q`` reconstruction section. That part explains why
the saved components are enough to rebuild the reduced statistic and is the
best place to connect the code to the estimator formula.

Finally, treat the multipole section as an advanced extension. It is still part
of the same notebook, but it is more computationally demanding and more useful
once the standard 3PCF workflow is already familiar.
