Corr_2PCF
=========

``corr2pcf.ipynb`` covers both the standard isotropic two-point correlation
function ``xi(s)`` and the anisotropic redshift-space views ``(s, mu)`` and
``(rp, pi)``.

What this notebook covers
-------------------------

The notebook has two main halves:

1. isotropic 2PCF on a one-dimensional separation grid
2. anisotropic 2PCF in redshift space, including line-of-sight changes,
   smoothing choices, and alternative pair-window families

It also contains lower-level sections for custom pair windows and direct
estimator comparisons. Those sections are useful when you want to understand
what the task wrapper is doing under the hood.

What is lightweight and what is not
-----------------------------------

Small smoke-test runs and direct API examples are meant to be executed inside
the notebook.

The production-style outputs compared later in the notebook are heavier and are
not committed to the repository. At each relevant point, the notebook tells you
which script and YAML file to run. Those jobs are intended for your own laptop,
workstation, or cluster.

Tracked files
-------------

- ``examples/notebooks/corr2pcf.ipynb``
- ``examples/scripts/run_2pcf.py``
- ``examples/configs/param_2pcf.yaml``
- the additional anisotropic configs in ``examples/configs/param_2pcf_*.yaml``

Typical command-line runs look like:

.. code-block:: bash

   cd examples
   python ./scripts/run_2pcf.py ./configs/param_2pcf.yaml
   mpirun -np 4 python ./scripts/run_2pcf.py ./configs/param_2pcf_smu_test.yaml

Conceptual focus
----------------

This is the notebook where PyHermes' pair-window abstraction becomes important.
Instead of hard-coding one estimator shape, the task combines:

- a prepared field
- optional smoothing windows
- a pair window
- a sampling grid

That is why the same task can cover ``xi(s)``, ``xi(s, mu)``, and
``xi(rp, pi)`` in one interface.

What to carry forward
---------------------

If you mainly care about real-space 2PCF, the first half is enough.

If you care about redshift-space distortions, the second half is the more
important reference because it shows how line-of-sight choice, smoothing, and
window family affect the result.
