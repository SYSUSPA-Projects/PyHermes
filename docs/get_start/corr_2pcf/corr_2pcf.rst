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

Mathematical idea
-----------------

PyHermes treats a 2PCF bin as a windowed copy of the same fluctuation field:

.. math::

   \xi_P =
   \left\langle
   \delta(\mathbf{x})\,
   (W_P\circ\delta)(\mathbf{x})
   \right\rangle.

For real-space ``xi(s)``, :math:`W_P` is a spherical shell. In the thin-shell
limit,

.. math::

   W_{\rm shell}(r;R)
   =
   {1\over 4\pi R^2}\delta_{\rm D}(r-R),
   \qquad
   \widehat W_{\rm shell}(k;R)
   =
   {\sin(kR)\over kR}.

For redshift-space ``(s, mu)`` and ``(rp, pi)`` measurements, the same
estimator uses line-of-sight-aware windows. A thin ring window has

.. math::

   \widehat W(k_\perp,k_\parallel)
   =
   e^{i k_\parallel r_\parallel}J_0(k_\perp r_\perp),

with finite-bin and real-valued variants implemented by the built-in ring,
disk, cylinder, and ``cylshell`` windows. Random fields provide the ``RR``
normalization and the data-minus-random correction used in the estimator.
