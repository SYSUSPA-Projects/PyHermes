Quick Start
===========

``quick_start.ipynb`` is the smallest PyHermes example. It uses
``examples/configs/param_convols.yaml`` to build a field, then demonstrates the
core numerical idea in one short path:

1. build a ``ConvolsData`` field from particle positions
2. subtract a uniform random field to form ``delta``
3. smooth the field with a spherical window
4. estimate a simple two-point statistic with shell convolutions

When to use this notebook
-------------------------

Use ``quick_start.ipynb`` when the example halo data have already been prepared
and you want a compact end-to-end calculation.

Use ``convols.ipynb`` first when you are starting from a fresh clone, because
that notebook prepares the example data used by ``param_convols.yaml``.

Input expectation
-----------------

The notebook reads the same configuration file used by the main ``Convols``
example:

.. code-block:: text

   examples/configs/param_convols.yaml

That config points to the local Quijote halo example prepared by
``convols.ipynb``:

.. code-block:: text

   examples/data/quijote_halos/8000

If that directory is not present yet, start with :doc:`convols/convols`.

What you should take away
-------------------------

The important idea is that PyHermes turns particle data into a reusable field
object. Later notebooks keep the same pattern, but add task-level workflows,
saved outputs, redshift-space variants, and heavier estimators.

Mathematical idea
-----------------

The quick start uses the core in-situ 2PCF identity:

.. math::

   \xi(R)
   =
   \left\langle
   \delta_W(\mathbf{x})\,
   (W_{\rm shell}(R)\circ\delta_W)(\mathbf{x})
   \right\rangle.

Here ``Convols`` supplies the normalized field, the notebook forms
:math:`\delta`, smooths it into :math:`\delta_W`, and each shell convolution
reads one separation bin. For the full notation, see :doc:`../math`.
