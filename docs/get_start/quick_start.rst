Quick Start
===========

``quick_start.ipynb`` is the smallest PyHermes example. It does not try to
teach every task interface. Instead, it shows the core numerical idea in one
short path:

1. build a ``ConvolsData`` field from particle positions
2. subtract a uniform random field to form ``delta``
3. smooth the field with a spherical window
4. estimate a simple two-point statistic with shell convolutions

When to use this notebook
-------------------------

Use ``quick_start.ipynb`` when you want to understand what the field
representation is doing before you move on to the more complete task notebooks.

Use ``convols.ipynb`` instead when you want the repository-tracked example
workflow, because that notebook includes the data-preparation steps for the
main Quijote halo example.

Input expectation
-----------------

The notebook assumes a small local particle catalog at:

.. code-block:: text

   examples/data/quijote10000.bin

That file is not committed to the repository. If you do not already have it,
start with :doc:`convols/convols`.

What you should take away
-------------------------

The important idea is that PyHermes turns particle data into a reusable field
object. Later notebooks keep the same pattern, but replace this toy
end-to-end example with task-level workflows, saved outputs, redshift-space
variants, and heavier estimators.
