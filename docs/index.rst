PyHermes documentation
======================

PyHermes is a workflow-oriented package for particle-based cosmic statistics.
The central idea is simple: project a particle catalog into a reusable
``SFCField`` field, act on that field with ``WindowFunc`` convolution
operators, and read out one-point, two-point, three-point, multipole, or
weighted-field measurements from the resulting field products.

The guide follows the example workflow in ``examples/notebooks/``. New users
should start with the introduction, install the package, then work through the
getting-started notebooks. The mathematical and window-function pages provide
the compact reference behind those examples.

The repository tracks notebooks, scripts, and YAML configs. Example data and
most outputs are generated locally while you work through the tutorials.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   intro
   install
   get_start/get_start
   math
   windows
   param/param
   benchmark
