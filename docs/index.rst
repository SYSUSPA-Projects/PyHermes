PyHermes documentation
======================

PyHermes is a workflow-oriented package for particle-based cosmic statistics.
This guide follows the same structure as the repository examples: install the
package, open the notebooks in ``examples/notebooks/``, build a field with
``Convols``, learn the field/window algebra with ``WindowFunc``, and then reuse
that field for ``Counting``, ``Corr_2PCF``, and ``Corr_3PCF``. After the main
multipoint-statistics workflow, the guide also includes a weighted-field
application showing how the same machinery can construct velocity and
momentum-density fields.

The repository tracks notebooks, scripts, and YAML configs. Example data and
most outputs are generated locally while you work through the tutorials.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   intro
   math
   windows
   install
   get_start/get_start
   param/param
   benchmark
