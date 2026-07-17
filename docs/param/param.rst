Configuration reference
=======================

PyHermes task configurations are YAML or JSON5 dictionaries with one top-level
task section:

- ``SFCProjection``
- ``Counting``
- ``Corr_2PCF``
- ``Corr_3PCF``
- ``Corr_3PCF_Multipole``

The task class loads its bundled defaults and overlays the user section. A
configuration therefore needs to state only the scientific choices and paths
that differ from the defaults.

.. toctree::
   :maxdepth: 1

   defaults/defaults
   io/io
   cal/cal
   perform/perform

Reading a configuration
-----------------------

.. code-block:: python

   from pyhermes.param.parambase import read_param
   from pyhermes.theory import Corr_2PCF

   params = read_param(config_path="./configs/param_2pcf.yaml")
   result = Corr_2PCF(params).run()

The public scripts accept the same file as their first positional argument:

.. code-block:: bash

   cd examples
   python scripts/run_2pcf.py configs/param_2pcf.yaml

Paths in a configuration are ordinary process-relative paths. The tracked
examples are written to run with ``examples/`` as the working directory.

Nested replacement rules
------------------------

Most dictionaries are merged recursively. A few structural dictionaries are
replaced as a whole because retaining an omitted coordinate or mapping from a
default would change the requested estimator:

- ``Corr_2PCF.binning_window`` and ``Corr_2PCF.sampling``
- ``Corr_3PCF_Multipole.binning_window12``
- ``Corr_3PCF_Multipole.binning_window13``
- ``Corr_3PCF_Multipole.sampling`` and ``sample_mpi``

Write those blocks completely in user configurations. Unknown keys produce a
warning but may still be retained; a warning is not evidence that the runtime
uses the key.

Programmatic changes
--------------------

When varying structural parameters from Python, modify the parameter dictionary
*before* constructing the task. Constructors normalize sampling arrays, expand
product dependencies, and build mapped window templates.

.. code-block:: python

   params = read_param(config_path="./configs/param_3pcf_multipole_lmax14.yaml")
   cfg = params["Corr_3PCF_Multipole"]
   cfg["l_max"] = 20
   cfg["sampling"]["r13"] = {"start": 40.0, "stop": 140.0, "step": 5.0}

   task = Corr_3PCF_Multipole(params)
   data = task.run()

Simple runtime choices such as ``threads`` and ``products`` are synchronized by
the task, but constructing a new task from the final dictionary is clearer and
less error-prone for a parameter scan.

Where to begin
--------------

The most reliable templates are the tracked files in ``examples/configs/``.
Read them beside the matching notebook and task guide; use this section when a
field needs a precise definition.
