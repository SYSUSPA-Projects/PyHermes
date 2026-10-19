Two-point correlation functions
===============================

``Corr_2PCF`` evaluates products of filtered data and random fields over a
sampled family of separation-binning windows. The same estimator handles
isotropic, anisotropic, auto-, and cross-correlations.

The :doc:`complete 2PCF notebook </notebooks/corr2pcf>` connects the task
interface to low-level field--window operations and includes isotropic and
redshift-space visualisations.

Isotropic 2PCF
--------------

The smallest configuration uses thin spherical shells:

.. literalinclude:: ../../../examples/configs/param_2pcf.yaml
   :language: yaml

The tracked configuration opts into the automatic spectral path. Omit the four
spectral/reuse settings to retain the convolution defaults. Install the same
source checkout before using its configurations; see :doc:`/install`.

``random: uniform`` uses the analytic constant reference density. Supply an
``SFCField`` or path instead when the random catalogue is spatially varying.

Run and load:

.. code-block:: bash

   cd examples
   python scripts/run_2pcf.py configs/param_2pcf.yaml

.. code-block:: python

   from pyhermes.io import Corr2PCFData
   from pyhermes.param.parambase import read_param

   params = read_param(config_path="./configs/param_2pcf.yaml")
   corr = Corr2PCFData(data_path=params["Corr_2PCF"]["fout_path"])
   print(corr.sampling_names)  # ('s',)
   print(corr.s, corr.xi)

Choosing optional spectral execution
-------------------------------------

Use ``pair_reduction: spectral``, ``reuse_input_fields: true``,
``spectral_window: auto`` and ``spectral_fft: shared`` with the default
``memory_strategy: speed``. This is one configuration for supported built-in
isotropic, grid-axis axial and Cartesian windows, rather than a separate
public switch for every window. Explicit ``shell_edges_radial`` remains
available for built-in finite shells; ordinary workloads can start with
``auto``. Unsupported windows retain their applicable fallback.

Parseval reduction avoids inverse FFTs on the supported path. Every original
alias keeps its own scaling weight and folding multiplicity; reordered
floating-point sums need not be bitwise identical. Shared FFT preparation
computes spectra on root and broadcasts real pair powers. Caches last for one
run, and caller-owned fields must not be mutated concurrently. The ``memory``
strategy retains its original execution path and does not activate the spectral
or input-reuse settings.

The main loop includes window planning, tables, reductions and communication.
Initial FFT/power setup is outside that interval but inside task total. Compare
matching total-time boundaries when assessing application performance. See
:doc:`optimization` for defaults, formulas, resource guards, fallback eligibility
and the controlled measurements.

Direct evaluation of prepared fields
------------------------------------

For a local calculation on existing real ``SFCField`` objects, use
``first.pair_with(second)`` and ``pair.evaluate(windows)``. Omitting the second
field gives an autocorrelation. The returned array contains windowed pair
means in input order; apply the estimator normalization explicitly. This shares
the same spectral and automatic-window backends as the optimized task, while
``Corr_2PCF`` continues to handle input preparation, MPI sampling and saved
result products. See :ref:`corr2pcf-local-pair-evaluation` for a full example.

Redshift-space anisotropy
-------------------------

A ``ring`` bin maps each :math:`(s,\mu)` sample to transverse radius
:math:`R=s\sqrt{1-\mu^2}` and line-of-sight offset :math:`H=s\mu`:

.. literalinclude:: ../../../examples/configs/param_2pcf_smu.yaml
   :language: yaml

After preparing the RSD field as described in the projection tutorial, run
from ``examples/``:

.. code-block:: bash

   python scripts/run_2pcf.py configs/param_2pcf_smu.yaml

The result has ``xi.shape == (len(s), len(mu))``. The plotting helper converts
it to the :math:`(s_\perp,s_\parallel)` plane:

.. code-block:: python

   from pyhermes.io import Corr2PCFData
   from pyhermes.utils.plot import plot_corr2pcf_2d
   from pyhermes.param.parambase import read_param

   params_smu = read_param(config_path="./configs/param_2pcf_smu.yaml")
   corr_smu = Corr2PCFData(data_path=params_smu["Corr_2PCF"]["fout_path"])
   plot_corr2pcf_2d(corr_smu)

.. figure:: ../../_static/paper/corr2pcf_ring_gaussian_ring_smu.png
   :width: 92%
   :align: center

   The same redshift-space estimator with a thin ring and a transversely
   Gaussian-blurred ring. The window changes; the field product does not.

Smoothing and binning are separate
----------------------------------

Use ``window`` to smooth both vertices, or ``window1`` and ``window2`` for
independent filters. Use ``binning_window`` for the pair separation:

.. code-block:: yaml

   Corr_2PCF:
      sfc_field: ./output/quijote8000_snap004_sfc.pkl
      random: uniform
      window:
         type: gaussian
         len_args: {R: 5.0}
      binning_window:
         type: thick_shell
         len_args:
            R: null
            delta_R: 6.0
         other_args: {}
         mapping: s_to_R
      sampling:
         s: {min: 10.0, max: 150.0, n: 29}
      products: xi

The named mapping fills ``R`` from ``s`` and preserves the fixed
``delta_R``. For a non-standard parameter mapping, provide a Python callable;
2PCF mappings are either one of the named mappings or a callable, not a YAML
dictionary. See :doc:`../../windows` for all built-in geometries.

Result products
---------------

``products`` may request:

``dd``
   Data--data field product.

``dr`` and ``rd``
   Ordered data--random cross-products.

``delta_dd``
   Product of :math:`\Delta=D-R` fields.

``rr``
   Random--random normalisation.

``xi``
   ``delta_dd / rr``. Dependencies are computed automatically.

Only requested products and their dependencies are retained in the result.

Cross-correlations
------------------

Use independent vertices for a cross-correlation:

.. code-block:: python

   from pyhermes.theory.corr2pcf import Corr_2PCF

   task = Corr_2PCF()
   task.sfc_field1 = halo_field
   task.sfc_field2 = mass_field
   task.random1 = "uniform"
   task.random2 = "uniform"
   task.window1 = {"type": "gaussian", "len_args": {"R": 5.0}}
   task.window2 = {"type": "gaussian", "len_args": {"R": 10.0}}
   task.binning_window = {
       "type": "shell",
       "len_args": {"R": None},
       "other_args": {},
       "mapping": "s_to_R",
   }
   task.sampling = {"s": [20.0, 40.0, 60.0, 80.0]}
   task.products = ["xi"]
   cross = task.run(save_result=False)

All field vertices must share the same MRA geometry and basis. Task-level
``weight_normalization`` converts compatible catalogue fields to one common
normalisation before products are formed.

Python overrides
----------------

.. code-block:: python

   import numpy as np
   import copy

   from pyhermes.param.parambase import read_param
   from pyhermes.theory.corr2pcf import Corr_2PCF

   params = read_param(config_path="./configs/param_2pcf.yaml")

   def s_to_gaussian_shell(sample, template):
       mapped = copy.deepcopy(template)
       mapped["len_args"]["R_shell"] = sample["s"]
       return mapped

   task = Corr_2PCF(params)
   task.sampling = {"s": np.linspace(0.0, 180.0, 46)}
   task.binning_window = {
       "type": "gaussian_shell",
       "len_args": {"R_smooth": 5.0},
       "mapping": s_to_gaussian_shell,
   }
   task.products = ["xi"]
   result = task.run(save_result=False)

Resolution and bin width
------------------------

The smallest trustworthy scale is controlled jointly by field resolution and
the binning window. A narrow bin does not recover structure unresolved by
``J``; a broad bin can intentionally average small-scale variation.

The direct periodic pair-counting comparison and its :math:`J` convergence are
shown on :doc:`../../benchmark`.  Residual differences are concentrated where
the separation approaches the field resolution.

Performance controls
--------------------

Sampling points are distributed across MPI ranks. The convolution path builds
and applies a window for each bin and required product. Optional spectral
execution reuses input spectra and eligible window plans across samples;
its cost depends on the window structure and complete setup/compute workload.

``memory_strategy: speed`` keeps reusable fields in memory. ``memory`` reloads
or rebuilds more aggressively to reduce peak storage. ``binning_window_cache``
can persist expensive kernels across repeated runs; set an explicit
``binning_window_cache_dir`` for production jobs.

Public tutorial
---------------

The :doc:`rendered 2PCF notebook </notebooks/corr2pcf>` contains high- and
low-level isotropic reconstruction and anisotropic window composition. It
computes the full :math:`46\times51` ring and Gaussian-ring curves directly
in one local process with eight threads, using the projected SFC fields as
inputs and plotting the returned results without precomputed correlation
files. ``examples/scripts/run_2pcf.py`` remains the production entry point.
