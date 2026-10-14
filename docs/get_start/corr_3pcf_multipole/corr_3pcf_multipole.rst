3PCF multipoles
===============

``Corr_3PCF_Multipole`` measures rotationally invariant Legendre multipoles of
the three-point correlation function. It replaces explicit orientation
sampling by multipole-filtered fields, then contracts those fields into
:math:`\zeta_\ell`. The estimator uses the same ``SFCField`` and
``WindowFunc`` infrastructure as the rest of PyHermes; only the radial and
angular window roles change.

The multipole sections of the
:doc:`complete 3PCF notebook </notebooks/corr3pcf>` load saved runs, compare
resolutions and :math:`\ell_{\max}`, and visualise split triangle
configurations.

Summation method and backend
-----------------------------

The task accepts two methods through the same input and result interface:

.. list-table:: Available combinations
   :header-rows: 1

   * - Calculation
     - ``summation_method``
     - ``summation_backend``
   * - Grid CPU
     - ``gamma``
     - ``cpu``
   * - Grid CPU + GPU
     - ``gamma``
     - ``gpu``
   * - Particle CPU (experimental)
     - ``particle``
     - ``cpu``

Defaults remain ``gamma`` and ``gpu`` for compatibility. Set both method and
backend explicitly when choosing particle summation. CPU FFTs construct the
harmonic arms in all combinations; only Gamma contraction can use CUDA.

.. important::

   Version 1.2 corrects the Gamma triple-overlap table. Recompute affected
   Gamma measurements from 1.1; the original SFC catalogue projections can be
   reused. CPU/GPU agreement in the old implementation did not establish that
   the common overlap table was correct. See :doc:`../../release_notes`.

Particle input and a runnable example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Particle summation reconstructs each harmonic arm at the first vertex's
catalogue positions and sums their product with its effective projection
weights. It requires a ``catalog_field`` with a retrievable particle catalogue.
During ``SFCProjection``, use ``save_particle_data: true`` and optionally
``particle_data_path`` to retain the companion ``pos``, ``catalog_weight``,
and ``field_value`` arrays. Keep that file accessible when moving the SFC
field; its path is stored in the metadata. An accessible original reader input
can also supply the catalogue. A derived field cannot replace the centre.

The first vertex must be unfiltered: omit ``window`` and ``window1``.
``window2`` and ``window3`` support endpoint smoothing as usual. A three-vertex
smoothing request therefore requires the Gamma method. Particle CUDA is not
implemented. Unsupported combinations raise an error.

The CPU-only synthetic example needs no external data:

.. code-block:: bash

   python examples/scripts/demo_3pcf_particle.py --output-dir ./examples/output/particle_demo

It projects 512 synthetic points at low resolution, saves the particle
companion and SFC field, computes three multipoles, and writes both the native
PKL and a numeric NPZ. It demonstrates the interface, not cosmological accuracy.

For the public Quijote field prepared by the Quick Start, run:

.. code-block:: bash

   cd examples
   python scripts/run_3pcf_multipole.py configs/param_3pcf_multipole_particle.yaml

The configuration uses ``summation_method: particle``, ``summation_backend:
cpu``, ``execution_mode: serial``, :math:`\ell=0,\ldots,7`, thin shells at
:math:`(20,40)\,h^{-1}\mathrm{Mpc}`, and endpoint sphere smoothing at
:math:`R=5\,h^{-1}\mathrm{Mpc}`. The driver also accepts explicit method,
backend, execution-mode, and thread overrides. Existing files are not silently
overwritten; ``--overwrite`` opts into replacement.

Particle subtraction preserves the normal product rules. Uniform random
centres use a two-field orthonormal integral, and explicit random catalogues
use their own particle readout. Catalogue weights, marks, and normalization
follow the saved field and task settings; do not normalize the returned
``zeta_l`` again.

At finite J, the methods differ: particle summation evaluates
:math:`A_J(\mathbf{x}_i)B_J(\mathbf{x}_i)`, while the Gamma centre integral
corresponds to :math:`P_J[A_JB_J](\mathbf{x}_i)`. They share a continuum target
but need not agree coefficient by coefficient at fixed J. Check convergence
and independent references rather than assuming either is exact.

From shell windows to radial profiles
-------------------------------------

For an infinitesimally thin shell of radius :math:`R`, the Fourier-space
harmonic window has the form

.. math::

   \widehat W_R^{\ell m}(\mathbf{k})
   =4\pi i^\ell j_\ell(2\pi kR)Y_\ell^m(\widehat{\mathbf{k}}),

where PyHermes uses Fourier phases :math:`e^{2\pi i\mathbf{k}\cdot\mathbf{x}}`.
For a general normalized isotropic radial profile :math:`w_R(r)`, the spherical
Bessel factor is replaced by

.. math::

   U_\ell(k)=4\pi\int_0^\infty r^2 w_R(r)
   j_\ell(2\pi kr)\,dr,
   \qquad
   \widehat W_R^{\ell m}\propto U_\ell(k)Y_\ell^m(\widehat{\mathbf{k}}).

This ``radial profile x spherical harmonic`` factorisation lets the same
multipole task use thin shells, finite-width shells, Gaussian shells, or other
isotropic binning profiles without changing the angular contraction.

.. list-table:: Radial profiles available to the multipole task
   :header-rows: 1
   :widths: 22 32 46

   * - Type
     - Length arguments
     - Numerical path
   * - ``shell``
     - ``R``
     - Analytic :math:`j_\ell(2\pi kR)` for every :math:`\ell`.
   * - ``thick_shell``
     - ``R``, ``delta_R``
     - Exact monopole transfer; higher orders from its analytic real-space
       profile and a one-dimensional Hankel table.
   * - ``sphere``
     - ``R``
     - Exact monopole transfer; higher orders from the top-hat real-space
       profile.
   * - ``gaussian``
     - ``R``
     - Exact monopole transfer; higher orders from the Gaussian real-space
       profile.
   * - ``gaussian_shell``
     - ``R_shell``, ``R_smooth``
     - Exact monopole transfer and a general inverse-Hankel route for higher
       orders.
   * - ``custom_real``, ``custom_kspace``
     - User-defined profile arguments
     - Tabulated real-space Hankel or inverse-Hankel route; see
       :doc:`../../windows`.

The task records radial-profile diagnostics and can warn when the tabulated
profile does not reproduce its monopole transfer within the configured
tolerance.

A fixed triangle
----------------

Radii are not top-level task parameters. They are sampled values mapped into
the two edge windows:

.. code-block:: yaml

   Corr_3PCF_Multipole:
      sfc_field: "./output/quijote8000_snap004_sfc.pkl"
      random: "uniform"
      window2:
         type: "sphere"
         len_args: {R: 5.0}
      window3:
         type: "sphere"
         len_args: {R: 5.0}
      binning_window12:
         type: "shell"
         len_args: {}
         other_args: {}
         mapping:
            R: "r12"
      binning_window13:
         type: "shell"
         len_args: {}
         other_args: {}
         mapping:
            R: "r13"
      sampling:
         r12: 20.0
         r13: 40.0
      l_max: 14
      gpu_device_id: 0
      execution_mode: "pair_mpi"
      products: "zeta_l"
      threads: 4
      fout_path: "./output/quijote8000_snap004_3pcf_multipole_lmax14.pkl"

``mapping`` assigns each sampled name to a ``len_args`` key. A target can also
be written explicitly as ``len_args.R`` or ``other_args.some_name``. Fixed
values remain in the template, so only parameters that actually vary need to
appear in ``sampling``. This block matches the tracked
``param_3pcf_multipole_lmax14.yaml``: ``l_min: 0`` and
``summation_backend: "gpu"`` are inherited from the task defaults.

Scanning radial configurations
------------------------------

``sampling.mode: grid`` forms the Cartesian product of all sampled arrays.
``paired`` zips arrays of the same length and broadcasts scalar entries. For
example, this scans :math:`r_{13}` while holding :math:`r_{12}` and both shell
widths fixed:

.. code-block:: yaml

   binning_window12:
      type: "thick_shell"
      len_args: {}
      mapping:
         R: "r12"
         delta_R: "delta_r12"
   binning_window13:
      type: "thick_shell"
      len_args: {}
      mapping:
         R: "r13"
         delta_R: "delta_r13"
   sampling:
      mode: "grid"
      r12: 20.0
      delta_r12: 6.0
      r13:
         start: 40.0
         stop: 140.0
         step: 5.0
      delta_r13: 6.0
   l_min: 0
   l_max: 0

The result has one row per expanded sample and one column per multipole order.
The original sampled values are stored in ``sample_params`` so rows remain
self-describing.

.. figure:: ../../_static/paper/Fig_3PCF_monopole_shell_thickness.png
   :width: 90%
   :align: center
   :alt: 3PCF monopole for thin and finite-width radial shells

   A broader radial bin averages neighbouring triangle configurations and
   suppresses small-scale fluctuations without changing the task structure.

Parallel execution
------------------

The task separates two kinds of parallel work:

``serial``
   One rank evaluates all samples and multipole modes. Use this for a small
   local check.

``pair_mpi``
   MPI-rank pairs share the :math:`(\ell,m)` convolution work for one sample.
   It requires an even number of ranks and is well suited to one or a few
   configurations with many multipoles.

``sample_mpi``
   Samples are assigned to ranks by static round robin. It is the natural mode
   for a long monopole or radial-window scan. The current implementation uses
   ``sample_mpi.ranks_per_sample: 1``; choose approximately one rank per
   simultaneous sample, then use ``threads`` for the local FFT and kernel work.

In ``sample_mpi``, rank 0 reads each file-backed ``SFCField`` once and
broadcasts it. This avoids every rank independently stressing shared storage.

CPU and GPU backends
--------------------

``summation_backend`` controls the final contraction and grid sum:

- ``cpu`` uses the multithreaded CPU kernel.
- ``gpu`` transfers the filtered fields to the selected CUDA device and uses
  ``gpu_threads_per_block`` for the final contraction.

The preceding FFT convolution stage remains CPU/MPI work in both cases. Thus a
GPU changes the contraction backend, not the estimator, radial windows, or
input fields. At high :math:`\ell_{\max}` the GPU path is usually most useful;
for a short monopole scan, sample-level CPU parallelism may be simpler and
competitive.

Corrected CPU/GPU agreement and the three-repeat runtime/host-memory
comparison are documented on :doc:`../../benchmark`. The production CUDA
contraction uses a flattened one-dimensional block: ``[8,8,8]`` sets its
product to 512 threads, not a three-dimensional launch.

Gamma CPU contexts reuse exactly nonzero signed overlap rows and work arrays.
Convolution contexts retain a source FFT across modes, so a source coefficient
array must remain immutable until its context is discarded. A complex128
spectrum costs 256 MiB at J=8, or 2 GiB at J=9, per source context. This
optimization applies to both Gamma and particle harmonic arms.

Products and output
-------------------

Available products are ``ddd_l``, ``rrr_l``, ``delta_ddd_l``, and ``zeta_l``.
Requesting ``zeta_l`` expands the connected-data and random dependencies
automatically. A fully uniform random field uses an analytic monopole-only
``rrr_l`` shortcut.

.. code-block:: python

   from pyhermes.io import Corr3PCFMultipoleData

   data = Corr3PCFMultipoleData(data_path="./output/threepcf_multipoles.pkl")
   print(data.sample_params)  # sampled names -> one-dimensional arrays
   print(data.l)              # [l_min, ..., l_max]
   print(data.zeta_l.shape)   # (n_samples, n_l)

Saved ``zeta_l`` contains the coefficients of
:math:`\zeta(\mu)=\sum_\ell\zeta_\ell P_\ell(\mu)`; do not multiply them by
``2*ell+1`` when reconstructing the angular curve:

.. code-block:: python

   import numpy as np
   from numpy.polynomial.legendre import legval

   mu = np.linspace(-1.0, 1.0, 201)
   zeta_mu = legval(mu, data.zeta_l[0])  # for l_min=0

The J value belongs to the input projection; it is not a multipole task radius
or resolution parameter. Metadata records the method, backend, input basis,
windows, and sampling. ``zeta_l`` has shape ``(n_samples, n_l)``.

``zeta_condition`` records the conditioning of the random-multipole solve.
Inspect unusually large values before interpreting a noisy high-order result.

.. figure:: ../../_static/paper/corr3pcf_multipole_lmax20_split_overview.png
   :width: 94%
   :align: center
   :alt: 3PCF multipoles for smoothing and triangle configurations

   Current ``J=8``, ``lmax=20`` grouped results.  Left: fixed
   :math:`(r_{12},r_{13})=(20,40)\,h^{-1}\mathrm{Mpc}` with top-hat and
   Gaussian vertex smoothing at matched radii.  Right: fixed top-hat
   :math:`R=5\,h^{-1}\mathrm{Mpc}` with six triangle side-length pairs.
   Multipoles compactly expose how smoothing and triangle geometry redistribute
   angular information.
