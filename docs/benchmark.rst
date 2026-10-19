Validation and Performance
==========================

This page separates two questions that are easy to mix together:

* **Validation:** does a field--window estimator recover the expected statistic,
  and does it converge when the numerical controls are tightened?
* **Performance:** once the estimator is fixed, which part of the workflow sets
  the runtime and memory footprint?

The performance overview below uses the verified optimized 2PCF and pure
angular DDD measurements, with main-loop and preprojected-field task totals
kept separate. Detailed 2PCF window-family and angular DDD/Q results are in
their task guides. The multipole section compares grid CPU, grid CPU + GPU
and particle CPU with three measurements per configuration.

Reference Data
--------------

Unless stated otherwise, the validation examples use the Quijote
fiducial-cosmology halo catalogue from realisation 8000, snapshot 004
(``z = 0``):

* 406,728 FoF haloes in a periodic cube of side
  ``1000 h^-1 Mpc``;
* the ``db2`` scaling-function basis;
* ``J = 8`` for the main tests, with selected ``J = 9`` comparisons;
* the plane-parallel approximation for redshift-space measurements.

These are controlled algorithm tests, not a complete survey-analysis recipe.
Survey masks and spatially varying selection functions require an explicit
random catalogue or random ``SFCField``.

Numerical Validation
--------------------

Anisotropic 2PCF
~~~~~~~~~~~~~~~~

The same ``Corr2PCF`` task can measure real- and redshift-space fields by
changing the input ``SFCField`` while keeping the binning-window family fixed.
The plot below maps :math:`s^2\xi(s,\mu)` to the
:math:`(s_\perp,s_\parallel)` plane.  Joining two measurements at
:math:`s_\perp=0` makes the line-of-sight response immediately visible and
also provides a compact check of tracer weighting.

.. figure:: _static/results/docs_2pcf_real_rsd_smu.png
   :alt: Real-space, redshift-space, and mass-weighted anisotropic two-point correlation functions
   :width: 100%

   Current grouped-test outputs.  Left: real space and unit-weight redshift
   space.  Right: unit- and mass-weighted redshift space.  The two halves of
   each circle share a colour normalization, while the two panels have
   separate colour bars because mass weighting has a larger dynamic range.
   The white centre excludes :math:`s<20\,h^{-1}{\rm Mpc}`.

For an independent estimator-level check, the paper compares the isotropic
Hermes result with direct periodic pair counting.  Agreement improves with
increasing ``J`` and is best interpreted only above the effective resolution
scale of the reconstructed field.

.. figure:: _static/paper/benchmark_2pcf_isotropic_s2xi_curves.png
   :alt: Isotropic PyHermes and direct pair-counting comparison
   :width: 86%

Standard 3PCF
~~~~~~~~~~~~~

The conventional ``Corr3PCF`` task estimates translational and rotational
averages by Monte Carlo sampling.  Convergence must therefore be checked in
``n_rot`` rather than inferred from one visually smooth curve.  In the current
test, the low-rotation result fluctuates visibly while the curves and their RMS
difference settle rapidly as ``n_rot`` increases.

.. figure:: _static/results/docs_3pcf_rotation_convergence.png
   :alt: Standard three-point correlation rotation convergence
   :width: 100%

   Left: :math:`Q(\theta)` for several rotation counts.  Right: RMS difference
   from the ``n_rot = 2000`` result.

3PCF Multipoles
~~~~~~~~~~~~~~~

.. warning::

   The next two convergence plots predate the 1.2 Gamma correction and are
   retained as historical analysis examples, not validation of 1.2. Corrected
   paper consistency and smoothing figures are in the multipole task guide.


Multipole measurements have two separate convergence controls.  ``lmax``
truncates the angular expansion, while ``J`` controls the MRA field resolution.
The left panel below verifies that increasing ``lmax`` appends higher orders
without changing the already computed low-order multipoles.  The right panel
compares those common modes between ``J = 8`` and ``J = 9``.

.. figure:: _static/results/docs_3pcf_multipole_lmax_resolution.png
   :alt: Three-point multipole truncation and field-resolution comparison
   :width: 100%

For non-shell radial profiles, PyHermes tabulates the required radial transform.
The default and refined tables should agree before a profile is used for
production measurements.  The following check covers the thin-shell analytic
path and the numerical thick- and Gaussian-shell paths.

.. figure:: _static/results/docs_3pcf_radial_profile_convergence.png
   :alt: Multipole radial-profile convergence at J equals 7 and 8
   :width: 100%

An Ensemble-Level Check
~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   These historical Kun figures predate the Gamma correction. Recompute
   affected multipoles before interpreting amplitudes; they are not current
   validation. The documentation does not distribute collaborator catalogues
   or their new production labels.


The Kun scan provides a larger application test: ``r12 = 20 h^-1 Mpc`` is held
fixed while ``r13`` varies.  The 129 mocks used here span different cosmological
parameters.  Their spread is therefore **not** a covariance estimate for one
cosmology; it is shown to inspect the stability of the full projection and
monopole pipeline over a deliberately broad simulation ensemble.

.. figure:: _static/results/docs_kun_monopole_ensemble.png
   :alt: Thin- and thick-shell three-point monopoles for 129 Kun mocks
   :width: 100%

The binning profile is part of the observable, not merely a plotting choice.
For one mock, widening a finite shell or increasing the Gaussian-shell width
averages a broader neighbourhood of triangle configurations:

.. figure:: _static/results/docs_kun_mock0_binning_windows.png
   :alt: Kun mock zero monopole measured with thin, thick, and Gaussian shells
   :width: 86%

   One input field and one radial scan measured with six binning profiles. The
   smooth variation between profiles is the expected finite-bin response.

Performance Model
-----------------

The catalogue is projected once onto
:math:`N_{\mathrm{MRA}}=2^{3J}` coefficients.  Gamma measurements operate on fields and windows, so their leading cost
depends on resolution and window evaluations. Particle summation additionally
scales with the number of centre particles and the readout stencil size.

For the convolution path with ``N_w`` window applications on a matching FFT grid,

.. math::

   T_{\mathrm{conv}}
   = \mathcal{O}\!\left(N_w N_{\mathrm{MRA}}
     \log N_{\mathrm{MRA}}\right).

For convolution-based ``Corr2PCF``, the sampled separation windows number
``N_s * N_mu`` or ``N_rp * N_pi``. Multipole calculations build harmonic arm
fields for :math:`N_m=(\ell_{\max}+1)(\ell_{\max}+2)/2` non-negative-m modes,
with additional factors for products, radii and endpoint arms.

Standard angular ``Corr3PCF`` instead prepares its filtered endpoint fields
before Monte Carlo sampling. Its sampled-triplet work scales with
``N_centres * N_rotations * N_angles`` and the interpolation stencil cost;
those combinations are not separate FFT/window applications. Pair
postprocessing for Q is an additional interval, absent from a pure DDD request.

The optional spectral 2PCF path instead prepares each distinct source FFT
once, then applies exact radial or separable reductions; it does not perform
an inverse FFT for every supported sample. Its cost depends on window
structure, occupied frequency classes and batching. The convolution expression
above is therefore not a per-bin FFT cost model for the optimized path. See
:doc:`get_start/corr_2pcf/optimization` for the reduction and timing definitions.

Increasing ``J`` by one multiplies the number of three-dimensional field
coefficients by eight.  Since some work buffers are rank-local, a high-``J``
job often benefits from fewer MPI ranks and more threads per rank.

Current 2PCF and angular DDD measurements
-----------------------------------------

All three rows use J=8 and unit-weight Quijote haloes in a periodic
1000 Mpc/h box. Times are independent minima of three fresh-process
measurements. The last timing column divides the unrounded main-loop minimum
by the number of samples or centre--rotation--angle combinations. Phase minima
must not be added to reconstruct total wall time.

.. csv-table:: Current loop and task times, with batch memory
   :file: _static/results/current_dd_ddd_20261017.csv
   :header-rows: 1
   :widths: 27 12 14 14 18 15

* **DD ring:** 46 separations and 51 mu values, or 2346 samples, on 32 distinct
  physical cores (8 ranks x 4 threads). This row retains the historical DD
  label for the anisotropic 2PCF workflow: the actual request is ``xi`` with
  analytic uniform-random normalization. It uses the original RSD benchmark
  field and the verified three-run cohort, job 127025.
* **Particle-centre DDD:** 406728 centres x 1000 rotations x 20 angles,
  or 8,134,560,000 combinations, on 128 distinct physical cores (16 x 8).
* **Box-random DDD:** 8000000 centres x 200 rotations x 20 angles,
  or 32,000,000,000 combinations, on the same 128-core layout. Both DDD rows
  are pure triplet requests from job 127098, using the original real-space
  field, r12=20/r13=40 Mpc/h, endpoint sphere-R5 windows and no centre smoothing.

Task total starts before reading an already projected field and includes input
IO, preparation and computation. It excludes catalogue projection, imports,
small-J compilation warmup and output IO/audits. The 2PCF main loop includes
window planning, tables, reductions and communication, but excludes initial
FFT/power setup. The angular DDD main loop includes per-product preparation,
broadcasts, sampled triplets and the final collective; initial field/centre
setup and pair postprocessing are outside it. Q additionally needs connected
products and lower-order pair postprocessing, and is not raw DDD.

Memory is Slurm **batch-step MaxRSS**, excluding GPU device memory. DD uses
the separate same-configuration resource job 127035: 4579592 KiB = 4.367 GiB.
Its one-run time is not pooled with the three-run timing cohort. Each DDD
memory value belongs to one centre/product array element, including its gates
and three fresh processes. These are not minima of three memory measurements,
per-rank RSS or process-tree sums.

The current DD and DDD measurements use different inputs, core counts and
allocations; they are not a controlled speedup comparison with each other or
with the historical paper table. That older table reported approximately
120/84/281 seconds. The archived particle example requested Q, and the
available box log does not substantiate the old 281-second value. Those numbers
must not be relabelled as current pure-DDD loop times or used to infer an
algorithm-only speedup.

The detailed task guides retain all repetitions, numerical checks and input
provenance: :doc:`2PCF windows <get_start/corr_2pcf/optimization>` and
:doc:`angular DDD/Q <get_start/corr_3pcf/corr_3pcf>`. Download the
:download:`overview CSV <_static/results/current_dd_ddd_20261017.csv>`,
:download:`2PCF source statistics <_static/results/twopcf_window_strategies_20261017.json>`
or :download:`DDD/Q source statistics <_static/results/angular_3pcf_catalogue_20261017.json>`.

The CPU nodes have two AMD EPYC 9754 processors (256 physical cores in
total); each measurement uses only its stated allocation. GPU measurements
use one NVIDIA GeForce RTX 4090 with CUDA 12.4.
``ranks x threads`` below describes the CPU layout; GPU jobs use the same
CPU-side convolution layout and an ``(8, 8, 8)`` CUDA block for the final
contraction.

Multipole performance comparison
--------------------------------

These measurements use J=8, lmax=7, thin shells at (20,40),
endpoint sphere R5, and an unfiltered first vertex. Each method/layout has
three samples; no slow sample is removed. Values below are medians with
minimum/maximum ranges. Kernel-level speedups are not substituted for total
wall time.

.. list-table:: PyHermes on 32 physical CPU cores
   :header-rows: 1

   * - Layout
     - Method
     - Total [s] (min--max)
     - Host RSS [GiB] (min--max)
   * - ``4x8``
     - grid CPU
     - 137.151 (136.273--137.222)
     - 6.144 (6.126--6.199)
   * - ``4x8``
     - grid CPU + GPU
     - 82.204 (80.089--85.062)
     - 5.143 (4.990--5.182)
   * - ``4x8``
     - particle CPU
     - 73.955 (72.053--75.441)
     - 5.226 (5.104--5.283)
   * - ``8x4``
     - grid CPU
     - 138.250 (137.756--143.143)
     - 10.955 (10.949--10.964)
   * - ``8x4``
     - grid CPU + GPU
     - 82.438 (82.243--82.524)
     - 8.708 (8.597--9.122)
   * - ``8x4``
     - particle CPU
     - 76.719 (75.787--77.550)
     - 9.262 (9.167--9.317)

.. figure:: _static/results/threepcf_v1_2_runtime_memory.png
   :alt: Grid CPU, grid CPU + GPU and particle CPU runtime and process-tree RSS
   :width: 100%

   Bars: median; black dots: all three runs; whiskers: min/max.
   The two layouts each use 32 physical cores. Host memory excludes GPU memory.

Total time is the sum of fresh preparation and measurement process wall times,
including initialization and output, and excludes queue time. Host RSS is a
0.5-second sampled process-tree sum; shared pages may be counted more than
once. It is not the Slurm MaxRSS used for the 2PCF/angular-DDD overview above.
GPU device samples reach 4.00--4.25 GiB and are reported separately.

One RTX4090 is an additional resource. Some CPU methods ran within batches
that retained a GPU allocation. Layouts and repeats received different CPU
IDs, NUMA ratios, and run times; differences cannot all be attributed to the
rank/thread split. Convolved-field caching was disabled; Numba disk compilation
could be reused, and OS caches were not cleared. GPU initialization and
compilation are included. These are hardware-specific observations, not a
global optimum or an equal-cost/equal-accuracy comparison.

CPU and GPU Gamma results pass allclose at rtol=1e-8, atol=1e-11. Independent
periodized-basis integrals test the overlap separately. Particle
summation is a different finite-J discretization, so its coefficient agreement
is not an absolute-accuracy measure.

All 18 numeric samples and result/source hashes are available in
:download:`the benchmark JSON <_static/results/threepcf_v1_2_benchmark.json>`.
The :doc:`3PCF notebook </notebooks/corr3pcf>` includes the plotting code.

Reproducing the Checks
----------------------

Run the particle task from the :doc:`3PCF notebook </notebooks/corr3pcf>`
or its YAML configuration to exercise catalogue readout and the task interface.
Numerical validation includes independent Gamma/particle integrals,
source-FFT reuse checks, and serial, pair-MPI and sample-MPI comparisons.
CUDA checks require a real device or an explicitly labelled simulator.

The benchmark JSON above is sufficient to replot the published timing/RSS
figure without access to a private filesystem. Full raw production logs and
large catalogue fields remain local. Do not sum overlapping profile phases
or infer absolute accuracy from backend agreement.
