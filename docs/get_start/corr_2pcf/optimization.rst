2PCF spectral computation
=========================

The optional spectral path evaluates the same field products and folded
windows used by :doc:`corr_2pcf`. Automatic strategies cover built-in radial and grid-axis separable windows.
The explicit finite-shell path also remains available. Original estimator normalizations, bin edges, scaling
weights and folding multiplicities are retained.

Enable the finite-shell path
----------------------------

Add these settings to an existing ``Corr_2PCF`` configuration with its chosen
input fields, products and sampling:

.. code-block:: yaml

   Corr_2PCF:
      memory_strategy: speed
      pair_reduction: spectral
      reuse_input_fields: true
      spectral_window: shell_edges_radial
      spectral_fft: shared
      binning_window:
         type: thick_shell
         len_args: {R: null, delta_R: 5.0}
         mapping: s_to_R

This example chooses a width of 5 in the field's length units. Preserve the
scientific window parameters of your own calculation. Explicit boundaries use
``binned_shell`` with ``R_in`` and ``R_out``; see :doc:`/windows`.

Defaults remain ``pair_reduction: convolution``, ``reuse_input_fields: false``,
``spectral_window: materialized`` and ``spectral_fft: per_rank``. Enabling one
switch does not implicitly enable the others.

Automatic window strategies
---------------------------

Use the same four options above with ``spectral_window: auto`` to choose a
strategy by mathematical structure instead of selecting a backend per window:

- Built-in radial windows (``shell``, ``gaussian_shell``, ``sphere``,
  ``gaussian``, ``cw``, ``cws``, ``gdw``, ``laplacian`` and
  ``inverse_laplacian``) share exact squared-frequency grouping. The original
  function is evaluated on every occupied class. Built-in ``thick_shell`` and
  ``binned_shell`` retain adjacent ball-boundary sharing.
- ``ring``, ``disk``, ``cylshell`` and ``cylinder`` with a line of sight exactly
  on any grid axis share bounded transverse/longitudinal factor tables. The
  original special functions, scaling weights and fold are retained. An almost
  aligned line of sight is not rounded onto an axis by this strategy.
- Other real octant windows use generic fused scalar reduction; unsupported,
  complex, disk-cached and composite windows retain their existing paths.

The radial planner is collective and is used when all queries are radial,
there are at least as many samples as MPI ranks (and at least two samples), and
scratch fits the 256 MiB per-rank guard. Otherwise it uses fused reduction;
finite-shell scratch overflow retains the existing batch-edge fallback. This
is a resource guard, not an assertion that automatic execution is always
fastest. One field pair is grouped/evaluated at a time to bound scratch memory.
The inverse Laplacian removes the entire folded DC coefficient, including its
nonzero aliases, as in the materialized implementation.

Separable factor tables use at most 16 MiB of persistent LRU cache per rank and
at most 64 MiB for the table pair currently being evaluated. Oversized tables
fall back to fused reduction. These limits exclude input fields, pair powers,
Numba compilation/runtime memory and the rest of the task. Caches last only
for one run. Kernels taking dynamic window-function arguments compile once per
process instead of persisting dispatcher weak references in disk-cache indices;
other eligible kernels retain disk caches. Record startup/prewarm time separately.
Planning, table construction and all associated collectives count
inside the reported main loop, including work for the first sample.

An explicit custom callable remains authoritative even if ``type`` names a
built-in window. It uses generic evaluation by default. Advanced authors may
set ``structure: radial`` in its window dictionary to promise a real radial
function under the existing octant contract; incorrect declarations change
the meaning of the calculation and are the author's responsibility. No
structure is inferred from the callable name. Existing configurations need
no declaration.

Compatibility with existing calls
---------------------------------

The existing ``Corr_2PCF(param_task=...)`` constructor,
``prepare_input_fields(...)`` and ``run(save_result=True, overwrite=False)``
entry points are unchanged. Existing YAML/Python configurations continue to
use convolution unless the new options are set explicitly. The result remains
``Corr2PCFData`` with the same sampling arrays, requested product arrays and
shapes. Additional runtime metrics are diagnostic metadata.

The four added optional settings are ``pair_reduction``,
``reuse_input_fields``, ``spectral_window`` and ``spectral_fft``. The new
built-in ``binned_shell`` accepts explicit inner/outer edges; existing
``thick_shell`` configurations remain supported, and custom callables remain
authoritative. Estimator and weight-normalization conventions are unchanged.

Parseval reduction
------------------

For real input fields and a real folded window response, the required spatial
average can be evaluated directly in Fourier space. Let :math:`N` be the total
number of grid cells, and :math:`\widehat K` the complete folded window response.
With the default unnormalized forward FFT and :math:`1/N` inverse FFT:

.. math::

   C_{fg} = \frac{1}{N}\sum_{\boldsymbol x}
       f(\boldsymbol x)(K*g)(\boldsymbol x)
     = \frac{1}{N^2}\sum_{\boldsymbol k}
       \operatorname{Re}\left[\widehat f(\boldsymbol k)^*
                              \widehat g(\boldsymbol k)\right]
       \widehat K(\boldsymbol k).

The real-FFT implementation gives the final-axis zero-frequency plane and,
for even grid lengths, the Nyquist plane weight one; other retained final-axis
frequencies have weight two. This preserves the full-spectrum sum. Existing
physical and estimator normalization is applied by the task as before.

Each distinct real input needs its FFT only once per run. The supported path
needs no inverse FFT and no full convolution-result field. ``shared`` computes
spectra and real pair powers on rank zero and broadcasts the powers after the
task has replicated the inputs. It is not zero-copy shared memory: each rank
still holds its real power arrays. Multiple inputs or requested products can
require more than one distinct FFT.

Window work and exact radial classes
------------------------------------

The optimization also reduces work around the spectral sum:

- Identical input vertices share normalization preparation, broadcasts and
  data-minus-random fields when their normalization and vertex windows agree.
- Fused scalar reduction avoids constructing full three-dimensional octant and
  folded-window arrays. Identical one-dimensional scaling basis arrays are
  reused within the run.
- Adjacent spherical shells share ball-boundary integrals. For 31 contiguous
  bins with 32 distinct edges, this requires 32 ball integrals per distinct
  pair power instead of 62 independently evaluated inner/outer balls.
- The batch path evaluates several radii in one traversal. Its benefit depends
  on resolution and layout; it remains available as an alternative and a
  radial-memory fallback.
- The radial path first applies each original term's scaling and transposed-fold
  power weights, then groups terms with exactly equal integer squared norm
  :math:`m=a^2+b^2+c^2`. Ball windows are evaluated on these radial classes.

Writing :math:`A_{abc}` for the original alias-specific scaling weight and
:math:`P^{\rm fold}_{abc}` for the transposed-fold power weight:

.. math::

   H_m = \sum_{a^2+b^2+c^2=m} P^{\rm fold}_{abc}\,A_{abc},
   \qquad
   C_{\rm ball}(R) = \frac{1}{N^2}\sum_m H_m\,
       W_{\rm ball}\left(\frac{\sqrt m}{L},R\right).

Here :math:`L` is the grid side length and the ball function uses the same
cycle-frequency convention, including its internal :math:`2\pi` factor, as the
original implementation. No scaling factor is extracted across different
aliases. Integer classes introduce no radial interpolation, tolerance-based
merging, frequency cutoff or change to folding multiplicities. Reordered
floating-point sums need not be bitwise identical.

Options and fallbacks
---------------------

.. list-table::
   :header-rows: 1
   :widths: 24 19 57

   * - Setting
     - Default
     - Behavior
   * - ``pair_reduction``
     - ``convolution``
     - ``spectral`` enables the real-power scalar reduction; complex or
       unsupported kernels retain convolution.
   * - ``reuse_input_fields``
     - ``false``
     - Reuse compatible input preparation within one task execution.
   * - ``spectral_fft``
     - ``per_rank``
     - ``shared`` computes distinct input spectra on root and broadcasts powers.
   * - ``spectral_window``
     - ``materialized``
     - Alternatives: ``fused``, ``shell_edges``, ``shell_edges_batch`` and
       ``shell_edges_radial``; ``auto`` selects a structural strategy. Shared-edge
       paths require built-in finite shells
       to activate their shell-specific optimization.
   * - ``memory_strategy``
     - ``speed``
     - The optional spectral execution uses ``speed``. ``memory`` retains its
       original calculation strategy.

A custom callable remains authoritative; a name matching a built-in shell does
not make it eligible for built-in boundary sharing. Directional/complex,
materialized/composite and disk-cached windows keep the applicable existing
paths. No new approximation is substituted for an unsupported window.

Radial grouping uses thread-private temporary histograms and a collective sum.
A conservative 256 MiB per-rank scratch estimate triggers a collective fallback
to batch edges. This limit does not include all input fields, power arrays or
other task memory. The J9 test estimate was about 60 MiB/rank with four threads
or 84 MiB/rank with eight threads. All power, basis, radial and scalar caches
are released after each run. Caller-owned fields must not be modified
concurrently with a task.

Timing and controlled measurements
-----------------------------------

``runtime_metrics`` distinguishes initial spectral setup and the existing
rank-zero main-loop interval. Radial grouping, allocations, histogram MPI
reductions, boundary evaluation and result assembly are inside the main loop.
Initial FFT/power preparation and power broadcasts are outside that interval
but inside setup and total time. Use
``compute_including_spectral_setup_seconds`` to include initial spectra in the
correlation computation, and separately measure projection-to-result time for
application performance. Per-rank maximum cumulative durations can overlap
and must not be added to reconstruct total wall time.

The 2026-10-16 controlled finite-shell test used 406,728 Quijote halos in a
periodic box of side 1000, unit weights, and 31 bins with
``edges = linspace(5, 150, 32)``. Each process rebuilt its full-resolution field
and powers. Imports, small-grid JIT warmup, public input/output IO and diagnostic
hashing were outside the measured interval; Numba disk and OS caches were
allowed. All slow samples were retained.

Each 32-core layout was measured three times. The table takes the minimum per
layout, then the smaller of 8 ranks x 4 threads and 4 ranks x 8 threads,
independently for each timing metric. All rows below selected 8 x 4:

.. list-table:: Complete 31-bin curve, seconds
   :header-rows: 1
   :widths: 20 40 40

   * - Resolution
     - Projection-to-result total
     - Main loop
   * - J=7
     - 0.169722
     - 0.029206
   * - J=8
     - 0.532926
     - 0.070165
   * - J=9
     - 2.687215
     - 0.292769

Within-allocation comparisons used medians to select methods, giving total-time
reductions of 41--69% across the tested J/layout combinations. The J7/J8 change
also enabled shared FFT; the J9 comparison already used shared FFT on both
sides. These are combined-path gains for this workload. Different J values
represent different finite-resolution approximations, not equal-accuracy
measurements.

The historical pycorr/Corrfunc and TreeCorr full-curve times were 0.373190 and
1.813970 seconds on 32 cores. They were reused without retiming in this
campaign and were not split into a matching PyHermes main-loop interval.

All 36 production curves and six gates passed checks against the original
implementation with ``rtol=1e-8, atol=1e-11``; maximum absolute xi difference
was about :math:`1.2\times10^{-12}`. Input/output/source hashes and physical-core
binding were checked. Process-tree RSS was sampled every 0.1 seconds and can
count shared pages more than once; whole-job Slurm MaxRSS is a separate value.

The :download:`detailed measurement report
<../../2pcf_radial_optimization_results_20261016.md>` retains the complete
validation and timing context. No absolute-accuracy or universal-speed claim
is inferred from this comparison.
