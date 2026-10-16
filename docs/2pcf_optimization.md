# 2PCF spectral optimization

The validated implementation is integrated into local main and remains opt-in.
The [current user guide](get_start/corr_2pcf/optimization.rst) explains Parseval
reduction, exact radial classes, options, fallbacks, and measured timing.
For built-in finite spherical shells, the final tested configuration is
`pair_reduction="spectral"`, `reuse_input_fields=True`,
`spectral_window="shell_edges_radial"`, `spectral_fft="shared"`,
with `memory_strategy="speed"`.

The sections below retain the successive experimental stages and their
measurement conditions; intermediate J-dependent recommendations are historical.

The defaults retain the convolution implementation. To opt into the two independent experiments:

```python
options = {'Corr_2PCF': {
    # Other normal Corr_2PCF options ...
    'memory_strategy': 'speed',
    'pair_reduction': 'spectral',
    'reuse_input_fields': True,
}}
```

`reuse_input_fields` avoids unnecessary normalization copies, shares identical prepared data vertices and broadcasts, and shares their delta field when both backgrounds are the same. Different inputs or vertex windows remain independent. Caller-owned input values are unchanged. Fields must not be mutated concurrently with a task.

`pair_reduction='spectral'` prepares the needed input rFFTs and real cross powers once per run and MPI rank. Each bin uses the **existing folded window kernel** in a weighted half-spectrum scalar reduction. The DC plane and the even-grid Nyquist plane have weight one; other final-axis frequencies have weight two. For the default SciPy FFT normalization the sum is divided by the square of the total number of grid cells. There is no inverse FFT on the supported path. Unsupported/complex kernels use the original convolution fallback. Memory strategy retains the original computation.

The two switches can be tested separately. Their caches are scoped to one `run()` and released after it. They do not reuse production results between runs or alter bin edges, estimator conventions, physical normalization, scaling functions, or frequency folding. In particular, no scaling factor is moved outside a folding sum. Any future change to that mathematics requires separate equivalence tests and is deferred.

Rank-zero `runtime_metrics` reports preparation, broadcast/delta, FFT/power, window, reduction/convolution and overall intervals. Per-rank durations overlap and cannot be added as a wall-clock total. The existing main-loop timer excludes initial spectral preparation; use **`compute_including_spectral_setup_seconds`** for a comparison that includes the first FFT and cross power. For application performance also time the complete RAM-catalog-to-curve call, including projection and preparation.

Tests: `tests/test_corr2pcf_optimization.py`; real MPI checks: `mpiexec -n 2 python -m pytest tests/test_corr2pcf_mpi.py`. The latter skips outside MPI. The isolated Slurm pilot in `benchmarks/2pcf_optimization/` compares main, candidate defaults, preparation only, spectral only and both, at J8/8 ranks x 4 threads, three independent processes per arm, with five small-J gates first. Every arm uses the same physical cores and original 31 finite shell bins. All sample times are retained; no speedup is assumed in advance.

Measured results and limits: [2026-10-16 controlled CPU tests](2pcf_optimization_results_20261016.md). The source is still opt-in; scaling and folding changes remain deferred.


## Window experiments (phase 2)

With `pair_reduction="spectral"`, set `spectral_window="fused"` or `"shell_edges"`.
The default remains `"materialized"`. Both candidates reuse byte-identical 1D scaling
samples/powers within one run; this changes no scaling or folding formula.

`fused` applies the transpose of the existing eight-term octant fold to the real
pair power and directly accumulates the original weighted alias terms. It avoids
allocating the full octant and folded 3D window. DC, Nyquist, mirror-plane
multiplicities and the inverse-Laplacian zero-mode convention remain explicit.
Complex/directional, full-rFFT, composite/materialized, and disk-cached windows
keep the materialized path. Generic custom octant functions remain authoritative.

`shell_edges` additionally shares scalar ball integrals across bins and ranks for
built-in `binned_shell` (`R_in`, `R_out`) and `thick_shell` windows only. No custom
function is inferred from its name. Explicit common edges allow exactly 32 sphere
integrals for 31 contiguous shells, per distinct pair power across the MPI job.
The volume-normalized difference is taken after these linear integrals; finite
precision cancellation must be checked against all original bin values. Edges
are deduplicated by exact values, never rounded or merged by tolerance.

All edge work and collectives are inside the existing main-loop timer. The first
FFT/power stays in the separately reported initial setup and in total timing.
Per-rank `window_fused_seconds`, `shell_edge_compute_seconds`, and
`shell_edge_collective_seconds` overlap between ranks; do not add maxima.
All scalar/basis caches are cleared after each run. Input fields, original
convolution/folding helpers, and opt-in defaults remain unchanged.

Before production: 39 local numerical checks and four real-MPI tests per rank
pass, including asymmetric powers, alias bandwidths 1/2, all products with
cross/nonuniform random fields, fallback, cache refresh, and more ranks than bins.

Measured phase-2 results: [window optimization, 2026-10-16](2pcf_window_optimization_results_20261016.md).


## Batched boundaries and shared spectral preparation (phase 3)

Experimental `spectral_window="shell_edges_batch"` computes up to eight assigned
ball radii in one grid traversal. It reuses reflected-power lookups and each
original alias's coordinates and scaling product across radii. It retains the
alias sum, DC/Nyquist multiplicities and ball normalization unchanged. Scratch
space is O(L * batch_size); no 3D array per radius is made. Empty edge ranks do
no traversal. Batches, metadata and scalar collectives remain in the main loop.

Experimental `spectral_fft="shared"` computes each distinct real input FFT and
pair power only on rank zero, then broadcasts real powers. The precondition is
replicated input fields, already guaranteed by Corr_2PCF's field broadcasts.
Local field identity keys are not transmitted. Shape/pair counts are checked
collectively. This is not zero-copy shared memory: a real power array remains on
each rank, and its broadcast cost must be included in setup/total timing.
The default `spectral_fft="per_rank"` remains unchanged. Complex/unsupported
window fallback and the memory strategy retain existing behavior.

All power/basis/scalar caches remain scoped to one run. A shared FFT may move
rank waiting from the main loop into setup: compare the complete call and the
FFT-inclusive compute interval, not only the main loop. Both paths are independent
opt-in candidates; adoption depends on controlled same-allocation measurements.

Pre-production checks: 54 local tests and seven real-MPI tests per rank on four
ranks, including asymmetric powers, alias bandwidths 1/2, 0/1/4/8 radii, bounded
chunks, all products/cross/nonuniform fields, cache refresh and global FFT counts.

Measured phase-3 results: [batch/shared spectral measurements](2pcf_batch_optimization_results_20261016.md). J9 benefits in both layouts; J7/J8 retain the phase-2 recommendation.


### Exact integer radial classes (experimental)

`spectral_window="shell_edges_radial"` is opt-in and applies only to the same
built-in finite spherical shells as `shell_edges`. It groups original octant
terms by the **integer** `a*a+b*b+c*c`, without floating-point tolerances,
interpolation, radial binning or frequency cutoffs. For each pair power:

```
H[m] = sum_{a*a+b*b+c*c=m} transposed_fold_power * phi[a]*phi[b]*phi[c]
C_ball(R) = sum_m H[m] * W_ball(sqrt(m)/L, R) / Ncells**2
```

Each original alias has its own scaling weight before grouping. This is a
reordering of the original sum, not extraction of a common scaling factor from
aliases; all mirror-plane/DC/Nyquist multiplicities remain. Summation order and
radial norm evaluation differ in floating point, so equivalence is checked to
stated tolerances rather than bit equality. Signed cross-power is supported.

MPI ranks partition the original i planes and SUM their histograms, then
partition the unique shell boundaries. Numba parallel work items own private
histograms: O(threads * (bandwidth*L)**2) scratch, without a 3D window. A
conservative 256 MiB per-rank scratch guard falls back collectively to batch
edges. Histograms and scalar caches are rebuilt per run; no full-resolution
result is retained across samples. Histogram construction, compaction, MPI
histogram reduction, ball evaluation and edge gathering all stay **inside** the
existing main-loop timer and total timer. Source FFT/power distribution remains
in total setup as before. Custom callbacks, directional/complex kernels and
disk-cached windows retain existing fallbacks. Defaults are unchanged.

Performance requires controlled measurement including projection, setup,
communication and RSS; no universal speedup is inferred from kernel-only probes.

Controlled 36-sample results: [exact radial grouping measurements](2pcf_radial_optimization_results_20261016.md). Both tested layouts improve total median for J7/J8/J9; the new mode remains opt-in.
