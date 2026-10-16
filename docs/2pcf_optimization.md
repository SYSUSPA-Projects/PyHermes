# Experimental 2PCF optimization

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
