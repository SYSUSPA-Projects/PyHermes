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
