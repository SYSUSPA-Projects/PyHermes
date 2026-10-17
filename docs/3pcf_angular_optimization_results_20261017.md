# Experimental standard angular 3PCF optimization, 2026-10-17

Branch: `codex/3pcf-angular-optimization`; baseline main 849de94; implementation 73df42f. These are local synthetic experiments, not a production/server performance table.

## Implemented changes

1. Box-random `delta_ddd` now constructs its three D-R fields once per product/rank. It releases them after that product and on exceptions. For the 20-angle Q+DDD workload, grid contrast allocations fall from 66 to 9, including the 6 contrasts required by lower-order pair terms. Particle-centre products retain their existing contrast preparation.
2. Root-only lower-order pairs use a local series helper built from the existing exact 2PCF power reducer and automatic window planner. Powers are shared across radii and released after each series. Planning uses COMM_SELF, never a collective Corr_2PCF task on rank zero alone. Unsupported windows retain convolution; their temporary full kernels are released after each radius so a planning list cannot retain one full grid per bin. For these uniform-random workloads,22 pair-convolution inverse FFTs become 0 and the spectral source FFT count is 6; for explicit random fields,44 inverse FFTs become 0 and 12 spectral source FFTs are used. This does not mean there is only one FFT for an entire Q task.

No user option or class-method signature is added/changed. MonteCarlo kernels, interpolation code, window/fold/reduction kernels, defaults and multipole scientific source are byte-identical to baseline. Centers, seeds, rotations and weights are preserved. The prior RR=0 exception behavior is retained.

The profiles below were measured on implementation 73df42f. A subsequent fallback-only memory guard does not affect the supported thin-shell paths in those profiles.

## Controlled local profiles

Each row uses one process, 2 threads, 20 angles and three independent processes per source arm. Order alternates between arms. The first two rows use J7 with512 box centers and 4 rotations; the final row uses J5 with8192 centers and 32 rotations to make MonteCarlo dominate. Fields are normalized synthetic positive grids, not the paper catalogue. Import/small-J warmup is excluded; task total includes in-memory preparation, the unchanged sampled triplets and root pair postprocessing. There is no projection, file IO or RSS observer in the measured interval.

|Workload|Baseline total / s|Optimized total / s|Speedup|Baseline pair post / s|Optimized pair post / s|
|---|---:|---:|---:|---:|---:|
|J7, uniform random, light MC|0.916287|0.162027|5.655|0.718307|0.103001|
|J7, explicit random fields, light MC|1.629245|0.278083|5.859|1.402700|0.197105|
|J5, uniform random, MC dominated|0.489728|0.466139|1.051|0.020298|0.007449|

Minima are selected independently for each phase and total; phase minima cannot be summed to reconstruct a total. All slow samples and ranges remain in the numeric data. The first MC-dominated baseline total was 0.857431 s, versus 0.502416 s and 0.489728 s later; it is retained. That row shows only about 5% lower minimum total, so the large gains in the lightly sampled rows must not be extrapolated to long production MonteCarlo tasks.

[Displayed CSV](_static/results/angular_3pcf_optimization_20261017.csv) and [all raw repetitions, counters and provenance](_static/results/angular_3pcf_optimization_20261017.json).

## Verification and boundaries

-18 fresh-process measurements, 9 same-input baseline/candidate comparisons passed rtol=3e-10, atol=1e-25. DDD,RRR,delta_DDD and zeta match exactly in these profiles; maximum absolute xi difference is 1.491e-18. Maximum relative Q difference is 3.221e-12. Independent synthetic cross fields can have tiny zeta_H and very large Q (up to 8.246e9 here); the maximum absolute Q difference 0.026554 is consistent with that relative bound. This is not an absolute-accuracy or cosmological validation.
-Full MPI-free suite:234 passed, 30 skipped (27 real-MPI cases separately covered;3 CUDA cases skipped without a device). Four actual MPI ranks each passed 27 tests, including the 4 new angular cases and 23 existing 2PCF cases. Weighted particle centres, explicit/nonuniform random fields, heterogeneous smoothing, empty centre ranks, duplicate/zero radii, tilted-window fallback, caller input ownership, repeated runs and exception cleanup are covered.
-MPI comparisons use identical rank counts and seeds on each arm. Box-center seeds depend on rank; changing the rank count changes the finite sample and is not an optimization-equivalence test.
-No production matrix, server job, GPU operation, article change or main merge is part of this experiment. The implementation remains on its isolated branch. Actual catalogue/Slurm performance and memory behavior still need a same-allocation experiment before publication.

## Next decisions

If a representative task is dominated by pair postprocessing, this prototype is ready for a bounded same-input server comparison. If sampled interpolation dominates, profile that kernel first; central-vertex interpolation is independent of angle and may be reusable, but any change needs fixed-seed, weighted and nonuniform checks. Identical normalized vertex/broadcast sharing remains a separate ownership/lifetime optimization. No approximation, seed change, reduced rotation count or scaling factorization across aliases is proposed.
