# Experimental standard angular 3PCF optimization, 2026-10-17

Branch: `codex/3pcf-angular-optimization`; baseline main 849de94; implementation 73df42f. The initial profiles below are local synthetic experiments. A subsequent J8 catalogue measurement on the same branch is reported separately, with its own timing and memory boundaries.

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
-The initial local profiles did not use Slurm or GPU resources. The later CPU catalogue measurements below remain on the isolated branch; no article change or main merge is included. They do not include a full-size baseline on the same allocation, so they are not a controlled production speedup measurement.

## J8 catalogue DDD and Q measurement

CPU job127098 ran four sequential array elements on cn2, using128 distinct physical cores,16MPI ranks x8threads,64GiB requested memory and no GPU. Each centre/product combination has three fresh-process measurements. The original remote preprojected real-space field (epsilon SHA f27eff57...) and unit-weight halo catalogue (SHA de53fe0d...) were copied byte-for-byte and verified; no projection or full-result reuse was performed. This input is not the subsequently refreshed local example field (epsilon836f...).

Both centre modes use J8, a periodic1000Mpc/h box, r12=20/r13=40Mpc/h,20 equally spaced theta values from0 to pi, seed42, no centre smoothing and sphereR5 windows only at vertices2 and3. Particle centres use406728 haloes and1000rotations (8,134,560,000 centre-rotation-angle combinations); box-random centres use8,000,000 positions and200rotations (32,000,000,000 combinations).

|Centre / product|Main loop / s|Compute + post / s|Task total / s|Main ns / comb.|Batch MaxRSS / GiB|
|---|---:|---:|---:|---:|---:|
|particle / ddd|78.077|78.077|79.540|9.598|7.747|
|particle / Q|78.664|80.461|82.154|9.670|11.616|
|box_random / ddd|294.019|294.019|295.491|9.188|9.906|
|box_random / Q|297.656|299.476|301.081|9.302|15.913|

Time columns take independent minima of three measurements; phase minima must not be added together. Main-loop ranges across the three runs are78.077–78.530s (particle DDD),78.664–79.442s (particle Q),294.019–305.358s (box DDD) and297.656–305.598s (box Q). All repetitions are retained in the numeric data.

The native main-loop timer includes product preparation/broadcasts, sampled triplets and the final collective; it excludes initial field/centre setup and pair postprocessing. Compute+post adds the root postprocessing. Task total begins before reading the preprojected field and includes field IO, endpoint convolution, centre preparation and computation; projection, imports, small-J compilation warmup, input audits and output IO are excluded. Q requests the connected sampled product (d_delta_dd for particle centres, delta_ddd for box centres) and its lower-order pairs; it is not the same product as raw DDD. Pair postprocessing takes about1.8s in these Q runs, and Monte Carlo interpolation dominates both Q and DDD.

Memory is the Slurm **batch-step MaxRSS for one centre/product array element**, including its gates and all three fresh processes. It is neither a minimum of three memory measurements nor per-rank/process-tree RSS. Rank masks are set before scientific imports; each rank binds its memory to the NUMA nodes represented by its actual core group, and transparent huge pages are disabled for those processes only. No host-wide policy is changed. These runtime/allocation conditions differ from historical runs.

All12 measurements passed independent SHA, saved-result array, native timer, physical-core/thread-mask and same-layout repetition checks. Each element first passed small-J baseline/candidate DDD/Q comparisons on16ranks; no full-size baseline was rerun. The original kernels, seeds, rotation counts and alias folding remain unchanged. Therefore this table establishes current candidate performance, not an algorithm-only speedup relative to the paper.

The historical particle example requested Q:75.8514s main loop plus postprocessing gives83.4886s, near the paper's84s. The available box example instead reports474.6013s main/482.1340s compute, and does not substantiate the paper's281s. The historical input SHA and batch-memory provenance for those paper numbers cannot be fully reconstructed from these logs. No paper value or old result has been replaced automatically.

[Catalogue table CSV](_static/results/angular_3pcf_catalogue_20261017.csv) and [all12 timings, ranges and provenance](_static/results/angular_3pcf_catalogue_20261017.json).

## Next decisions

If a representative task is dominated by pair postprocessing, this prototype is ready for a bounded same-input server comparison. If sampled interpolation dominates, profile that kernel first; central-vertex interpolation is independent of angle and may be reusable, but any change needs fixed-seed, weighted and nonuniform checks. Identical normalized vertex/broadcast sharing remains a separate ownership/lifetime optimization. No approximation, seed change, reduced rotation count or scaling factorization across aliases is proposed.
