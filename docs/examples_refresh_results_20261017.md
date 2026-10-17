# Verified example refresh (2026-10-17)

All four affected 2PCF examples and three Gamma multipole examples were refreshed from the exact current local example fields, verified, and installed under new versioned filenames. Historical outputs are retained. These are one measured run per example, not three-repeat timing statistics or a new best-layout comparison. All use 32 distinct physical cores on cn5 (AMD EPYC 9754).

## Measured times

Task total starts from already projected SFC files and includes their IO, preparation, initial FFTs and the complete requested task. It contains no catalogue projection. Main loop excludes input preparation and initial FFT setup. Imports, small-J serial/MPI gates and compilation warmup, output IO and post-run audits are outside task total; warmup and complete launcher wall time are retained separately in the CSV. Eligible compilation and OS caches may be warm.

|Example|MPI×threads|Main / s|Task / s|Case process-tree RSS / GiB|
|---|---|---:|---:|---:|
|2PCF isotropic|8x4|0.143807|0.907627|5.421|
|2PCF ring_small|8x4|0.235753|0.880393|5.125|
|2PCF ring_full|8x4|6.661952|7.319434|5.065|
|2PCF gaussian_ring_full|8x4|7.070271|7.776855|5.106|
|Gamma J8_l14|8x4|453.844500|455.978248|11.003|
|Gamma J8_l14_nonuniform|8x4|1473.035600|1476.199515|11.383|
|Gamma J9_l14|2x16|4049.553000|4059.775936|22.741|

Case RSS sums launcher and descendants in each observation; shared pages can repeat. Whole-job Slurm batch-step MaxRSS for Gamma job 127044 is 28022292 KiB = 26.724140 GiB, including all three examples and their warmups. It is not any individual example RSS. The job completed successfully in 01:41:15 with a 64 GiB reservation.

## Numerical and input verification

- All saved example arrays exactly equal the raw measurement NPZ arrays. Frozen source/input/output hashes, finite products, real physical-core groups, sampled worker-thread masks, THP policy and NUMA masks passed. All three Gamma cases used the same NUMA node in their actual allocated core groups. Raw-run phases/RSS agree with the reported CSV.
- A full fused ring baseline was evaluated once with the current example input, giving maximum absolute xi difference 4.352074256530614e-14 from the optimized path. The two earlier valid example outputs were retained after their campaign stopped at a historical-reference assertion; only missing outputs were recovered.
- The independent Gamma check builds the product matrix using Legendre-polynomial multiplication, rather than the production Wigner-3j builder. Solving A=B*zeta reproduces the saved coefficients with maximum difference 6.939e-18; the explicit-random matrix condition is 1.000019647. Uniform-random conditions are 1. This complements the independent basis-integral and serial/MPI regression gates.
- The example RSD field differs from the earlier benchmark field. Its historical ring reference is byte-identical to the old benchmark reference, so the roughly 3.25e-7 old-versus-new difference is an input-provenance difference, not the same-input implementation error. Gamma examples likewise use the actual local fields, not coefficients copied from a different benchmark input.

Canonical input full-file SHA256 values:

- `quijote8000_snap004_sfc.pkl`: `6f341ea2d0c184c97420601b87c20e3417fa75c8ba52f821197218fe85700d96`; epsilon `836f9d81f54b63dcc6c93f3ba241aee59b93a6aaf0ae9bf6b819c79ac6f32faa`.
- `quijote8000_snap004_sfc_J9.pkl`: `7a25e30b95bc9c643211f34c4c75ea4f5fe42d0ed0cbf4e378219a2877316918`; epsilon `dea0853c4ae205ba33232240f61d10ba881102d546f8643ad802fd4176e5ee92`.
- `quijote8000_snap004_rsd_sfc.pkl`: `fa817098c78b9903730abf598a6009a1394ff6305a89861ad32416e53ca0be4e`; epsilon `a80db1bc0599a9d0d7a5735e6631ec20a925d28eaccfb519775ac62c5a3064f9`.
- `random_sfc.pkl`: `ef4023a31e6d5841ecc283abdf9ab09b4b9c19932e7b92fc010d76b704f4582e`; epsilon `b3421365d1273c414b2be17ee3485fa229c59c296d5e309959c4f4fc3fd3fb8b`.

## Notebook and measurement boundaries

The 2PCF notebook actually executes the affected configurations/results and Gaussian-ring plots. The 3PCF notebook executes cells 2,30,39,41,42,44,45 from saved results, with embedded figures and new filenames. Both updated figure sets were visually checked. Gamma examples explicitly use CPU contraction, an unsmoothed first vertex, sphere R=5 on vertices 2/3, r12=20, r13=40 and lmax=14. The raw-component plot now divides physical-density products by physical mean density cubed; its earlier grid-density normalization left a spurious resolution-dependent factor. Saved products and zeta_l are unchanged by this plotting correction.

Historical direct-angular curves retain finite Monte Carlo rotations and do not store an exact coefficient-array input SHA. Their reconstruction overlay and low-ell projection are diagnostics, not strict identical-input/identical-estimator validation. The observed relative RMS differences (1.696% for particle centres, 6.360% for random centres) must not be treated as Gamma implementation errors or as an absolute accuracy test.

RSS polling sleeps nominally 0.1 seconds between observations; actual cadence includes observation work. The parent observer runs concurrently. Its elapsed wall time is neither a measured CPU cost nor a subtractable component of science time. These are observed timings, with that possible interference included.

|Gamma example|Observer wall / s|Samples|Mean effective cadence / s|
|---|---:|---:|---:|
|J8_l14|222.277|2318|0.196712|
|J8_l14_nonuniform|710.553|7571|0.194981|
|J9_l14|1926.566|21165|0.191816|

## Source and scope

The scientific source is f19380ad86ee2409b6f974cdb26492de7970e5b9; example configurations are frozen at 8604e6d71d839b0d437767b2b98a82ec0c922a96 with identical scientific files. Gamma manifest: 4d514735f95e4eea846cade200b767efa1cb67e518ae7b71a8dc4fb6ec301f1c. Earlier wrapper failures and all valid measurements remain preserved. There were no duplicate successful measurements.

The 2PCF core report separately retains 85 formal runs and 31 gates, three-repeat layout statistics, exact alias scaling/folding and independent resource measurements. The nonmultipole 3PCF/software review identifies product-local contrast reuse, common local pair reduction and preparation sharing as future candidates; no unrequested production matrix or nonmultipole scientific change was introduced. Main, protected branch tips, old article figures and Table 3 remain unchanged.
