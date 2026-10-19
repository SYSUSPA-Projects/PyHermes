# Exact shared window strategies for 2PCF (2026-10-17)

These controlled CPU measurements extend the earlier finite-shell optimization. They use the same original preprojected J=8 RSD field (406,728 unit-weight haloes, periodic box 1000 Mpc/h, db2, bandwidth 1) on AMD EPYC 9754 CPU nodes: the initial cohorts ran on cn5 and the full axial sampling follow-up ran on cn1. Every case uses 32 distinct physical cores. Main/default interfaces and the per-alias scaling/folding relation are unchanged. The new strategies are enabled with spectral reduction, input reuse, automatic windows and shared FFTs; the defaults remain unchanged.

**Timing boundary:** input is a saved, already projected SFC field. Task total includes reading that field, preparation, initial FFT/power/broadcast, planning, tables and the complete sampling loop. It contains no catalogue projection. Main loop includes all radial grouping/factor planning/batch work and communication, but excludes initial FFT setup. Imports, small-J compilation warmup and output IO/audits are outside both numbers. Every formal run is a new process with fresh production fields/windows/powers; OS and eligible Numba disk caches may be warm. Startup and warmup are retained in raw records. These are not cold application timings or the projection-inclusive Fig.13 timings.

## Optimized workloads and amortized cost

The latest summary combines the earlier radial/ring measurements with the
user-requested full axial sampling in job 127085. Every row is a three-process
minimum at J=8,32 physical cores,8x4. Allocations differ between jobs, so this is
not a same-allocation cross-window speed comparison. Task/sample uses unrounded
task-time minima and includes amortized input/setup/FFT cost; it is not the
incremental cost of adding one sample. Original64-point measurements are retained
in the controlled stage tables below.

|Window|Samples|Main / s|Task / s|Main/sample / ms|Task/sample / ms|
|---|---:|---:|---:|---:|---:|
|shell|31|0.085|0.453|2.752|14.606|
|thick_shell|31|0.082|0.470|2.632|15.159|
|gaussian_shell|31|0.083|0.471|2.661|15.185|
|ring|2346|6.869|7.285|2.928|3.105|
|gaussian_ring|2346|6.690|7.100|2.852|3.026|
|disk|2346|6.839|7.395|2.915|3.152|
|cylshell|2346|6.136|6.599|2.615|2.813|
|cylinder|2346|6.186|6.849|2.637|2.920|
|cubic|31|0.264|0.669|8.528|21.589|

The main/sample and task/sample columns divide the respective unrounded minima
by the sampling count. Initial input/setup/FFT cost is included only in the task
column. Raw repetitions, minima, medians, ranges and source jobs are bundled in
[the numeric JSON](_static/results/twopcf_window_strategies_20261017.json);
[the CSV](_static/results/twopcf_window_strategies_20261017.csv) contains the displayed table.

## Matched axial sampling follow-up

Job 127085 measures disk, cylshell and cylinder on cn1, each with the exact ring
sampling arrays: 46 s values from 0 to 180 in steps of 4 Mpc/h, and 51 mu values
from 0 to 1 in steps of 0.02, for 2346 samples. The built-in smu mapping retains
R=s sqrt(1-mu^2), H=s mu and each window's original definition. The same s/mu
coordinates do not imply identical averaging geometry between different windows.

Scientific source d733177 is frozen independently; no mathematical, default or
interface change is introduced. Nine fresh-process formal runs and six small-J
fused-versus-auto gates passed. Each complete axial curve agrees exactly at the
64 coordinates shared with the earlier verified 127025 curve (maximum absolute
xi difference 0.0). The 32 physical cores, all recorded worker masks, input/output
SHA, one shared forward FFT, no inverse FFT, batches of at most 8, and the
16 MiB/rank factor-cache bounds were independently checked. The nominal 0.1 s
parent RSS observation policy is retained; its concurrent wall cost is recorded
and cannot be subtracted from task time.

Cubic reuses its valid 127025 measurements: the same 31 s values as shell, from
5 to 150 Mpc/h. This workload scales a rectangular window with fixed
Lx:Ly:Lz=1:1.3:0.7; it is a one-dimensional size scan, not an independent
three-length grid or an equal-sided cube. No new cubic or ring run is needed.
The superseded configuration 127082 was cancelled during its first gate with
zero formal measurements after the user changed the proposed 46-point cubic scan
back to shell's 31 points. Both frozen configurations and the cancelled scene
remain preserved.

Full unrounded statistics, all repetitions, source jobs and resource records are
in the local measurement archive, in analysis/report.md and combined_table.csv. Main-loop and task-total minima are selected independently;
do not add phase minima to reconstruct a task total. Independent server archive:
deliveries/2pcf_matched_sampling_20261017_v2.

## First stage: structure-aware evaluation

Job 127023 compares the public-main baseline with c5b2220 in one fixed allocation, alternating order, three independent processes per arm. Baseline uses fused reduction except the already optimized finite-shell path. Minimum of three is shown; all slow samples, medians and ranges are retained. Radial windows have 31 samples, ring has 46×51=2346, and the other axial windows have 64.

|Window|Samples|Baseline main / s|Structured main / s|Baseline task / s|Structured task / s|
|---|---:|---:|---:|---:|---:|
|shell|31|0.669133|0.085326|1.071305|0.452790|
|gaussian_shell|31|0.983433|0.082504|1.421420|0.470730|
|thick_shell|31|0.073523|0.081583|0.466400|0.469944|
|ring|2346|91.904749|21.281215|92.315879|21.695307|
|disk|64|2.123592|0.757219|2.540001|1.144365|
|cylshell|64|2.210477|0.724519|2.601957|1.127223|
|cylinder|64|2.027632|0.727249|2.398902|1.111281|

Thin and Gaussian shells benefit from evaluating exact occupied radial frequency classes once. The already optimized thick shell has no additional speedup here. Grid-axis axial windows share bounded transverse/longitudinal factors. This stage adds neither interpolation nor a frequency cutoff.

## Second stage: bounded separable query batches

Job 127025 compares single-query d197163 with bounded batches 96c0b34 in a separate fixed allocation. Both arms already use automatic window strategies. Groups of at most eight queries share traversal/power reads and each alias’s scaling product, retaining each window’s original alias terms. The same backend also handles Cartesian sinc factors.

|Window|Samples|Single main / s|Batched main / s|Single task / s|Batched task / s|
|---|---:|---:|---:|---:|---:|
|ring|2346|21.930703|6.868846|22.319228|7.285165|
|disk|64|0.687420|0.279807|1.062427|0.677211|
|cylshell|64|0.702096|0.335833|1.110350|0.778101|
|cylinder|64|0.692123|0.288339|1.096434|0.758653|
|cubic|31|0.327215|0.264376|0.722708|0.669257|

The two stages have independent allocations; multiply their speedups only as a descriptive comparison, not as a single controlled baseline-versus-final measurement.

## Gaussian-ring and layout checks

Job 127033 compares the original example custom Gaussian-ring formula with the built-in implementation using automatic factors, on final scientific source f19380a. The full 2346-sample curve is measured three times per arm. Four small gates also cover zero and nonzero Gaussian width.

|Arm|Main / s|Including initial FFT / s|Task / s|
|---|---:|---:|---:|
|baseline|86.864036|86.997670|87.384169|
|candidate|6.689863|6.799746|7.099754|

Maximum absolute Gaussian-ring xi difference is 3.136e-13; all tested products satisfy rtol=1e-8, atol=1e-11. Gaussian-ring uses the common factor backend with multiple length arguments. Its transverse Gaussian damping remains distinct from an isotropic Gaussian shell.

Ring layout comparison in job 127025 (three processes/layout, same 32 cores):

|MPI×threads|Main / s|Including FFT / s|Task / s|Case process-tree RSS range / GiB|
|---|---:|---:|---:|---:|
|8x4|6.868846|6.974539|7.285165|4.869–4.883|
|4x8|7.418561|7.511775|7.843683|2.686–2.698|
|2x16|8.450196|8.515610|8.801880|1.361–1.377|

8×4 is fastest among these tested layouts. A separate final-source job 127035 contains just one full ring case plus small-J warmup/gate: main 6.590892s, task 6.997884s. Its Slurm batch-step MaxRSS is 4.367439 GiB (4579592 KiB). This single resource/calibration run is not pooled with the three-repeat timing cohort. Process-tree RSS polling uses a nominal 0.1s sleep between observations and sums launcher/descendants; the actual cadence also includes sampling work, whose elapsed time is retained in each raw timing record; shared pages may repeat. Slurm batch-step MaxRSS is a separate measurement and includes startup/warmup. Observer work runs concurrently and its wall time cannot be subtracted from the scientific task wall time to obtain an unobserved runtime. No mixed-workload batch RSS is presented as one case's RSS.

## Validity and boundaries

- All 85 formal measurements and 31 small gates passed source/input/output SHA, physical-core/thread-mask and numerical checks. Independent raw-run statistics agree with JSON/CSV tables; protected branch tips are unchanged. All original slow samples and startup records remain available.
- Exact radial grouping evaluates original functions on integer squared-frequency classes. Inverse Laplacian preserves removal of the entire folded DC coefficient, including its aliases.
- Exactly grid-aligned axial windows and Cartesian cubic windows share factor reduction. Tilted, unsupported, complex and composite/cached cases retain safe existing paths. A custom callable is authoritative even if its dictionary has a built-in name.
- Per-rank guards bound radial scratch to 256 MiB, factor scratch to 64 MiB and the factor LRU to 16 MiB. These exclude field/FFT/runtime memory. Small or oversized groups fall back; automatic mode is not a universal speed guarantee.
- Dynamic window-function kernels compile per process rather than storing unsafe weak dispatcher references in disk indices. Eligible numeric kernels retain disk caching. Cold startup may dominate a short task; retain/report warmup separately.
- The first zero-formal job 127021 failed a gate and remains preserved. The subsequent wrapper/cache/progress repairs are documented separately; no valid scientific samples were discarded or rerun.
- Example results use different input-file hashes and were refreshed separately.
  This measurement campaign did not edit the article. The current public
  benchmark overview draws its timings from these archived repetitions;
  subsequent article edits are separate from this frozen experiment.
