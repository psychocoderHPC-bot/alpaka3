# M1 caveats and uncertainty

## 1. ISA and tile are confounded in this A/B
The `T_TN` the planner emits is a function of the compiled SIMD width (see
`M1-perf-kernel.md`). Baseline runs T_TN=4, native runs T_TN=8. The ~1.66× 1-thread
speedup is therefore a sum of (a) SSE2->AVX2+FMA and (b) per-worker tile widening
T_TN 4->8. Neither is isolated.

To separate them you would need a third build, e.g.:
- `b-native-tile4`: `-ALPAKA_GEMM_NATIVE_ISA=ON` **and** a compile-time pin of
  `getArchSimdWidth<float>` to 4 (forcing the planner back to T_TN=4), so ISA is AVX2/FMA
  but the tile stays 4; or
- `b-base-tile8`: baseline -march (SSE2) with `T_TN` forced to 8 (would need the
  planner to honor a lane that the SSE2 execution cannot actually fill; useful as a
  "wider register tile at baseline width" probe).
Neither was built this run (task scope: build only `gemm_bench` in the two given configs).

## 2. Governor = schedutil (frequency not pinned)
The CPU ran at idle 1.5 GHz and escalated under the 100% kernels. All comparisons are
internally consistent (same window/governor) so **ratios** are meaningful, but the
**absolute GFLOP/s** is below a fixed-frequency `performance`-governor run. The task
specified the build options, not a governor, so no governor was changed.

## 3. Disabled earlier data (contamination)
Between 17:17 and 17:23 a foreign workload (`/tmp/v4/run_device.sh a770 2`) pinned a
core. Worse, an `OMP_NUM_THREADS` shell env variable leaked from a 64-thread run into
the "1-thread" native 1024³ run, producing a wrong 11.5 ms / 186 GFLOP/s. That row and
its siblings were discarded and re-measured in the clean 17:29-17:31 window presented
here. The leak is a harness bug, not a physical result, and was caught by the scaling
diagnostic (T=1/2/4/16 gave monotonic 11/22/44/153 GF/s confirming OMP is honored).

## 4. 512³-64t dependency noise
At 64 threads, 512³ tile count is small (512/8 = 64 block rows × 64 block cols =
4096 tiles; each tile is 8×8 register FP32). Under the "native" 153.9 GF/s ceiling a
single EPYC-7452 board peaks around 230-260 GF/s (2350 MHz × 2 FMA × 8-lanes / lane,
× 2 uops per lane for FP32) so 153.9 is already ~60-65% of peak physical throughput.
The native/baseline 512³-64t speedup of 1.227× is therefore partly ceiling-limited.

## 5. fp16-GFLOP/s is a bookkeeping number
For mixed-precision the benchmark still counts 2*M*N*K (fp32-equivalent), so the raw
GFLOP/s values (1.5-2.3) are not comparable across precisions. Use the ms and the
speedup.

## 6. SMT caveat
`taskset -c 0-63` does not disable SMT. At 64 threads on 32 physical cores two OMP
threads share a physical core (SMT siblings), which limits scaling below 32x. 512³ at
64t also exposes SMT contention; the 153.9 GF/s number is "32-physical, SMT on".
Not required to change for the M1 "isolate the gain" goal.
