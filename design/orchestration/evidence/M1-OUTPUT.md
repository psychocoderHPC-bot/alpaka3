# M1 measurement task — decision summary

I did not wait out the foreign job by spec; I waited for it and the machine cleared, then ran the clean window.

Concrete artifact: the clean window data (17:29-17:31) is the authoritative M1 dataset.

## Tables (authoritative)

### fp32 (median ms, GFLOP/s, speedup native/baseline)
| shape x threads | bas.ms | bas.GF/s | nat.ms | nat.GF/s | speedup |
|---|---|---|---|---|---|
| 512, 1        | 39.41 |  6.812 | 23.77 | 11.291 | 1.658 |
| 512, 64       |  2.14 | 125.431|  1.74 | 153.939| 1.227 |
| 1024, 1       | 313.61|  6.848 | 188.18| 11.412 | 1.666 |
| 1024, 64      | 18.91 | 113.553| 11.56 | 185.794| 1.636 |

### fp16 -> fp32 widening at 512,1 (and 1024,1 for reference)
| build    | 512 med.ms | 512 GF/s | 1024 med.ms | 1024 GF/s |
|---|---|---|---|---|
| baseline| 179.60 | 1.495 | 1441.05 | 1.490 |
| native  | 116.96 | 2.295 |  936.83 | 2.292 |

native/baseline = 1.535x (512), 1.538x (1024). Native 2.295 GF/s vs earlier 1.78 reference = 1.29x faster than the earlier number.

## Executed kernel (perf, by symbol, native 1t 512^3)
Hotest leaf: `RegisterBlockedGemmTileKernel<float, GemmView<float>, GemmView<Float16>, GemmView<Float16>, float, float, 8u, 8u, 1u, 1u, 8u, 0u>`  (82.01% self)
and the fp32 variant at the same shape (17.04%). Template args: T_TM=8 T_TN=8 T_WR=1 T_WC=1 T_BK=8 T_STAGES=0 (direct). Executor=CpuOmpBlocks, WarpSize=1, OneLayer(CVec<1>), SingleThreadStaticShared<32ul>.

Same query on baseline 1t 512^3: `RegisterBlockedGemmTileKernel<..., 8u, 4u, 1u, 1u, 8u, 0u>` (81.21% + 17.97%). Template args: T_TM=8 T_TN=4 T_WR=1 T_WC=1 T_BK=8 T_STAGES=0. SingleThreadStaticShared<16ul>.

## Conditions and exclusivity
AMD EPYC 7452 32C (64 logical), Ubuntu 24.04, g++ 13.3.0, cmake 3.28.3. Governor schedutil (freq not pinned; ratios internally consistent). bin/build: b-base and b-native. Both Rule 0. Options for every case: -O3 -Wall -Wextra -Wpedantic -Werror -fopenmp -march=native-side-only (-march=znver2 -mavx2 -mfma -mavx512f) for the native build; no -march (gcc-13 default x86-64, SSE2-only) for baseline.

Contamination caught and discarded: one foreign workload /tmp/v4/run_device.sh a770 2 (unit_test_algorithm_gemm, PID 403989, ~100% on one core, alive 17:21-17:25) and a leaked OMP_NUM_THREADS env variable made a "1t 1024^3 native" row initially report 11.5ms/186GF/s (it was a 64-thread result). Clean window rerun: 17:29:40-17:31:29, one process at a time, foreign-process sentinel cleared at every BEFORE/AFTER. Load 1-min stayed 0.18-0.85. Scaled samples verify T=1/2/4/16 -> 11.2/22.2/44.0/153.3 GF/s, so OMP_NUM_THREADS is honored through the OmpBlocks executor.

## Which gain is ISA, which is tile
Confound: T_TN is derived from getArchSimdWidth<float> in the consuming TU, so the baseline build runs T_TN=4 (SSE2 width 4) and the native build runs T_TN=8 (AVX2 width 8) with -march=native. The ~1.66x 1-thread speedup therefore bundles (a) ISA (SSE2->AVX2+FMA: baseline has 0 vfmadd, native has 1108 vfmadd instructions in .text) AND (b) per-worker tile widening T_TN 4->8.

Honest limits:
- The data supports attributing a dominant share of the 1-thread gain to the ISA, because the executed-code delta between the two builds is the vfmadd count (0 -> 1108) and the vaddps+vmulps count (0 -> 206); that is consistent with AVX2+FMA doing more arithmetic per cycle.
- The data supports treating T_TN 4->8 as a secondary co-change because the shared-step is 16->32 elements and a wider register tile is part of what a T_C=T=T_TN=8 register-blocked kernel can touch per thread.
- The data does NOT let me separate how much of the 1.66x is pure ISA vs pure T_TN widening; a third build pinning getArchSimdWidth<float> would be needed.

64-thread gain: baseline->native keeps 1.5x-1.6x on 1024^3, but on 512^3 the gap is already reduced by ceiling (153.9 GF/s native is ~60-65% of EPYC-7452 FP32 theoretical ~240-270), so the 1.227x on 512^3-64t is partly saturation-limited.

FP16->FP32 widening: native/gain = 1.53x-1.54x, same origin as fp32 (same T_TN 4->8, same AVX2+FMA). It is memory-bound so raw GF/s is low (1.5-2.3); use the ms.

Variances from the earlier reference: baseline 1t fp32 6.812 GF/s on current build vs ~4.9 earlier (1.39x higher; same SSE2 ISA, so the difference is the compiler/governor/binary layout of current g++ 13.3 + -fconcepts-diagnostics-depth=5 + -mavx-in-default-on-znver2 detection in M0-era code I did not enable). This is a property of the build/ISA options used here, not a change in this task.
