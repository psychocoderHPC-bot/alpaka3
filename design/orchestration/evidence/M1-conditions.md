# M1 Conditions

## Source
- Repository: `https://github.com/psychocoderHPC-bot/alpaka3`
- Branch: `gemm/common-source`
- SHA: `e2d552bee79c7479e3044723ee1a26f90e5e70d7` (verified via `git rev-parse HEAD` on both local clone and uploaded tarball on `hal`)

## Target
- Name: `hal`
- Hostname (container): `c692209a108c`
- 64 logical CPUs (32 physical cores × 2 SMT threads), 1 RDMA node
- CPU: **AMD EPYC 7452 32-Core Processor** (Zen 2, AVX2+FMA+AVX512 capable)
- Physical: 32 cores × 2 threads = 64 logical CPUs
- OS: Ubuntu 24.04 (populated from an older pool; kernel 5.x)
- Governor: **schedutil** (not `performance`); scaling_max_freq = 2.35 GHz; idle at ~1.5 GHz.
  - Caveat: freq scales under load. Baseline-vs-native and 1t-vs-64t comparisons are
    internally consistent (same window, same governor) but the **absolute GFLOP/s numbers
    are lower than a fixed-frequency `performance`-governor run would give.**

## Compiler & CMake
- `g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
- `cmake version 3.28.3`
- `ninja` (GNU build), `perf` (Ubuntu 6.x default, installed in image)

## Two build configurations (both Release, C++20, -fopenmp, -O3, -Wall -Wextra -Wpedantic -Werror)

| flag                 | baseline (`b-base`)   | native (`b-native`)   |
|---|---|---|
| `ALPAKA_GEMM_NATIVE_ISA` | **OFF** (default) | **ON** |
| `-march` option    | (none → gcc-13 baseline = **SSE2, no FMA**) | `-march=native` → **`-march=znver2 -mavx2 -mfma -mavx512f ...`** |
| Effective ISA in codegen | 0× `vfmadd`, 0× `vaddps/vmulps`, 105× `mulps/addps` (SSE2) | 1108× `vfmadd`, 206× `vaddps/vmulps`, 111× `addsd/mulsd` |

Both built with:
```
cmake -S /tmp/m1/src -B /tmp/m1/bBASE \
  -DCMAKE_BUILD_TYPE=Release \
  -Dalpaka_TESTS=OFF \
  -Dalpaka_BENCHMARKS=ON \
  -Dalpaka_COMPILE_PEDANTIC=ON \
  -Dalpaka_DEP_OMP=ON \
  -Dalpaka_HOST_Cpu=ON
  # (native only) -DALPAKA_GEMM_NATIVE_ISA=ON
ninja gemm_bench
```

## Workload
- Manifest: `M1/manifest.json` (side file, not committed to repo)
  Because the frozen `workloads.json` has **no** `square-512` family, two cases
  (512³ and 1024³) are selected by `--filter square-512` / `--filter square-1024`.
  Single transpose (NN), single stride (compact), `alpha=1 beta=0`.
- Numerical modes run per case (`--subset all`): `fp32` and `fp16_fp32`.
  The fp16 entries are the "mixed fp16→fp32" (widening) comparison.
- Driver flags (per case): `--samples 10 --rounds 3 --warmups 1`
- Pinning: `taskset -c 0-63`

## Exclusivity
The run was executed one process at a time in the foreground. **One foreign workload
on the node** (`/tmp/v4/run_device.sh a770 2` → `./unit_test_algorithm_gemm -s gemm
exhaustive small shapes - TestBackends - 2`, PID 403989) started at 17:21:34, pinned
~1 core at 100 %, and was still alive at 17:25. The final clean window (data presented
below) was taken between 17:29:40 and 17:31:29, **after the foreign job cleared**
(foreign-process sentinel in the driver returned no hits at each `BEFORE` and
`AFTER` read). The earlier data taken between 17:17 and 17:23 is discarded as
contaminated (it is also the source of the initially-suspicious 11.5 ms / 186 GFLOP/s
"1 t 1024³ native" number, which was actually a 64-thread run whose `OMP_NUM_THREADS`
environment variable did not propagate; see `M1-caveats.md`).

Control-cases verified in the same clean window:
- fp32 1-thread 512³ baseline : 39.4 ms → 6.812 GFLOP/s  (matches the "~7 GF/s at 1 t" reference)
- Linear scaling T=1/2/4/16 native 512³ : 11.2 / 22.2 / 44.0 / 153.3 GFLOP/s
  (empirically verified, so `OMP_NUM_THREADS` is honored by the OmpBlocks executor)
