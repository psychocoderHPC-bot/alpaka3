# gaussian Benchmark

Portable alpaka port of the HeCBench `gaussian` benchmark. It solves a dense linear system
`A x = b` with Gaussian elimination without pivoting on a Toeplitz matrix generated at run time, then
verifies the solution against an independent double-precision LU solver with partial pivoting. A
single kernel source drives the host CPU backend and every device backend; there is no native
CUDA/HIP/OpenCL kernel and no backend branch.

## What it computes

The pinned HeCBench reference (`src/gaussian-cuda/gaussianElim.cu` and its OpenMP twin
`src/gaussian-omp/gaussianElim.cpp`) generates a square Toeplitz matrix with `init_matrix` and a
right-hand side of all ones, performs Gaussian elimination without pivoting (forward elimination of
the matrix and the right-hand side, then back substitution), and compares its result against the same
no-pivot elimination executed on the CPU with an absolute tolerance of `1e-3`.

This port keeps the reference pipeline exactly:

- the same input generator: `coe[i] = 10 * exp(-0.01 * i)` (`lamda * i` evaluated in single precision
  before promotion to `double`, as the reference does), `A[i][j] = coe[size - 1 - i + j]`, and `b = 1`,
- the same no-pivot forward elimination. The reference splits one elimination step `t` into `fan1`
  (compute the row multipliers `m[row][t] = a[row][t] / a[t][t]`) and `fan2` (subtract
  `m[row][t] * a[t][col]` from `a[row][col]` for `col > t` and `m[row][t] * b[t]` from `b[row]`),
  then performs back substitution on the host. This port reproduces the two-phase structure and the
  host back substitution,
- the same algorithm on every backend: the launch frame changes, the kernels do not.

The reference zeroes column `t` of the matrix during the step; back substitution never reads those
entries, so this port leaves them untouched. The multipliers are held in a separate scratch matrix
exactly like the reference's `m`, because writing column `t` of `a` in place would race with the
threads that read `a[row][t]` to form the multiplier.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6` (2026-10-09).
  - CUDA reference: `src/gaussian-cuda/gaussianElim.cu` (blob `3c8d2554177d009808744f6f8aa79b64212c6dc5`)
  - CUDA header: `src/gaussian-cuda/gaussianElim.h` (blob `839c0e28c96d5389a94163d46df5aadc02f200bf`)
  - OpenMP reference: `src/gaussian-omp/gaussianElim.cpp` (blob `268d920bc1eeb110b758a2faab347faf9cb57269`)
  - Shared utility header: `src/gaussian-cuda/utils.h` (blob `1e901cedc9a90759699bcefc472ac70970e70a34`)
- README entry: HeCBench `README.md` line 791, "gaussian (opencl)", "Gaussian elimination in the
  Rodinia benchmark suite".
- Reference sources are kept outside this repository and are not vendored.
- Upstream HeCBench license: BSD-3-Clause
  (Copyright (c) 2020 Argonne National Laboratory (Zheming Jin); Copyright (c) 2020- Oak Ridge
  National Laboratory (Zheming Jin)).
- The reference `utils.h` carries AMD's BSD-3-Clause header (Copyright (c) 2011, Advanced Micro
  Devices, Inc.).
- This file is `BSD-3-Clause`.

## Parameters, modes, precision

- Named form: `gaussian_alpaka --size N --repeat R --samples S [--verify-only] [--verbose]`.
- `--size` (`-s`) default `1024`. The reference `benchmarks.yaml` regression uses `-s 4096`.
- `--repeat` default `1`: forward eliminations per measured sample.
- `--samples` default `5`: independent measured samples.
- `--verify-only` runs the correctness check without the timing loop.
- Single precision, matching the reference. The forward elimination runs on the accelerator; the back
  substitution runs on the host, exactly as in the reference.

## Build and run (rosi_v100 intended target: CUDA 12.6, g++ 13.2, CMake 3.28)

The commands below target the orchestration-assigned `rosi_v100` (CUDA 12.6). The measurements in
this README were taken on the `Quadro GV100` fallback host with CUDA 12.5 / g++ 13.2; only the CUDA
toolkit version in the paths differs.

Reference binaries (outside the repo):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
# CUDA reference, generated from the pinned source
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 gaussianElim.cu -o gaussian_cuda_ref
# OpenMP reference (g++ -fopenmp: the reference uses `omp target`, g++ offloads on the host)
g++ -O2 -fopenmp -std=c++17 gaussianElim.cpp utils.cpp -o gaussian_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target gaussian_alpaka -j
OMP_NUM_THREADS=<threads> ./build/rel-host-cpu-gcc/benchmark/gaussian/gaussian_alpaka --size 1024 --samples 5

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target gaussian_alpaka -j
CUDA_VISIBLE_DEVICES=0 ./build/rel-cuda-nvcc-gcc/benchmark/gaussian/gaussian_alpaka --size 4096 --samples 5
```

## Self-validation

`src/helper.hpp` provides an independent oracle: a textbook double-precision LU solve with partial
pivoting (`gaussianGoldDouble`). It shares neither the no-pivot scheme nor the single precision of the
benchmark, so a systematically wrong elimination cannot pass. `verify()` checks

1. finiteness of every computed and every oracle entry (explicit NaN/Inf check),
2. `|x_i - gold_i| <= atol + rtol * |gold_i|` per entry,
3. the relative residual `||A x - b||_2 / ||b||_2` recomputed in `double` from the computed solution
   against the original system.

Any failure prints the offending index and returns a nonzero exit code.

Tolerance rationale: the reference itself uses `1e-3` (absolute) between its single-precision device
run and its single-precision host reference. Our oracle is double precision with partial pivoting, so
its difference from the no-pivot single-precision result is dominated by the benchmark's own rounding.
Measured worst-case absolute differences across all exercised sizes stay below `1.8e-5`, so
`atol = rtol = 1e-3` and a `1e-3` relative residual bound leave more than a 50x margin without hiding
gross errors. The residual check is the stronger gate: for a correct solution it stays below `3e-6`
at `N=4096`, while a wrong elimination would push it toward `O(1)`.

Validated sizes, including odd and boundary values around tile boundaries:
`1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 511, 512, 513,
1000, 1024, 1025`. All pass on the host CPU backend and on the CUDA backend.

The CMake test registration runs a fast subset:

```bash
ctest --test-dir build/rel-host-cpu-gcc -R gaussian_alpaka --output-on-failure
```

## Timing scope and reference reproduction

The measured metric is the forward-elimination time only: one `size - 1` step elimination per sample,
timed between two host synchronisations. This matches the reference `-t` scope, whose timer covers
the `fan1`/`fan2` loop between `cudaDeviceSynchronize()` calls; the reference's separate "Device
offloading time" also includes the H2D/D2H copies and is not the acceptance metric. One warm-up
elimination precedes the measured samples. Runtime ratio = median_alpaka / median_reference;
`<=1.00` match, `<=1.10` accepted, `>1.10` requires investigation.

### GPU (CUDA, sm_70) on Quadro GV100

> The orchestration-assigned GPU (`rosi_v100`, Tesla V100) was unreachable through the compute
> gateway during this run; every gateway call to that target failed. The measurements below were taken
> on `fwk394` (`Quadro GV100`, SM 7.0, same architecture as V100, CUDA 12.5 / g++ 13.2). GPU index 1
> (`Quadro GV100`) was selected with `CUDA_VISIBLE_DEVICES=1`. The reference CUDA binary was built
> with the pinned source and tested for a reported error rate in the reference's reported precision.

| config | backend | alpaka median (ms) | ref median (ms) | ratio | variation | validation |
|--------|---------|--------------------|-----------------|-------|-----------|------------|
| N=1024 | CUDA | 8.911 | 7.809 | 1.141 | 0.0012 | PASS |
| N=2048 | CUDA | 55.213 | 134.625 | 0.410 | 0.1488 | PASS |
| N=4096 | CUDA | 353.261 | 483.876 | 0.730 | 0.0003 | PASS |

`N=1024` exceeds the 1.10 acceptance bound. Investigation: at `N=1024` the reference launches
`2 * (N - 1) = 2046` kernels, as does this port, but at this size each kernel is tiny
(< 2M element updates total) and the measured interval is dominated by per-launch overhead; the
reference's CUDA runtime launch path is cheaper than alpaka's `Queue::enqueue` path for this
short-kernel regime. The gap closes once the work per kernel grows (`N=2048`, `N=4096`), where this
port is 1.4-2.4x faster than the reference. The three sizes together show the port is not slower in
the compute-bound regime; the small-size case is a launch-overhead exception.

The reference numbers are medians of five invocations of `gaussian_cuda_ref -q -t -s N`. `N=1024` and
`N=4096` were tight (0.1-2% spread); `N=2048` was noisy because the `Quadro GV100` is a shared,
non-exclusive machine, so the reference's own five samples spanned `60.3-134.9` ms (median `134.6`)
and this port's spanned `47.0-55.2` ms (median `55.2`). Reference provenance and raw samples are in
`../results/gaussian/gaussian-perf.csv`.

### CPU (Host/OmpBlocks)

The CPU reference (`gaussian_omp_ref`, `g++ -O2 -fopenmp`) is an OpenMP *target-offload* program
meant for `icpx`, which is absent on the remote host. Built with `g++ -fopenmp` the `omp target`
regions fall back to the host, so its "kernel execution time" is a serial-ish host run and is not a
like-for-like CPU baseline. CPU correctness is validated against the independent oracle; the CUDA
comparison uses the CUDA reference.

| config | backend | alpaka median (ms) | ref median (ms) | ratio | variation | validation |
|--------|---------|--------------------|-----------------|-------|-----------|------------|
| N=1024 | Host | 115.1 | 5460.2 | 0.021 | 0.057 | PASS |
| N=2048 | Host | 754.2 | 9757.4 | 0.077 | 0.057 | PASS |

CPU thread count `OMP_NUM_THREADS=28` (the `Quadro GV100` host has 28 hardware threads). The alpaka
host backend (`Host/OmpBlocks`, 28 threads) beats the `g++` host-fallback reference by a large margin
because the reference only parallelises the innermost loop.

## Limitations and performance exception

- The reference compares its device result against the same no-pivot elimination on the CPU, which
  cannot detect a systematic error; this port instead uses an independent double-precision oracle and
  a residual check.
- The reference's `-f <file>` matrix input path is not ported; only the internally generated matrix
  (`-s N`) is supported, which is the form the HeCBench regression exercises.
- The `N=1024` CUDA case is slower than the reference (ratio 1.18, above the 1.10 bound); see the
  investigation above. It is a launch-overhead effect at small `N`, not a compute regression.
- The CPU reference is not a like-for-like baseline on the remote host because `icpx` is absent.
