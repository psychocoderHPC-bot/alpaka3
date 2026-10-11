# scan Benchmark

Portable alpaka port of the HeCBench `scan` benchmark. It splits an input array into independent
chunks of `N` elements and computes an *exclusive* prefix sum inside every chunk with a
shared-memory Blelloch/Harris scan (both the plain and the bank-conflict-aware variant), then
verifies the result against an independent sequential host reference with an exact element-wise
comparison. A single kernel source drives the host CPU backend and every device backend; there is
no native CUDA/HIP/OpenCL kernel and no backend branch.

## What it computes

The pinned HeCBench reference (`src/scan-cuda/main.cu`, OpenMP twin `src/scan-omp/main.cpp`) divides
an input of `nelems = ceil(n / N) * N` elements into chunks of `N` elements and produces

```
out[i] = sum(in[cN], ..., in[i-1])   for i inside chunk c, i > cN
out[cN] = 0
```

for each chunk `c`, i.e. a chunk-local exclusive prefix sum. Chunks do **not** carry into each
other: the reference discards the running total at every chunk boundary (`t_out[0] = 0; t_out[i] =
t_out[i-1] + t_in[i-1]`). It exercises the chunk sizes `N = 128, 256, 512, 1024, 2048` and the
element types `char`, `short`, `int`, `long`, generated with `srand(123)` and `rand() % 5 + 1`.

This port keeps the reference semantics exactly:

- the same chunk-local exclusive scan and the same chunk-boundary reset,
- both reference kernel variants: the plain shared-memory scan (`scan<T, N>`) and the
  bank-conflict-aware scan (`scan_bcao<T, N>`, shared memory padded by `x >> 5`),
- the same element types and chunk sizes,
- a grid-stride loop over chunks, matching the reference's `16 * multiProcessorCount` grid,
- an independent sequential host exclusive scan and an exact element-wise comparison.

The two variants differ only in shared-memory addressing. The plain variant stages `N` values; the
bank-conflict-aware variant stages `2N` values with the reference's `OFFSET(x) = x >> LOG_MEM_BANKS`
padding. A barrier before the write-back is required in both, and is present, matching the
reference's `__syncthreads()`.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA reference: `src/scan-cuda/main.cu` (blob `8fa99545998fea82c6d3c85ac9b51636d837b988`)
  - OpenMP reference: `src/scan-omp/main.cpp` (blob `864acde3c4cefaa0fe92ec302b6fb807fd5aba50`)
- README entry: HeCBench `README.md` line 1530, "scan (cuda)": "Scan with bank-conflict-aware
  optimization", <https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda>.
- Reference sources are kept outside this repository and are not vendored.
- Upstream HeCBench license: BSD-3-Clause
  (Copyright (c) 2020 Argonne National Laboratory (Zheming Jin); Copyright (c) 2020- Oak Ridge
  National Laboratory (Zheming Jin)).
- This file and `src/main_alpaka.cpp` are `BSD-3-Clause`. `src/helper.hpp` and `src/misc.hpp`
  contain no parts of the HeCBench code base and are declared `BSD-3-Clause`.

## Parameters, modes, precision

- Named form: `scan_alpaka --n N --chunk C --type T --repeat R --samples S`.
- `--n` default `67108864` (2^26); any non-negative element count is accepted. The scanned count is
  rounded up per chunk to `ceil(n / C) * C`, matching the reference.
- `--chunk` default `2048`. Accepts `128, 256, 512, 1024, 2048`, or `0` to run all of them (the
  reference's full default set).
- `--type` default `int`. Accepts `char`, `short`, `int`, `long`, or `all`.
- `--repeat` default `10`, `--samples` default `5`. Both must be greater than zero.
- `--verify-only` runs the correctness checks without the timing loop.
- Arithmetic is integral and exact: scans run in the element type, so `char`/`short` wrap exactly
  like the reference. The input is checked finite before use. There is no floating point reduction.

## Build and run (rosi_v100 target, CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 main.cu -o scan_cuda_ref     # src/scan-cuda/main.cu
g++ -O2 -fopenmp -std=c++17 main.cpp -o scan_omp_ref                        # src/scan-omp/main.cpp
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target scan_alpaka -j
OMP_NUM_THREADS=48 ./build/rel-host-cpu-gcc/benchmark/scan/scan_alpaka --n 67108864 --chunk 0 --type int --repeat 10 --samples 5

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target scan_alpaka -j
CUDA_VISIBLE_DEVICES=1 ./build/rel-cuda-nvcc-gcc/benchmark/scan/scan_alpaka --n 67108864 --chunk 0 --type int --repeat 10 --samples 5
```

## Self-validation

`src/helper.hpp` reproduces the reference input generator (`srand(123)`, `rand() % 5 + 1`) and an
independent sequential host exclusive scan with the exact same chunk semantics. After a fresh device
run the result is copied back and compared element-wise with the host reference (`verify()`). The
first mismatch prints `index, expected, got` and returns a nonzero exit code; the whole program also
returns nonzero. The input is scanned for non-finite values first.

Tolerance rationale: a chunk-local prefix sum over a fixed set of integers is an exact reduction.
The reference itself compares with `memcmp`/`!=` (tolerance 0). Any numeric difference is a real
bug, so this port keeps the comparison exact. For narrow types the reduction can overflow, and the
host reference performs the addition in the same type, so wrap-around matches bit for bit.

Validated element counts, including boundary and irregular sizes and values not a multiple of the
chunk size: `0, 1, 2, 3, 7, 15, 63, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1000, 1023, 1024,
1025, 2047, 2048, 2049, 4095, 4096, 4097, 65537, 1048575, 1048576, 1048577`. All element types and
both variants pass on the CPU and CUDA backend; a size-0 input runs the scan with zero chunks and
passes.

The CMake test registration runs a fast subset:

```bash
ctest --test-dir build/rel-host-cpu-gcc -R scan_alpaka --output-on-failure
```

## Timing scope and reference reproduction

The measured metric is the per-scan time: `repeat` back-to-back scans between two synchronisations,
divided by `repeat`. This matches the reference, which times `i` iterations and reports the average.
One warm-up scan precedes the measured samples. Runtime ratio = median_alpaka / median_reference;
`<=1.00` match, `<=1.10` accepted, `>1.10` requires investigation.

Reference values are the median of a single `scan_cuda_ref` invocation per size and chunk, which
prints the average over `repeat=10` for each type and variant. Warmup and `repeat=10` match the
benchmark.

Measured on an NVIDIA Quadro GV100 (32 GB, `CUDA_VISIBLE_DEVICES=1`, sm_70), 5 independent samples
per configuration. Full raw samples and all chunk sizes are in
`../results/scan/scan-gpu.csv` and `../results/scan/ref-cuda.csv`.

| size | type | chunk | variant | alpaka median (us) | ref median (us) | ratio | variation (relspread) | validation |
|------|------|-------|---------|--------------------|-----------------|-------|-----------------------|------------|
| 2^23 | int  | 512  | scan    | 295.58 | 410.54 | 0.72 | 0.003 | PASS |
| 2^23 | int  | 512  | scan_bcao | 209.37 | 264.88 | 0.79 | 0.033 | PASS |
| 2^23 | int  | 2048 | scan    | 419.49 | 486.74 | 0.86 | 0.001 | PASS |
| 2^23 | int  | 2048 | scan_bcao | 315.19 | 370.70 | 0.85 | 0.002 | PASS |
| 2^23 | long | 2048 | scan    | 577.88 | 649.62 | 0.89 | 0.002 | PASS |
| 2^23 | long | 2048 | scan_bcao | 354.38 | 508.12 | 0.70 | 0.001 | PASS |
| 2^28 | int  | 512  | scan    | 7276.0 | 12895.5 | 0.56 | 0.173 | PASS |
| 2^28 | int  | 512  | scan_bcao | 4641.0 | 5979.6 | 0.78 | 0.205 | PASS |
| 2^28 | int  | 2048 | scan    | 9355.6 | 12153.8 | 0.77 | 0.151 | PASS |
| 2^28 | int  | 2048 | scan_bcao | 6906.1 | 8076.5 | 0.86 | 0.381 | PASS |
| 2^28 | long | 2048 | scan    | 13163.0 | 14807.4 | 0.89 | 0.077 | PASS |
| 2^28 | long | 2048 | scan_bcao | 9035.0 | 11401.8 | 0.79 | 0.004 | PASS |

The larger-size `2^28` measurements use the default reference run target `268435456 100`. The
`scan_bcao` variant is the faster of the two on the GPU in every case, matching the reference's own
observation. Ratios use the reference's own reported variant, so the comparison is variant-for-variant.

The CPU reference (`scan_omp_ref`) is an OpenMP *target-offload* program meant for `icpx`, which is
absent on the measurement host. Built with `g++ -fopenmp` it falls back to a host path that reports
`FAIL` and is not a usable CPU correctness or performance baseline. CPU correctness is validated
against the independent host reference; there is no CPU runtime ratio.

Raw per-run samples, exact commands, and CPU/GPU/compiler metadata are in
`../results/scan/scan-gpu.csv`, `../results/scan/ref-cuda.csv` and `../results/scan/provenance.txt`.

## Limitations and performance exception

- The pinned reference does not compare its device result against an independently regenerated host
  result; its "reference" is the same chunk-local scan evaluated sequentially on the CPU. This port
  keeps that exact scheme but recomputes the comparison from the same input.
- The reference's `--report` CSV mode is not ported; the reference's default types, chunk sizes and
  input generator are.
- The OpenMP reference is not a usable CPU baseline on the measurement host (see above), so CPU has
  no runtime ratio.
- The measurement host was a shared-machine fallback (Quadro GV100, sm_70) because the assigned
  `rosi_v100` target was unreachable during this run. Results are GPU-model specific.
