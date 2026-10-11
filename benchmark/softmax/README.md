# softmax Benchmark

Portable [alpaka](https://github.com/alpaka-group/alpaka) implementation of the HeCBench `softmax`
benchmark. One C++20 source file is compiled unchanged for the host CPU and for CUDA GPUs; there is
no backend-specific kernel and no native CUDA/HIP/OpenMP code in this repository.

## What it computes

For a row-major tensor of `numSlice` independent rows (slices) of `sliceSize` values, the benchmark
computes the numerically stable softmax per row:

```
m   = max_j x[j]
e_j = exp(x[j] - m)
s   = sum_j e_j
y_j = e_j / s
```

Max subtraction, exponential, sum and divide are applied in exactly this order, matching the HeCBench
references. The measured kernel (`softMax`, implementation 0) assigns one thread to one full row and
walks the row three times (max, exp-sum, normalization). The alpaka kernel uses the same
one-worker-per-row structure, so the scheme, precision, sizes and verification are preserved.
Implementation 1 of the reference (`softMax2`, a warp-cooperative variant) computes identical values
and is not part of the measured baseline.

## Reference and provenance

The implementation is a faithful adaptation of HeCBench at commit
[`0a17ee47ca6072192101036d4bf292e6612f22e6`](https://github.com/ORNL/HeCBench/tree/0a17ee47ca6072192101036d4bf292e6612f22e6):

* `src/softmax-cuda/main.cu` (blob SHA `5bb28c61d0778280f7a5ca9f337bf21d67791c3a`)
* `src/softmax-omp/main.cpp` (blob SHA `6c813ab25fa0772bfebbc920f8807b40a05d5b18`)

The referenced CUDA/OpenMP sources are **not** vendored here. HeCBench is distributed under the BSD
3-Clause license (`Copyright (c) 2020 Argonne National Laboratory (Zheming Jin)` and
`Copyright (c) 2020-, Oak Ridge National Laboratory (Zheming Jin)`). The port and the host reference
in `src/helper.hpp` re-derive the standard softmax formulation and do not copy HeCBench code.

Data generation matches the references: `std::srand(2)` and `input[i] = rand() % 13` per element,
so inputs are small non-negative integers and the feeds are reproducible.

## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--size <rows>` | `100000` | Number of independent rows (slices) |
| `--slice-size <n>` | `784` | Number of elements per row |
| `--repeat <n>` | `100` | Number of measured iterations |
| `--dtype <t>` | `float` | `float`, `double` or `both` (the reference is FP32) |

The defaults `--size 100000 --slice-size 784 --repeat 100` are the HeCBench test arguments
(`benchmarks.yaml`: `args: ["100000", "784", "0", "100"]`). The reference's third argument selects the
implementation; the measured baseline is implementation 0.

For every kernel one warm-up iteration runs first and is discarded. `--repeat` measured iterations
follow; the reported value is the **median** of those samples.

## Validation

The host reference and the checker live in `src/helper.hpp`:

* The reference is an **independent** row-wise softmax computed in **double precision**: the FP32
  inputs are widened to `double` before the max/exp/sum/divide, so the reference is not limited by the
  single-precision rounding of the tested kernel. The checker re-derives the same double reference
  directly from the stored inputs.
* Results are compared with an absolute tolerance of `1e-3` for FP32, matching the HeCBench check
  `fabsf(output_cpu[i] - output_gpu[i]) > 1e-3`. The reference touches every element of the input,
  so the independently computed reference shares no code path with the kernel. A small relative term
  (`1e-4 * |ref|`) is added so near-zero normalized entries are still compared meaningfully; the
  tolerance is never widened beyond the HeCBench check.
* Non-finite (NaN/Inf) output samples are rejected explicitly and counted as failures.
* FP64 uses `1e-9`.
* Both checks print `Verification PASSED` or `Verification FAILED`; any failure makes the process
  return non-zero.

The checker is exercised with irregular sizes (`1x1`, `1x2`, `2x1`, `3x7`, `5x13`, `7x3`, `17x31`,
`101x3`, `255x5`, `256x257`, `257x256`, `4096x129`), a row length that is not a multiple of the
256-worker block, and the production size `100000x784`, on both the host CPU and the CUDA GPU.

## Build and run

The benchmark is registered in `benchmark/CMakeLists.txt` and is built with the rest of the alpaka
benchmarks.

Host CPU (GCC):

```bash
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target softmax_alpaka -j
./build/rel-host-cpu-gcc/benchmark/softmax/softmax_alpaka --size 100000 --slice-size 784 --repeat 5
```

CUDA GPU (nvcc, e.g. sm_70):

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake --preset rel-cuda-nvcc-gcc -Dalpaka_COMPILE_PEDANTIC=ON -DCMAKE_CUDA_ARCHITECTURES=70 .
cmake --build build/rel-cuda-nvcc-gcc --target softmax_alpaka -j
CUDA_VISIBLE_DEVICES=0 ./build/rel-cuda-nvcc-gcc/benchmark/softmax/softmax_alpaka \
    --size 100000 --slice-size 784 --repeat 100
```

## Timing scope

Only the kernel enqueue and the device synchronization are timed. Allocation, host/device copies and
the reference computation are outside the timed region. The acceptance metric is the median kernel
runtime in microseconds reported by `softmax kernel runtime median: <x> us`; `--repeat` raw samples
are printed as `iteration <i>: <x> us`.

## Reference reproduction

The HeCBench CUDA reference (`softMax`, implementation 0) was compiled outside this repository with
`nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 --use_fast_math` and run with
`./softmax_cuda_ref 100000 784 0 100`. The OpenMP reference (`src/softmax-omp/main.cpp`) was compiled
with `g++ -O2 -fopenmp -std=c++17 -ffast-math`; it requires an OpenMP-offload toolchain to run on a
GPU and is reported as a host build only.

## Limitations

* The alpaka port targets the reference's measured naive scheme (one worker per row, two `exp`
  evaluations per output). The reference's warp-cooperative `softMax2` variant is not ported.
* On the tested Tesla V100 (sm_70) the alpaka kernel runs at a median ratio of 0.96x for the
  production shape `100000x784`, 1.00x for `10000x1024`, 1.00x for `100000x100` and 0.76x for
  `1000000x16` relative to the naive CUDA reference (kernel-only timing, 3 independent runs of
  `--repeat 100`, `CUDA_VISIBLE_DEVICES=0`). The exact numbers, raw samples and the reference
  provenance are recorded under the run results directory outside this repository.
