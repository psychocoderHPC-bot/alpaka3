# axpby Benchmark

Portable [alpaka3](https://github.com/alpaka-group/alpaka3) implementation of the
HeCBench `axpby` workload: the fused elementwise update `out = a * x + b * y`
over one or more independent dense tensors (multi-tensor apply).

## What it computes

For every tensor `t` and every element `i` the output is

```
out_t[i] = a * static_cast<float>(x_t[i]) + b * static_cast<float>(y_t[i])
```

with the HeCBench defaults `a = 1`, `b = 1`. Each tensor has its own length; a
single kernel launch covers all tensors by flattening the work into chunks of
`chunkSize` elements and assigning one chunk of one tensor to each thread block.
This reproduces the `multi_tensor_apply` dispatch of the HeCBench reference.

## Provenance and attribution

- HeCBench upstream: <https://github.com/ORNL/HeCBench>
- Pinned commit: `0a17ee47ca6072192101036d4bf292e6612f22e6`
- Reference sources: `src/axpby-cuda/main.cu`, `src/axpby-cuda/multi_tensor_apply.cuh`,
  `src/axpby-omp/main.cpp` (kept outside this repository, not vendored).
- HeCBench is distributed under its own license; see the upstream repository.
  The CUDA reference originates from the PyTorch multi-tensor-apply kernels.

The alpaka port in `src/` is new work and is licensed under BSD-3-Clause
(`SPDX-License-Identifier: BSD-3-Clause`). The helper files
(`helper.hpp`, `misc.hpp`) carry the same license and contain no HeCBench code.

## Portable design

One computational source (`src/main_alpaka.cpp`) compiles for the host CPU and
for CUDA GPUs. There is no backend branch and no native CUDA/HIP/OpenMP kernel.
The chunk list is built on the host exactly like the reference launcher, then
passed to the kernel; each block reads its `(tensor, chunk)` entry from
`chunkTensor`/`chunkIndex`, computes its element range from `offsets`/`lengths`,
and threads within the block stride over the chunk.

## Parameters and precisions

| option | default | meaning |
|---|---|---|
| `--tensors` | 1000 | number of independent tensors |
| `--chunk-size` | 512 | elements per chunk (one chunk per block) |
| `--repeat` | 5 | measured iterations (plus one warm-up) |
| `--fixed-length` | 0 | force every tensor length (0 = HeCBench `srand(123)` draw) |

Precision is single (`float`), matching both references. Input generation uses
`srand(123)` and the exact draw order of the reference: per tensor first the
length `rand() % (1024*1024) + 1024`, then the `x` values, then the `y` values.

## Self-validation

The host-side `axpby::verify()` recomputes every element in double precision from
the stored inputs and compares against the accelerator output. Each tensor is
checked independently. Non-finite output fails the check explicitly. Exit code
is non-zero on any failure. The `ctest` case exercises four tensors of odd
length 4093 with chunk size 512.

Tolerance rationale: the inputs are non-negative integers below 2^20 and
`a = b = 1`, so `x + y < 2^24` and the double reference is exact for the default
configuration. The check uses `atol = 1e-3` (the same threshold as the HeCBench
`fabsf(...) > 1e-3` check) plus `rtol = 1e-6`; the relative term only absorbs
compiler contraction differences and is never widened to hide a wrong kernel.

## Build and run

CPU (GCC, OpenMP):

```bash
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON \
      -Dalpaka_TESTS=OFF -Dalpaka_EXAMPLES=OFF -Dalpaka_DOCS=OFF -S . -B build/rel-host-cpu-gcc
cmake --build build/rel-host-cpu-gcc --target axpby_alpaka -j
OMP_NUM_THREADS=14 ./build/rel-host-cpu-gcc/benchmark/axpby/axpby_alpaka \
      --tensors 50 --chunk-size 512 --repeat 5
ctest --test-dir build/rel-host-cpu-gcc -R axpby_alpaka
```

GPU (CUDA 12.6, sm_70):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 \
      -Dalpaka_COMPILE_PEDANTIC=ON -Dalpaka_TESTS=OFF -Dalpaka_EXAMPLES=OFF -Dalpaka_DOCS=OFF \
      -S . -B build/rel-cuda-nvcc-gcc
cmake --build build/rel-cuda-nvcc-gcc --target axpby_alpaka -j 8
CUDA_VISIBLE_DEVICES=2 ./build/rel-cuda-nvcc-gcc/benchmark/axpby/axpby_alpaka \
      --tensors 1000 --chunk-size 512 --repeat 5
```

## Timing scope and reference reproduction

Timing brackets only the kernel enqueue and the device synchronization, matching
the reference `multi_tensor_axpby`, which wraps the launch with
`cudaDeviceSynchronize()`. Allocation and host/device transfers are excluded.
One warm-up launch is discarded; the median of the measured launches is reported.
The acceptance metric is the kernel runtime median.

The HeCBench references are built outside the repository:

```bash
# CUDA (sm_70)
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 main.cu -o axpby_cuda_ref
# OpenMP CPU
g++ -O2 -fopenmp -std=c++17 main.cpp -o axpby_omp_ref
```

## Limitations and performance notes

- The reference only draws lengths up to 1 MiB and launches one block per chunk;
  the default configuration is therefore dominated by launch/dispatch overhead.
  The port reproduces the same dispatch, so the comparison is like-for-like.
- The HeCBench OpenMP reference is an OpenMP-offload program. On a plain host its
  `target teams` fallback serialises work and is far slower than the alpaka
  `CpuOmpBlocks` executor; CPU ratios are indicative rather than like-for-like.
- Only FP32 is exercised, matching both references.
