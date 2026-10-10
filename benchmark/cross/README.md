# cross Benchmark

Portable [alpaka](https://github.com/alpaka-group/alpaka) implementation of the HeCBench `cross`
benchmark (3D vector cross product). One C++20 source file is compiled unchanged for the host CPU
and for CUDA GPUs; there is no backend-specific kernel and no native CUDA/HIP/OpenMP code in this
repository.

## What it computes

For two dense tensors `x1` and `x2` with shape `(numRows, 3)`, the benchmark computes the 3D cross
product per row:

```
out[row] = x1[row] x x2[row]
```

with

```
out0 = x1_1 * x2_2 - x1_2 * x2_1
out1 = x1_2 * x2_0 - x1_0 * x2_2
out2 = x1_0 * x2_1 - x1_1 * x2_0
```

This is exactly the `cross_kernel` of the HeCBench reference with unit strides. The `cross2_kernel`
and `cross3_kernel` variants of the reference compute the same values (only the load pattern
differs), so a single portable kernel covers the computation; the reference uses the three variants
solely to check an in-place result against each other.

## Reference and provenance

The implementation is a faithful adaptation of HeCBench at commit
[`0a17ee47ca6072192101036d4bf292e6612f22e6`](https://github.com/zjin-lcf/HeCBench/tree/0a17ee47ca6072192101036d4bf292e6612f22e6):

* `src/cross-cuda/main.cu` (SHA `6a5fa81e4b73b8a233cb3613469d7f1413f79bd0`)
* `src/cross-omp/main.cpp` (SHA `b64dc198b2da54e2fb915278691e808cbc4a7b31`)

The referenced CUDA/OpenMP sources are **not** vendored here. The original benchmark is derived from
the PyTorch definition of `torch.linalg.cross`
(<https://pytorch.org/docs/stable/generated/torch.linalg.cross.html>).

Data generation matches the references: `std::default_random_engine engine(123)` and
`std::uniform_real_distribution<T> distr(-2.f, 2.f)` for both inputs.

## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--size <rows>` | `10000000` | Number of rows, i.e. number of 3D vectors per tensor |
| `--repeat <n>` | `100` | Number of measured iterations |
| `--dtype <t>` | `both` | `float`, `double` or `both` (the reference runs FP32 then FP64) |

For every kernel one warm-up iteration runs first and is discarded. `--repeat` measured iterations
follow; the reported value is the **median** of those samples.

## Validation

The host reference and the checker live in `src/helper.hpp`:

* The reference cross product is computed in **double precision** from the stored inputs (both the
  FP32 and the FP64 case). Results are compared against the device output with an absolute tolerance
  of `1e-3` for FP32 (matching the HeCBench check `fabs(o[i] - o2[i]) > 1e-3`) plus a small relative
  term; FP64 uses `1e-9`. Non-finite (NaN/Inf) output samples are rejected explicitly.
* The reference uses the same inputs it verifies against, so the check is self-contained.

Both checks print `PASSED` or `FAILED`; any failure makes the process return non-zero.

## Build and run

The benchmark is registered in `benchmark/CMakeLists.txt` and is built with the rest of the alpaka
benchmarks.

Host CPU (GCC):

```bash
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target cross_alpaka -j
./build/rel-host-cpu-gcc/benchmark/cross/cross_alpaka --size 10000000 --repeat 5
```

CUDA (nvcc + GCC), `sm_70` for a V100:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target cross_alpaka -j
./build/rel-cuda-nvcc-gcc/benchmark/cross/cross_alpaka --size 10000000 --repeat 5
```

`cross_alpaka` runs every enabled backend/executor and reports each one separately.

The HeCBench references are built outside this repository, at the pinned commit:

```bash
# CUDA reference, V100
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 src/cross-cuda/main.cu -o cross_cuda_ref
./cross_cuda_ref 10000000 5

# OpenMP host reference
g++ -O2 -fopenmp -std=c++17 src/cross-omp/main.cpp -o cross_omp_ref
OMP_NUM_THREADS=14 ./cross_omp_ref 10000000 5
```

## Timing scope

Only the kernel enqueue plus the device synchronization is timed
(`queue.enqueue()` + `onHost::wait()`); host/device copies, allocation and verification are
excluded. The queue is blocking to keep launch overhead representative for host devices.

## Limitations

* The OpenMP HeCBench variant targets the OpenMP offload model (`#pragma omp target ... num_threads(256)`).
  Without a device it runs on the host with 256 threads, which oversubscribes a typical node and
  depresses its throughput. The CPU numbers below are therefore an order-of-magnitude comparison,
  not a like-for-like one.
* The benchmark reports the `cross_kernel` algorithm only; the HeCBench reference's `cross2` and
  `cross3` variants compute identical values.
* `--repeat` defaults to 100 for parity with the reference; use at least 5 samples and prefer more
  for stable medians.
* The HeCBench CUDA reference allocates the host buffers for three intermediate outputs and does not
  check the allocation. At `--size 100000000` the FP64 pass exceeds the container memory limit and
  the reference process dies with SIGSEGV, so the comparison below uses 50M as the larger size.

## Measured performance

Warm-up plus five measured iterations, median, `--dtype both`. The ratio is
`alpaka / reference`; values below 1 mean the alpaka version is faster. The CPU reference is the
OpenMP `cross1`/`cross3` kernel (the reference prints three near-identical variants), the GPU
reference the CUDA `cross1`/`cross3` kernels.

### GPU (V100-SXM2-32GB, sm_70, CUDA 12.6)

| rows | dtype | alpaka GpuCuda us | CUDA ref us | ratio |
| --- | --- | ---: | ---: | ---: |
| 10M | FP32 | 590.1 | 557.3 / 528.8 | 1.06 / 1.12 |
| 10M | FP64 | 1222.5 | 1093.7 / 1101.8 | 1.12 / 1.11 |
| 50M | FP32 | 2776.3 | 2645.3 / 2640.5 | 1.05 / 1.05 |
| 50M | FP64 | 6160.2 | 5407.3 / 5348.2 | 1.14 / 1.15 |

### CPU (14-core x86_64, GCC 13.3.0)

The OpenMP reference oversubscribes 14 cores with `num_threads(256)` in its offload form, so the
CPU ratios are indicative rather than like-for-like.

| rows | dtype | alpaka CpuOmpBlocks us | OMP ref us | ratio |
| --- | --- | ---: | ---: | ---: |
| 10M | FP32 | 7587.2 | 28532.9 / 24478.3 | 0.27 / 0.31 |
| 10M | FP64 | 18079.9 | 33043.4 / 36937.4 | 0.55 / 0.49 |
| 50M | FP32 | 55095.1 | 61579.9 / 59908.3 | 0.89 / 0.92 |
| 50M | FP64 | 143037.5 | 94768.5 / 93516.7 | 1.51 / 1.53 |

## Reference environment

Measured on this port's CI environment (2026-10-10):

* CPU: 14-core x86_64 GNU/Linux, GCC 13.3.0, `-O3`.
* GPU: NVIDIA Tesla V100-SXM2-32GB (sm_70), CUDA 12.6, GCC 13.2.
