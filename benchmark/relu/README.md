# relu Benchmark

Portable [alpaka](https://github.com/alpaka-group/alpaka) implementation of the HeCBench `relu`
benchmark. One C++20 source file is compiled unchanged for the host CPU and for CUDA GPUs; there
is no backend-specific kernel and no native CUDA/HIP/OpenMP code in this repository.

## What it computes

The benchmark covers the two kernels of the HeCBench `relu` program, which are derived from
TensorFlow:

* **ReluGrad** (fp16): elementwise `(feature > 0) ? gradient : 0`. Two fp16 elements are processed
  per iteration, mirroring the `half2` fast path of the CUDA reference.
* **Relu** (packed int8x4): `max(value, 0)` applied independently to the four signed bytes of every
  `int32` word, i.e. the vectorized form of a ReLU on 8 bit activations.

## Reference and provenance

The implementation is a faithful adaptation of HeCBench at commit
[`0a17ee47ca6072192101036d4bf292e6612f22e6`](https://github.com/zjin-lcf/HeCBench/tree/0a17ee47ca6072192101036d4bf292e6612f22e6):

* [`src/relu-cuda/main.cu`](https://github.com/zjin-lcf/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/relu-cuda/main.cu) (SHA `3a5d1f330a5733e7f12fab15a9b3d919067bd900`)
* [`src/relu-omp/main.cpp`](https://github.com/zjin-lcf/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/relu-omp/main.cpp) (SHA `b749abbba3f0a40e99e1da7005aa7ed68298acf7`)
* [`src/relu-cuda/reference.h`](https://github.com/zjin-lcf/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/relu-cuda/reference.h) (SHA `371d8e6a6d666e72c21b9d67415db18b8ec1f5c5`)

The referenced CUDA/OpenMP sources are **not** vendored here. They are TensorFlow code licensed
under the Apache License 2.0; the original benchmark is described as "Rectified linear unit"
(<https://github.com/tensorflow>).

Data generation matches the references: `std::mt19937 engine(19937)`,
`std::uniform_real_distribution<float> real_dist(-1.f, 1.f)` for the fp16 inputs (gradient fixed at
1) and `std::uniform_int_distribution<unsigned char> int_dist(0, 255)` for the packed integers.
As in the CUDA reference, the integer distribution is consumed after the same number of fp16 draws,
so the engine state and the generated values are identical.

## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--size <count>` | `10000000` | Number of elements |
| `--repeat <n>` | `100` | Number of measured iterations |

For every kernel one warm-up iteration runs first and is discarded. `--repeat` measured iterations
follow; the reported value is the **median** of those samples.

## Validation

The host reference and the checker live in `src/helper.hpp`:

* ReluGrad is computed in **double precision** on the host from the fp16 inputs. Results are
  compared against the device output with a tolerance of `1e-3`, matching the HeCBench check. With
  features in `(-1, 1)` and a gradient of 1 the exact result is always 0 or 1, so the tolerance is
  not exercised in practice. NaN and Inf in the produced data are rejected explicitly.
* Relu is compared **bit-exact** against the individually recomputed packed-int reference.

Both checks print `PASSED` or `FAILED`; any failure makes the process return non-zero.

### fp16 representation

`_Float16` is not accepted as a CUDA device type by nvcc 12.6 (`Internal Compiler Error
(codegen): "unsupported float variant!"`), therefore the benchmark stores fp16 values as `uint16_t`
bit patterns and converts with small portable `floatToHalf` / `halfToFloat` helpers. The rounding
is round-to-nearest, ties-to-even and was verified to match the host `_Float16` conversion exactly
over 3 million random bit patterns, except that NaN payloads are canonicalized (as CUDA
`__float2half_rn` does).

## Build and run

The benchmark is registered in `benchmark/CMakeLists.txt` and is built with the rest of the alpaka
benchmarks.

Host CPU (GCC):

```bash
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target relu_alpaka -j
./build/rel-host-cpu-gcc/benchmark/relu/relu_alpaka --size 10000000 --repeat 5
```

CUDA (nvcc + GCC), 16-byte aligned allocations, `sm_70` for a V100:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target relu_alpaka -j
./build/rel-cuda-nvcc-gcc/benchmark/relu/relu_alpaka --size 10000000 --repeat 5
```

`relu_alpaka` runs every enabled backend/executor and reports each one separately.

The HeCBench references are built outside this repository, at the pinned commit:

```bash
# CUDA reference, V100
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 src/relu-cuda/main.cu -o relu_cuda_ref
./relu_cuda_ref 10000000 5

# OpenMP host reference
g++ -O2 -fopenmp -std=c++17 -I src/relu-cuda src/relu-omp/main.cpp -o relu_omp_ref
OMP_NUM_THREADS=14 ./relu_omp_ref 10000000 5
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
* The fp16 ReluGrad kernel evaluates the predicate on the raw binary16 lanes with alpaka's SIMD
  facilities (`onAcc::SimdAlgo`), so wide vector loads/stores are issued on every backend. On the GPU
  it now reaches the bandwidth of the vectorized HeCBench `float4` path; see the measured table above.
* `--repeat` defaults to 100 for parity with the reference; use at least 5 samples and prefer more
  for stable medians.

## Reference environment

Measured on this port's CI environment (2026-10-10):

* CPU: 14-core x86_64 GNU/Linux, GCC 13.3.0, `-O3`.
* GPU: NVIDIA Tesla V100-SXM2-32GB (sm_70), CUDA 12.6, GCC 13.2.
