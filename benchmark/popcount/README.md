# popcount Benchmark

Portable [alpaka](https://github.com/alpaka-group/alpaka) implementation of the HeCBench
`popcount` benchmark. One C++20 source file is compiled unchanged for the host CPU and for CUDA
GPUs; there is no backend-specific kernel and no native CUDA/HIP/OpenMP code in this repository.

## What it computes

For every element of a 64 bit integer array the kernel returns the number of set bits:

```
output[i] = popcount(input[i])     // 0 .. 64
```

The HeCBench reference contains six equivalent implementations of the same operation (SWAR bit
tricks, Kernighan's `x &= x - 1` loop, a 256 entry lookup table and the hardware `__popcll`
instruction) to compare their throughput. This port benchmarks the operation through the alpaka
intrinsic abstraction `alpaka::popcount`. On the host it lowers to `std::popcount` and on NVIDIA
hardware to the native `__popcll`, i.e. the fastest of the reference variants. Using the alpaka
abstraction keeps exactly one portable algorithm for every backend.

## Reference and provenance

The implementation is a faithful adaptation of HeCBench at commit
[`0a17ee47ca6072192101036d4bf292e6612f22e6`](https://github.com/ORNL/HeCBench/tree/0a17ee47ca6072192101036d4bf292e6612f22e6):

* [`src/popcount-cuda/main.cu`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/popcount-cuda/main.cu) (SHA `65c16b9ead0a525e3fd303710c422b7eea8047f5`)
* [`src/popcount-cuda/Makefile`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/popcount-cuda/Makefile) (SHA `3186f2bef84ad9cf947a90c0d9e250ac37f26803`)
* [`src/popcount-omp/main.cpp`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/popcount-omp/main.cpp) (SHA `a5e2c49bc85c853d3f81ac20b2a89bb85abf7799`)
* [`src/popcount-omp/Makefile`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/popcount-omp/Makefile) (SHA `86bc4254cfd0676085a4db121a8aab3e367e7850`)

The referenced CUDA/OpenMP sources are **not** vendored here.

Data generation matches the reference: `srand(2)` and two `rand()` draws per element, the first
shifted into the high 32 bits and OR-ed with the second, i.e. `data[i] = ((uint64_t)rand() << 32) | rand()`.
This reproduces the reference input bit for bit.

## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--size <count>` | `16777216` | Number of elements |
| `--repeat <n>` | `100` | Number of measured iterations |

The defaults match the HeCBench `run` target (`./main 16777216 1000` has a longer repetition count;
`--repeat` is exposed so a run can be kept short). One warm-up iteration runs first and is
discarded; `--repeat` measured iterations follow and the reported value is the **median** of those
samples.

## Validation

The host reference and the checker live in `src/helper.hpp`. `popcountReference` is the exact
Kernighan loop of the HeCBench reference `popcount_ref`. The device output is compared **bit-exact**
against it; because popcount is an exact integer operation no floating point tolerance applies.
Every result must also lie in `[0, 64]`, which rejects any corrupt value. The check prints `PASSED`
or `FAILED` and a failure makes the process return non-zero.

## Build and run

The benchmark is registered in `benchmark/CMakeLists.txt` and is built with the rest of the alpaka
benchmarks.

Host CPU (GCC):

```bash
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target popcount_alpaka -j
./build/rel-host-cpu-gcc/benchmark/popcount/popcount_alpaka --size 16777216 --repeat 5
```

CUDA (nvcc + GCC), `sm_70` for a V100:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target popcount_alpaka -j
./build/rel-cuda-nvcc-gcc/benchmark/popcount/popcount_alpaka --size 16777216 --repeat 5
```

`popcount_alpaka` runs every enabled backend/executor and reports each one separately.

The HeCBench references are built outside this repository, at the pinned commit:

```bash
# CUDA reference, V100
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 src/popcount-cuda/main.cu -o popcount_cuda_ref
./popcount_cuda_ref 16777216 5

# OpenMP host reference
g++ -O2 -fopenmp -std=c++17 src/popcount-omp/main.cpp -o popcount_omp_ref
OMP_NUM_THREADS=<n> ./popcount_omp_ref 16777216 5
```

The OpenMP reference is written for the OpenMP offload model (`#pragma omp target teams distribute
parallel for thread_limit(256)`). Without a device the pragmas execute on the host.

## Timing scope

Only the kernel enqueue plus the device synchronization is timed (`queue.enqueue()` +
`onHost::wait()`); host/device copies, allocation and verification are excluded. The queue is
blocking to keep launch overhead representative for host devices. The reported number is the median
of the measured iterations after one discarded warm-up.

## Limitations

* The reference prints a separate timing for each of its six implementations. This port benchmarks
  only the hardware/intrinsic equivalent (`pc6`); the other reference variants exist to compare
  software strategies and are not reproduced.
* The OpenMP reference is written for offload and its host fallback oversubscribes the node, so the
  CPU ratio is indicative rather than like-for-like.
* `--repeat` defaults to 100 for parity with the reference; use at least 5 samples and prefer more
  for stable medians.
