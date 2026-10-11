# reverse Benchmark

Portable [alpaka3](https://github.com/alpaka-group/alpaka3) implementation of the HeCBench
`reverse` workload: an in-place reversal of a dense `int` array on the device, applied a random
number of times and verified `memcmp`-style against the expected identity or reversed sequence.

## What it computes

For an array `d` of `len` elements, one reverse computes

```
d'[i] = d[len - 1 - i]      for i in [0, len)
```

The HeCBench reference draws a count in `[100, 9999]` per test case with
`std::default_random_engine(123)` / `std::uniform_int_distribution<int>(100, 9999)`, launches that
many reverses back to back, then compares the device buffer to `gold_even[i] = i` (even count) or
`gold_odd[i] = len - i - 1` (odd count) with `memcmp` over the whole buffer.

The port runs the same in-place reverse in a single alpaka kernel source. On the device the reverse is
a set of independent mirrored swaps `i <-> len - 1 - i` for `i < len / 2`: exactly one logical worker
owns each swap, so no two workers read or write the same slot and no block or grid synchronization is
needed. For odd `len` the middle element is not touched. This is the same permutation the reference
computes through its shared-memory staging buffer and produces byte-identical results.

## Provenance and attribution

- HeCBench upstream: <https://github.com/ORNL/HeCBench>
- Pinned commit: `0a17ee47ca6072192101036d4bf292e6612f22e6`
- Reference sources (kept outside this repository, not vendored):
  - CUDA: `src/reverse-cuda/main.cu`
  - OpenMP: `src/reverse-omp/main.cpp`
- HeCBench is distributed under its own license; see the upstream repository.

The alpaka port in `src/` is new work licensed under BSD-3-Clause
(`SPDX-License-Identifier: BSD-3-Clause`). The helper files (`helper.hpp`, `misc.hpp`) contain no
HeCBench code and carry the same license.

## Parameters and precisions

- Positional (HeCBench form): `reverse_alpaka <iterations>`.
- Named form: `reverse_alpaka --iterations <N> --length <L> --repeat <R>`.
- `iterations` default `100` (the reference `run` target uses `./main 100`); `length` default `256`
  (the reference `const int len = 256`); `repeat` default `5` (independent measured repetitions).
- Data type `int` (32-bit), matching both references. The reverse performs byte-preserving copies, so
  the comparison is exact.

## Build and run (rosi_v100, CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo, `/home/ubuntu/reverse-ref-src`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
cd /home/ubuntu/reverse-ref-src/reverse-cuda
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 main.cu -o reverse_cuda_ref
cd /home/ubuntu/reverse-ref-src/reverse-omp
g++ -O2 -fopenmp -std=c++17 main.cpp -o reverse_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target reverse_alpaka -j
OMP_NUM_THREADS=8 ./build/rel-host-cpu-gcc/benchmark/reverse/reverse_alpaka --length 256 --iterations 5

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target reverse_alpaka -j
CUDA_VISIBLE_DEVICES=2 ./build/rel-cuda-nvcc-gcc/benchmark/reverse/reverse_alpaka --length 256 --iterations 5
```

## Self-validation

`src/helper.hpp` contains the host reference. `makeGold()` builds the same `gold_even` / `gold_odd`
arrays as the reference, and `verify()` (a) checks every element exactly, (b) performs the reference's
`std::memcmp` over the raw bytes as an independent cross-check, and (c) rejects non-finite values
explicitly. The comparison is exact; the tolerance is zero and is never widened. A nonzero exit code
signals failure.

The benchmark first runs a full warm-up pass with verification enabled (all test cases checked), then
runs `repeat` timed passes. Validation covers the reference length `256` and odd/irregular lengths:
`1, 2, 3, 255, 257, 511, 512, 513, 1000, 1024, 1025`. Both CPU and CUDA backends were validated on
all of these on rosi_v100.

The `ctest` case runs `reverse_alpaka --length 256 --iterations 4 --repeat 2`.

## Timing scope and reference reproduction

The measured metric is the kernel-compute time: within each test case, the `count` back-to-back
kernel launches between two device synchronizations. The host-to-device reset and the device-to-host
copy are outside the timed region. This matches the reference, which wraps the launch loop with
`cudaDeviceSynchronize()` and accumulates the per-case times. The reported value is the sum over all
test cases. Warm-up is one full verified pass before the measured passes; the median of `repeat`
independent passes is reported. Because each pass already aggregates thousands of launches, run the
program several times and take the median of the printed medians for the final report.

Runtime ratio = median_alpaka / median_reference. `<=1.00` match, `<=1.10` accepted, `>1.10`
requires investigation. The queue is a non-blocking in-order queue so that the enqueue loop matches
the reference's stream semantics; a blocking queue would add a host round-trip per launch and inflate
the ratio by about 3.5x.

## Limitations and performance exception

- The `reverse-omp` reference is an OpenMP *target-offload* program (`#pragma omp target teams
  num_teams(1) thread_limit(len)` plus an inner `#pragma omp barrier`). It is designed for `icpx`,
  which is absent on rosi_v100. Built with `g++ -fopenmp` it falls back to the host, where the
  offload barrier semantics are not preserved and the reference reports `FAIL`. It is therefore not a
  usable CPU performance baseline. CPU correctness is still validated against the host reference; the
  GPU comparison uses the CUDA reference.
- The benchmark is launch-bound: the default configuration enqueues thousands of tiny 256-element
  kernels. The ratio is dominated by launch overhead, which the port keeps deliberately close to the
  reference (same per-case launch loop, one block, `len` threads).
