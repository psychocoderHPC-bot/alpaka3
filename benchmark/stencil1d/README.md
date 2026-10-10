# stencil1d Benchmark

Portable alpaka port of the HeCBench `stencil1d` benchmark. One radius-7 1D sliding-sum stencil
with a per-block shared-memory window. A single kernel source runs on the host CPU and on NVIDIA GPUs;
there is no native CUDA/HIP/OpenMP kernel and no backend branch.

## What it computes

For `length` output elements and input `a[i] = i`:

```
out[i] = sum_{k=-RADIUS}^{RADIUS} a[i+k]     with RADIUS = 7
```

Positions before the domain are zero. Positions in `[length, length + RADIUS)` continue the padded
sequence; positions at or beyond `length + RADIUS` are zero. Each thread block stages a
`BLOCK_SIZE + 2*RADIUS` window (`BLOCK_SIZE = 256`) in shared memory and fills the left and right
halo, then sums 15 neighbours per element.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA reference: `src/stencil1d-cuda/stencil_1d.cu` (blob `06a00dbd679946f5785fceba51246b7000ce9802`)
  - OpenMP reference: `src/stencil1d-omp/stencil_1d.cpp` (blob `46affbf33919c5302aa80b208346358f77bfdbaa`)
- Algorithm origin: [NVIDIA CUDA shared-memory tutorial slides](https://www.olcf.ornl.gov/wp-content/uploads/2019/12/02-CUDA-Shared-Memory.pdf).
- Reference sources are kept outside this repository and are not vendored.
- Upstream HeCBench license: see the HeCBench repository. This file is `BSD-3-Clause`.

## Parameters, modes, precision

- Positional (HeCBench form): `stencil1d_alpaka <length> <repeat>`.
- Named form: `stencil1d_alpaka --length <N> --repeat <R>`.
- `length` default `1 << 20`, `repeat` default `5`. Any positive `length` is accepted, including
  values that are not a multiple of the tile size.
- Data type `int32_t` (the reference uses `int`, i.e. 32-bit). The host reference accumulates in
  `double`.
- One algorithm for all backends. The launch configuration (`FrameSpec`) maps a frame to a host
  CPU thread block or to a CUDA thread block; the kernel body is identical.

## Build and run (rosi_v100, CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo, `/home/ubuntu/stencil1d-ref-src`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
cd /home/ubuntu/stencil1d-ref-src/stencil1d-cuda
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 stencil_1d.cu -o stencil1d_cuda_ref
cd /home/ubuntu/stencil1d-ref-src/stencil1d-omp
g++ -O2 -fopenmp -std=c++17 stencil_1d.cpp -o stencil1d_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target stencil1d_alpaka -j
OMP_NUM_THREADS=48 ./build/rel-host-cpu-gcc/benchmark/stencil1d/stencil1d_alpaka --length 1048576 --repeat 5

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target stencil1d_alpaka -j
CUDA_VISIBLE_DEVICES=0 ./build/rel-cuda-nvcc-gcc/benchmark/stencil1d/stencil1d_alpaka --length 1048576 --repeat 5
```

## Shared-memory window layout

The kernel stages a `tileSize + 2*RADIUS = 256 + 14 = 270` element window:
indices `[0, RADIUS)` are the left halo, `[RADIUS, RADIUS + tileSize)` the centre elements and
`[RADIUS + tileSize, tileSize + 2*RADIUS)` the right halo. Threads `localId < RADIUS` load the halo:

- left halo `temp[localId] = in[blockStart + localId - RADIUS]`, zero when the index is before the
  domain (block 0 only);
- right halo `temp[localId + RADIUS + tileSize] = in[blockStart + RADIUS + tileSize + localId]`, zero
  when the index reaches the padded input end;
- centre `temp[localId + RADIUS] = in[blockStart + localId]`, zero past the padded input.

Every one of the 270 entries is written, so no entry is ever uninitialised.

## Self-validation

`src/helper.hpp` contains a double-precision host reference. It is an **independent oracle**: it sums
the window `[i - RADIUS, i + RADIUS]` by reading the original padded input array element by element,
not by reusing the shared-memory window or its indexing, and not from the generation rule `a[i] = i`.
A wrong halo load or wrong boundary handling therefore cannot pass; it would have to reproduce the
correct value computed from the input independently. The oracle sanity was checked against a
deliberately broken halo scheme: the broken output fails at the first block boundary (index 249) and
the correct output passes.

Because the arithmetic is integer addition, the reference is exact, so `verify()` compares bit-exactly
(tolerance 0). The tolerance is not widened; a mismatch is a real bug. `verify()` rejects non-finite
values explicitly (NaN/Inf check).

A nonzero exit code signals failure. Validation covers a small fast case and representative sizes,
including odd/irregular lengths that are not multiples of the tile size (boundary correctness):
`1, 2, 7, 8, 255, 256, 257, 1024`, plus the default `1 << 20` and larger. Both the CPU and the CUDA
backend were validated on all of these.

## Timing scope and reference reproduction

The measured metric is the kernel-compute time: `repeat` back-to-back kernel launches between two
synchronisations, divided by `repeat`. This is exactly the HeCBench reference timing scope. Separate
end-to-end per-launch time (one launch plus one synchronisation) is also printed for context.

Warm-up: one kernel launch before the measured region. The program prints one value per invocation;
run it at least five times and take the median for the report.

Runtime ratio = median_alpaka / median_reference. `<=1.00` match, `<=1.10` accepted, `>1.10`
requires investigation.

## Limitations and performance exception

- The `stencil1d-omp` reference is an OpenMP *target-offload* program (`#pragma omp target teams`).
  It is meant for `icpx`; that compiler is absent on rosi_v100. Built with `g++ -fopenmp` it falls
  back to the host and is pathologically slow (about 72 s per 1M launch), so it is not a usable
  performance baseline. CPU correctness is still validated against the double host reference; the
  GPU comparison uses the CUDA reference.
- On the CUDA backend the alpaka kernel is about 1.18x the native reference at 1M and 16M elements.
  The reference kernel is `__restrict__`-qualified and compiles to 20 registers / 56 SASS
  instructions, while the generic alpaka wrapper compiles to 23 registers / 80 SASS instructions
  (bounds checks, tensor-index mapping, shared-memory declarations). The extra ~20% is generic-layer
  overhead per tiny kernel, not an algorithmic difference. This is documented as an accepted
  exception; the runtime ratio is reported honestly in the results.
