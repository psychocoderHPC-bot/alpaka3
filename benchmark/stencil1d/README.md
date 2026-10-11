# stencil1d Benchmark

Portable alpaka port of the HeCBench `stencil1d` benchmark. One radius-7 1D sliding-sum stencil
with a per-block shared-memory window. A single kernel source runs on the host CPU and on NVIDIA GPUs;
there is no native CUDA/HIP/OpenMP kernel and no backend branch.

## What it computes

For `length` output elements and a padded input `a` with `length + RADIUS` elements:

```
out[i] = sum_{k=-RADIUS}^{RADIUS} a[i+k]     with RADIUS = 7
```

Positions before the domain and at or beyond the padded buffer `[0, length + RADIUS)` are zero.
Each thread block stages a `BLOCK_SIZE + 2*RADIUS` window (`BLOCK_SIZE = 256`) in shared memory and
fills the left and right halo, then sums 15 neighbours per element.

The input values are a deterministic pseudo-random pattern (`makeInput`), not the monotone ramp
`a[i] = i` used by the upstream harness. The ramp is avoided on purpose: for a symmetric sliding sum
a duplicated or shifted halo reproduces the same sum of consecutive integers, so a ramp can hide an
aliasing index error. The pseudo-random pattern is not shift invariant.

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
- right halo `temp[localId + RADIUS + tileSize] = in[blockStart + localId + tileSize]`, zero
  when the index reaches the padded input end;
- centre `temp[localId + RADIUS] = in[blockStart + localId]`, zero past the padded input.

The right-halo destination is `RADIUS + tileSize + localId`, never the centre `RADIUS + localId`.

Every one of the 270 entries is written, so no entry is ever uninitialised.

## Self-validation

`src/helper.hpp` contains a double-precision host reference. It is an **independent oracle**: it sums
the window `[i - RADIUS, i + RADIUS]` by reading the original padded input array element by element,
not by reusing the shared-memory window or its indexing, and not from the generation rule `a[i] = i`.
A wrong halo load or wrong boundary handling therefore cannot pass; it would have to reproduce the
correct value computed from the input independently.

Two guards back this up.

- The default input is non-trivial (see above). A ramp would let an aliasing error pass; the
  pseudo-random pattern does not.
- `oracleSanityCheck()` runs before every verification. It builds the correct output and a
  deliberately broken output (the right-halo threads writing into the centre region instead of the
  right halo) from the *same* input, and requires the comparison to accept the first and reject the
  second. A comparison that had lost the power to detect an index error would fail this check and the
  program exits nonzero.

Because the arithmetic is integer addition, the reference is exact, so `verify()` compares bit-exactly
(tolerance 0). The tolerance is not widened; a mismatch is a real bug. `verify()` rejects non-finite
values explicitly (NaN/Inf check).

A nonzero exit code signals failure. Validation covers boundary-critical lengths around the tile and
radius: `1, 2, RADIUS-1 (=6), RADIUS (=7), RADIUS+1 (=8), 255, 256, 257, 511, 512, 513`,
`2*BLOCK_SIZE-1/2*BLOCK_SIZE/2*BLOCK_SIZE+1 (=511/512/513)` and `1 << 20`, plus `1 << 20 - 1` and
`1 << 20 + 1`. Both the CPU (`CpuOmpBlocks` and `CpuSerial`) and the CUDA backend pass all of them,
and the oracle sanity check passes on each.

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
- On the CUDA backend the alpaka kernel is about 1.15x the native reference at 1M elements and about
  1.16x at 16M elements. The reference kernel is `__restrict__`-qualified and compiles to 20 registers
  (1080 bytes shared memory), while the generic alpaka wrapper compiles to 25 registers (same
  1080 bytes shared memory): the alpaka launch wrapper adds index mapping and bounds handling.
  The extra ~15% is generic-layer overhead on a kernel that runs in ~20 µs, not an algorithmic
  difference. The ratio is measured on a downclocked idle V100 (SM clock ~135-202 MHz of 1530 MHz)
  and is reported honestly in the results; it stays above the 1.10 acceptance threshold.
