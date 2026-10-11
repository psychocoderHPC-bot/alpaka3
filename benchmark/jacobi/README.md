# jacobi Benchmark

Portable alpaka port of the HeCBench `jacobi` benchmark. One five-point Jacobi relaxation step
runs on the host CPU and on NVIDIA GPUs from a single kernel source; there is no native
CUDA/HIP/OpenMP kernel and no backend branch.

## What it computes

The 2D Laplace equation on an `N x N` grid with a five-point Jacobi stencil. Every iteration
updates the interior:

```
f[i][j] = 0.25 * (f_old[i-1][j] + f_old[i+1][j] + f_old[i][j-1] + f_old[i][j+1])   for 1 <= i,j <= N-2
```

The four grid edges hold a fixed sinusoidal boundary condition and the interior starts at zero
(see `initializeData`). The solver iterates until the normalised L2 change per step falls below
`tol`:

```
df        = f_new[i][j] - f_old[i][j]
error     = sqrt( sum(df^2) / (N*N) )
stop when error <= tol   (or iter == max_iters)
```

With `N = 2048` and `tol = 1e-5` the reference converges after 6072 iterations (verified on
rosi_v100); the iteration cap is 10000.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA reference: `src/jacobi-cuda/main.cu`
  - OpenMP reference: `src/jacobi-omp/main.cpp`
- UA Dr. Zheming Jin (ORNL) author credit is recorded in the HeCBench repository README.
- Reference sources are kept outside this repository and are not vendored.
- Upstream HeCBench license: see the HeCBench repository. This file and the benchmark sources are
  `BSD-3-Clause`. The `src/misc.hpp` CLI parser is `BSD-3-Clause` with the attribution note in the
  file header.

## Parameters, modes, precision

- Positional form: `jacobi_alpaka <n> [repeat]`.
- Named form: `jacobi_alpaka --n N --tolerance T --max-iters M --repeat R`.
- Defaults match HeCBench: `n = 2048`, `tolerance = 1e-5`, `max-iters = 10000`, `repeat = 5`.
- Data type `float` (single precision), matching the reference.
- One algorithm for all backends. A `FrameSpec` maps a logical frame to a host OpenMP thread block
  or to a CUDA thread block; the kernel body is identical. The field is a flat `n*n` row-major
  buffer, so linear indexing is independent of the backend's multi-dimensional pitch convention.

## Build and run (rosi_v100, CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo, `/home/ubuntu/jacobi-ref-src`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
cd /home/ubuntu/jacobi-ref-src/jacobi-cuda
nvcc -std=c++17 -arch=sm_70 -O3 main.cu -o jacobi_cuda_ref
cd /home/ubuntu/jacobi-ref-src/jacobi-omp
g++ -O2 -fopenmp -std=c++17 main.cpp -o jacobi_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target jacobi_alpaka -j
OMP_NUM_THREADS=8 ./build/rel-host-cpu-gcc/benchmark/jacobi/jacobi_alpaka --n 2048 --repeat 5

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target jacobi_alpaka -j
CUDA_VISIBLE_DEVICES=2 ./build/rel-cuda-nvcc-gcc/benchmark/jacobi/jacobi_alpaka --n 2048 --repeat 5
```

## Self-validation

`src/helper.hpp` contains an independent double-precision host reference. It re-derives the
sinusoidal boundary from `sin(...)`, applies the plain five-point average over the full domain and
computes the L2 change directly. It shares no indexing, tiling or reduction helper with the device
kernel, so a kernel with a wrong halo, wrong boundary update or wrong reduction cannot pass by
construction.

`verify()` checks:

- the reported iteration count is positive;
- every device value is finite (NaN/Inf rejected explicitly);
- the final field matches the double reference computed to the same iteration within
  `jacobi::verifyTolerance`.

The device result is `float`, the reference is `double`. The update is a convex average of the
previous field and the boundary stays in `[-1, 1]`, so `|u| <= 1` and one `float` ulp near 1 is
`2^-24 = 5.96e-8`. The verified worst-case difference over the validated sizes is a few ulps
(largest observed `7.12e-7` at N=2048); `verifyTolerance = 1e-4` keeps roughly a 140x margin over
the observed residual rather than hiding a bug. A nonzero exit code signals failure.

Validation covered `3, 5, 7, 9, 15, 16, 17, 31, 33, 64, 65, 100, 128, 129, 257` (including sizes
that are not a multiple of the 16x16 tile and odd sizes), plus the default `N = 2048`, on both the
CPU and CUDA backends.

## Timing scope and reference reproduction

The measured metric is the wall time of the whole `while (error > tol && iters < maxIters)` loop,
divided by the iteration count. This is exactly the HeCBench reference scope: each reference
invocation runs the convergence loop once and reports its own average time per iteration. The
initial device upload and the final result copy are outside the timed region. The alpaka program
runs a warm-up solve and then `repeat` measured solves; the reported value is the median per-solve
time, and `per_iteration_us = median / iters`.

Runtime ratio = median_alpaka / median_reference. `<=1.00` match, `<=1.10` accepted, `>1.10`
requires investigation (see limitations).

## Limitations and performance exception

- The `jacobi-omp` reference is an OpenMP *target-offload* program (`#pragma omp target ...`). It
  is meant for `icpx`; that compiler is absent on rosi_v100. Built with `g++ -fopenmp` it falls
  back to the host; CPU correctness is validated against the double host reference, and the GPU
  comparison uses the CUDA reference.
- On the CUDA backend the alpaka kernel matches or slightly beats the native reference per
  iteration (about 69 us vs 74 us, ratio about 0.93). The reference specialises an 18x18
  shared-memory halo tile and a warp-shuffle reduction; the port uses a flat row-major tile with a
  warp-shuffle reduction plus a single shared step across warps, which keeps the same global-memory
  access pattern and a comparable reduction cost. The remaining difference is small and in favour
  of the port.
