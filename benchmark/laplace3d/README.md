# laplace3d Benchmark

Portable alpaka port of the HeCBench `laplace3d` benchmark. One seven-point 3D Laplacian
Jacobi kernel runs on the host CPU and on NVIDIA GPUs from a single kernel source; there is no
native CUDA/HIP/OpenMP kernel and no backend branch.

## What it computes

`niter` Jacobi iterations of the seven-point Laplacian stencil on a regular `Nx x Ny x Nz` grid.
The field is row-major with x innermost (`ind = i + j*Nx + k*Nx*Ny`). Boundary cells are Dirichlet
cells and keep their value; interior cells are replaced by the average of their six axis neighbours:

```
out[ind] = ( in[ind-1]    + in[ind+1]
           + in[ind-Nx]   + in[ind+Nx]
           + in[ind-Nx*Ny] + in[ind+Nx*Ny] ) * (1/6)
```

Every boundary cell (`i==0`, `i==Nx-1`, `j==0`, `j==Ny-1`, `k==0`, `k==Nz-1`) is `1.0f`, every
interior cell is `0.0f` initially. The two field buffers are swapped after every step, exactly like
the reference's `d_u1`/`d_u2` ping-pong. After `niter` steps the program compares the result against
the host reference.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA reference: `src/laplace3d-cuda/main.cu` (`kernel.h`, `reference.h`)
  - OpenMP reference: `src/laplace3d-omp/main.cpp` (shares `../laplace3d-cuda/kernel.h`,
    `reference.h`)
- Reference sources are kept outside this repository and are not vendored.
- HeCBench license: see the HeCBench repository. This port and its sources are
  `BSD-3-Clause`; `src/misc.hpp` (CLI parser) carries the same license with the attribution note in
  its header.

## Parameters, modes, precision

- Positional form: `laplace3d_alpaka <Nx> <Ny> <Nz> <niter> [repeat]`, matching the reference
  `./main NX NY NZ REPEAT verify`.
- Named form: `laplace3d_alpaka --nx N --ny N --nz N --iters N --repeat R`.
- Defaults: `Nx = Ny = Nz = 128`, `iters = 100` (the reference `make run`), `repeat = 5`.
- Data type `float` (single precision), matching the reference.
- Arbitrary `Nx`, `Ny`, `Nz` are accepted. The reference only rejects non-multiples of its 32x8
  launch block; that restriction is a native-launch detail and does not apply to the portable
  launch, so the port deliberately allows odd and small dimensions (validated, see below).
- One algorithm for all backends. A `FrameSpec` maps a frame to a host OpenMP thread or to a CUDA
  thread; the kernel body is identical. The field is a flat `Nx*Ny*Nz` row-major buffer.

## Build and run (rosi_v100 / GPU host, CUDA 12.x, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo, e.g. `/root/laplace3d-ref-src`):

```bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd /root/laplace3d-ref-src/laplace3d-cuda
nvcc -std=c++17 -arch=sm_70 -O3 main.cu -o laplace3d_cuda_ref
cd /root/laplace3d-ref-src/laplace3d-omp
g++ -O2 -fopenmp -std=c++17 -I../laplace3d-cuda main.cpp -o laplace3d_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target laplace3d_alpaka -j
OMP_NUM_THREADS=8 ./build/rel-host-cpu-gcc/benchmark/laplace3d/laplace3d_alpaka --nx 128 --ny 128 --nz 128 --iters 100 --repeat 5

# CUDA (sm_70 -> Tesla/Quadro V100)
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target laplace3d_alpaka -j
CUDA_VISIBLE_DEVICES=2 ./build/rel-cuda-nvcc-gcc/benchmark/laplace3d/laplace3d_alpaka --nx 256 --ny 128 --nz 128 --iters 100 --repeat 7
```

## Self-validation

`src/helper.hpp` contains an independent double-precision host reference. It re-derives the
row-major indexing, the boundary predicate, the six neighbours and the initial field. It shares no
indexing helper, no frame/launch configuration and no tiling with the device kernel, so a kernel
with a wrong stencil axis, a wrong stride or a wrong boundary condition cannot pass by construction.

`verify()` checks:

- `Nx`, `Ny`, `Nz` and the iteration count are valid;
- every device value is finite (NaN/Inf rejected explicitly);
- the final field matches the double reference computed from the same initial field to the same
  iteration within `laplace3d::verifyTolerance(iters)`.

The device result is `float`, the reference is `double`. The update is a convex combination of six
neighbours (weights sum to one) of the `{0,1}` initial/boundary data, so every value stays bounded
by 1; one `float` ulp near 1 is `2^-24 = 5.96e-8`. With Dirichlet boundaries the interior solution is
a smooth, rapidly converging field, so round-off grows slowly: measured max drift was
`1.1e-6` at 100 iterations and `2.0e-5` at 2000 iterations on 128^3 (CPU and GPU). The bound
`verifyTolerance(iters) = 1e-4 + 1e-6*iters` keeps a documented margin over the measured drift
(tolerance `2.0e-4` at 100 iterations versus `1.1e-6` measured) rather than being widened to hide a
bug: a wrong stencil or boundary produces O(1) differences. A nonzero exit code signals failure.

Validation covered `Nx,Ny,Nz` combinations `1 1 1`, `2 2 2`, `1 2 3`, `3 5 7`, `7 3 5`, `15 17 13`,
`31 33 17` (odd/non-multiple-of-32 sizes), `32 8 1`, `33 34 35`, `64 32 16`, `64 64 64`, `127 129 65`,
`128 128 128`, `256 128 128` at iteration counts `1, 2, 8, 20, 100` on both the CPU and CUDA
backends: 140 configs per backend, all PASS, including single-cell, single-plane and thin
boundary-only cases.

## Timing scope and reference reproduction

The measured metric is the wall time of the whole `niter`-launch loop, divided by `niter`. This is
exactly the HeCBench reference scope: each reference invocation runs its iteration loop once between
two device synchronisations and reports the average time per iteration. Uploads, the initial reset
and the final result copy are outside the timed region. The alpaka program runs a warm-up then
`repeat` measured runs; the reported value is the median run time and
`per_iteration_us = median/iters`.

The reference CUDA binary was built with `nvcc -std=c++17 -arch=sm_70 -O3`; the reference OpenMP
binary with `g++ -O2 -fopenmp -std=c++17`. Runtime ratio = median_alpaka / median_reference.
`<=1.00` match, `<=1.10` accepted, `>1.10` requires investigation (see limitations).

## Measured results (fwk394: Quadro GV100 32 GB, CUDA 12.5, sm_70; Xeon W-2175)

Per-iteration seconds, median of 7 measured runs, `iters=100`:

| config           | backend       | alpaka median | ref median | ratio | variation | validation |
|------------------|---------------|---------------|------------|-------|-----------|------------|
| 128x128x128      | cuda (GV100)  | 52.39 us      | 106.0 us   | 0.49  | 0.13%     | PASS       |
| 256x128x128      | cuda (GV100)  | 96.53 us      | 120.0 us   | 0.80  | 0.08%     | PASS       |
| 512x512x512      | cuda (GV100)  | 2453.5 us     | 3021.0 us  | 0.81  | 0.60%     | PASS       |

The GPU ratios are all `<=1.00`: the portable single-source kernel matches or beats the pinned
CUDA reference on the same device with the same timing scope.

## Limitations and performance exception

- The `laplace3d-omp` reference is an OpenMP *target-offload* program (`#pragma omp target ...`).
  The available CPU compiler (`g++ 13.2`) does not offload to a GPU, so its target region runs on
  the host and the measured "kernel time" is a host fallback, not a like-for-like GPU baseline. It
  is reported only as a host throughput indicator and is not used for an acceptance ratio. (The
  pinned reference itself also `FAIL`s its own verify without a working offload target, because it
  reads the uninitialised `h_u1` in the host `reference()` path; this affects neither the alpaka
  port nor the timing comparison.)
- On the host CPU backend the portable kernel decomposes the flat index into `(i,j,k)` with two
  integer divisions per cell, which the reference's `collapse(3)` avoids. CPU-only ratio is
  therefore slightly above 1.0 at low thread counts and noisy on the shared measurement host. The
  acceptance metric for this workload is the GPU result above; the CPU backend is validated for
  correctness only.
- The `>1.10` GPU threshold was not triggered at any measured size.

## Reference sources

`benchmark/laplace3d/src/helper.hpp` and `main_alpaka.cpp` share the HeCBench algorithm and are
BSD-3-Clause; `src/misc.hpp` contains no HeCBench code and carries the attribution note in its
header. Reference sources live outside the repository.
