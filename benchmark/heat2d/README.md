# heat2d Benchmark

Portable alpaka port of the HeCBench `heat2d` benchmark. One explicit 2D heat-equation kernel
runs on the host CPU and on NVIDIA GPUs from a single kernel source; there is no native
CUDA/HIP/OpenMP kernel and no backend branch.

## What it computes

The 2D explicit heat equation on a fully periodic `Lx x Ly` grid. Every cell is updated with a
five-point Laplacian and the two field buffers are swapped after each step:

```
out[y*Lx + x] = norm * in[y*Lx + x]
                + delta * ( in[y*Lx + (x+1)%Lx] + in[y*Lx + (Lx+x-1)%Lx]
                            + in[((y+1)%Ly)*Lx + x] + in[((Ly+y-1)%Ly)*Lx + x] )

delta = sigma / (1 + 4*sigma)
norm  = 1 / (1 + 4*sigma)
sigma = 0.01   (fixed in the reference)
```

Neighbours wrap around in both directions; there is no halo. The initial field is the reference's
`{0,1}` stamp pattern (see below). The program applies `niter` steps and then compares the result
against the host reference.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA reference: `src/heat2d-cuda/main.cu`
  - OpenMP reference: `src/heat2d-omp/main.cpp`
- HeCBench README credits the original code to the Cyprus Research Promotion Foundation (RPF)
  project "GPU Clusterware".
- Reference sources are kept outside this repository and are not vendored.
- Upstream HeCBench license: see the HeCBench repository. This port and its sources are
  `BSD-3-Clause`; `src/misc.hpp` (CLI parser) carries the same license with the attribution note in
  its header.

## Parameters, modes, precision

- Positional form: `heat2d_alpaka <Lx> <Ly> <niter> [repeat]`, matching the reference
  `./main LX LY NITER`.
- Named form: `heat2d_alpaka --lx L --ly L --iters N --repeat R`.
- Defaults: `Lx = Ly = 4096`, `iters = 1000` (the reference `make run`), `repeat = 5`.
- Data type `float` (single precision), matching the reference.
- Arbitrary `Lx`, `Ly` are accepted. The reference only rejects non-multiples of its 16x16 launch
  block; that restriction is a native-launch detail and does not apply to the portable launch, so
  the port deliberately allows odd and small dimensions (validated, see below).
- One algorithm for all backends. A `FrameSpec` maps a frame to a host OpenMP thread block or to a
  CUDA thread block; the kernel body is identical. The field is a flat `Lx*Ly` row-major buffer.

## Build and run (rosi_v100, CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo, e.g. `/root/heat2d-ref-src`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cd /root/heat2d-ref-src/heat2d-cuda
nvcc -std=c++17 -arch=sm_70 -O3 main.cu -o heat2d_cuda_ref
cd /root/heat2d-ref-src/heat2d-omp
g++ -O2 -fopenmp -std=c++17 main.cpp -o heat2d_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target heat2d_alpaka -j
OMP_NUM_THREADS=8 ./build/rel-host-cpu-gcc/benchmark/heat2d/heat2d_alpaka --lx 4096 --ly 4096 --iters 1000 --repeat 5

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target heat2d_alpaka -j
CUDA_VISIBLE_DEVICES=1 ./build/rel-cuda-nvcc-gcc/benchmark/heat2d/heat2d_alpaka --lx 4096 --ly 4096 --iters 1000 --repeat 5
```

## Self-validation

`src/helper.hpp` contains an independent double-precision host reference. It re-derives the
periodic neighbours, the coefficients and the initial field, and applies the plain five-point update
over the whole domain. It shares no indexing, tiling or launch helper with the device kernel, so a
kernel with a wrong neighbour, a wrong wrap or a wrong coefficient cannot pass by construction.

`verify()` checks:

- `Lx`, `Ly` and the iteration count are valid;
- every device value is finite (NaN/Inf rejected explicitly);
- the final field matches the double reference computed from the same initial field to the same
  iteration within `heat2d::verifyTolerance(iters)`.

The device result is `float`, the reference is `double`. The update is a convex combination
(`norm + 4*delta == 1`) of the previous field, so every value stays bounded by the `{0,1}` initial
data and one `float` ulp near 1 is `2^-24 = 5.96e-8`. Single-precision round-off accumulates with the
iteration count at roughly `1e-8` per iteration or less (measured `1.15e-5` at 1000 and `7.0e-5` at
10000 iterations for a 256x256 grid). `verifyTolerance(iters) = 1e-5 + 1e-7*iters` keeps about a
tenfold margin over the measured drift (tolerance `1.1e-4` at 1000 iterations versus `1.15e-5`
measured) rather than widening the bound to hide a bug; at the reference default of 100000 iterations
it reaches the reference's own documented `1e-2` tolerance. A nonzero exit code signals failure.

Validation covered `Lx,Ly` in `{15, 16, 17, 31, 33}` (odd and non-multiple-of-16 sizes that the
reference itself rejects), `{1, 2, 3, 5, 7}` boundary sizes, plus `{64, 128, 256}`, on both the CPU
and CUDA backends.

## Timing scope and reference reproduction

The measured metric is the wall time of the whole `niter`-launch loop, divided by `niter`. This is
exactly the HeCBench reference scope: each reference invocation runs its iteration loop once between
two device synchronisations and reports the average time per iteration. Uploads and the final result
copy are outside the timed region. The alpaka program runs a warm-up then `repeat` measured runs;
the reported value is the median run time, and `per_iteration_us = median/iters`.

The reference CUDA binary was built with `nvcc -std=c++17 -arch=sm_70 -O3`; the reference OpenMP
binary with `g++ -O2 -fopenmp -std=c++17`. Runtime ratio = median_alpaka / median_reference.
`<=1.00` match, `<=1.10` accepted, `>1.10` requires investigation (see limitations).

## Limitations and performance exception

- The `heat2d-omp` reference is an OpenMP *target-offload* program (`#pragma omp target ...`),
  intended for `icpx`, which is absent on rosi_v100. Built with `g++ -fopenmp` it falls back to the
  host; CPU correctness is validated against the double host reference and the GPU comparison uses
  the CUDA reference.
- The reference allocates its initial buffer with `malloc` and fills only the stamped rows/columns,
  leaving the rest uninitialised; its own `PASS` therefore compares two runs that share the same
  garbage. This port zero-initialises first, so the initial field is deterministic and the reference
  math is unchanged.
- The native CUDA reference allows `niter` up to very large values and uses a 1D grid of
  `Lx/16*Ly/16` blocks of 256 threads with `%`/`/` index arithmetic, one thread per cell. The port
  uses the same one-thread-per-cell mapping; the per-iteration cost is dominated by the same
  streaming read/write pattern, so the ratio is expected near 1. Any ratio above 1.10 would need
  investigation.
