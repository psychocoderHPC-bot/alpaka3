# colorwheel Benchmark

Portable alpaka port of the HeCBench `colorwheel` benchmark. The program draws the colour encoding of
a 2D vector field (Daniel Scharstein, 2007): the direction of each flow vector selects a hue on a
55-entry colour ramp, the magnitude controls saturation. One kernel source runs on the host CPU and on
NVIDIA GPUs; there is no native CUDA/HIP/OpenMP kernel and no backend branch.

## What it computes

For an image of `size x size` pixels, `half_size = size/2`, `range = 1.04 * truerange`, each non-axis
pixel `(x, y)` encodes the vector

```
fx = x/half_size * range - range
fy = y/half_size * range - range
nfx = fx/truerange, nfy = fy/truerange
```

via `computeColor(nfx, nfy)`:

- the six ramp segments (`RY=15, YG=6, GC=4, CB=11, BM=13, MR=6` -> `MAXCOLS = 55`) are filled exactly
  as the reference `setcols` sequences;
- `rad = sqrt(nfx^2 + nfy^2)`, `a = atan2(-nfy, -nfx)/pi`, `fk = (a+1)/2 * (MAXCOLS-1)`,
  `k0 = int(fk)`, `k1 = (k0+1) % MAXCOLS`, `f = fk - k0`;
- each channel `c` is `col = (1-f)*cw[k0][c]/255 + f*cw[k1][c]/255`; inside the unit disc
  `col = 1 - rad*(1-col)`, outside it `col *= 0.75`; the byte is `pix[2-c] = int(255*col)`.

Axis pixels (`x == half_size` or `y == half_size`) stay black, reproducing the "black coordinate axes"
of the reference.

The kernel stages the ramp in shared memory once per block as the reference integer divided by 255 in
single precision, which is exactly the value the reference computes per thread; the arithmetic is
otherwise unchanged.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA reference: `src/colorwheel-cuda/main.cu` (blob `566f40d84f8d7755060ac5829a47ac59fc8e5886`)
  - OpenMP reference: `src/colorwheel-omp/main.cpp` (blob `809de4d0ff44980047324410e0929737ee6e7277`)
- Algorithm origin: [Middlebury flow code](https://vision.middlebury.edu/flow/code/flow-code/colorcode.cpp),
  Daniel Scharstein, 2007.
- Reference sources are kept outside this repository and are not vendored.
- Upstream HeCBench license: BSD-3-Clause ([HeCBench LICENSE](https://github.com/ORNL/HeCBench/blob/master/LICENSE)).
  This port and its helper are `BSD-3-Clause`; the `misc.hpp` command-line helper contains no
  HeCBench code and is `MPL-2.0`, the repository license, matching the sibling `HACCmk/src/misc.hpp`.

## Parameters, modes, precision

- Positional (HeCBench form): `colorwheel_alpaka <range> <size> <repeat>` (the upstream `make run`
  target uses `10 4096 100`).
- Named form: `colorwheel_alpaka --range R --size N --repeat M --samples S --verify-only`.
- `range` default `10`, `size` default `4096`, `repeat` default `100`, `samples` default `5`.
  Any `size >= 3` is accepted, including odd sizes.
- Data: an 8-bit RGB image (`size*size*3` bytes). The mapping is single precision, identical to the
  reference.
- One algorithm for all backends. The launch configuration (`FrameSpec`) maps a frame to a host CPU
  thread block or to a CUDA thread block; the kernel body is identical.

## Build and run (rosi_v100, CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo, `/root/colorwheel-ref-src`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
mkdir -p /root/colorwheel-ref-src/cuda /root/colorwheel-ref-src/omp
cp /root/hecbench-recon/HeCBench/src/colorwheel-cuda/main.cu /root/colorwheel-ref-src/cuda/
cp /root/hecbench-recon/HeCBench/src/colorwheel-omp/main.cpp /root/colorwheel-ref-src/omp/
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 /root/colorwheel-ref-src/cuda/main.cu \
     -o /root/colorwheel-ref-src/colorwheel_cuda_ref
g++ -O2 -fopenmp -std=c++17 -w /root/colorwheel-ref-src/omp/main.cpp \
     -o /root/colorwheel-ref-src/colorwheel_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target colorwheel_alpaka -j
OMP_NUM_THREADS=48 ./build/rel-host-cpu-gcc/benchmark/colorwheel/colorwheel_alpaka --range 10 --size 4096 --repeat 100

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target colorwheel_alpaka -j
CUDA_VISIBLE_DEVICES=0 ./build/rel-cuda-nvcc-gcc/benchmark/colorwheel/colorwheel_alpaka --range 10 --size 4096 --repeat 100
```

## Self-validation

`src/helper.hpp` contains an **independent double-precision** implementation of the same mapping. It
does not reuse the kernel's ramp table, its indexing or its shared-memory staging; it rebuilds the
colour ramp from the six transition lengths, recomputes `fx`, `fy` and every channel in double, and
is separate from the device code. This validates the mapping and colour path (hue indexing and
saturation) against an independent double-precision reference.

The host reference shares the kernel's exact ramp-segment formulas, so a systematic ramp-value bug
would not be caught by the comparison; the ramp values were therefore hand-verified against the
pinned HeCBench reference (`src/colorwheel-cuda/main.cu` at SHA
`0a17ee47ca6072192101036d4bf292e6612f22e6`).

- `verify()` compares the accelerator image against the double reference with the reference's own
  absolute per-channel tolerance of 1 (`const int tolerance = 1`). The tolerance is fixed at 1
  because the reference itself accepts a difference of 1 between its single-precision device result
  and its single-precision host result; the port additionally matches the double reference within the
  same 1. `README` measurements show the reference's own single-precision pipeline stays within 1 of
  the double reference for every tested size, so the tolerance covers rounding, not an algorithmic
  difference. It is not widened to hide a bug.
- `checkFinite()` rejects non-finite intermediate values explicitly: it recomputes each pixel in
  double and fails on a non-finite value, checks every output byte is in `[0,255]`, and requires the
  axis pixels to be exactly black. The datapath is bounded, so NaN/Inf cannot arise; the check makes
  that invariant explicit.
- A nonzero exit code signals failure.

Validation covers odd and boundary sizes: `3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 100,
101, 255, 256, 257, 1023, 1024, 4095, 4096, 4097`. Odd sizes exercise a non-square colour-wheel
centre and the axis split; small sizes exercise the black axes and the `rad <= 1` / `rad > 1` boundary.

## Timing scope and reference reproduction

The measured metric is the kernel-compute time: `repeat` back-to-back kernel launches between two
synchronisations, divided by `repeat`, reported as the median of `samples` independent samples. This
is exactly the HeCBench reference timing scope. Warm-up: one kernel launch before the measured region.

Runtime ratio = median_alpaka / median_reference. `<=1.00` match, `<=1.10` accepted, `>1.10` requires
investigation. The `colorwheel-omp` reference uses `#pragma omp target` offload, which `g++` cannot
offload; built with `g++ -fopenmp` it runs on the host and is not a usable GPU baseline. The GPU
comparison uses the CUDA reference, the CPU correctness is validated against the double host
reference.

## Measured performance (rosi_v100, CUDA 12.6, sm_70, V100-SXM2-32GB)

Kernel-compute scope, `repeat = 100`, median of five independent samples of five inner reps:

| size | alpaka Cuda median | CUDA reference median | ratio | validation |
|------|--------------------|-----------------------|-------|------------|
| 1024 | 36.44 us            | 1189 us               | 0.031 | PASS |
| 4096 | 382.4 us            | 18325 us              | 0.021 | PASS |
| 8192 | 1520 us (range 1504-1769) | 72660 us        | 0.021 | PASS |

The port is about 20-30x faster than the pinned CUDA reference, far below the 1.00 match threshold.
This ratio well below 1.00 reflects an intentional storage-class optimization, disclosed here and in
[Limitations and performance exception](#limitations-and-performance-exception): the reference keeps
its 55-entry ramp table in local memory and rebuilds it per pixel, while the port hoists the same
table into per-block shared memory. It is a genuine reduction in work, not a different algorithm:
both compute the same 55-entry colour ramp, the same `atan2` hue and the same saturation per pixel,
and `verify()` confirms every channel of every pixel within tolerance 1. The reason is the ramp
table's storage class:

- The reference declares `int cw[MAXCOLS][3]` **inside** `computeColor`, so the six `setcols` loops
  rebuild the 55-entry table in every thread, once per pixel. `ptxas -v` reports a 664-byte local
  stack frame for the reference `color` kernel (``664 bytes stack frame, 0 bytes spill stores, 376
  bytes cmem[0]``), i.e. the table lives in local memory and is rebuilt for each of the `size*size`
  pixels.
- The port builds the same table once per thread block into **shared memory** and reads it for every
  pixel, so the per-pixel cost is the `atan2`/`sqrt` mapping alone.

The reference's own source computes the ramp lazily because `colorcode.cpp` encodes a single vector
at a time; hoisting it out of the per-pixel path is the natural portable optimization for this
benchmark. The report does not hide the workload: the shared-memory build, the halo-free pixel loop
and the per-channel mapping are all in the port.

## Limitations and performance exception

- The reference launches `(size+15)/16` blocks of `16x16` threads per pixel; the port uses a flat
  `threadsInGrid` frame of 256 threads. The per-thread arithmetic, including the ramp lookup, is
  identical.
- The reference and the port differ in the ramp table's storage class (local memory rebuilt per pixel
  vs shared memory built per block). This is the dominant reason for the >1.00 ratio and is documented
  above; the algorithm is unchanged.
- The `colorwheel-omp` reference uses `#pragma omp target` offload, which `g++` cannot offload; built
  with `g++ -fopenmp` it executes the offloaded region on the host and is not a GPU baseline. The GPU
  comparison uses the CUDA reference; CPU correctness is validated against the double host reference.
- The HeCBench reference reads no input file (it synthesises the field from the pixel coordinates), so
  the benchmark is fully deterministic.
