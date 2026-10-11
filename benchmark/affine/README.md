# affine Benchmark

Portable alpaka port of the HeCBench `affine` benchmark. It applies a fixed 2D affine image
transform (rotate 30 degrees, scale 0.5, translate 0) to a 16-bit grayscale image. One source drives
the host CPU backend and every device backend; there is no native CUDA/HIP/OpenMP kernel and no
backend branch.

## What it computes

For every output pixel `(x, y)` the transform maps back to a source coordinate `(x_new, y_new)`
through the inverse of the fixed forward matrix:

```
affine = [[0.5*cos(30 deg), 0.5*sin(0 deg)],
          [0.5*sin(30 deg), 0.5*cos(0 deg)]]   beta = [0, 0]
```

The result is a bilinear sample of the four surrounding source pixels, truncated to
`unsigned short`. A sample that lands exactly on the last row or column uses the nearest edge pixel;
everything outside the image is filled with `WHITE = 1`. All arithmetic is single precision
(`float`), matching the reference.

The forward coefficients are constant for the whole image. The reference recomputes them per pixel
and its compiler folds the constant `cos`/`sin` calls; this port hoists them to the host and passes
the inverse matrix and zero offset as kernel arguments. The single-precision inverse matrix is
bit-identical to the reference's, so the per-pixel transform is unchanged.

Parallelism is expressed with `onAcc::makeIdxMap` over a 2D `IdxRange{width, height}`, the canonical
device-agnostic alpaka form. The logical frame extents (64x4) are a launch-shape hint; `makeIdxMap`
maps the logical range onto each backend's native thread hierarchy, so the same source drives the
host CPU and CUDA.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA: `src/affine-cuda/main.cu`, `src/affine-cuda/kernel.h`, `src/affine-cuda/reference.h`
    (the kernel is shared by the CUDA, HIP and SYCL variants).
  - OpenMP: `src/affine-omp/main.cpp` (the same arithmetic inline, inside an OpenMP `target` region).
  - HeCBench README entry: "affine (opencl): Affine transformation
    (https://github.com/Xilinx/SDAccel_Examples/tree/master/vision/affine)".
- Original: the Xilinx SDAccel examples, `vision/affine`. The source headers carry
  "Copyright (c) 2017, Xilinx, Inc. All rights reserved." under a 3-clause BSD licence. The
  reference sources are kept outside this repository and are not vendored; only this clean-room
  alpaka reimplementation is committed. The implementation files in this directory are
  `BSD-3-Clause`.

The reference reads an external RAW image (`data/CT-MONO2-16-brain.raw`, 512x512 unsigned short).
This port is self-contained: it generates a deterministic 16-bit grayscale image in `[0, 4095]`
(a typical 12-bit CT range) with a xorshift64* generator (`src/helper.hpp`, `generateImage`).

## Parameters, modes, precision

- `affine_alpaka --width W --height H --repeat R --samples S [--verify-only]`.
- `--width` / `--height` default `512` / `512`, matching the pinned reference `X_SIZE`/`Y_SIZE`.
- `--repeat` default `100`: transform passes per measured sample, comparable to the reference
  positional `<iterations>` argument.
- `--samples` default `5`: independent measured samples.
- `--verify-only`: run the correctness check only, no timing loop.
- The transform (rotation, scale, translation) is fixed by the reference and is not configurable.

Data is 16-bit unsigned integer (`unsigned short`). The coordinate transform and the bilinear
interpolation run in single precision `float`, exactly as the reference; the result is truncated to
`unsigned short`. The comparison against the host reference uses a tolerance of one grey level (see
below).

## Build and run

```bash
# CPU (host backend, g++)
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target affine_alpaka -j
OMP_NUM_THREADS=14 ./build/rel-host-cpu-gcc/benchmark/affine/affine_alpaka --repeat 2000 --samples 5

# CUDA (sm_70, g++ host compiler, nvcc 12.x)
export PATH=/usr/local/cuda/bin:$PATH
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target affine_alpaka -j
CUDA_VISIBLE_DEVICES=1 ./build/rel-cuda-nvcc-gcc/benchmark/affine/affine_alpaka --repeat 100000 --samples 5
```

The reference binaries (outside the repo):

```bash
# CUDA reference (nvcc, sm_70)
mkdir -p refs/affine-cuda && cp <HeCBench>/src/affine-cuda/{main.cu,kernel.h,reference.h} refs/affine-cuda/
nvcc -std=c++17 -arch=sm_70 -O3 refs/affine-cuda/main.cu -o affine_cuda_ref
./affine_cuda_ref input512.raw out.raw 1000000

# Host CPU reference: the OpenMP variant uses an OpenMP `target` offload region that g++ cannot
# compile. The only host compiler available is g++, and icpx/nvc are absent, so the CPU baseline is
# the reference source compiled for the host without the offload pragmas (a plain parallel loop).
g++ -O2 -fopenmp -std=c++17 -Irefs/affine-cuda affine_omp_hostpar.cpp -o affine_omp_hostpar
OMP_NUM_THREADS=14 ./affine_omp_hostpar input512.raw out.raw 2000
```

## Self-validation

Two trusted host references are recomputed from the same deterministic input:

- `affineReference` (`src/helper.hpp`) is a direct translation of the pinned `reference.h` in single
  precision; the accelerator result is compared against it by `verify`.
- `affineReferenceDouble` evaluates the same geometry in double precision, an independent oracle.
  `verifyReferenceAgainstDouble` asserts the two references differ by at most one grey level. On the
  generated images they differ in at most 717 pixels of 262,144 and the maximum channel delta is 1;
  any larger gap is a bug in a reference and fails the check.

`verify` reports the number of differing pixels, the maximum absolute difference and its location,
and returns a nonzero exit code when the maximum exceeds the tolerance. The justified tolerance is
**one grey level**. The result is a truncated integer bilinear sample of a single-precision
transform; a device and a host evaluation can round the transform (in particular the last digit of
the interpolation) slightly differently, which only changes the truncated integer when the
interpolated value is within the rounding error of an integer. The pinned reference harness reports
the same bound: on the pinned input the CUDA reference binary itself prints "Max output error is 1"
against its own host reference. A correctly implemented port therefore cannot differ by more than
one unit, and only at such pixels; any larger difference is a real defect. The tolerance is never
widened to hide bugs.

`checkFinite` re-derives every source sample and the double-precision transform intermediates,
asserts they are finite, asserts the source samples are inside `[0, 65535]`, and asserts every output
value is finite and inside `[0, 65535]`. The finite-input datapath cannot produce NaN or Inf; this
check makes that explicit. A nonzero exit code signals failure.

Validated on both the CPU and CUDA backends over the sizes
`1x1, 2x3, 3x1, 7x5, 15x16, 16x16, 17x19, 31x33, 64x64, 65x63, 127x129, 255x257, 256x256, 257x255,
512x512`, covering 1-pixel, tiny, irregular, odd and non-multiple-of-16 sizes and the image borders.
All verify PASS. On CUDA the observed maximum delta is 1 at 257x255 and 512x512 (matching the
reference's own reported error) and 0 elsewhere.

The CMake test registration runs a fast subset:

```bash
ctest --test-dir build/rel-host-cpu-gcc -R affine_alpaka --output-on-failure
```

## Timing scope and reference reproduction

The measured metric is the per-pass transform time: `repeat` back-to-back enqueues between two
synchronisations, divided by `repeat`. This matches the reference timing, which wraps the kernel
launch loop in `cudaDeviceSynchronize()` on both ends and divides by `<iterations>`. One warm-up pass
precedes the measured samples. A non-blocking queue is used so the measured passes pipeline instead
of synchronising per enqueue. `RESULT` prints the median, min, max and relative spread of the samples
plus every raw sample. Runtime ratio = median_alpaka / median_reference; `<=1.00` match, `<=1.10`
accepted, `>1.10` requires investigation.

Host: **Intel Xeon** class CPU, 14 threads pinned via `OMP_NUM_THREADS=14`. GPU: **Quadro GV100 (SM
7.0)**, CUDA 12.5.82, nvcc driver-built, `CUDA_VISIBLE_DEVICES=1`. Compilers: g++ 13.3 (local CPU
build), g++ 13.2 + nvcc 12.5 (remote CUDA build). Flags:
`-Dalpaka_COMPILE_PEDANTIC=ON` on both, `-DCMAKE_CUDA_ARCHITECTURES=70` for CUDA.

| config | backend | size | alpaka median | ref median | ratio | relspread | validation |
|--------|---------|------|---------------|------------|-------|-----------|------------|
| GPU | Cuda | 512x512 | 4.53 us | 3.73 us | 1.21 | ~0.01 | PASS |
| GPU | Cuda | 2048x2048 | 57.9 us | — | — | 0.01 | PASS |
| GPU | Cuda | 257x255 | 3.92 us | — | — | 0.25 | PASS |
| CPU | Host (OmpBlocks, 14 threads) | 512x512 | 631 us | 273 us | 2.31 | 0.17 | PASS |
| CPU | Host (OmpBlocks, 14 threads) | 2048x2048 | 17084 us | — | — | 0.17 | PASS |
| CPU | Host (serial) | 512x512 | 1915 us | 735 us | 2.61 | 0.33 | PASS |

The reference is a fixed 512x512 image (`X_SIZE`/`Y_SIZE` are compile-time constants), so the only
like-for-like GPU ratio is at 512x512. For the CPU comparison the host-parallel OpenMP compilation
of the reference source (14 threads) is the like-for-like baseline at 273 us; alpaka's OmpBlocks
executor is 631 us (ratio 2.31, see the exception below). The 735 us figure is the same source
compiled by g++ with the offload pragmas ignored, which runs serially; it is reported for
completeness only.

`results.json` holds the machine-readable measurement rows.

## Limitations and performance exception

- The GPU ratio at the default 512x512 is **1.21**, above the 1.10 bound. The gap is the
  per-element index remapping: `onAcc::makeIdxMap` computes the logical-to-native index mapping for
  every element, which is measurable for a transform this small, while the reference derives its
  index from `blockIdx*blockDim+threadIdx` with no remapping. The kernel runs one element per
  logical frame, and a frame-shape sweep (16x16, 32x8, 64x4, 64x8, 128x2, 256x1, 512x1) showed no
  shape closes the gap; `ncu` kernel-only times are 6.7 us (alpaka) vs 5.4 us (reference). An
  explicit thread-index kernel reaches the reference's speed (3.97 us, ratio 0.97) but is not
  host-portable: the host OpenMP executor (`OmpBlocks`) requires one thread per block and produces
  wrong results with an exact multi-thread `ThreadSpec`. The single-source portability rule therefore
  takes precedence, and the gap is accepted as a documented trade-off. The full investigation is
  recorded in `results.json` under `performance_investigation`.
- The CPU ratio against the host-parallel reference is **2.31**, also above the bound. The reference
  host-parallel loop gets better per-thread work than alpaka's `OmpBlocks` executor for this tiny
  per-pixel transform (the same index-map overhead as on the GPU, plus the host executor's per-block
  dispatch); the measured alpaka OmpBlocks speedup over its own serial executor is only ~3x on 14
  threads. This is a portability trade-off, not a correctness issue.
- The remote GPU host is shared, so wall-clock samples carry occasional outliers; medians are used
  and the raw samples are retained. An earlier uncontended reference run gave a 3.73 us median; under
  contention the same reference ranged 3.8-6.2 us.
- The serial host executor is included by the harness; the OmpBlocks executor is the meaningful CPU
  figure. Pin `OMP_NUM_THREADS` for stable host numbers.
