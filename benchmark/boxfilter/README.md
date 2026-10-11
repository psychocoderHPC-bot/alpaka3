# boxfilter Benchmark

Portable alpaka port of the HeCBench `boxfilter` benchmark. It applies a separable 2D box blur with
radius 10 to an RGBA image: a row pass averages each pixel horizontally, a column pass averages the
intermediate image vertically. A single kernel source drives the host CPU backend and every device
backend; there is no native CUDA/HIP/OpenCL kernel and no backend branch.

## What it computes

The pinned HeCBench reference (`src/boxfilter-cuda/main.cu` and its OpenMP twin
`src/boxfilter-omp/main.cpp`) filters an RGBA image with a separable 2D box blur of radius
`RADIUS = 10`. `rgbaUintToFloat4` unpacks a 32-bit RGBA pixel into four floats, and
`rgbaFloat4ToUint` packs four values back as `(unsigned int)(channel * SCALE)` masked to 8 bits, with
`SCALE = 1 / (2*RADIUS + 1)`. The device runs a `row_kernel` over the image rows and a `col_kernel`
over the columns; the host `BoxFilterHost` (`src/boxfilter-sycl/reference.cpp`) computes the same
result. The harness compares the device output against `BoxFilterHost` exactly.

This port keeps the reference pipeline:

- the same packed 32-bit RGBA layout, decoded and repacked per channel,
- the same row pass: each output pixel is the truncated average of the window
  `[max(0,x-radius), min(width-1,x+radius)]` on its row; pixels beyond a row end contribute zero,
  matching the reference shared-memory padding,
- the same column pass: each output pixel is the truncated average of the clamped window
  `[max(0,y-radius), min(height-1,y+radius)]` in its column (replicate boundary),
- the same quantization `floor(sum / (2*RADIUS+1))`,
- an independent host reference recomputed from the same input and an exact comparison over the ROI
  the HeCBench harness checks, `rows [radius, height - radius)`.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6` (2026-10-09).
  - CUDA reference: `src/boxfilter-cuda/main.cu` (blob `ae1b370fbf9f704527072a5eb4ebed8a80232a1f`)
  - OpenMP reference: `src/boxfilter-omp/main.cpp` (blob `c3809848f97a8c9ca8b99463c7acf19d5973e774`)
  - Host reference: `src/boxfilter-sycl/reference.cpp` (blob `d90b296f0d032d6a245f9a7e8ed2b070d4fb0e58`)
  - PPM loader: `src/boxfilter-sycl/shrUtils.cpp` (blob `4d11068676eccda6a844932c0e11eecce0db4cef`)
    and `src/boxfilter-sycl/shrUtils.h` (blob `43ffd7761eb6c3d838e8cd66679676e2a13b9e93`)
- README entry: HeCBench `README.md`, "boxfilter (cuda/omp/sycl)".
- Reference sources are kept outside this repository and are not vendored.
- The reference CUDA/OpenMP/host sources carry NVIDIA's BSD-3-Clause header
  (Copyright 1993-2010 NVIDIA Corporation).
- Upstream HeCBench is BSD-3-Clause.
- This file and the benchmark sources are `BSD-3-Clause`.

## Parameters, modes, precision

- Named form: `boxfilter_alpaka --width W --height H --radius R --repeat N --samples S`.
- `--width`/`--height` default `1024x1024`, the size of the reference input `lenaRGB.ppm`. Any positive
  size is accepted, including odd and non-square sizes; `height` must be greater than `2*radius` so the
  compared ROI is non-empty.
- `--radius` default `10`, matching the reference `RADIUS`.
- `--repeat` default `100`: back-to-back filter passes per measured sample, comparable to the reference
  positional `<repeat>` argument.
- `--samples` default `5`: independent measured samples.
- `--verify-only` runs the correctness checks without the timing loop.
- Data is an 8-bit RGBA image and the filter datapath is integer; there is no floating point
  accumulation. Result precision is exact.

## Build and run (rosi_v100, CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo, `/root/boxfilter-work/ref`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
# CUDA reference, sm_70 for Tesla V100 (ARCH changed from sm_60 to sm_70)
cd /root/boxfilter-work/ref/boxfilter-cuda
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 -I. -I../boxfilter-sycl \
  main.cu ../boxfilter-sycl/shrUtils.cpp ../boxfilter-sycl/reference.cpp \
  ../boxfilter-sycl/cmd_arg_reader.cpp -o boxfilter_cuda_ref
# OpenMP reference (target-offload variant; see limitations)
cd ../boxfilter-omp
g++ -O2 -fopenmp -std=c++17 -Wall -I../boxfilter-sycl \
  main.cpp ../boxfilter-sycl/shrUtils.cpp ../boxfilter-sycl/reference.cpp \
  ../boxfilter-sycl/cmd_arg_reader.cpp -o boxfilter_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target boxfilter_alpaka -j
OMP_NUM_THREADS=48 ./build/rel-host-cpu-gcc/benchmark/boxfilter/boxfilter_alpaka \
  --width 1920 --height 1080 --radius 10 --repeat 20 --samples 5

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target boxfilter_alpaka -j
CUDA_VISIBLE_DEVICES=0 ./build/rel-cuda-nvcc-gcc/benchmark/boxfilter/boxfilter_alpaka \
  --width 1920 --height 1080 --radius 10 --repeat 100 --samples 5
```

## Self-validation

`src/helper.hpp` recomputes the reference filter on the host in 64-bit integer arithmetic from the
same input image (`boxFilterGold`), then `verify()` compares the accelerator output pixel by pixel
over rows `[radius, height-radius)` with an exact comparison (tolerance 0). The host reference was
independently cross-checked against the pinned HeCBench `BoxFilterHost` on `lenaRGB.ppm` for radii
0, 1, 3, 5, 8, 10: the full-image difference is exactly the `radius` boundary rows at the top and
bottom and zero everywhere else, and the ROI difference is zero, confirming the ROI equality. The
alpha channel of the generated input is zero, so a non-zero output alpha is treated as a bug.
`checkFinite` also re-derives the row sums in double and rejects non-finite values and packed values
outside an 8-bit channel range.

Tolerance rationale: the reference row and column passes both truncate exact integer sums
(`floor(sum / (2r+1))`), so the comparison is exact (tolerance 0). For the pinned radius 10 the
reference's single-precision scale `1.0f/21` agrees with exact integer division `sum / 21` for every
achievable sum `0..255*21`, so the integer formulation reproduces the reference bit for bit. Widening
the tolerance would hide boundary and quantization bugs, so no tolerance is applied. A nonzero exit
code signals failure.

Validated sizes (both CPU and CUDA, all PASS): widths `1, 2, 3, 21, 23, 33, 64, 65, 100, 129, 257,
1080, 1024, 1023, 1025, 1920` against heights `21, 22, 25, 65, 127, 257, 300, 1023, 1024, 1025,
1080, 1920`, including `1x21`, `21x1024`, `1023x1025`, `1025x1023`, `1920x1080`, `1080x1920`.
Validated radii `0, 1, 2, 3, 5, 7, 10` (with appropriate sizes). Heights `<= 2*radius` are rejected
with a nonzero exit.

The CMake test registration runs a fast subset:

```bash
ctest --test-dir build/rel-host-cpu-gcc -R boxfilter_alpaka --output-on-failure
```

## Timing scope and reference reproduction

The measured metric is the per-filter time: `repeat` back-to-back filter passes (row kernel then
column kernel) between two synchronisations, divided by `repeat`. This matches the reference, which
times `<repeat>` iterations and reports the average. One warm-up pass precedes the measured samples.
Runtime ratio = median_alpaka / median_reference; `<=1.00` match, `<=1.10` accepted, `>1.10` requires
investigation.

Measured on a Tesla V100-SXM2-32GB (`CUDA_VISIBLE_DEVICES=0`), 5 independent samples of `--repeat 100`.
The reference is `boxfilter_cuda_ref <ppm> 100`, median of 5 runs at the same size (the reference
defaults to an external PPM of the same dimensions). The CPU reference cannot be built usefully on
rosi_v100 (see limitations), so CPU has no runtime ratio.

| config | backend | alpaka median (us) | ref median (us) | ratio | variation | validation |
|--------|---------|--------------------|-----------------|-------|-----------|------------|
| 1024x1024 | CUDA | 111.42 | 515.32 | 0.22 | 0.18% | PASS |
| 1920x1080 | CUDA | 201.08 | 618.51 | 0.33 | 0.21% | PASS |
| 4096x4096 | CUDA | 1269.98 | n/a | n/a | 0.31% | PASS |
| 1024x1024 | Host/OmpBlocks (48 threads) | 2800 | n/a | n/a | 5.9% | PASS |
| 1920x1080 | Host/OmpBlocks (48 threads) | 5008 | n/a | n/a | 4.5% | PASS |

The 4096x4096 size exceeds the reference's `uiMaxImageWidth/Height = 1920/1080` guard, so no CUDA
reference number exists for it; it is kept as a scaling data point and still self-validates.

Raw per-run samples, exact commands, and CPU/GPU/compiler metadata are in
`../results/boxfilter/boxfilter.csv`.

## Limitations and performance exception

- The pinned reference reads an external PPM (`lenaRGB.ppm`); this port generates a deterministic
  random RGBA image internally so the benchmark is self-contained. The algorithm and its input format
  are unchanged; alpha is zero because `shrLoadPPM4ub` pads the fourth channel with zero.
- For `height < 2*RADIUS + 1` the reference's column boundary loops leave some output rows unwritten
  (a quirk of a degenerate size). This port uses the clamp/replicate formulation, which is
  algebraically identical to the reference for `height >= 2*RADIUS + 1` (the regime the pinned
  benchmark uses). Verification restricts the comparison to rows `[radius, height-radius)`, exactly the
  region the HeCBench harness checks.
- The OpenMP reference (`boxfilter_omp_ref`) is an OpenMP *target-offload* program meant for `icpx`,
  which is absent on rosi_v100. Built with `g++ -fopenmp` it runs on the host, reports a large
  `Average kernel execution time` and `FAIL` because the `omp target` device path is not exercised; it
  is not a usable CPU performance or correctness baseline. CPU correctness is validated against the
  exact host reference; the GPU comparison uses the CUDA reference.
- The HeCBench reference hardcaps image dimensions at 1920x1080; larger sizes are validated here but
  cannot be compared against the reference binary.
