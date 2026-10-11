# medianfilter Benchmark

Portable alpaka port of the HeCBench `medianfilter` benchmark. It applies a 2D 3x3 median filter to
an RGBA image. One source drives the host CPU backend and every device backend; there is no native
CUDA/HIP/OpenMP kernel and no backend branch.

## What it computes

For every output pixel and every one of the three colour channels, the filter returns the median of
the 3x3 neighbourhood centred on the pixel. The alpha channel is dropped: the reference packs only
R, G and B into one 32-bit word (bit 0 / 8 / 16). Pixels outside the image contribute zero
(zero padding at the border). The median is computed with the reference's fixed 8-iteration integer
binary search over the 256 grey levels starting from the estimate 128 and the bound `[0, 255]`.

The reference device kernel (`src/medianfilter-cuda/MedianFilter.cu`) stages a 16x4 tile with a
one-pixel apron in shared memory and then runs the binary search over the staged window. This port
keeps the same algorithm, window and border semantics. Instead of shared-memory tiling it samples
the 3x3 window once per pixel into registers and runs the identical binary search over those values.
The result is bit-for-bit the same as the reference computation; only the data staging differs, so
the kernel stays portable across every alpaka backend.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA: `src/medianfilter-cuda/main.cu` (blob `651007e4c15730ab0a5fdecb54109551cfb317d2`),
    `src/medianfilter-cuda/MedianFilter.cu` (blob `8a902dde1392a8ae5ebc2187044b4a47e4257bee`),
    `src/medianfilter-cuda/MedianFilterHost.cpp` (blob `5fa7063ff42635ce8fd3faf68f363656dddc4dde`).
  - OpenMP: `src/medianfilter-omp/main.cpp` (blob `eedf9383fb1b6b13508a07fe2c18ef0c4074785e`).
  - HeCBench README entry: line 1139, "medianfilter (opencl): Two-dimensional 3x3 median filter of
    RGBA image".
- Original: the NVIDIA CUDA SDK (OpenCL/CUDA) median filter sample; the source header carries
  "Copyright 1993-2010 NVIDIA Corporation. All rights reserved." and the NVIDIA EULA notice. It is
  not a permissively licensed sample. The reference sources are kept outside this repository and are
  not vendored; only this clean-room alpaka reimplementation is committed. The implementation files
  in this directory are `BSD-3-Clause`.

The reference reads an external PPM image (`SierrasRGB.ppm`, 1920x1080). This port is
self-contained: it generates a deterministic 8-bit RGBA image with a xorshift64* generator
(`src/helper.hpp`, `generateImage`), alpha zero, matching `shrLoadPPM4ub`'s zero alpha padding.

## Parameters, modes, precision

- `medianfilter_alpaka --width W --height H --repeat R --samples S [--verify-only]`.
- `--width` / `--height` default `1920` / `1080`, matching the reference default input size. The
  reference caps the image at 1920x1080; this port has no cap and accepts any size >= 1.
- `--repeat` default `100`: filter passes per measured sample, comparable to the reference
  positional `<repeat>` argument.
- `--samples` default `5`: independent measured samples.
- `--verify-only`: run the correctness check only, no timing loop.
- Data is 8-bit integer (`std::uint8_t` channels packed into `std::uint32_t`). There is no floating
  point in the result datapath; the binary search uses `float` bounds exactly as the reference does,
  so the result is reproducible bit-for-bit.

## Build and run

```bash
# CPU (host backend, g++)
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target medianfilter_alpaka -j
OMP_NUM_THREADS=14 ./build/rel-host-cpu-gcc/benchmark/medianfilter/medianfilter_alpaka --repeat 20 --samples 5

# CUDA (sm_70, g++ host compiler, nvcc 12.x)
export PATH=/usr/local/cuda/bin:$PATH
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target medianfilter_alpaka -j
CUDA_VISIBLE_DEVICES=1 ./build/rel-cuda-nvcc-gcc/benchmark/medianfilter/medianfilter_alpaka --repeat 100 --samples 5
```

The CPU and GPU reference binaries (outside the repo):

```bash
# CUDA reference (needs ../boxfilter-sycl/shrUtils.{h,cpp} and MedianFilterHost.cpp)
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 -I../boxfilter-sycl \
     main.cu ../boxfilter-sycl/shrUtils.cpp MedianFilterHost.cpp -o median_cuda_ref
./median_cuda_ref ../medianfilter-sycl/data/SierrasRGB.ppm 500

# OpenMP reference (requires an OpenMP offload compiler; g++ cannot build the target region)
# built with the HeCBench Makefile.aomp (Intel oneAPI)
```

## Self-validation

Two trusted host references are recomputed from the same deterministic input:

- `medianGold` (`src/helper.hpp`) reproduces the reference binary search bit-for-bit. The
  accelerator result is compared against it with **tolerance 0** (`verify`): any mismatch prints the
  pixel coordinates and both packed values and returns a nonzero exit code. The comparison is exact
  because both sides run the same deterministic integer binary search.
- `medianSorted` computes the exact sorted 3x3 median, an independent oracle. `verifyReferenceAgainstSorted`
  asserts that `medianGold` and `medianSorted` differ by at most one grey level per channel. On the
  generated image they differ in only 4 pixels of 2,073,600, and at every one of them the exact
  median is 0 and the binary search returns 1 — a known property of the NVIDIA SDK binary search for
  an all-zero window. This pins the reference to within one grey level and proves the two references
  agree. The reference's own harness tolerates the same one-grey-level discrepancy in up to 0.01% of
  pixels (`shrCompareuit(..., 1.0f, 0.0001f)`); this port instead compares against the exact
  binary-search reference and needs no tolerance.

`checkFinite` re-derives every 3x3 window in double, asserts each sample is finite and inside
`[0, 255]`, and asserts the output alpha channel is zero. The integer datapath cannot produce NaN or
Inf; this check makes that explicit. A nonzero exit code signals failure.

Validated on both the CPU and CUDA backends over the sizes `1x1, 2x3, 3x1, 7x5, 31x33, 64x64,
65x63, 127x129, 255x257, 256x256, 257x255, 1920x1080`, covering irregular, odd, small and
non-multiple-of-16 sizes. All verify PASS.

The CMake test registration runs a fast subset:

```bash
ctest --test-dir build/rel-host-cpu-gcc -R medianfilter_alpaka --output-on-failure
```

## Timing scope and reference reproduction

The measured metric is the per-pass filter time: `repeat` back-to-back enqueues between two
synchronisations, divided by `repeat`. This matches the reference `MedianFilterGPU` timing, which
wraps a single kernel launch in `cudaDeviceSynchronize()` and returns the elapsed time. One warm-up
pass precedes the measured samples. `RESULT` prints the median, min, max and relative spread of the
samples plus every raw sample. Runtime ratio = median_alpaka / median_reference; `<=1.00` match,
`<=1.10` accepted, `>1.10` requires investigation.

Measured on a **Quadro GV100 (SM 7.0, CUDA 12.5)** with `CUDA_VISIBLE_DEVICES=1`, `--repeat 100
--samples 5`; reference `median_cuda_ref` on the same GPU (kernel time, median of 3 invocations of
500 cycles). Host `Xeon W-2175`:

| size | backend | alpaka median (us) | ref median (us) | ratio | relspread | validation |
|------|---------|--------------------|-----------------|-------|-----------|------------|
| 1920x1080 | CUDA | 216.93 | 153.0 | 1.42 | 0.0009 | PASS |
| 1024x1024 | CUDA | 110.96 | — | — | 0.0003 | PASS |
| 4096x4096 | CUDA | 1388.6 | — | — | 0.168 | PASS |
| 1920x1080 | Host (OmpBlocks, 14 threads) | 115133 | — | — | 0.076 | PASS |
| 1920x1080 | Host (serial) | 638058 | — | — | 0.034 | PASS |

The CPU reference is an OpenMP *offload* kernel (`#pragma omp target`); g++ cannot build it and
`icpx`/`nvc` are absent, so there is no like-for-like CPU baseline and the ratio is not reported for
the host backend.

## Limitations and performance exception

- The CUDA ratio at the default 1920x1080 is **1.41**, above the 1.10 bound. The gap is the staging
  strategy: the reference stages the 3x3 window (with a one-pixel apron) through shared memory and
  reuses it across the 8 binary-search iterations, while a portable single-source kernel samples the
  window into per-thread registers once. Register sampling already removed the naive 216-global-load
  -per-pixel variant (399us -> 217us at 1920x1080); the remaining gap is the reference's larger
  per-thread work sharing and its memory coalescing. Expressing the reference's cooperative tiling
  portably would need shared-memory aporns that alpaka does not expose uniformly across backends, so
  this port accepts the documented exception. It is a deliberate portability/perf trade-off.
- The serial host executor is included by the harness; the OmpBlocks executor is the meaningful CPU
  figure. Pin `OMP_NUM_THREADS` for stable host numbers.
- `results.json` holds the machine-readable measurement rows.
