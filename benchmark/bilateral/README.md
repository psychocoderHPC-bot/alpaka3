# Bilateral Filter Benchmark

The benchmark applies a **bilateral image filter** to a single-channel `float` image. For every output
pixel `(idx, idy)` it accumulates a normalised weighted average over a `(2R+1) x (2R+1)` window
centred on the pixel:

```
range   = -(I - I_w)^2 / (2 * variance_I)                              // range (intensity) kernel
spatial = -((idk-idx)^2 + (idl-idy)^2) / (2 * variance_spatial)        // spatial (domain) kernel
weight  = a_square * expf(spatial + range)                             // a_square = 0.5f / (variance_I * pi)
out     = sum(I_w * weight) / sum(weight)
```

The window is gathered with the reference's "mirror edges" index folding. The pinned reference runs
window radii `R = 3, 6, 9` on a 2960x1440 image with `variance_I = variance_spatial = 0.5`.

This is a portable, single-source [alpaka3](https://github.com/alpaka-group/alpaka3) port. One kernel
source drives the host CPU backend and every device backend; only the launch configuration changes.
There is no native CUDA/HIP kernel and no backend-specific algorithm branch. The HeCBench reference
sources are **not vendored** and are kept outside this repository.

## Provenance and attribution

- HeCBench reference: <https://github.com/ORNL/HeCBench> at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6` (pinned).
  - CUDA: `src/bilateral-cuda/main.cu` (blob `10b605eeb0262b40f02a35adcea6f75cd6fd565d`), with the
    device/host reference `src/bilateral-cuda/reference.h` (blob
    `b3310f1e9b1ea3f066dcbf62bfaef0f579fa714a`).
  - OpenMP: `src/bilateral-omp/main.cpp` (blob `dcc95252e8e01a523591b5948a00a509e0687682`).
  - HeCBench `README` entry: "bilateral" under algorithm benchmarks.
- Original algorithm: bilateral filter, <https://en.wikipedia.org/wiki/Bilateral_filter>.
- HeCBench license is the repository's own (see the HeCBench `LICENSE`); the reference is used only as
  an executable oracle and is not redistributed here.
- alpaka3 implementation: BSD-3-Clause (see the file headers).

## Layout

```
benchmark/bilateral/
  CMakeLists.txt        target bilateral_alpaka
  README.md             this file
  src/main_alpaka.cpp   portable kernel + driver (single source, CPU + GPU)
  src/helper.hpp        input generator (glibc-compatible), double host reference, verify()
  src/misc.hpp          command-line parsing
```

## Input reproduction

The reference generates its input with `srand(123); for i: src[i] = rand() % 256;`. To reproduce the
exact pinned input on any host libc, `helper.hpp` reimplements glibc's `random()` (`TYPE_3`, degree
31, separation 3) as `GlibcRandom`. It was cross-checked against the system `rand()` for the first
4096 outputs (exact match). This lets the unmodified pinned reference binary be rebuilt outside the
repository and compared against the port.

## Build and run

Prerequisites: a C++20 compiler and CMake >= 3.25; `catch2` is fetched by the top-level benchmark
build. Always pass `-Dalpaka_COMPILE_PEDANTIC=ON` so the CI warning set (`-Werror`) is reproduced.

CPU (host, OpenMP blocking executor):

```bash
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target bilateral_alpaka -j
OMP_NUM_THREADS=$(nproc) ./build/rel-host-cpu-gcc/benchmark/bilateral/bilateral_alpaka \
    --width 2960 --height 1440 --repeat 1000 --samples 5
```

CUDA (single source, sm_70):

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target bilateral_alpaka -j
./build/rel-cuda-nvcc-gcc/benchmark/bilateral/bilateral_alpaka \
    --width 2960 --height 1440 --repeat 1000 --samples 5
```

## Parameters, modes and precision

| Option | Default | Meaning |
| --- | --- | --- |
| `--width` | 2960 | image width in pixels |
| `--height` | 1440 | image height in pixels |
| `--intensity` | 0.5 | range-kernel variance (`variance_I`) |
| `--spatial` | 0.5 | spatial-kernel variance (`variance_spatial`) |
| `--repeat` | 1000 | inner filter passes per measured sample |
| `--samples` | 5 | independent measured samples |
| `--radius` | -1 | window radius; `-1` runs the reference set {3, 6, 9} |
| `--verify-only` | off | skip timing, run the correctness check only |

Compute precision is single (`float`), as in the reference. Accumulation order is the same as the
reference (window loops in `i` then `j`, `normalization` and `res` updated per tap), so the device
result reproduces the reference's own single-precision result closely.

## Self-validation

The port computes an **independent double-precision host reference** (`bilateralReference<R>` in
`helper.hpp`) from the same input and compares the accelerator output against it after a fresh run.

- Explicit NaN/Inf check on every accelerator output value; a non-finite value or a zero/NaN
  normalisation in the host reference fails the run with a message.
- The comparison tolerance is `1e-3`, the pinned reference harness's own threshold for the identical
  computation. It is justified by the single-precision reduction of up to `(2R+1)^2 = 361` terms
  (R=9) of magnitude up to ~255 divided by a normalisation of order 0.5: the accumulated rounding is
  a few `1e-4`. The observed max absolute difference is <= 9.2e-5 (see `results/`), comfortably inside
  the tolerance; the tolerance is not widened to hide bugs.
- The double host reference was cross-checked against the pinned `reference.h` for R=3,6,9 on
  multiple sizes, agreeing to < 1.2e-4 (R=3,6) and < 1e-3 (R=9); the port's own output agrees with
  the double reference to <= 9.2e-5.
- A nonzero exit code signals failure (the driver returns `EXIT_FAILURE` on the first mismatch).

Exercised sizes include odd and non-square dimensions and boundary cases (`4x4`, `4x6`, `6x4`,
`10x11`, `11x10`, `20x21`, `33x17`, `64x48`, `1920x1080`, `2960x1440`), each at the reference radii.
The reference folding is only well defined for `width, height >= R+1`; the driver rejects smaller
images.

## Timing scope and acceptance

Timing is the **kernel/compute time only**: `--repeat` filter passes are enqueued back to back and
bracketed by device synchronisation, divided by `--repeat`. Input generation, host-reference
computation and the final validation run are outside the timed region. One warm-up pass runs before
the measured samples. Five independent samples are taken; the reported median is the acceptance
metric and the variation is `(max - min) / median`.

Runtime ratio = median_alpaka / median_reference, where the reference is the pinned HeCBench binary
(CUDA reference built with `nvcc -O3 -arch=sm_70`; OpenMP reference built with `g++ -O2 -fopenmp`).
Target: <= 1.00 match, <= 1.10 accepted, > 1.10 requires investigation.

## Reference reproduction (outside this repository)

```bash
# CUDA reference (pinned source), sm_70
nvcc -std=c++17 -Xcompiler -fopenmp -Xcompiler -Wall -arch=sm_70 -O3 main.cu -o bilateral_cuda_ref
CUDA_VISIBLE_DEVICES=1 ./bilateral_cuda_ref 2960 1440 0.5 0.5 <repeat>

# OpenMP reference (pinned source)
g++ -O2 -fopenmp -std=c++17 main.cpp -o bilateral_omp_ref
OMP_NUM_THREADS=$(nproc) ./bilateral_omp_ref 2960 1440 0.5 0.5 <repeat>
```

The OpenMP reference uses `#pragma omp target ...`, i.e. it is written for device offload; a plain
`g++ -fopenmp` build ignores the target directives and executes the loop serially on the host. It is
still a valid CPU reference for correctness and for a serial CPU timing point.

## Limitations and performance exceptions

- The reference index folding is undefined for images smaller than the window. The driver enforces
  `width, height >= R+1` and exits on violation.
- The reference's strict correctness target is the pinned harness's `1e-3`; no attempt is made to
  reproduce the reference's exact floating-point bit pattern, which is neither required nor portable.
- The GPU reference and the alpaka CUDA backend may differ in the last rounding bit; the float
  comparison and its tolerance are unchanged from the pinned harness.
- On hosts without a device backend the driver prints the error and exits non-zero; the CPU preset is
  used for the host validation.
