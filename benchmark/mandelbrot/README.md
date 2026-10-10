# mandelbrot Benchmark

Portable [alpaka](https://github.com/alpaka-group/alpaka) implementation of the HeCBench
`mandelbrot` benchmark. One C++20 source file is compiled unchanged for the host CPU and for CUDA
GPUs; there is no backend branch and no native CUDA/HIP/OpenMP kernel.

## What it computes

For every pixel of a `rows x cols` image the kernel evaluates the escape-time iteration of the
Mandelbrot set and stores the iteration count at which the orbit leaves the disk of radius 2, or
`max_iterations` when the point is classified as interior:

```
for (i, j) in image:
    c = ( scaleRow(i) , scaleCol(j) )     // scaleRow: -1.5 .. 0.5, scaleCol: -1.0 .. 1.0
    z = 0
    count = 0
    while count < max_iterations and |z|^2 < 4:
        z = z*z + c
        count++
    output[i*cols + j] = count
```

The arithmetic (float `ComplexF`, the `Point` loop, `ScaleRow`, `ScaleCol`) is copied verbatim from
the HeCBench reference; only the loop that walks the image differs between the CUDA, OpenMP, SYCL
and HIP references, and this port replaces that loop with a single portable alpaka kernel.

## Reference and provenance

Faithful adaptation of HeCBench at commit
[`0a17ee47ca6072192101036d4bf292e6612f22e6`](https://github.com/ORNL/HeCBench/tree/0a17ee47ca6072192101036d4bf292e6612f22e6):

* [`src/mandelbrot-cuda/main.cu`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/mandelbrot-cuda/main.cu) (SHA `0735959f663a4ab1f503ca4728aae6e74ad4dd44`)
* [`src/mandelbrot-cuda/mandel.hpp`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/mandelbrot-cuda/mandel.hpp) (SHA `9382be88975fc856e255e0609044b41a75c43739`)
* [`src/mandelbrot-cuda/common.hpp`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/mandelbrot-cuda/common.hpp) (SHA `0a756e491f904507870fe013a8c552d25c55ff88`)
* [`src/mandelbrot-omp/main.cpp`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/mandelbrot-omp/main.cpp) (SHA `81f7b962e4fc5f5ecbb93cd7db24690c18732f68`)
* [`src/mandelbrot-omp/mandel.hpp`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/mandelbrot-omp/mandel.hpp) (SHA `0dea8f7c2d60d31e9d72459102f22c429e89ada2`)
* [`src/mandelbrot-omp/util.hpp`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/mandelbrot-omp/util.hpp) (SHA `01e061ffa0d2503a2c6d8f29c97a2882a6a0f9ed`)

The referenced CUDA/OpenMP sources are **not** vendored here. The exact file SHAs recorded during
reconnaissance are reproduced in `results/mandelbrot/provenance.txt`.

**License / attribution.** The algorithm and the copied parameter struct originate from the HeCBench
project (`src/mandelbrot-cuda/mandel.hpp`, `src/mandelbrot-omp/mandel.hpp`), Copyright (c) 2019
Intel Corporation, SPDX-License-Identifier: MIT. The reference attributes the code to the
[oneAPI-src/oneAPI-samples](https://github.com/oneapi-src/oneAPI-samples) `DirectProgramming`
examples. The alpaka adaptation is Copyright 2026 René Widera, SPDX-License-Identifier: BSD-3-Clause.

## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--rows <n>` | `1024` | Image rows; `ScaleRow` maps `0..rows` to `-1.5..0.5` |
| `--cols <n>` | `1024` | Image columns; `ScaleCol` maps `0..cols` to `-1.0..1.0` |
| `--max-iter <n>` | `1000` | Escape-time iteration limit |
| `--repeat <n>` | `5` | Number of measured repetitions |

The HeCBench `run` target uses `./main 1000`, i.e. 1000 repetitions with the source dimensions. The
source ships `row_size = 1080`, `col_size = 1920`, `max_iterations = 100`; the benchmark protocol
adopts the `1024 x 1024` geometry with `max_iterations = 1000`. Both are supported through the
options above, so the reference geometry can be reproduced with
`--rows 1080 --cols 1920 --max-iter 100`. One warm-up iteration is discarded; the reported number is
the **median** of the measured samples.

## Validation

The host references and the checker live in `src/helper.hpp`, and the result is checked three ways.

1. **Faithful float reference.** `MandelParameters::point` is shared with the kernel, so the host
   CPU and the serial float reference use bit-identical arithmetic; the host CPU run reports
   `0 / N` mismatches for every tested configuration. On the GPU the compiler is free to contract
   `a*b+c` into an FMA, which changes the last bit of the iteration for boundary pixels; the GPU
   therefore differs from the serial float reference only on boundary pixels, again bounded by the
   tolerance below.
2. **Independent double reference.** `pointDouble` re-implements the same grid in `double`
   precision. A defect that also exists in the shared float helper (a wrong scale, a wrong loop
   bound, a wrong update) would be exposed here.
3. **Interior random spot check.** A deterministic set of pixels (seed 12345) is recomputed with the
   `double` reference. Only deep-interior pixels, whose eight neighbours are also interior in
   `double`, are asserted; those points never leave the disk, so their classification does not
   depend on rounding and the check is exact.

Every output value must also be finite (`std::isfinite`) and lie in `[0, maxIterations]`, which
rejects NaN/Inf and any out-of-range count. A failure makes the process return non-zero.

### Tolerance rationale

The escape-time map is chaotic at the parabolic boundary between the interior and the exterior, so a
last-bit precision difference can change when a boundary pixel crosses the escape radius. The
kernel and the faithful float reference do not have this problem (identical arithmetic, exact
agreement). The comparison against the independent `double` reference allows a small fraction of
boundary pixels to differ.

The float/double classification divergence is a deterministic property of the grid, so it is
machine independent. A standalone probe of the same grid gives:

| Geometry | max_iter | mismatched pixels | ratio |
| --- | --- | --- | --- |
| `1024 x 1024` | 1000 | 5288 / 1048576 | 0.0050 |
| `1080 x 1920` | 100 | 2949 / 2073600 | 0.0014 |
| `1080 x 1920` | 1000 | 14145 / 2073600 | 0.0068 |
| `1080 x 1920` | 10000 | 15480 / 2073600 | 0.0075 |
| `100 x 100` | 2000 | 43 / 10000 | 0.0043 |

The benchmark's own runs reproduce these values (host `0.005043` at `1024 x 1024 x 1000` and
`0.001422` at `1080 x 1920 x 100`).

The tolerance is `0.05` (5 %), the same value the HeCBench reference `Verify` uses. It sits about 7x
above the worst measured float/double divergence, which is safely below the toll a genuine defect
would take (a wrong scale, loop bound or update affects entire regions, not isolated boundary
pixels). The mismatch ratio is printed for every run, so a regression that inflates it is visible
even when it stays under the bound. The tolerance is never widened to hide a bug: the exact float
comparison, the finite/range guards and the exact interior spot check are independent hard checks
that must pass on their own.

## Build and run

The benchmark is registered in `benchmark/CMakeLists.txt` and is built with the rest of the alpaka
benchmarks.

Host CPU (GCC):

```bash
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target mandelbrot_alpaka -j
./build/rel-host-cpu-gcc/benchmark/mandelbrot/mandelbrot_alpaka \
    --rows 1024 --cols 1024 --max-iter 1000 --repeat 5
```

CUDA (nvcc + GCC), `sm_70` for a V100:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target mandelbrot_alpaka -j
CUDA_VISIBLE_DEVICES=1 ./build/rel-cuda-nvcc-gcc/benchmark/mandelbrot/mandelbrot_alpaka \
    --rows 1024 --cols 1024 --max-iter 1000 --repeat 5
```

`mandelbrot_alpaka` runs every enabled backend/executor and reports each one separately.

The HeCBench references are built outside this repository, at the pinned commit:

```bash
# CUDA reference, V100
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 src/mandelbrot-cuda/main.cu -o mandelbrot_cuda_ref
CUDA_VISIBLE_DEVICES=1 ./mandelbrot_cuda_ref 1000

# OpenMP host reference
g++ -O2 -fopenmp -std=c++17 src/mandelbrot-omp/main.cpp -o mandelbrot_omp_ref
OMP_NUM_THREADS=<n> ./mandelbrot_omp_ref 1000
```

The OpenMP reference is written for the OpenMP offload model (`#pragma omp target teams distribute
parallel for simd collapse(2) thread_limit(256)`); without a device the pragmas execute on the host.

## Timing scope

Only the kernel enqueue plus the device synchronization is timed (`queue.enqueue()` +
`onHost::wait()`); host/device copies, reference evaluation, allocation and verification are
excluded. The queue is blocking to keep launch overhead representative for host devices. The
reported number is the median of the measured iterations after one discarded warm-up. The reference
program times its own `MandelParallel::Evaluate()` kernel the same way, so the ratio is
like-for-like.

## Limitations

* The HeCBench source is single-precision only; there is no double output phase to reproduce. The
  port keeps `float` exactly.
* The serial `MandelSerial` phase of the reference is used only for validation and is reproduced by
  the host reference here; it is not a timed artifact of this benchmark.
* The `Print()` debug output of the reference (only for images up to 128 pixels per side) is omitted;
  the computed image is validated numerically instead.
