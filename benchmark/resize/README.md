# resize Benchmark

Portable alpaka port of the HeCBench `resize` benchmark. It resizes a stack of single-channel 8-, 
16- and 32-bit images with a nearest-neighbour and a bilinear kernel. One kernel source drives the
host CPU backend and every device backend; there is no native CUDA/HIP/OpenCL kernel and no backend
branch that selects a different algorithm.

## What it computes

The pinned HeCBench reference (`src/resize-cuda/main.cu` and its OpenMP twin `src/resize-omp/main.cpp`)
resizes `num_channels` single-channel images, each `in_height x in_width`, to
`out_height x out_width`. Two kernels are provided:

- `resize` (nearest neighbour): the source coordinate is `lroundf((idx + 0.5f) * o2i)` and is clamped
  to the source extent.
- `resize_bilinear`: the source coordinate is `fmaxf((idx + 0.5f) * o2i - 0.5f, 0)`; the four
  neighbours `v_00, v_01, v_10, v_11` are combined as
  `v_00 + T(fy)*T(v_10-v_00) + T(fx)*T(v_01-v_00) + T(fy)*T(fx)*T(v_11-v_01-v_10+v_00)`.

`resize_image<T>` runs each kernel `repeat` times and prints the average kernel time and an effective
bandwidth.

This port preserves the reference pipeline exactly, including four behaviours that a textbook resize
would get wrong:

1. **Integer-division scaling ratios.** `resize_image` computes `const float fx = in_width / out_width;`
   with integer division, so e.g. `1920/256 == 7` rather than `7.5`. The coordinate mapping therefore
   samples the source on an integer-step grid.
2. **Swapped axis ratios.** `resize_image` passes `(fx, fy)` to the kernel parameters `(o2i_fy, o2i_fx)`;
   the kernel's `o2i_fy` therefore receives `in_width / out_width` and its `o2i_fx` receives
   `in_height / out_height`.
3. **Integer weight cast.** The bilinear weights `T(in_y - in_y0)` and `T(in_x - in_x0)` are cast to the
   pixel type `T`. For the integer pixel types the fractional parts truncate to `0`, so the reference
   bilinear kernel degenerates to `v_00`.
4. **Group-of-eight channel truncation.** Both kernels process `CHANNELS_PER_ITER = 8` channels per work
   item and compute `num_channel_iters_per_xy = num_channels / 8` with integer division. Channels
   `[(num_channels/8)*8, num_channels)` are never written and keep their zero-initialised value.

One kernel template drives the host CPU backend and all device backends; the launch configuration
changes per backend, the algorithm does not.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6` (2026-10-09).
  - CUDA reference: `src/resize-cuda/main.cu` (blob `9caba398942bd6517678f2d9029d4039dc1745cc`)
  - CUDA Makefile flags: `src/resize-cuda/Makefile` (blob `b5f24dacd4f7bf54b4a3332925056ee68ad7f3bb`)
  - OpenMP reference: `src/resize-omp/main.cpp` (blob `94a0f2d105e42ebca67a9af860fda6500e17a2de`)
  - SYCL reference (same algorithm, consulted for the cast semantics):
    `src/resize-sycl/main.cpp` (blob `c6130b64a0ac625817d0e655123bd77d2430f8b4`)
- README entry: HeCBench `README.md`, "resize (cuda)" — "Resize images (https://github.com/opencv/)".
- Reference sources are kept outside this repository and are not vendored.
- HeCBench is `BSD-3-Clause` (Copyright 2020, Argonne National Laboratory, Zheming Jin); the
  per-benchmark `src/resize-cuda/LICENSE` copy is Apache-2.0.
- This file and the benchmark sources are `Apache-2.0`.

## Parameters, modes, precision

- Named form:
  `resize_alpaka --in-width W --in-height H --out-width W2 --out-height H2 --channels C --repeat N --samples S`.
- `--in-width/--in-height` default `1920x1080` and `--out-width/--out-height` default `256x256`, matching
  the HeCBench reference `run` target `1920 1080 256 256 512 100`.
- `--channels` defaults to `512`, matching the reference `run` target.
- `--repeat` default `100`: back-to-back kernel passes per measured sample, comparable to the reference
  positional `<repeat>` argument.
- `--samples` default `5`: independent measured samples.
- `--types` selects the pixel type sizes in bytes to run (`1`, `2`, `4`); default all three. The option
  is repeatable.
- `--nearest` / `--bilinear` select the kernels; both are enabled by default.
- `--verify-only` runs the correctness checks without the timing loop.
- Pixel types are `unsigned char`, `unsigned short` and `unsigned int` (the reference order). The
  datapath is integer for all three; there is no floating point accumulation. Result precision is exact.

## Build and run (fwk394, CUDA 12.5, g++ 13.2, CMake 3.28)

Reference binaries (built outside the repo, `$HOME/resize-work/ref`):

```bash
# CUDA reference, sm_70 for the Quadro GV100
cd $HOME/resize-work/refsrc/resize-cuda
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 main.cu -o $HOME/resize-work/ref/resize_cuda_ref
# OpenMP reference (target-offload variant; see limitations)
cd ../resize-omp
g++ -O2 -fopenmp -std=c++17 -Wall main.cpp -o $HOME/resize-work/ref/resize_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target resize_alpaka -j
OMP_NUM_THREADS=28 ./build/rel-host-cpu-gcc/benchmark/resize/resize_alpaka \
  --in-width 1920 --in-height 1080 --out-width 256 --out-height 256 --channels 512 --repeat 20 --samples 5

# CUDA (sm_70 -> Quadro GV100)
export PATH=/usr/local/cuda-12.5/bin:$PATH CUDA_HOME=/usr/local/cuda-12.5
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target resize_alpaka -j
CUDA_VISIBLE_DEVICES=1 ./build/rel-cuda-nvcc-gcc/benchmark/resize/resize_alpaka \
  --in-width 1920 --in-height 1080 --out-width 256 --out-height 256 --channels 512 --repeat 100 --samples 5
```

## Self-validation

`src/helper.hpp` recomputes the reference resize on the host in the same pixel type `T` from the same
input image: the input is filled with `(i+1) % 13`, the output buffer is zero-initialised (matching the
reference `cudaMemset`), the integer-division ratios and the swapped `(o2i_fy, o2i_fx)` order are
reproduced, and the same expression (including the integer weight cast and the group-of-eight channel
truncation) is evaluated. `verify()` then compares the accelerator output element by element with an
exact comparison (tolerance 0). `checkSane()` makes the non-finite check explicit (relevant only if a
floating point pixel type is instantiated) and a nonzero exit code signals failure.

The host reference was independently cross-checked against the pinned CUDA reference binary on the GPU:
for each pixel type and mode, the device output dumped to a file matched `helper.hpp` element for
element (zero differences) across `1920x1080 -> 256x256`, the upscale `256x256 -> 1920x1080`, odd
`100x100 -> 37x53` and `37x53 -> 100x100`, and the boundary case `13x7 -> 29x31`.

Tolerance rationale: the reference arithmetic is integer for all three pinned pixel types and the host
reference evaluates the identical expression in the same type, so the comparison is exact (tolerance 0).
The single precision coordinate mapping (`lroundf`, `fmaxf`) is evaluated identically on host and
accelerator; for every validated size the float and a double counterpart select the same integer source
coordinates, so the exact comparison does not mask a precision bug. Widening the tolerance would hide
coordinate, cast and channel-truncation bugs, so no tolerance is applied.

Validated sizes (both CPU `rel-host-cpu-gcc` and CUDA `rel-cuda-nvcc-gcc`, all PASS, all three pixel
types and both modes): `1x1`, `1x1 -> 5x7`, `3x5 -> 7x11`, `13x7 -> 29x31`, `100x100 -> 37x53`,
`37x53 -> 100x100`, `257x129 -> 65x33`, `1920x1080 -> 256x256`, `1920x1080 -> 1x1`,
`1x1 -> 1920x1080`, `1023x1025 -> 1025x1023`, `4096x4096 -> 17x19`, `2x2 -> 4000x4000`, `7x7 -> 7x7`,
`255x255 -> 1x255`, plus `channels = 1, 3, 8, 16` (exercising the group-of-eight truncation for the
non-multiples). All pass.

The CMake test registration runs a fast subset:

```bash
ctest --test-dir build/rel-host-cpu-gcc -R resize_alpaka --output-on-failure
```

## Timing scope and reference reproduction

The measured metric is the per-kernel-pass time: `repeat` back-to-back kernel passes between two
synchronisations, divided by `repeat`. This matches the reference, which times `<repeat>` launches and
reports the average. One warm-up pass precedes the measured samples. Runtime ratio =
median_alpaka / median_reference; `<=1.00` match, `<=1.10` accepted, `>1.10` requires investigation.

Measured on a Quadro GV100 (`CUDA_VISIBLE_DEVICES=1`), 7 independent interleaved rounds of
`--repeat 100` (the reference and the port are run back to back in each round so both see the same
GPU contention). The reference is `resize_cuda_ref 1920 1080 256 256 512 100`. The host is an AMD
EPYC node (fwk394) and the GPU is shared, so the reference itself shows occasional contention
outliers; the medians below are the acceptance metric. The OpenMP reference cannot be built usefully
on g++ (see limitations), so CPU has no runtime ratio.

| config | mode/type | alpaka median (us) | ref median (us) | ratio | variation | validation |
|--------|-----------|--------------------|-----------------|-------|-----------|------------|
| 1920x1080 -> 256x256 | nearest u8 | 209.70 | 222.15 | 0.94 | 11.4% | PASS |
| 1920x1080 -> 256x256 | nearest u16 | 414.90 | 424.73 | 0.98 | 0.6% | PASS |
| 1920x1080 -> 256x256 | nearest u32 | 825.01 | 834.67 | 0.99 | 0.0% | PASS |
| 1920x1080 -> 256x256 | bilinear u8 | 367.22 | 385.84 | 0.95 | 0.0% | PASS |
| 1920x1080 -> 256x256 | bilinear u16 | 720.62 | 731.79 | 0.98 | 0.0% | PASS |
| 1920x1080 -> 256x256 | bilinear u32 | 1442.30 | 1428.30 | 1.01 | 0.0% | PASS |

All modes and pixel types are within the accepted `<=1.10` band; the only ratio above 1.00 is the
u32 bilinear at 1.01. The large u8-nearest variation comes from two contended rounds (338 us,
2419 us); the uncontended plateau is ~209 us.

Raw per-run samples, exact commands, and CPU/GPU/compiler metadata are in
`../results/resize/resize.csv`.

## Limitations and performance exception

- The OpenMP reference (`resize_omp_ref`) uses `#pragma omp target` with `num_teams(29184)`; it is meant
  for `icpx`, which is absent on fwk394. Built with `g++ -fopenmp` the offload pragma is ignored and the
  loop runs on the host with a pathological thread league, so the binary does not complete usefully. It
  is not a usable CPU performance or correctness baseline. CPU correctness is validated against the
  exact host reference; the GPU comparison uses the CUDA reference.
- The HeCBench reference hard-codes the launch to `<<<29184, 256>>>` with `CHANNELS_PER_ITER = 8`. This
  port keeps the group-of-eight channel decomposition to preserve the channel-truncation behaviour but
  lets alpaka choose the launch configuration, so the measured numbers are for the portable launch.
