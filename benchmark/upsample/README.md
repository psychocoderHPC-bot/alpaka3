# upsample Benchmark

Portable alpaka port of the HeCBench `upsample` benchmark. It implements the 2x nearest-neighbour
upsampling layer of a small U-Net: a forward pass that replicates every input sample into a 2x2 output
tile, and a backward pass that sums the four `dout` values of each 2x2 tile. One source drives the host
CPU backend and every device backend; there is no native CUDA/HIP/OpenMP kernel and no backend branch.

## What it computes

The pinned HeCBench reference (`src/upsample-cuda/main.cu`, OpenMP twin `src/upsample-omp/main.cpp`)
operates on an `N, C, H, W` tensor (batch, channel, height, width) with row-major contiguous storage.
It defines two passes with two kernel formulations each:

- forward: `upsample_forward_kernel` (flat, one thread per input element) and
  `upsample_forward_kernel2` (2D tile kernel). Both replicate the input element `(b, c, i, j)` into the
  four output positions `(2i, 2j)`, `(2i, 2j+1)`, `(2i+1, 2j)`, `(2i+1, 2j+1)` of the doubled image.
- backward: `upsample_backward_kernel` (flat) and `upsample_backward_kernel2` (2D). Both set the
  gradient `dx[b, c, i, j]` to the sum of the four `dout` values of the same 2x2 tile.

All four kernels compute the same two mathematical operations; they differ only in how the work is
tiled. This port keeps the flat one-thread-per-element formulation, which is the base variant the
reference benchmarks first (`upsample_forward1` / `upsample_backward1`). The index decomposition is the
same as the reference flat kernel, and the accumulation order of the backward sum is the reference's
`dout_sum += ...` order `(((v00 + v01) + v10) + v11)`, so a correct port reproduces the reference
single-precision arithmetic exactly. There is one portable kernel per pass and no backend branch.

The reference sources are kept outside this repository and are not vendored.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA: `src/upsample-cuda/main.cu` (blob `c1ada8e559123592890ffa4a862d012473b85161`),
    `src/upsample-cuda/reference.h` (blob `ebdebbe9af8a2d4dbe9750f80ed95f1da0b724fb`),
    `src/upsample-cuda/common.h` (blob `93f77134e4869c8ec8003a106614f7a88058bfa3`).
  - OpenMP: `src/upsample-omp/main.cpp` (blob `42320cf804cbc763e1caf8dcc7772e4db64514b4`).
  - HeCBench README entry: line 1803, "### upsample (cuda) -- Nearest interpolation of samples
    (https://github.com/clu0/unet.cu)".
- Original: [`clu0/unet.cu`](https://github.com/clu0/unet.cu) `train_unet.cu`, `upsample_forward_kernel`
  and `upsample_backward_kernel` (github.com/clu0/unet.cu, `train_unet.cu` lines 361 and 407 at the
  time of writing). The repository carries no explicit license file; it is reproduced here only as a
  provenance pointer and is not vendored.
- Upstream HeCBench license: BSD-3-Clause (Copyright (c) 2020 Argonne National Laboratory (Zheming
  Jin); Copyright (c) 2020- Oak Ridge National Laboratory (Zheming Jin)).
- The implementation files in this directory are `BSD-3-Clause`.

## Parameters, modes, precision

- `upsample_alpaka --batch N --channels C --height H --width W --repeat R --samples S
  [--seed S] [--verify-only]`.
- `--batch` (`N`) default `32`, `--channels` (`C`) default `64`, `--height` (`H`) default `64`,
  `--width` (`W`) default `64`, matching the reference `run` target (`32 64 64 64 100`).
- `--repeat` default `100`: passes per measured sample, matching the reference `<repeat>` argument.
- `--samples` default `5`: independent measured samples per pass.
- `--seed` default `0`: seed of the internal deterministic generator. The reference calls `srand(0)`
  and fills its buffers with `rand()/RAND_MAX*2-1` in `[-1, 1)`; that generator is not thread-safe and
  depends on the OpenMP thread count, so this port uses a deterministic xorshift64* generator with the
  same range.
- `--verify-only`: run the correctness check without the timing loop.
- Single precision, matching the reference. There are no other precision or mode options.

## Build and run

The measurements in `results.json` were taken on the `fwk394` host (`Quadro GV100`, Volta `sm_70`,
CUDA 12.5, g++ 13.3). The orchestration-assigned `rosi_v100` was unreachable during this task
(`computeMCP` connection failure for that target, the documented `B-ROSI` blocker), so `fwk394` was
used as the fallback and is the host the numbers below belong to.

```bash
# CPU (rel-host-cpu-gcc, g++ 13.3, OpenMP)
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build build/rel-host-cpu-gcc --target upsample_alpaka -j
OMP_NUM_THREADS=28 ./build/rel-host-cpu-gcc/benchmark/upsample/upsample_alpaka \
  --batch 32 --channels 64 --height 64 --width 64 --repeat 100 --samples 5

# CUDA (rel-cuda-nvcc-gcc, nvcc 12.5, sm_70)
cmake --preset rel-cuda-nvcc-gcc -Dalpaka_COMPILE_PEDANTIC=ON -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build/rel-cuda-nvcc-gcc --target upsample_alpaka -j
CUDA_VISIBLE_DEVICES=1 ./build/rel-cuda-nvcc-gcc/benchmark/upsample/upsample_alpaka \
  --batch 32 --channels 64 --height 64 --width 64 --repeat 100 --samples 5
```

The reference binaries are built outside this repository:

```bash
# CUDA reference (pinned HeCBench sources)
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 main.cu -o upsample_cuda_ref
# CPU reference: the HeCBench omp main uses `#pragma omp target` and cannot run under g++ -fopenmp
# (the target region is a no-op and leaves the device buffers zero). The timed CPU baseline uses the
# pinned host implementation from reference.h (`upsample_forward_reference` / `upsample_backward_reference`).
g++ -O3 -fopenmp -std=c++17 -I. cpu_ref.cpp -o upsample_cpu_ref
```

The reference prints per-block-size timings; the acceptance comparison uses the reference
`block_size 256` forward and backward lines, which match this port's effective 256-thread mapping.

## How self-validation works

Two trusted references are computed on the host from the same `N, C, H, W` definition:

- `upsampleBench::forwardReference` / `backwardReference`: single-precision references in the
  reference's exact accumulation order. The accelerator result is compared against these; because both
  operations are a fixed-order sum of at most four values, a correct port matches them bit-for-bit and
  the measured relative-L2 is exactly zero.
- `upsampleBench::forwardOracleDouble` / `backwardOracleDouble`: independent double-precision oracles
  that do not share the single-precision accumulation order. `verifyReferenceAgreement` pins the
  single-precision references to within fp32 rounding (global relative-L2 `< 1e-6`; measured
  `0` forward, `3.6e-8` backward). This guards against a shared bug in two single-precision
  reimplementations.

`verifyForward` compares the forward output tile-by-tile with a per-tile relative-L2 metric
`sqrt(sum_tile(delta^2) / sum_tile(ref^2))`; `verifyBackward` uses `|got - ref| / |ref|` per element.
Both reject any NaN/Inf with a nonzero exit. The per-tile metric cannot hide a single wrong tile inside
a large tensor the way one global norm can. The acceptance threshold is `1e-8`.

Tolerance rationale: both passes are exact fixed-order summations of at most four fp32 values, so a
correct portable port matches the single-precision reference bit-for-bit (measured rel-L2 `0.0` on both
CPU and GPU). `1e-8` is below one fp32 unit roundoff (`~6e-8`) and therefore leaves essentially no room
for a wrong tile mapping, border rule or offset to pass, while tolerating only a backend that reorders
the four-element sum (not observed here). It was not widened to hide a discrepancy.

Boundary and irregular sizes are exercised: `H=1/W=1`, `W=1` with `H>1`, odd `H`/`W` (`33x31`), and
`5x7`, in addition to the reference `64x64`. Both CPU (`CpuOmpBlocks`) and CUDA pass all of them with
rel-L2 `0.0`.

## Timing scope

The reported metric is the per-pass time: each measured sample runs `repeat` launches end-to-end on a
non-blocking queue and divides the wall time by `repeat`, matching the reference `benchmark_kernel`
scope. The host reference/oracle construction and the validation run are outside the timed region.
`forward` and `backward` are timed and reported separately. Median, min, max and relative spread
`(max-min)/median` over `samples` independent samples are printed; the acceptance metric is the median.
Raw samples are printed as `raw_samples_us=`.

## Limitations / performance exception

- GPU (primary target): no performance exception. At the reference size the CUDA forward and backward
  passes match the reference `block_size 256` time (forward ratio ~0.99, backward ~1.00); at `128x128`
  the ratios are ~1.00. The portable flat kernel is the same work decomposition the reference
  benchmarks first (`upsample_forward1` / `upsample_backward1`).
- CPU performance exception (ratio >1.10): the portable flat kernel decomposes each element's flat
  index with two integer divisions, while the reference CPU baseline uses hand-written nested loops
  whose indices are loop counters and therefore does no division. Measured interleaved at the reference
  size with `repeat=20`: forward ~5.64 ms vs reference ~4.04 ms (ratio ~1.40), backward ~6.15 ms vs
  reference ~2.64 ms (ratio ~2.33).
  - Investigation: a row-based variant (one work item per row, inner contiguous loop over `W`, no
    inner divisions) is correct and removes the CPU division cost, but collapses GPU parallelism (it
    measured ~13 ms forward on the GV100, ~40x slower). A 2D/3D `makeIdxMap` range that would carry the
    index without division is not reliable on the host executor: the host realizes one worker per
    block (`backends/host-cpu-execution.md`), and the 2D/3D variants produced wrong results and a heap
    corruption. The one-portable-kernel rule forbids a backend-specific branch, so the division-free
    variants were rejected. The single flat kernel is retained because it is bit-exact and matches the
    reference on the GPU, which is the benchmark's target.
- `rosi_v100` (CUDA 12.6, 4x V100-32GB) was unreachable; the CUDA evidence is from `fwk394`
  (`Quadro GV100`, CUDA 12.5, `sm_70`, `CUDA_VISIBLE_DEVICES=1`).
