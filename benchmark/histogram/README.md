# histogram Benchmark

Portable alpaka port of the HeCBench `histogram` benchmark. It histograms an internally generated
image of random 8-bit RGBA pixels into 256 bins per active channel with block-private shared-memory
counters, then verifies the result against an independent host reference with an exact integer
comparison. A single kernel source drives the host CPU backend and every device backend; there is no
native CUDA/HIP/OpenCL kernel and no backend branch.

## What it computes

The pinned HeCBench reference (`src/histogram-cuda/histogram_compare.cu` and its OpenMP twin
`src/histogram-omp/histogram_compare_base.cpp`) generates an image of 8-bit RGBA pixels with a
Mersenne Twister (MT19937 seeded by `init_by_array({0x123, 0x234, 0x345, 0x456}, 4)` and the
`RandomBits()` reduction), then computes a per-channel histogram. This port keeps the reference
pipeline exactly:

- the same MT19937 seeding and the same per-channel `RandomBits()` generation,
- the same three genres with 256 bins: 1-channel `uchar1`, 3-channel `uchar4`, and 3-channel `float4`
  in `[0, 1)`,
- the same bucket scheme: 8-bit samples are used directly, float samples `k/256` are scaled by 256 and
  truncated,
- the shared-memory privatised algorithm of `histogram_smem_atomics`: each block owns a padded
  (`+ CHANNEL`) shared counter array, atomically increments it, and writes a per-block partial
  histogram that a second kernel accumulates. This is the fast variant of the reference.

The reference implements both a shared-memory and a global-memory variant; this port reproduces the
shared-memory one, which is the variant the reference reports as fastest.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6` (2026-10-09).
  - CUDA reference: `src/histogram-cuda/histogram_compare.cu` (blob `852ad69eab4fcbf0abc65a1c3157bed89f29dde4`)
  - Shared-memory kernel: `src/histogram-cuda/histogram_smem_atomics.h` (blob `b401fbb6fd56a1bf1479de351486a9c5e8267802`)
  - RNG: `src/histogram-cuda/mersenne.h` (blob `38644513f6df8b48ff8b62f8d4fe614561a108fd`)
  - Harness: `src/histogram-cuda/test_util.h` (blob `5d2f3e604f6ef588b8c82dc8e194f87f67c16abe`)
  - OpenMP reference: `src/histogram-omp/histogram_compare_base.cpp` (blob `7495741104ea702291d641c6b22fd93118b0f6f4`)
- README entry: HeCBench `README.md` line 893, "histogram (cuda)".
- Reference sources are kept outside this repository and are not vendored.
- Upstream HeCBench license: BSD-3-Clause
  (Copyright (c) 2020 Argonne National Laboratory (Zheming Jin); Copyright (c) 2020- Oak Ridge
  National Laboratory (Zheming Jin)).
- The reference CUDA/OpenMP sources carry NVIDIA's BSD-3-Clause header
  (Copyright (c) 2011-2018, NVIDIA CORPORATION; Copyright (c) 2011, Duane Merrill).
- The MT19937 implementation reproduced in `src/helper.hpp` is
  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura, BSD-3-Clause.
- This file is `BSD-3-Clause`.

## Parameters, modes, precision

- Named form: `histogram_alpaka --pixels N --repeat R --samples S --entropy E`.
- `--pixels` default `1920*1080 = 2073600` (the reference `width=1920 height=1080` default). Any
  positive pixel count is accepted, including values not divisible by the block size.
- `--repeat` default `100`, matching the reference `--i=100`: inner histogram computations per
  measured sample.
- `--samples` default `5` independent measured samples per genre.
- `--entropy` default `0`, matching the reference `--entropy=0`. `-1` is the reference's all-zero
  image; positive values AND more MT words per sample.
- `--verify-only` runs the correctness checks without the timing loop.
- Data is exact integer counts; there is no floating point accumulation in the histogram itself. The
  `float4` input samples are checked finite before use.

## Build and run (rosi_v100, CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo, `/root/histogram-ref-src/histogram-ref`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
cd /root/histogram-ref-src/histogram-ref/histogram-cuda
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 histogram_compare.cu -o histogram_cuda_ref
cd ../histogram-omp
g++ -O2 -fopenmp -std=c++17 histogram_compare_base.cpp -o histogram_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target histogram_alpaka -j
OMP_NUM_THREADS=48 ./build/rel-host-cpu-gcc/benchmark/histogram/histogram_alpaka --pixels 2073600 --repeat 20 --samples 5

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target histogram_alpaka -j
CUDA_VISIBLE_DEVICES=1 ./build/rel-cuda-nvcc-gcc/benchmark/histogram/histogram_alpaka --pixels 2073600 --repeat 100 --samples 5
```

## Self-validation

`src/helper.hpp` reimplements the reference MT19937 and `RandomBits()` so the same pixels are
generated on the host, and it computes an independent host histogram with the reference bucket
scheme. After a fresh device run the result is copied back and compared bin by bin with an **exact
integer comparison** (`verify()`), plus a total-count invariant `sum(hist) == ACTIVE_CHANNELS *
numPixels`. Any difference prints the first mismatching `(channel, bin)` and returns a nonzero exit
code.

Tolerance rationale: a histogram is an exact integer reduction of exact bucket assignments. The
reference itself uses `CompareDeviceResults` with `!=` for `unsigned int` (tolerance 0). Widening the
tolerance would hide bucket-assignment bugs, so this port keeps the comparison exact. A nonzero exit
code signals failure.

Validated pixel counts, including odd and non-multiple-of-block sizes and boundaries:
`1, 2, 3, 7, 15, 63, 127, 255, 256, 257, 511, 1000, 1023, 1024, 1025, 4095, 4096, 4097, 65537,
1048575, 1048576, 1048577`. All three genres pass on both the CPU and the CUDA backend. Entropy
values `-1, 0, 1, 2, 3, 4` pass on both backends. The `float4` input is additionally checked finite.

The CMake test registration runs a fast subset:

```bash
ctest --test-dir build/rel-host-cpu-gcc -R histogram_alpaka --output-on-failure
```

## Timing scope and reference reproduction

The measured metric is the per-histogram time: `repeat` back-to-back histogram computations between
two synchronisations, divided by `repeat`. This matches the reference, which times `i` iterations and
reports the average. One warm-up computation precedes the measured samples. Runtime ratio =
median_alpaka / median_reference; `<=1.00` match, `<=1.10` accepted, `>1.10` requires investigation.

Measured on a Tesla V100-SXM2-32GB (`CUDA_VISIBLE_DEVICES=1`), 5 independent invocations of
`--repeat 100 --samples 5` each. Reference values are the shared-memory (`smem atomics`) variant,
median of 5 invocations of `histogram_cuda_ref --i=100`.

| config | backend | genre | alpaka median (us) | ref median (us) | ratio | validation |
|--------|---------|-------|--------------------|-----------------|-------|------------|
| 1920x1080 (2073600) | CUDA | uchar1 | 24.46 | 47.19 | 0.52 | PASS |
| 1920x1080 (2073600) | CUDA | uchar4 | 65.18 | 75.04 | 0.87 | PASS |
| 1920x1080 (2073600) | CUDA | float4 | 85.30 | 91.84 | 0.93 | PASS |
| 4096x4096 (16777216) | CUDA | uchar1 | 134.51 | 244.40 | 0.55 | PASS |
| 4096x4096 (16777216) | CUDA | uchar4 | 206.45 | 294.88 | 0.70 | PASS |
| 4096x4096 (16777216) | CUDA | float4 | 373.63 | 415.00 | 0.90 | PASS |

The `smem` partial-histogram algorithm is faster than the reference's `gmem` variant (85.6, 210.0 and
219.8 us at 1920x1080), so the reported ratios are conservative relative to the reference's own
fastest variant.

The CPU reference (`histogram_omp_ref`) is an OpenMP *target-offload* program meant for `icpx`, which
is absent on rosi_v100. Built with `g++ -fopenmp` it falls back to the host and additionally reports
`INCORRECT`/`FAIL` because the `omp declare target` decode path is not exercised; it is not a usable
CPU performance or correctness baseline. CPU correctness is validated against the exact host
reference; the GPU comparison uses the CUDA reference. CPU throughput (Host/OmpBlocks, N=2073600,
`--repeat 20`): uchar1 ~1.0 ms, uchar4 ~1.2 ms, float4 ~1.5 ms (48 threads, with run-to-run
variation from shared-machine load).

Raw per-run samples, exact commands, and CPU/GPU/compiler metadata are in
`../results/histogram/histogram-gpu.csv`.

## Limitations and performance exception

- The pinned reference does not compare its device result against an independently regenerated host
  result; its "reference" is the same bucket scheme evaluated on the CPU. This port keeps that exact
  bucket scheme but recomputes the comparison from the same input pixels, and adds a total-count
  invariant.
- The reference's default `bandwidth_GBs = 41` is a hardcoded constant and is not used here; this
  port reports time, not a percentage of a fictitious peak.
- The HeCBench `--report` CSV mode and the `.tga` file-image tests are not ported; the reference's
  random-image genres are.
- The OpenMP reference is not a usable CPU baseline on rosi_v100 (see above), so CPU has no runtime
  ratio.
