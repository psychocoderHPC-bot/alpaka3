# dct8x8 Benchmark

Portable alpaka port of the HeCBench `dct8x8` benchmark. It applies a separable 8x8 DCT
(`DCT_FORWARD`, 666 in the HeCBench source) or its inverse (`DCT_INVERSE`, 777) to every full 8x8
block of a float image, then verifies the result against an independent double-precision host
reference with a relative-L2 comparison. A single kernel source drives the host CPU backend and the
CUDA backend; there is no native CUDA/HIP kernel and no backend branch.

## What it computes

The pinned HeCBench reference (`src/dct8x8-cuda/main.cu` + `kernels.cu` + `DCT8x8_gold.cu`, with the
OpenMP twin `src/dct8x8-omp/`) transforms a `width x height` float image in place, block by block:

- the forward DCT is an orthonormal DCT-II (`sqrt(1/8)` for the DC term, `sqrt(2/8)` otherwise),
  applied separably over rows then columns,
- the inverse is the transpose (DCT-III),
- the reference uses the hardcoded AAN fast transform from the NVIDIA CUDA SDK sample with the
  constants `C_a`..`C_f`, `C_norm` in `DCT8x8.h`; it was verified numerically to equal the
  orthonormal DCT-II/DCT-III pair above,
- only full 8x8 blocks are transformed (`i + 8 - 1 < imageH` and `j + 8 - 1 < imageW`); cells outside
  full blocks are left at the reference's uninitialized output.

`main` fills the image with `srand(2009); x = (float)rand() / (float)RAND_MAX;`, runs the forward
then the inverse transform `repeat` times each, and prints the relative L2 norm of the doubled result
against `DCT8x8CPU`.

This port keeps the transform definition, scaling and block layout exactly:

- the same hardcoded 8-point (i)DCT and constants as `DCT8x8.h`,
- the same row-then-column pass order and `BLOCK_SIZE = 8` full-block rule,
- the same comparison: relative L2 `sqrt(sum((got-ref)^2) / sum(ref^2)) < 1e-6`, here against an
  independent double-precision CPU reference, plus explicit NaN/Inf rejection.

Every work item owns one 8x8 block, loads it into registers, transforms rows then columns, and stores
it back. The row-then-column order and per-element mapping match `DCT8x8CPU`. One source runs on the
host CPU and on NVIDIA GPUs.

### Remainder cells

The reference allocates the output but never initializes cells outside full blocks, so their value is
undefined there and `Verify()` compares them anyway. This port defines those cells as a copy of the
input, which is deterministic and does not change the transform of any full block. The port also
transforms only full blocks, exactly like the reference.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6` (2026-10-09).
  - CUDA harness: `src/dct8x8-cuda/main.cu` (blob `f3686f55a784678c029399bb975ac16fe44c56c0`)
  - CUDA kernels: `src/dct8x8-cuda/kernels.cu` (blob `3b9607e718eb3a2faf682f59ad25d3b6ff47654a`)
  - CUDA gold: `src/dct8x8-cuda/DCT8x8_gold.cu` (blob `5909bcd4eaae4063c77590fd63c6e3b15f496dcb`)
  - Header: `src/dct8x8-cuda/DCT8x8.h` (blob `434d2862f5026a118684bdd22f24b80724d260d2`)
  - OpenMP harness: `src/dct8x8-omp/main.cpp` (blob `7b1035924130e78849882ba8caec546f7a7b95e6`)
  - OpenMP kernels: `src/dct8x8-omp/kernels.cpp` (blob `6a2b95fa659303444ee098461819b12506a74a35`)
  - OpenMP gold: `src/dct8x8-omp/DCT8x8_gold.cpp` (blob `5909bcd4eaae4063c77590fd63c6e3b15f496dcb`)
  - OpenMP header: `src/dct8x8-omp/DCT8x8.h` (blob `434d2862f5026a118684bdd22f24b80724d260d2`)
- README entry: HeCBench `README.md`, "dct8x8 (cuda), dct8x8 (omp)". The 8x8 DCT is the
  H.264-style integer-friendly 8-point transform; the reference is the NVIDIA CUDA SDK `dct8x8`
  sample.
- Reference sources are kept outside this repository and are not vendored.
- Licensing: the NVIDIA CUDA SDK source in `src/dct8x8-cuda` / `src/dct8x8-omp` carries NVIDIA's
  `DCT8x8.h` header (NVIDIA SDK EULA); the HeCBench repository is MIT (ORNL). This benchmark's files
  are `BSD-3-Clause` unless stated otherwise (`misc.hpp` notes its differing license).

## Parameters, modes, precision

- Named form: `dct8x8_alpaka --width W --height H --repeat R --samples S --seed X [--inverse]`.
- `--width`/`--height` default `8192 x 8192` (the HeCBench `run` target). Any positive size is
  accepted, including sizes that are not a multiple of 8 and sizes smaller than 8.
- `--repeat` default `100`, matching the reference `run` target: transforms per measured sample.
- `--samples` default `5`: independent measured samples.
- `--seed` default `2009`, matching the reference `srand(2009)`.
- `--inverse` runs the inverse DCT (the reference's second half). Default is the forward DCT.
- `--verify-only` runs the correctness checks without the timing loop.
- Data is `float`; the transform runs in `float` on the device and is compared against a
  double-precision reference.

## Build and run (reference environment: CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo, `/root/dct8x8-ref-src/`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
cd /root/dct8x8-ref-src/dct8x8-cuda
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 main.cu kernels.cu DCT8x8_gold.cu -o dct8x8_cuda_ref
cd ../dct8x8-omp
g++ -O2 -fopenmp -std=c++17 main.cpp kernels.cpp DCT8x8_gold.cpp -o dct8x8_omp_ref
```

The `dct8x8-omp` reference is an `omp target teams` offload program for `icpx`; built with
`g++ -fopenmp` it ignores the offload pragmas and runs the teams loop on the host. It is a functional
CPU check, not a tuned CPU baseline.

alpaka benchmark (the `-Dalpaka_COMPILE_PEDANTIC=ON` flag promotes warnings such as `-Wsign-compare`
to the `-Werror` failures CI sees):

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target dct8x8_alpaka -j
OMP_NUM_THREADS=48 ./build/rel-host-cpu-gcc/benchmark/dct8x8/dct8x8_alpaka \
  --width 999 --height 777 --repeat 20 --samples 3

# CUDA (sm_70 -> Tesla V100; on a V100 set the GPU with CUDA_VISIBLE_DEVICES)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target dct8x8_alpaka -j
CUDA_VISIBLE_DEVICES=3 ./build/rel-cuda-nvcc-gcc/benchmark/dct8x8/dct8x8_alpaka \
  --width 8192 --height 8192 --repeat 100 --samples 5
```

## Self-validation

`verify()` compares the device result against an independent, deliberately different double-precision
host implementation (`referenceTransform` in `helper.hpp`): an explicit O(8^2) orthonormal
DCT-II/DCT-III applied separably to each full block, with the same row-then-column order and
full-block rule. It reproduces the reference's relative-L2 metric and additionally rejects any NaN or
Inf. The process exits non-zero on failure.

The tolerance is the reference's own `1e-6`. It is not widened: the float fast transform versus the
double oracle measures `8.2e-8` (8x8) up to `8.8e-8` (8192x8192) for the forward DCT and
`1.3e-7` for the inverse, three orders of magnitude below the threshold. The ported kernel reaches
`7.9e-8`..`9.5e-8` on GPU, the same order as the reference's own `3.5e-8`/`8.4e-8` relative L2
against its CPU reference.

Validation covers small and irregular sizes and boundary cases on both backends: `8x8`, `8x7`,
`7x9`, `9x9`, `15x15`, `16x16`, `20x12`, `999x777`, `1000x776` and `8192x8192`, in both directions.
Sizes with no full block (`8x7`, `7x9`) yield an exact copy (relative L2 `0`) as expected.

## Timing scope and results

Timing is per transform, end-to-end on the queue: one measured sample runs `repeat` transforms and
the reported value is `total / repeat`. The measured metric is the median of `samples` independent
samples after one warm-up. This is directly comparable to the reference's `Average DCT8x8 kernel
execution time` (which measures the kernel loop only). Validation runs a separate single transform
after the timing loop.

See `results/dct8x8/` for the machine-readable CSV and provenance.

## Limitations / performance exception

**The GPU kernel is slower than the pinned CUDA reference (ratio > 1.10) on the measured workload.**
The reference is the hand-tuned NVIDIA CUDA SDK kernel: each thread owns one 8x8 block, the block
transposes through shared memory so both passes are coalesced, and it compiles to 32 registers with
512-thread blocks, giving high occupancy. This port expresses the same algorithm through alpaka's
portable work distribution (one work item per block, block held in registers). The register tile
costs 216 registers per thread, which caps occupancy, and the measured ratio on a V100 is ~1.5x for
`8192x8192` and ~2.6x for `999x777`. Several portable variants were measured (in-place two-pass,
per-block shared memory, and a coalesced shared-memory group kernel with 32 registers); none matched
the reference on the 8192x8192 workload. The port keeps the transform, scaling, block layout and
comparison exact, and this is a documented performance exception rather than a correctness gap.
