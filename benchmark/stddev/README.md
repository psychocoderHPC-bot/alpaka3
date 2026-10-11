# stddev Benchmark

Portable alpaka port of the HeCBench `stddev` benchmark. It reduces every column of a row-major
matrix to a root-mean-square (sum-of-squares) deviation. A single source drives the host CPU backend
and every device backend; there is no native CUDA/HIP/OpenMP kernel and no backend branch.

## What it computes

For every column `c` of an `N x D` row-major `float` matrix the pinned HeCBench reference computes

```
std[c] = sqrt( sum_{r=0}^{N-1} data[r*D + c]^2 / sampleSize )
```

with `sampleSize = N - 1` (the reference always passes `sample = true`). Despite the name this is a
sum-of-squares deviation, not a mean-centred standard deviation: a mean-centred value would not match
the kernel. The reference device pipeline has two kernels, `sopKernel` (per-column sum of squares via
thread-block and global atomics) and `sampleKernel` (normalise by `sampleSize` and square root), plus
a `cudaMemset` that zeroes the output before every launch because the accumulation uses atomics.

This port reproduces the exact computation, the `float` value type, the two-stage pipeline, the atomic
accumulation and the pre-launch zeroing (`onHost::memset`). It keeps the reference's key decomposition
property: a 2D grid that splits both the column (fast) and row (slow) dimensions, so a warp reads 32
consecutive columns and the worker count stays near a fixed target independent of the matrix shape.
The row split is chosen as `D * rowSplits ~= 1048576` (capped so every worker keeps at least 8 rows).

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA reference: `src/stddev-cuda/main.cu` (Apache-2.0, "Copyright (c) 2018-2020, NVIDIA CORPORATION"; derived from NVIDIA RAPIDS [`raft`](https://github.com/rapidsai/raft))
  - OpenMP reference: `src/stddev-omp/main.cpp` (OpenMP *target-offload* program)
  - Shared reference header: `src/stddev-cuda/reference.h`
- README entry: HeCBench `README.md` line 1692, "stddev (cuda): Standard deviation".
- `benchmarks.yaml` test metadata: `args: ["65536", "16384", "100"]`, regex
  `Average execution time of stddev kernels: ...`.
- The HeCBench sources are kept outside this repository and are not vendored.
- Upstream HeCBench license: `BSD-3-Clause` (see the HeCBench repository). The ported files are
  `BSD-3-Clause`; the CMake file is `ISC` like the neighbouring alpaka3 benchmarks.

## Parameters, modes, precision

- Positional (HeCBench form): `stddev_alpaka <D> <N> <repeat>` (e.g. `65536 16384 100`).
- Named form: `stddev_alpaka --D <cols> --N <rows> --repeat <r> --samples <s>`.
- `D` (columns) default `65536`; any value is accepted, not only multiples of 32. The reference
  documents a multiple-of-32 requirement; odd and prime `D` are exercised here on purpose.
- `N` (rows) default `16384`; must be at least 2 (sample stddev divides by `N-1`).
- `--repeat` default `100` (the HeCBench inner repetition count). A sample times `repeat` pipelines
  back-to-back between two synchronisations.
- `--samples` default `5` independent measured samples per configuration.
- Precision is `float` (`Type = float`), matching the reference. The host oracle is `double`.

## Build and run (rosi_v100, CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binary (outside the repo, `/root/stddev-ref-src`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cd /root/stddev-ref-src
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 -use_fast_math main.cu -o stddev_cuda_ref
# OpenMP reference (offload program; g++ fallback is host-only, see limitations)
g++ -O2 -fopenmp -std=c++17 -Wall main.cpp -I. -o stddev_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target stddev_alpaka -j
OMP_NUM_THREADS=48 ./build/rel-host-cpu-gcc/benchmark/stddev/stddev_alpaka --D 4096 --N 4096 --repeat 50 --samples 7

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target stddev_alpaka -j
CUDA_VISIBLE_DEVICES=3 ./build/rel-cuda-nvcc-gcc/benchmark/stddev/stddev_alpaka --D 4096 --N 4096 --repeat 100 --samples 7
```

The CMake test registration runs a fast irregular subset:

```bash
ctest --test-dir build/rel-host-cpu-gcc -R stddev_alpaka --output-on-failure
```

## Self-validation

`src/helper.hpp` regenerates the input with the reference's `srand(123)` / `rand()/(float)RAND_MAX`
sequence and evaluates the exact pinned formula independently in `double`: pass 1 accumulates the sum
of squares streaming the matrix in linear order (opposite memory order to the device's strided
reduction) and pass 2 normalises and takes the square root in `double`. `verify()` rejects NaN/Inf in
either buffer and compares with `|got - ref| <= 1e-3 + 1e-3 * |ref|`.

Tolerance rationale: the pinned reference itself accepts `fabsf(ref - got) <= 1e-3` on the same data.
The only deviation source is the `float` accumulation order (atomics and independent partials). The
absolute term is therefore the reference's own tolerance, not widened; the relative term scales it for
large values. Correctness is checked on every run before timing is reported; a nonzero exit signals
failure.

Validated shapes (CUDA and CPU, each within the 32-bit element extent): `1, 31, 33, 255, 257, 1000,
4096, 65536` columns against `2, 4, 17, 65, 257, 4096, 16384` rows, including prime and non-multiple-
of-width `D`, `N = 2` (minimum sample size), and `D=1`. The registered CTest case uses `D=320, N=97`.

## Timing scope and reference reproduction

The measured metric is the per-pipeline kernel time: `repeat` back-to-back
`memset + sop + normalise` pipelines between two synchronisations, divided by `repeat`. Kernel time
only; the one device->host result copy and the host oracle are outside the timed region. One warm-up
pipeline precedes the measured samples. The reference measures exactly the same scope
(`repeat` iterations between two `cudaDeviceSynchronize`, divided by `repeat`).
Runtime ratio = median_alpaka / median_reference; `<=1.00` match, `<=1.10` accepted, `>1.10` requires
investigation.

Measured on a Tesla V100-SXM2-32GB (`CUDA_VISIBLE_DEVICES=3`), `--repeat 100`, 7 samples:

| config (D x N) | backend | alpaka median (us) | ref median (us) | ratio | variation | validation |
|----------------|---------|--------------------|-----------------|-------|-----------|------------|
| 4096 x 4096    | CUDA V100 | 100.81 | 94.0 | 1.072 | 0.5% | PASS |
| 1024 x 65536   | CUDA V100 | 330.01 | 400.0 | 0.825 | 0.7% | PASS |
| 65536 x 16384  | CUDA V100 | 5351.4 | 6119.0 | 0.874 | 0.4% | PASS |

The exact per-run samples, reference provenance and full repro commands are in the machine-readable
`results/stddev/results.json` run artifact (kept outside the repository, under the orchestration run
directory).

## Limitations and performance exception

- The pinned OpenMP reference is an OpenMP *target-offload* program meant for `icpx`
  (`-fiopenmp -fopenmp-targets=spir64`), which is absent on rosi_v100. Built with
  `g++ -fopenmp -std=c++17` and run with `OMP_NUM_THREADS=48 ./stddev_omp_ref 320 97 10`
  it prints a timing and then **FAIL**: the g++ host fallback of the `omp target` regions does not
  reproduce the offload reduction semantics this program relies on. The g++ OpenMP build is therefore
  neither a correctness nor a performance baseline; the CPU comparison is indicative only and CPU
  correctness is validated against the independent `double` host oracle.
- The reference's `main` may report `PASS` on the target even when the offload reduction is
  architecture-sensitive; this port validates against a `double` oracle instead.
- The reference at the default `65536 x 16384` spends most of its wall time in its cache-hostile
  single-threaded `stddev_ref` host oracle (tens of minutes); only the kernel timing line is used for
  the ratio.
- The 4096 x 4096 ratio (1.06-1.07 across two independent runs) is above the ideal `<=1.00` but within the accepted `<=1.10` band. The
  reference uses one thread per (32-column, row-chunk) tile with two shared-memory atomic phases; the
  port trades that for one atomic add per worker, which is close but not identical on the smallest
  grids. Larger matrices are at or below parity.
