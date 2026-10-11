# morphology Benchmark

Portable alpaka port of the HeCBench `morphology` benchmark. It applies a 2D separable
morphological dilation and erosion with a rectangular structuring element to an 8-bit image
(`dilation` then `erosion`, repeated), then verifies the result against an independent host
reference with an exact byte comparison. A single kernel source drives the host CPU backend and
every device backend; there is no native CUDA/HIP kernel and no backend branch.

## What it computes

The pinned HeCBench reference (`src/morphology-cuda/main.cu` + `morphology.cu`, with the OpenMP
twin `src/morphology-omp/main.cpp` + `morphology.cpp`) runs the van Herk/Gil-Werman (vHGW)
algorithm for two separable morphological operators:

- **dilation**: per output cell the maximum over the structuring-element window, border value `255`
  (`WHITE`),
- **erosion**: per output cell the minimum over the window, border value `0` (`BLACK`).

`main` builds a black image with a single white pixel at `(height/2 - 1, width/2 - 1)`, then
applies the `dilate`+`erode` pair `repeat` times in place and prints `PASS` iff
`sum(image) == 255`.

This port keeps the reference pipeline exactly:

- the same pass order: one horizontal pass then one vertical pass per operation,
- the same structuring element (`hsize` x `vsize`, origin in the centre),
- the same odd/even radii and border handling,
- the same comparison: exact bytes after `repeat` `dilate`+`erode` pairs, plus a reproduction of
  the reference's `sum(image) == 255` report line.

One source runs on the host CPU and on NVIDIA GPUs.

### Structuring element, border and parity semantics

The pinned reference uses a two-way scan over `2 * selSize` samples per work item and writes an
output at index `c + selSize / 2`. Its exact semantics (validated exhaustively against the pinned
CUDA binary; see below) are:

- horizontal pass: outputs only columns `c` with `L <= c < W - hsize / 2`, where `L = hsize / 2`
  and the right window edge is `R = (hsize odd) ? hsize / 2 : hsize / 2 - 1`. Columns outside that
  interval stay at the intermediate buffer's initial `0`.
- vertical pass: rows outside `[vsize / 2, H - vsize / 2)` are set to the operation's border value
  for the whole row; the interior uses the window `[r - vsize / 2, r + Rv]` with
  `Rv = (vsize odd) ? vsize / 2 : vsize / 2 - 1`.

This means the first and last `hsize / 2` (resp. `vsize / 2`) rows of the final image are always
the border value, and untouched intermediate columns act as `0`. The reference accepts both odd
and even structuring-element sizes; this port does the same.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6` (2026-10-09).
  - CUDA harness: `src/morphology-cuda/main.cu` (blob `f6f0e8829ff5b05f0da8d6009637e41b031964da`)
  - CUDA kernels: `src/morphology-cuda/morphology.cu` (blob `7ae013891c83bcfea0227572309e29f62c27faef`)
  - Header: `src/morphology-cuda/morphology.h` (blob `9d5c87995b8a5408b2a04095eb1cfbfc4e6e4471`)
  - OpenMP harness: `src/morphology-omp/main.cpp` (blob `8203b6bb999b3956246488cbeb2582943384ecb9`)
  - OpenMP kernels: `src/morphology-omp/morphology.cpp` (blob `9f8fe8803ca92d7b31170df0078680b7cc5c7cbe`)
  - OpenMP header: `src/morphology-omp/morphology.h` (blob `c5d0df4522526a635c0a91e4ed323540b509ea91`)
- README entry: HeCBench `README.md` line 1220, "morphology (cuda)"; original algorithm
  <https://github.com/yszheda/CUDA-Morphology>. The vHGW algorithm is from van Herk,
  *Pattern Recognition Letters* 13 (1992) 517-521, and Gil and Werman, *IEEE Trans. PAMI*
  15(5) (1993) 504-507; see the leptonica binary-morphology notes.
- Reference sources are kept outside this repository and are not vendored.
- Upstream HeCBench license: MIT (ORNL). The `morphology-cuda` / `morphology-omp` directories also
  carry the repository's `LICENSE` file (GNU GPL v3 in the reference tree) for the original
  CUDA-Morphology code.
- This file is `BSD-3-Clause`.

## Parameters, modes, precision

- Named form:
  `morphology_alpaka --hsize K --vsize K --width W --height H --repeat R --samples S --seed X`.
- `--width`/`--height` default `999 x 777` (the irregular HeCBench-derived measured frame). Any
  positive size is accepted, including odd and non-square sizes.
- `--hsize`/`--vsize` default `71` (the measured workload uses ~70). Both odd and even positive
  values are accepted up to `256`.
- `--repeat` default `100`, matching the reference `run` target: inner `dilate`+`erode` pairs per
  measured sample.
- `--samples` default `5`: independent measured samples of the operation time.
- `--seed` default `0` reproduces the reference's single-white-pixel image. A nonzero seed
  generates a deterministic pseudo-random 8-bit image for broader coverage.
- `--verify-only` runs the correctness checks without the timing loop.
- Data is an 8-bit image and the datapath is integer (`min`/`max`); there is no floating point
  accumulation and the result is exact.

The CPU and CUDA builds are the same source; only the preset/executor changes.

## Build and run (rosi_v100, CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo, `/root/morphology-ref-src/`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
cd /root/morphology-ref-src/morphology-cuda
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 main.cu morphology.cu -o morphology_cuda_ref
cd ../morphology-omp
g++ -O2 -fopenmp -std=c++17 main.cpp morphology.cpp -o morphology_omp_ref
```

alpaka benchmark (the `-Dalpaka_COMPILE_PEDANTIC=ON` flag promotes warnings such as
`-Wsign-compare` to the `-Werror` failures CI sees):

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target morphology_alpaka -j
OMP_NUM_THREADS=48 ./build/rel-host-cpu-gcc/benchmark/morphology/morphology_alpaka \
  --width 999 --height 777 --hsize 71 --vsize 71 --repeat 20 --samples 5 --seed 12345

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target morphology_alpaka -j
CUDA_VISIBLE_DEVICES=3 ./build/rel-cuda-nvcc-gcc/benchmark/morphology/morphology_alpaka \
  --width 999 --height 777 --hsize 71 --vsize 71 --repeat 100 --samples 5 --seed 12345
```

## Algorithm

The portable kernel replaces the fixed-size block scan of the CUDA reference with a thread-local
sliding window (a monotonic deque) over a segment of the row (horizontal pass) or column (vertical
pass). It is O(1) per output element and depends on the structuring-element size only through the
segment overlap, so it stays fast for large structuring elements.

- `HorizontalPass`: a work item handles `segLen` consecutive output columns of one row; the window
  `[c - hsize/2, c + R]` is maintained with a monotonic deque.
- `VerticalPass`: a work item handles `segLen` consecutive output rows of one column; border rows
  are written directly and the interior uses the same deque.

On a host backend a work item maps to one OpenMP iteration; on CUDA it maps to one thread. The
algorithm is identical on both.

## Self-validation

`src/helper.hpp` contains an independent, deliberately different O(size-per-output) host reference:
for every output cell it scans the full `[c - L, c + R]` window with the reference's radii and
border rules, once for dilation and once for erosion. `fullReference()` applies the `repeat`
`dilate`+`erode` pairs exactly like the reference `main`. After a fresh accelerator run the result
is copied back and compared byte by byte (`verify()`); the first mismatch reports both the index and
the worst absolute difference and returns a nonzero exit code. For the reference's black/white image
(`seed 0`) the reference's own `sum(image) == 255` line is likewise reproduced and a non-binary
output is rejected.

Tolerance rationale: the datapath is integer `min`/`max`; the reference itself compares with `!=`
and its `PASS` test is an exact integer sum. Widening the tolerance would hide border and window
bugs, so this port keeps the comparison exact (tolerance 0). A nonzero exit code signals failure.

Validated on rosi_v100 (CPU `rel-host-cpu-gcc` and CUDA `rel-cuda-nvcc-gcc`, both with
`alpaka_COMPILE_PEDANTIC=ON`):

- odd, even, `1`, and large structuring elements (`1, 2, 3, 4, 5, 6, 7, 8, 33, 71, 255`),
- image sizes `1, 2, 5, 16, 31, 33, 64, 99, 100, 127, 129, 255, 257, 511, 777, 999, 1023, 1025`
  in both dimensions, including 1-row/1-column and non-multiples of the segment length,
- seeds `0` (reference image) and `12345`/`777` (random images),
- both backends match the host reference exactly (0 mismatches).

The structured-element and border semantics were additionally cross-checked against the pinned CUDA
reference binary itself over ~6000 (size, sel-size, seed) combinations with 0 mismatches.

The CMake test registration runs a fast subset:

```bash
ctest --test-dir build/rel-host-cpu-gcc -R morphology_alpaka --output-on-failure
```

## Timing scope and reference reproduction

The measured metric is the average time of **one morphological operation** (a dilation or an
erosion, i.e. one horizontal + one vertical pass): `2 * repeat` operations are timed between two
synchronisations and divided by `2 * repeat`. This is directly comparable to the reference's
`Average kernel execution time (dilate/erode)`. It includes the per-operation intermediate-buffer
memset, which the reference performs inside `morphology()` before its timed kernels. One warm-up
pair precedes the measured samples. Runtime ratio = median_alpaka / median_reference; `<=1.00`
match, `<=1.10` accepted, `>1.10` requires investigation.

Measured on a Tesla V100-SXM2-32GB (`CUDA_VISIBLE_DEVICES=3`), host CPU Intel Xeon Gold 6136
@3.00GHz, 5 independent invocations of the alpaka binary per row; the reference value is the median
of one operation from `morphology_cuda_ref` invocations at the same configuration.

| config | backend | alpaka median (us/op) | ref median (us/op) | ratio | validation |
|--------|---------|-----------------------|--------------------|-------|------------|
| 999x777, h=71 | CUDA | 57.98 | 83.5 | 0.69 | PASS |
| 999x777, h=32 | CUDA | 47.54 | 95.0 | 0.50 | PASS |
| 999x777, h=5 | CUDA | 33.48 | 449.5 | 0.07 | PASS |
| 1920x1080, h=71 | CUDA | 127.67 | 216.5 | 0.59 | PASS |
| 4096x4096, h=32 | CUDA | 554.48 | 1935.5 | 0.29 | PASS |
| 8192x8192, h=32 | CUDA | 2429 | 7673.5 | 0.32 | PASS |
| 8192x8192, h=64 | CUDA | 2967 | 5750 | 0.52 | PASS |

The reference's block-scan kernels are fastest for large structuring elements; for small elements
the reference is much slower than the sliding-window kernel, which is where the very low ratios
come from. The measured workload (large structuring element) is a match.

CPU throughput (Host/OmpBlocks, 48 threads, `--repeat 20`, N=999x777, h=71): ~1.2 ms/op with
run-to-run variation from shared-machine load. The OpenMP reference (`morphology_omp_ref`) is an
OpenMP *target-offload* program meant for `icpx`, which is absent on rosi_v100; built with
`g++ -fopenmp` it falls back to the host and hangs under `omp target teams` for this benchmark, so
it is not a usable CPU performance or correctness baseline. CPU correctness is validated against the
exact independent host reference.

Raw per-run samples, exact commands, and CPU/GPU/compiler metadata are in
`../results/morphology/morphology-gpu.csv`.

## Limitations and performance exception

- The pinned reference does not compare its device result against an independently regenerated host
  result; its "reference" is the same algorithm on the CPU. This port keeps the reference's exact
  semantics and recomputes an independent O(size-per-output) host oracle.
- The reference's `morphology-omp` build is not usable on rosi_v100 (see above), so CPU has no
  runtime ratio.
- The reference's default `32 32 8192 8192 100` and the derived `999x777` measured frame are both
  supported. The task states `dimensions ~999x777`; that is the default here.
- The reference `main` always exits 0 even when it prints `FAIL`; this port reproduces that report
  line but additionally exits nonzero on any byte mismatch against the host reference.
