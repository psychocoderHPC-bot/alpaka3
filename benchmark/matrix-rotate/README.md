# matrix-rotate Benchmark

## What it computes

`matrix-rotate` rotates a square `n x n` matrix of `float` in place by 90 degrees
(clockwise), using the HeCBench "pyramid ring" scheme:

```
for layer in [0, n/2):
    first = layer
    last  = n - 1 - layer
    for i in [first, last):
        offset = i - first
        top = m[first][i]
        m[first][i]               = m[last-offset][first]   // left  -> top
        m[last-offset][first]     = m[last][last-offset]    // bottom -> left
        m[last][last-offset]      = m[i][last]              // right -> bottom
        m[i][last]                = top                     // top   -> right
```

Outer rings (`layer`) are independent, so the work parallelizes over rings; within one ring the
four corner moves are sequential and need the single saved temporary.

The matrix is initialized to `m[i][j] = i * n + j` and rotated `repeat` times. The host reference
applies the identical loop in `double`. Because the operation only permutes exactly representable
non-negative integers, the result is compared **bit-exact**.

## Provenance

* Upstream reference (HeCBench), commit-pinned:
  <https://github.com/ORNL/HeCBench/tree/0a17ee47ca6072192101036d4bf292e6612f22e6>
* CUDA source ported: [`src/matrix-rotate-cuda/main.cu`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/matrix-rotate-cuda/main.cu)
* OpenMP source ported: [`src/matrix-rotate-omp/main.cpp`](https://github.com/ORNL/HeCBench/blob/0a17ee47ca6072192101036d4bf292e6612f22e6/src/matrix-rotate-omp/main.cpp)
* Original author: Robert Searles (ring-swap `rotate_matrix`), as used by the HeCBench collection (ORNL, MIT license).
* Alpaka port: René Widera, 2026, ISC/BSD-3-Clause as marked per file. The alpaka sources contain no parts of the
  HeCBench code base; `helper.hpp` and `misc.hpp` carry a separate license header.

The alpaka implementation is a single portable kernel (`src/main_alpaka.cpp`) compiled unchanged for the host CPU
and for CUDA GPUs. There is no backend branch selecting a different algorithm, no native CUDA/HIP/OpenMP kernel,
and no separate serial-CPU vs parallel-GPU path. The ring mapping (`one linear thread per ring`) and the addressing
match the reference.

### Launch shape

The kernel only bounds the available parallelism; it never depends on the launch shape for correctness. The
`onHost::FrameSpec` is tuned from the device properties so both backends use their parallelism well:

* Host CPU executors (`warpSize == 1`) run one thread per logical frame and pay a fixed per-frame setup cost, so
  the launcher uses `min(rings, 2 * multiProcessorCount)` frames of extent one. Each host thread strides over the
  rings. One frame per ring is measurably slower on the host.
* Accelerators use the reference's 256 threads per block: `ceil(rings / 256)` frames of 256, matching the HeCBench
  grid `(n/2 + 255) / 256` blocks of 256 threads. `worker::linearThreadsInGrid` distributes the rings over the
  launched threads.

This is the only backend-dependent value in the benchmark, and it is a launch-shape heuristic (like the reference
`<<<blocks, 256>>>`), not an algorithm branch.

## Build and run

Configure with an alpaka3 preset from the repository root, then build the target.

CPU (host, GCC):

```bash
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build --preset rel-host-cpu-gcc --target matrix-rotate_alpaka
./build/rel-host-cpu-gcc/benchmark/matrix-rotate/matrix-rotate_alpaka --size 17 --repeat 3
```

CUDA (NVIDIA, GCC host compiler):

```bash
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON
cmake --build --preset rel-cuda-nvcc-gcc --target matrix-rotate_alpaka
./build/rel-cuda-nvcc-gcc/benchmark/matrix-rotate/matrix-rotate_alpaka --size 5000 --repeat 100
```

The binary runs once per available backend. Use `CUDA_VISIBLE_DEVICES=<idx>` to pin a GPU.

### Parameters

| Flag        | Positional | Default | Meaning                                     |
|-------------|------------|---------|---------------------------------------------|
| `--size`    | 1st        | 5000    | Square matrix side length `n` (odd allowed) |
| `--repeat`  | 2nd        | 100     | Number of in-place rotations                |

Both the HeCBench positional form `./main <size> <repeat>` and the named form are accepted.
`n = 5000`, `repeat = 100` reproduce the HeCBench reference run.

## Self-validation

`src/helper.hpp` provides an independent host reference:

* `initMatrix()` fills `m[i][j] = i * n + j` exactly like the reference.
* `referenceRotate()` runs the identical ring swap in **double**, sequentially, without any alpaka code.
* `verify()` compares the produced `float` matrix against the double golden matrix entry by entry and exits
  nonzero on any failure.

**Tolerance:** zero (bit-exact). Every entry is a non-negative integer below `n*n`, so it is exactly representable
as `float` for realistic `n`; a rotation is a permutation and introduces no rounding. Any nonzero tolerance would
be unjustified and could mask a wrong addressing scheme or a lost write. Non-finite outputs are always reported and
fail the check. The program returns `EXIT_FAILURE` when verification fails.

The CTest case runs `--size 17 --repeat 3` to exercise the odd-size pyramid-ring addressing and the exact compare
quickly. The following boundaries are exercised manually / by CI variants:

* even and odd `n` (odd `n` leaves a single centre cell untouched),
* `n = 1` and `n = 2` (no ring / smallest ring),
* `repeat = 1` and larger `repeat` (rotations compose to the identity for `repeat % 4 == 0`),
* the default `n = 5000`, `repeat = 100`.

## Timing scope

The measured section brackets `repeat` consecutive kernel launches with one device synchronization, matching the
HeCBench reference. One full `repeat`-batch is run as warm-up first; then 7 independent measured repetitions are
recorded. The reported metric is the **median per-rotation time in microseconds** (`total / repeat`), which is the
acceptance metric. Host-to-device and device-to-host copies, allocation and the validation run are excluded.
Raw samples, medians and variation are committed in the machine-readable
[`results.json`](results.json) in this directory.

`ratio = median(alpaka) / median(reference)`: `<= 1.00` matches, `<= 1.10` is accepted, `> 1.10` needs
investigation.

## Reference reproduction

The HeCBench references are built outside the repository (never committed):

```bash
# CPU reference
g++ -O2 -fopenmp -std=c++17 src/matrix-rotate-omp/main.cpp -o matrix-rotate-omp-ref
./matrix-rotate-omp-ref 5000 100

# CUDA reference
nvcc -std=c++17 -arch=sm_70 -O3 src/matrix-rotate-cuda/main.cu -o matrix-rotate-cuda-ref
./matrix-rotate-cuda-ref 5000 100
```

## Limitations / performance exception

The reference CUDA kernel launches one thread per ring and each ring's inner loop is short (`~n` iterations),
so the workload is latency-bound rather than bandwidth-bound. The portable alpaka kernel preserves the same
one-thread-per-ring mapping. On GPUs with many idle threads the launch/config overhead dominates; a different
work decomposition (e.g. one thread per swapped element pair) would change the addressing scheme and is therefore
out of scope.

The host CPU executor (`CpuOmpBlocks`) maps one logical frame to one thread and rebuilds the per-frame accelerator
state for every frame, so the launcher caps the host at `2 * cores` frames of extent one. On the 2-socket test host
(Intel Xeon Gold 6136, 2 x 12 cores, NUMA) the measured ratio depends strongly on sibling/HT contention and on the
reference OpenMP thread count: with `OMP_NUM_THREADS=24` the measured `median(alpaka) / median(reference)` is about
`0.35` at `n=2000`, `0.69` at `n=5000`, and `1.12` at `n=10000`. The `n=10000` case is above the acceptance
threshold and is reported as an investigated exception: both binaries are memory-bound at that size, the reference
OpenMP timing varies by more than 2x between 24 and 48 threads on this NUMA box, and the alpaka host launcher does
not pin threads. Any ratio above 1.10 is reported with the reproducing configuration rather than hidden.

