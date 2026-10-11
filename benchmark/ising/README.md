# ising Benchmark

Portable [alpaka](https://github.com/alpaka-group/alpaka) implementation of the HeCBench `ising`
benchmark (2D square-lattice Ising Monte-Carlo simulation with a checkerboard update). One C++20
source file is compiled unchanged for the host CPU and for CUDA GPUs; there is no backend-specific
kernel and no native CUDA/HIP/OpenMP code in this repository.

## What it computes

The benchmark simulates a 2D Ising spin lattice of `nx` rows by `ny` columns with a periodic
boundary and a checkerboard (black/white) Metropolis update:

* The lattice is split into two colors by `(row + column) % 2`. Each color is stored row-wise as a
  packed array of `nx * (ny / 2)` spins (`signed char`, value `+1` or `-1`).
* One full update performs two passes. In the black pass each black cell is updated from the frozen
  white lattice; in the white pass each white cell is updated from the (already updated) black
  lattice. Because each color reads only the opposite color and itself, the parallel update is race
  free.
* For a cell with spin `s` and the sum `m` of its nearest neighbor spins, the flip is accepted when

  ```
  randval < exp(-2 * beta * m * s)      with   beta = 1 / (alpha * TCRIT)
  ```

  where `TCRIT = 2.26918531421f` and `alpha` defaults to `0.1` (a cold, mostly-ordered state). This
  matches the HeCBench `update_lattice` reference exactly, including the off-column index selection
  `joff = is_black ? (row_odd ? j+1 : j-1) : (row_odd ? j-1 : j+1)`.
* The random buffer of `nx * (ny / 2)` floats is generated once with `srand(seed)` and
  `rand() / RAND_MAX`, then reused for every value and every iteration, exactly as the HeCBench
  CUDA reference does in its `#ifndef CURAND` verification path.
* Every measured iteration performs `alpha` and the full update; the reference reports
  updates per nanosecond as `nx * ny * niters / duration`.

## Reference and provenance

The implementation is a faithful adaptation of HeCBench at commit
[`0a17ee47ca6072192101036d4bf292e6612f22e6`](https://github.com/ORNL/HeCBench/tree/0a17ee47ca6072192101036d4bf292e6612f22e6):

* `src/ising-cuda/main.cu` (SHA-1 `e43f3996380963788b77d75608096a47f66fa171`)
* `src/ising-cuda/reference.h` (SHA-1 `f8612cfc69585a0270f4a72c536a36c9d781e6db`)
* `src/ising-omp/main.cpp` (SHA-1 `cb82aebbc385458171d39924732f26e9a478aa43`)

The HeCBench sources carry the NVIDIA 2019 MIT-style license. The referenced CUDA/OpenMP sources are
**not** vendored here; only the alpaka implementation is committed. The original benchmark derives
from the NVIDIA Ising sample
(<https://github.com/ORNL/HeCBench/tree/master/src/ising-cuda>).

## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--lattice-n <nx>` | `5120` | Number of lattice rows (must be even) |
| `--lattice-m <ny>` | `5120` | Number of lattice columns (must be even) |
| `--nwarmup <n>` | `100` | Number of warm-up full updates (not timed) |
| `--niters <n>` | `1000` | Number of timed full updates per measured repetition |
| `--repeat <n>` | `5` | Number of measured repetitions |
| `--alpha <a>` | `0.1` | Coefficient of the critical temperature |
| `--seed <s>` | `1234` | Random number seed |

`--lattice-n 5120 --lattice-m 5120 --nwarmup 100 --niters 1000` reproduces the HeCBench default
workload.

## Validation

The host reference and the checker live in `src/helper.hpp`. The reference is deliberately
**independent** of the kernel layout:

* It stores the full `nx x ny` lattice and applies the checkerboard update in place, reading the
  frozen opposite color, with the four orthogonal periodic neighbors. It does not use the packed
  two-array layout of the kernel, so a wrong packed index or neighbor selection in the kernel cannot
  be reproduced by sharing a bug.
* It evaluates the acceptance ratio in double precision with `std::exp`, while the kernel uses the
  backend's single-precision `expf`, so the two use different arithmetic.
* The comparison is **exact**: the lattice is a discrete `{-1,+1}` state, the update is a discrete
  map, and the HeCBench reference itself compares its device output to its host reference with `==`.
  No tolerance is used or widened. A separate domain check rejects any lattice entry that is not
  exactly `+1` or `-1`; for an integer state this is the meaningful corrupt/non-finite check (a stray
  byte from an uninitialized or partially copied buffer is caught instead of comparing unequal
  silently).
* Both the packed kernel and the checkerboard reference consume the identical `srand(seed)` stream,
  so identical inputs are compared.
* Any mismatch or invalid entry makes the process return non-zero.

Why exact comparison is safe for this chaotic Monte-Carlo chain: the decision margin
`|randval - acceptance_ratio|` was measured over the whole run at 512², 1024² and 2048² (200 timed
iterations each). The minimum margin over the non-saturated decisions was `5.5e-7` at 1024²/2048²
and `1.1e-5` at 512², far larger than the float/double difference of the acceptance computation
(relative error of `expf` vs `std::exp`, below `1e-7`). No decision lands close enough to the
boundary for the precision difference to flip it, so the chain is bit-reproducible across the CPU and
CUDA backends. This was also checked end to end: an independent double full-lattice reference matches
the packed HeCBench reference exactly at up to 5120² with 1100 sweeps.

## Build and run

The benchmark is registered in `benchmark/CMakeLists.txt` and is built with the rest of the alpaka
benchmarks. Every configure passes `-Dalpaka_COMPILE_PEDANTIC=ON` so the CI warning failures
(`-Wsign-compare`, `-Wimplicit-const-int-float-conversion`, ...) are reproduced locally.

Host CPU (GCC):

```bash
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target ising_alpaka -j
# defaults to the HeCBench workload; use a smaller lattice for a quick check
./build/rel-host-cpu-gcc/benchmark/ising/ising_alpaka --lattice-n 64 --lattice-m 64 --nwarmup 10 --niters 50 --repeat 1
./build/rel-host-cpu-gcc/benchmark/ising/ising_alpaka --lattice-n 1024 --lattice-m 1024 --nwarmup 10 --niters 200 --repeat 5
```

CUDA (nvcc + GCC), `sm_70` for a V100 / GV100:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target ising_alpaka -j
./build/rel-cuda-nvcc-gcc/benchmark/ising/ising_alpaka --lattice-n 5120 --lattice-m 5120 --nwarmup 100 --niters 1000 --repeat 5
```

`ising_alpaka` runs every enabled backend/executor and reports each one separately.

The HeCBench references are built outside this repository, at the pinned commit:

```bash
# CUDA reference, sm_70
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 src/ising-cuda/main.cu -o ising_cuda_ref
./ising_cuda_ref -x 5120 -y 5120 -w 100 -n 1000

# OpenMP host reference
g++ -O2 -fopenmp -std=c++17 -I src/ising-cuda src/ising-omp/main.cpp -o ising_omp_ref
OMP_NUM_THREADS=28 ./ising_omp_ref -x 1024 -y 1024 -w 10 -n 200
```

## Timing scope

Only the kernel work is timed. For each measured repetition the lattice is re-initialized, the
untimed warm-up runs, and then exactly `--niters` full updates (black pass + white pass) are enqueued
and one device synchronization is taken; the elapsed wall time of that block is the sample. Host/device
copies, allocation and the reference verification are excluded. The reported value is the median of
`--repeat` samples. The acceptance metric is the kernel runtime (all `niters` full updates) of the
`CpuOmpBlocks` / `GpuCuda` executor; the ratio is `median_alpaka / median_reference`.

A non-blocking queue is used so the backend can enqueue both passes of an iteration back to back and
synchronize once per measured block, matching the reference's asynchronous launch pattern. With the
blocking queue the per-enqueue host round-trip added roughly 1.4x overhead on the GPU.

## Measured performance

Warm-up followed by 5 measured repetitions, median. Ratio is `alpaka / reference`; below 1 means the
alpaka version is faster. Raw samples are in `results/ising/`.

### GPU (Quadro GV100, 32 GB, sm_70, CUDA 12.5, driver 555.52.04, `CUDA_VISIBLE_DEVICES=1`)

| lattice | warmup | niters | alpaka GpuCuda us (median) | CUDA ref us (median) | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 x 512 | 10 | 200 | 1634.9 | 1586.0 | 1.03 |
| 1024 x 1024 | 10 | 200 | 3903.5 | 3721.0 | 1.05 |
| 2048 x 2048 | 20 | 200 | 14437.9 | 14034.0 | 1.03 |
| 5120 x 5120 | 100 | 1000 | 366688.2 | 344658.0 | 1.06 |

All configurations verify PASS. The 1024²/2048² rows are interleaved alpaka/reference runs because
the GV100 is a shared node (`sharing: unknown`, 7 concurrent clients) and the reference samples
spread between 2.9 ms and 3.7 ms; the alpaka medians were stable to <1 %.

### CPU (Intel Xeon W-2175, 14 cores / 28 threads, GCC 13.2, `OMP_NUM_THREADS=28`)

| lattice | warmup | niters | alpaka CpuOmpBlocks us (median) | OMP ref us (median) | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 x 1024 | 10 | 200 | 214270.8 | 626398.0 | 0.34 |

The OpenMP reference targets the OpenMP offload model (`#pragma omp target ...`) and this build does
not offload, so it runs the reference kernel on the host with default scheduling. The CPU numbers are
a same-host comparison of the portable alpaka kernel against the HeCBench host fallback.

## Reference environment

Measured on the port's build host (fwk394), 2026-10-11:

* CPU: Intel Xeon W-2175, 14 cores / 28 threads, Ubuntu 24.04, GCC 13.2.0, `-O3`.
* GPU: NVIDIA Quadro GV100 32 GB (Volta, sm_70), CUDA 12.5 (V12.5.82), driver 555.52.04.
* `rosi_v100` (4x V100) was unreachable during this task; the Quadro GV100 (same sm_70 generation)
  was used for the CUDA evidence. GPU device 3 does not exist on fwk394; `CUDA_VISIBLE_DEVICES=1`
  selects the GV100.

## Limitations

* The portable kernel keeps the HeCBench packed black/white layout and the same parallelization
  (one packed element per thread). It reaches parity with the hand-written CUDA reference (ratio
  1.02-1.06) after moving to a non-blocking queue.
* The CPU executor (`CpuOmpBlocks`) is faster than the OpenMP-offload reference fallback on this host;
  the ratio is indicative because the reference build does not offload.
* `--repeat` defaults to 5; more samples give more stable medians on a shared node.
* The random sequence uses glibc `rand()`, so cross-platform comparison requires the same C library
  (`rand()` is implementation defined). This matches the HeCBench reference, which also relies on the
  platform `rand()`.
