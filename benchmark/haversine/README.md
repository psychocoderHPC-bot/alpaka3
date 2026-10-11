# haversine Benchmark

Portable alpaka port of the HeCBench `haversine` benchmark. It computes the great-circle distance
between a large table of query locations and a small set of reference cities with the haversine
formula. One source drives the host CPU backend and the CUDA backend; there is no native
CUDA/HIP/OpenMP kernel and no backend branch.

## What it computes

For every query pair `i` the kernel reads a `double4` `(a_lat, a_lon, b_lat, b_lon)` and evaluates

```
ax = a_lon * DEGREE_TO_RADIAN ; ay = a_lat * DEGREE_TO_RADIAN
bx = b_lon * DEGREE_TO_RADIAN ; by = b_lat * DEGREE_TO_RADIAN
x  = (bx - ax) / 2              ; y  = (by - ay) / 2
d  = 2 * EARTH_RADIUS_KM * asin( sqrt( sin(y)^2 + sin(x)^2 * cos(ay) * cos(by) ) )
```

with `DEGREE_TO_RADIAN = pi/180` and `EARTH_RADIUS_KM = 6371`, all in `double` precision. The input
construction is exactly the pinned reference's: `num_cities` distinct locations are replicated once
per reference city, and for each block the reference-city latitude/longitude is copied into the
`z`/`w` slots using the pinned 1-based `index_map` `{436483, 1952407, 627919, 377884, 442703,
1863423}` (bombay, melbourne, waltham, moscow, glasgow, morocco). One thread produces one distance;
there is no reduction or reordering.

Parallelism is expressed with `onAcc::makeIdxMap` over a flat 1D `IdxRange`. A flat 1D range with
`worker::linearThreadsInGrid` is the alpaka form that is correct on the host CPU executors as well as
on CUDA; a multidimensional `threadsInGrid` range is not honoured by the host executors. The frame
extent (`--frame`, default 256) is a launch-shape hint only.

The port also reproduces the reference's host-side verification, but as a real check: it recomputes
an independent double-precision reference (`haversineBench::hostReference` in `src/helper.hpp`),
compares with a justified tolerance, rejects NaN/Inf and exits nonzero on failure.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA: `src/haversine-cuda/main.cu`, `src/haversine-cuda/distance.cu`,
    `src/haversine-cuda/distance.h`.
  - OpenMP: `src/haversine-omp/main.cpp`, `src/haversine-omp/distance.cpp` (the same arithmetic in an
    OpenMP `target` region).
  - HeCBench README entry: `haversine (cuda): Haversine distance (https://github.com/rapidsai/cuspatial)`.
- Upstream license: the NVIDIA sources carry
  `Copyright (c) 2020, NVIDIA CORPORATION.` under **Apache-2.0** (see the `LICENSE` file in each
  `src/haversine-*` directory). The reference sources are kept outside this repository and are not
  vendored; only this clean-room alpaka reimplementation is committed. The implementation files in
  this directory are `BSD-3-Clause`.

## Parameters, modes, precision

- `haversine_alpaka --cities N --repeat R --samples S --seed X [--frame F] [--input FILE]
  [--dump FILE] [--verify-only]`.
- HeCBench positional form is also accepted: `haversine_alpaka <file> <repeat>`.
- `--cities` default `2097152` (`2^21`), the pinned `num_cities`; `num_ref_cities` is fixed at 6.
- `--repeat` default `100`, comparable to the reference positional `<repeat>`; `--samples` default `5`.
- `--seed` default `12345`, used by the built-in location generator. `--input FILE` instead reads the
  exact HeCBench `lat lon` format (one pair per line), so the pinned 42 MB `locations.txt` can be
  replayed. `--dump FILE` writes the raw device distance array (validation aid).
- `--verify-only` runs the correctness check only. `--frame` sets the logical frame extent.
- All arithmetic is `double`. The reference CUDA target is compiled with `-use_fast_math`; this port
  uses the default-precision libm/device math, which is not relaxed by the port.

The pinned HeCBench `locations.txt` is a DVC-tracked artifact
(`src/geodesic-sycl/locations.txt`, md5 `03ce97e6d0c09f8be45907295e5d78c4`, 42873157 bytes,
symlinked by the geodesic CUDA/HIP/OMP variants). This port is self-contained: by default it
generates deterministic coordinates with a xorshift64* generator, and the pinned file can be supplied
with `--input` (used in the validation below).

## Build and run

```bash
# CPU (host backend, g++)
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON -Dalpaka_BUILD_BENCHMARKS=ON .
cmake --build build/rel-host-cpu-gcc --target haversine_alpaka -j
OMP_NUM_THREADS=14 ./build/rel-host-cpu-gcc/benchmark/haversine/haversine_alpaka \
    --input locations.txt --repeat 20 --samples 7

# CUDA (sm_70, g++ host compiler, nvcc 12.x)
export PATH=/usr/local/cuda/bin:$PATH
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON -Dalpaka_BUILD_BENCHMARKS=ON .
cmake --build build/rel-cuda-nvcc-gcc --target haversine_alpaka -j
CUDA_VISIBLE_DEVICES=1 ./build/rel-cuda-nvcc-gcc/benchmark/haversine/haversine_alpaka \
    --input locations.txt --repeat 100 --samples 7
```

The reference binaries (outside the repo):

```bash
# CUDA reference (nvcc, sm_70)
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 -use_fast_math main.cu distance.cu -o hav_cuda_ref
CUDA_VISIBLE_DEVICES=1 ./hav_cuda_ref locations.txt 100

# Host CPU reference: the OpenMP variant uses an OpenMP `target` offload region that g++ cannot
# compile and icpx/nvc are absent, so the CPU baseline is the same source with the offload pragma
# replaced by a plain host parallel-for (a wrapper `distance_hostpar.cpp`). g++ 13.2, -O2.
g++ -O2 -fopenmp -std=c++17 -Wall main_timed.cpp distance_hostpar.cpp -o hav_omp_timed
OMP_NUM_THREADS=14 ./hav_omp_timed locations.txt 20
```

## Self-validation

- `hostReference` (`src/helper.hpp`) is an independent double-precision transcription of the pinned
  formula, evaluated in linear order. The accelerator result is compared against it by `verify`.
- `verify` reports the maximum absolute error and rejects NaN/Inf in both buffers, returning a nonzero
  exit code when any element exceeds the tolerance. `checkInputFinite` rejects non-finite or
  out-of-range input coordinates before the run.
- Tolerance: `|got - ref| <= 1e-6 km + 1e-9 * |ref|`. Both sides evaluate the same `double`
  expression; the only difference is the libm implementation of `sin`/`cos`/`asin`/`sqrt`, worth a few
  ULP. At distances of order `1e4 km` one ULP is about `2e-12 km`, so the observed deviation is far
  below the `1e-6 km` (1 mm) absolute term. The tolerance is not widened to hide bugs: a wrong formula
  produces km-scale differences. On the GPU the observed maximum deviation over the full default case
  is `5.9e-9 km`; on the CPU it is exactly `0`.
- Cross-check against the pinned reference binary: feeding the same 2^21 synthetic input to the pinned
  `src/haversine-cuda` binary (`nvcc sm_70 -use_fast_math`) and to the port and comparing the device
  distance arrays elementwise gives a maximum difference of `5.9e-9 km`; the reference binary's own
  device-vs-host deviation is `2.0e-9 km`. The port is therefore inside the reference's own rounding
  noise.

Validated on both the CPU and CUDA backends over `num_cities = 1, 2, 3, 5, 31, 32, 33, 255, 256,
257, 1000, 100003` (seeds 42 and 7), covering 1 city, odd, non-multiple-of-256 and boundary sizes; all
verify PASS. The CMake test registration runs a fast subset:

```bash
ctest --test-dir build/rel-host-cpu-gcc -R haversine_alpaka --output-on-failure
```

## Timing scope and reference reproduction

The measured metric is the per-kernel time: `repeat` back-to-back enqueues between two
synchronisations, divided by `repeat`, after one warm-up. This matches the reference, which wraps the
launch loop in `cudaDeviceSynchronize()` on both ends and divides by `<repeat>`. `RESULT` prints the
median, min, max and relative spread plus every raw sample. Runtime ratio = median_alpaka /
median_reference; `<=1.00` match, `<=1.10` accepted, `>1.10` requires investigation.

Host: **Intel Xeon W-2175**, 14 threads via `OMP_NUM_THREADS=14`. GPU: **Quadro GV100 (SM 7.0)**,
driver 555.52.04, CUDA 12.5 (`V12.5.82`), `CUDA_VISIBLE_DEVICES=1`; host compiler g++ 13.2. Local CPU
build g++ 13.3. Flags: `-Dalpaka_COMPILE_PEDANTIC=ON` on both, `-DCMAKE_CUDA_ARCHITECTURES=70` for CUDA.

| config | backend | size | alpaka median | ref median | ratio | relspread | validation |
|--------|---------|------|---------------|------------|-------|-----------|------------|
| GPU | Cuda | 2^21 cities (12582912 pairs) | 9383.6 us | 11427 us | 0.82 | 0.05 | PASS |
| CPU | Host (OmpBlocks, 14 threads) | 2^21 cities | 99989.6 us | 94009.9 us | 1.064 | 0.004 | PASS |
| CPU | Host (serial) | 2^21 cities | 1004400.6 us | 996880.3 us | 1.008 | 0.09 | PASS |

The GPU ratio is `0.82` (the alpaka kernel is faster than the `-use_fast_math` reference). The CPU
`OmpBlocks` ratio is `1.064`, inside the accepted bound; `CpuSerial` is `1.008` against the sequential
host-reference loop. `results.json` holds the machine-readable measurement rows.

## Limitations and performance exception

- `num_cities = 2^21` needs 42 MB for the location table and 6x that for the pair table (1.2 GB host
  plus device); the default run requires a GPU with at least ~2 GB free. Reduce `--cities` for small
  devices.
- The remote GPU host is shared; individual GPU samples carry occasional outliers (up to ~1.5x the
  median). Medians are used and all raw samples are retained.
- The 1-based `index_map` indices can exceed small `--cities` values; the port clamps the lookup index
  to `num_cities - 1` for validation sizes. For the default `2^21` the clamp is inactive and the input
  construction is identical to the reference.
