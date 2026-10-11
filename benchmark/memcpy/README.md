# memcpy Benchmark

Portable alpaka port of the HeCBench `memcpy` benchmark. It measures the bandwidth of byte-wise
memory copies between host and device. A single source drives the host CPU backend and every device
backend through `onHost::memcpy`; there is no native CUDA/HIP/OpenMP kernel and no backend branch.

## What it computes

The HeCBench reference `src/memcpy-cuda/main.cu` allocates a raw byte buffer per size, fills it, and
measures the per-copy latency of `cudaMemcpyAsync` in the host-to-device and device-to-host
directions. There is no compute kernel and the reference does not check the copied bytes.

This port reproduces that measurement and adds:

- a **device-to-device** mode (not present in the reference),
- a **byte-exact correctness check** on a freshly poisoned destination for every mode and size.

For each size and mode it enqueues `repeat` copies back-to-back between two synchronisations and
reports the wall time divided by `repeat`, which is exactly the reference timing scope. It prints
`median`, `min`, `max`, relative spread, effective bandwidth, and every raw sample.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6`.
  - CUDA reference: `src/memcpy-cuda/main.cu` (blob `4a9230f9c3cf98854c5eea1d52dcb782ac25cfb4`)
  - OpenMP reference: `src/memcpy-omp/main.cpp` (blob `4f8d7f585f7289d26398720d7be57e895b75ba5e`)
- README entry: HeCBench `README.md` line 1151, "memcpy (cuda): Evaluate bandwidth of memory copy
  between a host and a device".
- Reference sources are kept outside this repository and are not vendored.
- Upstream HeCBench license: see the HeCBench repository. This file is `BSD-3-Clause`.

The pinned reference uses 16 sizes `size[i] = 1 << (i + 6)`, i.e. **64 B .. 2 MiB**, not the
"64 B .. 32 GB" mentioned in the task brief. The default size list here therefore matches the pinned
reference exactly (64 B .. 2 MiB); larger sizes are reachable through `--sizes` and are bounded by
`--max-bytes`.

## Parameters, modes, precision

- Positional (HeCBench form): `memcpy_alpaka <repeat>`.
- Named form: `memcpy_alpaka --sizes A,B,C --repeat R --samples S --max-bytes N --mode M`.
- `--sizes` default: the 16 HeCBench sizes `64, 128, ..., 2097152` bytes. Comma separated.
- `--repeat` default `8192`, matching the HeCBench `./main 8192` invocation.
- `--samples` default `5` independent measured samples per (mode, size).
- `--max-bytes` default 4 GiB, further capped to one third of the device global memory so a
  device-to-device test can hold two payloads. Requests above the cap are skipped and reported.
- `--mode` is `h2d`, `d2h`, `d2d`, or `all` (default `all`).
- `--verify-only` runs correctness checks without the timing loop.
- Data is opaque bytes (`std::uint8_t`). There is no floating point arithmetic.

## Build and run (rosi_v100, CUDA 12.6, g++ 13.2, CMake 3.28)

Reference binaries (outside the repo, `/root/memcpy-ref-src`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
cd /root/memcpy-ref-src
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 main.cu -o memcpy_cuda_ref
g++ -O2 -fopenmp -std=c++17 main_omp.cpp -o memcpy_omp_ref
```

alpaka benchmark:

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target memcpy_alpaka -j
OMP_NUM_THREADS=48 ./build/rel-host-cpu-gcc/benchmark/memcpy/memcpy_alpaka --repeat 4 --samples 3

# CUDA (sm_70 -> Tesla V100)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target memcpy_alpaka -j
CUDA_VISIBLE_DEVICES=1 ./build/rel-cuda-nvcc-gcc/benchmark/memcpy/memcpy_alpaka --repeat 8192 --samples 5
```

## Self-validation

`src/helper.hpp` defines a deterministic, position-dependent byte pattern (`patternByte`). The
source buffer is filled with the pattern once. Before each correctness copy the destination is
poisoned with the bit-wise inverse pattern (`b ^ 0xFF`), then one real copy is performed, then the
destination is read back and compared byte for byte with `verifyBytes()`.

Because the destination starts as the inverse of the expected pattern, a copy that is skipped,
truncated, shifted, or partially applied leaves at least one poison byte and fails at a well-defined
offset. The comparison is exact (tolerance 0). A payload of arbitrary bytes interpreted as `float`
is frequently NaN, so a NaN/Inf test would be meaningless; the byte-exact comparison is strictly
stronger and is the tolerance rationale. A nonzero exit code signals failure.

Validated sizes include odd and non-multiple-of-width values and boundary values:
`1, 2, 3, 7, 15, 63, 64, 65, 127, 255, 256, 257, 511, 512, 513, 1000, 1023, 1024, 1025, 4095,
4096, 4097, 65535, 65536, 65537, 1048575, 1048576, 1048577` in all three modes, on both the CPU and
the CUDA backend.

The CMake test registration runs a fast subset:

```bash
ctest --test-dir build/rel-host-cpu-gcc -R memcpy_alpaka --output-on-failure
```

## Timing scope and reference reproduction

The measured metric is the per-copy transfer time: `repeat` back-to-back enqueues between two
synchronisations, divided by `repeat`. This matches the HeCBench reference. One warm-up batch of
`repeat` copies precedes the measured samples. Runtime ratio = median_alpaka / median_reference;
`<=1.00` match, `<=1.10` accepted, `>1.10` requires investigation.

Measured on a Tesla V100-SXM2-32GB (`CUDA_VISIBLE_DEVICES=1`) at `--repeat 8192 --samples 5`:

| mode | size | alpaka median (us) | ref median (us) | ratio |
|------|------|--------------------|-----------------|-------|
| H2D  | 64 B | 1.5580 | 1.5009 | 1.038 |
| D2H  | 64 B | 7.1503 | 7.0886 | 1.009 |
| H2D  | 64 KiB | 15.4948 | 15.5265 | 0.998 |
| D2H  | 64 KiB | 23.7103 | 23.7686 | 0.998 |
| H2D  | 2 MiB | 392.02 | 392.24 | 0.999 |
| D2H  | 2 MiB | 472.36 | 474.51 | 0.996 |
| H2D  | 64 MiB | 13376.1 | 13561.5 | 0.986 |
| D2H  | 64 MiB | 13699.9 | 14685.5 | 0.933 |

Device-to-device has no reference counterpart; it reaches 639 GB/s at 2 MiB and 403 GB/s at 64 MiB.
The raw per-run samples, reference provenance and full repro commands are in the
machine-readable [`results.json`](results.json) in this directory.

## Limitations and performance exception

- The pinned reference has no device-to-device mode, so D2D is validated against the host reference
  pattern only and has no runtime ratio.
- The reference stops at 2 MiB. The `--sizes 64,2097152,67108864` run demonstrates a "larger" size
  (64 MiB) safely within the 32 GiB device; 32 GiB is not attempted because a device-to-device test
  needs two payloads and the host buffer must be allocated as well. `--max-bytes` prevents an
  out-of-memory request.
- The OpenMP reference is an OpenMP *target-offload* program meant for `icpx`, which is absent on
  rosi_v100. Built with `g++ -fopenmp` it falls back to the host and moves no data (per-copy times
  below 0.01 us), so it is not a usable performance baseline. CPU correctness is validated against
  the byte-exact host reference; the GPU comparison uses the CUDA reference.
