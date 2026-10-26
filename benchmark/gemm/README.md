# GEMM benchmark harness

This directory holds the reusable benchmark tooling for the portable GEMM work.
The committed content is the workload manifest, the timing driver skeleton, and
the plotting script. Measured samples, figures, and profiler output stay in an
external run directory and are never committed.

`workloads.json` is the frozen workload manifest. Its `schema_version` changes
whenever shapes, numerical modes, layout classes, or alpha/beta cases change
after freezing. `ci_fast_subset` is a bounded CI selection; it is not a release
validation set.

## Build the M0 harness

The driver is portable C++20 and needs no GPU backend to compile. Build it on
its own:

```bash
cmake -S benchmark/gemm -B build/gemm-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEMM_BENCH_WERROR=ON
cmake --build build/gemm-bench --target gemm_bench -j
```

Add `-Dalpaka_COMPILE_PEDANTIC=ON` when configuring from the repository root so
the harness follows the same warning policy as the alpaka targets.

## Host-native ISA for performance measurements

The default build stays baseline-portable: the library is header-only and never
forces ISA flags on its consumers, and no target receives `-march=native`
implicitly. The planner and the register-blocked kernel evaluate the compiled
SIMD width in the consuming translation unit, so a default build plans and runs
with the baseline vector width.

Performance numbers require the opt-in host-native ISA for the GEMM targets:

```bash
cmake -S benchmark/gemm -B build/gemm-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DALPAKA_GEMM_NATIVE_ISA=ON
```

`ALPAKA_GEMM_NATIVE_ISA` (default `OFF`) applies `-march=native` to the host C++
translation units of `gemm_bench` and the GEMM unit tests only, using GCC/Clang
syntax. Device passes of accelerator builds keep their own flags. Correctness
results must be identical with and without the option; only the code generation
changes. Do not use this option for portable CI artifacts that are distributed
across machines.

## Smoke run (placeholder functor)

The M0 driver ships a placeholder `GemmFunctor` that reports
`not_implemented`; it does not fabricate timings. Write the smoke output outside
the checkout:

```bash
./build/gemm-bench/gemm_bench \
  --manifest benchmark/gemm/workloads.json \
  --output-dir /tmp/alpaka3-gemm-benchmark/smoke \
  --revision "$(git rev-parse HEAD)" \
  --target cpu-host \
  --subset ci-fast \
  --max-cases 2 \
  --smoke
```

Inspect the result:

```bash
python3 -m json.tool /tmp/alpaka3-gemm-benchmark/smoke/samples.json | head
```

## Timed run

Connect a real GEMM implementation by implementing `gemm::bench::GemmFunctor`
in `src/gemm_interface.hpp` and passing it to `runBenchmark()` in
`src/gemm_bench.cpp`. Timed runs enforce at least 10 samples per round over at
least 3 rounds, warm up once, synchronize after every repetition, and report
median/min/max wall time plus GFLOP/s from useful `2*M*N*K` operations.

```bash
./build/gemm-bench/gemm_bench \
  --manifest benchmark/gemm/workloads.json \
  --output-dir /tmp/alpaka3-gemm-benchmark/run-$(date +%Y%m%dT%H%M%S) \
  --revision "$(git rev-parse HEAD)" \
  --target cpu-host \
  --subset all \
  --samples 10 \
  --rounds 3
```

The driver refuses an output directory inside the source checkout.

## Device selection

The adapter picks the device from the command line. The default is the host
CPU, so an existing invocation keeps measuring the CPU:

```bash
--device-kind cpu          # alpaka Host API, Cpu device kind (default)
--device-kind cuda         # CUDA API, NVIDIA GPU
--device-kind hip          # HIP API, AMD GPU
--device-kind oneapi-cpu   # oneAPI API, CPU device
--device-kind oneapi-gpu   # oneAPI API, first available Intel/NVIDIA/AMD GPU
--executor NAME            # optional alpaka executor, e.g. CpuSerial or GpuCuda
```

Without `--executor` the device's default executor is used. The executor name
is matched case-insensitively against the executors compiled into the build.
Devices are only instantiated for toolkits enabled at configure time. A
requested combination that is not compiled in or has no available device fails
with a nonzero exit status; the adapter never falls back to the CPU.

Every run prints one identity line before timing:

```text
executor=GpuCuda api=Cuda device=NVIDIA A30 kind=NvidiaGpu
```

`samples.json` records the same identity at document and record level in the
`executor`, `api`, `device_kind`, `device_name` and `adapter` fields. The
adapter also checks that the created device reports the requested API and
device kind and aborts on a mismatch.

**A CPU-only build cannot produce GPU numbers.** If CUDA, HIP or oneAPI were
not enabled at configure time, `--device-kind cuda` (or the corresponding GPU
kind) exits with an error. Select the backend at configure time as well, for
example with the root configuration options, and confirm from the printed
identity line which device actually ran.

## Staging policy and plan reporting

The register-blocked kernel has two operand supply policies. The default is the
shipped compile-time configuration, so an existing invocation keeps its plan and
its timings:

```bash
--staging automatic    # default: resolved by the shipped compile-time policy
--staging direct       # read A and B from global memory in the K loop
--staging cooperative  # stage each A/B panel in block-shared memory
```

`automatic` resolves exactly as before. `direct` and `cooperative` are explicit
requests to the planner: they select only plans with that policy and never fall
back to the other one. An explicit `cooperative` request that no legal plan can
satisfy (for example because the bounded stage ring exceeds the device's
block-shared memory) exits nonzero and prints the planner's rejection reason
instead of silently running the direct path.

Every executed case prints one plan line before warm-up and timing:

```text
plan case=square-64 tileM=8 tileN=8 WR=1 WC=1 BK=16 stages=1 policy=direct simd=8 scratch=0
```

The same values are recorded in `samples.json` under the record's `plan` object:
`tile_m`, `tile_n`, `worker_rows`, `worker_cols`, `k_block`, `stages`,
`policy`, `simd_width` and `scratch_bytes` (the block-shared staging bound, zero
for a direct plan). They are copied from the planner diagnostics of the actual
run, so a measurement log always identifies the configuration that produced the
number. The record's existing `executor`, `api`, `device_kind`, `device_name`
and `adapter` fields carry the device identity. A case that does not select a
plan, or an external baseline, has no `plan` object.

## Plots

```bash
python3 benchmark/gemm/plot.py \
  /tmp/alpaka3-gemm-benchmark/run-*/samples.json \
  --output-dir /tmp/alpaka3-gemm-plots
```

Pass one file for a throughput chart. Pass two or more; the first file is the
baseline for `ratio.png`. The script refuses to write inside the checkout unless
`--allow-inside-repo` is given explicitly.

## Acceptance checker

`validation/check_acceptance.py` consumes the requirements YAML and an external
run root containing `acceptance.json` and `samples.json` files. It exits nonzero
for missing required entries, stale source SHAs, unapproved substitutions,
unresolved required review findings, failing thresholds, and unapproved skips.
See its module docstring for the evidence schema.

```bash
python3 validation/check_acceptance.py \
  --requirements validation/gemm-requirements.yaml \
  --run-root /tmp/alpaka3-gemm-evidence/<revision> \
  --head-sha "$(git rev-parse HEAD)"
```

Unit tests:

```bash
python3 -m pytest validation/tests/test_check_acceptance.py -q
```
