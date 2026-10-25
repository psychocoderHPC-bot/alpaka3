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
