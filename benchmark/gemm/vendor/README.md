# alpakaVendor reference baseline (benchmark only)

`gemm_vendor_bench` is an optional, separate executable that measures the
external [alpakaVendor](https://github.com/psychocoderHPC/alpakaVendor) BLAS
wrapper on the same `benchmark/gemm/workloads.json` shapes as `gemm_bench`.
It exists to cross-check the in-tree common-source `onHost::gemm` against a
vendor library. It is not part of the library and is never linked into it.

The integration is off by default. With `ALPAKA_GEMM_VENDOR_BASELINE=OFF` no
dependency is fetched, no extra target is created, and `gemm_bench` is
unchanged apart from the shared harness reporting optional provenance fields.

## Dependency and license

- `alpakaVendor` is a **separate dependency**, not a submodule and not vendored
  into this repository. CMake fetches it with `FetchContent` at the pinned
  commit `ccf128e3c1ff85ae0b1fc5d9b08fd9bc44f02aeb` (branch `dev`).
- It is licensed under **MPL-2.0**. It stays a separate work; this repository
  only links its header-only INTERFACE target into the benchmark executable.
- The baseline requires an OpenBLAS development package with a `pkg-config`
  module named `openblas` (Debian/Ubuntu: `libopenblas-dev`).

## Configure and build (option ON, OpenBLAS host backend)

Build from the checkout so the baseline compiles against this tree's alpaka3
headers rather than alpakaVendor's pinned alpaka3:

```bash
cmake -S benchmark/gemm -B build/gemm-vendor-baseline \
  -DCMAKE_BUILD_TYPE=Release \
  -Dalpaka_USE_SOURCE_TREE=ON \
  -DALPAKA_GEMM_VENDOR_BASELINE=ON
cmake --build build/gemm-vendor-baseline --target gemm_vendor_bench -j
```

If OpenBLAS is missing, configuration stops with a message that names the
missing `openblas` pkg-config module; it does not fall back or fabricate data.

## Run

```bash
./build/gemm-vendor-baseline/vendor/gemm_vendor_bench \
  --manifest benchmark/gemm/workloads.json \
  --output-dir /tmp/alpaka3-gemm-vendor/run-$(date +%Y%m%dT%H%M%S) \
  --revision "$(git rev-parse HEAD)" \
  --target cpu-host \
  --subset all \
  --samples 10 \
  --rounds 3
```

`--smoke` and `--filter` work exactly as in `gemm_bench`. The output file is
`samples.json` with the same schema, plus two fields per record:

- `"backend": "vendor:openblas"`
- `"device_name": "<alpaka host device name>"`

Records whose numerical mode the vendor interface cannot express carry
`"status": "vendor_unavailable"` and a `"detail"` reason. This baseline never
emits timings for such records.

## Supported scope in this integration

- Numerical modes: `fp32` and `fp64`. alpakaVendor's BLAS scalar concept covers
  `float`, `double` and alpaka complex values only; there is no fp16, bf16 or
  tf32 entry point, so `fp16_fp32` and `bf16_fp32` are reported as
  `vendor_unavailable`.
- Layouts: row-major dense with a column stride of one element, which covers the
  manifest's `compact` and `padded` stride classes. A/B `none`/`transpose` map to
  alpakaVendor's `Transposed` annotation.
- Backend: OpenBLAS host only. The cuBLAS, rocBLAS and oneMKL dispatches exist
  in alpakaVendor but require a backend-specific device in this functor; they
  are a follow-up, not wired here.
- The vendor interface has no capability predicate and no fallback. All
  availability decisions are made in `vendor_baseline.cpp` from the manifest
  mode and the compiled backend.

## Build the in-tree benchmark with the option OFF

```bash
cmake -S benchmark/gemm -B build/gemm-bench \
  -DCMAKE_BUILD_TYPE=Release \
  -Dalpaka_USE_SOURCE_TREE=ON
cmake --build build/gemm-bench --target gemm_bench -j
```
