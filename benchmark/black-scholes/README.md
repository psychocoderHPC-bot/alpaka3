# black-scholes Benchmark

Portable alpaka port of the HeCBench `black-scholes` benchmark. It prices a long stream of
European call and put options with the Black-Scholes analytic engine ported from QuantLib, then
verifies every price against an independent double-precision host reference. A single compute source
drives the host CPU backend and the CUDA backend; there is no native CUDA/HIP kernel and no backend
branch.

## What it computes

The pinned HeCBench reference (`src/black-scholes-cuda/blackScholesAnalyticEngine.cu`, with the
OpenMP twin `src/black-scholes-omp/blackScholesAnalyticEngine.cpp`) generates an option stream of
50,000,000 entries where entry `i` is `settings[i mod 37]`, then prices every entry:

- each setting is a plain-vanilla `CALL` or `PUT` with strike, spot, dividend yield `q`, risk-free
  rate `r`, time to maturity `t` and Black volatility `vol`;
- the per-option pipeline is the QuantLib Black calculator: dividend discount `exp(-q*t)`, risk-free
  discount `exp(-r*t)`, forward `spot*dividendDiscount/riskFreeDiscount`, standard deviation
  `sqrt(vol^2*t)`, `d1 = log(forward/strike)/stdDev + stdDev/2`, `d2 = d1 - stdDev`;
- the normal CDF and density use the reference's hand-rolled single-precision Cody-Waite `erf`
  approximation (`errorFunct`) with the constants from `errorFunctConsts.cuh`;
- the price is `riskFreeDiscount * (forward*alpha + strike*beta)` with the call/put `alpha`/`beta`
  conventions of `initBlackCalcVars`.

This port keeps the reference algorithm and single-precision arithmetic exactly. One work item owns
one option, so the same source is correct on the host CPU and on NVIDIA GPUs.

## Provenance

- Reference: [ORNL/HeCBench](https://github.com/ORNL/HeCBench) at commit
  `0a17ee47ca6072192101036d4bf292e6612f22e6` (2026-10-09).
  - CUDA harness: `src/black-scholes-cuda/blackScholesAnalyticEngine.cu`
    (blob `51fc19ea16e906c34bf0d4263ede703f303302cd`)
  - CUDA kernels: `src/black-scholes-cuda/blackScholesAnalyticEngineKernels.cu`
    (blob `668c24d9f0b58e6d7a657e72ce03b7302276268c`)
  - CUDA kernels header: `src/black-scholes-cuda/blackScholesAnalyticEngineKernels.cuh`
    (blob `215cb5b8a3d452757f848e8cd5b24c9e34ab850a`)
  - CUDA CPU kernels: `src/black-scholes-cuda/blackScholesAnalyticEngineKernelsCpu.cu`
    (blob `5133adc1ab6b5cd18ee727a5b809466119c7600c`)
  - CUDA CPU header: `src/black-scholes-cuda/blackScholesAnalyticEngineKernelsCpu.cuh`
    (blob `14ca159479e0626aff8c369a18251bf058c5a523`)
  - Structs: `src/black-scholes-cuda/blackScholesAnalyticEngineStructs.cuh`
    (blob `1dcfa47585c0cc409c0f4b6f3ef4d83ab4c12167`)
  - Constants: `src/black-scholes-cuda/errorFunctConsts.cuh`
    (blob `90e2056c1e9fe6a112c326204f2edf4d1ad6811f`)
  - OpenMP harness: `src/black-scholes-omp/blackScholesAnalyticEngine.cpp`
    (blob `c795c3efadee9ba09b75af79311ca4825f2ac452`)
  - OpenMP kernels: `src/black-scholes-omp/blackScholesAnalyticEngineKernels.cpp`
    (blob `01c63443514ddd50b12dcf9e50dc10e2f3e300a3`)
  - OpenMP CPU kernels: `src/black-scholes-omp/blackScholesAnalyticEngineKernelsCpu.cpp`
    (blob `4784072d98680ff08137c705d43b942ceaaafb3e`)
- README entry: HeCBench `README.md`, "black-scholes (cuda)", described as "The Black-Scholes
  simulation (https://github.com/cavazos-lab/FinanceBench)". The source headers credit
  Scott Grauer-Gray and the analytic engine is ported from QuantLib.
- Reference sources are kept outside this repository and are not vendored.
- Licensing: the HeCBench repository is BSD 3-Clause (Copyright (c) 2020 Argonne National
  Laboratory, Zheming Jin); the `black-scholes` sources carry Grauer-Gray/QuantLib lineage. This
  benchmark's files are `BSD-3-Clause` unless stated otherwise (`misc.hpp` notes its differing
  license).

## Parameters, modes, precision

- Named form: `black-scholes_alpaka --options N --repeat R --samples S [--verify-only]`.
- `--options` default `50000000`, matching the reference `main`.
- `--repeat` default `100`, matching the reference `run` target (`./main 100`): kernel launches per
  measured sample.
- `--samples` default `5`: independent measured samples.
- `--verify-only` runs the correctness check without the timing loop.
- Data is single precision (`optionInputStruct` fields and the result), matching the reference. The
  independent oracle is double precision.

## Build and run (reference environment: CUDA 12.6, g++ 13.2, CMake 3.28, Tesla V100 sm_70)

Reference binaries (outside the repo, `/root/bs-work/refsrc/`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
# CUDA reference (pinned HeCBench source)
nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 \
     src/black-scholes-cuda/blackScholesAnalyticEngine.cu -o blackscholes_cuda_ref
CUDA_VISIBLE_DEVICES=0 ./blackscholes_cuda_ref 100

# CPU baseline (pinned HeCBench CPU kernel, parallelised with OpenMP)
g++ -O2 -fopenmp -std=c++17 -I src/black-scholes-omp \
    -include blackScholesAnalyticEngineStructs.h -include cfloat \
    ref_cpu_main.cpp blackScholesAnalyticEngineKernelsCpu.cpp -o blackscholes_cpu_ref
OMP_NUM_THREADS=48 ./blackscholes_cpu_ref 1
```

The pinned `black-scholes-omp` harness is an `icpx` offload program (`omp target teams` with
`-fiopenmp -fopenmp-targets=spir64`); `icpx`/`nvc` are absent, and built with `g++ -fopenmp` its
`target teams` region runs on the host but does not parallelise well and is very slow. The CPU
baseline therefore calls the reference's own `getOutValOptionCpu` kernel directly from an OpenMP
parallel `for`, which is the same computation wide-parallel.

alpaka benchmark (the `-Dalpaka_COMPILE_PEDANTIC=ON` flag promotes warnings such as
`-Wsign-compare` to the `-Werror` failures CI sees):

```bash
# CPU
cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-host-cpu-gcc --target black-scholes_alpaka -j
OMP_NUM_THREADS=48 ./build/rel-host-cpu-gcc/benchmark/black-scholes/black-scholes_alpaka \
  --options 50000000 --repeat 100 --samples 5

# CUDA (sm_70 -> Tesla V100; on a V100 set the GPU with CUDA_VISIBLE_DEVICES)
export PATH=/usr/local/cuda-12.6/bin:$PATH CUDA_HOME=/usr/local/cuda-12.6
cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON .
cmake --build build/rel-cuda-nvcc-gcc --target black-scholes_alpaka -j
CUDA_VISIBLE_DEVICES=0 ./build/rel-cuda-nvcc-gcc/benchmark/black-scholes/black-scholes_alpaka \
  --options 50000000 --repeat 100 --samples 5
```

## Self-validation

`verify()` compares every device price against an independent, deliberately different double-precision
host oracle (`referencePrice` in `helper.hpp`) that evaluates the same Black-Scholes formulas with
`std::erfc` for the normal CDF and shares no code with the port. It rejects any NaN or Inf, reports
the maximum absolute deviation and the first offending index, and exits non-zero on failure. As a
secondary, informational check it also prints the maximum deviation from the published analytic prices
of the 37 settings.

The tolerance is the reference's own absolute tolerance `1.0e-4`, used in the reference `main` to
compare its GPU and CPU results against the published prices. It is not widened. The measured margin
is far below it: the port's maximum deviation from the double oracle is `9.9e-6` on the GPU and
`1.1e-5` on the host, and the deviation from the published prices is `5.4e-5` for both; the reference
prints the same order on the GPU (`Output price at index 25000000: 10.847914` vs CPU `10.847917`).
The dominant residual is the reference's single-precision `erf` approximation, not the port.

Validation covers boundary and irregular sizes on both backends: `7` (smaller than one 256-thread
block), `37` (one full cycle of settings), `3700`, `12345` (not a multiple of the block size),
`37000` and `50000000`.

## Timing scope and results

Timing is per kernel launch, end-to-end on the queue: one measured sample runs `repeat` launches and
the reported value is `total / repeat`. The measured metric (acceptance) is the median of `samples`
independent samples after one warm-up. This is directly comparable to the reference's
`Average kernel execution time on GPU`, which measures the kernel loop only. Validation runs a
separate single launch after the timing loop.

Measured on the exclusive `rosi_v100` node (Tesla V100-SXM2-32GB, CUDA 12.6, nvcc 12.6.85, g++
13.2, 48 CPU cores), 50,000,000 options, `repeat=100`, GPU device 0:

| config | backend | executor | options | alpaka median (ms) | ref median (ms) | ratio | variation | validation |
|---|---|---|---|---|---|---|---|---|
| GPU | Cuda | `GpuCuda` | 50000000 | 2.268 | 2.819 | 0.81 | 0.1% | PASS |
| CPU | Host | `CpuOmpBlocks` | 50000000 | 69.5 | 106.4 | 0.65 | <1% | PASS |
| CPU | Host | `CpuSerial` | 50000000 | 2628.0 | 106.4 | 24.7 | n/a | PASS |

Ratio = `median_alpaka / median_reference`; `<=1.00` matches, `<=1.10` is accepted. Both the GPU
(`GpuCuda`) and the parallel host (`CpuOmpBlocks`) ratios are below 1.0. The `CpuSerial` line is a
single-thread sanity point, not a competitive baseline.

Raw GPU samples (final run, `samples=5`):
`2.26838, 2.26864, 2.26844, 2.26834, 2.26851` ms; interleaved reference rounds: `2.818931`,
`2.805454`, `2.819754`, `2.821242`, `2.820264` ms. The node could not be clock-locked (shared host,
no permission), so a few early samples showed idle-clock outliers; the medians above are stable and
repeat within 1%.

See `results/black-scholes/` for the machine-readable CSV and provenance.

## Limitations / performance exception

None. Both the GPU and the parallel host backend are at or below the reference runtime, so the
benchmark needs no performance exception. The `CpuSerial` executor is intentionally single-threaded
and is not a baseline.
