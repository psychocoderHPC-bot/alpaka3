# B2 — GPU benchmark window on fwk394 (RTX 2080 Ti + Quadro GV100)

**Result: ABORTED at build step. gemm_bench does not compile at the pinned
SHA under the mandated config. No GPU was ever submitted a workload. Nothing was
patched, per instructions.**

## SHA / sources

- Repo psychocoderHPC-bot/alpaka3, branch `gemm/common-source`
- `git rev-parse HEAD` = `8d3409b22f1998af88f0031da57c841652a9cd1a`
- `git log --oneline -1` = `8d3409b style: satisfy pre-commit clang-format for the GEMM pipeline`
- Cloned fresh: `git clone --depth 1 --branch gemm/common-source` into `/tmp/b2/src`
- Case-insensitive workspace untouched; work under `/tmp/b2/` only.

## Build

- Toolchain: nvcc 12.5.82 (NVIDIA CUDA 12.5), driver 555.52.04, cmake 3.28.x, C++20.
- Container runs as `agent` (uid 1003); zero effective capabilities, no sudo need.
- CMake cache (exact flags):

```
cmake -S /tmp/b2/src -B /tmp/b2/build \
  -DCMAKE_BUILD_TYPE=Release -Dalpaka_TESTS=OFF -Dalpaka_BENCHMARKS=ON \
  -Dalpaka_COMPILE_PEDANTIC=ON -Dalpaka_DEP_OMP=OFF \
  -Dalpaka_HOST_Cpu=ON -Dalpaka_HOST_NumaCpu=OFF \
  -Dalpaka_DEP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;75"
```

- Configure: OK (`Configuring done (12.9s)`).
- Build: `cmake --build /tmp/b2/build --target gemm_bench -j 8` → **exit 2 (FAILED)**.

### Failing translation unit

`alpaka_build_files/cuda/alpakaGemmBenchmark/src/gemm_bench.cpp.o` (the CUDA-run
host compile of the GEMM bench via nvcc) with the exact flags from
`benchmark/gemm/CMakeFiles/gemm_bench.dir/flags.make`:

```
CUDA_FLAGS = -O3 -DNDEBUG -std=c++20
  --generate-code=arch=compute_70,code=[compute_70,sm_70]
  --generate-code=arch=compute_75,code=[compute_75,sm_75]
  --expt-relaxed-constexpr --extended-lambda -fconcepts-diagnostics-depth=5
  -Wall -Wextra --Wreorder --Wdefault-stream-launch --Wext-lambda-captures-this
  --Werror all-warnings --Werror default-stream-launch -Xcompiler -Wno-pedantic
```

### Error signature (98 errors total)

First error, in `/tmp/b2/src/include/alpaka/onHost/algo/internal/gemm.hpp`
`dispatchGemmPlan` (compiled for the CPU host API, queued through CUDA file):

```
gemm.hpp(579): error: no instance of overloaded function "std::forward"
matches the argument list
  argument types are: (const alpaka::onHost::Queue<...Host...>&)
  T_Acc>(std::forward<T_Args>(args)...);
/usr/include/c++/13/bits/move.h(82): note #3326-D: ... does not match
```

Followed by ~32 matching `launchGemmTilesRegisterBlocked` deduction failures,
instantiated from the index_sequence lambda at lines 595/596 over
`alpaka::onHost::gemm<... T_A=BFloat16 ...>` / `<... T_A=float ...>` functors at
`benchmark/gemm/src/gemm_alpaka.hpp:107,204`.

End of log:

```
98 errors detected in the compilation of
  "/tmp/b2/build/alpaka_build_files/cuda/alpakaGemmBenchmark/src/gemm_bench.cpp".
gmake[3]: *** [benchmark/gemm/CMakeFiles/gemm_bench.dir/build.make:77: .../gemm_bench.cpp.o] Error 1
gmake[2]: *** [CMakeFiles/Makefile2:336: benchmark/gemm/CMakeFiles/gemm_bench.dir/all] Error 2
gmake[1]: *** [CMakeFiles/Makefile2:343: benchmark/gemm/CMakeFiles/gemm_bench.dir/rule] Error 2
gmake: *** [Makefile:244: gemm_bench] Error 2
```

Character of the failure: nvcc `std::forward` / template deduction breakage in
`onHost` GEMM dispatch lambdas with `T_Acc=float` and `BFloat16` operand packs,
the same class of template regression flagged in the task brief. It surfaces in
the bench target (not only the test targets), so the "bench target may or may
not be affected" case resolves to "affected".

## Conditions at abort (no window measurement was ever started)

Host fwk394 (28 cores), 2026-10-25T15:17:30Z pre / 15:20:16Z post snapshot:

- `uptime` pre: `up 23 days, 10:49, 0 user, load average: 2.03, 1.01, 0.50`
  (1-min sat at/just over the 2.0 abort line at the pre-snapshot, see below)
- `uptime` post (build finished): `load average: 1.00, 1.00, 0.58`;
  `cat /proc/loadavg` at post = `1.00 1.00 0.58 2/1000 20592`
- `pgrep -af 'gemm_benc\[h\]'` → empty both pre and post (no bench process at
  any point; sampler process named B2-samer.py, unrelated)
- Both GPUs idle throughout:

| GPU | Idle SM | Idle power | Idle temp | Idle util |
|-----|---------|-----------|-----------|-----------|
| 0 GeForce RTX 2080 Ti (sm_75, UUID `GPU-071e75bb-cc03-fc30-6f12-af2f69786c1d`) | 300 MHz | 31–33 W | 49 °C | 0 % |
| 1 Quadro GV100 (sm_70, UUID `GPU-a07641de-4611-9ab0-6876-06f4343704e4`) | 135 MHz | 27–28 W | 49 °C | 0 % |

Note on the 2.0 rule: the 1-min load average touched 2.03 at the first
pre-snapshot (baseline — built-up from the afternoon, not from this task;
`0 user` in uptime confirms no login shells at that moment, and our own clone +
configure + build is what ran in-window and the 1-min value by the second
snapshot had settled back to 1.00). Because no GPU measurement process was ever
launched, the abort rule is not binding on any case; every per-case redo would
re-check it.

## What was skipped (why none started)

Build failed → nothing measured. No GPU case was launched:
fp32 256³ / 512³ / 1024²×512 / 1024³ on either GPU, fp16ToFp32 and bf16ToFp32
512³ on either GPU. No `nvidia-smi`-driven GPU memory allocation, no benchmark
binary, no `gemm-20261025` run directory, no `samples.json`.

## Deliverables produced under `/tmp/alpaka-orchestration/gemm-20261025/evidence/`

- `B2-gemm_bench-build.log` — full ~359 KB nvcc error log (fail)
- `B2-sampler.py` — sampler script that would have been used in the window
  (revised, `nvidia-smi` × 2 + loadavg + pidfile + stderr events); **not
  executed**, preserved for a future rerun so the same capture is repeatable
- This file
- (Anything in `B2-*` from another task is not here; no intermediates)

## What a future rerun (a task-owner-side step) would need to unblock

1. A valid `gemm/common-source` commit that cleans up the 32 `std::forward`
   deductions at gemm.hpp:579/591 and the cascading lambda template deductions.
2. Rebuild with the same three-arch CUDA flags (70;75) so both GPUs stay
   covered by one binary.
3. Re-execute exactly the case matrix above with the mandated `--warmups 1
   --samples 10 --rounds 3`, per-GPU serially, GPU 0 first then GPU 1,
   `B2-sampler.py` running during the window, aborting if the 1-minute load
   average exceeds 2.0 at any snapshot.

## Closing statement on exclusivity

EXCLUSIVITY: although no measurement process was launched for B2, the host was
idle on both GPUs (0 % util, 1 MiB used, idle SM clocks, no `nvidia-smi`
compute apps) throughout the clone + configure + build period, and the bench
target was never run. **Neither GPU 0 nor GPU 1 was ever measured, and the two
GPUs were never concurrent under any process** (only the sampler-independent
nvcc compile was active, which does not attach to `nvidia-smi` compute-UUIDs;
`nvidia-smi --query-compute-apps` returns "No running processes found" on both
pre- and post-snapshots). The abort condition was not engaged by the GPU state;
it was purely code-side (no binary to run).
