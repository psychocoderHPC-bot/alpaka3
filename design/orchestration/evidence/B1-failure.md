# B1 - GPU benchmark window on rosi_v100 - BLOCKED (build failed, no measurements)

Window: 2026-10-25 15:16Z - 15:34Z UTC (build + probes only; no measurement runs).
Status: gemm_bench does not build at the required SHA. No source edits attempted.

## SHA

Required SHA: 8d3409b22f1998af88f0031da57c841652a9cd1a
Checked out SHA: 8d3409b22f1998af88f0031da57c841652a9cd1a
Branch: gemm/common-source (branch tip == required SHA, verified via
git ls-remote at 2026-10-25T15:2xZ, so no newer "nvcc dispatch fix" commit
exists to note)
Commit message: "style: satisfy pre-commit clang-format for the GEMM pipeline"

## Host & device

Target: rosi_v100 (gv001)   [Slurm --exclusive: nodes=1, cpus=48, gpus=4/300GB]
OS inside dev container: Ubuntu 24.04 (apptainer shell, root)
CUDA 12.6 (nvcc Build cuda_12.6.r12.6/compiler.35059454_0), driver 570.158.01
CMake 3.28.3, Ninja 1.11.1, g++ 13.3.0

Device for the measurement window (CUDA_VISIBLE_DEVICES=0, single device):
  GPU 0: Tesla V100-SXM2-32GB, 32768 MiB
         UUID GPU-84d5313e-c8b7-15cf-c276-683b2c6f5e26
  (No run happened, so no clocks/peak utilization were recorded; the device
  was idle throughout the window, 0 % on all GPUs.)

Other GPUs (not touched): GPU-061e1427-c194-0ceb-0961-e6a4e54591d7,
GPU-bc53acac-6d2e-fdea-3cf7-2a0e31b1ef53, GPU-1430de85-bec0-42c4-96d6-3a94bc3d77b9.

Peak references (V100-SXM2-32GB, labelled, not claims):
  FP32 CUDA-core peak    ~15.7 TFLOP/s (the value the task marks as the yardstick)
  FP16 tensor-core peak  ~125 TFLOP/s (dense, reference only)

## Build result

BUILD FAILED - gemm_bench cannot be built at SHA 8d3409b22f1998af88f0031da57c841652a9cd1a.

CMake configuration: SUCCESS (CMAKE_RC=0).
CMake log head:
  -- The CXX compiler identification is GNU 13.3.0
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  ... (full log in B1-cmake-config.log)

Ninja build: FAILED
  ninja -C build -j32 gemm_bench
  -> 1481 lines in build.log, grep -c 'error:' = 98
  -> last line: "ninja: build stopped: subcommand failed."
  -> binary not produced; no gemm_bench ELF under build tree.

First errors (build.log lines 6/23/40):
  /tmp/bx/src/include/alpaka/onHost/algo/internal/gemm.hpp(579):
    error: no instance of overloaded function "std::forward" matches the argument list
  /tmp/bx/src/include/alpaka/onHost/algo/internal/gemm.hpp(591): same
  /tmp/bx/src/include/alpaka/onHost/algo/internal/gemm.hpp(583):
    error: no instance of function template "alpaka::onHost::internal::launchGemmTilesRegisterBlocked"

Why this is not fixable in-window:
  - The failing TUs are the GEMM internal header (onHost/algo/internal/gemm.hpp)
    included by gemm_bench.cpp through the public API.
  - Per B1 task rules, source must not be patched at this step; the fix must
    come from the nvcc dispatch backport (M1), not from this window.
  - The task explicitly specified "report that and stop" for this failure mode.

## Conditions & window discipline

Pre-build probe (15:16 UTC): pgrep -af 'gemm_benc[h]' -> no processes;
nvidia-smi utilization 0 % on all 4 GPUs; uptime clean.
Post-build probe (15:35 UTC), after the failed build and session close:
  uptime:  15:35:32 up 193 days, 4:11, load average: 1.02, 0.80, 0.55
  nvidia-smi compute apps: (none)
  nvidia-smi utilization: all GPUs 0 %
  1-min load 1.02 is well below the 2.0 abort threshold -> no abort
  triggered; the window was clean before, during and after the build.
No gemm_bench process was running at any point (the only processes seen
were this task's own probes). Clocks: not captured, because no measurement
run was executed (there is nothing whose clocks to report).

## Exclusivity statement

Target rosi_v100 was allocated via Slurm --exclusive (resolved_plan:
nodes=1, cpus_per_node=48, gpus_per_node=4, exclusive=true).
During this window (15:16-15:34 UTC), loadavg at the end of the window was
1.20 0.69 0.48 at 48 cores, which is below the 2.0 abort threshold.
No other gemm_bench process was present at any point (checked pre-build and
via session logs; only the build log on one session).
GPU utilization held at 0 across the window (no measurement run, only a
failed build); the window was clean.

Note on exclusivity vs other targets on the gateway: hal, jureca, fwk388 and
fwk394 were concurrently connected via the computeMCP gateway, but they are
physically separate machines and do not share resources with rosi_v100.
The SLURM --exclusive flag guarantees no other job shares this node.

## Skipped

All 6 measurement cases are SKIPPED because the binary could not be built:
  - fp32   256x256x256, 512x512x512, 1024x1024x512, 1024x1024x1024
  - fp16ToFp32  512x512x512
  - bf16ToFp32  512x512x512

## Artifact set

  /tmp/bx/cmake.log          full CMake configuration log
  /tmp/bx/build.log          full ninja build log (1481 lines, 98 errors)
  /tmp/bx/src/               checkout (no source edits; pristine at SHA)
  /tmp/bx/build/             partial build tree (no gemm_bench binary)

Evidence copy in this directory:
  B1-alpaka-orchestration/gemm-20261025/evidence/B1-failure.md  (this file)

## Recommendation

Re-attempt task after M1 (nvcc dispatch backport) has been merged into
gemm/common-source and the branch tip advances past
8d3409b22f1998af88f0031da57c841652a9cd1a.  The next B run should:
  1. Re-clone at the backported SHA (no source edits).
  2. Re-run this exact build (the same CMake invocation, already validated).
  3. Open the window identically ( CUDA_VISIBLE_DEVICES=0, nvcc -j
     for build only, no builder during measurement, idle GPUs).
  4. Run the same 6 cases with the same --warmups 1 --samples 10 --rounds 3.
