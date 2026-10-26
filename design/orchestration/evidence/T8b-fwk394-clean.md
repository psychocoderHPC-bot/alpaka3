# T8b — fwk394 clean GPU validation + NUMA capability proof

Date: 2026-10-25, executed by primary agent (no delegation)
Source SHA256 fingerprint: 6-keyfile, identical to /tmp/alpaka-orchestration/gemm-20261025/evidence/src/local-hash.txt
  Verified on both fwk394 and hal before each run.

## fwk394 hardware
  GPU 0: NVIDIA GeForce RTX 2080 Ti (sm_75), UUID GPU-071e75bb-cc03-fc30-6f12-af2f69786c1d
  GPU 1: Quadro GV100 (sm_70), UUID GPU-a07641de-4611-9ab0-6876-06f4343704e4
  CPU: Intel Xeon W-2175, 1 NUMA node, default ulimit -l = 64 (MB of locked memory)
  nvcc: 12.5, container CapEff = 0 (zero effective capabilities)

## Multi-arch build (fixes NoKernelImage)

CMake -DCMAKE_CUDA_ARCHITECTURES="70;75" -Dalpaka_DEP_CUDA=ON -Dalpaka_TESTS=ON -Dalpaka_BENCHMARKS=ON
  -Dalpaka_COMPILE_PEDANTIC=ON -Dalpaka_DEP_OMP=OFF -Dalpaka_HOST_Cpu=ON

Build dir: /tmp/t8a/build/cuda-70-75 (fresh, ninja, 272 targets)

### ctest -R lockstep, per-GPU, CUDA_VISIBLE_DEVICES selected

| GPU | CUDA_VISIBLE_DEVICES | Device (nvidia-smi) | pass | fail | skip | time |
|-----|---------------------|---------------------|------|------|------|------|
| 0   | 0                   | RTX 2080 Ti (sm_75) | 10   | 5    | 0    | 1.60 s |
| 1   | 1                   | Quadro GV100 (sm_70)| 10   | 5    | 0    | 2.84 s |

All 5 failures on both GPUs are the same: `TestBackends - 1` = Host NumaCpu,
`hwloc_set_area_membind failed: Operation not permitted`.
The NoKernelImage errors from T8a are **gone**.
TestBackends - 2 (GPU) passes on both devices (745 assertions each).
TestBackends - 0 (Host Cpu) passes on both devices.

## NUMA exclusion (repo-native option)

Option found in cmake/alpakaHwloc.cmake line 50-53:
  option(alpaka_HOST_NumaCpu "Enable support for api::host and deviceKind::numaCpu ..." ON)
When OFF, defines ALPAKA_DISABLE_Host_NumaCpu (include/alpaka/onHost/DeviceSpec.hpp:105),
removing NumaCpu from allBackends. No ctest label needed.

Build dir: /tmp/t8a/build/cuda-70-75-nohwloc
  cmake ... -Dalpaka_HOST_NumaCpu=OFF -DCMAKE_CUDA_ARCHITECTURES="70;75" (same as above)

| GPU | CUDA_VISIBLE_DEVICES | Device (nvidia-smi) | pass | fail | skip | time |
|-----|---------------------|---------------------|------|------|------|------|
| 0   | 0                   | RTX 2080 Ti (sm_75) | 10   | 0    | 0    | 1.35 s |
| 1   | 1                   | Quadro GV100 (sm_70)| 10   | 0    | 0    | 2.29 s |

10 tests (5 lockstep cases × 2 backends: CpuSerial/Host-Cpu + GpuCuda).
No remaining failures.

## NUMA failure is environmental — not code

Container capability state (fwk394):
  CapEff:   0000000000000000   (zero effective caps)
  CapBnd:   00000000a80425fb   (bounding set lacks CAP_SYS_RESOURCE bit 24)
  ulimit -l: 64 (MB)           (RLIMIT_MEMLOCK default, not unbounded)

hwloc_set_area_membind requires CAP_SYS_RESOURCE (or the memory region must be
pinned within RLIMIT_MEMLOCK). The container has neither, so the call fails with
EPERM before any lockstep kernel launches. The exact error line from the test:
  hwloc_set_area_membind failed: Operation not permitted

hal (A30, RX 7900, Arc) does not exhibit this error because its container
grants the required capability; NumaCpu tests pass there (15/15).

## hal reproducibility (same source fingerprint)

SHA256 of 6 key files on hal /tmp/t8a/src: identical to local-hash.txt.

| Target | Build dir            | Backend  | pass | fail | time |
|--------|---------------------|----------|------|------|------|
| hal    | /tmp/t8a/build/cuda-a30    | CUDA sm_80 (A30)     | 15   | 0    | 2.48 s |
| hal    | /tmp/t8a/build/hip-rx7900  | HIP gfx1100 (RX 7900)| 15   | 0    | 1.43 s |

Same counts as T8a. Reproducible.

## Summary

- NoKernelImage: **eliminated** by multi-arch 70;75 build.
- NumaCpu EPERM: **proven environmental** (CapEff=0, no CAP_SYS_RESOURCE, RLIMIT_MEMLOCK=64MB).
  Eliminated by repo-native -Dalpaka_HOST_NumaCpu=OFF.
- Remaining unexplained failures: **none**.
