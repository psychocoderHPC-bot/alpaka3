# Intermittent HIP CI failures on shared gfx1030 (page fault + hang) — investigation report

## Confirmed
- **Alpaka3-side defect (fixed):** `~Device()` used the throwing `ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK` in an implicitly-`noexcept` destructor, turning any HIP error reported during teardown into `std::terminate` (observed as `Subprocess aborted`). Fix PR: https://github.com/psychocoderHPC-bot/alpaka3/pull/9. This is an **amplifier, not the root cause** of the page fault/hang.
- **The failure is HIP-specific in this project's CI.** Over 34 GPU pipelines: HIP 34/206 jobs failed; CUDA 4/226 failed, and every CUDA failure is a Catch2 math-assertion (NaN) failure — no CUDA page fault or hang. All CUDA failures are addressed by the math-test NaN comparison change (alpaka-group/alpaka3#704).
- **The fault string is a generic device page fault.** A native HIP out-of-bounds store on dev-hal prints the identical `Memory access fault by GPU node-N ... Reason: Page not present or supervisor privilege.`
- **alpaka3's test path is not the source of the fault.** Source review at the failing SHA found in-bounds indexing for all 27 functor tuples (capacity 1000, grid covers exactly 1000), no device access to unpinned host memory, no host threads, and correct stream/buffer lifetime ordering.

## Evidence (CI)
```
CI EVIDENCE — alpaka3 HIP intermittent failures
Project: hzdr/crp/alpaka3 (GitLab)
Pipeline: https://gitlab.com/hzdr/crp/alpaka3/-/pipelines/2892995985
Ref: pr-703, commit 9d708ca41aebe10761d56b1d9ceec2d22177dcb8, 2026-09-29
Runner: ci-amd-rocm-1 tGjCdpw7b (system ID s_87e51d4f87db), host ci03
GPU: AMD Radeon PRO W6800, gfx1030, APCI_AMD_GPU_ARCH=gfx1030
Concurrency: 4 GPU jobs on the same host via concurrent-0..3 slots; ctest is serial per job.

FAILING JOB A (abort + page fault)
  job: https://gitlab.com/hzdr/crp/alpaka3/-/jobs/16805240086
  ROCm 7.0 (hip-runtime/libraries), Debug, ctest start 11:39:08Z, end 11:41:39Z
  test #74 mathOpsComplexDouble - TestAccFunctorTuplesComplex - 20
  11:39:44.618  Start 74
  11:39:48.371  74/394 ... Subprocess aborted***Exception:   3.75 sec
  11:39:48.371  Filters: "mathOpsComplexDouble - TestAccFunctorTuplesComplex - 20"
  11:39:48.371  Randomness seeded to: 1450375331
  11:39:48.371  Memory access fault by GPU node-4 (Agent handle: 0x2c9f01b0) on address 0x7a18afcc5000. Reason: Page not present or supervisor privilege.
  11:39:48.371  (next test 75 starts and passes)
  The test process aborted right after the GPU page fault; subsequent tests were unaffected.

FAILING JOB B (hang / no progress)
  job: https://gitlab.com/hzdr/crp/alpaka3/-/jobs/16805240089
  ROCm 6.3 (hipcc6.3 release image), Release, ctest start 11:38:52Z, end 12:07:48Z
  test #85 mathOpsComplexDouble - TestAccFunctorTuplesComplex - 12
  11:39:44.271  Start 85
  12:04:44.411  85/394 ... ***Timeout 1500.14 sec
  12:04:44.411  Filters: "mathOpsComplexDouble - TestAccFunctorTuplesComplex - 12"
  12:04:44.411  Randomness seeded to: 1279827042
  12:04:44.411  (no further output; process produced nothing for 1500 s)
  Normal duration of this test is ~0.35 s (release) / ~0.6 s (debug).

TIMING CORRELATION
  Job A abort: 11:39:48.371Z. Job B hang begins: 11:39:44.271Z (never returns).
  The two failing processes overlapped within ~4 s on the same GPU.
  A third HIP job on the same host (7.14, slot concurrent-3) started ctest ~11:42:00Z and passed.

WITHIN-PIPELINE VERSION MATRIX (same code, same GPU)
  hipcc6.3 release   FAIL (hang, test 85)
  hipcc7.0 debug     FAIL (page fault abort, test 74)
  hipcc6.4 release   pass
  hipcc7.1           pass
  hipcc7.2 debug     pass
  hipcc7.14 release  pass
  hipcc10.0          pass
  The failing tuple index is different each run; index 20 passes in job B and index 12 passes in job A.

CROSS-PIPELINE STATISTICS (34 GPU child pipelines, 2026-07..09)
  HIP  jobs: 34 failed / 206 total (~16.5%)
  CUDA jobs:  4 failed / 226 total (~1.8%)
  All 4 CUDA failures are Catch2 math-assertion (NaN) failures:
    REQUIRE(test::isApproxEqual(results(i), stdExpectedResult)) == false
  No CUDA job has produced a GPU page-fault string or a ctest timeout.

GENERIC FAULT STRING CONFIRMATION (dev-hal, gfx1100, ROCm 7.2.4)
  A native HIP out-of-bounds store prints the identical message:
    Memory access fault by GPU node-1 (Agent handle: ...) on address ... Reason: Page not present or supervisor privilege.
  i.e. the CI line is a generic device page-fault report, not an alpaka-specific message.

EARLIER REPORTS OF THE SAME WINDOW/STRING
  ROCm/hip#3906  "Memory access fault by GPU node-4 after stream creation"
                 gfx1030 W6800, ROCm 7.0 (also 6.4-7.2), >=8 CI runners, still open.
  ROCm/legacy-rocm-build#6527 "deadlock if single device is used with multiple processes"
                 gfx1100, ROCm 7.2.0; maintainer: SDMA queue exhaustion in KFD.
```

## Reproduction attempts on dev-hal (gfx1100, KFD 7.1.3)
# Reproduction notes (dev-hal)

## Target environment
- GPU: AMD Radeon RX 7900 XTX, gfx1100 (NOT the CI's gfx1030 W6800)
- Host KFD/ROCk: 7.1.3.31500000
- Userspace ROCm tested: 7.2.4 (native), 7.0.0 and 6.3.0 (installed side-by-side;
  7.0/6.3 runtimes selected via LD_LIBRARY_PATH / LD_PRELOAD)
- 64 CPU cores, 251 GiB RAM

## What was run and the result

| Experiment | Processes/lanes | Work | ROCm runtime | Result |
|---|---|---|---|---|
| short-lived single-stream procs | 8 lanes x 400 | init, 1 stream, alloc+kernel+sync, teardown | 7.2.4 | 400/400 ok |
| short-lived single-stream procs | 16 lanes x 200 | same | 7.2.4 | 0 failures over 41 rounds |
| short-lived single-stream procs | 8 lanes x 300 | same | 7.0.0 | 300/300 ok |
| short-lived single-stream procs | 8 lanes x 312 | same | 6.3.0 | 312/312 ok |
| reset churn | 16 / 24 procs x 100 | hipDeviceReset each iteration | 7.2.4 | 0 failures |
| multi-stream multiplexing | 8 procs x 32 streams | async work on 32 streams, sync all | 7.2.4 | 0 failures, several rounds |
| multi-stream multiplexing | 8 procs x 32 streams x 6 rounds | same | 7.0.0 | 48/48 ok |
| alpaka3 unit_test_math_ComplexDouble as short-lived procs | 4 lanes x 20 rounds | real failing test binary | 7.2.4 | 540/540 ok |
| alpaka3 unit_test_math_ComplexDouble full suite | 4 concurrent suites x 6 rounds | real failing test binary | 6.3.0 runtime (LD_PRELOAD) | 24/24 suites ok |

The multi-stream mode deliberately exceeds `GPU_MAX_HW_QUEUES` (default 4): the 5th
and later streams were verified (with `AMD_LOG_LEVEL=3`) to map onto pooled hardware
queues, i.e. the shared-AQL-ring multiplexing path.

## Interpretation
- The failure did NOT reproduce on gfx1100 with host KFD 7.1.3, even with the CI's
  userspace ROCm 7.0/6.3 and heavy concurrency.
- This does not exonerate the HIP runtime; it identifies a strong hardware/KFD
  dependence (gfx1030 + the CI host's KFD/driver) that a different GPU does not show.
- A native HIP OOB store reproduces the exact fault string, confirming the CI message
  is a generic device page fault.

## What is still needed
- A gfx1030 host (the CI runner) to run the same reproducer under N-way concurrency.
- `dmesg`/amdgpu KFD output captured at the moment of failure (queue create / SDMA /
  page-fault events). The CI jobs capture no kernel log, which is the single most
  valuable missing artifact.
- `AMD_LOG_LEVEL=3` output from the failing process around `hipStreamCreateWithFlags`
  and the fault, to see whether the fault occurs during/after queue creation.

## Reproducer
See `hip_multiproc_repro.cpp` (standalone HIP, no alpaka3 dependency). Build with
`hipcc -O2 hip_multiproc_repro.cpp -o repro`. Modes: `short`, `streams <K> <iters>`,
`reset <reps>`.

## Standalone HIP reproducer (no alpaka3)
```cpp
// Standalone HIP reproducer for concurrent multi-process / multi-stream use of a
// single AMD GPU. It has no dependency on alpaka3.
//
// It stresses the paths implicated in intermittent CI failures such as
// ROCm/hip#3906 (GPU memory fault after stream creation with >=8 runners) and
// ROCm/legacy-rocm-build#6527 (deadlock / no-progress with many processes):
//   - per-process HIP context init (hipSetDevice)
//   - non-blocking stream creation (hipStreamCreateWithFlags)
//   - hipMalloc / hipMemcpyAsync / kernel launch / hipStreamSynchronize
//   - hipFree / hipStreamDestroy / hipDeviceReset
//   - a mode that keeps many streams alive per process, forcing the runtime to
//     multiplex them onto the shared hardware-queue pool
//     (GPU_MAX_HW_QUEUES, default 4).
//
// Build:
//   hipcc -O2 hip_multiproc_repro.cpp -o repro
//
// Run a single-stream short-lived process (mimics one short test binary):
//   ./repro short
//
// Run N processes concurrently, R rounds, W parallel lanes:
//   seq 0 $((N-1)) | xargs -P W -I{} ./repro short
//
// Keep K streams alive per process and churn work (forces HW-queue multiplexing):
//   ./repro streams <K> <iters>
//
// Exercise hipDeviceReset churn (what a per-process teardown does):
//   ./repro reset <reps>
//
// Capture runtime diagnostics (very verbose) while running:
//   AMD_LOG_LEVEL=3 ./repro streams 16 200
//
// Any "Memory access fault by GPU node-N ... Page not present or supervisor
// privilege" or a process that never prints DONE with a timeout is a hit.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

namespace
{
    char const* g_tag = "p";

    void fail(hipError_t e, char const* what)
    {
        std::fprintf(
            stderr,
            "[%d %s] HIP ERROR in %s: %s\n",
            static_cast<int>(getpid()),
            g_tag,
            what,
            hipGetErrorString(e));
        std::fflush(stderr);
        std::exit(2);
    }

#define CK(x, what)                 \
    do                              \
    {                               \
        hipError_t e_ = (x);        \
        if(e_ != hipSuccess)        \
            fail(e_, what);         \
    } while(0)

    __global__ void iotaKernel(int* out, int n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < n)
            out[i] = i * 2 + 1;
    }

    // One short-lived test process: init, one stream, small alloc + kernel +
    // sync, teardown, exit.
    int runShort()
    {
        constexpr int N = 256;
        int h[N];
        hipStream_t s;
        CK(hipSetDevice(0), "hipSetDevice");
        CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        int* d = nullptr;
        CK(hipMalloc(reinterpret_cast<void**>(&d), N * sizeof(int)), "hipMalloc");
        iotaKernel<<<1, N, 0, s>>>(d, N);
        CK(hipGetLastError(), "launch");
        CK(hipMemcpyAsync(h, d, N * sizeof(int), hipMemcpyDeviceToHost, s), "hipMemcpyAsync");
        CK(hipStreamSynchronize(s), "hipStreamSynchronize");
        CK(hipFree(d), "hipFree");
        CK(hipStreamDestroy(s), "hipStreamDestroy");
        return 0;
    }

    // Keep K streams alive and issue async work on all of them before syncing.
    // With K > GPU_MAX_HW_QUEUES the runtime multiplexes streams onto a shared
    // hardware queue / AQL ring.
    int runStreams(int K, int iters)
    {
        constexpr int N = 4096;
        std::vector<int> h(N);
        for(int i = 0; i < N; ++i)
            h[i] = i;
        CK(hipSetDevice(0), "hipSetDevice");
        std::vector<hipStream_t> s(K);
        std::vector<int*> dA(K), dB(K);
        for(int j = 0; j < K; ++j)
        {
            CK(hipStreamCreateWithFlags(&s[j], hipStreamNonBlocking), "hipStreamCreateWithFlags");
            CK(hipMalloc(reinterpret_cast<void**>(&dA[j]), N * sizeof(int)), "hipMalloc");
            CK(hipMalloc(reinterpret_cast<void**>(&dB[j]), N * sizeof(int)), "hipMalloc");
        }
        for(int r = 0; r < iters; ++r)
        {
            for(int j = 0; j < K; ++j)
            {
                CK(hipMemcpyAsync(dA[j], h.data(), N * sizeof(int), hipMemcpyHostToDevice, s[j]), "H2D");
                iotaKernel<<<(N + 255) / 256, 256, 0, s[j]>>>(dB[j], N);
                CK(hipGetLastError(), "launch");
                CK(hipMemcpyAsync(h.data(), dB[j], N * sizeof(int), hipMemcpyDeviceToHost, s[j]), "D2H");
            }
            for(int j = 0; j < K; ++j)
                CK(hipStreamSynchronize(s[j]), "hipStreamSynchronize");
            if((r % 20) == 0)
            {
                std::printf("[%d %s] rep %d ok\n", static_cast<int>(getpid()), g_tag, r);
                std::fflush(stdout);
            }
        }
        std::printf("[%d %s] DONE streams K=%d iters=%d\n", static_cast<int>(getpid()), g_tag, K, iters);
        std::fflush(stdout);
        return 0;
    }

    // hipDeviceReset churn: closest to a program that tears down its context
    // once per short-lived run.
    int runReset(int reps)
    {
        constexpr int N = 256;
        int h[N];
        for(int r = 0; r < reps; ++r)
        {
            hipStream_t s;
            CK(hipSetDevice(0), "hipSetDevice");
            CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "hipStreamCreateWithFlags");
            int* d = nullptr;
            CK(hipMalloc(reinterpret_cast<void**>(&d), N * sizeof(int)), "hipMalloc");
            iotaKernel<<<1, N, 0, s>>>(d, N);
            CK(hipMemcpyAsync(h, d, N * sizeof(int), hipMemcpyDeviceToHost, s), "hipMemcpyAsync");
            CK(hipStreamSynchronize(s), "hipStreamSynchronize");
            CK(hipFree(d), "hipFree");
            CK(hipStreamDestroy(s), "hipStreamDestroy");
            CK(hipDeviceReset(), "hipDeviceReset");
        }
        std::printf("[%d %s] DONE reset reps=%d\n", static_cast<int>(getpid()), g_tag, reps);
        std::fflush(stdout);
        return 0;
    }
} // namespace

int main(int argc, char** argv)
{
    static char tagbuf[32];
    std::snprintf(tagbuf, sizeof(tagbuf), "pid%d", static_cast<int>(getpid()));
    g_tag = tagbuf;

    std::string mode = (argc > 1) ? argv[1] : "short";

    if(mode == "short")
        return runShort();
    if(mode == "streams")
    {
        int K = (argc > 2) ? std::atoi(argv[2]) : 16;
        int iters = (argc > 3) ? std::atoi(argv[3]) : 200;
        return runStreams(K, iters);
    }
    if(mode == "reset")
    {
        int reps = (argc > 2) ? std::atoi(argv[2]) : 100;
        return runReset(reps);
    }
    std::fprintf(stderr, "usage: %s [short | streams <K> <iters> | reset <reps>]\n", argv[0]);
    return 1;
}
```

## Proposed ROCm patch — assignment-vs-comparison typo in queue-pool accounting

In `rocclr/device/rocm/rocdevice.cpp`, `acquireQueue()` contains:

```cpp
if (!managed && (cuMask.size() == 0) && (qIndex = QueuePriority::Normal)) {
    num_normal_queues_++;
}
```

`QueuePriority::Normal == 1`, so `(qIndex = ...)` **assigns** instead of comparing: the
condition is always true and `qIndex` is clobbered. `num_normal_queues_` is used by
`ReleaseActiveNormalQueue()` to decide queue reuse, so the accounting is wrong. Present in
7.0.0 (`rocdevice.cpp:3143`) and still in 7.2.0 (`rocdevice.cpp:3047`).

Patch files attached below (7.0.0 and 7.2.0). This is a correctness bug independent of the
intermittent page fault; it is **not** claimed to be the root cause, but it affects the same
dynamic hardware-queue pool implicated in the failure.

### patch (7.0.0)
```diff
--- a/rocdevice.cpp	2026-09-29 21:22:10.356522415 +0200
+++ b/rocdevice.cpp	2026-09-29 21:22:10.379522107 +0200
@@ -3140,7 +3140,7 @@
   qInfo.refCount = 1;
   ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "acquireQueue refCount: %p (%d)",
           result.first->first->base_address, result.first->second.refCount);
-  if (!managed && (cuMask.size() == 0) && (qIndex = QueuePriority::Normal)) {
+  if (!managed && (cuMask.size() == 0) && qIndex == QueuePriority::Normal) {
     num_normal_queues_++;
   }
   return queue;
```

### patch (7.2.0)
```diff
--- a/rocdevice.cpp	2026-09-29 21:22:13.192484443 +0200
+++ b/rocdevice.cpp	2026-09-29 21:22:13.224484015 +0200
@@ -3044,7 +3044,7 @@
   qInfo.refCount = 1;
   ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "acquireQueue refCount: %p (%d)",
           result.first->first->base_address, result.first->second.refCount);
-  if (!managed && (cuMask.size() == 0) && (qIndex = QueuePriority::Normal)) {
+  if (!managed && (cuMask.size() == 0) && qIndex == QueuePriority::Normal) {
     num_normal_queues_++;
   }
   return queue;
```

## Root-cause status (calibrated)

**Established:** intermittent, HIP-only, same-host/same-GPU concurrent failures; one GPU
page fault and one no-output hang in the same ~4 s window; the abort is amplified by an
alpaka3 destructor defect (now fixed); the fault string is generic.

**Hypothesis (not proven):** a ROCm runtime/KFD race when multiple processes create streams
and exceed `GPU_MAX_HW_QUEUES` (default 4), multiplexing onto shared hardware AQL rings.
Supporting evidence: `Device::acquireQueue()` holds `active_queue_access_` across the whole
`hsa_queue_create` loop; upstream fix rocm-systems#8856 ("Serialize shared HW-ring publish to
fix MT hang") and open rocm-systems#10377 ("Order shared AQL ring publication") describe
exactly this class; ROCm/hip#3906 reports the same fault string on the same gfx1030 hardware
with >=8 runners; ROCm/legacy-rocm-build#6527 has a maintainer attributing a multi-process
deadlock to KFD SDMA queue exhaustion.

**Not reproduced** on dev-hal gfx1100 (host KFD 7.1.3) using userspace ROCm 7.2.4, 7.0.0 and
6.3.0, even with 32 multiplexed streams per process and 24 concurrent reset-churn processes.
This indicates a hardware/KFD dependence (gfx1030 + the CI runner) and is the main open
question.

**Most valuable missing artifact:** `dmesg`/amdgpu KFD + HSA logs captured at the moment of
failure on the gfx1030 CI runner; the CI jobs currently capture no kernel log.
