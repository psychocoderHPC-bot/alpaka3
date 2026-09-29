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
