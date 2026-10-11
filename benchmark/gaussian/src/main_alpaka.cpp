/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `gaussian` benchmark.
 *
 * The pinned HeCBench reference (`src/gaussian-cuda/gaussianElim.cu` and its OpenMP twin
 * `src/gaussian-omp/gaussianElim.cpp`) solves a dense system `A x = b` for an internally generated
 * Toeplitz matrix (`init_matrix`) and `b = 1`. It performs Gaussian elimination without pivoting:
 * the forward elimination runs on the accelerator (CUDA `fan1`/`fan2` kernels, or the OpenMP target
 * loops), back substitution runs on the host, and the result is compared against the same no-pivot
 * elimination executed on the CPU with an absolute tolerance of `1e-3`.
 *
 * This port keeps the reference algorithm exactly:
 *  - the same input generator (`gaussianBench::initMatrix`) and `b = 1`,
 *  - the same no-pivot forward elimination, expressed as few portable kernels,
 *  - the same host back substitution (`gaussianBench::backSubstitutionFloat`),
 *  - one source that runs on the host CPU and on CUDA GPUs, with no backend-specific algorithm.
 *
 * It replaces the reference's self-comparison with an independent double-precision LU solve with
 * partial pivoting (`gaussianBench::gaussianGoldDouble`) plus a residual check, so a systematically
 * wrong elimination cannot pass.
 *
 * The forward elimination order is preserved bit-for-bit with respect to the reference upper triangle
 * and the resulting solution: at step `t` each row `row > t` subtracts
 * `(a[row][t] / a[t][t]) * a[t][col]` from `a[row][col]` for `col > t`, and `b[row]` subtracts
 * `(a[row][t] / a[t][t]) * b[t]`. The reference also writes the below-diagonal column `t` to zero;
 * that value is never read by back substitution, so omitting the write cannot change the solution and
 * it removes a read/write race between the multiplier and the column-`t` element. The multiplier is
 * recomputed per element from the same operands the reference's `fan1` used, which yields identical
 * products.
 *
 * The reference sources are kept outside this repository and are not vendored.
 */

#include <alpaka/alpaka.hpp>

#include "helper.hpp"
#include "misc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gaussianAlpaka
{
    using namespace gaussianBench;

    //! Logical threads per frame.
    constexpr std::uint32_t threadsPerBlock = 256u;

    /** First phase of one elimination step: compute the row multipliers (`fan1`).
     *
     * For every row `row > t` the multiplier `m[row*size + t] = a[row*size + t] / a[t*size + t]` is
     * written once. The multiplier buffer is written and read by distinct logical threads; there is no
     * read/write race because the second phase starts only after this one completes.
     */
    struct MultiplierKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto a,
            alpaka::concepts::IMdSpan auto m,
            std::uint32_t size,
            std::uint32_t t,
            std::uint32_t numRows) const
        {
            using namespace alpaka;

            for(auto iMD : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{numRows}))
            {
                std::uint32_t const row = t + 1u + iMD.x();
                m[static_cast<std::size_t>(row) * size + t]
                    = a[static_cast<std::size_t>(row) * size + t] / a[static_cast<std::size_t>(t) * size + t];
            }
        }
    };

    /** Second phase of one elimination step: update rows `row > t` (`fan2`).
     *
     * The frame covers the `rows * cols` entries of the tile `[t+1, size) x [t, size)`, enumerated in
     * row-major order. Every entry is touched by exactly one logical thread:
     *  - at column `col == t` the right-hand side is updated (`b[row] -= mult * b[t]`),
     *  - elsewhere the matrix entry is updated (`a[row][col] -= mult * a[t][col]`).
     *
     * Column `t` of the matrix is intentionally left untouched: the reference zeroes it and back
     * substitution never reads it, so omitting the write cannot change the solution and it removes a
     * read/write race on that element.
     */
    struct UpdateKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto a,
            alpaka::concepts::IMdSpan auto b,
            alpaka::concepts::IMdSpan auto m,
            std::uint32_t size,
            std::uint32_t t,
            std::uint32_t numRows,
            std::uint32_t numCols) const
        {
            using namespace alpaka;

            for(auto linearIdxMD :
                onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{Vec{numRows * numCols}}))
            {
                std::uint32_t const linearIdx = linearIdxMD.x();
                std::uint32_t const row = t + 1u + linearIdx / numCols;
                std::uint32_t const col = t + linearIdx % numCols;
                float const mult = m[static_cast<std::size_t>(row) * size + t];
                if(col == t)
                    b[row] -= mult * b[t];
                else
                    a[static_cast<std::size_t>(row) * size + col]
                        -= mult * a[static_cast<std::size_t>(t) * size + col];
            }
        }
    };

    struct Stats
    {
        double median = 0.0;
        double min = 0.0;
        double max = 0.0;
        double spread = 0.0;
    };

    inline Stats summarize(std::vector<double> samples)
    {
        Stats s;
        std::sort(samples.begin(), samples.end());
        s.min = samples.front();
        s.max = samples.back();
        std::size_t const n = samples.size();
        s.median = (n % 2u == 0u) ? 0.5 * (samples[n / 2u - 1u] + samples[n / 2u]) : samples[n / 2u];
        s.spread = (s.median > 0.0) ? (s.max - s.min) / s.median : 0.0;
        return s;
    }

    /** Launch the whole forward elimination once (the `size - 1` steps).
     *
     * The multiplier buffer is reused across all steps; only column `t` is written each step, so a
     * single zero-initialised buffer is enough. The reference `fan1`/`fan2` pairing is preserved.
     */
    inline void forwardElimination(
        auto& queue,
        auto&& a,
        auto&& b,
        auto&& m,
        std::uint32_t size,
        auto const& exec)
    {
        using namespace alpaka;

        for(std::uint32_t t = 0; t + 1u < size; ++t)
        {
            std::uint32_t const rows = size - 1u - t;
            std::uint32_t const cols = size - t;
            std::uint32_t const numFrames = divCeil(rows, threadsPerBlock);
            queue.enqueue(
                onHost::FrameSpec{Vec{numFrames}, Vec{threadsPerBlock}, exec},
                KernelBundle{MultiplierKernel{}, a, m, size, t, rows});

            std::uint32_t const work = rows * cols;
            std::uint32_t const updateFrames = divCeil(work, threadsPerBlock);
            queue.enqueue(
                onHost::FrameSpec{Vec{updateFrames}, Vec{threadsPerBlock}, exec},
                KernelBundle{UpdateKernel{}, a, b, m, size, t, rows, cols});
        }
    }

    /** Run and validate one matrix size.
     *
     * The device runs the no-pivot forward elimination, the host completes the no-pivot back
     * substitution, and the solution is compared against the independent double-precision oracle. A
     * fresh copy of the original matrix is uploaded before each measured sample and before the
     * correctness run, because forward elimination destroys its input.
     */
    int runSize(
        auto const& deviceSpec,
        auto const& exec,
        char const* backend,
        std::uint32_t size,
        gaussianBench::Params const& params)
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);

        std::size_t const n = size;
        std::size_t const numElements = n * n;

        // Original system on the host (single precision, exactly the reference generator).
        auto hA = onHost::allocHost<float>(numElements);
        auto hB = onHost::allocHost<float>(n);
        gaussianBench::initMatrix(onHost::data(hA), onHost::data(hB), size);

        // Independent double-precision oracle.
        std::vector<double> gold(n);
        gaussianBench::gaussianGoldDouble(onHost::data(hA), onHost::data(hB), gold.data(), size);

        // Device copies of the original system.
        auto dA = onHost::alloc<float>(device, numElements);
        auto dB = onHost::alloc<float>(device, n);
        // Scratch multiplier matrix, only column `t` is ever written.
        auto dM = onHost::alloc<float>(device, numElements);
        onHost::memset(queue, dM, std::uint8_t{0});

        // Host buffers for the device result after forward elimination.
        auto hAw = onHost::allocHost<float>(numElements);
        auto hBw = onHost::allocHost<float>(n);
        std::vector<float> solution(n);

        auto const uploadAndRun = [&]()
        {
            onHost::memcpy(queue, dA, hA);
            onHost::memcpy(queue, dB, hB);
            onHost::wait(queue);
            gaussianAlpaka::forwardElimination(queue, dA, dB, dM, size, exec);
            onHost::wait(queue);
        };

        // Warm-up.
        uploadAndRun();

        if(!params.verifyOnly)
        {
            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                // Each measured elimination starts from the original system. Only the elimination
                // itself is timed; the reset is outside the timer, matching the reference `-t` scope
                // (its timer starts after the H2D copies and the synchronisation).
                double total = 0.0;
                for(int r = 0; r < params.repeat; ++r)
                {
                    onHost::memcpy(queue, dA, hA);
                    onHost::memcpy(queue, dB, hB);
                    onHost::wait(queue);
                    auto const start = std::chrono::steady_clock::now();
                    gaussianAlpaka::forwardElimination(queue, dA, dB, dM, size, exec);
                    onHost::wait(queue);
                    auto const end = std::chrono::steady_clock::now();
                    total += static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                }
                samples.push_back(total * 1e-6 / static_cast<double>(params.repeat));
            }

            Stats const stats = gaussianAlpaka::summarize(samples);
            std::printf(
                "RESULT backend=%s size=%u median_ms=%.6f min_ms=%.6f max_ms=%.6f relspread=%.4f\n",
                backend,
                size,
                stats.median,
                stats.min,
                stats.max,
                stats.spread);
            std::printf("  raw_samples_ms=");
            for(std::size_t i = 0; i < samples.size(); ++i)
                std::printf("%s%.6f", i == 0 ? "" : ",", samples[i]);
            std::printf("\n");
        }

        // Correctness: fresh forward elimination, host back substitution, oracle comparison.
        uploadAndRun();
        onHost::memcpy(queue, hAw, dA);
        onHost::memcpy(queue, hBw, dB);
        onHost::wait(queue);

        gaussianBench::backSubstitutionFloat(onHost::data(hAw), onHost::data(hBw), solution.data(), size);

        if(params.verbose)
        {
            for(std::uint32_t i = 0; i < size; ++i)
                std::printf("  x[%u] = %.8g (gold %.8g)\n", i, static_cast<double>(solution[i]), gold[i]);
        }

        int const rc = gaussianBench::verify(
            solution.data(),
            gold.data(),
            onHost::data(hA),
            onHost::data(hB),
            size);
        std::printf("gaussian size=%u verification %s\n", size, rc == EXIT_SUCCESS ? "PASSED" : "FAILED");
        return rc;
    }

    auto example(auto const& deviceSpec, auto const& exec, gaussianBench::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        std::string const backend = deviceSpec.getApi().getName();

        std::printf(
            "Using alpaka backend %s on %s, size=%u, repeat=%d, samples=%d\n",
            backend.c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            params.size,
            params.repeat,
            params.samples);

        return gaussianAlpaka::runSize(deviceSpec, exec, backend.c_str(), params.size, params);
    }
} // namespace gaussianAlpaka

auto main(int argc, char* argv[]) -> int
{
    gaussianBench::Params params;
    if(int const ret = gaussianBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return gaussianAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
