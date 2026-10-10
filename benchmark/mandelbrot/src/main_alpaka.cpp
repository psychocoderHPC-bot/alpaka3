/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `mandelbrot` benchmark.
 *
 * The HeCBench references (`src/mandelbrot-cuda`, `src/mandelbrot-omp`) evaluate the escape-time
 * iteration of the Mandelbrot set over a fixed image. `MandelParameters::Point` is the exact same
 * scalar iteration in every reference; only the loop that walks the image differs (a CUDA kernel
 * with a 16x16 thread block or an OpenMP `target teams distribute parallel for collapse(2)`).
 *
 * This port follows the protocol: one portable kernel source, compiled unchanged for the host CPU
 * and for CUDA GPUs. There is no backend branch and no native CUDA/HIP/OpenMP kernel. The sampling
 * grid and the iteration body are shared with the host reference in `helper.hpp`, so the reference
 * and the kernel are arithmetically identical.
 */

#include <alpaka/alpaka.hpp>

#include "helper.hpp"
#include "misc.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace mandelbrotAlpaka
{
    /** Escape-time iteration for every pixel of the image.
     *
     * The image is walked as a 2D range (the reference walks the same 2D image with
     * `collapse(2)`); the per-pixel work is the shared `MandelParameters::Point`.
     * `MandelParameters` is passed by value into the kernel, exactly like the reference copies it to
     * the device.
     */
    struct MandelKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto output,
            mandel::MandelParameters params,
            int rows,
            int cols) const
        {
            using namespace alpaka;
            for(auto [j, i] :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec{cols, rows}}))
            {
                output[i * cols + j] = static_cast<float>(
                    params.point(mandel::ComplexF{params.scaleRow(i), params.scaleCol(j)}));
            }
        }
    };

    /** Build a 2D frame specification with one pixel per worker, 16 x 16 per frame.
     *
     * The dimension order follows the index range: dimension 0 is the column and dimension 1 is the
     * row.
     */
    auto makeFrameSpec(auto const&, auto const& exec, uint32_t rows, uint32_t cols)
    {
        using namespace alpaka;
        constexpr uint32_t chunk = 16u;
        uint32_t const blocksCols = divCeil(std::max(cols, 1u), chunk);
        uint32_t const blocksRows = divCeil(std::max(rows, 1u), chunk);
        return onHost::FrameSpec{Vec{std::max(blocksCols, 1u), std::max(blocksRows, 1u)}, Vec{chunk, chunk}, exec};
    }

    /** Run one warm-up iteration followed by `repeat` measured iterations.
     *
     * @return the median runtime in microseconds.
     */
    template<typename T_KernelBundle>
    double timeKernel(auto& queue, auto const& frameSpec, T_KernelBundle const& kernel, int repeat)
    {
        using namespace alpaka;
        queue.enqueue(frameSpec, kernel);
        onHost::wait(queue);

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeat));
        for(int i = 0; i < repeat; ++i)
        {
            auto const start = std::chrono::steady_clock::now();
            queue.enqueue(frameSpec, kernel);
            onHost::wait(queue);
            auto const end = std::chrono::steady_clock::now();
            double const us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-3;
            samples.push_back(us);
            std::printf("  iteration %d: %.3f us\n", i, us);
        }
        std::sort(samples.begin(), samples.end());
        double const median = samples.empty() ? 0.0 : samples[samples.size() / 2u];
        return median;
    }

    auto example(auto const deviceSpec, auto const exec, int rows, int cols, int maxIterations, int repeat) -> int
    {
        using namespace alpaka;

        long long const total = static_cast<long long>(rows) * static_cast<long long>(cols);
        std::cout << "Image size: " << rows << " x " << cols << " (" << total << " pixels)" << std::endl;
        std::cout << "Max iterations: " << maxIterations << std::endl;
        std::cout << "Measured iterations: " << repeat << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device device = devSelector.makeDevice(0);
        onHost::Queue queue = device.makeQueue(queueKind::blocking);

        Vec<int, 1u> const extent(static_cast<int>(total));
        auto hOut = onHost::allocHost<float>(extent);
        auto hReferenceFloat = onHost::allocHost<float>(extent);
        auto hReferenceDouble = onHost::allocHost<double>(extent);

        // Host references: the faithful float iteration and an independent double iteration.
        mandel::evaluateReferenceFloat(rows, cols, maxIterations, hReferenceFloat.data());
        mandel::evaluateReferenceDouble(rows, cols, maxIterations, hReferenceDouble.data());

        auto dOut = onHost::alloc<float>(device, extent);
        // Poison the device image with a NaN bit pattern so a pixel the kernel does not write is
        // caught by the finite check in verify() instead of silently matching the reference.
        onHost::memset(queue, dOut, 0xFFu);
        onHost::wait(queue);

        mandel::MandelParameters const params{rows, cols, maxIterations};
        auto const frameSpec = makeFrameSpec(device, exec, static_cast<uint32_t>(rows), static_cast<uint32_t>(cols));
        KernelBundle const kernel{MandelKernel{}, dOut, params, rows, cols};

        std::printf(
            "Mandelbrot kernel runtime median: %.3f us\n",
            timeKernel(queue, frameSpec, kernel, repeat));

        onHost::memcpy(queue, hOut, dOut);
        onHost::wait(queue);

        return mandel::verify(hOut, rows, cols, maxIterations, hReferenceFloat.data(), hReferenceDouble.data());
    }
} // namespace mandelbrotAlpaka

auto main(int argc, char* argv[]) -> int
{
    int rows = mandel::defaultRowSize;
    int cols = mandel::defaultColSize;
    int maxIterations = mandel::defaultMaxIterations;
    int repeat = mandel::defaultRepetitions;

    if(int const ret = mandel::parseCmd(argc, argv, rows, cols, maxIterations, repeat))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        {
            return mandelbrotAlpaka::example(
                onHost::makeDeviceSpec(backend),
                getExecutor(backend),
                rows,
                cols,
                maxIterations,
                repeat);
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
