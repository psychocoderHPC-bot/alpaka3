/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `matrix-rotate` benchmark.
 *
 * The workload rotates a square `n x n` matrix of `float` in place by 90 degrees, using the HeCBench "pyramid ring"
 * scheme: for every ring `layer` the four corners (`top`, `left`, `bottom`, `right`) are swapped with a single
 * saved temporary, walking `i` from `first = layer` to `last = n - 1 - layer`.
 *
 * The exact same kernel source is compiled for the host CPU and for CUDA GPUs; there is no backend branch and no
 * native CUDA/HIP/OpenMP kernel. The addressing scheme, the CLI (`<matrix size> <repeat>`), the input
 * initialization `m[i][j] = i * n + j` and the exact host comparison all match `src/matrix-rotate-cuda/main.cu`
 * (and the equivalent OpenMP reference `src/matrix-rotate-omp/main.cpp`) at HeCBench commit
 * 0a17ee47ca6072192101036d4bf292e6612f22e6.
 *
 * Kernel mapping mirrors the reference: one linear thread in the grid owns one ring
 * (`layer = blockIdx.x * blockDim.x + threadIdx.x`). Threads with `layer >= n / 2` are idle, exactly like the
 * reference guard. The independent host reference applies the identical ring swap in `double`, and the comparison
 * is bit-exact because the rotation only permutes exactly representable integer entries.
 */

#include "helper.hpp"
#include "misc.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

namespace matrixRotateAlpaka
{
    /** In-place 90-degree rotation of every ring of a square matrix.
     *
     * Each linear thread in the grid owns exactly one ring, matching the reference
     * `int layer = blockIdx.x * blockDim.x + threadIdx.x;`. The four-way corner swap follows the reference
     * byte for byte.
     */
    struct RotateKernel
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, alpaka::concepts::IMdSpan auto matrix, int n) const
        {
            using namespace alpaka;

            int const half = n / 2;
            for(auto layerIdx : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{half}))
            {
                int const layer = static_cast<int>(layerIdx);
                if(layer >= half)
                    continue;

                int const first = layer;
                int const last = n - 1 - layer;
                for(int i = first; i < last; ++i)
                {
                    int const offset = i - first;

                    float const top = matrix[static_cast<std::size_t>(first) * n + i]; // save top
                    // left -> top
                    matrix[static_cast<std::size_t>(first) * n + i]
                        = matrix[static_cast<std::size_t>(last - offset) * n + first];
                    // bottom -> left
                    matrix[static_cast<std::size_t>(last - offset) * n + first]
                        = matrix[static_cast<std::size_t>(last) * n + (last - offset)];
                    // right -> bottom
                    matrix[static_cast<std::size_t>(last) * n + (last - offset)]
                        = matrix[static_cast<std::size_t>(i) * n + last];
                    // top -> right
                    matrix[static_cast<std::size_t>(i) * n + last] = top;
                }
            }
        }
    };

    /** Build the 1-D launcher specification for the ring kernel.
     *
     * Each ring is an independent parallel task. The launch shape is derived from the device properties so that
     * both backends expose the algorithm's parallelism efficiently:
     *
     * - Devices with a unit warp (`warpSize == 1`, the host CPU executors) use one thread per frame with
     *   `2 * multiProcessorCount` frames. The host executor flattens each frame to a single thread and pays a fixed
     *   per-frame accelerator setup cost, so `2 * cores` frames is enough to use all cores while keeping the
     *   per-frame overhead low; each host thread strides over the rings. One frame per ring is measurably slower.
     * - Accelerators use the reference's 256 threads per block: `ceil(rings / 256)` frames of 256, matching the
     *   reference grid `(n/2 + 255) / 256` blocks of 256 threads.
     *
     * This is launch-shape tuning only. The kernel and its ring addressing are identical on both backends, and the
     * index map strides over every ring regardless of the shape, so correctness never depends on it.
     */
    inline auto makeFrameSpec(auto const& device, auto const exec, int n)
    {
        using namespace alpaka;
        auto const props = device.getDeviceProperties();
        int const numRings = n / 2;
        uint32_t const rings = static_cast<uint32_t>(std::max(numRings, 1));

        if(props.warpSize <= 1u)
        {
            uint32_t const cores = std::max(props.multiProcessorCount, 1u);
            uint32_t const numFrames = std::min<uint32_t>(rings, cores * 2u);
            return onHost::FrameSpec{Vec<uint32_t, 1u>(numFrames), Vec<uint32_t, 1u>(1u), exec};
        }

        constexpr uint32_t threadsPerBlock = 256u;
        uint32_t const numFrames = (rings + threadsPerBlock - 1u) / threadsPerBlock;
        return onHost::FrameSpec{Vec<uint32_t, 1u>(numFrames), Vec<uint32_t, 1u>(threadsPerBlock), exec};
    }

    /** Rotate the device matrix `repeat` times inside a single timed section.
     *
     * The timing scope matches the HeCBench reference, which brackets the loop of `repeat` launches with one device
     * synchronization. A full batch of `repeat` rotations is run once as warm-up.
     *
     * @return the median per-rotation runtime in microseconds.
     */
    double timeRotations(auto& queue, auto const& frameSpec, auto const& kernel, int repeat, int measureReps)
    {
        using namespace alpaka;

        // warm-up: one full batch
        for(int i = 0; i < repeat; ++i)
            queue.enqueue(frameSpec, kernel);
        onHost::wait(queue);

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(measureReps));
        for(int r = 0; r < measureReps; ++r)
        {
            auto const start = std::chrono::steady_clock::now();
            for(int i = 0; i < repeat; ++i)
                queue.enqueue(frameSpec, kernel);
            onHost::wait(queue);
            auto const end = std::chrono::steady_clock::now();
            double const us
                = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-3 / repeat;
            samples.push_back(us);
            printf("  measured rep %d: %.4f us/rotation\n", r, us);
        }
        std::sort(samples.begin(), samples.end());
        return samples[samples.size() / 2u];
    }

    int run(auto const deviceSpec, auto const exec, int n, int repeat)
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device device = devSelector.makeDevice(0);
        // A blocking queue reduces the launch overhead for host devices.
        onHost::Queue queue = device.makeQueue(queueKind::blocking);

        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        std::size_t const count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
        printf("matrix %d x %d (%zu floats, %zu MiB), repeat %d\n", n, n, count, (count * sizeof(float)) >> 20, repeat);

        std::vector<float> initial;
        matrixRotate::initMatrix(initial, n);
        std::vector<double> gold(initial.begin(), initial.end());
        matrixRotate::referenceRotate(gold, n, repeat);

        Vec<std::size_t, 1u> const extent{count};
        auto hInitial = onHost::allocHost<float>(extent);
        auto hResult = onHost::allocHost<float>(extent);
        for(std::size_t i = 0; i < count; ++i)
            hInitial[i] = initial[i];

        auto dMatrix = onHost::alloc<float>(device, extent);

        int const half = n / 2;
        auto const frameSpec = makeFrameSpec(device, exec, n);
        KernelBundle const kernel{RotateKernel{}, dMatrix, n};

        // ---- validation run (untimed, device path) ----
        onHost::memcpy(queue, dMatrix, hInitial);
        onHost::wait(queue);
        if(half > 0)
        {
            for(int i = 0; i < repeat; ++i)
                queue.enqueue(frameSpec, kernel);
        }
        onHost::wait(queue);

        onHost::memcpy(queue, hResult, dMatrix);
        onHost::wait(queue);

        int const validRc = matrixRotate::verify(std::data(hResult), gold, n);

        // ---- timing run ----
        if(half > 0)
        {
            double const median = timeRotations(queue, frameSpec, kernel, repeat, 7);
            printf("median kernel time: %.4f us/rotation\n", median);
        }
        else
        {
            printf("median kernel time: n/a (n < 2, no ring to rotate)\n");
        }

        return validRc;
    }

    auto example(auto const deviceSpec, auto const exec, int n, int repeat) -> int
    {
        using namespace alpaka;
        return run(deviceSpec, exec, n, repeat);
    }
} // namespace matrixRotateAlpaka

auto main(int argc, char* argv[]) -> int
{
    int size = matrixRotate::defaultSize;
    int repeat = matrixRotate::defaultRepeat;

    if(int const ret = matrixRotate::parseCmd(argc, argv, size, repeat))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return matrixRotateAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), size, repeat); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
