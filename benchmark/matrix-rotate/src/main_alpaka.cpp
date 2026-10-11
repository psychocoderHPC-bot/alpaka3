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
 * Work division: the HeCBench reference launches one linear thread per ring, which is latency bound because the
 * number of rings grows only linearly with `n` while each thread performs an O(n) serial walk. This implementation
 * instead flattens the (ring, element) iteration space into a single linear index so that one thread owns exactly
 * one four-corner swap. The swaps form a bijection onto the `(layer, i)` space with `layer` in `[0, n/2)` and
 * `i` in `[layer, n - 1 - layer)`, so the total number of independent swaps is
 * `P = sum_{layer=0}^{n/2 - 1} (n - 1 - 2 * layer) = (n/2) * (n - n/2)`.
 *
 * A thread recovers `(layer, i)` from its linear id `workId` in O(1) with a closed-form inverse of the layer
 * prefix sum `start(layer) = layer * (n - layer)`: solving `layer^2 - n * layer + workId = 0` gives the estimate
 * `layer ~ (n - sqrt(n^2 - 4 * workId)) / 2`, then two small correction loops snap to the exact layer. The
 * correction loops make the decode exact regardless of floating-point rounding, so the permutation is unchanged.
 * The linear id wraps around `P`, so a grid with `ceil(P / 256)` frames of 256 threads never executes a thread with
 * `workId >= P` and the launch is not padded.
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
    /** Number of independent four-corner swaps for an `n x n` matrix.
     *
     * `P = sum_{layer=0}^{n/2 - 1} (n - 1 - 2 * layer) = (n/2) * (n - n/2)`, computed in `std::size_t` so the
     * intermediate products cannot overflow for any `n` whose swap count still fits the 32-bit index map.
     */
    inline std::size_t swapCount(int n)
    {
        std::size_t const half = static_cast<std::size_t>(n) / 2u;
        return half * (static_cast<std::size_t>(n) - half);
    }

    /** In-place 90-degree rotation of every ring of a square matrix.
     *
     * Each linear thread owns exactly one four-corner swap, decoded from its linear id. The swap itself is the
     * same byte-for-byte sequence the reference performs inside its per-ring loop.
     */
    struct RotateKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto matrix,
            int n,
            int totalSwaps) const
        {
            using namespace alpaka;

            int const half = n / 2;
            int const nInt = n;
            for(auto workIdx : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{totalSwaps}))
            {
                int const workId = static_cast<int>(workIdx);

                // Closed-form estimate of the ring `layer` whose prefix sum `layer * (n - layer)` contains
                // `workId`, followed by two exact correction loops.
                float const nf = static_cast<float>(nInt);
                float disc = nf * nf - 4.0f * static_cast<float>(workId);
                disc = disc > 0.0f ? disc : 0.0f;
                int layer = static_cast<int>(0.5f * (nf - math::sqrt(disc)));
                layer = layer < 0 ? 0 : (layer >= half ? half - 1 : layer);
                while(layer > 0 && static_cast<std::size_t>(layer) * (nInt - layer) > workId)
                    --layer;
                while(layer + 1 < half && static_cast<std::size_t>(layer + 1) * (nInt - (layer + 1)) <= workId)
                    ++layer;

                std::size_t const start = static_cast<std::size_t>(layer) * (nInt - layer);
                int const offsetInRing = static_cast<int>(static_cast<std::size_t>(workId) - start);
                int const first = layer;
                int const last = nInt - 1 - layer;
                int const i = first + offsetInRing;
                int const offset = offsetInRing;

                float const top = matrix[static_cast<std::size_t>(first) * nInt + i]; // save top
                // left -> top
                matrix[static_cast<std::size_t>(first) * nInt + i]
                    = matrix[static_cast<std::size_t>(last - offset) * nInt + first];
                // bottom -> left
                matrix[static_cast<std::size_t>(last - offset) * nInt + first]
                    = matrix[static_cast<std::size_t>(last) * nInt + (last - offset)];
                // right -> bottom
                matrix[static_cast<std::size_t>(last) * nInt + (last - offset)]
                    = matrix[static_cast<std::size_t>(i) * nInt + last];
                // top -> right
                matrix[static_cast<std::size_t>(i) * nInt + last] = top;
            }
        }
    };

    /** Build the 1-D launcher specification for the swap kernel.
     *
     * The flattened swap space of size `totalSwaps` is split into frames so that both backends expose the
     * algorithm's parallelism efficiently:
     *
     * - Devices with a unit warp (`warpSize == 1`, the host CPU executors) use one thread per frame with
     *   `2 * multiProcessorCount` frames; the host executor pays a fixed per-frame setup cost, so this keeps the
     *   per-frame overhead low while each host thread strides over the swaps.
     * - Accelerators use the reference's 256 threads per block: `ceil(totalSwaps / 256)` frames of 256.
     *
     * This is launch-shape tuning only. The kernel and its swap addressing are identical on both backends, and the
     * index map strides over every swap regardless of the shape, so correctness never depends on it.
     */
    inline auto makeFrameSpec(auto const& device, auto const exec, uint32_t totalSwaps)
    {
        using namespace alpaka;
        auto const props = device.getDeviceProperties();
        uint32_t const swaps = std::max(totalSwaps, 1u);

        if(props.warpSize <= 1u)
        {
            uint32_t const cores = std::max(props.multiProcessorCount, 1u);
            uint32_t const numFrames = std::min<uint32_t>(swaps, cores * 2u);
            return onHost::FrameSpec{Vec<uint32_t, 1u>(numFrames), Vec<uint32_t, 1u>(1u), exec};
        }

        constexpr uint32_t threadsPerBlock = 256u;
        uint32_t const numFrames = (swaps + threadsPerBlock - 1u) / threadsPerBlock;
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
        std::size_t const numSwaps = swapCount(n);
        auto const frameSpec = makeFrameSpec(device, exec, static_cast<uint32_t>(numSwaps));
        KernelBundle const kernel{RotateKernel{}, dMatrix, n, static_cast<int>(numSwaps)};

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
