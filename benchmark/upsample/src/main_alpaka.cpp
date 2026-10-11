/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `upsample` benchmark.
 *
 * The pinned HeCBench reference (`src/upsample-cuda/main.cu`, OpenMP twin `src/upsample-omp/main.cpp`)
 * implements the 2x nearest-neighbour upsampling layer of a small U-Net. The reference defines two
 * passes with two kernel formulations each (a flat one-thread-per-element kernel and a 2D tile kernel),
 * but all four kernels perform the same two mathematical operations on an `N, C, H, W` tensor:
 *
 *  - forward: for every input element `(b, c, i, j)` its value is replicated into the 2x2 output tile
 *    at rows `2i, 2i+1` and columns `2j, 2j+1` of the `2H x 2W` output image;
 *  - backward: for every element `(b, c, i, j)` the gradient is the sum of the four `dout` values of
 *    that 2x2 tile.
 *
 * This port keeps the reference algorithms and layouts exactly. One portable kernel per pass is
 * enqueued per backend; there is no native CUDA/HIP/OpenMP kernel and no backend branch. The result is
 * compared against an independent double-precision host reference with a per-tile relative-L2 metric
 * (see `helper.hpp` and the README).
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

namespace upsampleAlpaka
{
    using namespace upsampleBench;

    //! Logical threads per frame. The executor decides the physical mapping; the kernel is agnostic.
    constexpr std::uint32_t threadsPerBlock = 256u;

    //! Decode the flat tile index `p` of an `N, C, H, W` tensor into `(b*C + c, i, j)`.
    struct Decomposed
    {
        std::uint64_t bc;
        std::uint64_t i;
        std::uint64_t j;
    };

    ALPAKA_FN_ACC inline Decomposed decompose(std::uint64_t p, std::uint64_t H, std::uint64_t W)
    {
        std::uint64_t const img = H * W;
        Decomposed d;
        // Recovering the remainder with a subtraction keeps this to two integer divisions instead of
        // four (`%` is a second division for the same divisor); this is the CPU hot path.
        d.bc = p / img;
        std::uint64_t const rem = p - d.bc * img;
        d.i = rem / W;
        d.j = rem - d.i * W;
        return d;
    }

    /** Forward pass: replicate every input element into its 2x2 output tile.
     *
     * One work item per input element, exactly like the reference's flat kernel, so the memory traffic
     * is coalesced on the GPU. The pointer arithmetic reproduces `upsample_forward_kernel`: the
     * channel-plane offset is `bc * imgOut`, and the four tile entries sit at rows `2i, 2i+1` and
     * columns `2j, 2j+1` with stride `wOut` between the two rows.
     */
    struct ForwardKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto x,
            alpaka::concepts::IMdSpan auto out,
            std::uint64_t H,
            std::uint64_t W) const
        {
            using namespace alpaka;

            std::uint64_t const numTiles = static_cast<std::uint64_t>(x.getExtents().x());
            std::uint64_t const wOut = 2u * W;
            std::uint64_t const imgOut = (2u * H) * wOut;

            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{Vec{numTiles}}))
            {
                std::uint64_t const p = idx.x();
                Decomposed const d = decompose(p, H, W);
                std::uint64_t const base = d.bc * imgOut + (2u * d.i) * wOut + 2u * d.j;

                float const value = x[static_cast<std::size_t>(p)];
                out[static_cast<std::size_t>(base)] = value;
                out[static_cast<std::size_t>(base + 1u)] = value;
                out[static_cast<std::size_t>(base + wOut)] = value;
                out[static_cast<std::size_t>(base + wOut + 1u)] = value;
            }
        }
    };

    /** Backward pass: sum the four `dout` values of each 2x2 tile.
     *
     * One work item per gradient element, same flat map as the forward pass. The accumulation order per
     * element is `(((v00 + v01) + v10) + v11)`, exactly the reference's nested loop with
     * `dout_sum += ...`, so a correct port reproduces the reference float arithmetic.
     */
    struct BackwardKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto dout,
            alpaka::concepts::IMdSpan auto dx,
            std::uint64_t H,
            std::uint64_t W) const
        {
            using namespace alpaka;

            std::uint64_t const numTiles = static_cast<std::uint64_t>(dx.getExtents().x());
            std::uint64_t const wOut = 2u * W;
            std::uint64_t const imgOut = (2u * H) * wOut;

            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{Vec{numTiles}}))
            {
                std::uint64_t const p = idx.x();
                Decomposed const d = decompose(p, H, W);
                std::uint64_t const base = d.bc * imgOut + (2u * d.i) * wOut + 2u * d.j;

                float doutSum = 0.0f;
                doutSum += dout[static_cast<std::size_t>(base)];
                doutSum += dout[static_cast<std::size_t>(base + 1u)];
                doutSum += dout[static_cast<std::size_t>(base + wOut)];
                doutSum += dout[static_cast<std::size_t>(base + wOut + 1u)];
                dx[static_cast<std::size_t>(p)] = doutSum;
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

    //! Enqueue the forward pass once over the whole tensor.
    inline void runForward(
        auto& queue,
        auto const& exec,
        auto& dX,
        auto& dOut,
        std::uint64_t numTiles,
        std::uint64_t H,
        std::uint64_t W)
    {
        using namespace alpaka;

        std::uint32_t const frames = static_cast<std::uint32_t>((numTiles + threadsPerBlock - 1u) / threadsPerBlock);
        auto const frameSpec = onHost::FrameSpec{Vec{frames}, Vec{threadsPerBlock}, exec};
        queue.enqueue(frameSpec, KernelBundle{ForwardKernel{}, dX, dOut, H, W});
    }

    //! Enqueue the backward pass once over the whole tensor.
    inline void runBackward(
        auto& queue,
        auto const& exec,
        auto& dDout,
        auto& dDx,
        std::uint64_t numTiles,
        std::uint64_t H,
        std::uint64_t W)
    {
        using namespace alpaka;

        std::uint32_t const frames = static_cast<std::uint32_t>((numTiles + threadsPerBlock - 1u) / threadsPerBlock);
        auto const frameSpec = onHost::FrameSpec{Vec{frames}, Vec{threadsPerBlock}, exec};
        queue.enqueue(frameSpec, KernelBundle{BackwardKernel{}, dDout, dDx, H, W});
    }

    /** Run, validate and measure both passes for one device and executor. */
    auto example(auto const& deviceSpec, auto const& exec, upsampleBench::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);
        std::string const backend = deviceSpec.getApi().getName();
        std::string const execName = onHost::demangledName(exec);

        std::uint64_t const B = static_cast<std::uint64_t>(params.batch);
        std::uint64_t const C = static_cast<std::uint64_t>(params.channels);
        std::uint64_t const H = static_cast<std::uint64_t>(params.height);
        std::uint64_t const W = static_cast<std::uint64_t>(params.width);
        std::uint64_t const img = H * W;
        std::uint64_t const numTiles = B * C * img;
        std::uint64_t const numOut = numTiles * 4u;

        std::printf(
            "Using alpaka backend %s on %s exec=%s, B=%llu C=%llu H=%llu W=%llu repeat=%d samples=%d\n",
            backend.c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            execName.c_str(),
            static_cast<unsigned long long>(B),
            static_cast<unsigned long long>(C),
            static_cast<unsigned long long>(H),
            static_cast<unsigned long long>(W),
            params.repeat,
            params.samples);

        // Deterministic input for the forward pass and for the backward pass.
        auto hX = onHost::allocHost<float>(numTiles);
        auto hDout = onHost::allocHost<float>(numOut);
        upsampleBench::makeRandomFloat(onHost::data(hX), numTiles, params.seed);
        upsampleBench::makeRandomFloat(onHost::data(hDout), numOut, params.seed + 1u);

        // Independent single-precision CPU references (the comparison target) and independent
        // double-precision oracles that pin those references to within fp32 rounding.
        std::vector<float> refOut;
        std::vector<float> refDx;
        std::vector<double> oracleOut;
        std::vector<double> oracleDx;
        auto const t0 = std::chrono::steady_clock::now();
        upsampleBench::forwardReference(onHost::data(hX), refOut, B, C, H, W);
        upsampleBench::backwardReference(onHost::data(hDout), refDx, B, C, H, W);
        upsampleBench::forwardOracleDouble(onHost::data(hX), oracleOut, B, C, H, W);
        upsampleBench::backwardOracleDouble(onHost::data(hDout), oracleDx, B, C, H, W);
        auto const t1 = std::chrono::steady_clock::now();
        double const hostRefMs
            = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) * 1e-6;
        std::printf("host fp32 references and fp64 oracles computed in %.3f ms\n", hostRefMs);

        int agreeRc = upsampleBench::verifyReferenceAgreement(
            refOut.data(),
            oracleOut,
            refOut.size(),
            upsampleBench::referenceAgreementTolerance,
            "forward-reference-vs-oracle");
        if(agreeRc != EXIT_SUCCESS)
            return agreeRc;
        agreeRc = upsampleBench::verifyReferenceAgreement(
            refDx.data(),
            oracleDx,
            refDx.size(),
            upsampleBench::referenceAgreementTolerance,
            "backward-reference-vs-oracle");
        if(agreeRc != EXIT_SUCCESS)
            return agreeRc;

        auto dX = onHost::alloc<float>(device, numTiles);
        auto dOut = onHost::alloc<float>(device, numOut);
        auto dDout = onHost::alloc<float>(device, numOut);
        auto dDx = onHost::alloc<float>(device, numTiles);
        auto hOut = onHost::allocHost<float>(numOut);
        auto hDx = onHost::allocHost<float>(numTiles);

        onHost::memcpy(queue, dX, hX);
        onHost::memcpy(queue, dDout, hDout);
        onHost::wait(queue);

        if(!params.verifyOnly)
        {
            // warm-up
            runForward(queue, exec, dX, dOut, numTiles, H, W);
            runBackward(queue, exec, dDout, dDx, numTiles, H, W);
            onHost::wait(queue);

            std::vector<double> fwdSamples;
            std::vector<double> bwdSamples;
            fwdSamples.reserve(static_cast<std::size_t>(params.samples));
            bwdSamples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                for(int r = 0; r < params.repeat; ++r)
                    runForward(queue, exec, dX, dOut, numTiles, H, W);
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                fwdSamples.push_back(
                    static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())
                    * 1e-3 / static_cast<double>(params.repeat));

                onHost::wait(queue);
                auto const startB = std::chrono::steady_clock::now();
                for(int r = 0; r < params.repeat; ++r)
                    runBackward(queue, exec, dDout, dDx, numTiles, H, W);
                onHost::wait(queue);
                auto const endB = std::chrono::steady_clock::now();
                bwdSamples.push_back(
                    static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(endB - startB).count())
                    * 1e-3 / static_cast<double>(params.repeat));
            }

            char const* const modeNames[2] = {"forward", "backward"};
            std::vector<double> const* const sampleSets[2] = {&fwdSamples, &bwdSamples};
            for(int m = 0; m < 2; ++m)
            {
                Stats const stats = summarize(*sampleSets[m]);
                std::printf(
                    "RESULT backend=%s exec=%s mode=%s B=%llu C=%llu H=%llu W=%llu median_us=%.4f min_us=%.4f "
                    "max_us=%.4f relspread=%.4f\n",
                    backend.c_str(),
                    execName.c_str(),
                    modeNames[m],
                    static_cast<unsigned long long>(B),
                    static_cast<unsigned long long>(C),
                    static_cast<unsigned long long>(H),
                    static_cast<unsigned long long>(W),
                    stats.median,
                    stats.min,
                    stats.max,
                    stats.spread);
                std::printf("  raw_samples_us=");
                for(std::size_t i = 0; i < sampleSets[m]->size(); ++i)
                    std::printf("%s%.4f", i == 0 ? "" : ",", (*sampleSets[m])[i]);
                std::printf("\n");
            }
        }

        // Validation: run both passes once and compare against the double reference.
        runForward(queue, exec, dX, dOut, numTiles, H, W);
        runBackward(queue, exec, dDout, dDx, numTiles, H, W);
        onHost::memcpy(queue, hOut, dOut);
        onHost::memcpy(queue, hDx, dDx);
        onHost::wait(queue);

        int rc = upsampleBench::verifyForward(
            onHost::data(hOut),
            refOut,
            B,
            C,
            H,
            W,
            upsampleBench::defaultTolerance,
            "upsample-forward");
        if(rc != EXIT_SUCCESS)
        {
            std::printf("upsample forward verification FAILED\n");
            return rc;
        }
        rc = upsampleBench::verifyBackward(
            onHost::data(hDx),
            refDx,
            B,
            C,
            H,
            W,
            upsampleBench::defaultTolerance,
            "upsample-backward");
        if(rc != EXIT_SUCCESS)
        {
            std::printf("upsample backward verification FAILED\n");
            return rc;
        }

        std::printf("upsample B=%llu C=%llu H=%llu W=%llu verification PASSED\n",
                    static_cast<unsigned long long>(B),
                    static_cast<unsigned long long>(C),
                    static_cast<unsigned long long>(H),
                    static_cast<unsigned long long>(W));
        return EXIT_SUCCESS;
    }
} // namespace upsampleAlpaka

auto main(int argc, char* argv[]) -> int
{
    upsampleBench::Params params;
    if(int const ret = upsampleBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return upsampleAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
