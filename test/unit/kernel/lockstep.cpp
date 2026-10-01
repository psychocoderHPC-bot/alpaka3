/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

namespace
{
    ALPAKA_FN_HOST_ACC constexpr auto linearValue(auto const& value)
    {
        if constexpr(alpaka::concepts::Simd<std::decay_t<decltype(value)>>)
        {
            using SimdType = std::decay_t<decltype(value)>;
            return Simd<int32_t, SimdType::width()>{
                [&](auto laneIdx) constexpr { return static_cast<int32_t>(value[static_cast<uint32_t>(laneIdx)]); }};
        }
        else
        {
            return static_cast<int32_t>(value);
        }
    }

    template<uint32_t T_numThreads>
    struct ManualLinearGroup
    {
        ALPAKA_FN_ACC constexpr auto getThreadSpace(auto const& acc) const
        {
            auto const linearThreadIdx = linearize(acc[layer::thread].count(), acc[layer::thread].idx());
            return ThreadSpace{Vec{linearThreadIdx}, CVec<uint32_t, T_numThreads>{}};
        }
    };

    struct WarpFrameExtent
    {
        constexpr auto operator()(auto const& device) const
        {
            return Vec{device.getDeviceProperties().warpSize};
        }
    };

    template<typename T_LogicalExtent, typename T_WorkGroup>
    struct LockstepKernel
    {
        T_WorkGroup workGroup;

        ALPAKA_FN_ACC void operator()(auto const& acc, concepts::IDataSource auto out) const
        {
            auto scope = onAcc::makeLockstep(acc, workGroup, T_LogicalExtent{});
            auto tmp = scope.template var<int32_t>();

            scope.concurrent(
                [](auto const& idx, auto tmpRef) { tmpRef = linearValue(idx.linear()) + int32_t{1}; },
                tmp);

            scope.template concurrent<int32_t>(
                [](auto const& idx, auto outRef, auto tmpRef)
                { outRef = tmpRef.load() * int32_t{2} - linearValue(idx.linear()); },
                onAcc::map(out),
                tmp);
        }
    };

    template<typename T_LogicalExtent, typename T_WorkGroup, typename T_FrameExtent>
    void runCase(auto cfg, T_WorkGroup const& workGroup, T_FrameExtent const& frameExtent, char const* label)
    {
        auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
        onHost::Device device = test::getDevice(deviceExec);
        concepts::Executor auto exec = test::getExecutor(deviceExec);

        auto queue = device.makeQueue();
        auto const logicalExtent = Vec{T_LogicalExtent{}};
        auto const resolvedFrameExtent = [&]
        {
            if constexpr(requires { frameExtent(device); })
                return frameExtent(device);
            else
                return frameExtent;
        }();

        INFO("device name: " << device.getName());
        INFO("executor   : " << exec.getName());
        INFO("case       : " << label);

        auto outDev = onHost::alloc<int32_t>(device, logicalExtent);
        auto outHost = onHost::allocHostLike(outDev);
        auto expectedHost = onHost::allocHostLike(outDev);

        onHost::memset(queue, outDev, 0u);
        queue.enqueue(
            onHost::FrameSpec{resolvedFrameExtent.fill(1u), resolvedFrameExtent, exec},
            KernelBundle{LockstepKernel<T_LogicalExtent, T_WorkGroup>{workGroup}, outDev});
        onHost::memcpy(queue, outHost, outDev);
        onHost::wait(queue);

        meta::ndLoopIncIdx(
            logicalExtent,
            [&](auto idx)
            {
                auto const linearIdx = linearize(T_LogicalExtent{}, idx);
                expectedHost[idx] = static_cast<int32_t>(linearIdx) + int32_t{2};
                CHECK(outHost[idx] == expectedHost[idx]);
            });
    }

    /** Concurrent functor proving full, exactly-once coverage and recording the SIMD width actually taken.
     *
     * Each logical element increments its own per-element counter once (a duplicate write makes the counter > 1). The
     * maximum SIMD width taken by any iteration is recorded through an atomic max, so the caller can assert that the
     * wide path was used when the architecture SIMD width permits it. The written output value encodes the linear
     * index, giving an independent coverage check.
     *
     * @tparam T_LogicalExtent compile-time logical extent (1D)
     */
    template<typename T_LogicalExtent, typename T_WorkGroup>
    struct TailCoverageKernel
    {
        T_WorkGroup workGroup;

        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            concepts::IMdSpan auto count,
            concepts::IMdSpan auto out,
            concepts::IMdSpan auto maxWidth) const
        {
            auto scope = onAcc::makeLockstep(acc, workGroup, T_LogicalExtent{});
            scope.template concurrent<int32_t>(
                [&](auto const& idx, auto outRef)
                {
                    using IdxType = std::remove_cvref_t<decltype(idx)>;
                    constexpr uint32_t width = IdxType::width();
                    onAcc::atomicMax(acc, maxWidth.data(), static_cast<int32_t>(width));

                    auto const linear = idx.linear();
                    if constexpr(width == 1u)
                    {
                        onAcc::atomicAdd(acc, count.data() + static_cast<uint32_t>(linear), int32_t{1});
                    }
                    else
                    {
                        for(uint32_t lane = 0u; lane < width; ++lane)
                            onAcc::atomicAdd(acc, count.data() + static_cast<uint32_t>(linear[lane]), int32_t{1});
                    }
                    outRef = linearValue(linear) + int32_t{1};
                },
                onAcc::map(out));
        }
    };

    /** Run TailCoverageKernel over the 1D compile-time logical size T_logical.
     *
     * The worker group is threadsInGrid; the frame extent is the number of grid workers, i.e. the launched frame
     * count, so the participating worker count equals @p frameExtent on every host backend (CpuSerial executes them
     * sequentially, CpuOmpBlocks in parallel).
     *
     * Verifies that every logical element is covered exactly once (per-element counter == 1 and the written value is
     * the linear index + 1) and returns the maximum resolved SIMD width observed.
     */
    template<uint32_t T_logical>
    int32_t runTailCoverageCase(auto cfg, uint32_t frameExtent, char const* label)
    {
        auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
        onHost::Device device = test::getDevice(deviceExec);
        concepts::Executor auto exec = test::getExecutor(deviceExec);

        auto queue = device.makeQueue();
        auto const logicalExtent = Vec{CVec<uint32_t, T_logical>{}};

        auto countDev = onHost::alloc<int32_t>(device, logicalExtent);
        auto outDev = onHost::alloc<int32_t>(device, logicalExtent);
        auto maxWidthDev = onHost::alloc<int32_t>(device, Vec{1u});
        auto countHost = onHost::allocHostLike(countDev);
        auto outHost = onHost::allocHostLike(outDev);
        auto maxWidthHost = onHost::allocHostLike(maxWidthDev);

        onHost::memset(queue, countDev, 0u);
        onHost::memset(queue, outDev, 0u);
        onHost::memset(queue, maxWidthDev, 0u);

        INFO("case              : " << label);
        queue.enqueue(
            onHost::FrameSpec{Vec{frameExtent}, Vec{1u}, exec},
            KernelBundle{
                TailCoverageKernel<CVec<uint32_t, T_logical>, ALPAKA_TYPEOF(onAcc::worker::threadsInGrid)>{
                    onAcc::worker::threadsInGrid},
                countDev,
                outDev,
                maxWidthDev});
        onHost::memcpy(queue, countHost, countDev);
        onHost::memcpy(queue, outHost, outDev);
        onHost::memcpy(queue, maxWidthHost, maxWidthDev);
        onHost::wait(queue);

        INFO("executor          : " << exec.getName());
        INFO("logical size      : " << T_logical);
        INFO("frame extent      : " << frameExtent);
        INFO("max resolved simd : " << maxWidthHost[Vec{0u}]);

        for(uint32_t i = 0u; i < T_logical; ++i)
        {
            CAPTURE(i);
            // Exactly once: a duplicate write would make the per-element counter > 1.
            CHECK(countHost[Vec{i}] == int32_t{1});
            // Full coverage: the written value is injective per element.
            CHECK(outHost[Vec{i}] == static_cast<int32_t>(i) + int32_t{1});
        }

        return maxWidthHost[Vec{0u}];
    }

    /** Verify idx[d] == delinearize(logicalExtent, idx.linear())[d] for all d over a 3D logical extent.
     *
     * A separate output records a per-element match flag (1) so missing coverage is detected as well.
     */
    template<typename T_WorkGroup>
    struct Delinearize3dKernel
    {
        T_WorkGroup workGroup;

        ALPAKA_FN_ACC void operator()(auto const& acc, concepts::IMdSpan auto ok) const
        {
            constexpr auto logicalExtent = CVec<uint32_t, 5u, 7u, 11u>{};
            auto scope = onAcc::makeLockstep(acc, workGroup, logicalExtent);
            scope.template concurrent<int32_t>(
                [&](auto const& idx, auto okRef)
                {
                    using IdxType = std::remove_cvref_t<decltype(idx)>;
                    constexpr uint32_t width = IdxType::width();

                    Simd<int32_t, width> match{[]([[maybe_unused]] auto laneIdx) constexpr { return int32_t{1}; }};
                    for(uint32_t lane = 0u; lane < width; ++lane)
                    {
                        if constexpr(width == 1u)
                        {
                            auto const expected
                                = onAcc::internal::delinearize(logicalExtent, static_cast<uint32_t>(idx.linear()));
                            for(uint32_t d = 0u; d < 3u; ++d)
                            {
                                if(idx[d] != expected[d])
                                    match[lane] = int32_t{0};
                            }
                        }
                        else
                        {
                            auto const expected = onAcc::internal::delinearize(
                                logicalExtent,
                                static_cast<uint32_t>(idx.linear()[lane]));
                            for(uint32_t d = 0u; d < 3u; ++d)
                            {
                                if(idx[d][lane] != expected[d])
                                    match[lane] = int32_t{0};
                            }
                        }
                    }
                    okRef = match;
                },
                onAcc::map(ok));
        }
    };

    /** Architecture SIMD width (element count) for int32_t for the backend of T_Device. */
    template<typename T_Device>
    consteval uint32_t archSimdWidthFor()
    {
        using Api = std::remove_cvref_t<decltype(std::declval<T_Device const&>().getApi())>;
        using DevKind = std::remove_cvref_t<decltype(std::declval<T_Device const&>().getDeviceKind())>;
        return getArchSimdWidth<int32_t>(Api{}, DevKind{});
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("lockstep distributed vars", "[kernel][lockstep]", TestBackends)
{
    auto cfg = TestType::makeDict();

    runCase<CVec<uint32_t, 9u, 11u>>(cfg, onAcc::worker::threadsInBlock, CVec<uint32_t, 2u, 4u>{}, "threadsInBlock");
    runCase<CVec<uint32_t, 17u, 19u>>(cfg, ManualLinearGroup<1u>{}, CVec<uint32_t, 1u>{}, "manual");

    runCase<CVec<uint32_t, 17u, 19u>>(
        cfg,
        onAcc::worker::linearThreadsInWarp,
        WarpFrameExtent{},
        "linearThreadsInWarp");
}

TEMPLATE_LIST_TEST_CASE("lockstep worker count relative to logical size", "[kernel][lockstep]", TestBackends)
{
    auto cfg = TestType::makeDict();

    // workerCount > logicalSize: more workers than elements; surplus workers must not process any element.
    {
        int32_t const maxWidth = runTailCoverageCase<5u>(cfg, 8u, "workers>logical");
        CHECK(maxWidth >= 1);
    }
    // workerCount == logicalSize: exactly one element per worker, so only the scalar path can be taken.
    {
        int32_t const maxWidth = runTailCoverageCase<8u>(cfg, 8u, "workers==logical");
        CHECK(maxWidth >= 1);
    }
    // workerCount < logicalSize: every worker owns several elements.
    {
        int32_t const maxWidth = runTailCoverageCase<21u>(cfg, 4u, "workers<logical");
        CHECK(maxWidth >= 1);
    }
}

TEMPLATE_LIST_TEST_CASE(
    "lockstep simd tail covers every logical element exactly once",
    "[kernel][lockstep]",
    TestBackends)
{
    auto cfg = TestType::makeDict();

    // Deterministic SIMD tail regression case.
    //
    // The logical extent is shaped `k * width * workerCount + r` with r > 0 and k >= 1. For the int32_t value type
    // the selected width is 4 on SSE2+ x86 builds and can be larger elsewhere; with k=2, width=4 and workers=3 the
    // core part is 24 and r=3 gives 27 elements. The final wide iteration of each worker is followed by a scalar tail
    // and the total is not a multiple of width * workerCount, so a wrong tail bound either drops the trailing elements
    // or writes them twice; both are caught by the counter and value checks.
    constexpr uint32_t logical = 2u * 4u * 3u + 3u;
    constexpr uint32_t workers = 3u;

    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device device = test::getDevice(deviceExec);
    constexpr uint32_t archWidth = archSimdWidthFor<ALPAKA_TYPEOF(device)>();

    int32_t const maxWidth = runTailCoverageCase<logical>(cfg, workers, "simd tail");

    INFO("arch simd width for int32_t: " << archWidth);
    if constexpr(archWidth > 1u)
    {
        // The architecture SIMD width permits the wide path, so at least one iteration must have taken it.
        CHECK(maxWidth > 1);
    }
    else
    {
        // No wide path is available on this build: full, exactly-once coverage (checked above) must still hold.
        CHECK(maxWidth == 1);
    }
}

TEMPLATE_LIST_TEST_CASE("lockstep simd tail with larger remainder", "[kernel][lockstep]", TestBackends)
{
    auto cfg = TestType::makeDict();

    // Second `k * width * workerCount + r` shape with a larger remainder and fewer workers, so only part of the
    // workers can still take the wide path near the end of the iteration space.
    constexpr uint32_t logical = 5u * 4u * 2u + 5u;
    constexpr uint32_t workers = 2u;

    int32_t const maxWidth = runTailCoverageCase<logical>(cfg, workers, "simd tail");
    CHECK(maxWidth >= 1);
}

TEMPLATE_LIST_TEST_CASE("lockstep 3d index delinearization", "[kernel][lockstep]", TestBackends)
{
    auto cfg = TestType::makeDict();

    auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    auto queue = device.makeQueue();

    // 3D logical extent in the style of CVec<uint32_t, 5u, 7u, 11u>: 385 elements. 512 workers make workerCount
    // larger than the logical size, so the resolved SIMD width is clamped to 1 and the scalar index path is used.
    constexpr auto logicalExtent = CVec<uint32_t, 5u, 7u, 11u>{};
    constexpr uint32_t workers = 512u;

    auto okDev = onHost::alloc<int32_t>(device, Vec{logicalExtent});
    auto okHost = onHost::allocHostLike(okDev);
    // Sentinel 0; every covered element must be overwritten with a match flag (1).
    onHost::memset(queue, okDev, 0u);

    queue.enqueue(
        onHost::FrameSpec{Vec{workers}, Vec{1u}, exec},
        KernelBundle{
            Delinearize3dKernel<ALPAKA_TYPEOF(onAcc::worker::threadsInGrid)>{onAcc::worker::threadsInGrid},
            okDev});
    onHost::memcpy(queue, okHost, okDev);
    onHost::wait(queue);

    // Every logical element must be covered exactly once and satisfy
    // idx[d] == delinearize(logicalExtent, idx.linear())[d] for all d.
    meta::ndLoopIncIdx(Vec{logicalExtent}, [&](auto idx) { CHECK(okHost[idx] == int32_t{1}); });
}
