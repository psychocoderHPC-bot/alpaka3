/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

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
