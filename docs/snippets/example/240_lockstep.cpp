/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <numeric>
#include <vector>

using namespace alpaka;

// BEGIN-TUTORIAL-lockstepKernel
struct LockstepDistributedVarKernel
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc, concepts::IMdSpan auto out) const
    {
        // The logical iteration space is compile-time known: 8 elements in one dimension.
        auto scope = onAcc::makeLockstep(acc, onAcc::worker::threadsInBlock, CVec<uint32_t, 8u>{});

        // var() is a distributed per-worker array of registers/locals, partitioned over the workers.
        auto tmp = scope.template var<int32_t>();

        // First lockstep pass: fill this worker's elements of `tmp`.
        scope.concurrent([](auto const& idx, auto tmpRef) { tmpRef = idx.linear() + 1u; }, tmp);

        // Second lockstep pass: read the distributed array back and write the result to global memory.
        scope.concurrent(
            []([[maybe_unused]] auto const& idx, auto outRef, auto tmpRef) { outRef = tmpRef.load() * int32_t{2}; },
            out,
            tmp);
    }
};

// END-TUTORIAL-lockstepKernel

TEMPLATE_LIST_TEST_CASE("tutorial lockstep distributed array", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict());
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::vector<int32_t> hostOutput(8u, 0);
    auto outputBuffer = onHost::allocLike(device, hostOutput);

    // BEGIN-TUTORIAL-lockstepLaunch
    // One frame covering the logical iteration space; the workers of the block process it in lockstep.
    // The frame extent is compile-time known, so the worker count seen by var() is a compile-time value.
    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{Vec{1u}, CVec<uint32_t, 8u>{}};
    queue.enqueue(frameSpec, KernelBundle{LockstepDistributedVarKernel{}, outputBuffer});
    // END-TUTORIAL-lockstepLaunch

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(uint32_t i = 0u; i < hostOutput.size(); ++i)
        CHECK(hostOutput[i] == static_cast<int32_t>((i + 1u) * 2u));
}

// BEGIN-TUTORIAL-lockstepExplicitWidth
struct LockstepExplicitWidthKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto outDeduced,
        concepts::IMdSpan auto outExplicit,
        concepts::IDataSource auto const& in) const
    {
        auto scope = onAcc::makeLockstep(acc, onAcc::worker::threadsInBlock, CVec<uint32_t, 8u>{});

        // Deduced overload: the SIMD width is derived from the value type of the first data argument (here int32_t).
        scope.concurrent(
            []([[maybe_unused]] auto const& idx, auto outRef, auto inRef) { outRef = inRef.load() + int32_t{1}; },
            outDeduced,
            in);

        // Explicit overload: the value type given as template parameter selects the SIMD width.
        // Use this overload when the data has no value type (e.g. mapped data) or to steer the width deliberately.
        scope.template concurrent<int32_t>(
            []([[maybe_unused]] auto const& idx, auto outRef, auto inRef) { outRef = inRef.load() + int32_t{1}; },
            outExplicit,
            in);
    }
};

// END-TUTORIAL-lockstepExplicitWidth

TEMPLATE_LIST_TEST_CASE("tutorial lockstep explicit simd width", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict());
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::vector<int32_t> hostInput(8u);
    std::iota(hostInput.begin(), hostInput.end(), 0);
    std::vector<int32_t> hostDeduced(8u, 0);
    std::vector<int32_t> hostExplicit(8u, 0);

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto deducedBuffer = onHost::allocLike(device, hostDeduced);
    auto explicitBuffer = onHost::allocLike(device, hostExplicit);

    onHost::memcpy(queue, inputBuffer, hostInput);

    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{Vec{1u}, Vec{8u}};
    queue.enqueue(frameSpec, KernelBundle{LockstepExplicitWidthKernel{}, deducedBuffer, explicitBuffer, inputBuffer});

    onHost::memcpy(queue, hostDeduced, deducedBuffer);
    onHost::memcpy(queue, hostExplicit, explicitBuffer);
    onHost::wait(queue);

    for(uint32_t i = 0u; i < hostInput.size(); ++i)
    {
        CHECK(hostDeduced[i] == hostInput[i] + 1);
        CHECK(hostExplicit[i] == hostInput[i] + 1);
    }
}

// BEGIN-TUTORIAL-lockstepMapped
struct LockstepMappedKernel
{
    ALPAKA_FN_ACC void operator()(
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& in) const
    {
        // 7 logical elements; the shifted read below needs one extra input element as halo.
        auto scope = onAcc::makeLockstep(acc, onAcc::worker::threadsInBlock, CVec<uint32_t, 7u>{});

        // The explicit value type is required because mapped data carries no value_type for deduction.
        scope.template concurrent<int32_t>(
            []([[maybe_unused]] auto const& idx, auto outRef, auto current, auto next)
            { outRef = current.load() + next.load(); },
            // Identity mapping: accesses `out` at the logical index of every SIMD lane.
            onAcc::map(out),
            // Identity mapping for the current input element.
            onAcc::map(in),
            // Offset mapping: adds Vec{1u} to the logical index, i.e. reads the right neighbor.
            onAcc::map(in, Vec{1u}));
    }
};

// END-TUTORIAL-lockstepMapped

TEMPLATE_LIST_TEST_CASE("tutorial lockstep mapped indexing", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict());
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::vector<int32_t> hostInput(8u);
    std::iota(hostInput.begin(), hostInput.end(), 0);
    std::vector<int32_t> hostOutput(7u, 0);

    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto outputBuffer = onHost::allocLike(device, hostOutput);

    onHost::memcpy(queue, inputBuffer, hostInput);

    onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{Vec{1u}, Vec{7u}};
    queue.enqueue(frameSpec, KernelBundle{LockstepMappedKernel{}, outputBuffer, inputBuffer});

    onHost::memcpy(queue, hostOutput, outputBuffer);
    onHost::wait(queue);

    for(uint32_t i = 0u; i < hostOutput.size(); ++i)
        CHECK(hostOutput[i] == hostInput[i] + hostInput[i + 1u]);
}
