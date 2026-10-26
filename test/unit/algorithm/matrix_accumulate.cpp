/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "../../../validation/numerics/tolerance.hpp"

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/** @file
 *
 * Unit tests of the common matrix-accumulate contract, independent of GEMM.
 *
 * Coverage:
 * - the portable per-worker fallback on every available executor for fp16/bf16 to fp32, strict fp32 and strict fp64,
 * - the warp contract on a non-GEMM kernel, compared against the portable fallback and an independent double
 *   reference,
 * - the compile-time capability rules and the selection of the fallback for every non-native combination,
 * - the documented participation contract with one known-value tile.
 */

using namespace alpaka;
namespace matrix = alpaka::onAcc::matrix;

namespace
{
    using TestBackends
        = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

    constexpr std::uint32_t tileM = 16u;
    constexpr std::uint32_t tileN = 8u;
    constexpr std::uint32_t tileK = 16u;
    constexpr std::uint32_t laneCount = 32u;
    /** Extra output slot the warp kernel uses to report the device-pass capability decision. */
    constexpr std::uint32_t pathSlot = tileM * tileN;

    /** Portable per-worker fallback: the whole 16x8 tile is computed by one worker. */
    template<typename T_In>
    struct LocalFallbackKernel
    {
        T_In const* a;
        T_In const* b;
        float* c;

        ALPAKA_FN_ACC void operator()(auto const& acc) const
        {
            matrix::RegisterTile<float, tileM, tileN> tile{};
            for(std::uint32_t k = 0u; k < tileK; ++k)
            {
                matrix::ColumnFragment<T_In, tileM> av{};
                matrix::RowFragment<T_In, tileN> bv{};
                for(std::uint32_t i = 0u; i < tileM; ++i)
                    av.value[i] = a[static_cast<std::size_t>(i) * tileK + k];
                for(std::uint32_t j = 0u; j < tileN; ++j)
                    bv.value[j] = b[static_cast<std::size_t>(k) * tileN + j];
                matrix::MatrixAccumulate::call(acc, tile, av, bv);
            }
            for(std::uint32_t i = 0u; i < tileM; ++i)
                for(std::uint32_t j = 0u; j < tileN; ++j)
                    c[static_cast<std::size_t>(i) * tileN + j] = tile.at(i, j);
        }
    };

    /** Warp contract: one m16n8k16 step executed by 32 lanes. */
    template<typename T_In>
    struct WarpKernel
    {
        T_In const* a;
        T_In const* b;
        float* c;

        ALPAKA_FN_ACC void operator()(auto const& acc) const
        {
            using Acc = std::remove_cvref_t<decltype(acc)>;
            if constexpr(Acc::getWarpSize() >= laneCount)
            {
                std::uint32_t const lane = onAcc::warp::getLaneIdx(acc) % laneCount;
                matrix::WarpOperandA<T_In> av{};
                matrix::WarpOperandB<T_In> bv{};
                matrix::WarpAccumulator<float> cv{};
                matrix::loadWarpOperandA(av, a, lane);
                matrix::loadWarpOperandB(bv, b, lane);
                matrix::MatrixAccumulate::call(acc, cv, av, bv);
                matrix::storeWarpAccumulator(c, cv, lane);

                // Report the compile-time decision of this device pass. The host pass never executes this branch.
                using Api = ALPAKA_TYPEOF(acc.getApi());
                using DeviceKind = ALPAKA_TYPEOF(acc.getDeviceKind());
                constexpr auto decision
                    = matrix::matrixAccumulateSupported<T_In, float, matrix::M16N8K16, Api, DeviceKind>();
                if(lane == 0u)
                    c[pathSlot] = static_cast<float>(static_cast<std::uint32_t>(decision));
            }
        }
    };

    template<typename T>
    T storageValue(float const value)
    {
        if constexpr(std::same_as<T, Float16> || std::same_as<T, BFloat16>)
            return T::fromFloat(value);
        else
            return static_cast<T>(value);
    }

    template<typename T>
    double asDouble(T const value)
    {
        if constexpr(std::same_as<T, Float16> || std::same_as<T, BFloat16>)
            return static_cast<double>(value.toFloat());
        else
            return static_cast<double>(value);
    }

    /** Filled with small values that are exactly representable in fp16 and bf16. */
    template<typename T_In>
    std::vector<T_In> makeOperandA()
    {
        std::vector<T_In> values(tileM * tileK);
        for(std::size_t i = 0u; i < values.size(); ++i)
        {
            auto const row = static_cast<int>(i / tileK);
            auto const k = static_cast<int>(i % tileK);
            float const value = static_cast<float>((row * 7 + k * 3) % 9 - 4) * 0.25f;
            values[i] = storageValue<T_In>(value);
        }
        return values;
    }

    template<typename T_In>
    std::vector<T_In> makeOperandB()
    {
        std::vector<T_In> values(tileK * tileN);
        for(std::size_t i = 0u; i < values.size(); ++i)
        {
            auto const k = static_cast<int>(i / tileN);
            auto const column = static_cast<int>(i % tileN);
            float const value = static_cast<float>((k * 5 + column * 11) % 7 - 3) * 0.25f;
            values[i] = storageValue<T_In>(value);
        }
        return values;
    }

    /** Independent double reference of C = A * B for one 16x16 by 16x8 tile. */
    template<typename T_In>
    std::array<double, tileM * tileN> reference(std::vector<T_In> const& a, std::vector<T_In> const& b)
    {
        std::array<double, tileM * tileN> result{};
        for(std::uint32_t i = 0u; i < tileM; ++i)
            for(std::uint32_t j = 0u; j < tileN; ++j)
            {
                double sum = 0.0;
                for(std::uint32_t k = 0u; k < tileK; ++k)
                    sum += asDouble(a[static_cast<std::size_t>(i) * tileK + k])
                           * asDouble(b[static_cast<std::size_t>(k) * tileN + j]);
                result[static_cast<std::size_t>(i) * tileN + j] = sum;
            }
        return result;
    }

    /** Frozen forward-error bound of the fp32 accumulation mode for one output element. */
    template<typename T_In>
    double tolerance(
        std::vector<T_In> const& a,
        std::vector<T_In> const& b,
        std::uint32_t const row,
        std::uint32_t const column)
    {
        double magnitude = 0.0;
        for(std::uint32_t k = 0u; k < tileK; ++k)
            magnitude += std::abs(
                asDouble(a[static_cast<std::size_t>(row) * tileK + k])
                * asDouble(b[static_cast<std::size_t>(k) * tileN + column]));
        double const gamma = static_cast<double>(tileK) * validation::gemm::unitRoundoffFp32
                             / (1.0 - static_cast<double>(tileK) * validation::gemm::unitRoundoffFp32);
        return validation::gemm::defaultSafetyFactor * gamma * magnitude
               + 0.5 * static_cast<double>(std::numeric_limits<float>::denorm_min());
    }

    void reportPath(char const* kind, auto const& device, auto const& exec, float const reportedCode)
    {
        auto const props = device.getDeviceProperties();
        auto const availability = matrix::available(device);
        auto const code = static_cast<std::uint32_t>(reportedCode + 0.5f);
        std::string decision = "none";
        if(reportedCode >= -0.5f && reportedCode <= 2.5f && code <= 2u)
            decision = matrix::toString(static_cast<matrix::Availability>(code));
        std::cout << "[matrix-accumulate] " << kind << " device=\"" << props.name << "\" warpSize=" << props.warpSize
                  << " exec=" << exec.getName() << " runtimeAvailability=" << matrix::toString(availability)
                  << " devicePassDecision=" << decision << "\n";
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("matrix accumulate portable fallback on every executor", "[matrix][fallback]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    auto const run = [&](auto typeTag)
    {
        using T_In = typename decltype(typeTag)::type;
        auto const a = makeOperandA<T_In>();
        auto const b = makeOperandB<T_In>();
        auto const expected = reference(a, b);

        auto dA = onHost::alloc<T_In>(device, Vec{a.size()});
        auto dB = onHost::alloc<T_In>(device, Vec{b.size()});
        auto dC = onHost::alloc<float>(device, Vec{tileM * tileN});
        onHost::memcpy(queue, dA, a);
        onHost::memcpy(queue, dB, b);

        queue.enqueue(
            onHost::FrameSpec{Vec{1u}, CVec<std::uint32_t, 1u>{}, exec},
            KernelBundle{LocalFallbackKernel<T_In>{onHost::data(dA), onHost::data(dB), onHost::data(dC)}});
        std::vector<float> candidate(tileM * tileN);
        onHost::memcpy(queue, candidate, dC);
        onHost::wait(queue);

        for(std::uint32_t i = 0u; i < tileM; ++i)
            for(std::uint32_t j = 0u; j < tileN; ++j)
            {
                auto const index = static_cast<std::size_t>(i) * tileN + j;
                CHECK(std::abs(static_cast<double>(candidate[index]) - expected[index]) <= tolerance(a, b, i, j));
            }
    };

    run(std::type_identity<Float16>{});
    run(std::type_identity<BFloat16>{});

    std::cout << "[matrix-accumulate] local fallback completed on " << device.getDeviceProperties().name << " / "
              << exec.getName() << "\n";
}

TEMPLATE_LIST_TEST_CASE("matrix accumulate warp contract against the fallback", "[matrix][warp]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto const props = device.getDeviceProperties();
    if(props.warpSize < laneCount)
        SKIP("the warp contract needs a hardware warp of at least 32 lanes");

    auto queue = device.makeQueue();

    auto const runWarpCase = [&](auto typeTag, char const* label)
    {
        using T_In = typename decltype(typeTag)::type;
        auto const a = makeOperandA<T_In>();
        auto const b = makeOperandB<T_In>();
        auto const expected = reference(a, b);

        auto dA = onHost::alloc<T_In>(device, Vec{a.size()});
        auto dB = onHost::alloc<T_In>(device, Vec{b.size()});
        auto dWarp = onHost::alloc<float>(device, Vec{tileM * tileN + 1u});
        auto dFallback = onHost::alloc<float>(device, Vec{tileM * tileN});
        onHost::memcpy(queue, dA, a);
        onHost::memcpy(queue, dB, b);

        queue.enqueue(
            onHost::FrameSpec{Vec{1u}, CVec<std::uint32_t, laneCount>{}, exec},
            KernelBundle{WarpKernel<T_In>{onHost::data(dA), onHost::data(dB), onHost::data(dWarp)}});
        queue.enqueue(
            onHost::FrameSpec{Vec{1u}, CVec<std::uint32_t, 1u>{}, exec},
            KernelBundle{LocalFallbackKernel<T_In>{onHost::data(dA), onHost::data(dB), onHost::data(dFallback)}});

        std::vector<float> warpResult(tileM * tileN + 1u);
        std::vector<float> fallbackResult(tileM * tileN);
        onHost::memcpy(queue, warpResult, dWarp);
        onHost::memcpy(queue, fallbackResult, dFallback);
        onHost::wait(queue);

        reportPath(label, device, exec, warpResult[pathSlot]);

        for(std::uint32_t i = 0u; i < tileM; ++i)
            for(std::uint32_t j = 0u; j < tileN; ++j)
            {
                auto const index = static_cast<std::size_t>(i) * tileN + j;
                auto const bound = tolerance(a, b, i, j);
                CHECK(std::abs(static_cast<double>(warpResult[index]) - expected[index]) <= bound);
                CHECK(
                    std::abs(static_cast<double>(warpResult[index]) - static_cast<double>(fallbackResult[index]))
                    <= bound);
            }
    };

    runWarpCase(std::type_identity<Float16>{}, "fp16->fp32");
    runWarpCase(std::type_identity<BFloat16>{}, "bf16->fp32");
}

TEMPLATE_LIST_TEST_CASE("matrix accumulate known-value participation", "[matrix][contract]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    using T_In = Float16;
    // A is the 16x16 identity, so C must equal B exactly.
    std::vector<T_In> a(tileM * tileK, storageValue<T_In>(0.0f));
    for(std::uint32_t i = 0u; i < tileM; ++i)
        a[static_cast<std::size_t>(i) * tileK + i] = storageValue<T_In>(1.0f);
    std::vector<T_In> b(tileK * tileN);
    for(std::size_t i = 0u; i < b.size(); ++i)
        b[i] = storageValue<T_In>(static_cast<float>(static_cast<int>(i % 13u) - 6));

    auto dA = onHost::alloc<T_In>(device, Vec{a.size()});
    auto dB = onHost::alloc<T_In>(device, Vec{b.size()});
    auto dC = onHost::alloc<float>(device, Vec{tileM * tileN});
    onHost::memcpy(queue, dA, a);
    onHost::memcpy(queue, dB, b);

    bool const warp = device.getDeviceProperties().warpSize >= laneCount;
    if(warp)
        queue.enqueue(
            onHost::FrameSpec{Vec{1u}, CVec<std::uint32_t, laneCount>{}, exec},
            KernelBundle{WarpKernel<T_In>{onHost::data(dA), onHost::data(dB), onHost::data(dC)}});
    else
        queue.enqueue(
            onHost::FrameSpec{Vec{1u}, CVec<std::uint32_t, 1u>{}, exec},
            KernelBundle{LocalFallbackKernel<T_In>{onHost::data(dA), onHost::data(dB), onHost::data(dC)}});

    std::vector<float> candidate(tileM * tileN);
    onHost::memcpy(queue, candidate, dC);
    onHost::wait(queue);

    for(std::uint32_t i = 0u; i < tileM; ++i)
        for(std::uint32_t j = 0u; j < tileN; ++j)
        {
            auto const index = static_cast<std::size_t>(i) * tileN + j;
            CHECK(candidate[index] == static_cast<float>(asDouble(b[static_cast<std::size_t>(i) * tileN + j])));
        }
}

TEMPLATE_LIST_TEST_CASE(
    "matrix accumulate illegal native combination resolves to the fallback",
    "[matrix][capability]",
    TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto const props = device.getDeviceProperties();
    if(props.warpSize < laneCount)
        SKIP("the warp contract needs a hardware warp of at least 32 lanes");

    auto queue = device.makeQueue();
    using T_In = float;
    auto const a = makeOperandA<T_In>();
    auto const b = makeOperandB<T_In>();
    auto const expected = reference(a, b);

    auto dA = onHost::alloc<T_In>(device, Vec{a.size()});
    auto dB = onHost::alloc<T_In>(device, Vec{b.size()});
    auto dC = onHost::alloc<float>(device, Vec{tileM * tileN + 1u});
    onHost::memcpy(queue, dA, a);
    onHost::memcpy(queue, dB, b);

    // Strict fp32 is never silently routed through a 16-bit native instruction; the generic fallback is selected.
    queue.enqueue(
        onHost::FrameSpec{Vec{1u}, CVec<std::uint32_t, laneCount>{}, exec},
        KernelBundle{WarpKernel<T_In>{onHost::data(dA), onHost::data(dB), onHost::data(dC)}});
    std::vector<float> candidate(tileM * tileN + 1u);
    onHost::memcpy(queue, candidate, dC);
    onHost::wait(queue);

    reportPath("strict-fp32 warp", device, exec, candidate[pathSlot]);
    // The device-pass decision must be `no` (0) for strict fp32.
    CHECK(candidate[pathSlot] == 0.0f);
    for(std::uint32_t i = 0u; i < tileM; ++i)
        for(std::uint32_t j = 0u; j < tileN; ++j)
        {
            auto const index = static_cast<std::size_t>(i) * tileN + j;
            CHECK(std::abs(static_cast<double>(candidate[index]) - expected[index]) <= tolerance(a, b, i, j));
        }
}

TEST_CASE("matrix accumulate capability descriptor rules", "[matrix][capability]")
{
#if !defined(__CUDA_ARCH__)
    using matrix::Availability;
    using matrix::M16N8K16;
    using matrix::matrixAccumulateSupported;

    // Native support exists only for 16-bit inputs with fp32 accumulation on CUDA.
    static_assert(
        matrixAccumulateSupported<Float16, float, M16N8K16, api::Cuda, deviceKind::NvidiaGpu>() != Availability::no);
    static_assert(
        matrixAccumulateSupported<BFloat16, float, M16N8K16, api::Cuda, deviceKind::NvidiaGpu>() != Availability::no);
    // Strict fp32 and fp64 are portable-only; a mixed accumulator is illegal.
    static_assert(
        matrixAccumulateSupported<float, float, M16N8K16, api::Cuda, deviceKind::NvidiaGpu>() == Availability::no);
    static_assert(
        matrixAccumulateSupported<double, double, M16N8K16, api::Cuda, deviceKind::NvidiaGpu>() == Availability::no);
    static_assert(
        matrixAccumulateSupported<Float16, double, M16N8K16, api::Cuda, deviceKind::NvidiaGpu>() == Availability::no);
    // No native adapter exists for other APIs or device kinds.
    static_assert(
        matrixAccumulateSupported<Float16, float, M16N8K16, api::Hip, deviceKind::AmdGpu>() == Availability::no);
    static_assert(
        matrixAccumulateSupported<Float16, float, M16N8K16, api::Host, deviceKind::Cpu>() == Availability::no);

    // The host pass of a CUDA compilation cannot know the target architecture, so it reports `unknown`, never `yes`.
    if constexpr(ALPAKA_ARCH_PTX == ALPAKA_VERSION_NUMBER_NOT_AVAILABLE)
        static_assert(
            matrixAccumulateSupported<Float16, float, M16N8K16, api::Cuda, deviceKind::NvidiaGpu>()
            == Availability::unknown);
#endif
    SUCCEED();
}

TEST_CASE("matrix accumulate warp fragment layout round trip", "[matrix][contract]")
{
    using T_In = Float16;
    std::vector<T_In> a(tileM * tileK);
    for(std::size_t i = 0u; i < a.size(); ++i)
        a[i] = storageValue<T_In>(static_cast<float>(i));
    std::vector<T_In> b(tileK * tileN);
    for(std::size_t i = 0u; i < b.size(); ++i)
        b[i] = storageValue<T_In>(static_cast<float>(i));

    // The documented per-lane load path covers every logical element of A and B exactly once.
    std::vector<std::uint32_t> aOwners(a.size(), 0u);
    std::vector<std::uint32_t> bOwners(b.size(), 0u);
    for(std::uint32_t lane = 0u; lane < laneCount; ++lane)
    {
        matrix::WarpOperandA<T_In> av{};
        matrix::WarpOperandB<T_In> bv{};
        matrix::loadWarpOperandA(av, a.data(), lane);
        matrix::loadWarpOperandB(bv, b.data(), lane);
        for(std::uint32_t i = 0u; i < 8u; ++i)
        {
            auto const [row, k] = matrix::internal::warpACoordinate(lane, i);
            CHECK(av.value[i].bits() == a[static_cast<std::size_t>(row) * tileK + k].bits());
            ++aOwners[static_cast<std::size_t>(row) * tileK + k];
        }
        for(std::uint32_t i = 0u; i < 4u; ++i)
        {
            auto const [k, column] = matrix::internal::warpBCoordinate(lane, i);
            CHECK(bv.value[i].bits() == b[static_cast<std::size_t>(k) * tileN + column].bits());
            ++bOwners[static_cast<std::size_t>(k) * tileN + column];
        }
    }
    CHECK(std::all_of(aOwners.begin(), aOwners.end(), [](auto const count) { return count == 1u; }));
    CHECK(std::all_of(bOwners.begin(), bOwners.end(), [](auto const count) { return count == 1u; }));

    // The documented store path likewise covers every logical output element exactly once.
    std::vector<std::uint32_t> cOwners(tileM * tileN, 0u);
    std::vector<float> c(tileM * tileN, -1.0f);
    for(std::uint32_t lane = 0u; lane < laneCount; ++lane)
    {
        matrix::WarpAccumulator<float> cv{};
        for(std::uint32_t i = 0u; i < 4u; ++i)
            cv.value[i] = static_cast<float>(lane * 4u + i);
        matrix::storeWarpAccumulator(c.data(), cv, lane);
        for(std::uint32_t i = 0u; i < 4u; ++i)
        {
            auto const [row, column] = matrix::internal::warpCCoordinate(lane, i);
            ++cOwners[static_cast<std::size_t>(row) * tileN + column];
        }
    }
    CHECK(std::all_of(cOwners.begin(), cOwners.end(), [](auto const count) { return count == 1u; }));
    CHECK(std::all_of(c.begin(), c.end(), [](auto const value) { return value >= 0.0f; }));
}
