/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

namespace
{
    template<typename T_ValueType>
    struct FloatToNarrow
    {
        ALPAKA_FN_ACC auto operator()(concepts::Simd auto const& in) const
        {
            constexpr uint32_t width = ALPAKA_TYPEOF(in)::width();
            Simd<T_ValueType, width> out = Simd<T_ValueType, width>::fill(T_ValueType::fromFloat(0.0f));
            for(uint32_t i = 0u; i < width; ++i)
                out[i] = T_ValueType::fromFloat(in[i]);
            return out;
        }
    };

    struct NarrowToFloat
    {
        ALPAKA_FN_ACC auto operator()(concepts::Simd auto const& in) const
        {
            using ValueType = GetValueType_t<ALPAKA_TYPEOF(in)>;
            constexpr uint32_t width = ALPAKA_TYPEOF(in)::width();
            // broadcast, lane access and a local scalar value must work for the storage type
            Simd<ValueType, width> const broadcast = Simd<ValueType, width>::fill(ValueType::fromFloat(1.0f));
            Simd<float, width> out;
            for(uint32_t i = 0u; i < width; ++i)
            {
                ValueType const local = in[i];
                out[i] = local.toFloat() * broadcast[i].toFloat();
            }
            return out;
        }
    };

    std::vector<float> testValues()
    {
        std::vector<float> values{
            0.0f,
            -0.0f,
            1.0f,
            -1.0f,
            0.5f,
            -2.5f,
            1.0f / 3.0f,
            65504.0f,
            1.0e-7f,
            std::ldexp(1.0f, -24),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
        };
        for(uint32_t i = 0u; i < 64u; ++i)
            values.push_back(float(i) * 0.0625f - 1.5f);
        return values;
    }

    template<typename T_ValueType>
    void runBackendTest(auto const& cfg)
    {
        auto deviceExec = test::getDeviceExecutorOrSkipTest(cfg);
        onHost::Device device = test::getDevice(deviceExec);
        concepts::Executor auto exec = test::getExecutor(deviceExec);
        onHost::Queue queue = device.makeQueue();

        CHECK(getArchSimdWidth<T_ValueType>(queue) > 0u);
        CHECK(getNumElemPerThread<T_ValueType>(queue) >= getArchSimdWidth<T_ValueType>(queue));

        auto values = testValues();
        for(auto extent : {Vec{1001u}, Vec{37u}, Vec{8u}})
        {
            auto const numElements = static_cast<size_t>(extent.x());
            std::vector<float> input(numElements);
            for(size_t i = 0u; i < numElements; ++i)
                input[i] = values[i % values.size()];

            // host reference computed with the same public conversion, the conversion itself is verified elsewhere
            std::vector<T_ValueType> expectedNarrow(numElements);
            std::vector<float> expectedFloat(numElements);
            for(size_t i = 0u; i < numElements; ++i)
            {
                expectedNarrow[i] = T_ValueType::fromFloat(input[i]);
                expectedFloat[i] = expectedNarrow[i].toFloat();
            }

            auto deviceFloatIn = onHost::alloc<float>(device, extent);
            auto deviceFloatOut = onHost::alloc<float>(device, extent);
            auto deviceNarrow = onHost::alloc<T_ValueType>(device, extent);
            auto hostFloatIn = onHost::allocHostLike(deviceFloatIn);
            auto hostFloatOut = onHost::allocHostLike(deviceFloatOut);
            auto hostNarrow = onHost::allocHostLike(deviceNarrow);

            for(size_t i = 0u; i < numElements; ++i)
                hostFloatIn[i] = input[i];

            onHost::memcpy(queue, deviceFloatIn, hostFloatIn);
            onHost::transform(queue, exec, deviceNarrow, FloatToNarrow<T_ValueType>{}, deviceFloatIn);
            onHost::transform(queue, exec, deviceFloatOut, NarrowToFloat{}, deviceNarrow);
            onHost::memcpy(queue, hostNarrow, deviceNarrow);
            onHost::memcpy(queue, hostFloatOut, deviceFloatOut);
            onHost::wait(queue);

            for(size_t i = 0u; i < numElements; ++i)
            {
                INFO("extent " << extent << " element " << i << " input " << input[i]);
                CHECK(hostNarrow[i].bits() == expectedNarrow[i].bits());
                CHECK(std::bit_cast<uint32_t>(hostFloatOut[i]) == std::bit_cast<uint32_t>(expectedFloat[i]));
            }
        }
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("fp16 simd storage and conversion", "[precision][fp16][simd]", TestBackends)
{
    runBackendTest<Float16>(TestType::makeDict());
}

TEMPLATE_LIST_TEST_CASE("bf16 simd storage and conversion", "[precision][bf16][simd]", TestBackends)
{
    runBackendTest<BFloat16>(TestType::makeDict());
}

TEST_CASE("narrow simd load and store", "[precision][simd]")
{
    // explicit load/store of a narrow SIMD pack on the host, including an unaligned start offset
    constexpr uint32_t width = 8u;
    Float16 input[width + 1u];
    for(uint32_t i = 0u; i < width + 1u; ++i)
        input[i] = Float16::fromFloat(float(i) * 0.25f);

    Simd<Float16, width> pack = Simd<Float16, width>::fill(Float16::fromFloat(-1.0f));
    CHECK(pack.width() == width);
    for(uint32_t i = 0u; i < width; ++i)
        CHECK(pack[i].toFloat() == -1.0f);

    pack.copyFrom(input, Alignment<alignof(Float16)>{});
    for(uint32_t i = 0u; i < width; ++i)
        CHECK(pack[i].toFloat() == float(i) * 0.25f);

    Float16 output[width + 1u];
    pack.copyTo(output, Alignment<alignof(Float16)>{});
    for(uint32_t i = 0u; i < width; ++i)
        CHECK(output[i].bits() == input[i].bits());

    // the y/z/w accessors and the revert helper are storage agnostic
    auto const reverted = pack.revert();
    CHECK(reverted[0].toFloat() == pack[width - 1u].toFloat());
}
