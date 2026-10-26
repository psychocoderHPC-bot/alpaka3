/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "NarrowFloatReference.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

using namespace alpaka;

namespace
{
    constexpr uint16_t mantissaMask = 0x3ffu;

    uint16_t expectedNaNBits(float const value)
    {
        uint32_t const floatBits = std::bit_cast<uint32_t>(value);
        uint32_t const sign = (floatBits >> 31u) << 15u;
        uint32_t const payload = ((floatBits & 0x7f'ffffu) >> 13u) | 0x200u;
        return uint16_t(sign | 0x7c00u | payload);
    }

    bool isNaNBits(uint16_t const bits)
    {
        return (bits & 0x7c00u) == 0x7c00u && (bits & mantissaMask) != 0u;
    }

    bool isSignalingNaNBits(uint16_t const bits)
    {
        return isNaNBits(bits) && (bits & 0x200u) == 0u;
    }
} // namespace

TEST_CASE("fp16 type properties", "[precision][fp16]")
{
    STATIC_REQUIRE(sizeof(Float16) == 2u);
    STATIC_REQUIRE(alignof(Float16) == 2u);
    STATIC_REQUIRE(std::is_trivially_copyable_v<Float16>);
    STATIC_REQUIRE(std::is_standard_layout_v<Float16>);
    STATIC_REQUIRE(std::is_same_v<GetValueType_t<Float16>, Float16>);
    STATIC_REQUIRE(std::is_same_v<GetValueType_t<Float16 const>, Float16 const>);
    STATIC_REQUIRE(isKernelArgumentTriviallyCopyable_v<Float16>);

    // the conversion is usable in constant expressions
    STATIC_REQUIRE(Float16::fromFloat(1.0f).bits() == 0x3c00u);
    STATIC_REQUIRE(Float16::fromFloat(1.5f).bits() == 0x3e00u);
    STATIC_REQUIRE(Float16::fromFloat(-0.0f).bits() == 0x8000u);
    STATIC_REQUIRE(Float16::fromBits(0x3c00u).toFloat() == 1.0f);
    STATIC_REQUIRE(static_cast<float>(Float16::fromFloat(2.0f)) == 2.0f);
}

TEST_CASE("fp16 exhaustive widening", "[precision][fp16]")
{
    uint32_t mismatches = 0u;
    uint32_t firstMismatch = 0u;
    for(uint32_t raw = 0u; raw <= 0xffffu; ++raw)
    {
        auto const bits = uint16_t(raw);
        float const value = Float16::fromBits(bits).toFloat();

        uint32_t expectedBits;
        if(isNaNBits(bits))
        {
            expectedBits = (uint32_t(bits & 0x8000u) << 16u) | 0x7f80'0000u | (uint32_t(bits & mantissaMask) << 13u);
        }
        else if((bits & 0x7c00u) == 0x7c00u)
        {
            expectedBits = (uint32_t(bits & 0x8000u) << 16u) | 0x7f80'0000u;
        }
        else
        {
            expectedBits = std::bit_cast<uint32_t>(narrowFloatTest::referenceWiden(bits, narrowFloatTest::fp16Format));
        }

        if(std::bit_cast<uint32_t>(value) != expectedBits)
        {
            if(mismatches == 0u)
                firstMismatch = raw;
            ++mismatches;
        }
    }
    INFO("first mismatching bit pattern 0x" << std::hex << firstMismatch);
    CHECK(mismatches == 0u);
}

TEST_CASE("fp16 exhaustive round trip", "[precision][fp16]")
{
    uint32_t mismatches = 0u;
    uint32_t firstMismatch = 0u;
    for(uint32_t raw = 0u; raw <= 0xffffu; ++raw)
    {
        auto const bits = uint16_t(raw);
        uint16_t const roundTrip = Float16::fromFloat(Float16::fromBits(bits).toFloat()).bits();
        // a signaling NaN is quieted by the narrowing conversion, all other patterns are reproduced
        uint16_t const expectedBits = isSignalingNaNBits(bits) ? uint16_t(bits | 0x200u) : bits;
        if(roundTrip != expectedBits)
        {
            if(mismatches == 0u)
                firstMismatch = raw;
            ++mismatches;
        }
    }
    INFO("first mismatching bit pattern 0x" << std::hex << firstMismatch);
    CHECK(mismatches == 0u);
}

TEST_CASE("fp16 exhaustive narrowing rounding", "[precision][fp16]")
{
    auto const& format = narrowFloatTest::fp16Format;
    uint32_t mismatches = 0u;
    uint32_t firstMismatch = 0u;
    auto checkValue = [&](float const value, uint16_t const expectedBits, uint32_t const id)
    {
        uint16_t const got = Float16::fromFloat(value).bits();
        if(got != expectedBits)
        {
            if(mismatches == 0u)
                firstMismatch = id;
            ++mismatches;
        }
    };

    // all adjacent positive pairs up to the largest finite value, the sign is tested via the negated values
    for(uint32_t raw = 0u; raw < 0x7bffu; ++raw)
    {
        auto const lowerBits = uint16_t(raw);
        auto const upperBits = uint16_t(raw + 1u);
        float const lower = Float16::fromBits(lowerBits).toFloat();
        float const upper = Float16::fromBits(upperBits).toFloat();

        checkValue(lower, lowerBits, raw);
        checkValue(-lower, uint16_t(lowerBits | 0x8000u), raw);

        double const middle = (double(lower) + double(upper)) / 2.0;
        float const middleFloat = float(middle);
        if(double(middleFloat) != middle)
            continue;

        checkValue(middleFloat, narrowFloatTest::referenceNarrow(middleFloat, format), raw);
        checkValue(-middleFloat, narrowFloatTest::referenceNarrow(-middleFloat, format), raw);

        float const belowMiddle = std::nextafter(middleFloat, lower);
        float const aboveMiddle = std::nextafter(middleFloat, upper);
        checkValue(belowMiddle, narrowFloatTest::referenceNarrow(belowMiddle, format), raw);
        checkValue(aboveMiddle, narrowFloatTest::referenceNarrow(aboveMiddle, format), raw);
        checkValue(-belowMiddle, narrowFloatTest::referenceNarrow(-belowMiddle, format), raw);
        checkValue(-aboveMiddle, narrowFloatTest::referenceNarrow(-aboveMiddle, format), raw);
    }
    INFO("first mismatching pair 0x" << std::hex << firstMismatch);
    CHECK(mismatches == 0u);
}

TEST_CASE("fp16 special values", "[precision][fp16]")
{
    // signed zero
    CHECK(Float16::fromFloat(0.0f).bits() == 0x0000u);
    CHECK(Float16::fromFloat(-0.0f).bits() == 0x8000u);
    CHECK(std::bit_cast<uint32_t>(Float16::fromBits(0x0000u).toFloat()) == 0x0000'0000u);
    CHECK(std::bit_cast<uint32_t>(Float16::fromBits(0x8000u).toFloat()) == 0x8000'0000u);

    // ties to even in both directions
    float const onePlusHalfUlp = 1.0f + std::ldexp(1.0f, -11); // tie between 0x3c00 and 0x3c01
    float const onePlusOneAndHalfUlp = 1.0f + std::ldexp(1.0f, -10) + std::ldexp(1.0f, -11); // tie between 0x3c01
                                                                                             // and 0x3c02
    CHECK(Float16::fromFloat(onePlusHalfUlp).bits() == 0x3c00u);
    CHECK(Float16::fromFloat(onePlusOneAndHalfUlp).bits() == 0x3c02u);
    CHECK(Float16::fromFloat(-onePlusHalfUlp).bits() == 0xbc00u);
    CHECK(Float16::fromFloat(-onePlusOneAndHalfUlp).bits() == 0xbc02u);

    // smallest subnormal and its ties
    float const minSubnormal = std::ldexp(1.0f, -24);
    CHECK(Float16::fromFloat(minSubnormal).bits() == 0x0001u);
    CHECK(Float16::fromFloat(std::ldexp(1.0f, -25)).bits() == 0x0000u); // tie to even zero
    CHECK(Float16::fromFloat(std::ldexp(3.0f, -25)).bits() == 0x0002u); // tie to even two
    CHECK(Float16::fromFloat(-std::ldexp(1.0f, -25)).bits() == 0x8000u);
    CHECK(Float16::fromFloat(std::nextafter(std::ldexp(1.0f, -25), 1.0f)).bits() == 0x0001u);

    // subnormal to normal boundary, largest subnormal is odd so the tie goes to the smallest normal
    float const largestSubnormal = std::ldexp(1023.0f, -24);
    float const boundaryTie = std::ldexp(1023.5f, -24);
    CHECK(Float16::fromBits(0x03ffu).toFloat() == largestSubnormal);
    CHECK(Float16::fromFloat(largestSubnormal).bits() == 0x03ffu);
    CHECK(Float16::fromFloat(boundaryTie).bits() == 0x0400u);
    CHECK(Float16::fromFloat(std::nextafter(boundaryTie, 0.0f)).bits() == 0x03ffu);

    // smallest and largest normal
    CHECK(Float16::fromBits(0x0400u).toFloat() == std::ldexp(1.0f, -14));
    CHECK(Float16::fromFloat(std::ldexp(1.0f, -14)).bits() == 0x0400u);
    CHECK(Float16::fromFloat(65504.0f).bits() == 0x7bffu);
    CHECK(Float16::fromBits(0x7bffu).toFloat() == 65504.0f);

    // overflow to infinity
    CHECK(Float16::fromFloat(65520.0f).bits() == 0x7c00u);
    CHECK(Float16::fromFloat(std::nextafter(65520.0f, 0.0f)).bits() == 0x7bffu);
    CHECK(Float16::fromFloat(-65520.0f).bits() == 0xfc00u);
    CHECK(Float16::fromFloat(std::numeric_limits<float>::max()).bits() == 0x7c00u);

    // infinity
    CHECK(Float16::fromFloat(std::numeric_limits<float>::infinity()).bits() == 0x7c00u);
    CHECK(Float16::fromFloat(-std::numeric_limits<float>::infinity()).bits() == 0xfc00u);
    CHECK(std::bit_cast<uint32_t>(Float16::fromBits(0x7c00u).toFloat()) == 0x7f80'0000u);
    CHECK(std::bit_cast<uint32_t>(Float16::fromBits(0xfc00u).toFloat()) == 0xff80'0000u);

    // NaN: the sign is kept, the quiet bit is forced and the most significant payload bits survive
    float const payloadNaN = std::bit_cast<float>(uint32_t{0x7fc1'2345u});
    CHECK(Float16::fromFloat(payloadNaN).bits() == expectedNaNBits(payloadNaN));
    float const signalingNaN = std::bit_cast<float>(uint32_t{0x7f80'0001u});
    CHECK(Float16::fromFloat(signalingNaN).bits() == 0x7e00u);
    CHECK(std::isnan(Float16::fromFloat(payloadNaN).toFloat()));
    CHECK(std::bit_cast<uint32_t>(Float16::fromFloat(payloadNaN).toFloat()) == 0x7fc1'2000u);
    CHECK(std::bit_cast<uint32_t>(Float16::fromBits(0xfe01u).toFloat()) == 0xffc0'2000u);

    // quiet NaN payloads survive a full round trip
    for(uint32_t raw = 0x7c01u; raw <= 0x7fffu; ++raw)
    {
        auto const bits = uint16_t(raw);
        if(isSignalingNaNBits(bits))
            continue;
        CHECK(Float16::fromFloat(Float16::fromBits(bits).toFloat()).bits() == bits);
        CHECK(
            Float16::fromFloat(Float16::fromBits(uint16_t(bits | 0x8000u)).toFloat()).bits()
            == uint16_t(bits | 0x8000u));
    }
}
