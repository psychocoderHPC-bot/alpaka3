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
    constexpr uint16_t mantissaMask = 0x7fu;

    uint16_t expectedNaNBits(float const value)
    {
        uint32_t const floatBits = std::bit_cast<uint32_t>(value);
        uint32_t const sign = (floatBits >> 31u) << 15u;
        uint32_t const payload = ((floatBits & 0x7f'ffffu) >> 16u) | 0x40u;
        return uint16_t(sign | 0x7f80u | payload);
    }

    bool isNaNBits(uint16_t const bits)
    {
        return (bits & 0x7f80u) == 0x7f80u && (bits & mantissaMask) != 0u;
    }

    bool isSignalingNaNBits(uint16_t const bits)
    {
        return isNaNBits(bits) && (bits & 0x40u) == 0u;
    }
} // namespace

TEST_CASE("bf16 type properties", "[precision][bf16]")
{
    STATIC_REQUIRE(sizeof(BFloat16) == 2u);
    STATIC_REQUIRE(alignof(BFloat16) == 2u);
    STATIC_REQUIRE(std::is_trivially_copyable_v<BFloat16>);
    STATIC_REQUIRE(std::is_standard_layout_v<BFloat16>);
    STATIC_REQUIRE(std::is_same_v<GetValueType_t<BFloat16>, BFloat16>);
    STATIC_REQUIRE(std::is_same_v<GetValueType_t<BFloat16 const>, BFloat16 const>);
    STATIC_REQUIRE(isKernelArgumentTriviallyCopyable_v<BFloat16>);

    STATIC_REQUIRE(BFloat16::fromFloat(1.0f).bits() == 0x3f80u);
    STATIC_REQUIRE(BFloat16::fromFloat(1.5f).bits() == 0x3fc0u);
    STATIC_REQUIRE(BFloat16::fromFloat(-0.0f).bits() == 0x8000u);
    STATIC_REQUIRE(BFloat16::fromBits(0x3f80u).toFloat() == 1.0f);
    STATIC_REQUIRE(static_cast<float>(BFloat16::fromFloat(2.0f)) == 2.0f);
}

TEST_CASE("bf16 exhaustive widening", "[precision][bf16]")
{
    uint32_t mismatches = 0u;
    uint32_t firstMismatch = 0u;
    for(uint32_t raw = 0u; raw <= 0xffffu; ++raw)
    {
        auto const bits = uint16_t(raw);
        float const value = BFloat16::fromBits(bits).toFloat();

        uint32_t expectedBits;
        if(isNaNBits(bits))
        {
            expectedBits = (uint32_t(bits & 0x8000u) << 16u) | 0x7f80'0000u | (uint32_t(bits & mantissaMask) << 16u);
        }
        else if((bits & 0x7f80u) == 0x7f80u)
        {
            expectedBits = (uint32_t(bits & 0x8000u) << 16u) | 0x7f80'0000u;
        }
        else
        {
            expectedBits = std::bit_cast<uint32_t>(narrowFloatTest::referenceWiden(bits, narrowFloatTest::bf16Format));
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

TEST_CASE("bf16 exhaustive round trip", "[precision][bf16]")
{
    uint32_t mismatches = 0u;
    uint32_t firstMismatch = 0u;
    for(uint32_t raw = 0u; raw <= 0xffffu; ++raw)
    {
        auto const bits = uint16_t(raw);
        uint16_t const roundTrip = BFloat16::fromFloat(BFloat16::fromBits(bits).toFloat()).bits();
        uint16_t const expectedBits = isSignalingNaNBits(bits) ? uint16_t(bits | 0x40u) : bits;
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

TEST_CASE("bf16 exhaustive narrowing rounding", "[precision][bf16]")
{
    auto const& format = narrowFloatTest::bf16Format;
    uint32_t mismatches = 0u;
    uint32_t firstMismatch = 0u;
    auto checkValue = [&](float const value, uint16_t const expectedBits, uint32_t const id)
    {
        uint16_t const got = BFloat16::fromFloat(value).bits();
        if(got != expectedBits)
        {
            if(mismatches == 0u)
                firstMismatch = id;
            ++mismatches;
        }
    };

    for(uint32_t raw = 0u; raw < 0x7f7fu; ++raw)
    {
        auto const lowerBits = uint16_t(raw);
        auto const upperBits = uint16_t(raw + 1u);
        float const lower = BFloat16::fromBits(lowerBits).toFloat();
        float const upper = BFloat16::fromBits(upperBits).toFloat();

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

TEST_CASE("bf16 special values", "[precision][bf16]")
{
    // signed zero
    CHECK(BFloat16::fromFloat(0.0f).bits() == 0x0000u);
    CHECK(BFloat16::fromFloat(-0.0f).bits() == 0x8000u);
    CHECK(std::bit_cast<uint32_t>(BFloat16::fromBits(0x0000u).toFloat()) == 0x0000'0000u);
    CHECK(std::bit_cast<uint32_t>(BFloat16::fromBits(0x8000u).toFloat()) == 0x8000'0000u);

    // ties to even in both directions
    float const onePlusHalfUlp = 1.0f + std::ldexp(1.0f, -8);
    float const onePlusOneAndHalfUlp = 1.0f + std::ldexp(1.0f, -7) + std::ldexp(1.0f, -8);
    CHECK(BFloat16::fromFloat(onePlusHalfUlp).bits() == 0x3f80u);
    CHECK(BFloat16::fromFloat(onePlusOneAndHalfUlp).bits() == 0x3f82u);
    CHECK(BFloat16::fromFloat(-onePlusHalfUlp).bits() == 0xbf80u);
    CHECK(BFloat16::fromFloat(-onePlusOneAndHalfUlp).bits() == 0xbf82u);

    // the conversion rounds, it does not truncate the mantissa
    float const tieToEven = std::bit_cast<float>(uint32_t{0x3f81'8000u});
    float const roundUp = std::bit_cast<float>(uint32_t{0x3f81'c000u});
    float const roundDown = std::bit_cast<float>(uint32_t{0x3f81'4000u});
    CHECK(BFloat16::fromFloat(tieToEven).bits() == 0x3f82u);
    CHECK(BFloat16::fromFloat(roundUp).bits() == 0x3f82u);
    CHECK(BFloat16::fromFloat(roundDown).bits() == 0x3f81u);
    CHECK(BFloat16::fromBits(0x3f80u).toFloat() == 1.0f);

    // subnormals share the binary32 exponent range
    float const minSubnormal = std::ldexp(1.0f, -133);
    CHECK(BFloat16::fromBits(0x0001u).toFloat() == minSubnormal);
    CHECK(BFloat16::fromFloat(minSubnormal).bits() == 0x0001u);
    CHECK(BFloat16::fromFloat(std::ldexp(1.0f, -134)).bits() == 0x0000u); // tie to even zero
    CHECK(BFloat16::fromFloat(std::ldexp(3.0f, -134)).bits() == 0x0002u); // tie to even two
    CHECK(BFloat16::fromFloat(std::nextafter(std::ldexp(1.0f, -134), 1.0f)).bits() == 0x0001u);

    float const largestSubnormal = std::ldexp(127.0f, -133);
    float const subnormalBoundaryTie = std::ldexp(127.5f, -133);
    CHECK(BFloat16::fromBits(0x007fu).toFloat() == largestSubnormal);
    CHECK(BFloat16::fromFloat(largestSubnormal).bits() == 0x007fu);
    CHECK(BFloat16::fromFloat(subnormalBoundaryTie).bits() == 0x0080u);
    CHECK(BFloat16::fromFloat(std::nextafter(subnormalBoundaryTie, 0.0f)).bits() == 0x007fu);

    // smallest and largest normal value
    CHECK(BFloat16::fromBits(0x0080u).toFloat() == std::ldexp(1.0f, -126));
    CHECK(BFloat16::fromFloat(std::ldexp(1.0f, -126)).bits() == 0x0080u);
    float const largestNormal = BFloat16::fromBits(0x7f7fu).toFloat();
    CHECK(BFloat16::fromFloat(largestNormal).bits() == 0x7f7fu);

    // overflow to infinity, the tie between the largest finite value and infinity rounds to even (infinity)
    CHECK(BFloat16::fromFloat(std::numeric_limits<float>::max()).bits() == 0x7f80u);
    CHECK(BFloat16::fromFloat(std::bit_cast<float>(uint32_t{0x7f7f'8000u})).bits() == 0x7f80u);
    CHECK(BFloat16::fromFloat(std::bit_cast<float>(uint32_t{0x7f7f'7fffu})).bits() == 0x7f7fu);
    CHECK(BFloat16::fromFloat(-std::numeric_limits<float>::max()).bits() == 0xff80u);

    // infinity
    CHECK(BFloat16::fromFloat(std::numeric_limits<float>::infinity()).bits() == 0x7f80u);
    CHECK(std::bit_cast<uint32_t>(BFloat16::fromBits(0x7f80u).toFloat()) == 0x7f80'0000u);
    CHECK(std::bit_cast<uint32_t>(BFloat16::fromBits(0xff80u).toFloat()) == 0xff80'0000u);

    // NaN: the sign is kept, the quiet bit is forced and the most significant payload bits survive
    float const payloadNaN = std::bit_cast<float>(uint32_t{0x7fc1'2345u});
    CHECK(BFloat16::fromFloat(payloadNaN).bits() == expectedNaNBits(payloadNaN));
    CHECK(BFloat16::fromFloat(payloadNaN).bits() == 0x7fc1u);
    float const signalingNaN = std::bit_cast<float>(uint32_t{0x7f80'0001u});
    CHECK(BFloat16::fromFloat(signalingNaN).bits() == 0x7fc0u);
    CHECK(std::isnan(BFloat16::fromFloat(payloadNaN).toFloat()));
    CHECK(std::bit_cast<uint32_t>(BFloat16::fromFloat(payloadNaN).toFloat()) == 0x7fc1'0000u);
    CHECK(std::bit_cast<uint32_t>(BFloat16::fromBits(0xffc1u).toFloat()) == 0xffc1'0000u);

    // quiet NaN payloads survive a full round trip
    for(uint32_t raw = 0x7f80u; raw <= 0x7fffu; ++raw)
    {
        auto const bits = uint16_t(raw);
        if(isSignalingNaNBits(bits))
            continue;
        CHECK(BFloat16::fromFloat(BFloat16::fromBits(bits).toFloat()).bits() == bits);
        CHECK(
            BFloat16::fromFloat(BFloat16::fromBits(uint16_t(bits | 0x8000u)).toFloat()).bits()
            == uint16_t(bits | 0x8000u));
    }
}

namespace
{
    // Name the active host compiler in the diagnostics below, so a failure points at the toolchain
    // that enabled FTZ/DAZ rather than at the BF16 conversion.
#if defined(__INTEL_LLVM_COMPILER)
    constexpr char const* hostCompilerName = "IntelLLVM (icx/icpx)";
#elif defined(__INTEL_COMPILER)
    constexpr char const* hostCompilerName = "Intel Classic (icc/icpc)";
#elif defined(__clang__)
    constexpr char const* hostCompilerName = "Clang";
#elif defined(__GNUC__)
    constexpr char const* hostCompilerName = "GCC";
#else
    constexpr char const* hostCompilerName = "unknown compiler";
#endif
} // namespace

TEST_CASE("bf16 subnormal multiply-accumulate keeps exact FP32 subnormal bits", "[precision][bf16][fp-contract]")
{
    // This pins the precision contract, not a compiler flag. Intel C++ at -O3 with the default FP
    // model sets MXCSR FTZ and DAZ for the whole process and never restores them; every operand or
    // result in the FP32 subnormal range is then silently replaced by zero. Widened BF16 subnormals
    // live exactly there, so the expected bit patterns below are the contract.
    struct Case
    {
        uint16_t input;
        uint32_t expectedMacBits;
    };

    constexpr Case cases[]{
        {0x0001u, 0x0000'8000u}, // min subnormal 2^-133, times 0.5 is 2^-134
        {0x8001u, 0x8000'8000u}, // negative min subnormal
        {0x007fu, 0x003f'8000u}, // largest BF16 subnormal
        {0x0080u, 0x0040'0000u}, // smallest BF16 normal, the product is an FP32 subnormal
    };

    for(auto const& testCase : cases)
    {
        // volatile keeps the widening and the multiply-accumulate at run time, so the test observes
        // the process FP environment instead of a constant-folded result.
        float volatile operand = BFloat16::fromBits(testCase.input).toFloat();
        float const mac = std::fmaf(static_cast<float>(operand), 0.5f, 0.0f);
        uint32_t const got = std::bit_cast<uint32_t>(mac);

        INFO("BF16 input 0x" << std::hex << testCase.input);
        INFO("expected 0x" << std::hex << testCase.expectedMacBits << ", got 0x" << std::hex << got);
        if(got != testCase.expectedMacBits)
        {
            FAIL(
                "BF16 subnormal widening was flushed to zero by compiler "
                << hostCompilerName
                << ". The host FP environment has FTZ (flush-to-zero) and DAZ (denormals-are-zero) "
                   "enabled: icx/icpx at -O3 with the default -ffp-model write MXCSR bits 15 and 6 "
                   "process-wide. Compile the whole program, including the entry point, with "
                   "-ffp-model=precise, or deliberately opt in to flushed subnormals with "
                   "alpaka_FAST_MATH=ON/alpaka_FTZ=ON and the reduced-precision contract.");
        }
        CHECK(got == testCase.expectedMacBits);
    }
}
