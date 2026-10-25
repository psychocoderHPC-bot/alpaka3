/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cmath>
#include <cstdint>

/** Scalar reference implementation for the 16 bit storage type conversions.
 *
 * The reference is independent of the bit manipulation used by `alpaka::Float16` and `alpaka::BFloat16`: it decodes
 * values with `std::ldexp()` and rounds with `std::nearbyint()`.
 */
namespace narrowFloatTest
{
    struct Format
    {
        int mantissaBits;
        int exponentBits;
    };

    inline constexpr Format fp16Format{10, 5};
    inline constexpr Format bf16Format{7, 8};

    inline uint32_t maxExponentField(Format const& format)
    {
        return (uint32_t{1u} << format.exponentBits) - 1u;
    }

    inline uint32_t mantissaMask(Format const& format)
    {
        return (uint32_t{1u} << format.mantissaBits) - 1u;
    }

    inline uint32_t exponentBias(Format const& format)
    {
        return maxExponentField(format) >> 1u;
    }

    /** Widen a 16 bit value with `std::ldexp`.
     *
     * The result is a value, not a bit pattern. NaN and infinity must be handled by the caller.
     */
    inline float referenceWiden(uint16_t const bits, Format const& format)
    {
        bool const negative = (bits & 0x8000u) != 0u;
        uint32_t const exponent = (uint32_t(bits) >> format.mantissaBits) & maxExponentField(format);
        uint32_t const mantissa = uint32_t(bits) & mantissaMask(format);
        float value;
        if(exponent == 0u)
            value = std::ldexp(float(mantissa), 1 - int(exponentBias(format)) - format.mantissaBits);
        else
            value = std::ldexp(
                float(mantissa | (uint32_t{1u} << format.mantissaBits)),
                int(exponent) - int(exponentBias(format)) - format.mantissaBits);
        return negative ? -value : value;
    }

    /** Narrow a finite binary32 value with round to nearest, ties to even. */
    inline uint16_t referenceNarrow(float const value, Format const& format)
    {
        bool const negative = std::signbit(value);
        uint16_t const sign = negative ? uint16_t{0x8000u} : uint16_t{0u};
        double const magnitude = std::fabs(double(value));
        if(magnitude == 0.0)
            return sign;
        int exponentOfValue = 0;
        // value = fraction * 2^exponentOfValue, fraction in [0.5, 1)
        double const fraction = std::frexp(magnitude, &exponentOfValue);
        int const exponentField = exponentOfValue - 1 + int(exponentBias(format));
        if(exponentField >= 1)
        {
            if(exponentField >= int(maxExponentField(format)))
                return uint16_t(sign | (maxExponentField(format) << format.mantissaBits));
            double rounded = std::nearbyint(std::ldexp(fraction, format.mantissaBits + 1));
            int resultExponent = exponentField;
            if(rounded == double(uint32_t{1u} << (format.mantissaBits + 1u)))
            {
                rounded /= 2.0;
                ++resultExponent;
                if(resultExponent >= int(maxExponentField(format)))
                    return uint16_t(sign | (maxExponentField(format) << format.mantissaBits));
            }
            return uint16_t(
                sign | (uint32_t(resultExponent) << format.mantissaBits) | (uint32_t(rounded) & mantissaMask(format)));
        }
        // subnormal result or zero, a carry reaches the smallest normal value
        double const rounded = std::nearbyint(std::ldexp(fraction, exponentField + format.mantissaBits));
        return uint16_t(sign | uint16_t(rounded));
    }
} // namespace narrowFloatTest
