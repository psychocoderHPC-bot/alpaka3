/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <bit>
#include <cstdint>

/** @file
 *
 * 16 bit floating point storage types.
 *
 * Both types are storage and transport formats. Convert them to `float` for arithmetic and convert the result back
 * with `fromFloat()`. Narrowing results from `fromFloat()` use round to nearest, ties to even. The conversion is
 * implemented with integer arithmetic and must not depend on backend specific instructions.
 */

namespace alpaka
{
    namespace internal
    {
        /** Round an unsigned integer to the right by @p shift bits using round to nearest, ties to even.
         *
         * @param significand value to round
         * @param shift number of bits which are removed
         * @return rounded value
         */
        ALPAKA_FN_HOST_ACC constexpr uint32_t roundShiftRight(uint32_t const significand, uint32_t const shift)
        {
            if(shift == 0u)
                return significand;
            // the value is smaller than half of the smallest representable value
            if(shift > 24u)
                return 0u;
            uint32_t const truncated = significand >> shift;
            uint32_t const remainder = significand & ((uint32_t{1u} << shift) - 1u);
            uint32_t const halfway = uint32_t{1u} << (shift - 1u);
            bool const roundUp = remainder > halfway || (remainder == halfway && (truncated & 1u) != 0u);
            return truncated + (roundUp ? 1u : 0u);
        }

        /** Encode a binary32 value into a 16 bit float with @p T_numMantissaBits mantissa and @p T_numExponentBits
         * exponent bits.
         *
         * The sign is at bit 15 for all supported layouts. Subnormals, overflow, infinity and NaN are handled
         * explicitly. The returned mantissa payload of a NaN keeps the most significant payload bits of the input and
         * forces the quiet bit.
         *
         * @param floatBits bit pattern of the binary32 value
         * @return bit pattern of the 16 bit value
         */
        template<uint32_t T_numMantissaBits, uint32_t T_numExponentBits>
        ALPAKA_FN_HOST_ACC constexpr uint16_t float32ToFloat16(uint32_t const floatBits)
        {
            static_assert(T_numMantissaBits + T_numExponentBits + 1u == 16u);
            constexpr uint32_t mantissaMask = (uint32_t{1u} << T_numMantissaBits) - 1u;
            constexpr uint32_t maxExponentField = (uint32_t{1u} << T_numExponentBits) - 1u;
            constexpr uint32_t exponentBias = maxExponentField >> 1u;
            constexpr uint32_t quietBit = uint32_t{1u} << (T_numMantissaBits - 1u);

            uint32_t const signBit = (floatBits >> 31u) << 15u;
            uint32_t const exponent = (floatBits >> 23u) & 0xffu;
            uint32_t const mantissa = floatBits & 0x7f'ffffu;

            if(exponent == 0xffu)
            {
                uint32_t const infinityOrNan = maxExponentField << T_numMantissaBits;
                if(mantissa == 0u)
                    return uint16_t(signBit | infinityOrNan);
                uint32_t const payload = (mantissa >> (23u - T_numMantissaBits)) | quietBit;
                return uint16_t(signBit | infinityOrNan | payload);
            }

            /* value = significand * 2^(exponent - 150) for normal values and
             * value = significand * 2^-149 for subnormal values, the hidden bit is only present for normal values
             */
            uint32_t const significand = exponent == 0u ? mantissa : (mantissa | 0x80'0000u);
            int32_t const narrowExponent = int32_t(exponent) - int32_t(127u - exponentBias);
            if(narrowExponent >= 1)
            {
                if(narrowExponent >= int32_t(maxExponentField))
                    return uint16_t(signBit | (maxExponentField << T_numMantissaBits));
                uint32_t rounded = roundShiftRight(significand, 23u - T_numMantissaBits);
                // the significand can overflow into the exponent field
                if(rounded == (uint32_t{1u} << (T_numMantissaBits + 1u)))
                {
                    rounded >>= 1u;
                    if(narrowExponent + 1 >= int32_t(maxExponentField))
                        return uint16_t(signBit | (maxExponentField << T_numMantissaBits));
                    return uint16_t(signBit | ((narrowExponent + 1) << T_numMantissaBits) | (rounded & mantissaMask));
                }
                return uint16_t(signBit | (narrowExponent << T_numMantissaBits) | (rounded & mantissaMask));
            }

            /* Subnormal result or zero. The value is expressed as multiples of the smallest subnormal
             * 2^(1 - exponentBias - T_numMantissaBits). A rounding carry lands in the exponent field and yields the
             * smallest normal value.
             */
            int32_t const shift = exponent == 0u
                                      ? int32_t(150u - exponentBias - T_numMantissaBits)
                                      : int32_t(151u - exponentBias - T_numMantissaBits) - int32_t(exponent);
            uint32_t const rounded = roundShiftRight(significand, uint32_t(shift));
            return uint16_t(signBit | (rounded & 0x7fffu));
        }

        /** Decode a 16 bit float with @p T_numMantissaBits mantissa and @p T_numExponentBits exponent bits into
         * binary32.
         *
         * The conversion is exact for every finite value. A NaN keeps the sign and the payload in the most significant
         * mantissa bits.
         *
         * @param bits bit pattern of the 16 bit value
         * @return bit pattern of the binary32 value
         */
        template<uint32_t T_numMantissaBits, uint32_t T_numExponentBits>
        ALPAKA_FN_HOST_ACC constexpr uint32_t float16ToFloat32(uint16_t const bits)
        {
            static_assert(T_numMantissaBits + T_numExponentBits + 1u == 16u);
            constexpr uint32_t mantissaMask = (uint32_t{1u} << T_numMantissaBits) - 1u;
            constexpr uint32_t maxExponentField = (uint32_t{1u} << T_numExponentBits) - 1u;
            constexpr uint32_t exponentBias = maxExponentField >> 1u;

            uint32_t const signBit = uint32_t(bits & 0x8000u) << 16u;
            uint32_t const exponent = (uint32_t(bits) >> T_numMantissaBits) & maxExponentField;
            uint32_t const mantissa = uint32_t(bits) & mantissaMask;

            if(exponent == maxExponentField)
                return signBit | 0x7f80'0000u | (mantissa << (23u - T_numMantissaBits));
            if constexpr(exponentBias == 127u)
            {
                // exponent range and subnormal range are identical to binary32, the value is a prefix of binary32
                return signBit | (exponent << 23u) | (mantissa << (23u - T_numMantissaBits));
            }
            else
            {
                if(exponent == 0u)
                {
                    if(mantissa == 0u)
                        return signBit;
                    // normalize the subnormal value, all subnormals are in the range of binary32 normal values
                    uint32_t msb = T_numMantissaBits - 1u;
                    while(((mantissa >> msb) & 1u) == 0u)
                        --msb;
                    uint32_t const exponentField = msb + 1u - exponentBias - T_numMantissaBits + 127u;
                    return signBit | (exponentField << 23u) | ((mantissa << (23u - msb)) & 0x7f'ffffu);
                }
                return signBit | ((exponent - exponentBias + 127u) << 23u) | (mantissa << (23u - T_numMantissaBits));
            }
        }
    } // namespace internal

    /** IEEE 754 binary16 storage type.
     *
     * Use @ref toFloat() for arithmetic and @ref fromFloat() to convert the result back.
     */
    struct alignas(2) Float16
    {
        constexpr Float16() = default;

        /** Create a value from a raw bit pattern.
         *
         * @param bits the 16 bit pattern, no encoding is altered
         */
        ALPAKA_FN_HOST_ACC static constexpr Float16 fromBits(uint16_t const bits)
        {
            return Float16{bits};
        }

        /** Convert a binary32 value to binary16.
         *
         * Subnormals are preserved, values which overflow the binary16 range become infinity. Rounding is round to
         * nearest, ties to even.
         *
         * @param value binary32 value
         */
        ALPAKA_FN_HOST_ACC static constexpr Float16 fromFloat(float const value)
        {
            return Float16{internal::float32ToFloat16<10u, 5u>(std::bit_cast<uint32_t>(value))};
        }

        /** Convert to binary32.
         *
         * The conversion is exact. A NaN keeps the sign and the most significant payload bits.
         */
        ALPAKA_FN_HOST_ACC constexpr float toFloat() const
        {
            return std::bit_cast<float>(internal::float16ToFloat32<10u, 5u>(m_bits));
        }

        ALPAKA_FN_HOST_ACC constexpr explicit operator float() const
        {
            return toFloat();
        }

        /** Raw bit pattern of the value. */
        ALPAKA_FN_HOST_ACC constexpr uint16_t bits() const
        {
            return m_bits;
        }

    private:
        constexpr explicit Float16(uint16_t const bits) : m_bits{bits}
        {
        }

        uint16_t m_bits{0u};
    };

    /** bfloat16 storage type.
     *
     * The type shares the exponent range of binary32 and keeps the most significant mantissa bits. Use
     * @ref toFloat() for arithmetic and @ref fromFloat() to convert the result back.
     */
    struct alignas(2) BFloat16
    {
        constexpr BFloat16() = default;

        /** Create a value from a raw bit pattern.
         *
         * @param bits the 16 bit pattern, no encoding is altered
         */
        ALPAKA_FN_HOST_ACC static constexpr BFloat16 fromBits(uint16_t const bits)
        {
            return BFloat16{bits};
        }

        /** Convert a binary32 value to bfloat16.
         *
         * The conversion rounds the binary32 value, it does not truncate the mantissa. Subnormals are preserved,
         * rounding is round to nearest, ties to even.
         *
         * @param value binary32 value
         */
        ALPAKA_FN_HOST_ACC static constexpr BFloat16 fromFloat(float const value)
        {
            return BFloat16{internal::float32ToFloat16<7u, 8u>(std::bit_cast<uint32_t>(value))};
        }

        /** Convert to binary32.
         *
         * The conversion is exact. A NaN keeps the sign and the most significant payload bits.
         */
        ALPAKA_FN_HOST_ACC constexpr float toFloat() const
        {
            return std::bit_cast<float>(internal::float16ToFloat32<7u, 8u>(m_bits));
        }

        ALPAKA_FN_HOST_ACC constexpr explicit operator float() const
        {
            return toFloat();
        }

        /** Raw bit pattern of the value. */
        ALPAKA_FN_HOST_ACC constexpr uint16_t bits() const
        {
            return m_bits;
        }

    private:
        constexpr explicit BFloat16(uint16_t const bits) : m_bits{bits}
        {
        }

        uint16_t m_bits{0u};
    };
} // namespace alpaka
