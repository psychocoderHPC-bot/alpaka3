/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Helpers shared by the relu benchmark: portable fp16 bit conversion, double precision host reference
 *        implementations and verification.
 *
 * The benchmark is a faithful adaptation of the HeCBench `relu` CUDA/OpenMP programs which are derived from the
 * TensorFlow ReluGrad and Relu kernels (Apache-2.0). Only the alpaka implementation lives in this repository.
 */

#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include <bit>
#include <cstdlib>

namespace relu
{
    /** Convert an IEEE-754 binary32 value into a binary16 bit pattern.
     *
     * The rounding is round-to-nearest, ties-to-even which is identical to `_Float16` / `__float2half_rn` for the
     * finite range. Subnormals, overflow to infinity and NaN payloads are handled explicitly.
     */
    ALPAKA_FN_HOST_ACC constexpr uint16_t floatToHalf(float value)
    {
        uint32_t const bits = std::bit_cast<uint32_t>(value);
        uint32_t const sign = (bits >> 16) & 0x8000u;
        uint32_t const mant = bits & 0x007FFFFFu;
        uint32_t const rawExp = (bits >> 23) & 0xFFu;
        int exp = static_cast<int>(rawExp) - 127 + 15;

        if(rawExp == 0xFFu)
        {
            // Inf or NaN. Canonicalize NaN to a quiet value without payload.
            if(mant == 0u)
                return static_cast<uint16_t>(sign | 0x7C00u);
            return static_cast<uint16_t>(sign | 0x7E00u);
        }
        if(exp <= 0)
        {
            // subnormal or zero
            if(exp < -10)
                return static_cast<uint16_t>(sign);
            uint32_t m = mant | 0x00800000u;
            unsigned const shift = static_cast<unsigned>(14 - exp);
            uint32_t half = m >> shift;
            uint32_t const rem = m & ((1u << shift) - 1u);
            uint32_t const halfBit = 1u << (shift - 1u);
            if(rem > halfBit || (rem == halfBit && (half & 1u)))
                ++half;
            return static_cast<uint16_t>(sign | half);
        }
        if(exp >= 31)
        {
            // finite value that overflows binary16
            return static_cast<uint16_t>(sign | 0x7C00u);
        }
        uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
        uint32_t const rem = mant & 0x1FFFu;
        if(rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
            ++half;
        return static_cast<uint16_t>(half);
    }

    /** Convert a binary16 bit pattern into a binary32 value. */
    ALPAKA_FN_HOST_ACC constexpr float halfToFloat(uint16_t h)
    {
        uint32_t const sign = static_cast<uint32_t>(h & 0x8000u) << 16;
        uint32_t const exp = (h >> 10) & 0x1Fu;
        uint32_t const mant = h & 0x3FFu;
        uint32_t f;
        if(exp == 0u)
        {
            if(mant == 0u)
            {
                f = sign;
            }
            else
            {
                // subnormal
                int e = -1;
                uint32_t m = mant;
                do
                {
                    ++e;
                    m <<= 1;
                } while((m & 0x400u) == 0u);
                m &= 0x3FFu;
                f = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (m << 13);
            }
        }
        else if(exp == 31u)
        {
            f = sign | 0x7F800000u | (mant << 13);
        }
        else
        {
            f = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
        }
        return std::bit_cast<float>(f);
    }

    inline bool isNan(float value)
    {
        return isnan(value) != 0;
    }

    inline bool isInf(float value)
    {
        return isinf(value) != 0;
    }

    /** Packed-integer ReLU reference: `max(value, 0)` applied independently to the four signed bytes of every
     * 32 bit word. Matches HeCBench `Relu_reference` bit for bit.
     */
    inline int32_t reluReference(int32_t input)
    {
        auto const c0 = static_cast<int8_t>(input & 0xFF);
        auto const c1 = static_cast<int8_t>((input >> 8) & 0xFF);
        auto const c2 = static_cast<int8_t>((input >> 16) & 0xFF);
        auto const c3 = static_cast<int8_t>((input >> 24) & 0xFF);
        uint32_t const r0 = static_cast<uint32_t>(c0 > 0 ? c0 : 0);
        uint32_t const r1 = static_cast<uint32_t>(c1 > 0 ? c1 : 0);
        uint32_t const r2 = static_cast<uint32_t>(c2 > 0 ? c2 : 0);
        uint32_t const r3 = static_cast<uint32_t>(c3 > 0 ? c3 : 0);
        return static_cast<int32_t>((r3 << 24) | (r2 << 16) | (r1 << 8) | r0);
    }

    /** fp16 ReluGrad reference computed in double precision.
     *
     * Inputs and outputs are binary16 bit patterns. The operation is `(feature > 0) ? gradient : 0`. Because the
     * result of this benchmark (feature in (-1, 1), gradient == 1) is exactly representable in binary16, the double
     * precision intermediate is converted back losslessly.
     */
    inline uint16_t reluGradReference(uint16_t feature, uint16_t gradient)
    {
        double const f = static_cast<double>(halfToFloat(feature));
        double const g = static_cast<double>(halfToFloat(gradient));
        double const result = (f > 0.0) ? g : 0.0;
        return floatToHalf(static_cast<float>(result));
    }

    /** Verify the packed-integer ReLU result against the host reference. */
    template<typename T_Out, typename T_In>
    int verifyInt(int count, T_Out const& out, T_In const& in)
    {
        int error = EXIT_SUCCESS;
        for(int i = 0; i < count; ++i)
        {
            int32_t const expected = reluReference(static_cast<int32_t>(in[i]));
            if(static_cast<int32_t>(out[i]) != expected)
            {
                printf(
                    "error at relu[%d]: got %d expected %d\n",
                    i,
                    static_cast<int>(out[i]),
                    static_cast<int>(expected));
                error = EXIT_FAILURE;
                break;
            }
        }
        printf("Relu (packed int8x4) verification %s\n", error ? "FAILED" : "PASSED");
        return error;
    }

    /** Verify the fp16 ReluGrad result against the double precision host reference.
     *
     * A tolerance of 1e-3 is used, matching the HeCBench reference. Inputs are fp16 values in (-1, 1) and the
     * gradient is 1, therefore the expected result is always exactly 0 or 1 and the tolerance is not reached in
     * practice. NaN/Inf in the produced data is treated as a failure.
     */
    template<typename T_OutFp16, typename T_Feature, typename T_Gradient>
    int verifyReluGrad(int count, T_OutFp16 const& out, T_Feature const& feature, T_Gradient const& gradient)
    {
        int error = EXIT_SUCCESS;
        float const tolerance = 1e-3f;
        for(int i = 0; i < count; ++i)
        {
            float const got = halfToFloat(static_cast<uint16_t>(out[i]));
            if(isNan(got) || isInf(got))
            {
                printf("non-finite result at backprop[%d]: %f\n", i, static_cast<double>(got));
                error = EXIT_FAILURE;
                break;
            }
            uint16_t const expectedBits = reluGradReference(
                static_cast<uint16_t>(feature[i]),
                static_cast<uint16_t>(gradient[i]));
            float const expected = halfToFloat(expectedBits);
            if(fabsf(got - expected) > tolerance)
            {
                printf(
                    "error at backprop[%d]: got %f expected %f\n",
                    i,
                    static_cast<double>(got),
                    static_cast<double>(expected));
                error = EXIT_FAILURE;
                break;
            }
        }
        printf("ReluGrad (fp16) verification %s\n", error ? "FAILED" : "PASSED");
        return error;
    }

} // namespace relu
