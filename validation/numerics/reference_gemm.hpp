/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

/** @file
 *
 * Independent scalar reference for ``C := alpha * op(A) * op(B) + beta * C``.
 *
 * This header is validation-only. It is not included by the alpaka3 library, it
 * is not part of the public API, and it deliberately shares no code with the
 * GEMM under test: the loops are a naive triple loop without SIMD, blocking,
 * packing or cache blocking. The only shared definitions are the small value
 * conversion helpers at the bottom, which make raw FP16/BF16 bit patterns
 * readable before the common storage types land.
 *
 * Accumulation happens in the caller-selected ``Acc`` type through
 * ``gemm_reference<Acc>()``:
 *
 * * ``Acc = float``  models a strict FP32/FMA accumulation path and is only
 *   used by tests as a candidate stand-in, never as the certified reference.
 * * ``Acc = double`` is the reference for FP32 and for converted FP16/BF16
 *   inputs (the guide requires FP64 reference accumulation for those).
 * * ``Acc = long double`` (through ``gemm_reference_extended()``) is the
 *   bounded reference for FP64 data. It is used only when the platform has a
 *   wider-than-double long double; see TOLERANCE.md for the documented limit.
 */

namespace alpaka::validation::gemm
{
    enum class Layout
    {
        rowMajor,
        colMajor
    };

    /** Whether an operand is used as stored (N) or transposed (T). */
    enum class Op
    {
        none,
        transpose
    };

    /** Non-owning matrix view with a padded leading dimension.
     *
     * ``rows`` and ``cols`` describe the stored matrix. ``ld`` is the leading
     * dimension in elements: the row stride for row-major storage and the
     * column stride for column-major storage. A padded or shifted subview is
     * expressed by a smaller logical extent plus an offset ``data`` pointer.
     * For output views ``op`` must be ``none``.
     */
    template<typename T>
    struct MatrixView
    {
        T* data = nullptr;
        std::size_t rows = 0;
        std::size_t cols = 0;
        std::size_t ld = 0;
        Layout layout = Layout::rowMajor;
        Op op = Op::none;

        [[nodiscard]] constexpr std::size_t logicalRows() const
        {
            return op == Op::none ? rows : cols;
        }

        [[nodiscard]] constexpr std::size_t logicalCols() const
        {
            return op == Op::none ? cols : rows;
        }

        [[nodiscard]] constexpr std::size_t index(std::size_t row, std::size_t col) const
        {
            return layout == Layout::rowMajor ? row * ld + col : col * ld + row;
        }

        [[nodiscard]] constexpr T& stored(std::size_t row, std::size_t col) const
        {
            return data[index(row, col)];
        }

        /** Element ``(i, k)`` of the logical operand. */
        [[nodiscard]] constexpr T& at(std::size_t i, std::size_t k) const
        {
            return op == Op::none ? stored(i, k) : stored(k, i);
        }
    };

    // ---------------------------------------------------------------------
    // Raw FP16/BF16 storage helpers.
    //
    // These decode raw bit patterns exactly. Until the common FP16/BF16 value
    // types are merged, tests pass Fp16Bits/Bf16Bits views to the reference.
    // Once a library type exists it only needs a ``referenceValue`` overload
    // found by argument-dependent lookup; nothing here has to change.
    // ---------------------------------------------------------------------

    struct Fp16Bits
    {
        std::uint16_t bits = 0;
    };

    struct Bf16Bits
    {
        std::uint16_t bits = 0;
    };

    inline double fp16BitsToDouble(std::uint16_t bits) noexcept
    {
        auto const sign = bits >> 15;
        auto const exponent = (bits >> 10) & 0x1fu;
        auto const mantissa = bits & 0x3ffu;
        double value = 0.0;
        if(exponent == 0)
            value = std::ldexp(static_cast<double>(mantissa), -24);
        else if(exponent == 0x1fu)
            value = mantissa == 0 ? std::numeric_limits<double>::infinity() : std::numeric_limits<double>::quiet_NaN();
        else
            value = std::ldexp(static_cast<double>(mantissa | 0x400u), static_cast<int>(exponent) - 25);
        return sign ? -value : value;
    }

    inline double bf16BitsToDouble(std::uint16_t bits) noexcept
    {
        auto const asFloat = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
        return static_cast<double>(asFloat);
    }

    inline std::uint16_t floatToFp16Bits(float value) noexcept
    {
        auto const x = std::bit_cast<std::uint32_t>(value);
        auto const sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
        auto const exponent = (x >> 23) & 0xffu;
        auto const mantissa = x & 0x7fffffu;

        if(exponent == 0xffu)
            return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));

        if(exponent >= 113u && exponent <= 142u)
        {
            auto const significand = mantissa | 0x800000u;
            auto const rounded = significand + (1u << 12) - 1u + ((significand >> 13) & 1u);
            auto e = exponent - 112u + (rounded >> 24);
            auto const outMantissa = (rounded >> 13) & 0x3ffu;
            if(e >= 31u)
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            return static_cast<std::uint16_t>(sign | (e << 10) | outMantissa);
        }

        if(exponent > 142u)
            return static_cast<std::uint16_t>(sign | 0x7c00u);

        // Subnormal FP16 or FP32 zero: round the value in units of 2^-24.
        double const magnitude = std::fabs(static_cast<double>(value));
        double const units = std::nearbyint(std::ldexp(magnitude, 24));
        if(units >= 1024.0)
            return static_cast<std::uint16_t>(sign | 0x0400u);
        return static_cast<std::uint16_t>(sign | static_cast<std::uint16_t>(units));
    }

    inline std::uint16_t floatToBf16Bits(float value) noexcept
    {
        auto const x = std::bit_cast<std::uint32_t>(value);
        if((x & 0x7fffffffu) > 0x7f800000u)
            return static_cast<std::uint16_t>((x >> 16) | 0x0040u);
        auto const roundBit = (x >> 16) & 1u;
        auto const rounded = x + 0x7fffu + roundBit;
        return static_cast<std::uint16_t>(rounded >> 16);
    }

    inline float fp16FromFloat(float value) noexcept
    {
        return static_cast<float>(fp16BitsToDouble(floatToFp16Bits(value)));
    }

    inline float bf16FromFloat(float value) noexcept
    {
        return static_cast<float>(bf16BitsToDouble(floatToBf16Bits(value)));
    }

    /** Exact stored-value read used by the reference.
     *
     * The overload set is intentionally open: a future storage type only adds
     * an overload next to that type and argument-dependent lookup finds it.
     */
    inline double referenceValue(Fp16Bits value) noexcept
    {
        return fp16BitsToDouble(value.bits);
    }

    inline double referenceValue(Bf16Bits value) noexcept
    {
        return bf16BitsToDouble(value.bits);
    }

    template<std::floating_point T>
    constexpr T referenceValue(T value) noexcept
    {
        return value;
    }

    template<std::integral T>
    constexpr auto referenceValue(T value) noexcept
    {
        return value;
    }

    namespace detail
    {
        template<typename T>
        inline void requireStride(MatrixView<T> const& view, char const* name)
        {
            if(view.data == nullptr || view.rows == 0 || view.cols == 0)
                return;
            auto const required = view.layout == Layout::rowMajor ? view.cols : view.rows;
            if(view.ld < required)
            {
                throw std::invalid_argument(
                    std::string("gemm reference: leading dimension of ") + name + " is smaller than its extent");
            }
        }

        template<typename AlphaT, typename BetaT>
        constexpr void requireScalar()
        {
            static_assert(std::is_arithmetic_v<std::remove_cvref_t<AlphaT>>, "alpha must be an arithmetic scalar");
            static_assert(std::is_arithmetic_v<std::remove_cvref_t<BetaT>>, "beta must be an arithmetic scalar");
        }
    } // namespace detail

    /** Naive reference GEMM.
     *
     * Zero-size contract, matching the project guide:
     * * ``M == 0`` or ``N == 0``: no-op, and no operand is read.
     * * ``K == 0`` or ``alpha == 0``: computes ``beta * C`` without reading A or B.
     * * ``beta == 0``: the old C value is not read.
     * * ``beta == 1`` with ``K == 0``: no-op; the stored C bits are preserved.
     *
     * Products are formed in ``Acc`` and accumulated in ascending ``k`` order.
     * The function performs a separate multiply and add; a compiler is free to
     * contract those into an FMA, which is covered by the tolerance model.
     */
    template<typename Acc, typename OutT, typename AT, typename BT, typename AlphaT, typename BetaT>
    void gemm_reference(
        MatrixView<OutT> c,
        MatrixView<AT> a,
        MatrixView<BT> b,
        AlphaT alpha,
        BetaT beta)
    {
        static_assert(std::is_arithmetic_v<Acc>, "the accumulator must be an arithmetic type");
        detail::requireScalar<AlphaT, BetaT>();
        if(c.op != Op::none)
            throw std::invalid_argument("gemm reference: output view must not be transposed");

        std::size_t const m = c.rows;
        std::size_t const n = c.cols;
        std::size_t const k = a.logicalCols();

        if(a.logicalRows() != m || b.logicalCols() != n || b.logicalRows() != k)
            throw std::invalid_argument("gemm reference: inconsistent matrix dimensions");
        detail::requireStride(a, "A");
        detail::requireStride(b, "B");
        detail::requireStride(c, "C");

        if(m == 0 || n == 0)
            return;

        bool const alphaIsZero = alpha == AlphaT(0);
        bool const betaIsZero = beta == BetaT(0);
        bool const betaIsOne = beta == BetaT(1);

        for(std::size_t i = 0; i < m; ++i)
        {
            for(std::size_t j = 0; j < n; ++j)
            {
                if(alphaIsZero || k == 0)
                {
                    if(betaIsZero)
                        c.stored(i, j) = static_cast<OutT>(0);
                    else if(!betaIsOne)
                        c.stored(i, j) = static_cast<OutT>(beta * static_cast<Acc>(c.stored(i, j)));
                    // beta == 1 reuses the stored bits unchanged.
                    continue;
                }

                Acc sum = Acc{0};
                for(std::size_t kk = 0; kk < k; ++kk)
                {
                    auto const av = static_cast<Acc>(referenceValue(a.at(i, kk)));
                    auto const bv = static_cast<Acc>(referenceValue(b.at(kk, j)));
                    sum += av * bv;
                }

                Acc value;
                if(betaIsZero)
                    value = static_cast<Acc>(alpha) * sum;
                else if(betaIsOne)
                    value = static_cast<Acc>(alpha) * sum + static_cast<Acc>(c.stored(i, j));
                else
                    value = static_cast<Acc>(alpha) * sum + static_cast<Acc>(beta) * static_cast<Acc>(c.stored(i, j));
                c.stored(i, j) = static_cast<OutT>(value);
            }
        }
    }

    /** Whether the platform long double carries more precision than double. */
    inline constexpr bool hasExtendedPrecisionReference
        = std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits;

    /** Higher precision reference for FP64 data.
     *
     * Only usable where long double is wider than double (x86-64: 64 mantissa
     * bits). The documented useful range and limits are in TOLERANCE.md.
     */
    template<typename OutT, typename AT, typename BT, typename AlphaT, typename BetaT>
    void gemm_reference_extended(
        MatrixView<OutT> c,
        MatrixView<AT> a,
        MatrixView<BT> b,
        AlphaT alpha,
        BetaT beta)
    {
        static_assert(
            hasExtendedPrecisionReference,
            "gemm_reference_extended needs a long double wider than double; use a multiprecision "
            "reference or restrict the test");
        gemm_reference<long double>(c, a, b, alpha, beta);
    }

} // namespace alpaka::validation::gemm
