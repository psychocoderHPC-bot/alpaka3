/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "reference_gemm.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

/** @file
 *
 * Forward-error tolerance model for the alpaka3 GEMM tests.
 *
 * The derivation, the operation counts, the safety factor and the documented
 * limits live in TOLERANCE.md next to this header. The short version is
 *
 *     |C_hat_ij - C_ij| <= safety * gamma_n * (|alpha| * sum_k |A_ik B_kj|
 *                                           + |beta| * |C_ij|)
 *                          + 0.5 * denorm_min(output)
 *     gamma_n = n * u / (1 - n * u),  n = effective rounded operations
 *
 * where ``u`` is the unit roundoff of the accumulation format. The bound is
 * uninformative when ``n * u >= 1``; TOLERANCE.md describes the alternative
 * validation and this header reports that state instead of pretending a
 * vacuous bound is a pass.
 *
 * Validation-only header: no library code includes it.
 */

namespace alpaka::validation::gemm
{
    enum class NumericMode
    {
        fp32,
        fp64,
        fp16ToFp32,
        bf16ToFp32
    };

    inline constexpr double unitRoundoffFp32 = 0x1p-24;
    inline constexpr double unitRoundoffFp64 = 0x1p-53;
    inline constexpr double unitRoundoffFp16 = 0x1p-11; // 10 explicit fraction bits
    inline constexpr double unitRoundoffBf16 = 0x1p-8; // 7 explicit fraction bits

    /** Unit roundoff of the accumulation format of a mode. */
    constexpr double accumulationUnitRoundoff(NumericMode mode) noexcept
    {
        switch(mode)
        {
        case NumericMode::fp64:
            return unitRoundoffFp64;
        case NumericMode::fp32:
        case NumericMode::fp16ToFp32:
        case NumericMode::bf16ToFp32:
            return unitRoundoffFp32;
        }
        return unitRoundoffFp32;
    }

    /** Unit roundoff of the stored input format of a mode.
     *
     * Zero for FP32/FP64 because stored and computed operands are identical;
     * nonzero for the storage-only modes, where it describes the separate
     * input-quantization error, never the accumulation error.
     */
    constexpr double inputUnitRoundoff(NumericMode mode) noexcept
    {
        switch(mode)
        {
        case NumericMode::fp16ToFp32:
            return unitRoundoffFp16;
        case NumericMode::bf16ToFp32:
            return unitRoundoffBf16;
        case NumericMode::fp32:
        case NumericMode::fp64:
            return 0.0;
        }
        return 0.0;
    }

    inline constexpr double defaultSafetyFactor = 8.0;

    /** Configuration of one error bound.
     *
     * ``accumulations`` is K. ``alphaNonZero``, ``betaNonZero`` and
     * ``betaIsOne`` must describe the same alpha/beta that the comparison
     * passes. ``separateMultiplyAdd`` selects the non-FMA operation count.
     * ``forcedOperationCount`` exists only to unit-test the uninformative
     * ``n * u >= 1`` branch without allocating a matrix with millions of
     * columns; it must not be used for real evidence.
     */
    struct ToleranceModel
    {
        NumericMode mode = NumericMode::fp32;
        double unitRoundoff = unitRoundoffFp32;
        double safetyFactor = defaultSafetyFactor;
        std::size_t accumulations = 0;
        bool alphaNonZero = true;
        bool betaNonZero = false;
        bool betaIsOne = false;
        bool outputConversion = false;
        bool separateMultiplyAdd = false;
        std::optional<std::size_t> forcedOperationCount = std::nullopt;

        /** Number of rounding operations in the candidate's evaluation.
         *
         * K FMA roundings (or 2K for a separate multiply and add), one alpha
         * multiply, one add for the beta term when beta is one, two roundings
         * (beta multiply and add) otherwise, and one output conversion when
         * the accumulator type differs from the stored output type. The
         * degenerate paths (K == 0 or alpha == 0) reduce to the beta term
         * alone. Every term is counted once; the count is conservative for
         * alpha == 1 and beta == 0, which only makes the bound larger, never
         * smaller.
         */
        [[nodiscard]] std::size_t effectiveOperationCount() const noexcept
        {
            if(forcedOperationCount)
                return *forcedOperationCount;
            if(accumulations == 0 || !alphaNonZero)
                return (betaNonZero && !betaIsOne ? 1u : 0u) + (outputConversion ? 1u : 0u);
            std::size_t n = accumulations * (separateMultiplyAdd ? 2u : 1u);
            n += 1u; // alpha multiply
            n += betaNonZero ? (betaIsOne ? 1u : 2u) : 0u;
            n += outputConversion ? 1u : 0u;
            return n;
        }

        [[nodiscard]] long double nu() const noexcept
        {
            return static_cast<long double>(effectiveOperationCount()) * unitRoundoff;
        }

        /** False when n * u >= 1; the forward bound is then uninformative. */
        [[nodiscard]] bool informative() const noexcept
        {
            return nu() < 1.0L;
        }

        /** gamma_n = n*u/(1-n*u), infinite when n*u >= 1. */
        [[nodiscard]] long double gamma() const noexcept
        {
            auto const value = nu();
            if(value >= 1.0L)
                return std::numeric_limits<long double>::infinity();
            return value / (1.0L - value);
        }

        /** Per-element absolute bound for a given magnitude scale. */
        template<typename OutT>
        [[nodiscard]] long double elementBound(long double scale) const noexcept
        {
            // Each of the n roundings can hide an underflowed term of up to
            // 0.5 denorm_min; one extra floor covers the final store.
            auto const floor = (static_cast<long double>(effectiveOperationCount()) + 1.0L) * absoluteFloor<OutT>();
            return safetyFactor * gamma() * scale + floor;
        }

        template<typename OutT>
        [[nodiscard]] static long double absoluteFloor() noexcept
        {
            if constexpr(std::numeric_limits<OutT>::has_denorm != std::denorm_absent)
                return static_cast<long double>(std::numeric_limits<OutT>::denorm_min());
            else
                return 0.0L;
        }
    };

    /** Build the tolerance model for one comparison. */
    inline ToleranceModel makeToleranceModel(
        NumericMode mode,
        std::size_t accumulations,
        bool alphaNonZero = true,
        bool betaNonZero = true,
        bool betaIsOne = false,
        bool outputConversion = false,
        bool separateMultiplyAdd = false)
    {
        ToleranceModel model;
        model.mode = mode;
        model.unitRoundoff = accumulationUnitRoundoff(mode);
        model.accumulations = accumulations;
        model.alphaNonZero = alphaNonZero;
        model.betaNonZero = betaNonZero;
        model.betaIsOne = betaIsOne;
        model.outputConversion = outputConversion;
        model.separateMultiplyAdd = separateMultiplyAdd;
        return model;
    }

    struct ComparisonReport
    {
        bool informative = false;
        bool withinBound = false;
        std::size_t operations = 0;
        long double gamma = std::numeric_limits<long double>::infinity();

        long double maxAbsError = 0.0L;
        std::size_t worstErrorI = 0;
        std::size_t worstErrorJ = 0;
        long double boundAtWorstError = 0.0L;

        long double maxScaledError = 0.0L;
        std::size_t worstScaledI = 0;
        std::size_t worstScaledJ = 0;

        long double maxBound = 0.0L;
        std::size_t maxBoundI = 0;
        std::size_t maxBoundJ = 0;
    };

    namespace detail
    {
        inline long double magnitude(long double value) noexcept
        {
            return std::fabs(value);
        }

        template<typename T>
        long double asLongDouble(T const& value)
        {
            return static_cast<long double>(referenceValue(value));
        }

        template<typename RefT, typename CandT>
        long double difference(RefT const& reference, CandT const& candidate) noexcept
        {
            auto const r = static_cast<long double>(reference);
            auto const c = static_cast<long double>(candidate);
            if(std::isfinite(r) && std::isfinite(c))
                return std::fabs(c - r);
            // Equal non-finite values (both infinities of the same sign) are
            // treated as no error; NaN never compares equal.
            return (r == c) ? 0.0L : std::numeric_limits<long double>::infinity();
        }
    } // namespace detail

    /** Compare a candidate result against a higher-precision reference.
     *
     * ``candidate`` and ``reference`` are logical M x N views (no transpose).
     * ``a`` and ``b`` are the operands as the candidate consumed them, so S_ij
     * uses the quantized values for FP16/BF16 storage modes. ``cOld`` is only
     * read when beta is nonzero, matching the reference's zero-size contract.
     * ``referenceValue()`` is applied to the operands only, not to C.
     */
    template<typename OutT, typename RefT, typename AT, typename BT, typename CT, typename AlphaT, typename BetaT>
    ComparisonReport compareGemm(
        ToleranceModel const& model,
        MatrixView<OutT> candidate,
        MatrixView<RefT> reference,
        MatrixView<AT> a,
        MatrixView<BT> b,
        MatrixView<CT> cOld,
        AlphaT alpha,
        BetaT beta)
    {
        detail::requireScalar<AlphaT, BetaT>();
        if(candidate.op != Op::none || reference.op != Op::none)
            throw std::invalid_argument("compareGemm: C views must not be transposed");

        std::size_t const m = candidate.rows;
        std::size_t const n = candidate.cols;
        std::size_t const k = a.logicalCols();
        if(reference.rows != m || reference.cols != n || a.logicalRows() != m || b.logicalCols() != n
           || b.logicalRows() != k)
            throw std::invalid_argument("compareGemm: inconsistent matrix dimensions");
        if(model.accumulations != k)
            throw std::invalid_argument("compareGemm: tolerance model was built for a different K");
        if(m == 0 || n == 0)
            return {};

        bool const betaIsZero = beta == BetaT(0);

        ComparisonReport report;
        report.operations = model.effectiveOperationCount();
        report.gamma = model.gamma();
        report.informative = model.informative();
        report.withinBound = true;

        for(std::size_t i = 0; i < m; ++i)
        {
            for(std::size_t j = 0; j < n; ++j)
            {
                long double products = 0.0L;
                for(std::size_t kk = 0; kk < k; ++kk)
                {
                    products += detail::magnitude(detail::asLongDouble(a.at(i, kk)))
                                * detail::magnitude(detail::asLongDouble(b.at(kk, j)));
                }

                long double scale = detail::magnitude(static_cast<long double>(alpha)) * products;
                if(!betaIsZero)
                    scale += detail::magnitude(static_cast<long double>(beta))
                             * detail::magnitude(static_cast<long double>(cOld.at(i, j)));

                long double const bound = model.template elementBound<OutT>(scale);
                long double const error = detail::difference(reference.stored(i, j), candidate.stored(i, j));
                long double const scaled = scale > 0.0L
                                               ? error / scale
                                               : (error == 0.0L ? 0.0L : std::numeric_limits<long double>::infinity());

                if(error > report.maxAbsError)
                {
                    report.maxAbsError = error;
                    report.worstErrorI = i;
                    report.worstErrorJ = j;
                    report.boundAtWorstError = bound;
                }

                if(scaled > report.maxScaledError)
                {
                    report.maxScaledError = scaled;
                    report.worstScaledI = i;
                    report.worstScaledJ = j;
                }
                if(bound > report.maxBound)
                {
                    report.maxBound = bound;
                    report.maxBoundI = i;
                    report.maxBoundJ = j;
                }
                if(error > bound)
                    report.withinBound = false;
            }
        }
        return report;
    }

    struct QuantizationReport
    {
        bool withinBound = false;
        long double maxAbsError = 0.0L;
        std::size_t worstI = 0;
        std::size_t worstJ = 0;
        long double boundAtWorstError = 0.0L;
        long double maxBound = 0.0L;
    };

    /** Separate input-quantization error of a storage-only mode.
     *
     * Returns ``quantized_reference - original_reference`` and checks it
     * against the exact first-order propagation
     *
     *     |alpha| * sum_k (|a_o| |b_o - b_q| + |b_q| |a_o - a_q|)
     *
     * plus a small safety factor. The beta*C term cancels because both
     * references use the same C. Accumulation error is not part of this
     * quantity; compare it separately with ``compareGemm``.
     */
    template<typename ResultT, typename AT, typename AQT, typename BT, typename BQT, typename AlphaT>
    QuantizationReport compareInputQuantization(
        MatrixView<ResultT> originalReference,
        MatrixView<ResultT> quantizedReference,
        MatrixView<AT> aOriginal,
        MatrixView<AQT> aQuantized,
        MatrixView<BT> bOriginal,
        MatrixView<BQT> bQuantized,
        AlphaT alpha,
        double safetyFactor = 4.0)
    {
        std::size_t const m = originalReference.rows;
        std::size_t const n = originalReference.cols;
        std::size_t const k = aOriginal.logicalCols();
        if(quantizedReference.rows != m || quantizedReference.cols != n || aQuantized.logicalCols() != k
           || bOriginal.logicalCols() != n || bQuantized.logicalCols() != n || bOriginal.logicalRows() != k
           || bQuantized.logicalRows() != k)
            throw std::invalid_argument("compareInputQuantization: inconsistent matrix dimensions");
        if(m == 0 || n == 0)
            return {};

        QuantizationReport report;
        report.withinBound = true;
        for(std::size_t i = 0; i < m; ++i)
        {
            for(std::size_t j = 0; j < n; ++j)
            {
                long double bound = 0.0L;
                for(std::size_t kk = 0; kk < k; ++kk)
                {
                    long double const ao = detail::asLongDouble(aOriginal.at(i, kk));
                    long double const aq = detail::asLongDouble(aQuantized.at(i, kk));
                    long double const bo = detail::asLongDouble(bOriginal.at(kk, j));
                    long double const bq = detail::asLongDouble(bQuantized.at(kk, j));
                    bound += detail::magnitude(ao) * detail::magnitude(bo - bq)
                             + detail::magnitude(bq) * detail::magnitude(ao - aq);
                }
                bound *= detail::magnitude(static_cast<long double>(alpha)) * safetyFactor;
                bound += ToleranceModel::template absoluteFloor<ResultT>();

                long double const error
                    = detail::difference(originalReference.stored(i, j), quantizedReference.stored(i, j));
                if(error > bound)
                    report.withinBound = false;
                if(bound > report.maxBound)
                    report.maxBound = bound;
                if(error > report.maxAbsError)
                {
                    report.maxAbsError = error;
                    report.worstI = i;
                    report.worstJ = j;
                    report.boundAtWorstError = bound;
                }
            }
        }
        return report;
    }
} // namespace alpaka::validation::gemm
