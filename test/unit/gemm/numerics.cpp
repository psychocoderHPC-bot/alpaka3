/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "../../../validation/numerics/tolerance.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

/** @file
 *
 * Numerical tests for the independent GEMM reference and its tolerance model.
 *
 * The GEMM kernel does not exist yet at M1, so these tests exercise the
 * numerical foundation the later kernel tests use:
 *
 * * the reference against the higher-precision reference and against
 *   hand-computed patterns,
 * * the forward-error bound on candidates built with a deliberately simple
 *   accumulation (the naive FP32 reference),
 * * discrimination of the bound at and beyond its boundary,
 * * zero-size read-elision rules,
 * * FP16/BF16 bit-pattern paths with a separate input-quantization report,
 * * the uninformative n*u >= 1 branch.
 *
 * Every comparison asserts with the derived bound and prints the maximum
 * absolute error, the maximum scale-normalized error and the worst coordinate
 * on failure.
 */

namespace
{
    using namespace alpaka::validation::gemm;

    template<typename T>
    struct HostMatrix
    {
        std::size_t rows;
        std::size_t cols;
        std::size_t ld;
        Layout layout;
        Op op = Op::none;
        std::vector<T> storage;

        HostMatrix(
            std::size_t rowsIn,
            std::size_t colsIn,
            Layout layoutIn = Layout::rowMajor,
            std::size_t pad = 0)
            : rows(rowsIn)
            , cols(colsIn)
            , ld((layoutIn == Layout::rowMajor ? colsIn : rowsIn) + pad)
            , layout(layoutIn)
        {
            auto const extent = layout == Layout::rowMajor ? rows : cols;
            storage.assign(std::max<std::size_t>(ld * extent, 1), T{});
        }

        [[nodiscard]] MatrixView<T> view()
        {
            return {storage.data(), rows, cols, ld, layout, op};
        }

        [[nodiscard]] MatrixView<const T> view() const
        {
            return {storage.data(), rows, cols, ld, layout, op};
        }
    };

    struct Rng
    {
        std::mt19937_64 engine;

        explicit Rng(std::uint64_t seed) : engine(seed)
        {
        }

        [[nodiscard]] float uniform(float low = -1.0f, float high = 1.0f)
        {
            return std::uniform_real_distribution<float>(low, high)(engine);
        }

        [[nodiscard]] int sign()
        {
            return std::uniform_int_distribution<int>(0, 1)(engine) == 0 ? -1 : 1;
        }
    };

    template<typename T>
    void fillRandom(HostMatrix<T>& matrix, Rng& rng, float low = -1.0f, float high = 1.0f)
    {
        for(auto& value : matrix.storage)
            value = static_cast<T>(rng.uniform(low, high));
    }

    void requireWithin(ComparisonReport const& report, std::string const& label)
    {
        INFO(
            label << ": informative=" << report.informative << " operations=" << report.operations
                  << " maxAbsError=" << static_cast<double>(report.maxAbsError)
                  << " maxScaledError=" << static_cast<double>(report.maxScaledError)
                  << " maxBound=" << static_cast<double>(report.maxBound) << " worst=(" << report.worstErrorI << ","
                  << report.worstErrorJ << ")"
                  << " worstScaled=(" << report.worstScaledI << "," << report.worstScaledJ << ")");
        REQUIRE(report.informative);
        REQUIRE(report.withinBound);
    }

    void copyInto(HostMatrix<double>& target, HostMatrix<float> const& source)
    {
        for(std::size_t j = 0; j < source.cols; ++j)
            for(std::size_t i = 0; i < source.rows; ++i)
                target.view().stored(i, j) = static_cast<double>(source.view().stored(i, j));
    }

    /** FP32 candidate against the FP64 reference for one configuration. */
    void checkFp32Case(
        std::string const& label,
        HostMatrix<float> const& a,
        HostMatrix<float> const& b,
        HostMatrix<float> const& cOld,
        float alpha,
        float beta)
    {
        HostMatrix<float> candidate = cOld;
        HostMatrix<double> reference(cOld.rows, cOld.cols, cOld.layout);
        copyInto(reference, cOld);

        gemm_reference<double>(reference.view(), a.view(), b.view(), alpha, beta);
        gemm_reference<float>(candidate.view(), a.view(), b.view(), alpha, beta);

        auto const model = makeToleranceModel(
            NumericMode::fp32,
            a.view().logicalCols(),
            alpha != 0.0f,
            beta != 0.0f,
            beta == 1.0f);
        auto const report = compareGemm(
            model,
            candidate.view(),
            reference.view(),
            a.view(),
            b.view(),
            cOld.view(),
            alpha,
            beta);
        requireWithin(report, label);
    }

} // namespace

TEST_CASE("gemm: reference GEMM agrees with the FP64 reference on small shapes and all layouts", "[gemm][numerics]")
{
    Layout const layouts[] = {Layout::rowMajor, Layout::colMajor};
    Op const operations[] = {Op::none, Op::transpose};
    std::size_t const extents[] = {1, 2, 3, 5};

    for(auto m : extents)
        for(auto n : extents)
            for(auto k : extents)
                for(auto layoutA : layouts)
                    for(auto layoutB : layouts)
                        for(auto layoutC : layouts)
                            for(auto opA : operations)
                                for(auto opB : operations)
                                {
                                    Rng rng(0x9e3779b97f4a7c15ULL * (m * 100 + n * 10 + k) + static_cast<int>(opA));
                                    std::size_t const storedRowsA = opA == Op::none ? m : k;
                                    std::size_t const storedColsA = opA == Op::none ? k : m;
                                    std::size_t const storedRowsB = opB == Op::none ? k : n;
                                    std::size_t const storedColsB = opB == Op::none ? n : k;

                                    HostMatrix<float> a(storedRowsA, storedColsA, layoutA, 2);
                                    HostMatrix<float> b(storedRowsB, storedColsB, layoutB, 3);
                                    HostMatrix<float> cOld(m, n, layoutC, 1);
                                    fillRandom(a, rng);
                                    fillRandom(b, rng);
                                    fillRandom(cOld, rng, -0.5f, 0.5f);
                                    a.op = opA;
                                    b.op = opB;

                                    checkFp32Case(
                                        "small-shape m=" + std::to_string(m) + " n=" + std::to_string(n) + " k="
                                            + std::to_string(k),
                                        a,
                                        b,
                                        cOld,
                                        1.5f,
                                        0.25f);
                                }
}

TEST_CASE("gemm: reference GEMM follows the zero-size and read-elision contract", "[gemm][numerics]")
{
    SECTION("M=0 and N=0 are no-ops")
    {
        MatrixView<const float> emptyA{nullptr, 0, 3, 3, Layout::rowMajor, Op::none};
        MatrixView<const float> emptyB{nullptr, 3, 4, 4, Layout::rowMajor, Op::none};
        HostMatrix<float> c(0, 4);
        gemm_reference<double>(c.view(), emptyA, emptyB, 2.0f, 3.0f);
        REQUIRE(c.storage.size() == 1);

        MatrixView<const float> aNull{nullptr, 3, 2, 2, Layout::rowMajor, Op::none};
        MatrixView<const float> bNull{nullptr, 2, 0, 0, Layout::rowMajor, Op::none};
        HostMatrix<float> cZeroCols(3, 0);
        gemm_reference<double>(cZeroCols.view(), aNull, bNull, 2.0f, 3.0f);
        REQUIRE(cZeroCols.storage.size() == 1);
    }

    SECTION("K=0 reads neither A nor B and computes beta*C")
    {
        HostMatrix<float> c(2, 3);
        for(std::size_t i = 0; i < c.storage.size(); ++i)
            c.storage[i] = 0.25f * static_cast<float>(i + 1);
        HostMatrix<float> const old = c;

        MatrixView<const float> aNull{nullptr, 2, 0, 0, Layout::rowMajor, Op::none};
        MatrixView<const float> bNull{nullptr, 0, 3, 3, Layout::rowMajor, Op::none};
        gemm_reference<double>(c.view(), aNull, bNull, 2.0f, 3.0f);

        for(std::size_t j = 0; j < 3; ++j)
            for(std::size_t i = 0; i < 2; ++i)
                REQUIRE(c.view().stored(i, j) == 3.0f * old.view().stored(i, j));
    }

    SECTION("alpha=0 reads neither A nor B even with K>0")
    {
        HostMatrix<float> c(2, 3);
        Rng alphaZeroRng(11);
        fillRandom(c, alphaZeroRng, -0.5f, 0.5f);
        HostMatrix<float> const old = c;
        MatrixView<const float> aNull{nullptr, 2, 4, 0, Layout::rowMajor, Op::none};
        MatrixView<const float> bNull{nullptr, 4, 3, 3, Layout::rowMajor, Op::none};

        gemm_reference<double>(c.view(), aNull, bNull, 0.0f, -0.5f);
        for(std::size_t i = 0; i < c.storage.size(); ++i)
            REQUIRE(c.storage[i] == -0.5f * old.storage[i]);
    }

    SECTION("beta=0 does not read the old C")
    {
        HostMatrix<float> a(2, 4);
        HostMatrix<float> b(4, 3);
        Rng rng(42);
        fillRandom(a, rng);
        fillRandom(b, rng);

        HostMatrix<float> c(2, 3);
        std::fill(c.storage.begin(), c.storage.end(), std::numeric_limits<float>::quiet_NaN());
        HostMatrix<double> reference(2, 3);
        gemm_reference<double>(reference.view(), a.view(), b.view(), 1.25f, 0.0f);
        gemm_reference<float>(c.view(), a.view(), b.view(), 1.25f, 0.0f);

        for(std::size_t j = 0; j < 3; ++j)
            for(std::size_t i = 0; i < 2; ++i)
            {
                REQUIRE(std::isfinite(c.view().stored(i, j)));
            }

        auto const report
            = compareGemm(
                makeToleranceModel(NumericMode::fp32, 4, true, false),
                c.view(),
                reference.view(),
                a.view(),
                b.view(),
                MatrixView<const float>{},
                1.25f,
                0.0f);
        requireWithin(report, "beta=0 with a NaN old C");
    }

    SECTION("beta=1 with K=0 preserves the stored bits")
    {
        HostMatrix<float> c(2, 2);
        c.view().stored(0, 0) = -0.0f;
        c.view().stored(0, 1) = std::numeric_limits<float>::quiet_NaN();
        c.view().stored(1, 0) = std::numeric_limits<float>::denorm_min();
        c.view().stored(1, 1) = 7.5f;
        auto const bits = [&](std::size_t i, std::size_t j)
        { return std::bit_cast<std::uint32_t>(c.view().stored(i, j)); };
        auto const before = std::array<std::uint32_t, 4>{bits(0, 0), bits(0, 1), bits(1, 0), bits(1, 1)};

        MatrixView<const float> aNull{nullptr, 2, 0, 0, Layout::rowMajor, Op::none};
        MatrixView<const float> bNull{nullptr, 0, 2, 2, Layout::rowMajor, Op::none};
        gemm_reference<double>(c.view(), aNull, bNull, 3.0f, 1.0f);

        REQUIRE(bits(0, 0) == before[0]);
        REQUIRE(bits(0, 1) == before[1]);
        REQUIRE(bits(1, 0) == before[2]);
        REQUIRE(bits(1, 1) == before[3]);
    }
}

TEST_CASE("gemm: reference GEMM rejects inconsistent dimensions and strides", "[gemm][numerics]")
{
    HostMatrix<float> a(2, 3);
    HostMatrix<float> b(3, 2);
    HostMatrix<float> c(2, 2);

    auto badA = a.view();
    badA.rows = 1;
    REQUIRE_THROWS_AS(gemm_reference<double>(c.view(), badA, b.view(), 1.0f, 0.0f), std::invalid_argument);

    auto badB = b.view();
    badB.cols = 1;
    REQUIRE_THROWS_AS(gemm_reference<double>(c.view(), a.view(), badB, 1.0f, 0.0f), std::invalid_argument);

    auto badStride = a.view();
    badStride.ld = 1;
    REQUIRE_THROWS_AS(gemm_reference<double>(c.view(), badStride, b.view(), 1.0f, 0.0f), std::invalid_argument);

    auto transposedOutput = c.view();
    transposedOutput.op = Op::transpose;
    REQUIRE_THROWS_AS(
        gemm_reference<double>(transposedOutput, a.view(), b.view(), 1.0f, 0.0f),
        std::invalid_argument);
}

TEST_CASE("gemm: reference GEMM is exact on identity, diagonal and rank-one patterns", "[gemm][numerics]")
{
    SECTION("identity A leaves B unchanged")
    {
        std::size_t const n = 5;
        HostMatrix<float> a(n, n, Layout::colMajor, 2);
        for(std::size_t i = 0; i < n; ++i)
            a.view().stored(i, i) = 1.0f;
        HostMatrix<float> b(n, 3);
        Rng rng(7);
        fillRandom(b, rng);
        HostMatrix<float> cOld(n, 3);
        fillRandom(cOld, rng, -0.5f, 0.5f);
        HostMatrix<float> candidate = cOld;
        HostMatrix<double> reference(n, 3);
        gemm_reference<double>(reference.view(), a.view(), b.view(), 1.0f, 0.0f);
        gemm_reference<float>(candidate.view(), a.view(), b.view(), 1.0f, 0.0f);
        for(std::size_t j = 0; j < 3; ++j)
            for(std::size_t i = 0; i < n; ++i)
                REQUIRE(candidate.view().stored(i, j) == b.view().stored(i, j));
    }

    SECTION("diagonal A scales rows of B")
    {
        std::size_t const n = 4;
        HostMatrix<float> a(n, n);
        for(std::size_t i = 0; i < n; ++i)
            a.view().stored(i, i) = static_cast<float>(i + 1);
        HostMatrix<float> b(n, 2);
        Rng rng(8);
        fillRandom(b, rng);
        HostMatrix<float> cOld(n, 2);
        fillRandom(cOld, rng, -0.5f, 0.5f);
        HostMatrix<float> candidate = cOld;

        HostMatrix<double> reference(n, 2);
        copyInto(reference, cOld);
        gemm_reference<double>(reference.view(), a.view(), b.view(), 2.0f, -1.0f);
        gemm_reference<float>(candidate.view(), a.view(), b.view(), 2.0f, -1.0f);
        checkFp32Case("diagonal", a, b, cOld, 2.0f, -1.0f);
    }

    SECTION("rank-one A and B reproduce the outer product")
    {
        HostMatrix<float> a(5, 1);
        HostMatrix<float> b(1, 4);
        Rng rng(9);
        fillRandom(a, rng);
        fillRandom(b, rng);
        HostMatrix<float> cOld(5, 4);
        fillRandom(cOld, rng, -0.5f, 0.5f);
        checkFp32Case("rank-one", a, b, cOld, 1.0f, 0.0f);
    }
}

TEST_CASE("gemm: reference GEMM survives cancellation and scale-separated magnitudes", "[gemm][numerics]")
{
    SECTION("near-total cancellation with random signs")
    {
        HostMatrix<float> a(3, 4, Layout::rowMajor, 2);
        HostMatrix<float> b(4, 2);
        Rng rng(1234);
        fillRandom(a, rng);
        for(std::size_t k = 0; k < 4; ++k)
            a.view().stored(0, k) = (k < 3 ? 1.0f : 2.0f);
        for(std::size_t k = 0; k < 4; ++k)
            for(std::size_t j = 0; j < 2; ++j)
                b.view().stored(k, j) = static_cast<float>(rng.sign());
        // Row 0: 1 - 1 + 1e-7 + 0 -> result near 1e-7 while sum|ab| is ~4.
        b.view().stored(0, 0) = 1.0f;
        b.view().stored(1, 0) = -1.0f;
        b.view().stored(2, 0) = 1.0e-7f;
        b.view().stored(3, 0) = 0.0f;

        HostMatrix<float> cOld(3, 2);
        fillRandom(cOld, rng, -0.5f, 0.5f);
        checkFp32Case("cancellation", a, b, cOld, 1.0f, 0.0f);
    }

    SECTION("scale-separated magnitudes including an underflowing product")
    {
        HostMatrix<float> a(2, 4);
        HostMatrix<float> b(4, 2);
        a.view().stored(0, 0) = 1.0e15f;
        a.view().stored(0, 1) = 1.0e-15f;
        a.view().stored(0, 2) = 1.0e-25f;
        a.view().stored(0, 3) = 3.0f;
        a.view().stored(1, 0) = -1.0e15f;
        a.view().stored(1, 1) = 1.0e-15f;
        a.view().stored(1, 2) = 1.0e-25f;
        a.view().stored(1, 3) = 2.0f;
        b.view().stored(0, 0) = 1.0e-15f;
        b.view().stored(0, 1) = -1.0e-15f;
        b.view().stored(1, 0) = 1.0e15f;
        b.view().stored(1, 1) = 1.0e15f;
        b.view().stored(2, 0) = 1.0e-25f;
        b.view().stored(2, 1) = 1.0e-25f;
        b.view().stored(3, 0) = 1.0f;
        b.view().stored(3, 1) = -1.0f;

        HostMatrix<float> cOld(2, 2);
        Rng rng(77);
        fillRandom(cOld, rng, -0.5f, 0.5f);
        checkFp32Case("scale-separated", a, b, cOld, 1.0f, 0.25f);
    }
}

TEST_CASE("gemm: padded leading dimensions and shifted subviews keep the declared coordinates", "[gemm][numerics]")
{
    for(auto layout : {Layout::rowMajor, Layout::colMajor})
    {
        std::size_t const rows = 4;
        std::size_t const cols = 3;
        // Base matrix with room for an offset and padding. The subview starts at
        // a deliberately misaligned element.
        std::size_t const baseRows = rows + 3;
        std::size_t const baseCols = cols + 4;
        HostMatrix<float> base(baseRows, baseCols, layout, 2);
        Rng rng(2024);
        fillRandom(base, rng, -2.0f, 2.0f);

        std::size_t const offset = layout == Layout::rowMajor ? base.ld + 2 : 2 * base.ld + 1;
        auto aView = base.view();
        aView.data += offset;
        aView.rows = rows;
        aView.cols = cols;

        HostMatrix<float> a(rows, cols, layout, 0);
        for(std::size_t j = 0; j < cols; ++j)
            for(std::size_t i = 0; i < rows; ++i)
                a.view().stored(i, j) = aView.stored(i, j);

        HostMatrix<float> identity(cols, cols);
        for(std::size_t i = 0; i < cols; ++i)
            identity.view().stored(i, i) = 1.0f;
        HostMatrix<float> cOld(rows, cols);
        fillRandom(cOld, rng, -0.5f, 0.5f);
        HostMatrix<float> candidate = cOld;
        gemm_reference<double>(candidate.view(), aView, identity.view(), 1.0f, 0.0f);

        for(std::size_t j = 0; j < cols; ++j)
            for(std::size_t i = 0; i < rows; ++i)
                REQUIRE(candidate.view().stored(i, j) == a.view().stored(i, j));
    }
}

TEST_CASE("gemm: tolerance model passes inside the bound and rejects beyond it", "[gemm][numerics]")
{
    Rng rng(555);
    HostMatrix<float> a(4, 8, Layout::rowMajor, 1);
    HostMatrix<float> b(8, 5, Layout::colMajor, 2);
    HostMatrix<float> cOld(4, 5);
    fillRandom(a, rng);
    fillRandom(b, rng);
    fillRandom(cOld, rng, -0.5f, 0.5f);

    HostMatrix<double> reference(4, 5);
    copyInto(reference, cOld);
    gemm_reference<double>(reference.view(), a.view(), b.view(), 2.0f, 0.5f);

    HostMatrix<float> candidate(4, 5);
    for(std::size_t j = 0; j < 5; ++j)
        for(std::size_t i = 0; i < 4; ++i)
            candidate.view().stored(i, j) = static_cast<float>(reference.view().stored(i, j));

    auto const model = makeToleranceModel(NumericMode::fp32, 8, true, true);
    auto const baseline = compareGemm(
        model,
        candidate.view(),
        reference.view(),
        a.view(),
        b.view(),
        cOld.view(),
        2.0f,
        0.5f);
    requireWithin(baseline, "rounded reference candidate");

    std::size_t const i = baseline.maxBoundI;
    std::size_t const j = baseline.maxBoundJ;

    // Half a bound of extra error at the coordinate with the largest bound
    // must still pass; twice the bound must be rejected.
    HostMatrix<float> inside = candidate;
    inside.view().stored(i, j)
        = static_cast<float>(static_cast<double>(inside.view().stored(i, j)) + 0.5 * baseline.maxBound);
    auto const insideReport
        = compareGemm(model, inside.view(), reference.view(), a.view(), b.view(), cOld.view(), 2.0f, 0.5f);
    requireWithin(insideReport, "perturbation at half the bound");

    HostMatrix<float> outside = candidate;
    outside.view().stored(i, j)
        = static_cast<float>(static_cast<double>(outside.view().stored(i, j)) + 2.0 * baseline.maxBound);
    auto const outsideReport
        = compareGemm(model, outside.view(), reference.view(), a.view(), b.view(), cOld.view(), 2.0f, 0.5f);
    INFO(
        "beyond-bound maxAbsError=" << static_cast<double>(outsideReport.maxAbsError)
                                    << " maxBound=" << static_cast<double>(outsideReport.maxBound));
    REQUIRE_FALSE(outsideReport.withinBound);
}

TEST_CASE("gemm: tolerance model covers FMA and separate multiply-add operation counts", "[gemm][numerics]")
{
    auto const fma = makeToleranceModel(NumericMode::fp32, 16, true, true, false, false, false);
    auto const separate = makeToleranceModel(NumericMode::fp32, 16, true, true, false, false, true);
    REQUIRE(fma.effectiveOperationCount() == 19);
    REQUIRE(separate.effectiveOperationCount() == 35);
    REQUIRE(separate.gamma() > fma.gamma());

    Rng rng(31415);
    HostMatrix<float> a(3, 16);
    HostMatrix<float> b(16, 4);
    HostMatrix<float> cOld(3, 4);
    fillRandom(a, rng);
    fillRandom(b, rng);
    fillRandom(cOld, rng, -0.5f, 0.5f);

    HostMatrix<float> candidate = cOld;
    HostMatrix<double> reference(3, 4);
    copyInto(reference, cOld);
    gemm_reference<double>(reference.view(), a.view(), b.view(), 1.0f, 1.0f);
    gemm_reference<float>(candidate.view(), a.view(), b.view(), 1.0f, 1.0f);

    auto const report
        = compareGemm(separate, candidate.view(), reference.view(), a.view(), b.view(), cOld.view(), 1.0f, 1.0f);
    requireWithin(report, "separate multiply-add bound");
}

TEST_CASE("gemm: output conversion is part of the bound", "[gemm][numerics]")
{
    Rng rng(2718);
    HostMatrix<float> a(4, 12);
    HostMatrix<float> b(12, 3);
    HostMatrix<float> cOld(4, 3);
    fillRandom(a, rng);
    fillRandom(b, rng);
    fillRandom(cOld, rng, -0.5f, 0.5f);

    HostMatrix<double> reference(4, 3);
    gemm_reference<double>(reference.view(), a.view(), b.view(), 1.0f, 0.0f);

    // Candidate computed in double and stored to float: exactly one output
    // conversion, which the model counts as an extra rounding.
    HostMatrix<float> candidate(4, 3);
    for(std::size_t j = 0; j < 3; ++j)
        for(std::size_t i = 0; i < 4; ++i)
            candidate.view().stored(i, j) = static_cast<float>(reference.view().stored(i, j));

    auto const model = makeToleranceModel(NumericMode::fp32, 12, true, false, false, true);
    auto const report
        = compareGemm(model, candidate.view(), reference.view(), a.view(), b.view(), MatrixView<const float>{}, 1.0f, 0.0f);
    requireWithin(report, "double accumulator stored to float");
}

TEST_CASE("gemm: FP16 and BF16 storage paths separate quantization from accumulation error", "[gemm][numerics]")
{
    Rng rng(1616);
    std::size_t const m = 3;
    std::size_t const n = 4;
    std::size_t const k = 7;

    HostMatrix<float> aOrig(m, k);
    HostMatrix<float> bOrig(k, n);
    HostMatrix<float> cOld(m, n);
    fillRandom(aOrig, rng);
    fillRandom(bOrig, rng);
    fillRandom(cOld, rng, -0.5f, 0.5f);
    // Values that exercise subnormal FP16, a tie at the FP16 precision and a
    // value near the BF16 precision.
    aOrig.view().stored(0, 0) = 1.0e-7f;
    aOrig.view().stored(0, 1) = 1.0f + std::ldexp(1.0f, -11);
    aOrig.view().stored(0, 2) = 1.0f + std::ldexp(1.0f, -8);
    bOrig.view().stored(0, 0) = 3.0f;

    SECTION("FP16 -> FP32")
    {
        HostMatrix<Fp16Bits> aQ(m, k);
        HostMatrix<Fp16Bits> bQ(k, n);
        for(std::size_t j = 0; j < k; ++j)
            for(std::size_t i = 0; i < m; ++i)
                aQ.view().stored(i, j) = Fp16Bits{floatToFp16Bits(aOrig.view().stored(i, j))};
        for(std::size_t j = 0; j < n; ++j)
            for(std::size_t i = 0; i < k; ++i)
                bQ.view().stored(i, j) = Fp16Bits{floatToFp16Bits(bOrig.view().stored(i, j))};

        HostMatrix<float> candidate = cOld;
        HostMatrix<double> reference(m, n);
        HostMatrix<double> originalReference(m, n);
        copyInto(reference, cOld);
        copyInto(originalReference, cOld);
        gemm_reference<float>(candidate.view(), aQ.view(), bQ.view(), 1.0f, 0.25f);
        gemm_reference<double>(reference.view(), aQ.view(), bQ.view(), 1.0f, 0.25f);
        gemm_reference<double>(originalReference.view(), aOrig.view(), bOrig.view(), 1.0f, 0.25f);

        auto const model = makeToleranceModel(NumericMode::fp16ToFp32, k, true, true, false);
        auto const report
            = compareGemm(model, candidate.view(), reference.view(), aQ.view(), bQ.view(), cOld.view(), 1.0f, 0.25f);
        requireWithin(report, "fp16 accumulation against quantized inputs");

        auto const quantization = compareInputQuantization(
            originalReference.view(),
            reference.view(),
            aOrig.view(),
            aQ.view(),
            bOrig.view(),
            bQ.view(),
            1.0f);
        INFO(
            "fp16 input quantization: maxAbsError=" << static_cast<double>(quantization.maxAbsError)
                                                   << " maxBound=" << static_cast<double>(quantization.maxBound)
                                                   << " worst=(" << quantization.worstI << ","
                                                   << quantization.worstJ << ")");
        REQUIRE(quantization.withinBound);
        REQUIRE(quantization.maxAbsError > 0.0L);
    }

    SECTION("BF16 -> FP32")
    {
        HostMatrix<Bf16Bits> aQ(m, k);
        HostMatrix<Bf16Bits> bQ(k, n);
        for(std::size_t j = 0; j < k; ++j)
            for(std::size_t i = 0; i < m; ++i)
                aQ.view().stored(i, j) = Bf16Bits{floatToBf16Bits(aOrig.view().stored(i, j))};
        for(std::size_t j = 0; j < n; ++j)
            for(std::size_t i = 0; i < k; ++i)
                bQ.view().stored(i, j) = Bf16Bits{floatToBf16Bits(bOrig.view().stored(i, j))};

        HostMatrix<float> candidate = cOld;
        HostMatrix<double> reference(m, n);
        HostMatrix<double> originalReference(m, n);
        copyInto(reference, cOld);
        copyInto(originalReference, cOld);
        gemm_reference<float>(candidate.view(), aQ.view(), bQ.view(), 1.0f, 0.25f);
        gemm_reference<double>(reference.view(), aQ.view(), bQ.view(), 1.0f, 0.25f);
        gemm_reference<double>(originalReference.view(), aOrig.view(), bOrig.view(), 1.0f, 0.25f);

        auto const model = makeToleranceModel(NumericMode::bf16ToFp32, k, true, true, false);
        auto const report
            = compareGemm(model, candidate.view(), reference.view(), aQ.view(), bQ.view(), cOld.view(), 1.0f, 0.25f);
        requireWithin(report, "bf16 accumulation against quantized inputs");

        auto const quantization = compareInputQuantization(
            originalReference.view(),
            reference.view(),
            aOrig.view(),
            aQ.view(),
            bOrig.view(),
            bQ.view(),
            1.0f);
        INFO(
            "bf16 input quantization: maxAbsError=" << static_cast<double>(quantization.maxAbsError)
                                                   << " maxBound=" << static_cast<double>(quantization.maxBound));
        REQUIRE(quantization.withinBound);
        REQUIRE(quantization.maxAbsError > 0.0L);
    }
}

TEST_CASE("gemm: FP16 and BF16 bit conversion edge cases", "[gemm][numerics]")
{
    REQUIRE(fp16BitsToDouble(0x3c00) == 1.0);
    REQUIRE(fp16BitsToDouble(0x0001) == std::ldexp(1.0, -24));
    REQUIRE(fp16BitsToDouble(0x8000) == 0.0);
    REQUIRE(std::signbit(fp16BitsToDouble(0x8000)));
    REQUIRE(floatToFp16Bits(-0.0f) == 0x8000);
    REQUIRE(floatToFp16Bits(1.0f) == 0x3c00);
    REQUIRE(floatToFp16Bits(65520.0f) == 0x7c00);
    REQUIRE(floatToFp16Bits(1.0f + std::ldexp(1.0f, -11)) == 0x3c00);
    REQUIRE(floatToFp16Bits(1.0f + std::ldexp(1.0f, -10)) == 0x3c01);
    REQUIRE(fp16FromFloat(1.0e-7f) > 0.0f);
    REQUIRE(fp16FromFloat(1.0e-7f) < std::ldexp(1.0f, -14));

    REQUIRE(bf16BitsToDouble(0x3f80) == 1.0);
    REQUIRE(floatToBf16Bits(1.0f) == 0x3f80);
    REQUIRE(floatToBf16Bits(-2.0f) == 0xc000);
    REQUIRE(floatToBf16Bits(1.0f + std::ldexp(1.0f, -8)) == 0x3f80);
    REQUIRE(floatToBf16Bits(1.0f + std::ldexp(1.0f, -7)) == 0x3f81);
    REQUIRE(bf16FromFloat(1.0f + std::ldexp(1.0f, -8)) == 1.0f);
}

TEST_CASE("gemm: n*u >= 1 is reported as uninformative and validated by exact data", "[gemm][numerics]")
{
    // A real FP32 accumulation reaches n*u >= 1 only near K = 2^24, which is
    // not practical in a unit test. The forced count unit-tests the branch.
    auto model = makeToleranceModel(NumericMode::fp32, 4);
    model.forcedOperationCount = std::size_t{1} << 26;
    REQUIRE_FALSE(model.informative());
    REQUIRE(std::isinf(static_cast<double>(model.gamma())));

    // Alternative validation for an uninformative bound: operands whose dot
    // product is exactly representable, so the candidate must reproduce it
    // exactly rather than merely satisfy a vacuous bound.
    HostMatrix<float> a(2, 4);
    HostMatrix<float> b(4, 3);
    a.view().stored(0, 0) = 1.0f;
    a.view().stored(0, 1) = 2.0f;
    a.view().stored(0, 2) = -4.0f;
    a.view().stored(0, 3) = 8.0f;
    a.view().stored(1, 0) = 3.0f;
    a.view().stored(1, 1) = 3.0f;
    a.view().stored(1, 2) = 3.0f;
    a.view().stored(1, 3) = 3.0f;
    for(std::size_t j = 0; j < 3; ++j)
        for(std::size_t i = 0; i < 4; ++i)
            b.view().stored(i, j) = static_cast<float>(i + 1 - static_cast<std::size_t>(j));

    HostMatrix<float> candidate(2, 3);
    HostMatrix<double> reference(2, 3);
    gemm_reference<float>(candidate.view(), a.view(), b.view(), 1.0f, 0.0f);
    gemm_reference<double>(reference.view(), a.view(), b.view(), 1.0f, 0.0f);
    for(std::size_t j = 0; j < 3; ++j)
        for(std::size_t i = 0; i < 2; ++i)
            REQUIRE(candidate.view().stored(i, j) == static_cast<float>(reference.view().stored(i, j)));

    auto const report = compareGemm(
        model,
        candidate.view(),
        reference.view(),
        a.view(),
        b.view(),
        MatrixView<const float>{},
        1.0f,
        0.0f);
    REQUIRE_FALSE(report.informative);
    REQUIRE(std::isinf(static_cast<double>(report.maxBound)));
}

TEST_CASE("gemm: seeds reproduce reference results bit for bit", "[gemm][numerics]")
{
    auto run = [](std::uint64_t seed, HostMatrix<float>& out)
    {
        Rng rng(seed);
        HostMatrix<float> a(4, 6);
        HostMatrix<float> b(6, 5);
        HostMatrix<float> cOld(4, 5);
        fillRandom(a, rng);
        fillRandom(b, rng);
        fillRandom(cOld, rng, -0.5f, 0.5f);
        out = cOld;
        HostMatrix<double> reference(4, 5);
        gemm_reference<double>(reference.view(), a.view(), b.view(), 1.0f, 0.0f);
        for(std::size_t j = 0; j < 5; ++j)
            for(std::size_t i = 0; i < 4; ++i)
                out.view().stored(i, j) = static_cast<float>(reference.view().stored(i, j));
    };

    HostMatrix<float> first(1, 1);
    HostMatrix<float> second(1, 1);
    HostMatrix<float> different(1, 1);
    run(99, first);
    run(99, second);
    run(100, different);

    for(std::size_t i = 0; i < first.storage.size(); ++i)
        REQUIRE(std::bit_cast<std::uint32_t>(first.storage[i]) == std::bit_cast<std::uint32_t>(second.storage[i]));
    bool anyDifferent = false;
    for(std::size_t i = 0; i < first.storage.size(); ++i)
        anyDifferent = anyDifferent
                    || std::bit_cast<std::uint32_t>(first.storage[i])
                           != std::bit_cast<std::uint32_t>(different.storage[i]);
    REQUIRE(anyDifferent);
}

TEST_CASE("gemm: FP64 candidate is validated by the extended-precision reference", "[gemm][numerics]")
{
    if constexpr(!hasExtendedPrecisionReference)
    {
        SKIP("long double is not wider than double on this platform; FP64 needs a multiprecision reference");
    }
    else
    {
        Rng rng(8080);
        HostMatrix<double> a(3, 9);
        HostMatrix<double> b(9, 4);
        HostMatrix<double> cOld(3, 4);
        fillRandom(a, rng, -1.0e8f, 1.0e8f);
        fillRandom(b, rng, -1.0e-8f, 1.0e-8f);
        fillRandom(cOld, rng, -0.5f, 0.5f);
        // Severe cancellation: the result is many orders below the products.
        a.view().stored(0, 0) = 1.0e8;
        a.view().stored(0, 1) = 1.0e8;
        b.view().stored(0, 0) = 1.0;
        b.view().stored(1, 0) = -1.0 + 1.0e-9;

        HostMatrix<double> candidate = cOld;
        HostMatrix<long double> reference(cOld.rows, cOld.cols, cOld.layout);
        for(std::size_t j = 0; j < cOld.cols; ++j)
            for(std::size_t i = 0; i < cOld.rows; ++i)
                reference.view().stored(i, j) = static_cast<long double>(cOld.view().stored(i, j));
        gemm_reference_extended(reference.view(), a.view(), b.view(), 1.0, 1.0);
        gemm_reference<double>(candidate.view(), a.view(), b.view(), 1.0, 1.0);

        auto const model = makeToleranceModel(NumericMode::fp64, 9, true, true);
        auto const report
            = compareGemm(model, candidate.view(), reference.view(), a.view(), b.view(), cOld.view(), 1.0, 1.0);
        requireWithin(report, "fp64 with cancellation");
    }
}
