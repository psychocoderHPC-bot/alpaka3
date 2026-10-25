/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "../../../validation/numerics/tolerance.hpp"
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace alpaka;

namespace
{
    using validation::gemm::Layout;
    using validation::gemm::MatrixView;
    using validation::gemm::NumericMode;
    using validation::gemm::Op;
    using validation::gemm::ToleranceModel;

    constexpr Layout toReferenceLayout(onHost::GemmLayout layout)
    {
        return layout == onHost::GemmLayout::rowMajor ? Layout::rowMajor : Layout::colMajor;
    }

    constexpr Op toReferenceOp(onHost::GemmOp op)
    {
        return op == onHost::GemmOp::none ? Op::none : Op::transpose;
    }

    /** Owned host matrix with logical extents, padded leading dimension and optional subview offset. */
    template<typename T>
    struct Matrix
    {
        std::vector<T> host;
        std::size_t logicalRows = 0;
        std::size_t logicalCols = 0;
        std::size_t storedRows = 0;
        std::size_t storedCols = 0;
        std::size_t ld = 0;
        std::size_t offset = 0;
        onHost::GemmLayout layout = onHost::GemmLayout::rowMajor;
        onHost::GemmOp op = onHost::GemmOp::none;

        [[nodiscard]] T* data()
        {
            return host.data() + offset;
        }

        [[nodiscard]] T const* data() const
        {
            return host.data() + offset;
        }

        [[nodiscard]] std::size_t elements() const
        {
            if(storedRows == 0u || storedCols == 0u)
                return 0u;
            if(layout == onHost::GemmLayout::rowMajor)
                return (storedRows - 1u) * ld + storedCols;
            return (storedCols - 1u) * ld + storedRows;
        }

        [[nodiscard]] MatrixView<T> referenceView() const
        {
            return MatrixView<T>{
                const_cast<T*>(host.data() + offset),
                storedRows,
                storedCols,
                ld,
                toReferenceLayout(layout),
                toReferenceOp(op)};
        }

        [[nodiscard]] onHost::GemmView<T> deviceView(T* pointer) const
        {
            return onHost::GemmView<T>{pointer, logicalRows, logicalCols, ld, layout, op};
        }
    };

    std::mt19937_64 makeGenerator(std::size_t index, std::uint64_t seed)
    {
        std::seed_seq sequence{
            static_cast<std::uint32_t>(seed),
            static_cast<std::uint32_t>(seed >> 32u),
            static_cast<std::uint32_t>(index),
            static_cast<std::uint32_t>(index >> 32u)};
        return std::mt19937_64{sequence};
    }

    template<typename T>
    Matrix<T> makeMatrix(
        std::size_t logicalRows,
        std::size_t logicalCols,
        onHost::GemmLayout layout,
        onHost::GemmOp op,
        std::size_t padding,
        std::size_t offset,
        std::uint64_t seed)
    {
        Matrix<T> matrix;
        matrix.logicalRows = logicalRows;
        matrix.logicalCols = logicalCols;
        matrix.layout = layout;
        matrix.op = op;
        matrix.offset = offset;
        matrix.storedRows = op == onHost::GemmOp::none ? logicalRows : logicalCols;
        matrix.storedCols = op == onHost::GemmOp::none ? logicalCols : logicalRows;
        std::size_t const extent = layout == onHost::GemmLayout::rowMajor ? matrix.storedCols : matrix.storedRows;
        matrix.ld = extent + padding;
        std::size_t const elements = matrix.elements();
        matrix.host.assign(std::max<std::size_t>(elements + offset, 1u), T{});
        auto generator = makeGenerator(elements + offset + logicalRows * 131u + logicalCols, seed);
        std::uniform_real_distribution<double> distribution{-1.0, 1.0};
        for(auto& value : matrix.host)
            value = static_cast<T>(distribution(generator));
        return matrix;
    }

    struct Shape
    {
        std::size_t m;
        std::size_t n;
        std::size_t k;
    };

    struct Variant
    {
        onHost::GemmLayout layout = onHost::GemmLayout::rowMajor;
        onHost::GemmOp opA = onHost::GemmOp::none;
        onHost::GemmOp opB = onHost::GemmOp::none;
        double alpha = 1.0;
        double beta = 0.0;
    };

    /** Run one GEMM case and compare it with the frozen reference and tolerance model. */
    template<typename T, typename RefT, typename T_Device, typename T_Exec, typename T_Queue>
    validation::gemm::ComparisonReport runCase(
        T_Device const& device,
        T_Exec const& exec,
        T_Queue& queue,
        Matrix<T> const& a,
        Matrix<T> const& b,
        Matrix<T> const& cOld,
        T alpha,
        T beta,
        NumericMode mode)
    {
        std::size_t const m = cOld.logicalRows;
        std::size_t const n = cOld.logicalCols;
        std::size_t const k = a.logicalCols;

        std::vector<RefT> reference(static_cast<std::size_t>(m) * n, RefT{0});
        MatrixView<RefT> referenceView{reference.data(), m, n, n, Layout::rowMajor, Op::none};
        MatrixView<T> const cOldView = cOld.referenceView();
        for(std::size_t i = 0u; i < m; ++i)
            for(std::size_t j = 0u; j < n; ++j)
                reference[i * n + j] = static_cast<RefT>(cOldView.stored(i, j));
        if(m != 0u && n != 0u)
        {
            if constexpr(std::same_as<RefT, long double>)
                validation::gemm::gemm_reference_extended(
                    referenceView,
                    a.referenceView(),
                    b.referenceView(),
                    alpha,
                    beta);
            else
                validation::gemm::gemm_reference<RefT>(
                    referenceView,
                    a.referenceView(),
                    b.referenceView(),
                    alpha,
                    beta);
        }

        auto allocate = [&](std::size_t count)
        {
            std::size_t const size = std::max<std::size_t>(count, 1u);
            return onHost::alloc<T>(device, Vec{size});
        };

        auto dA = allocate(a.host.size());
        auto dB = allocate(b.host.size());
        auto dC = allocate(cOld.host.size());
        onHost::memcpy(queue, dA, a.host);
        onHost::memcpy(queue, dB, b.host);
        onHost::memcpy(queue, dC, cOld.host);

        auto const viewA = a.deviceView(onHost::data(dA) + a.offset);
        auto const viewB = b.deviceView(onHost::data(dB) + b.offset);
        auto const viewC = cOld.deviceView(onHost::data(dC) + cOld.offset);

        onHost::gemm(
            queue,
            exec,
            viewC,
            viewA,
            viewB,
            alpha,
            beta,
            mode == NumericMode::fp64 ? onHost::GemmNumericMode::fp64 : onHost::GemmNumericMode::fp32);

        std::vector<T> candidate(cOld.host.size());
        onHost::memcpy(queue, candidate, dC);
        onHost::wait(queue);

        auto const model
            = validation::gemm::makeToleranceModel(mode, k, alpha != T{0}, beta != T{0}, beta == T{1}, false, false);

        MatrixView<T>
            candidateView{candidate.data() + cOld.offset, m, n, cOld.ld, toReferenceLayout(cOld.layout), Op::none};
        return validation::gemm::compareGemm(
            model,
            candidateView,
            referenceView,
            a.referenceView(),
            b.referenceView(),
            cOldView,
            alpha,
            beta);
    }

    template<typename T, typename RefT, typename T_Device, typename T_Exec, typename T_Queue>
    void checkCase(
        T_Device const& device,
        T_Exec const& exec,
        T_Queue& queue,
        Shape const& shape,
        Variant const& variant,
        NumericMode mode,
        std::size_t padding,
        std::size_t offset,
        std::uint64_t seed)
    {
        CAPTURE(shape.m, shape.n, shape.k, variant.alpha, variant.beta);
        CAPTURE(static_cast<int>(variant.layout), static_cast<int>(variant.opA), static_cast<int>(variant.opB));
        CAPTURE(padding, offset);

        auto const a = makeMatrix<T>(shape.m, shape.k, variant.layout, variant.opA, padding, offset, seed + 1u);
        auto const b = makeMatrix<T>(shape.k, shape.n, variant.layout, variant.opB, padding, offset, seed + 2u);
        auto const c
            = makeMatrix<T>(shape.m, shape.n, variant.layout, onHost::GemmOp::none, padding, offset, seed + 3u);

        auto const report = runCase<T, RefT>(
            device,
            exec,
            queue,
            a,
            b,
            c,
            static_cast<T>(variant.alpha),
            static_cast<T>(variant.beta),
            mode);

        if(shape.m == 0u || shape.n == 0u)
            return;

        INFO(
            "worst scaled error " << static_cast<double>(report.maxScaledError) << " at (" << report.worstScaledI
                                  << "," << report.worstScaledJ << "), bound at worst "
                                  << static_cast<double>(report.boundAtWorstError));
        CHECK(report.informative);
        CHECK(report.withinBound);
    }

    /** Verify that a no-op case leaves the complete C buffer, including padding, byte-identical. */
    template<typename T, typename T_Device, typename T_Exec, typename T_Queue>
    void checkUnchanged(
        T_Device const& device,
        T_Exec const& exec,
        T_Queue& queue,
        Shape const& shape,
        Variant const& variant,
        NumericMode mode,
        bool makeOperandsNull)
    {
        auto const a = makeMatrix<T>(shape.m, shape.k, variant.layout, variant.opA, 0u, 0u, 11u);
        auto const b = makeMatrix<T>(shape.k, shape.n, variant.layout, variant.opB, 0u, 0u, 12u);
        auto const c = makeMatrix<T>(shape.m, shape.n, variant.layout, onHost::GemmOp::none, 0u, 0u, 13u);

        auto dA = onHost::alloc<T>(device, Vec{std::max<std::size_t>(a.host.size(), 1u)});
        auto dB = onHost::alloc<T>(device, Vec{std::max<std::size_t>(b.host.size(), 1u)});
        auto dC = onHost::alloc<T>(device, Vec{std::max<std::size_t>(c.host.size(), 1u)});
        onHost::memcpy(queue, dA, a.host);
        onHost::memcpy(queue, dB, b.host);
        onHost::memcpy(queue, dC, c.host);

        auto viewA = a.deviceView(makeOperandsNull ? nullptr : onHost::data(dA));
        auto viewB = b.deviceView(makeOperandsNull ? nullptr : onHost::data(dB));
        auto viewC = c.deviceView(onHost::data(dC));

        onHost::gemm(
            queue,
            exec,
            viewC,
            viewA,
            viewB,
            static_cast<T>(variant.alpha),
            static_cast<T>(variant.beta),
            mode == NumericMode::fp64 ? onHost::GemmNumericMode::fp64 : onHost::GemmNumericMode::fp32);

        std::vector<T> result(c.host.size());
        onHost::memcpy(queue, result, dC);
        onHost::wait(queue);
        CHECK(result == c.host);
    }

    using TestBackends
        = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

    std::vector<double> alphaBetaValues()
    {
        return {0.0, 1.0, -1.0, 0.5, 2.5, -0.25};
    }
} // namespace

TEMPLATE_LIST_TEST_CASE("gemm exhaustive small shapes", "[gemm][numerics]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    std::uint64_t seed = 0x1234u;
    for(std::size_t m = 0u; m <= 17u; ++m)
        for(std::size_t n = 0u; n <= 17u; ++n)
            for(std::size_t k = 0u; k <= 17u; ++k)
            {
                Variant variant;
                variant.alpha = (m + n + k) % 2u == 0u ? 1.0 : 0.75;
                variant.beta = (m + n + k) % 3u == 0u ? 0.0 : -0.5;
                checkCase<float, double>(
                    device,
                    exec,
                    queue,
                    Shape{m, n, k},
                    variant,
                    NumericMode::fp32,
                    0u,
                    0u,
                    seed++);
            }
}

TEMPLATE_LIST_TEST_CASE("gemm transpose layout and alpha beta matrix", "[gemm][numerics]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    std::vector<Shape> const shapes{
        {1u, 1u, 1u},
        {1u, 8u, 5u},
        {3u, 5u, 7u},
        {4u, 4u, 4u},
        {7u, 9u, 8u},
        {8u, 8u, 8u},
        {9u, 7u, 15u},
        {15u, 17u, 16u},
        {16u, 16u, 17u},
        {17u, 15u, 31u},
        {31u, 33u, 32u},
        {33u, 31u, 16u}};

    auto const values = alphaBetaValues();
    std::uint64_t seed = 0xfeedu;
    for(auto const& shape : shapes)
        for(int layout = 0; layout < 2; ++layout)
            for(int op = 0; op < 4; ++op)
                for(double alpha : values)
                    for(double beta : values)
                    {
                        Variant variant;
                        variant.layout = layout == 0 ? onHost::GemmLayout::rowMajor : onHost::GemmLayout::colMajor;
                        variant.opA = (op & 1) != 0 ? onHost::GemmOp::transpose : onHost::GemmOp::none;
                        variant.opB = (op & 2) != 0 ? onHost::GemmOp::transpose : onHost::GemmOp::none;
                        variant.alpha = alpha;
                        variant.beta = beta;
                        checkCase<float, double>(
                            device,
                            exec,
                            queue,
                            shape,
                            variant,
                            NumericMode::fp32,
                            0u,
                            0u,
                            seed++);
                    }
}

TEMPLATE_LIST_TEST_CASE("gemm padded leading dimensions and subviews", "[gemm][numerics]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    std::vector<Shape> const shapes{{3u, 5u, 7u}, {7u, 9u, 8u}, {8u, 9u, 15u}, {15u, 17u, 16u}, {17u, 15u, 17u}};

    std::uint64_t seed = 0xab'cdefu;
    for(auto const& shape : shapes)
        for(int layout = 0; layout < 2; ++layout)
            for(int op = 0; op < 4; ++op)
                for(std::size_t offset : {0u, 1u})
                {
                    Variant variant;
                    variant.layout = layout == 0 ? onHost::GemmLayout::rowMajor : onHost::GemmLayout::colMajor;
                    variant.opA = (op & 1) != 0 ? onHost::GemmOp::transpose : onHost::GemmOp::none;
                    variant.opB = (op & 2) != 0 ? onHost::GemmOp::transpose : onHost::GemmOp::none;
                    variant.alpha = 1.5;
                    variant.beta = -0.5;
                    checkCase<float, double>(
                        device,
                        exec,
                        queue,
                        shape,
                        variant,
                        NumericMode::fp32,
                        7u,
                        offset,
                        seed++);
                }
}

TEMPLATE_LIST_TEST_CASE("gemm K tile boundaries and non divisible shapes", "[gemm][numerics]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    std::vector<Shape> const shapes{
        {1u, 1u, 1u},
        {2u, 3u, 4u},
        {3u, 2u, 8u},
        {5u, 7u, 16u},
        {4u, 4u, 32u},
        {17u, 19u, 15u},
        {19u, 17u, 31u},
        {16u, 16u, 33u},
        {31u, 17u, 65u}};

    std::uint64_t seed = 0x5151u;
    for(auto const& shape : shapes)
    {
        Variant variant;
        variant.alpha = -2.0;
        variant.beta = 0.25;
        checkCase<float, double>(device, exec, queue, shape, variant, NumericMode::fp32, 0u, 0u, seed++);
    }
}

TEMPLATE_LIST_TEST_CASE("gemm degenerate size contract", "[gemm][semantics]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    // M == 0 or N == 0: no-op, no operand is dereferenced, even with null A/B and C views.
    for(Shape const shape : {Shape{0u, 5u, 3u}, Shape{5u, 0u, 3u}, Shape{0u, 0u, 0u}})
    {
        Variant variant;
        variant.alpha = 2.0;
        variant.beta = 0.5;
        checkUnchanged<float>(device, exec, queue, shape, variant, NumericMode::fp32, true);
    }

    // K == 0 or alpha == 0: C = beta * C without reading A or B. Null A/B pointers prove the contract.
    for(Shape const shape : {Shape{4u, 5u, 0u}, Shape{4u, 5u, 3u}, Shape{5u, 4u, 0u}})
    {
        for(double beta : {0.0, 1.0, 2.0, -0.5})
        {
            auto const a
                = makeMatrix<float>(shape.m, shape.k, onHost::GemmLayout::rowMajor, onHost::GemmOp::none, 0u, 0u, 21u);
            auto const b
                = makeMatrix<float>(shape.k, shape.n, onHost::GemmLayout::rowMajor, onHost::GemmOp::none, 0u, 0u, 22u);
            auto const c
                = makeMatrix<float>(shape.m, shape.n, onHost::GemmLayout::rowMajor, onHost::GemmOp::none, 0u, 0u, 23u);
            auto dC = onHost::alloc<float>(device, Vec{c.host.size()});
            onHost::memcpy(queue, dC, c.host);
            auto const viewC = c.deviceView(onHost::data(dC));
            auto const viewA = a.deviceView(nullptr);
            auto const viewB = b.deviceView(nullptr);
            float const alpha = shape.k == 0u ? 1.0f : 0.0f;
            onHost::gemm(queue, exec, viewC, viewA, viewB, alpha, static_cast<float>(beta));
            std::vector<float> result(c.host.size());
            onHost::memcpy(queue, result, dC);
            onHost::wait(queue);
            for(std::size_t i = 0u; i < c.host.size(); ++i)
            {
                CAPTURE(i, beta, shape.m, shape.n, shape.k);
                float const expected
                    = beta == 0.0 ? 0.0f : (beta == 1.0 ? c.host[i] : static_cast<float>(beta) * c.host[i]);
                CHECK(result[i] == expected);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE("gemm FP64 mode", "[gemm][numerics]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    for(Shape const shape : {Shape{1u, 1u, 1u}, Shape{3u, 5u, 7u}, Shape{8u, 9u, 16u}, Shape{17u, 15u, 15u}})
    {
        for(int op = 0; op < 4; ++op)
        {
            Variant variant;
            variant.opA = (op & 1) != 0 ? onHost::GemmOp::transpose : onHost::GemmOp::none;
            variant.opB = (op & 2) != 0 ? onHost::GemmOp::transpose : onHost::GemmOp::none;
            variant.alpha = 1.25;
            variant.beta = -0.75;
            checkCase<double, long double>(
                device,
                exec,
                queue,
                shape,
                variant,
                NumericMode::fp64,
                3u,
                1u,
                0x6464u + static_cast<std::uint64_t>(op));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("gemm host validation rejects invalid problems", "[gemm][validation]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    auto dC = onHost::alloc<float>(device, Vec{64u});
    auto dA = onHost::alloc<float>(device, Vec{64u});
    auto dB = onHost::alloc<float>(device, Vec{64u});

    auto const makeView
        = [](float* pointer,
             std::size_t rows,
             std::size_t cols,
             std::size_t ld,
             onHost::GemmLayout layout,
             onHost::GemmOp op) { return onHost::GemmView<float>{pointer, rows, cols, ld, layout, op}; };

    SECTION("leading dimension smaller than the fast extent")
    {
        auto c = makeView(onHost::data(dC), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto a = makeView(onHost::data(dA), 4u, 4u, 3u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto b = makeView(onHost::data(dB), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        CHECK_THROWS_AS(onHost::gemm(queue, exec, c, a, b, 1.0f, 0.0f), std::invalid_argument);
    }

    SECTION("transposed output is rejected")
    {
        auto c = makeView(onHost::data(dC), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::transpose);
        auto a = makeView(onHost::data(dA), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto b = makeView(onHost::data(dB), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        CHECK_THROWS_AS(onHost::gemm(queue, exec, c, a, b, 1.0f, 0.0f), std::invalid_argument);
    }

    SECTION("inconsistent dimensions are rejected")
    {
        auto c = makeView(onHost::data(dC), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto a = makeView(onHost::data(dA), 4u, 5u, 5u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto b = makeView(onHost::data(dB), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        CHECK_THROWS_AS(onHost::gemm(queue, exec, c, a, b, 1.0f, 0.0f), std::invalid_argument);
    }

    SECTION("overlapping C and A are rejected")
    {
        auto c = makeView(onHost::data(dC), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto a = makeView(onHost::data(dC) + 1, 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto b = makeView(onHost::data(dB), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        CHECK_THROWS_AS(onHost::gemm(queue, exec, c, a, b, 1.0f, 0.0f), std::invalid_argument);
    }

    SECTION("a null output pointer with a store is rejected")
    {
        auto c = makeView(nullptr, 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto a = makeView(onHost::data(dA), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto b = makeView(onHost::data(dB), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        CHECK_THROWS_AS(onHost::gemm(queue, exec, c, a, b, 1.0f, 0.0f), std::invalid_argument);
    }

    SECTION("64-bit size arithmetic overflow is rejected")
    {
        std::size_t const huge = std::numeric_limits<std::size_t>::max() / 2u;
        auto c = makeView(onHost::data(dC), 2u, 2u, huge, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto a = makeView(onHost::data(dA), 2u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto b = makeView(onHost::data(dB), 4u, 2u, 2u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        CHECK_THROWS_AS(onHost::gemm(queue, exec, c, a, b, 1.0f, 0.0f), std::overflow_error);
    }

    SECTION("FP16 and BF16 modes are rejected until their storage types exist")
    {
        auto c = makeView(onHost::data(dC), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto a = makeView(onHost::data(dA), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto b = makeView(onHost::data(dB), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        CHECK_THROWS_AS(
            onHost::gemm(queue, exec, c, a, b, 1.0f, 0.0f, onHost::GemmNumericMode::fp16ToFp32),
            std::invalid_argument);
        CHECK_THROWS_AS(
            onHost::gemm(queue, exec, c, a, b, 1.0f, 0.0f, onHost::GemmNumericMode::bf16ToFp32),
            std::invalid_argument);
    }
}
