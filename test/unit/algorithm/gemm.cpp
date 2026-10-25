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
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace alpaka
{
    // ADL extension point of the frozen reference: let it read the common 16-bit storage types exactly without
    // modifying the validation-only reference header. The reference header documents this overload hook.
    inline double referenceValue(Float16 value) noexcept
    {
        return static_cast<double>(value.toFloat());
    }

    inline double referenceValue(BFloat16 value) noexcept
    {
        return static_cast<double>(value.toFloat());
    }
} // namespace alpaka

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
    validation::gemm::ComparisonReport checkCase(
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
            return report;

        INFO(
            "worst scaled error " << static_cast<double>(report.maxScaledError) << " at (" << report.worstScaledI
                                  << "," << report.worstScaledJ << "), bound at worst "
                                  << static_cast<double>(report.boundAtWorstError));
        CHECK(report.informative);
        CHECK(report.withinBound);
        return report;
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

    /** Copy a float matrix into a 16-bit storage matrix, preserving layout, padding and subview offset. */
    template<typename Storage>
    Matrix<Storage> quantizeMatrix(Matrix<float> const& source)
    {
        Matrix<Storage> result;
        result.logicalRows = source.logicalRows;
        result.logicalCols = source.logicalCols;
        result.storedRows = source.storedRows;
        result.storedCols = source.storedCols;
        result.ld = source.ld;
        result.offset = source.offset;
        result.layout = source.layout;
        result.op = source.op;
        result.host.resize(source.host.size());
        for(std::size_t i = 0u; i < source.host.size(); ++i)
            result.host[i] = Storage::fromFloat(source.host[i]);
        return result;
    }

    struct MixedReport
    {
        validation::gemm::ComparisonReport accumulation;
        validation::gemm::QuantizationReport quantization;
    };

    /** Run one mixed-precision case: quantized storage operands, FP32 accumulation and FP32 output.
     *
     * The accumulation error is compared against the frozen double reference fed with the quantized operands.
     * The input-quantization error is reported separately against the same reference fed with the original
     * float operands, so the two error sources are never conflated.
     */
    template<typename Storage, typename RefT, typename T_Device, typename T_Exec, typename T_Queue>
    MixedReport runMixedCase(
        T_Device const& device,
        T_Exec const& exec,
        T_Queue& queue,
        Matrix<float> const& aOriginal,
        Matrix<float> const& bOriginal,
        Matrix<float> const& cOld,
        float alpha,
        float beta,
        NumericMode mode,
        onHost::GemmNumericMode hostMode)
    {
        Matrix<Storage> const a = quantizeMatrix<Storage>(aOriginal);
        Matrix<Storage> const b = quantizeMatrix<Storage>(bOriginal);

        std::size_t const m = cOld.logicalRows;
        std::size_t const n = cOld.logicalCols;
        std::size_t const k = a.logicalCols;

        auto makeReference = [&](auto const& aView, auto const& bView)
        {
            std::vector<RefT> storage(static_cast<std::size_t>(m) * n, RefT{0});
            MatrixView<RefT> view{storage.data(), m, n, n, Layout::rowMajor, Op::none};
            auto const cView = cOld.referenceView();
            for(std::size_t i = 0u; i < m; ++i)
                for(std::size_t j = 0u; j < n; ++j)
                    view.stored(i, j) = static_cast<RefT>(cView.stored(i, j));
            if(m != 0u && n != 0u)
                validation::gemm::gemm_reference<RefT>(view, aView, bView, alpha, beta);
            return storage;
        };

        std::vector<RefT> const referenceStorage = makeReference(a.referenceView(), b.referenceView());
        std::vector<RefT> const originalStorage = makeReference(aOriginal.referenceView(), bOriginal.referenceView());
        MatrixView<RefT> const
            referenceView{const_cast<RefT*>(referenceStorage.data()), m, n, n, Layout::rowMajor, Op::none};
        MatrixView<RefT> const
            originalView{const_cast<RefT*>(originalStorage.data()), m, n, n, Layout::rowMajor, Op::none};

        auto allocate = [&](std::size_t count)
        {
            std::size_t const size = std::max<std::size_t>(count, 1u);
            return onHost::alloc<Storage>(device, Vec{size});
        };

        auto dA = allocate(a.host.size());
        auto dB = allocate(b.host.size());
        auto dC = onHost::alloc<float>(device, Vec{std::max<std::size_t>(cOld.host.size(), 1u)});
        onHost::memcpy(queue, dA, a.host);
        onHost::memcpy(queue, dB, b.host);
        onHost::memcpy(queue, dC, cOld.host);

        auto const viewA = a.deviceView(onHost::data(dA) + a.offset);
        auto const viewB = b.deviceView(onHost::data(dB) + b.offset);
        auto const viewC = cOld.deviceView(onHost::data(dC) + cOld.offset);

        onHost::gemm(queue, exec, viewC, viewA, viewB, alpha, beta, hostMode);

        std::vector<float> candidate(cOld.host.size());
        onHost::memcpy(queue, candidate, dC);
        onHost::wait(queue);

        MixedReport report;
        auto const model
            = validation::gemm::makeToleranceModel(mode, k, alpha != 0.0f, beta != 0.0f, beta == 1.0f, false, false);
        MatrixView<float> const
            candidateView{candidate.data() + cOld.offset, m, n, cOld.ld, toReferenceLayout(cOld.layout), Op::none};
        report.accumulation = validation::gemm::compareGemm(
            model,
            candidateView,
            referenceView,
            a.referenceView(),
            b.referenceView(),
            cOld.referenceView(),
            alpha,
            beta);
        if(k != 0u)
        {
            report.quantization = validation::gemm::compareInputQuantization(
                originalView,
                referenceView,
                aOriginal.referenceView(),
                a.referenceView(),
                bOriginal.referenceView(),
                b.referenceView(),
                alpha);
        }
        return report;
    }

    template<typename Storage, typename RefT, typename T_Device, typename T_Exec, typename T_Queue>
    MixedReport checkMixedCase(
        T_Device const& device,
        T_Exec const& exec,
        T_Queue& queue,
        Shape const& shape,
        Variant const& variant,
        NumericMode mode,
        onHost::GemmNumericMode hostMode,
        std::size_t padding,
        std::size_t offset,
        std::uint64_t seed)
    {
        CAPTURE(shape.m, shape.n, shape.k, variant.alpha, variant.beta);
        CAPTURE(static_cast<int>(variant.layout), static_cast<int>(variant.opA), static_cast<int>(variant.opB));
        CAPTURE(padding, offset);

        auto const a = makeMatrix<float>(shape.m, shape.k, variant.layout, variant.opA, padding, offset, seed + 1u);
        auto const b = makeMatrix<float>(shape.k, shape.n, variant.layout, variant.opB, padding, offset, seed + 2u);
        auto const c
            = makeMatrix<float>(shape.m, shape.n, variant.layout, onHost::GemmOp::none, padding, offset, seed + 3u);

        auto const report = runMixedCase<Storage, RefT>(
            device,
            exec,
            queue,
            a,
            b,
            c,
            static_cast<float>(variant.alpha),
            static_cast<float>(variant.beta),
            mode,
            hostMode);

        if(shape.m == 0u || shape.n == 0u)
            return report;

        INFO(
            "worst scaled error " << static_cast<double>(report.accumulation.maxScaledError) << " at ("
                                  << report.accumulation.worstScaledI << "," << report.accumulation.worstScaledJ
                                  << "), bound at worst "
                                  << static_cast<double>(report.accumulation.boundAtWorstError));
        CHECK(report.accumulation.informative);
        CHECK(report.accumulation.withinBound);

        if(shape.k != 0u)
        {
            INFO(
                "input quantization error " << static_cast<double>(report.quantization.maxAbsError) << " at ("
                                            << report.quantization.worstI << "," << report.quantization.worstJ
                                            << "), bound "
                                            << static_cast<double>(report.quantization.boundAtWorstError));
            CHECK(report.quantization.withinBound);
        }
        return report;
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
    long double worstScaled = 0.0L;
    for(std::size_t m = 0u; m <= 17u; ++m)
        for(std::size_t n = 0u; n <= 17u; ++n)
            for(std::size_t k = 0u; k <= 17u; ++k)
            {
                Variant variant;
                variant.alpha = (m + n + k) % 2u == 0u ? 1.0 : 0.75;
                variant.beta = (m + n + k) % 3u == 0u ? 0.0 : -0.5;
                auto const report = checkCase<float, double>(
                    device,
                    exec,
                    queue,
                    Shape{m, n, k},
                    variant,
                    NumericMode::fp32,
                    0u,
                    0u,
                    seed++);
                if(m != 0u && n != 0u)
                    worstScaled = std::max(worstScaled, report.maxScaledError);
            }
    std::cout << "[fp32] worst scaled error " << static_cast<double>(worstScaled) << "\n";
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

    long double worstScaled = 0.0L;
    for(Shape const shape : {Shape{1u, 1u, 1u}, Shape{3u, 5u, 7u}, Shape{8u, 9u, 16u}, Shape{17u, 15u, 15u}})
    {
        for(int op = 0; op < 4; ++op)
        {
            Variant variant;
            variant.opA = (op & 1) != 0 ? onHost::GemmOp::transpose : onHost::GemmOp::none;
            variant.opB = (op & 2) != 0 ? onHost::GemmOp::transpose : onHost::GemmOp::none;
            variant.alpha = 1.25;
            variant.beta = -0.75;
            auto const report = checkCase<double, long double>(
                device,
                exec,
                queue,
                shape,
                variant,
                NumericMode::fp64,
                3u,
                1u,
                0x6464u + static_cast<std::uint64_t>(op));
            worstScaled = std::max(worstScaled, report.maxScaledError);
        }
    }
    std::cout << "[fp64] worst scaled error " << static_cast<double>(worstScaled) << "\n";
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
        CHECK_THROWS_AS(onHost::gemm(queue, exec, c, a, b, 1.0f, 0.0f), std::invalid_argument);
    }

    SECTION("mixed modes reject matching storage inconsistently")
    {
        auto c = makeView(onHost::data(dC), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto a = makeView(onHost::data(dA), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        auto b = makeView(onHost::data(dB), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none);
        // Float storage with a mixed mode is a mismatch, not an implicit conversion request.
        CHECK_THROWS_AS(
            onHost::gemm(queue, exec, c, a, b, 1.0f, 0.0f, onHost::GemmNumericMode::fp16ToFp32),
            std::invalid_argument);
        CHECK_THROWS_AS(
            onHost::gemm(queue, exec, c, a, b, 1.0f, 0.0f, onHost::GemmNumericMode::bf16ToFp32),
            std::invalid_argument);
        // Strict FP32/FP64 do not silently consume a 16-bit storage type.
        auto dHalf = onHost::alloc<Float16>(device, Vec{64u});
        auto dBf = onHost::alloc<BFloat16>(device, Vec{64u});
        auto const makeHalfView = [](Float16* pointer)
        { return onHost::GemmView<Float16>{pointer, 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none}; };
        auto const makeBfView = [](BFloat16* pointer)
        {
            return onHost::GemmView<BFloat16>{pointer, 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none};
        };
        CHECK_THROWS_AS(
            onHost::gemm(queue, exec, c, makeHalfView(onHost::data(dHalf)), b, 1.0f, 0.0f),
            std::invalid_argument);
        CHECK_THROWS_AS(
            onHost::gemm(queue, exec, c, a, makeBfView(onHost::data(dBf)), 1.0f, 0.0f),
            std::invalid_argument);
        // The two 16-bit types are distinct: an FP16 mode never accepts BF16 storage, and vice versa.
        auto dOut = onHost::alloc<float>(device, Vec{64u});
        auto cFloat = onHost::GemmView<float>{
            onHost::data(dOut),
            4u,
            4u,
            4u,
            onHost::GemmLayout::rowMajor,
            onHost::GemmOp::none};
        CHECK_THROWS_AS(
            onHost::gemm(
                queue,
                exec,
                cFloat,
                makeBfView(onHost::data(dBf)),
                makeBfView(onHost::data(dBf)),
                1.0f,
                0.0f,
                onHost::GemmNumericMode::fp16ToFp32),
            std::invalid_argument);
        CHECK_THROWS_AS(
            onHost::gemm(
                queue,
                exec,
                cFloat,
                makeHalfView(onHost::data(dHalf)),
                makeHalfView(onHost::data(dHalf)),
                1.0f,
                0.0f,
                onHost::GemmNumericMode::bf16ToFp32),
            std::invalid_argument);
        // A mixed output type is rejected even when the operands match the mode.
        auto cHalf = makeHalfView(onHost::data(dHalf));
        CHECK_THROWS_AS(
            onHost::gemm(
                queue,
                exec,
                cHalf,
                makeHalfView(onHost::data(dHalf)),
                makeHalfView(onHost::data(dHalf)),
                1.0f,
                0.0f,
                onHost::GemmNumericMode::fp16ToFp32),
            std::invalid_argument);
    }
}

TEMPLATE_LIST_TEST_CASE(
    "gemm FP16/BF16 mixed precision exhaustive small shapes",
    "[gemm][numerics][fp16][bf16]",
    TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    auto const runMode = [&]<typename Storage>(NumericMode mode, onHost::GemmNumericMode hostMode, char const* label)
    {
        double worstScaled = 0.0;
        double worstQuantization = 0.0;
        std::uint64_t seed = 0x16u;
        for(std::size_t m = 0u; m <= 7u; ++m)
            for(std::size_t n = 0u; n <= 7u; ++n)
                for(std::size_t k = 0u; k <= 7u; ++k)
                {
                    Variant variant;
                    variant.alpha = (m + n + k) % 2u == 0u ? 1.0 : 0.75;
                    variant.beta = (m + n + k) % 3u == 0u ? 0.0 : -0.5;
                    auto const report = checkMixedCase<Storage, double>(
                        device,
                        exec,
                        queue,
                        Shape{m, n, k},
                        variant,
                        mode,
                        hostMode,
                        0u,
                        0u,
                        seed++);
                    if(m != 0u && n != 0u)
                    {
                        worstScaled
                            = std::max<double>(worstScaled, static_cast<double>(report.accumulation.maxScaledError));
                        if(k != 0u)
                            worstQuantization = std::max<double>(
                                worstQuantization,
                                static_cast<double>(report.quantization.maxAbsError));
                    }
                }
        std::cout << "[" << label << "] worst scaled error " << static_cast<double>(worstScaled)
                  << ", worst input quantization error " << static_cast<double>(worstQuantization) << "\n";
    };

    runMode.template operator()<Float16>(NumericMode::fp16ToFp32, onHost::GemmNumericMode::fp16ToFp32, "fp16ToFp32");
    runMode.template operator()<BFloat16>(NumericMode::bf16ToFp32, onHost::GemmNumericMode::bf16ToFp32, "bf16ToFp32");
}

TEMPLATE_LIST_TEST_CASE(
    "gemm FP16/BF16 mixed precision adversarial shapes",
    "[gemm][numerics][fp16][bf16]",
    TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    std::vector<Shape> const shapes{
        {1u, 1u, 1u},
        {1u, 8u, 5u},
        {3u, 5u, 7u},
        {5u, 7u, 16u},
        {4u, 4u, 32u},
        {17u, 19u, 15u},
        {19u, 17u, 31u},
        {16u, 16u, 33u},
        {31u, 17u, 65u}};

    auto const runMode = [&]<typename Storage>(NumericMode mode, onHost::GemmNumericMode hostMode, char const* label)
    {
        double worstScaled = 0.0;
        double worstQuantization = 0.0;
        std::uint64_t seed = 0x5a5au;
        for(auto const& shape : shapes)
            for(int layout = 0; layout < 2; ++layout)
                for(int op = 0; op < 4; ++op)
                    for(std::size_t offset : {0u, 1u})
                    {
                        Variant variant;
                        variant.layout = layout == 0 ? onHost::GemmLayout::rowMajor : onHost::GemmLayout::colMajor;
                        variant.opA = (op & 1) != 0 ? onHost::GemmOp::transpose : onHost::GemmOp::none;
                        variant.opB = (op & 2) != 0 ? onHost::GemmOp::transpose : onHost::GemmOp::none;
                        variant.alpha = 1.25;
                        variant.beta = -0.75;
                        auto const report = checkMixedCase<Storage, double>(
                            device,
                            exec,
                            queue,
                            shape,
                            variant,
                            mode,
                            hostMode,
                            3u,
                            offset,
                            seed++);
                        if(shape.m != 0u && shape.n != 0u)
                        {
                            worstScaled = std::max<double>(
                                worstScaled,
                                static_cast<double>(report.accumulation.maxScaledError));
                            if(shape.k != 0u)
                                worstQuantization = std::max<double>(
                                    worstQuantization,
                                    static_cast<double>(report.quantization.maxAbsError));
                        }
                    }
        std::cout << "[" << label << "] adversarial worst scaled error " << static_cast<double>(worstScaled)
                  << ", worst input quantization error " << static_cast<double>(worstQuantization) << "\n";
    };

    runMode.template operator()<Float16>(NumericMode::fp16ToFp32, onHost::GemmNumericMode::fp16ToFp32, "fp16ToFp32");
    runMode.template operator()<BFloat16>(NumericMode::bf16ToFp32, onHost::GemmNumericMode::bf16ToFp32, "bf16ToFp32");
}

TEMPLATE_LIST_TEST_CASE(
    "gemm FP16/BF16 stored edge values convert exactly as GEMM inputs",
    "[gemm][numerics][fp16][bf16]",
    TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    std::vector<std::uint16_t> const fp16Patterns{
        0x0000u, // +0
        0x8000u, // -0
        0x0001u, // smallest positive subnormal
        0x8001u, // smallest negative subnormal
        0x03ffu, // largest subnormal
        0x0400u, // smallest normal
        0x3bffu, // just below one
        0x3c00u, // one
        0x3c01u, // one + 2^-10
        0x7bffu, // largest finite
        0x7c00u, // +inf
        0xfc00u, // -inf
        0x7e00u, // quiet NaN
        0x7d00u, // NaN with payload
    };
    std::vector<std::uint16_t> const bf16Patterns{
        0x0000u, // +0
        0x8000u, // -0
        0x0001u, // smallest positive subnormal
        0x8001u, // smallest negative subnormal
        0x007fu, // largest subnormal
        0x0080u, // smallest normal
        0x3f7fu, // just below one
        0x3f80u, // one
        0x3f81u, // one + 2^-7
        0x7f7fu, // largest finite
        0x7f80u, // +inf
        0xff80u, // -inf
        0x7fc0u, // quiet NaN
        0x7f81u, // NaN with payload
    };

    auto const expectSameBits = [](float candidate, float expected, int caseIndex, char const* side)
    {
        CAPTURE(caseIndex, side);
        if(std::isnan(expected))
            CHECK(std::isnan(candidate));
        else
        {
            CHECK_FALSE(std::isnan(candidate));
            CHECK(std::bit_cast<std::uint32_t>(candidate) == std::bit_cast<std::uint32_t>(expected));
        }
    };

    // A-side: C = A * 1, so every output row carries the converted stored value.
    auto const checkASide = [&]<typename Storage>(
                                std::vector<std::uint16_t> const& patterns,
                                onHost::GemmNumericMode hostMode,
                                char const* label)
    {
        std::size_t const count = patterns.size();
        std::vector<Storage> aHost(count);
        for(std::size_t i = 0u; i < count; ++i)
            aHost[i] = Storage::fromBits(patterns[i]);
        std::vector<Storage> const bHost(1u, Storage::fromFloat(1.0f));
        std::vector<float> const cHost(count, 0.0f);

        auto dA = onHost::alloc<Storage>(device, Vec{count});
        auto dB = onHost::alloc<Storage>(device, Vec{1u});
        auto dC = onHost::alloc<float>(device, Vec{count});
        onHost::memcpy(queue, dA, aHost);
        onHost::memcpy(queue, dB, bHost);
        onHost::memcpy(queue, dC, cHost);

        onHost::GemmView<Storage> const
            viewA{onHost::data(dA), count, 1u, 1u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none};
        onHost::GemmView<Storage> const
            viewB{onHost::data(dB), 1u, 1u, 1u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none};
        onHost::GemmView<float> const
            viewC{onHost::data(dC), count, 1u, 1u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none};
        onHost::gemm(queue, exec, viewC, viewA, viewB, 1.0f, 0.0f, hostMode);

        std::vector<float> result(count);
        onHost::memcpy(queue, result, dC);
        onHost::wait(queue);
        for(std::size_t i = 0u; i < count; ++i)
        {
            INFO(label);
            expectSameBits(result[i], 0.0f + aHost[i].toFloat() * 1.0f, static_cast<int>(i), "A");
        }
    };

    // B-side: C = 1 * B, so every output column carries the converted stored value.
    auto const checkBSide = [&]<typename Storage>(
                                std::vector<std::uint16_t> const& patterns,
                                onHost::GemmNumericMode hostMode,
                                char const* label)
    {
        std::size_t const count = patterns.size();
        std::vector<Storage> bHost(count);
        for(std::size_t i = 0u; i < count; ++i)
            bHost[i] = Storage::fromBits(patterns[i]);
        std::vector<Storage> const aHost(1u, Storage::fromFloat(1.0f));
        std::vector<float> const cHost(count, 0.0f);

        auto dA = onHost::alloc<Storage>(device, Vec{1u});
        auto dB = onHost::alloc<Storage>(device, Vec{count});
        auto dC = onHost::alloc<float>(device, Vec{count});
        onHost::memcpy(queue, dA, aHost);
        onHost::memcpy(queue, dB, bHost);
        onHost::memcpy(queue, dC, cHost);

        onHost::GemmView<Storage> const
            viewA{onHost::data(dA), 1u, 1u, 1u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none};
        onHost::GemmView<Storage> const
            viewB{onHost::data(dB), 1u, count, count, onHost::GemmLayout::rowMajor, onHost::GemmOp::none};
        onHost::GemmView<float> const
            viewC{onHost::data(dC), 1u, count, count, onHost::GemmLayout::rowMajor, onHost::GemmOp::none};
        onHost::gemm(queue, exec, viewC, viewA, viewB, 1.0f, 0.0f, hostMode);

        std::vector<float> result(count);
        onHost::memcpy(queue, result, dC);
        onHost::wait(queue);
        for(std::size_t i = 0u; i < count; ++i)
        {
            INFO(label);
            expectSameBits(result[i], 0.0f + bHost[i].toFloat() * 1.0f, static_cast<int>(i), "B");
        }
    };

    checkASide.template operator()<Float16>(fp16Patterns, onHost::GemmNumericMode::fp16ToFp32, "fp16 A");
    checkASide.template operator()<BFloat16>(bf16Patterns, onHost::GemmNumericMode::bf16ToFp32, "bf16 A");
    checkBSide.template operator()<Float16>(fp16Patterns, onHost::GemmNumericMode::fp16ToFp32, "fp16 B");
    checkBSide.template operator()<BFloat16>(bf16Patterns, onHost::GemmNumericMode::bf16ToFp32, "bf16 B");
}
