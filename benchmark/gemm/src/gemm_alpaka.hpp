// SPDX-License-Identifier: MPL-2.0
//
// M2 CPU adapter for the benchmark harness: times the real common-source GEMM
// kernel instead of the M0 placeholder.

#pragma once

#include "gemm_interface.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace gemm::bench
{
    namespace detail
    {
        template<typename T>
        [[nodiscard]] T storedFromFloat(float value)
        {
            if constexpr(std::same_as<T, float>)
                return value;
            else
                return T::fromFloat(value);
        }
    } // namespace detail

    /** Alpaka host-CPU GEMM functor.
     *
     * The functor owns one device, queue and the matrix buffers of the active numerical mode. prepare() allocates
     * and seeds them for a problem; run() only enqueues the real kernel, so the harness measures queue latency plus
     * device work rather than allocation or host packing.
     *
     * Strict FP32, FP16-to-FP32 and BF16-to-FP32 are supported. The mixed modes keep A and B in their 16-bit
     * storage type and C in FP32; the common kernel performs the exact conversion. FP64 is not exposed by this M2
     * adapter.
     */
    class AlpakaGemmFunctor final : public GemmFunctor
    {
    public:
        using Device = decltype(alpaka::onHost::makeHostDevice<float>());
        using Executor = decltype(alpaka::onHost::defaultExecutor(std::declval<Device>()));

        AlpakaGemmFunctor()
            : m_device{alpaka::onHost::makeHostDevice<float>()}
            , m_exec{alpaka::onHost::defaultExecutor(m_device)}
            , m_queue{m_device.makeQueue()}
        {
        }

        [[nodiscard]] std::string_view name() const override
        {
            return "alpaka-host-cpu-m2";
        }

        [[nodiscard]] bool supports(Problem const& problem) const override
        {
            return problem.mode == "fp32" || problem.mode == "fp16_fp32" || problem.mode == "bf16_fp32";
        }

        [[nodiscard]] std::string unsupportedReason(Problem const& problem) const override
        {
            return "M2 alpaka adapter supports fp32, fp16_fp32 and bf16_fp32, requested " + problem.mode;
        }

        void prepare(Problem const& problem) override
        {
            setStrides(problem);
            if(problem.mode == "fp32")
                prepareImpl(problem, m_f32.emplace());
            else if(problem.mode == "fp16_fp32")
                prepareImpl(problem, m_f16.emplace());
            else if(problem.mode == "bf16_fp32")
                prepareImpl(problem, m_bf16.emplace());
            else
                throw std::runtime_error(unsupportedReason(problem));
        }

        void resetOutput(Problem const& problem) override
        {
            if(problem.mode == "fp32")
                resetImpl(m_f32);
            else if(problem.mode == "fp16_fp32")
                resetImpl(m_f16);
            else if(problem.mode == "bf16_fp32")
                resetImpl(m_bf16);
        }

        RunResult run(Problem const& problem) override
        {
            if(problem.mode == "fp32")
                return runImpl(problem, *m_f32, alpaka::onHost::GemmNumericMode::fp32);
            if(problem.mode == "fp16_fp32")
                return runImpl(problem, *m_f16, alpaka::onHost::GemmNumericMode::fp16ToFp32);
            if(problem.mode == "bf16_fp32")
                return runImpl(problem, *m_bf16, alpaka::onHost::GemmNumericMode::bf16ToFp32);
            throw std::runtime_error(unsupportedReason(problem));
        }

        void synchronize() override
        {
            alpaka::onHost::wait(m_queue);
        }

    private:
        template<typename T>
        using DeviceBuffer
            = std::decay_t<decltype(alpaka::onHost::alloc<T>(std::declval<Device>(), alpaka::Vec{std::size_t{1u}}))>;

        template<typename T_Operand, typename T_Output>
        struct Buffers
        {
            std::vector<T_Operand> a;
            std::vector<T_Operand> b;
            std::vector<T_Output> c;
            std::vector<T_Output> cOriginal;
            std::optional<DeviceBuffer<T_Operand>> dA;
            std::optional<DeviceBuffer<T_Operand>> dB;
            std::optional<DeviceBuffer<T_Output>> dC;
        };

        void setStrides(Problem const& problem)
        {
            m_lda = problem.lda;
            m_ldb = problem.ldb;
            m_ldc = problem.ldc;
            m_rowsA = problem.transpose_a == "transpose" ? problem.k : problem.m;
            m_colsA = problem.transpose_a == "transpose" ? problem.m : problem.k;
            m_rowsB = problem.transpose_b == "transpose" ? problem.n : problem.k;
            m_colsB = problem.transpose_b == "transpose" ? problem.k : problem.n;
        }

        template<typename T_Operand, typename T_Output>
        void prepareImpl(Problem const& problem, Buffers<T_Operand, T_Output>& buffers)
        {
            buffers.a = makeBuffer<T_Operand>(m_rowsA, m_colsA, m_lda, 1u);
            buffers.b = makeBuffer<T_Operand>(m_rowsB, m_colsB, m_ldb, 2u);
            buffers.c = makeBuffer<T_Output>(problem.m, problem.n, m_ldc, 3u);
            buffers.cOriginal = buffers.c;

            buffers.dA = allocate<T_Operand>(buffers.a.size());
            buffers.dB = allocate<T_Operand>(buffers.b.size());
            buffers.dC = allocate<T_Output>(buffers.c.size());
            alpaka::onHost::memcpy(m_queue, *buffers.dA, buffers.a);
            alpaka::onHost::memcpy(m_queue, *buffers.dB, buffers.b);
            alpaka::onHost::memcpy(m_queue, *buffers.dC, buffers.c);
            alpaka::onHost::wait(m_queue);
        }

        template<typename T_Operand, typename T_Output>
        void resetImpl(std::optional<Buffers<T_Operand, T_Output>>& buffers)
        {
            if(buffers && !buffers->cOriginal.empty())
                alpaka::onHost::memcpy(m_queue, *buffers->dC, buffers->cOriginal);
        }

        template<typename T_Operand, typename T_Output>
        RunResult runImpl(
            Problem const& problem,
            Buffers<T_Operand, T_Output>& buffers,
            alpaka::onHost::GemmNumericMode mode)
        {
            auto const viewC = alpaka::onHost::GemmView<T_Output>{
                alpaka::onHost::data(*buffers.dC),
                problem.m,
                problem.n,
                m_ldc,
                alpaka::onHost::GemmLayout::rowMajor,
                alpaka::onHost::GemmOp::none};
            auto const viewA = alpaka::onHost::GemmView<T_Operand>{
                alpaka::onHost::data(*buffers.dA),
                problem.m,
                problem.k,
                m_lda,
                alpaka::onHost::GemmLayout::rowMajor,
                problem.transpose_a == "transpose" ? alpaka::onHost::GemmOp::transpose : alpaka::onHost::GemmOp::none};
            auto const viewB = alpaka::onHost::GemmView<T_Operand>{
                alpaka::onHost::data(*buffers.dB),
                problem.k,
                problem.n,
                m_ldb,
                alpaka::onHost::GemmLayout::rowMajor,
                problem.transpose_b == "transpose" ? alpaka::onHost::GemmOp::transpose : alpaka::onHost::GemmOp::none};

            alpaka::onHost::gemm(
                m_queue,
                m_exec,
                viewC,
                viewA,
                viewB,
                static_cast<T_Output>(problem.alpha),
                static_cast<T_Output>(problem.beta),
                mode);
            return RunResult{true, {}};
        }

        template<typename T>
        [[nodiscard]] DeviceBuffer<T> allocate(std::size_t elements)
        {
            return alpaka::onHost::alloc<T>(m_device, alpaka::Vec{std::max<std::size_t>(elements, 1u)});
        }

        template<typename T>
        [[nodiscard]] static std::vector<T> makeBuffer(
            std::size_t rows,
            std::size_t cols,
            std::size_t ld,
            std::uint32_t seed)
        {
            if(rows == 0u || cols == 0u)
                return std::vector<T>(1u, detail::storedFromFloat<T>(0.0f));
            std::size_t const elements = (rows - 1u) * ld + cols;
            std::vector<T> buffer(elements);
            for(std::size_t i = 0u; i < elements; ++i)
                buffer[i]
                    = detail::storedFromFloat<T>(static_cast<float>(std::sin(0.001 * static_cast<double>(i + seed))));
            return buffer;
        }

        Device m_device;
        Executor m_exec;
        decltype(m_device.makeQueue()) m_queue;
        std::size_t m_lda{};
        std::size_t m_ldb{};
        std::size_t m_ldc{};
        std::size_t m_rowsA{};
        std::size_t m_colsA{};
        std::size_t m_rowsB{};
        std::size_t m_colsB{};
        std::optional<Buffers<float, float>> m_f32;
        std::optional<Buffers<alpaka::Float16, float>> m_f16;
        std::optional<Buffers<alpaka::BFloat16, float>> m_bf16;
    };
} // namespace gemm::bench
