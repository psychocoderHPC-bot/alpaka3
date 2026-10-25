// SPDX-License-Identifier: MPL-2.0
//
// M2 CPU adapter for the benchmark harness: times the real common-source GEMM
// kernel instead of the M0 placeholder.

#pragma once

#include "gemm_interface.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gemm::bench
{
    /** Alpaka host-CPU GEMM functor.
     *
     * The functor owns one device, queue and the three matrix buffers. prepare() allocates and seeds them for a
     * problem; run() only enqueues the real kernel, so the harness measures queue latency plus device work rather
     * than allocation or host packing.
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
            return problem.mode == "fp32";
        }

        [[nodiscard]] std::string unsupportedReason(Problem const& problem) const override
        {
            return "M2 alpaka adapter supports fp32 only, requested " + problem.mode;
        }

        void prepare(Problem const& problem) override
        {
            m_lda = problem.lda;
            m_ldb = problem.ldb;
            m_ldc = problem.ldc;
            m_rowsA = problem.transpose_a == "transpose" ? problem.k : problem.m;
            m_colsA = problem.transpose_a == "transpose" ? problem.m : problem.k;
            m_rowsB = problem.transpose_b == "transpose" ? problem.n : problem.k;
            m_colsB = problem.transpose_b == "transpose" ? problem.k : problem.n;

            m_a = makeBuffer(m_rowsA, m_colsA, m_lda, 1u);
            m_b = makeBuffer(m_rowsB, m_colsB, m_ldb, 2u);
            m_c = makeBuffer(problem.m, problem.n, m_ldc, 3u);
            m_cOriginal = m_c;

            m_dA = allocate(m_a.size());
            m_dB = allocate(m_b.size());
            m_dC = allocate(m_c.size());
            alpaka::onHost::memcpy(m_queue, *m_dA, m_a);
            alpaka::onHost::memcpy(m_queue, *m_dB, m_b);
            alpaka::onHost::memcpy(m_queue, *m_dC, m_c);
            alpaka::onHost::wait(m_queue);
        }

        void resetOutput(Problem const& problem) override
        {
            (void) problem;
            if(!m_cOriginal.empty())
                alpaka::onHost::memcpy(m_queue, *m_dC, m_cOriginal);
        }

        RunResult run(Problem const& problem) override
        {
            auto const viewC = alpaka::onHost::GemmView<float>{
                alpaka::onHost::data(*m_dC),
                problem.m,
                problem.n,
                m_ldc,
                alpaka::onHost::GemmLayout::rowMajor,
                alpaka::onHost::GemmOp::none};
            auto const viewA = alpaka::onHost::GemmView<float>{
                alpaka::onHost::data(*m_dA),
                problem.m,
                problem.k,
                m_lda,
                alpaka::onHost::GemmLayout::rowMajor,
                problem.transpose_a == "transpose" ? alpaka::onHost::GemmOp::transpose : alpaka::onHost::GemmOp::none};
            auto const viewB = alpaka::onHost::GemmView<float>{
                alpaka::onHost::data(*m_dB),
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
                static_cast<float>(problem.alpha),
                static_cast<float>(problem.beta),
                alpaka::onHost::GemmNumericMode::fp32);
            return RunResult{true, {}};
        }

        void synchronize() override
        {
            alpaka::onHost::wait(m_queue);
        }

    private:
        using Buffer = decltype(alpaka::onHost::alloc<float>(std::declval<Device>(), alpaka::Vec{std::size_t{1u}}));

        [[nodiscard]] Buffer allocate(std::size_t elements)
        {
            return alpaka::onHost::alloc<float>(m_device, alpaka::Vec{std::max<std::size_t>(elements, 1u)});
        }

        [[nodiscard]] static std::vector<float> makeBuffer(
            std::size_t rows,
            std::size_t cols,
            std::size_t ld,
            std::uint32_t seed)
        {
            if(rows == 0u || cols == 0u)
                return std::vector<float>(1u, 0.0f);
            std::size_t const elements = (rows - 1u) * ld + cols;
            std::vector<float> buffer(elements, 0.0f);
            for(std::size_t i = 0u; i < elements; ++i)
                buffer[i] = std::sin(0.001 * static_cast<double>(i + seed));
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
        std::vector<float> m_a;
        std::vector<float> m_b;
        std::vector<float> m_c;
        std::vector<float> m_cOriginal;
        std::optional<Buffer> m_dA;
        std::optional<Buffer> m_dB;
        std::optional<Buffer> m_dC;
    };
} // namespace gemm::bench
