/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace alpaka;

namespace
{
    using SerialBackends
        = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, std::make_tuple(exec::cpuSerial)))>;

} // namespace

TEMPLATE_LIST_TEST_CASE(
    "gemm cooperative staging rejects an unsupported stage count",
    "[gemm][staging]",
    SerialBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    auto dA = onHost::alloc<float>(device, Vec{16u});
    auto dB = onHost::alloc<float>(device, Vec{16u});
    auto dC = onHost::alloc<float>(device, Vec{16u});
    CHECK_THROWS_AS(
        onHost::gemm(
            queue,
            exec,
            onHost::GemmView{onHost::data(dC), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none},
            onHost::GemmView{onHost::data(dA), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none},
            onHost::GemmView{onHost::data(dB), 4u, 4u, 4u, onHost::GemmLayout::rowMajor, onHost::GemmOp::none},
            1.0f,
            0.0f,
            onHost::GemmNumericMode::fp32,
            onHost::GemmKernel::registerBlocked,
            onHost::GemmStagingConfig{onHost::GemmStaging::cooperative, 4u}),
        std::invalid_argument);
}

namespace
{

    /** Dense row-major GEMM reference in double precision. */
    std::vector<double> referenceGemm(
        std::vector<float> const& a,
        std::vector<float> const& b,
        std::vector<float> const& c,
        std::size_t m,
        std::size_t n,
        std::size_t k,
        float alpha,
        float beta)
    {
        std::vector<double> result(c.begin(), c.end());
        for(std::size_t i = 0u; i < m; ++i)
            for(std::size_t j = 0u; j < n; ++j)
            {
                double sum = 0.0;
                for(std::size_t kk = 0u; kk < k; ++kk)
                    sum += static_cast<double>(a[i * k + kk]) * static_cast<double>(b[kk * n + j]);
                result[i * n + j] = static_cast<double>(alpha) * sum + static_cast<double>(beta) * result[i * n + j];
            }
        return result;
    }
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "gemm cooperative staging completes on the sequential executor",
    "[gemm][staging][progress]",
    SerialBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    // Partial M/N tiles and a partial K tile force both a multi-stage ring and boundary zero fill.
    std::size_t const m = 37u;
    std::size_t const n = 23u;
    std::size_t const k = 67u;
    float const alpha = 1.5f;
    float const beta = -0.5f;

    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c(m * n);
    for(std::size_t i = 0u; i < a.size(); ++i)
        a[i] = static_cast<float>(static_cast<int>(i % 29u) - 14) / 7.0f;
    for(std::size_t i = 0u; i < b.size(); ++i)
        b[i] = static_cast<float>(static_cast<int>(i % 31u) - 15) / 9.0f;
    for(std::size_t i = 0u; i < c.size(); ++i)
        c[i] = static_cast<float>(static_cast<int>(i % 13u) - 6) / 5.0f;

    std::vector<double> const reference = referenceGemm(a, b, c, m, n, k, alpha, beta);

    for(std::uint32_t stages : {1u, 2u, 3u})
    {
        auto dA = onHost::alloc<float>(device, Vec{a.size()});
        auto dB = onHost::alloc<float>(device, Vec{b.size()});
        auto dC = onHost::alloc<float>(device, Vec{c.size()});
        onHost::memcpy(queue, dA, a);
        onHost::memcpy(queue, dB, b);
        onHost::memcpy(queue, dC, c);

        onHost::gemm(
            queue,
            exec,
            onHost::GemmView{onHost::data(dC), m, n, n, onHost::GemmLayout::rowMajor, onHost::GemmOp::none},
            onHost::GemmView{onHost::data(dA), m, k, k, onHost::GemmLayout::rowMajor, onHost::GemmOp::none},
            onHost::GemmView{onHost::data(dB), k, n, n, onHost::GemmLayout::rowMajor, onHost::GemmOp::none},
            alpha,
            beta,
            onHost::GemmNumericMode::fp32,
            onHost::GemmKernel::registerBlocked,
            onHost::GemmStagingConfig{onHost::GemmStaging::cooperative, stages});

        std::vector<float> candidate(c.size());
        onHost::memcpy(queue, candidate, dC);
        onHost::wait(queue);

        double worst = 0.0;
        for(std::size_t i = 0u; i < candidate.size(); ++i)
            worst = std::max(worst, std::abs(static_cast<double>(candidate[i]) - reference[i]));
        INFO("stage count " << stages << " worst absolute error " << worst);
        CHECK(worst < 1.0e-4);
    }
    std::cout << "[staging] sequential executor progress cases 3\n";
}
