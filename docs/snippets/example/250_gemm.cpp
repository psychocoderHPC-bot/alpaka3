/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace alpaka;

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms - gemm", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);

    std::vector<float> const hostA(16u, 1.0f);
    std::vector<float> const hostB(16u, 2.0f);
    std::vector<float> hostC(16u, 0.0f);

    auto dA = onHost::allocLike(device, hostA);
    auto dB = onHost::allocLike(device, hostB);
    auto dC = onHost::allocLike(device, hostC);
    onHost::memcpy(queue, dA, hostA);
    onHost::memcpy(queue, dB, hostB);
    onHost::memcpy(queue, dC, hostC);

    // BEGIN-TUTORIAL-gemmCall
    // A, B and C are 4x4 row-major views over existing device storage.
    constexpr auto size = std::size_t{4};
    auto const layout = onHost::GemmLayout::rowMajor;
    auto const noOp = onHost::GemmOp::none;

    auto viewA = onHost::GemmView<float>{onHost::data(dA), size, size, size, layout, noOp};
    auto viewB = onHost::GemmView<float>{onHost::data(dB), size, size, size, layout, noOp};
    auto viewC = onHost::GemmView<float>{onHost::data(dC), size, size, size, layout, noOp};

    // C = alpha * A * B + beta * C with the default executor.
    onHost::gemm(queue, viewC, viewA, viewB, 1.0f, 0.0f);
    // END-TUTORIAL-gemmCall

    onHost::memcpy(queue, hostC, dC);
    onHost::wait(queue);

    // A is all ones and B is all twos, so every element of A * B is 4 * 2.
    for(auto const value : hostC)
        CHECK(value == 8.0f);
}
