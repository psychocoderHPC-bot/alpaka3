/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * Multi-device correctness tests.
 *
 * alpaka3 keeps every queue bound to the device that created it. These tests
 * exercise more than one device at a time so that a wrong device or queue
 * binding is observable as a wrong result:
 *
 * - per-device vector add, one queue per device
 * - cross-device isolation with all devices allocated at once
 * - interleaved submission on all queues, waiting only at the end
 * - host -> device -> host roundtrip per device
 * - repeated allocation/reuse per device
 * - copy between two device buffers on different devices
 * - cross-device event synchronization
 * - submission from one std::thread per device
 *
 * Tests that need more than one device call SKIP() when the backend provides
 * fewer than two, so the suite passes on single-GPU and CPU-only hosts.
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <thread>
#include <vector>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

using Data = std::uint32_t;
using IdxVec = Vec<std::size_t, 1u>;

constexpr std::uint32_t numElements = 1u << 18; // 256 Ki elements, 1 MiB per buffer
constexpr std::uint32_t chunkSize = 256u;

//! C = A + B
//! Defined at namespace scope: an internal-linkage kernel triggers -Wattributes
//! under NVCC, which the pedantic CI build treats as an error.
struct VectorAddKernel
{
    ALPAKA_FN_ACC auto operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto const a,
        alpaka::concepts::IMdSpan auto const b,
        alpaka::concepts::IMdSpan auto const c) const -> void
    {
        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        simdGrid.concurrent(
            acc,
            a.getExtents(),
            [](auto const&, auto&& simdA, auto&& simdB, auto&& simdC) constexpr
            { simdC = simdA.load() + simdB.load(); },
            a,
            b,
            c);
    }
};

/** Build a selector for the backend under test, skipping when it has no device.
 *
 * @return the selector; SKIP() throws, so the caller never sees it when no
 *         device is available.
 */
inline auto makeSelectorOrSkip(auto const& cfg)
{
    auto deviceSpec = onHost::makeDeviceSpec(cfg);
    auto selector = onHost::makeDeviceSelector(deviceSpec);
    UNSCOPED_INFO("DeviceSpec: " << deviceSpec.getName());
    UNSCOPED_INFO("API: " << deviceSpec.getApi().getName());
    if(!selector.isAvailable())
        SKIP("No device available for " << deviceSpec.getName());
    UNSCOPED_INFO("Device count: " << selector.getDeviceCount());
    return selector;
}

inline auto frameSpecFor(IdxVec const& extent, auto const& queue, auto const& exec)
{
    std::uint32_t const elementsPerWorker = getNumElemPerThread<Data>(queue);
    return onHost::FrameSpec{
        divCeil(extent, IdxVec{chunkSize} * static_cast<std::size_t>(elementsPerWorker)),
        IdxVec{chunkSize},
        exec};
}

TEMPLATE_LIST_TEST_CASE("multiDevice per-device vector add and roundtrip", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = makeSelectorOrSkip(cfg);
    auto exec = alpaka::getExecutor(cfg);
    IdxVec const extent{numElements};

    for(std::uint32_t i = 0; i < selector.getDeviceCount(); ++i)
    {
        auto dev = selector.makeDevice(i);
        CAPTURE(i, dev.getName());
        auto queue = dev.makeQueue();

        auto hostA = onHost::allocHost<Data>(extent);
        auto hostB = onHost::allocHost<Data>(extent);
        auto hostC = onHost::allocHost<Data>(extent);
        for(std::size_t j = 0; j < numElements; ++j)
        {
            hostA[j] = static_cast<Data>(1000u * (i + 1u) + (j % 7u) + 1u);
            hostB[j] = static_cast<Data>((j % 5u) + 1u);
            hostC[j] = 0u;
        }

        auto devA = onHost::allocLike(dev, hostA);
        auto devB = onHost::allocLike(dev, hostB);
        auto devC = onHost::allocLike(dev, hostC);

        onHost::memcpy(queue, devA, hostA);
        onHost::memcpy(queue, devB, hostB);
        onHost::memset(queue, devC, uint8_t{0});
        queue.enqueue(frameSpecFor(extent, queue, exec), KernelBundle{VectorAddKernel{}, devA, devB, devC});
        onHost::memcpy(queue, hostC, devC);
        onHost::wait(queue);

        for(std::size_t j = 0; j < numElements; ++j)
        {
            Data const expected = static_cast<Data>(hostA[j] + hostB[j]);
            INFO("device " << i << " element " << j);
            REQUIRE(hostC[j] == expected);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("multiDevice host to device to host roundtrip", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = makeSelectorOrSkip(cfg);
    IdxVec const extent{numElements};

    for(std::uint32_t i = 0; i < selector.getDeviceCount(); ++i)
    {
        auto dev = selector.makeDevice(i);
        auto queue = dev.makeQueue();

        auto hostIn = onHost::allocHost<Data>(extent);
        auto hostOut = onHost::allocHost<Data>(extent);
        for(std::size_t j = 0; j < numElements; ++j)
        {
            hostIn[j] = static_cast<Data>(0x9e37'79b9u * static_cast<std::uint32_t>(j + 1u) + 7u * i);
            hostOut[j] = 0u;
        }

        auto devBuf = onHost::allocLike(dev, hostIn);
        onHost::memcpy(queue, devBuf, hostIn);
        onHost::memcpy(queue, hostOut, devBuf);
        onHost::wait(queue);

        for(std::size_t j = 0; j < numElements; ++j)
        {
            INFO("device " << i << " element " << j);
            REQUIRE(hostOut[j] == hostIn[j]);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("multiDevice repeated allocation and reuse", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = makeSelectorOrSkip(cfg);
    auto exec = alpaka::getExecutor(cfg);
    IdxVec const extent{4096u};
    constexpr std::uint32_t iterations = 50u;

    for(std::uint32_t i = 0; i < selector.getDeviceCount(); ++i)
    {
        auto dev = selector.makeDevice(i);
        auto queue = dev.makeQueue();
        for(std::uint32_t k = 0; k < iterations; ++k)
        {
            Data const value = static_cast<Data>(k + 1u);
            auto hostIn = onHost::allocHost<Data>(extent);
            auto hostOut = onHost::allocHost<Data>(extent);
            for(std::size_t j = 0; j < extent.x(); ++j)
            {
                hostIn[j] = value;
                hostOut[j] = 0u;
            }
            auto devIn = onHost::allocLike(dev, hostIn);
            auto devOut = onHost::allocLike(dev, hostOut);
            onHost::memcpy(queue, devIn, hostIn);
            onHost::memset(queue, devOut, uint8_t{0});
            // out = in + in
            queue.enqueue(frameSpecFor(extent, queue, exec), KernelBundle{VectorAddKernel{}, devIn, devIn, devOut});
            onHost::memcpy(queue, hostOut, devOut);
            onHost::wait(queue);

            Data const expected = static_cast<Data>(2u * value);
            for(std::size_t j = 0; j < extent.x(); ++j)
            {
                INFO("device " << i << " iteration " << k << " element " << j);
                REQUIRE(hostOut[j] == expected);
            }
        }
    }
}

TEMPLATE_LIST_TEST_CASE("multiDevice cross-device isolation", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = makeSelectorOrSkip(cfg);
    if(selector.getDeviceCount() < 2)
        SKIP("Requires at least two devices, found " << selector.getDeviceCount());
    auto exec = alpaka::getExecutor(cfg);
    IdxVec const extent{numElements};

    std::vector<decltype(selector.makeDevice(0))> devices;
    std::vector<decltype(selector.makeDevice(0).makeQueue())> queues;
    std::vector<decltype(onHost::allocHost<Data>(extent))> hostIn;
    std::vector<decltype(onHost::allocHost<Data>(extent))> hostOut;
    std::vector<decltype(onHost::allocLike(selector.makeDevice(0), onHost::allocHost<Data>(extent)))> devIn;
    std::vector<decltype(onHost::allocLike(selector.makeDevice(0), onHost::allocHost<Data>(extent)))> devOut;

    for(std::uint32_t i = 0; i < selector.getDeviceCount(); ++i)
    {
        devices.push_back(selector.makeDevice(i));
        queues.push_back(devices.back().makeQueue());
        hostIn.push_back(onHost::allocHost<Data>(extent));
        hostOut.push_back(onHost::allocHost<Data>(extent));
        devIn.push_back(onHost::allocLike(devices.back(), hostIn.back()));
        devOut.push_back(onHost::allocLike(devices.back(), hostOut.back()));

        Data const constant = static_cast<Data>(100u + i * 37u);
        for(std::size_t j = 0; j < numElements; ++j)
        {
            hostIn.back()[j] = constant;
            hostOut.back()[j] = 0u;
        }
        onHost::memcpy(queues.back(), devIn.back(), hostIn.back());
        onHost::memset(queues.back(), devOut.back(), uint8_t{0});
    }

    // Submit on every queue before waiting, so all devices are in flight.
    for(std::uint32_t i = 0; i < selector.getDeviceCount(); ++i)
        queues[i].enqueue(
            frameSpecFor(extent, queues[i], exec),
            KernelBundle{VectorAddKernel{}, devIn[i], devIn[i], devOut[i]});
    for(std::uint32_t i = 0; i < selector.getDeviceCount(); ++i)
    {
        onHost::memcpy(queues[i], hostOut[i], devOut[i]);
        onHost::wait(queues[i]);
    }

    for(std::uint32_t i = 0; i < selector.getDeviceCount(); ++i)
    {
        Data const constant = static_cast<Data>(100u + i * 37u);
        Data const expected = static_cast<Data>(constant + constant);
        for(std::size_t j = 0; j < numElements; ++j)
        {
            INFO("device " << i << " element " << j);
            REQUIRE(hostOut[i][j] == expected);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("multiDevice interleaved concurrent submission", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = makeSelectorOrSkip(cfg);
    if(selector.getDeviceCount() < 2)
        SKIP("Requires at least two devices, found " << selector.getDeviceCount());
    auto exec = alpaka::getExecutor(cfg);
    IdxVec const extent{numElements};

    std::vector<decltype(selector.makeDevice(0))> devices;
    std::vector<decltype(selector.makeDevice(0).makeQueue())> queues;
    std::vector<decltype(onHost::allocHost<Data>(extent))> hostA;
    std::vector<decltype(onHost::allocHost<Data>(extent))> hostB;
    std::vector<decltype(onHost::allocHost<Data>(extent))> hostOut;
    std::vector<decltype(onHost::allocLike(selector.makeDevice(0), onHost::allocHost<Data>(extent)))> devA;
    std::vector<decltype(onHost::allocLike(selector.makeDevice(0), onHost::allocHost<Data>(extent)))> devB;
    std::vector<decltype(onHost::allocLike(selector.makeDevice(0), onHost::allocHost<Data>(extent)))> devOut;

    for(std::uint32_t i = 0; i < selector.getDeviceCount(); ++i)
    {
        devices.push_back(selector.makeDevice(i));
        queues.push_back(devices.back().makeQueue());
        hostA.push_back(onHost::allocHost<Data>(extent));
        hostB.push_back(onHost::allocHost<Data>(extent));
        hostOut.push_back(onHost::allocHost<Data>(extent));
        devA.push_back(onHost::allocLike(devices.back(), hostA.back()));
        devB.push_back(onHost::allocLike(devices.back(), hostB.back()));
        devOut.push_back(onHost::allocLike(devices.back(), hostOut.back()));

        Data const a = static_cast<Data>(i + 1u);
        Data const b = static_cast<Data>(2u * (i + 1u));
        for(std::size_t j = 0; j < numElements; ++j)
        {
            hostA.back()[j] = a;
            hostB.back()[j] = b;
            hostOut.back()[j] = 0u;
        }
        onHost::memcpy(queues.back(), devA.back(), hostA.back());
        onHost::memcpy(queues.back(), devB.back(), hostB.back());
        onHost::memset(queues.back(), devOut.back(), uint8_t{0});
    }

    for(std::uint32_t i = 0; i < selector.getDeviceCount(); ++i)
        queues[i].enqueue(
            frameSpecFor(extent, queues[i], exec),
            KernelBundle{VectorAddKernel{}, devA[i], devB[i], devOut[i]});
    for(std::uint32_t i = 0; i < selector.getDeviceCount(); ++i)
    {
        onHost::memcpy(queues[i], hostOut[i], devOut[i]);
        onHost::wait(queues[i]);
    }

    for(std::uint32_t i = 0; i < selector.getDeviceCount(); ++i)
    {
        Data const expected = static_cast<Data>(3u * (i + 1u));
        for(std::size_t j = 0; j < numElements; ++j)
        {
            INFO("device " << i << " element " << j);
            REQUIRE(hostOut[i][j] == expected);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("multiDevice copy between two device buffers", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = makeSelectorOrSkip(cfg);
    if(selector.getDeviceCount() < 2)
        SKIP("Requires at least two devices, found " << selector.getDeviceCount());
    IdxVec const extent{numElements};

    auto devSrc = selector.makeDevice(0);
    auto devDst = selector.makeDevice(1);
    auto queueSrc = devSrc.makeQueue();
    auto queueDst = devDst.makeQueue();

    auto hostIn = onHost::allocHost<Data>(extent);
    auto hostOut = onHost::allocHost<Data>(extent);
    for(std::size_t j = 0; j < numElements; ++j)
    {
        hostIn[j] = static_cast<Data>(0xdead'beefu * static_cast<std::uint32_t>(j + 1u) + 3u);
        hostOut[j] = 0u;
    }

    auto devSrcBuf = onHost::allocLike(devSrc, hostIn);
    auto devDstBuf = onHost::allocLike(devDst, hostIn);

    onHost::memcpy(queueSrc, devSrcBuf, hostIn);
    onHost::memset(queueDst, devDstBuf, uint8_t{0});
    onHost::wait(queueSrc);
    onHost::wait(queueDst);

    // Copy a device buffer on device 0 into a device buffer on device 1. The
    // copy is issued on the destination queue; whether it uses peer access or
    // is rejected when the devices cannot reach each other is backend-defined.
    onHost::memcpy(queueDst, devDstBuf, devSrcBuf);
    onHost::memcpy(queueDst, hostOut, devDstBuf);
    onHost::wait(queueDst);

    for(std::size_t j = 0; j < numElements; ++j)
    {
        INFO("element " << j);
        REQUIRE(hostOut[j] == hostIn[j]);
    }
}

TEMPLATE_LIST_TEST_CASE("multiDevice cross-device event synchronization", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = makeSelectorOrSkip(cfg);
    if(selector.getDeviceCount() < 2)
        SKIP("Requires at least two devices, found " << selector.getDeviceCount());
    auto exec = alpaka::getExecutor(cfg);
    IdxVec const extent{numElements};

    auto devActive = selector.makeDevice(0);
    auto devWaiting = selector.makeDevice(1);
    auto queueActive = devActive.makeQueue();
    auto queueWaiting = devWaiting.makeQueue();

    auto hostIn = onHost::allocHost<Data>(extent);
    auto hostOut = onHost::allocHost<Data>(extent);
    for(std::size_t j = 0; j < numElements; ++j)
    {
        hostIn[j] = static_cast<Data>(j % 251u);
        hostOut[j] = 0u;
    }

    auto devA = onHost::allocLike(devActive, hostIn);
    auto devB = onHost::allocLike(devActive, hostIn);
    auto devOut = onHost::allocLike(devActive, hostOut);
    // Consumed by the waiting device, so the event dependency is observable.
    auto devWaitingOut = onHost::allocLike(devWaiting, hostOut);

    auto event = devActive.makeEvent(timing::disabled);

    onHost::memcpy(queueActive, devA, hostIn);
    onHost::memcpy(queueActive, devB, hostIn);
    onHost::memset(queueActive, devOut, uint8_t{0});
    queueActive.enqueue(frameSpecFor(extent, queueActive, exec), KernelBundle{VectorAddKernel{}, devA, devB, devOut});
    queueActive.enqueue(event); // signal after the kernel
    queueWaiting.waitFor(event); // device 1 waits for device 0

    // Device 1 must observe the device 0 result only after the event fired.
    // These copies run on the waiting device's queue and read the active
    // device's output; without the waitFor dependency the read races the kernel.
    onHost::memcpy(queueWaiting, devWaitingOut, devOut);
    onHost::memcpy(queueWaiting, hostOut, devWaitingOut);
    onHost::wait(queueWaiting);

    for(std::size_t j = 0; j < numElements; ++j)
    {
        Data const expected = static_cast<Data>(2u * (j % 251u));
        INFO("element " << j);
        REQUIRE(hostOut[j] == expected);
    }
}

TEMPLATE_LIST_TEST_CASE("multiDevice submission from multiple host threads", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto selector = makeSelectorOrSkip(cfg);
    if(selector.getDeviceCount() < 2)
        SKIP("Requires at least two devices, found " << selector.getDeviceCount());
    auto exec = alpaka::getExecutor(cfg);
    IdxVec const extent{4096u};

    std::uint32_t const deviceCount = selector.getDeviceCount();
    std::vector<int> result(deviceCount, -1);
    std::vector<std::string> message(deviceCount);
    std::vector<std::thread> threads;

    for(std::uint32_t i = 0; i < deviceCount; ++i)
    {
        threads.emplace_back(
            [&, i]
            {
                try
                {
                    auto dev = selector.makeDevice(i);
                    auto queue = dev.makeQueue();
                    auto hostIn = onHost::allocHost<Data>(extent);
                    auto hostOut = onHost::allocHost<Data>(extent);
                    Data const value = static_cast<Data>(7u * i + 3u);
                    for(std::size_t j = 0; j < extent.x(); ++j)
                    {
                        hostIn[j] = value;
                        hostOut[j] = 0u;
                    }
                    auto devIn = onHost::allocLike(dev, hostIn);
                    auto devOut = onHost::allocLike(dev, hostOut);
                    onHost::memcpy(queue, devIn, hostIn);
                    onHost::memset(queue, devOut, uint8_t{0});
                    queue.enqueue(
                        frameSpecFor(extent, queue, exec),
                        KernelBundle{VectorAddKernel{}, devIn, devIn, devOut});
                    onHost::memcpy(queue, hostOut, devOut);
                    onHost::wait(queue);

                    Data const expected = static_cast<Data>(2u * value);
                    for(std::size_t j = 0; j < extent.x(); ++j)
                    {
                        if(hostOut[j] != expected)
                        {
                            result[i] = 1;
                            message[i] = "device " + std::to_string(i) + " element " + std::to_string(j);
                            return;
                        }
                    }
                    result[i] = 0;
                }
                catch(std::exception const& e)
                {
                    result[i] = 1;
                    message[i] = std::string("exception: ") + e.what();
                }
            });
    }
    for(auto& thread : threads)
        thread.join();

    for(std::uint32_t i = 0; i < deviceCount; ++i)
    {
        INFO("thread for device " << i << ": " << message[i]);
        REQUIRE(result[i] == 0);
    }
}
