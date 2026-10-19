/* Copyright 2026 alpaka contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

using namespace alpaka;

namespace
{
    // A timeout here means the operation is correctly blocked by our gate.
    // CTest's separate timeout detects failure to finish after releasing it.
    // The started promises precede API entry; this delay only encourages the interleaving.
    constexpr auto blockingInterval = std::chrono::milliseconds{100};
    // Generous upper bound so a regression surfaces as a failed check instead of a hung test process.
    constexpr auto crossQueueWatchdog = std::chrono::milliseconds{5000};
    constexpr auto startupTimeout = std::chrono::milliseconds{2000};

    template<typename T_Device>
    struct CrossQueueWaitScenario
    {
        using BlockingQueue = decltype(std::declval<T_Device&>().makeQueue(queueKind::blocking));
        using NonBlockingQueue = decltype(std::declval<T_Device&>().makeQueue(queueKind::nonBlocking));
        using Event = decltype(std::declval<T_Device&>().makeEvent());

        T_Device device;
        BlockingQueue waitingQueue;
        BlockingQueue producingQueue;
        NonBlockingQueue recordingQueue;
        Event event;

        std::promise<void> releaseEventTask;
        std::promise<void> waitEntered;

        CrossQueueWaitScenario()
            : device(onHost::makeHostDevice())
            , waitingQueue(device.makeQueue(queueKind::blocking))
            , producingQueue(device.makeQueue(queueKind::blocking))
            , recordingQueue(device.makeQueue(queueKind::nonBlocking))
            , event(device.makeEvent())
        {
        }
    };

    /** Release a gate on destruction so a failing check cannot leave a thread blocked forever. */
    struct PromiseGate
    {
        std::promise<void>& promise;
        bool released = false;

        ~PromiseGate()
        {
            release();
        }

        void release()
        {
            if(!released)
            {
                released = true;
                promise.set_value();
            }
        }
    };
} // namespace

TEST_CASE("host blocking event wait allows concurrent re-recording", "[queue]")
{
    // Producer: gate -> E1. Consumer: wait(E1) -> E2, using the same event object.
    // Re-recording must leave the existing wait dependent on E1, not E2.
    auto device = onHost::makeHostDevice();
    auto producer = device.makeQueue(queueKind::nonBlocking);
    auto consumer = device.makeQueue(queueKind::blocking);
    auto event = device.makeEvent();

    std::promise<void> releaseProducer;
    auto producerGate = releaseProducer.get_future().share();
    producer.enqueueHostFn([producerGate] { producerGate.wait(); });
    producer.enqueue(event);

    std::promise<void> waitStarted;
    auto waiter = std::async(
        std::launch::async,
        [&]
        {
            waitStarted.set_value();
            consumer.waitFor(event);
        });
    waitStarted.get_future().wait();
    auto const waitStatus = waiter.wait_for(blockingInterval);

    std::promise<void> recordStarted;
    auto recorder = std::async(
        std::launch::async,
        [&]
        {
            recordStarted.set_value();
            consumer.enqueue(event);
        });
    recordStarted.get_future().wait();
    recorder.wait_for(blockingInterval);

    releaseProducer.set_value();
    // Releasing the gate must allow E1 -> consumer wait -> E2 to finish.
    // Regression: recorder holds the event lock while waiting for the consumer
    // lock; the consumer holds its lock while E1 needs the event lock to finish.
    waiter.get();
    recorder.get();
    onHost::wait(producer);
    CHECK(waitStatus == std::future_status::timeout);
    CHECK(event.isComplete());
}

TEST_CASE("host blocking event wait keeps the queue nonempty", "[queue]")
{
    // Producer: gate -> event. Consumer: wait(event), with no other work.
    // An unresolved dependency still counts as unfinished queue work. If waitFor
    // bypasses the busy flag, isEmpty() and wait(queue) incorrectly report completion.
    auto device = onHost::makeHostDevice();
    auto producer = device.makeQueue(queueKind::nonBlocking);
    auto consumer = device.makeQueue(queueKind::blocking);
    auto event = device.makeEvent();

    std::promise<void> releaseProducer;
    auto producerGate = releaseProducer.get_future().share();
    producer.enqueueHostFn([producerGate] { producerGate.wait(); });
    producer.enqueue(event);

    std::promise<void> waitStarted;
    auto eventWaiter = std::async(
        std::launch::async,
        [&]
        {
            waitStarted.set_value();
            consumer.waitFor(event);
        });
    waitStarted.get_future().wait();
    auto const eventWaitStatus = eventWaiter.wait_for(blockingInterval);
    bool const emptyWhileWaiting = consumer.isEmpty();

    // Queue synchronization must stay blocked for the same dependency.
    auto queueWaiter = std::async(std::launch::async, [&] { onHost::wait(consumer); });
    auto const queueWaitStatus = queueWaiter.wait_for(blockingInterval);
    releaseProducer.set_value();
    eventWaiter.get();
    queueWaiter.get();
    onHost::wait(producer);

    CHECK(eventWaitStatus == std::future_status::timeout);
    CHECK_FALSE(emptyWhileWaiting);
    CHECK(queueWaitStatus == std::future_status::timeout);
}

TEST_CASE("host queue wait includes callback destruction", "[queue]")
{
    // Queue completion must not be published before the callback captures are destroyed.
    // Hold the destructor open after the callback body has already returned.
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::nonBlocking);

    struct Cleanup
    {
        std::promise<void>& started;
        std::shared_future<void> release;

        ~Cleanup()
        {
            started.set_value();
            release.wait();
        }
    };

    std::promise<void> cleanupStarted;
    std::promise<void> releaseCleanup;
    std::promise<void> releaseTask;
    auto cleanup = std::shared_ptr<Cleanup>(new Cleanup{cleanupStarted, releaseCleanup.get_future().share()});
    auto taskGate = releaseTask.get_future().share();
    queue.enqueueHostFn([cleanup, taskGate] { taskGate.wait(); });
    // Leave the queued callback as the last owner before allowing its body to return.
    cleanup.reset();
    releaseTask.set_value();
    cleanupStarted.get_future().wait();

    // The queue entry stays until the task and its captures are destroyed, so wait(queue)
    // must not become ready while the destructor is held open.
    auto waiter = std::async(std::launch::async, [&] { onHost::wait(queue); });
    auto const waitStatus = waiter.wait_for(blockingInterval);
    releaseCleanup.set_value();
    waiter.get();

    CHECK(waitStatus == std::future_status::timeout);
}

TEST_CASE("host blocking event wait lets the event producer enter the queue", "[queue]")
{
    // Queue A waits on an event recorded on a non-blocking queue. The recording task needs
    // queue A's execution slot before it can complete, so holding A's lock across the wait
    // deadlocks. The producer task is held behind a gate and the wait is bounded by watchdogs.
    using Device = decltype(onHost::makeHostDevice());
    auto scenario = std::make_unique<CrossQueueWaitScenario<Device>>();
    auto& state = *scenario;
    PromiseGate releaseEventTask{state.releaseEventTask};

    state.recordingQueue.enqueueHostFn(
        [&state]
        {
            state.releaseEventTask.get_future().wait();
            state.producingQueue.enqueueHostFn([&state] { state.waitingQueue.enqueueHostFn([] {}); });
        });
    state.recordingQueue.enqueue(state.event);

    auto waiter = std::make_unique<std::future<void>>(std::async(
        std::launch::async,
        [&state]
        {
            state.waitEntered.set_value();
            state.waitingQueue.waitFor(state.event);
        }));

    bool const entered = state.waitEntered.get_future().wait_for(startupTimeout) == std::future_status::ready;

    // Wait until waitFor has registered the dependency as pending blocking work.
    bool dependencyPending = false;
    auto const dependencyDeadline = std::chrono::steady_clock::now() + startupTimeout;
    while(!dependencyPending && std::chrono::steady_clock::now() < dependencyDeadline)
    {
        dependencyPending = !state.waitingQueue.isEmpty();
        if(!dependencyPending)
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }

    releaseEventTask.release();
    auto const status = waiter->wait_for(crossQueueWatchdog);

    CHECK(entered);
    CHECK(dependencyPending);
    CHECK(status == std::future_status::ready);

    if(status == std::future_status::ready)
    {
        waiter->get();
        waiter.reset();
        scenario.reset();
    }
    else
    {
        /* The buggy implementation leaves the waiter and the producer blocked. Keep their state alive so the
         * watchdog can report the failure without joining the deadlocked thread. */
        waiter.release();
        scenario.release();
    }
}

TEST_CASE("host nonblocking queue wait propagates callback failures", "[queue]")
{
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::nonBlocking);
    queue.enqueueHostFn([] { throw std::runtime_error{"host task failure"}; });

    // The first callback failure is reported once; a repeated wait observes a clean queue.
    CHECK_THROWS_AS(onHost::wait(queue), std::runtime_error);
    CHECK_NOTHROW(onHost::wait(queue));
}

TEST_CASE("host event wait propagates preceding callback failures", "[queue]")
{
    // Queue: throwing callback -> event. The event must carry the failure that precedes it.
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue(queueKind::nonBlocking);
    auto event = queue.makeEvent();
    queue.enqueueHostFn([] { throw std::runtime_error{"host task failure"}; });
    queue.enqueue(event);

    CHECK_THROWS_AS(onHost::wait(event), std::runtime_error);
    CHECK(event.isComplete());
}
