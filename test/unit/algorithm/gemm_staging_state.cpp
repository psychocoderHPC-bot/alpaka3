/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace alpaka;

namespace
{
    template<std::uint32_t T_stages>
    struct ScheduleResult
    {
        std::uint64_t commits{};
        std::uint64_t consumes{};
        std::array<std::vector<std::uint64_t>, T_stages> generations;
    };

    /** Drive the bounded synchronous schedule over the generation-tracked ring and verify every invariant.
     *
     * The simulated panels store the generation of their last writer. A reader checks that the panel content is the
     * generation it was allowed to consume, which proves there is no read before ready and no stale read after a
     * wraparound.
     */
    template<std::uint32_t T_stages>
    ScheduleResult<T_stages> runSchedule(std::uint64_t batchCount)
    {
        using Tracker = onAcc::staging::StageTracker<T_stages>;
        using Token = typename Tracker::Token;

        Tracker tracker;
        std::array<std::uint64_t, T_stages> content{};
        ScheduleResult<T_stages> result;

        auto const fill = [&](Token token)
        {
            content[token.stage] = token.generation;
            result.generations[token.stage].push_back(token.generation);
        };
        auto const consume = [&](Token token)
        {
            CHECK(content[token.stage] == token.generation);
            ++result.consumes;
        };

        std::uint64_t const prologue = std::min<std::uint64_t>(T_stages, batchCount);
        for(std::uint64_t i = 0u; i < prologue; ++i)
        {
            auto const token = tracker.acquire();
            fill(token);
            CHECK(tracker.tryCommit(token));
            ++result.commits;
        }
        for(std::uint64_t i = 0u; i < batchCount; ++i)
        {
            auto const token = tracker.tokenOf(i);
            CHECK(tracker.tryWait(token));
            CHECK(tracker.tryConsume(token));
            consume(token);
            CHECK(tracker.tryComplete(token));
            CHECK(tracker.tryRelease(token));
            if(i + T_stages < batchCount)
            {
                auto const next = tracker.acquire();
                fill(next);
                CHECK(tracker.tryCommit(next));
                ++result.commits;
            }
        }

        CHECK(tracker.allFree());
        CHECK(tracker.issuedCount() == batchCount);
        CHECK(result.commits == batchCount);
        CHECK(result.consumes == batchCount);

        // Per slot, generations are strictly increasing and follow the ring order.
        for(std::uint32_t stage = 0u; stage < T_stages; ++stage)
        {
            auto const& history = result.generations[stage];
            for(std::size_t i = 0u; i < history.size(); ++i)
            {
                CHECK(history[i] == static_cast<std::uint64_t>(i) + 1u);
                if(i > 0u)
                    CHECK(history[i] > history[i - 1u]);
            }
        }
        CHECK(tracker.generation(0u) == result.generations[0u].size());
        return result;
    }
} // namespace

TEST_CASE("gemm staging state machine legal schedules", "[gemm][staging][state-machine]")
{
    std::vector<std::uint64_t> const batchCounts{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 12u};
    std::size_t cases = 0u;
    for(std::uint64_t batches : batchCounts)
    {
        INFO("batch count " << batches);
        runSchedule<1u>(batches);
        ++cases;
        runSchedule<2u>(batches);
        ++cases;
        runSchedule<3u>(batches);
        ++cases;
    }
    std::cout << "[staging] legal state-machine cases " << cases << "\n";
    CHECK(cases == batchCounts.size() * 3u);
}

TEST_CASE("gemm staging state machine rejects illegal transitions", "[gemm][staging][state-machine]")
{
    using Tracker = onAcc::staging::StageTracker<1u>;
    using Token = typename Tracker::Token;

    Tracker tracker;
    // No read before ready.
    Token token{};
    CHECK(tracker.tryAcquire(token));
    CHECK_FALSE(tracker.tryWait(token));
    CHECK_FALSE(tracker.tryConsume(token));
    CHECK_FALSE(tracker.tryComplete(token));
    CHECK_FALSE(tracker.tryRelease(token));

    // No overwrite before release, even if a second acquisition is attempted.
    Token duplicate{};
    CHECK_FALSE(tracker.tryAcquire(duplicate));

    // Commit publishes this generation only.
    CHECK(tracker.tryCommit(token));
    CHECK_FALSE(tracker.tryCommit(token));
    CHECK(tracker.tryWait(token));
    CHECK(tracker.tryConsume(token));
    CHECK_FALSE(tracker.tryConsume(token));
    CHECK_FALSE(tracker.tryRelease(token));
    CHECK(tracker.tryComplete(token));
    CHECK_FALSE(tracker.tryCommit(token));
    CHECK(tracker.tryRelease(token));
    CHECK_FALSE(tracker.tryConsume(token));

    // A stale completion cannot release the next, freshly acquired generation.
    CHECK(tracker.tryAcquire(token));
    Token fresh = tracker.tokenOf(1u);
    CHECK(fresh.stage == token.stage);
    CHECK(fresh == token);
    CHECK(tracker.tryCommit(token));
    CHECK(tracker.tryConsume(token));
    CHECK(tracker.tryComplete(token));
    CHECK(tracker.tryRelease(token));
    CHECK(tracker.tryAcquire(fresh));
    CHECK(fresh.generation > token.generation);
    CHECK_FALSE(tracker.tryWait(token));
    CHECK_FALSE(tracker.tryConsume(token));
    CHECK_FALSE(tracker.tryComplete(token));
    CHECK_FALSE(tracker.tryRelease(token));
}

TEST_CASE("gemm staging tracker rejects stale generations for every stage count", "[gemm][staging][state-machine]")
{
    auto const checkStale = [](auto stagesConstant)
    {
        constexpr std::uint32_t stages = decltype(stagesConstant)::value;
        using Tracker = onAcc::staging::StageTracker<stages>;
        Tracker tracker;
        std::array<typename Tracker::Token, stages> first{};
        for(auto& token : first)
            token = tracker.acquire();
        for(auto const& token : first)
            CHECK(tracker.tryCommit(token));
        for(auto const& token : first)
        {
            CHECK(tracker.tryConsume(token));
            CHECK(tracker.tryComplete(token));
            CHECK(tracker.tryRelease(token));
        }
        // Refill all slots: every token from the previous generation is now stale.
        std::array<typename Tracker::Token, stages> second{};
        for(auto& token : second)
            token = tracker.acquire();
        for(std::uint32_t i = 0u; i < stages; ++i)
        {
            CHECK(second[i].generation == first[i].generation + 1u);
            CHECK_FALSE(tracker.tryWait(first[i]));
            CHECK_FALSE(tracker.tryConsume(first[i]));
            CHECK_FALSE(tracker.tryComplete(first[i]));
            CHECK_FALSE(tracker.tryRelease(first[i]));
            CHECK_FALSE(tracker.tryCommit(first[i]));
        }
    };
    checkStale(std::integral_constant<std::uint32_t, 1u>{});
    checkStale(std::integral_constant<std::uint32_t, 2u>{});
    checkStale(std::integral_constant<std::uint32_t, 3u>{});
}
