/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "../../../validation/numerics/tolerance.hpp"
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace alpaka
{
    // ADL extension points of the frozen reference for the common 16-bit storage types.
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
using namespace alpaka::onAcc::gemm;

namespace
{
    using pipeline::Phase;
    using pipeline::PhaseCounts;
    using pipeline::StagePolicy;

    /** Host-side stage ring with no barriers.
     *
     * It follows the same generation-tracked transitions as `staging::SyncStageRing`, records the last written
     * generation per slot and verifies that a consumer only observes the generation it was allowed to consume. Every
     * transition runs through the progress guard, so a stalled schedule fails with a diagnostic instead of hanging.
     */
    template<std::uint32_t T_stages>
    struct TestRing
    {
        using Tracker = onAcc::staging::StageTracker<T_stages>;
        using Token = typename Tracker::Token;

        static constexpr std::uint64_t stepCap = 1'000'000u;

        Tracker tracker{};
        std::array<std::uint64_t, T_stages> content{};
        std::array<std::vector<std::uint64_t>, T_stages> generations{};
        PhaseCounts counts{};
        std::uint64_t steps{};
        std::uint64_t fills{};

        void progress(char const* where)
        {
            ++steps;
            if(steps > stepCap)
                throw std::runtime_error(
                    std::string{"gemm pipeline deadlock guard: more than "} + std::to_string(stepCap)
                    + " transitions without progress at " + where);
        }

        Token acquire()
        {
            progress("acquire");
            return tracker.acquire();
        }

        template<typename T_Acc>
        void commit(T_Acc const&, Token token)
        {
            progress("commit");
            CHECK(tracker.tryCommit(token));
            content[token.stage] = token.generation;
            generations[token.stage].push_back(token.generation);
        }

        template<typename T_Acc>
        void waitReady(T_Acc const&, Token token)
        {
            progress("wait");
            CHECK(tracker.tryWait(token));
            ++counts.waits;
        }

        void consume(Token token)
        {
            progress("consume");
            CHECK(tracker.tryConsume(token));
        }

        template<typename T_Acc>
        void complete(T_Acc const&, Token token)
        {
            progress("complete");
            CHECK(tracker.tryComplete(token));
            ++counts.completes;
        }

        void release(Token token)
        {
            progress("release");
            CHECK(tracker.tryRelease(token));
            ++counts.releases;
        }
    };

    /** Ring whose producer never publishes; used to prove the harness guard fails instead of hanging. */
    template<std::uint32_t T_stages>
    struct StalledRing
    {
        using Tracker = onAcc::staging::StageTracker<T_stages>;
        using Token = typename Tracker::Token;

        static constexpr std::uint32_t spinCap = 64u;

        Tracker tracker{};

        Token acquire()
        {
            return tracker.acquire();
        }

        template<typename T_Acc>
        void commit(T_Acc const&, Token)
        {
            // Deliberately does not publish the generation.
        }

        template<typename T_Acc>
        void waitReady(T_Acc const&, Token token)
        {
            for(std::uint32_t spin = 0u; spin < spinCap; ++spin)
                if(tracker.tryWait(token))
                    return;
            throw std::runtime_error("gemm pipeline deadlock guard: stage never became ready");
        }

        void consume(Token token)
        {
            tracker.consume(token);
        }

        template<typename T_Acc>
        void complete(T_Acc const&, Token token)
        {
            tracker.complete(token);
        }

        void release(Token token)
        {
            tracker.release(token);
        }
    };

    /** One legal schedule case: full tiles plus the optional partial tile. */
    struct TileCase
    {
        std::uint64_t fullTiles{};
        bool partialTile{};

        [[nodiscard]] std::uint64_t tiles() const
        {
            return pipeline::tileCount(fullTiles, partialTile);
        }
    };

    /** (L, S) pairs requested by the M6 state-machine matrix, with duplicate L values removed. */
    std::vector<TileCase> tileCases(std::uint32_t stages)
    {
        std::vector<TileCase> cases{
            {0u, true}, // only a partial tile: F = 0, R = 1
            {1u, false},
            {stages, false},
            {static_cast<std::uint64_t>(stages) + 1u, false},
            {2u * static_cast<std::uint64_t>(stages), false},
            {2u * static_cast<std::uint64_t>(stages) + 1u, false},
            {7u, false}};
        if(stages > 1u)
            cases.push_back({static_cast<std::uint64_t>(stages) - 1u, false});
        std::stable_sort(
            cases.begin(),
            cases.end(),
            [](TileCase const& lhs, TileCase const& rhs)
            {
                if(lhs.tiles() != rhs.tiles())
                    return lhs.tiles() < rhs.tiles();
                return lhs.partialTile < rhs.partialTile;
            });
        cases.erase(
            std::unique(
                cases.begin(),
                cases.end(),
                [](TileCase const& lhs, TileCase const& rhs)
                { return lhs.tiles() == rhs.tiles() && lhs.partialTile == rhs.partialTile; }),
            cases.end());
        return cases;
    }

    /** Run one policy over the canonical bounded schedule and verify every invariant and phase count. */
    template<std::uint32_t T_stages, StagePolicy T_policy>
    void runLegalSchedule(TileCase const& tileCase, std::uint64_t& runs, std::uint64_t& transitions)
    {
        using Ring = TestRing<T_stages>;
        using Pipeline = pipeline::BoundedStagePipeline<T_policy, T_stages, Ring>;
        using Token = typename Ring::Token;

        Ring ring;
        Pipeline pipe{ring};
        std::uint64_t const tiles = tileCase.tiles();

        auto const fill = [&](std::uint64_t batch, Token token, Phase phase)
        {
            CHECK(token == ring.tracker.tokenOf(batch));
            if(phase == Phase::prologue)
                ++pipe.ring.counts.prologueFills;
            else
            {
                CHECK(phase == Phase::steadyState);
                ++pipe.ring.counts.refills;
            }
            ++pipe.ring.fills;
        };
        auto const compute = [&](std::uint64_t batch, Token token, Phase phase)
        {
            // No read before ready and no stale read: the slot content must be this token's generation.
            CHECK(pipe.ring.content[token.stage] == token.generation);
            CHECK(pipe.ring.tracker.tokenOf(batch) == token);
            if(phase == Phase::steadyState)
                ++pipe.ring.counts.steadyComputes;
            else
            {
                CHECK(phase == Phase::drain);
                ++pipe.ring.counts.drainComputes;
            }
            ++pipe.ring.counts.computes;
        };

        pipe.run(0, tileCase.fullTiles, tileCase.partialTile, fill, compute);

        PhaseCounts const expected = pipeline::scheduleCounts(tiles, T_stages);
        CHECK(pipe.ring.counts == expected);
        CHECK(pipe.ring.fills == tiles);
        CHECK(pipe.ring.tracker.allFree());
        CHECK(pipe.ring.tracker.issuedCount() == tiles);

        // Generations are per slot monotonically increasing by exactly one and never reused early.
        for(std::uint32_t stage = 0u; stage < T_stages; ++stage)
        {
            auto const& history = pipe.ring.generations[stage];
            for(std::size_t i = 0u; i < history.size(); ++i)
            {
                CHECK(history[i] == static_cast<std::uint64_t>(i) + 1u);
                if(i > 0u)
                    CHECK(history[i] > history[i - 1u]);
            }
        }

        ++runs;
        transitions += pipe.ring.steps;
    }

    /** Enumerate every illegal transition of the tracker and assert it is rejected. */
    template<std::uint32_t T_stages>
    std::uint64_t rejectIllegalSchedules()
    {
        using Tracker = onAcc::staging::StageTracker<T_stages>;
        using Token = typename Tracker::Token;

        std::uint64_t rejections = 0u;
        auto const rejected = [&](bool ok)
        {
            CHECK_FALSE(ok);
            ++rejections;
        };

        Tracker tracker;
        std::array<Token, T_stages> tokens{};
        for(auto& token : tokens)
            CHECK(tracker.tryAcquire(token));

        // Ring full: no overwrite before release of an existing generation.
        Token extra{};
        rejected(tracker.tryAcquire(extra));

        // Read before ready and release without consume/complete are rejected.
        rejected(tracker.tryWait(tokens.front()));
        rejected(tracker.tryConsume(tokens.front()));
        rejected(tracker.tryComplete(tokens.front()));
        rejected(tracker.tryRelease(tokens.front()));

        // Matched commit/consume: a second commit is rejected.
        for(auto const& token : tokens)
            CHECK(tracker.tryCommit(token));
        rejected(tracker.tryCommit(tokens.front()));

        for(auto const& token : tokens)
        {
            CHECK(tracker.tryConsume(token));
            rejected(tracker.tryConsume(token));
            rejected(tracker.tryRelease(token));
            CHECK(tracker.tryComplete(token));
            rejected(tracker.tryComplete(token));
            CHECK(tracker.tryRelease(token));
        }

        // Forge a token with the right stage and the previous generation; every transition must reject it.
        Token const forged{tokens.front().stage, tokens.front().generation - 1u};
        rejected(tracker.tryWait(forged));
        rejected(tracker.tryConsume(forged));
        rejected(tracker.tryComplete(forged));
        rejected(tracker.tryRelease(forged));

        // A stale token cannot touch a slot refilled for the next generation.
        std::uint64_t const issuedBefore = tracker.issuedCount();
        Token const refilled = tracker.acquire();
        CHECK(refilled == tracker.tokenOf(issuedBefore));
        CHECK(refilled.stage == tokens.front().stage);
        CHECK(refilled.generation == tokens.front().generation + 1u);
        CHECK(tracker.tryCommit(refilled));
        CHECK(tracker.tryConsume(refilled));
        CHECK(tracker.tryComplete(refilled));
        for(auto const& stale : tokens)
            rejected(tracker.tryRelease(stale));
        rejected(tracker.tryWait(tokens.front()));
        rejected(tracker.tryConsume(tokens.front()));
        CHECK(tracker.tryRelease(refilled));

        // A token for a different stage is rejected too.
        if constexpr(T_stages > 1u)
        {
            Token const otherStage{static_cast<std::uint32_t>((refilled.stage + 1u) % T_stages), refilled.generation};
            rejected(tracker.tryWait(otherStage));
            rejected(tracker.tryConsume(otherStage));
        }
        return rejections;
    }

    using CpuBackends = std::decay_t<decltype(onHost::allBackends(
        onHost::enabledDeviceSpecs,
        std::make_tuple(exec::cpuSerial, exec::cpuOmpBlocks)))>;

    using validation::gemm::Layout;
    using validation::gemm::MatrixView;
    using validation::gemm::NumericMode;
    using validation::gemm::Op;

    struct Shape
    {
        std::size_t m;
        std::size_t n;
        std::size_t k;
    };

    /** Deterministic values with mixed signs and cancellation, independent of the random library. */
    std::vector<float> makeValues(std::size_t count, int stride, int divisor)
    {
        std::vector<float> values(count);
        for(std::size_t i = 0u; i < count; ++i)
            values[i] = static_cast<float>(static_cast<int>(i % static_cast<std::size_t>(stride)) - stride / 2)
                        / static_cast<float>(divisor);
        return values;
    }

    /** Quantize an FP32 host matrix into the storage type of a numerical mode. */
    template<typename Storage>
    std::vector<Storage> quantize(std::vector<float> const& source)
    {
        std::vector<Storage> result(source.size());
        if constexpr(std::same_as<Storage, float> || std::same_as<Storage, double>)
            for(std::size_t i = 0u; i < source.size(); ++i)
                result[i] = static_cast<Storage>(source[i]);
        else
            for(std::size_t i = 0u; i < source.size(); ++i)
                result[i] = Storage::fromFloat(source[i]);
        return result;
    }

    /** Real CPU execution of one pipeline case against the frozen reference and tolerance model. */
    template<typename Storage, typename Output, typename RefT, typename T_Device, typename T_Exec, typename T_Queue>
    long double runRealCase(
        T_Device const& device,
        T_Exec const& exec,
        T_Queue& queue,
        Shape const& shape,
        NumericMode mode,
        onHost::GemmNumericMode hostMode,
        onHost::GemmStaging policy,
        std::uint32_t stages)
    {
        std::size_t const m = shape.m;
        std::size_t const n = shape.n;
        std::size_t const k = shape.k;

        auto const aFloat = makeValues(m * k, 29, 7);
        auto const bFloat = makeValues(k * n, 31, 9);
        auto const cFloat = makeValues(m * n, 13, 5);

        auto const a = quantize<Storage>(aFloat);
        auto const b = quantize<Storage>(bFloat);
        auto const c = quantize<Output>(cFloat);

        std::vector<RefT> reference(m * n, RefT{0});
        for(std::size_t i = 0u; i < m * n; ++i)
            reference[i] = static_cast<RefT>(c[i]);
        MatrixView<RefT> const referenceView{reference.data(), m, n, n, Layout::rowMajor, Op::none};
        MatrixView<Storage> const aView{const_cast<Storage*>(a.data()), m, k, k, Layout::rowMajor, Op::none};
        MatrixView<Storage> const bView{const_cast<Storage*>(b.data()), k, n, n, Layout::rowMajor, Op::none};
        MatrixView<Output> const cOldView{const_cast<Output*>(c.data()), m, n, n, Layout::rowMajor, Op::none};
        Output const alpha = static_cast<Output>(1.5);
        Output const beta = static_cast<Output>(-0.5);
        validation::gemm::gemm_reference<RefT>(referenceView, aView, bView, alpha, beta);

        auto dA = onHost::alloc<Storage>(device, Vec{std::max<std::size_t>(a.size(), 1u)});
        auto dB = onHost::alloc<Storage>(device, Vec{std::max<std::size_t>(b.size(), 1u)});
        auto dC = onHost::alloc<Output>(device, Vec{std::max<std::size_t>(c.size(), 1u)});
        onHost::memcpy(queue, dA, a);
        onHost::memcpy(queue, dB, b);
        onHost::memcpy(queue, dC, c);

        onHost::gemm(
            queue,
            exec,
            onHost::GemmView<Output>{onHost::data(dC), m, n, n, onHost::GemmLayout::rowMajor, onHost::GemmOp::none},
            onHost::GemmView<Storage>{onHost::data(dA), m, k, k, onHost::GemmLayout::rowMajor, onHost::GemmOp::none},
            onHost::GemmView<Storage>{onHost::data(dB), k, n, n, onHost::GemmLayout::rowMajor, onHost::GemmOp::none},
            alpha,
            beta,
            hostMode,
            onHost::GemmKernel::registerBlocked,
            onHost::GemmStagingConfig{policy, stages});

        std::vector<Output> candidate(c.size());
        onHost::memcpy(queue, candidate, dC);
        onHost::wait(queue);

        auto const model = validation::gemm::makeToleranceModel(
            mode,
            k,
            alpha != Output{0},
            beta != Output{0},
            beta == Output{1},
            false,
            false);
        MatrixView<Output> const candidateView{candidate.data(), m, n, n, Layout::rowMajor, Op::none};
        auto const report
            = validation::gemm::compareGemm(model, candidateView, referenceView, aView, bView, cOldView, alpha, beta);
        INFO(
            "mode " << static_cast<int>(mode) << " policy " << static_cast<int>(policy) << " stages " << stages
                    << " shape " << m << "x" << n << "x" << k);
        CHECK(report.informative);
        CHECK(report.withinBound);
        return report.maxScaledError;
    }
} // namespace

TEST_CASE("gemm pipeline legal state machine and phase accounting", "[gemm][pipeline][state-machine]")
{
    std::uint64_t runs = 0u;
    std::uint64_t transitions = 0u;
    std::uint64_t pairs = 0u;

    auto const checkStageCount = [&](auto stagesConstant)
    {
        constexpr std::uint32_t stages = decltype(stagesConstant)::value;
        for(auto const& tileCase : tileCases(stages))
        {
            INFO("stages " << stages << " tiles " << tileCase.tiles() << " partial " << tileCase.partialTile);
            runLegalSchedule<stages, StagePolicy::synchronous>(tileCase, runs, transitions);
            runLegalSchedule<stages, StagePolicy::asynchronous>(tileCase, runs, transitions);
            ++pairs;
        }
    };
    checkStageCount(std::integral_constant<std::uint32_t, 1u>{});
    checkStageCount(std::integral_constant<std::uint32_t, 2u>{});
    checkStageCount(std::integral_constant<std::uint32_t, 3u>{});

    std::cout << "[pipeline] legal schedule matrix: " << pairs << " (L,S) pairs x 2 policies = " << runs
              << " schedules, " << transitions << " checked transitions\n";
    CHECK(pairs >= 18u);
    CHECK(runs == pairs * 2u);
}

TEST_CASE("gemm pipeline rejects illegal state transitions", "[gemm][pipeline][state-machine]")
{
    std::uint64_t rejections = 0u;
    rejections += rejectIllegalSchedules<1u>();
    rejections += rejectIllegalSchedules<2u>();
    rejections += rejectIllegalSchedules<3u>();
    std::cout << "[pipeline] illegal transitions rejected: " << rejections << "\n";
    CHECK(rejections > 30u);
}

TEST_CASE("gemm pipeline deadlock guard fails instead of hanging", "[gemm][pipeline][state-machine]")
{
    using Ring = StalledRing<2u>;
    using Pipeline = pipeline::BoundedStagePipeline<StagePolicy::synchronous, 2u, Ring>;
    using Token = typename Ring::Token;

    Ring ring;
    Pipeline pipe{ring};
    auto const fill = [](std::uint64_t, Token, Phase) {};
    auto const compute = [](std::uint64_t, Token, Phase) {};

    CHECK_THROWS_WITH(pipe.run(0, 2u, false, fill, compute), "gemm pipeline deadlock guard: stage never became ready");
}

TEMPLATE_LIST_TEST_CASE(
    "gemm pipeline real execution partial tiles every stage count and mode",
    "[gemm][pipeline][numerics]",
    CpuBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    std::vector<Shape> const shapes{{37u, 23u, 67u}, {13u, 17u, 31u}};
    std::vector<onHost::GemmStaging> const policies{
        onHost::GemmStaging::cooperative,
        onHost::GemmStaging::asynchronous};

    long double worstFp32 = 0.0L;
    long double worstFp64 = 0.0L;
    long double worstFp16 = 0.0L;
    long double worstBf16 = 0.0L;

    for(auto const& shape : shapes)
        for(auto policy : policies)
            for(std::uint32_t stages : {1u, 2u, 3u})
            {
                worstFp32 = std::max(
                    worstFp32,
                    runRealCase<float, float, double>(
                        device,
                        exec,
                        queue,
                        shape,
                        NumericMode::fp32,
                        onHost::GemmNumericMode::fp32,
                        policy,
                        stages));
                if constexpr(validation::gemm::hasExtendedPrecisionReference)
                    worstFp64 = std::max(
                        worstFp64,
                        runRealCase<double, double, long double>(
                            device,
                            exec,
                            queue,
                            shape,
                            NumericMode::fp64,
                            onHost::GemmNumericMode::fp64,
                            policy,
                            stages));
                worstFp16 = std::max(
                    worstFp16,
                    runRealCase<Float16, float, double>(
                        device,
                        exec,
                        queue,
                        shape,
                        NumericMode::fp16ToFp32,
                        onHost::GemmNumericMode::fp16ToFp32,
                        policy,
                        stages));
                worstBf16 = std::max(
                    worstBf16,
                    runRealCase<BFloat16, float, double>(
                        device,
                        exec,
                        queue,
                        shape,
                        NumericMode::bf16ToFp32,
                        onHost::GemmNumericMode::bf16ToFp32,
                        policy,
                        stages));
            }

    std::cout << "[pipeline] real execution worst scaled error fp32 " << static_cast<double>(worstFp32) << ", fp64 "
              << static_cast<double>(worstFp64) << ", fp16 " << static_cast<double>(worstFp16) << ", bf16 "
              << static_cast<double>(worstBf16) << "\n";

    // The real execution test runs 2 shapes x 2 policies x 3 stages x 4 modes per backend.
    CHECK(worstFp32 >= 0.0L);
    CHECK(worstFp16 >= 0.0L);
    CHECK(worstBf16 >= 0.0L);
}
