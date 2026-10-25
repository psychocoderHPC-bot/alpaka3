/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/** @file
 *
 * Host-only tests of the M5 GEMM planner.
 *
 * No device is created: facts are supplied explicitly, so the tests cover determinism, legality, minimal and
 * contradictory fact sets, the candidate cap, expert-override rejection and diagnostic content. Heuristic quality is
 * not measured here and no clock, random engine or filesystem state participates in selection.
 */

namespace
{
    using namespace alpaka;
    using onHost::GemmPlan;
    using onHost::internal::buildCandidateFamily;
    using onHost::internal::CandidateOutcome;
    using onHost::internal::compiledFact;
    using onHost::internal::deviceFact;
    using onHost::internal::estimatedRegisterBudgetBytes;
    using onHost::internal::Fact;
    using onHost::internal::GemmCandidateFamily;
    using onHost::internal::GemmFacts;
    using onHost::internal::GemmPlanRequest;
    using onHost::internal::GemmProblem;
    using onHost::internal::PlanSelection;
    using onHost::internal::PlanValidation;
    using onHost::internal::selectPlan;
    using onHost::internal::validatePlan;

    struct TargetProfile
    {
        char const* name;
        std::uint32_t simdWidth;
        std::uint32_t numPipelines;
        std::uint64_t cacheLineBytes;
        std::uint64_t sharedMemPerBlockBytes;
        std::uint64_t maxThreadsPerBlock;
        std::uint64_t maxBlocksPerGrid;
        std::uint64_t multiProcessorCount;
        std::uint64_t warpSize;
    };

    std::array<TargetProfile, 4> const targetProfiles{{
        {"width-1-serial", 1u, 1u, 64u, 32768u, 1u, 1u << 30, 1u, 1u},
        {"cpu-simd8", 8u, 2u, 64u, 32768u, 1u, 1u << 30, 8u, 8u},
        {"gpu-simd4", 4u, 2u, 16u, 49152u, 1024u, 1u << 30, 80u, 32u},
        {"gpu-simd16", 16u, 4u, 16u, 65536u, 256u, 65535u, 48u, 64u},
    }};

    struct ModeInfo
    {
        char const* name;
        onHost::GemmNumericMode mode;
        std::uint32_t accumulatorBytes;
        std::uint64_t aBytes;
        std::uint64_t bBytes;
        std::uint64_t cBytes;
    };

    std::array<ModeInfo, 4> const modes{{
        {"fp32", onHost::GemmNumericMode::fp32, 4u, 4u, 4u, 4u},
        {"fp64", onHost::GemmNumericMode::fp64, 8u, 8u, 8u, 8u},
        {"fp16ToFp32", onHost::GemmNumericMode::fp16ToFp32, 4u, 2u, 2u, 4u},
        {"bf16ToFp32", onHost::GemmNumericMode::bf16ToFp32, 4u, 2u, 2u, 4u},
    }};

    /** Facts with every optional query left unknown. */
    GemmFacts makeFacts(TargetProfile const& target, ModeInfo const& mode)
    {
        GemmFacts facts;
        facts.accumulatorBytes = mode.accumulatorBytes;
        facts.simdWidth = compiledFact(target.simdWidth, "simdWidth", "elements");
        facts.numPipelines = compiledFact(target.numPipelines, "numPipelines", "instructions");
        facts.cacheLineBytes = compiledFact(target.cacheLineBytes, "cacheLineBytes", "bytes");
        facts.mQuantum = compiledFact<std::uint32_t>(1u, "mQuantum", "elements");
        facts.nQuantum = compiledFact<std::uint32_t>(1u, "nQuantum", "elements");
        facts.kQuantum = compiledFact<std::uint32_t>(1u, "kQuantum", "elements");
        facts.sharedMemPerBlockBytes = deviceFact(target.sharedMemPerBlockBytes, "sharedMemPerBlockBytes", "bytes");
        facts.maxThreadsPerBlock = deviceFact(target.maxThreadsPerBlock, "maxThreadsPerBlock", "threads");
        facts.maxBlocksPerGrid = deviceFact(target.maxBlocksPerGrid, "maxBlocksPerGrid", "blocks");
        facts.multiProcessorCount = deviceFact(target.multiProcessorCount, "multiProcessorCount", "processors");
        facts.warpSize = deviceFact(target.warpSize, "warpSize", "threads");
        facts.globalMemCapacityBytes
            = deviceFact<std::uint64_t>(1ull << 34, "globalMemCapacityBytes", "bytes");
        return facts;
    }

    GemmCandidateFamily familyOf(GemmFacts const& facts)
    {
        return buildCandidateFamily(
            facts.simdWidth.get(),
            static_cast<std::uint32_t>(facts.maxThreadsPerBlock.get()),
            facts.kQuantum.get());
    }

    GemmProblem makeProblem(
        ModeInfo const& mode,
        std::uint64_t m,
        std::uint64_t n,
        std::uint64_t k,
        onHost::GemmOp aOp = onHost::GemmOp::none,
        onHost::GemmOp bOp = onHost::GemmOp::none)
    {
        GemmProblem problem;
        problem.m = m;
        problem.n = n;
        problem.k = k;
        problem.mode = mode.mode;
        problem.elementBytesA = mode.aBytes;
        problem.elementBytesB = mode.bBytes;
        problem.elementBytesC = mode.cBytes;
        problem.aOp = aOp;
        problem.bOp = bOp;
        return problem;
    }

} // namespace

TEST_CASE("gemm plan determinism", "[gemm][plan]")
{
    std::size_t cases = 0u;
    for(auto const& target : targetProfiles)
    {
        for(auto const& mode : modes)
        {
            GemmFacts const facts = makeFacts(target, mode);
            GemmCandidateFamily const family = familyOf(facts);
            std::array<std::array<std::uint64_t, 3>, 6> const shapes{{
                {1u, 1u, 1u},
                {7u, 13u, 5u},
                {64u, 64u, 64u},
                {65u, 63u, 31u},
                {1u, 1024u, 3u},
                {33u, 1u, 129u},
            }};
            for(auto const& shape : shapes)
            {
                GemmProblem const problem = makeProblem(mode, shape[0], shape[1], shape[2]);
                PlanSelection const first = selectPlan(problem, facts, family);
                for(std::uint32_t repeat = 0u; repeat < 4u; ++repeat)
                {
                    PlanSelection const again = selectPlan(problem, facts, family);
                    CHECK(again.plan == first.plan);
                    CHECK(again.diagnostics.selectedScore == first.diagnostics.selectedScore);
                    CHECK(again.diagnostics.fallbackReason == first.diagnostics.fallbackReason);
                    CHECK(again.diagnostics.candidates.size() == first.diagnostics.candidates.size());
                }
                ++cases;
            }
        }
    }
    std::cout << "[plan] determinism cases " << cases << "\n";
    CHECK(cases == targetProfiles.size() * modes.size() * 6u);
}

TEST_CASE("gemm plan selected plans pass launch validation", "[gemm][plan][property]")
{
    std::array<std::array<std::uint64_t, 3>, 20> const shapes{{
        {0u, 0u, 0u},
        {0u, 5u, 3u},
        {5u, 0u, 3u},
        {8u, 8u, 0u},
        {1u, 1u, 1u},
        {1u, 1u, 7u},
        {1u, 8u, 1u},
        {2u, 2u, 2u},
        {3u, 5u, 7u},
        {4u, 4u, 4u},
        {7u, 13u, 5u},
        {16u, 16u, 16u},
        {17u, 19u, 23u},
        {1u, 1024u, 1u},
        {1024u, 1u, 1u},
        {33u, 1u, 17u},
        {1u, 33u, 17u},
        {64u, 64u, 64u},
        {65u, 63u, 31u},
        {128u, 7u, 129u},
    }};
    std::array<std::pair<onHost::GemmOp, onHost::GemmOp>, 4> const layouts{{
        {onHost::GemmOp::none, onHost::GemmOp::none},
        {onHost::GemmOp::none, onHost::GemmOp::transpose},
        {onHost::GemmOp::transpose, onHost::GemmOp::none},
        {onHost::GemmOp::transpose, onHost::GemmOp::transpose},
    }};

    std::size_t combinations = 0u;
    std::size_t selected = 0u;
    for(auto const& target : targetProfiles)
    {
        for(auto const& mode : modes)
        {
            GemmFacts const facts = makeFacts(target, mode);
            GemmCandidateFamily const family = familyOf(facts);
            for(auto const& shape : shapes)
            {
                for(auto const& [aOp, bOp] : layouts)
                {
                    GemmProblem const problem = makeProblem(mode, shape[0], shape[1], shape[2], aOp, bOp);
                    INFO(
                        target.name << " " << mode.name << " shape " << shape[0] << "x" << shape[1] << "x"
                                    << shape[2]);
                    PlanSelection const selection = selectPlan(problem, facts, family);
                    PlanValidation const validation = validatePlan(selection.plan, problem, facts, family);
                    REQUIRE(validation.legal);
                    REQUIRE(validation.reason.empty());
                    REQUIRE(selection.plan.kernel == onHost::GemmKernel::registerBlocked);
                    REQUIRE(selection.diagnostics.selected == selection.plan);
                    REQUIRE(selection.diagnostics.mode == mode.mode);
                    REQUIRE(
                        (selection.diagnostics.stagingBytes > 0u
                         || selection.plan.policy == onHost::GemmStaging::direct));
                    ++selected;
                    ++combinations;
                }
            }
        }
    }
    std::cout << "[plan] property combinations " << combinations << ", selected plans " << selected << "\n";
    CHECK(combinations == targetProfiles.size() * modes.size() * shapes.size() * layouts.size());
    CHECK(selected == combinations);
}

TEST_CASE("gemm plan works with minimal facts", "[gemm][plan]")
{
    TargetProfile const& target = targetProfiles[0];
    ModeInfo const& mode = modes[0];
    GemmFacts facts = makeFacts(target, mode);
    GemmCandidateFamily const family = familyOf(facts);

    // All optional facts remain unknown.
    REQUIRE_FALSE(facts.l1CacheBytes.known());
    REQUIRE_FALSE(facts.l2CacheBytes.known());
    REQUIRE_FALSE(facts.registerBudgetBytes.known());
    REQUIRE_FALSE(facts.activeBlocks.known());
    REQUIRE_FALSE(facts.nativeMatrixCombinations.known());
    REQUIRE_FALSE(facts.asyncCopyAlignmentBytes.known());

    GemmProblem const problem = makeProblem(mode, 13u, 17u, 11u);
    PlanSelection const selection = selectPlan(problem, facts, family);
    CHECK(validatePlan(selection.plan, problem, facts, family).legal);
    CHECK(selection.diagnostics.selectedEngine.find("FMA/SIMD") != std::string::npos);

    auto const contains = [&](std::string const& text)
    {
        return std::any_of(
            selection.diagnostics.missingFacts.begin(),
            selection.diagnostics.missingFacts.end(),
            [&](std::string const& entry) { return entry.find(text) != std::string::npos; });
    };
    CHECK(contains("l1CacheBytes"));
    CHECK(contains("l2CacheBytes"));
    CHECK(contains("registerBudgetBytes"));
    CHECK(contains("activeBlocks"));
    CHECK(contains("nativeMatrixCombinations"));
    CHECK(contains("asyncCopyAlignmentBytes"));
    CHECK_FALSE(selection.diagnostics.factsUsed.empty());
}

TEST_CASE("gemm plan handles contradictory optional facts", "[gemm][plan]")
{
    TargetProfile const& target = targetProfiles[2];
    ModeInfo const& mode = modes[0];
    GemmFacts facts = makeFacts(target, mode);
    GemmCandidateFamily const family = familyOf(facts);
    GemmProblem const problem = makeProblem(mode, 64u, 64u, 64u);

    SECTION("advertised register budget is too small for every candidate")
    {
        facts.registerBudgetBytes = deviceFact<std::uint64_t>(8u, "registerBudgetBytes", "bytes/worker");
        REQUIRE_THROWS_AS(selectPlan(problem, facts, family), std::invalid_argument);
    }

    SECTION("advertised register budget is contradictory but one fact set still selects")
    {
        facts.registerBudgetBytes = deviceFact<std::uint64_t>(
            estimatedRegisterBudgetBytes,
            "registerBudgetBytes",
            "bytes/worker");
        PlanSelection const selection = selectPlan(problem, facts, family);
        CHECK(validatePlan(selection.plan, problem, facts, family).legal);
    }

    SECTION("missing shared-memory fact keeps cooperative plans out")
    {
        facts.sharedMemPerBlockBytes = onHost::internal::unavailableFact<std::uint64_t>(
            "sharedMemPerBlockBytes",
            "bytes",
            "mocked unavailable device query");
        PlanSelection const selection = selectPlan(problem, facts, family);
        CHECK(selection.plan.policy == onHost::GemmStaging::direct);
        CHECK(validatePlan(selection.plan, problem, facts, family).legal);
        for(auto const& outcome : selection.diagnostics.candidates)
            if(outcome.plan.policy == onHost::GemmStaging::cooperative)
            {
                CHECK_FALSE(outcome.legal);
                CHECK(outcome.reason.find("known block-shared limit") != std::string::npos);
            }
    }

    SECTION("contradictory shared-memory fact prunes the oversized stages only")
    {
        facts.sharedMemPerBlockBytes = deviceFact<std::uint64_t>(1024u, "sharedMemPerBlockBytes", "bytes");
        PlanSelection const selection = selectPlan(problem, facts, family);
        CHECK(selection.plan.policy == onHost::GemmStaging::direct);
        std::size_t rejectedStaging = 0u;
        std::size_t legalStaging = 0u;
        for(auto const& outcome : selection.diagnostics.candidates)
            if(outcome.plan.policy == onHost::GemmStaging::cooperative)
            {
                if(outcome.legal)
                    ++legalStaging;
                else
                {
                    ++rejectedStaging;
                    CHECK(outcome.reason.find("staging requires") != std::string::npos);
                }
            }
        // Some small one-stage plans fit in 1 KiB and are pruned lazily; the large rings are rejected.
        CHECK(rejectedStaging > 0u);
        CHECK(legalStaging + rejectedStaging > 0u);
    }

    SECTION("an unknown native-matrix capability does not invent a native engine")
    {
        facts.nativeMatrixCombinations = onHost::internal::unavailableFact<std::uint32_t>(
            "nativeMatrixCombinations",
            "combinations",
            "mocked unavailable capability");
        PlanSelection const selection = selectPlan(problem, facts, family);
        CHECK(selection.diagnostics.selectedEngine.find("FMA/SIMD") != std::string::npos);
        CHECK(validatePlan(selection.plan, problem, facts, family).legal);
    }
}

TEST_CASE("gemm plan candidate family is bounded", "[gemm][plan]")
{
    static_assert(onHost::internal::maxPlanCandidates == 64u);

    std::size_t checked = 0u;
    for(auto const& target : targetProfiles)
    {
        for(auto const& mode : modes)
        {
            GemmFacts const facts = makeFacts(target, mode);
            GemmCandidateFamily const family = familyOf(facts);
            REQUIRE(family.size <= onHost::internal::maxPlanCandidates);
            std::uint32_t const expected = target.maxThreadsPerBlock >= 16u ? 64u : 32u;
            CHECK(family.size == expected);

            // The minimal fallback is first and is a legal, direct plan for the smallest shapes.
            GemmPlan const& fallback = family.candidates[0];
            CHECK(fallback.policy == onHost::GemmStaging::direct);
            CHECK(fallback.threadM == 1u);
            CHECK(fallback.threadN == target.simdWidth);
            CHECK(fallback.workerCols == 1u);
            CHECK(fallback.stages == 1u);

            for(std::size_t i = 0; i < family.size; ++i)
            {
                GemmPlan const& plan = family.candidates[i];
                CHECK(plan.threadN % target.simdWidth == 0u);
                CHECK(plan.threadCount <= target.maxThreadsPerBlock);
                CHECK(plan.tileM == plan.threadM * plan.workerRows);
                CHECK(plan.tileN == plan.threadN * plan.workerCols);
                CHECK(plan.kBlock % 1u == 0u);
                for(std::size_t j = i + 1u; j < family.size; ++j)
                    CHECK_FALSE(plan == family.candidates[j]);
            }
            ++checked;
        }
    }
    std::cout << "[plan] candidate family target/type combinations " << checked << "\n";
    CHECK(checked == targetProfiles.size() * modes.size());
}

TEST_CASE("gemm plan expert override rejection", "[gemm][plan][override]")
{
    TargetProfile const& target = targetProfiles[2];
    ModeInfo const& mode = modes[0];
    GemmFacts facts = makeFacts(target, mode);
    GemmCandidateFamily const family = familyOf(facts);
    GemmProblem const problem = makeProblem(mode, 64u, 64u, 64u);

    PlanSelection const automatic = selectPlan(problem, facts, family);
    GemmPlanRequest valid;
    valid.expertOverride = automatic.plan;
    PlanSelection const overridden = selectPlan(problem, facts, family, valid);
    CHECK(overridden.diagnostics.overrideUsed);
    CHECK(overridden.plan == automatic.plan);
    CHECK(overridden.diagnostics.selectedEngine.find("FMA/SIMD") != std::string::npos);

    auto const expectRejected = [&](GemmPlan const& plan, GemmProblem const& problemIn, GemmFacts const& factsIn,
                                    std::string const& needle)
    {
        GemmPlanRequest request;
        request.expertOverride = plan;
        try
        {
            (void)selectPlan(problemIn, factsIn, family, request);
            FAIL("expected the override to be rejected");
        } catch(std::invalid_argument const& error)
        {
            INFO(error.what());
            CHECK(std::string{error.what()}.find(needle) != std::string::npos);
        }
    };

    SECTION("illegal tile")
    {
        GemmPlan plan = automatic.plan;
        plan.tileM += 1u;
        expectRejected(plan, problem, facts, "tile extent");
    }

    SECTION("too much scratch")
    {
        GemmPlan plan = family.candidates[family.size - 1u];
        REQUIRE(plan.policy == onHost::GemmStaging::cooperative);
        facts.sharedMemPerBlockBytes = deviceFact<std::uint64_t>(64u, "sharedMemPerBlockBytes", "bytes");
        expectRejected(plan, problem, facts, "staging requires");
    }

    SECTION("unsupported numerical mode")
    {
        GemmProblem badMode = problem;
        badMode.mode = static_cast<onHost::GemmNumericMode>(99);
        expectRejected(automatic.plan, badMode, facts, "numerical mode");
    }

    SECTION("mismatched layout")
    {
        GemmProblem badLayout = problem;
        badLayout.cOp = onHost::GemmOp::transpose;
        expectRejected(automatic.plan, badLayout, facts, "output view must not be transposed");
    }

    SECTION("plan outside the compiled family")
    {
        GemmPlan plan = automatic.plan;
        plan.kBlock += 4u;
        plan.stages = 1u;
        expectRejected(plan, problem, facts, "not part of the compiled candidate family");
    }
}

TEST_CASE("gemm plan diagnostics content", "[gemm][plan][diagnostics]")
{
    TargetProfile const& target = targetProfiles[3];
    ModeInfo const& mode = modes[1];
    GemmFacts facts = makeFacts(target, mode);
    GemmCandidateFamily const family = familyOf(facts);
    GemmProblem const problem = makeProblem(mode, 300u, 200u, 129u);
    PlanSelection const selection = selectPlan(problem, facts, family);

    REQUIRE(selection.diagnostics.candidates.size() == family.size);
    std::size_t legal = 0u;
    std::size_t rejected = 0u;
    for(auto const& outcome : selection.diagnostics.candidates)
    {
        if(outcome.legal)
        {
            ++legal;
            CHECK(outcome.reason.empty());
        }
        else
        {
            ++rejected;
            CHECK_FALSE(outcome.reason.empty());
        }
    }
    CHECK(legal + rejected == family.size);
    CHECK(legal > 0u);
    CHECK(rejected > 0u);

    CHECK(selection.diagnostics.selected == selection.plan);
    CHECK(selection.diagnostics.mode == mode.mode);
    CHECK(selection.diagnostics.stagingBytes
          == onHost::internal::stagedBytes(selection.plan, facts.accumulatorBytes));
    CHECK(selection.diagnostics.selectedEngine.find("register-blocked") != std::string::npos);
    CHECK(selection.diagnostics.selectedScore > 0.0);
    CHECK(selection.diagnostics.registerPressureNote.find("estimate") != std::string::npos);
    CHECK_FALSE(selection.diagnostics.factsUsed.empty());

    // Every rejected plan this profile produces is rejected for a specific, inspectable reason.
    for(auto const& outcome : selection.diagnostics.candidates)
        if(!outcome.legal)
            CHECK(outcome.plan != selection.plan);
    std::cout << "[plan] diagnostics candidates " << family.size << " legal " << legal << " rejected " << rejected
              << "\n";
}

TEST_CASE("gemm plan explicit cooperative requests stay legal", "[gemm][plan][staging]")
{
    std::array<std::array<std::uint64_t, 3>, 3> const shapes{{
        {7u, 13u, 5u},
        {64u, 64u, 64u},
        {1u, 1024u, 3u},
    }};
    std::size_t cases = 0u;
    for(auto const& target : targetProfiles)
    {
        for(auto const& mode : modes)
        {
            GemmFacts const facts = makeFacts(target, mode);
            GemmCandidateFamily const family = familyOf(facts);
            for(auto const& shape : shapes)
            {
                GemmProblem const problem = makeProblem(mode, shape[0], shape[1], shape[2]);
                for(std::uint32_t stages = 1u; stages <= 3u; ++stages)
                {
                    GemmPlanRequest request;
                    request.policy = onHost::GemmStaging::cooperative;
                    request.stages = stages;
                    PlanSelection const selection = selectPlan(problem, facts, family, request);
                    CHECK(selection.plan.policy == onHost::GemmStaging::cooperative);
                    CHECK(selection.plan.stages == stages);
                    CHECK(validatePlan(selection.plan, problem, facts, family).legal);
                    ++cases;
                }
            }
        }
    }
    std::cout << "[plan] explicit cooperative cases " << cases << "\n";
    CHECK(cases == targetProfiles.size() * modes.size() * shapes.size() * 3u);
}
