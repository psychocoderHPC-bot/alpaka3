/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/onAcc/staging.hpp"
#include "alpaka/onHost/DeviceProperties.hpp"
#include "alpaka/onHost/algo/internal/gemmTypes.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

/** @file
 *
 * M5 GEMM planner: facts, bounded candidate family, legality pruning, deterministic ranking, diagnostics and expert
 * overrides.
 *
 * The planner is host-only and has no device dependency beyond the facts the caller supplies, so every decision is
 * inspectable in a unit test without a device. It never reads a clock, a random engine or the filesystem, and it
 * contains no backend name: the only inputs are the compiled traits and the runtime `DeviceProperties`.
 *
 * Unknown is represented by `std::nullopt` with a provenance and an explicit reason. No fact is defaulted to zero,
 * unlimited, or an invented hardware constant. The single exception is documented and labelled as a heuristic
 * estimate (the register-pressure model), which is a planner policy and never reported as a device fact.
 */

namespace alpaka::onHost
{
    /** One fully specified register-blocked GEMM configuration.
     *
     * All values are runtime integers: the planner never turns a problem dimension into a template argument. The
     * bounded candidate family below contains the configurations the compiled kernel family can execute, so the
     * launcher can dispatch to a plan without instantiating a shape-dependent kernel.
     *
     * `tileM`/`tileN` are the logical block-tile extents, `threadM`/`threadN` the per-worker register sub-tile and
     * `workerRows`/`workerCols` the worker grid. `tileM == workerRows * threadM` and
     * `tileN == workerCols * threadN` are invariants. `threadCount == workerRows * workerCols` is the block size.
     * `stages` is the synchronous stage-ring depth for a cooperative plan and 1 for a direct plan.
     */
    struct GemmPlan
    {
        std::uint32_t tileM = 0;
        std::uint32_t tileN = 0;
        std::uint32_t threadM = 0;
        std::uint32_t threadN = 0;
        std::uint32_t workerRows = 0;
        std::uint32_t workerCols = 0;
        std::uint32_t kBlock = 0;
        std::uint32_t stages = 0;
        GemmStaging policy = GemmStaging::direct;
        GemmKernel kernel = GemmKernel::registerBlocked;
        std::uint32_t threadCount = 0;
        std::uint32_t simdWidth = 0;

        constexpr bool operator==(GemmPlan const&) const = default;
        constexpr auto operator<=>(GemmPlan const&) const = default;
    };
} // namespace alpaka::onHost

namespace alpaka::onHost::internal
{
    /** Where a fact came from. `estimate` is a planner policy value, never a device fact. */
    enum class FactSource : std::uint8_t
    {
        compiledTrait,
        deviceProperty,
        estimate,
        unavailable
    };

    constexpr char const* factSourceName(FactSource source)
    {
        switch(source)
        {
        case FactSource::compiledTrait:
            return "compiled-trait";
        case FactSource::deviceProperty:
            return "device-property";
        case FactSource::estimate:
            return "estimate";
        case FactSource::unavailable:
            return "unavailable";
        }
        return "unknown";
    }

    /** A fact value with provenance and units.
     *
     * `value` is empty when the fact is unknown. `unavailableReason` is non-null for every unknown fact so a
     * diagnostic can explain the gap instead of silently substituting a number.
     */
    template<typename T>
    struct Fact
    {
        std::optional<T> value{};
        FactSource source = FactSource::unavailable;
        char const* name = "";
        char const* units = "";
        char const* unavailableReason = "not queried";

        [[nodiscard]] constexpr bool known() const noexcept
        {
            return value.has_value();
        }

        [[nodiscard]] constexpr T get() const
        {
            return value.value();
        }
    };

    template<typename T>
    constexpr Fact<T> compiledFact(T value, char const* name, char const* units)
    {
        return Fact<T>{value, FactSource::compiledTrait, name, units, nullptr};
    }

    template<typename T>
    constexpr Fact<T> deviceFact(T value, char const* name, char const* units)
    {
        return Fact<T>{value, FactSource::deviceProperty, name, units, nullptr};
    }

    template<typename T>
    constexpr Fact<T> unavailableFact(char const* name, char const* units, char const* reason)
    {
        return Fact<T>{std::nullopt, FactSource::unavailable, name, units, reason};
    }

    /** Planner facts.
     *
     * The first group comes from compiled traits and is always available. The second group comes from
     * `DeviceProperties`; the planner treats a missing value as unknown and rejects plans it cannot prove legal. The
     * third group has no source in the project and is modelled as an optional unknown with a reason.
     */
    struct GemmFacts
    {
        std::uint32_t accumulatorBytes = 0;

        Fact<std::uint32_t> simdWidth = unavailableFact<std::uint32_t>("simdWidth", "elements", "not queried");
        Fact<std::uint32_t> numPipelines
            = unavailableFact<std::uint32_t>("numPipelines", "instructions", "not queried");
        Fact<std::uint64_t> cacheLineBytes = unavailableFact<std::uint64_t>("cacheLineBytes", "bytes", "not queried");
        Fact<std::uint32_t> mQuantum = unavailableFact<std::uint32_t>("mQuantum", "elements", "not queried");
        Fact<std::uint32_t> nQuantum = unavailableFact<std::uint32_t>("nQuantum", "elements", "not queried");
        Fact<std::uint32_t> kQuantum = unavailableFact<std::uint32_t>("kQuantum", "elements", "not queried");

        Fact<std::uint64_t> sharedMemPerBlockBytes
            = unavailableFact<std::uint64_t>("sharedMemPerBlockBytes", "bytes", "not queried");
        Fact<std::uint64_t> maxThreadsPerBlock
            = unavailableFact<std::uint64_t>("maxThreadsPerBlock", "threads", "not queried");
        Fact<std::uint64_t> maxBlocksPerGrid
            = unavailableFact<std::uint64_t>("maxBlocksPerGrid", "blocks", "not queried");
        Fact<std::uint64_t> multiProcessorCount
            = unavailableFact<std::uint64_t>("multiProcessorCount", "processors", "not queried");
        Fact<std::uint64_t> warpSize = unavailableFact<std::uint64_t>("warpSize", "threads", "not queried");
        Fact<std::uint64_t> globalMemCapacityBytes
            = unavailableFact<std::uint64_t>("globalMemCapacityBytes", "bytes", "not queried");

        Fact<std::uint64_t> l1CacheBytes = unavailableFact<std::uint64_t>(
            "l1CacheBytes",
            "bytes",
            "DeviceProperties exposes no L1 capacity query in this project");
        Fact<std::uint64_t> l2CacheBytes = unavailableFact<std::uint64_t>(
            "l2CacheBytes",
            "bytes",
            "DeviceProperties exposes no L2 or last-level capacity query in this project");
        Fact<std::uint64_t> registerBudgetBytes = unavailableFact<std::uint64_t>(
            "registerBudgetBytes",
            "bytes/worker",
            "DeviceProperties exposes no register budget and no compiled resource usage in this project");
        Fact<std::uint64_t> activeBlocks = unavailableFact<std::uint64_t>(
            "activeBlocks",
            "blocks/processor",
            "no kernel-specific occupancy query exists in this project");
        Fact<std::uint32_t> nativeMatrixCombinations = unavailableFact<std::uint32_t>(
            "nativeMatrixCombinations",
            "combinations",
            "no native matrix capability enumeration exists in this project");
        Fact<std::uint64_t> asyncCopyAlignmentBytes = unavailableFact<std::uint64_t>(
            "asyncCopyAlignmentBytes",
            "bytes",
            "no asynchronous-copy capability query exists in this project");
    };

    /** Build the facts for one compiled target and one device.
     *
     * Compiled traits provide the SIMD width, the instruction-parallelism estimate, the cache-line size and the
     * operation quanta. The scalar FMA/SIMD path has an operation quantum of one in every dimension; the optional
     * native-matrix capability stays unknown. `DeviceProperties` provides the runtime limits.
     */
    template<typename T_Acc, typename T_Api, typename T_DeviceKind>
    GemmFacts makeGemmFacts(T_Api const api, T_DeviceKind const deviceKind, DeviceProperties const& properties)
    {
        GemmFacts facts;
        facts.accumulatorBytes = static_cast<std::uint32_t>(sizeof(T_Acc));
        facts.simdWidth = compiledFact(
            std::max<std::uint32_t>(getArchSimdWidth<T_Acc>(api, deviceKind), 1u),
            "simdWidth",
            "elements");
        facts.numPipelines = compiledFact(
            std::max<std::uint32_t>(getNumPipelines(api, deviceKind), 1u),
            "numPipelines",
            "instructions");
        facts.cacheLineBytes = compiledFact(
            static_cast<std::uint64_t>(std::max<std::uint32_t>(getCachelineSize(api, deviceKind), 1u)),
            "cacheLineBytes",
            "bytes");
        // The FMA/SIMD engine is element-wise: one element per operation in each dimension. Native matrix shapes are
        // not compiled in this milestone and their capability is unknown.
        facts.mQuantum = compiledFact<std::uint32_t>(1u, "mQuantum", "elements");
        facts.nQuantum = compiledFact<std::uint32_t>(1u, "nQuantum", "elements");
        facts.kQuantum = compiledFact<std::uint32_t>(1u, "kQuantum", "elements");

        facts.sharedMemPerBlockBytes = deviceFact(
            static_cast<std::uint64_t>(properties.sharedMemPerBlockBytes),
            "sharedMemPerBlockBytes",
            "bytes");
        facts.maxThreadsPerBlock
            = deviceFact(static_cast<std::uint64_t>(properties.maxThreadsPerBlock), "maxThreadsPerBlock", "threads");
        facts.maxBlocksPerGrid
            = deviceFact(static_cast<std::uint64_t>(properties.maxBlocksPerGrid), "maxBlocksPerGrid", "blocks");
        facts.multiProcessorCount = deviceFact(
            static_cast<std::uint64_t>(properties.multiProcessorCount),
            "multiProcessorCount",
            "processors");
        facts.warpSize = deviceFact(static_cast<std::uint64_t>(properties.warpSize), "warpSize", "threads");
        facts.globalMemCapacityBytes = deviceFact(
            static_cast<std::uint64_t>(properties.globalMemCapacityBytes),
            "globalMemCapacityBytes",
            "bytes");
        return facts;
    }

    /** The logical problem, independent of pointers and storage. */
    struct GemmProblem
    {
        std::uint64_t m = 0;
        std::uint64_t n = 0;
        std::uint64_t k = 0;
        GemmLayout aLayout = GemmLayout::rowMajor;
        GemmLayout bLayout = GemmLayout::rowMajor;
        GemmLayout cLayout = GemmLayout::rowMajor;
        GemmOp aOp = GemmOp::none;
        GemmOp bOp = GemmOp::none;
        GemmOp cOp = GemmOp::none;
        GemmNumericMode mode = GemmNumericMode::fp32;
        std::uint64_t elementBytesA = 4;
        std::uint64_t elementBytesB = 4;
        std::uint64_t elementBytesC = 4;
        /** Pointer alignment in bytes; 0 means unknown and skips the alignment check. */
        std::uint64_t alignmentA = 0;
        std::uint64_t alignmentB = 0;
        std::uint64_t alignmentC = 0;
        /** False for the beta-only scale path, which never reads A or B. */
        bool readsOperands = true;
    };

    /** Hard cap of the compiled family per operation, type and target. */
    inline constexpr std::size_t maxPlanCandidates = 64;
    /** Structural bound of every plan field; keeps all resource products far away from 64-bit overflow. */
    inline constexpr std::uint32_t maxPlanDimension = 8192;
    /** Documented heuristic cap of the per-worker unrolled work (an estimate, not a device fact). */
    inline constexpr std::uint32_t maxUnrolledWork = 256;
    /** Largest per-worker M extent in the candidate family.
     *
     * Paired with `TN = V` this fills the vector register file: at V=8 (AVX2 fp32) the tile has
     * `8 * 8 = 64` fp32 accumulators, which is eight 256-bit YMM registers for the output. One
     * further YMM holds the hoisted eight-element B row and the eight A values are scalar
     * broadcasts, so about nine of the 16 architectural YMM registers are live, with room for
     * prefetched operands and temporaries. At V=1 (scalar CPU or GPU) the tile has eight
     * accumulators; with `TM + TN + 8 = 17` operands and temporaries that is 25 registers, well
     * below a GPU's 255-register per-thread budget. The register-pressure legality check below
     * still rejects any candidate that exceeds a reported smaller budget, so the planner falls
     * back instead of assuming the tile fits.
     */
    inline constexpr std::uint32_t maxRegisterTileRows = 8;
    /** Documented conservative register budget used only when no register fact is available (an estimate). */
    inline constexpr std::uint64_t estimatedRegisterBudgetBytes = 2048;

    struct GemmCandidateFamily
    {
        std::array<GemmPlan, maxPlanCandidates> candidates{};
        std::size_t size = 0;
    };

    /** Build the bounded candidate family from dimensionless multiples of the compiled quanta.
     *
     * The family has three groups, all generated from compiled capabilities only:
     *
     * - a small direct group (`TN = V`, one worker) that contains the minimal fallback and the
     *   width-one/tiny plans; it grows the per-worker row count `TM` in {1, 2, 4, 8};
     * - a wide direct group (`TN = V`, `TM = 8`, worker columns up to 16) that keeps the previous
     *   skinny-N plans;
     * - a reuse-oriented group that pairs the largest per-worker tile (`TM = 8`, `TN = 2V`) with a
     *   `WR x WC` worker grid. The grids are chosen so the legal block tiles cover 16x16, 32x32,
     *   64x64, 64x128, 128x64 and 128x128 for a four-element SIMD width, and the legality check
     *   prunes every plan whose worker count exceeds the target's block-thread capability. The
     *   same grid therefore yields many workers on a GPU and collapses to the one-worker group on
     *   a CPU, without a backend branch.
     *
     * Direct plans use the {8, 16} multiples of the operation K quantum because they have no panel
     * refill. Cooperative reuse plans use {8, 16} with one or two stages and 32 with one stage: a
     * 32-quantum panel pair with two stages already exceeds the smallest block-shared limit this
     * planner models, so the legal single-stage variant is the useful large-K member. The group
     * order puts the minimal direct fallback first, and `maxPlanCandidates` bounds the family.
     */
    constexpr GemmCandidateFamily buildCandidateFamily(
        std::uint32_t simdWidth,
        std::uint32_t maxThreadsPerBlock,
        std::uint32_t kQuantum)
    {
        GemmCandidateFamily family{};
        std::uint32_t const width = std::max<std::uint32_t>(simdWidth, 1u);
        std::uint32_t const quantum = std::max<std::uint32_t>(kQuantum, 1u);
        std::uint32_t const threadLimit = std::max<std::uint32_t>(maxThreadsPerBlock, 1u);

        auto const add = [&](GemmPlan const& candidate) constexpr
        {
            if(family.size >= maxPlanCandidates)
                return;
            for(std::size_t i = 0; i < family.size; ++i)
                if(family.candidates[i] == candidate)
                    return;
            family.candidates[family.size++] = candidate;
        };

        constexpr std::array<std::uint32_t, 4> threadRows{1u, 2u, 4u, maxRegisterTileRows};
        // Direct plans keep K blocks {8, 16}: the direct kernel has no panel refill, so a larger
        // K block only regroups the same outer loop and cannot change the score.
        constexpr std::array<std::uint32_t, 2> directKMultiples{8u, 16u};
        // Reuse-oriented worker grids. For V=4 with TM=8 and TN=2V they describe block tiles
        // 16x16, 32x32, 64x64, 64x128, 128x64 and 128x128. The thread count is the grid product,
        // so the build-time limit and the runtime legality check prune grids the target cannot run.
        constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 6> reuseGrids{{
            {2u, 2u},
            {4u, 4u},
            {8u, 8u},
            {8u, 16u},
            {16u, 8u},
            {16u, 16u},
        }};

        auto const make = [&](std::uint32_t threadM,
                              std::uint32_t threadN,
                              std::uint32_t workerRows,
                              std::uint32_t workerCols,
                              std::uint32_t kBlock,
                              GemmStaging policy,
                              std::uint32_t stages) constexpr
        {
            GemmPlan plan;
            plan.tileM = threadM * workerRows;
            plan.tileN = threadN * workerCols;
            plan.threadM = threadM;
            plan.threadN = threadN;
            plan.workerRows = workerRows;
            plan.workerCols = workerCols;
            plan.kBlock = kBlock;
            plan.stages = stages;
            plan.policy = policy;
            plan.kernel = GemmKernel::registerBlocked;
            plan.threadCount = workerRows * workerCols;
            plan.simdWidth = width;
            return plan;
        };

        auto const addWhenItFits = [&](GemmPlan const& candidate) constexpr
        {
            if(candidate.threadCount <= threadLimit)
                add(candidate);
        };

        // Small direct group. Index zero is the minimal fallback: TM = 1, TN = V, one worker,
        // BK = 8 quanta, direct loads.
        for(std::uint32_t threadM : threadRows)
            for(std::uint32_t multiplier : directKMultiples)
                add(make(threadM, width, 1u, 1u, quantum * multiplier, GemmStaging::direct, 1u));

        // Wide direct group: the previous skinny-N plans. The worker-column count is clamped, so a
        // one-thread target collapses these candidates onto the small group by deduplication.
        std::uint32_t const wideWorkers = std::min<std::uint32_t>(16u, threadLimit);
        for(std::uint32_t multiplier : directKMultiples)
            add(make(maxRegisterTileRows, width, 1u, wideWorkers, quantum * multiplier, GemmStaging::direct, 1u));

        // Reuse-oriented direct group: largest per-worker tile and a multi-worker block tile.
        for(auto const& [workerRows, workerCols] : reuseGrids)
            for(std::uint32_t multiplier : directKMultiples)
                addWhenItFits(make(
                    maxRegisterTileRows,
                    2u * width,
                    workerRows,
                    workerCols,
                    quantum * multiplier,
                    GemmStaging::direct,
                    1u));

        // Cooperative variants. The one-worker group keeps every stage count legal on a
        // thread-limited target; the reuse group offers one and two stages so a measured policy can
        // pick staging for a block tile that actually amortizes the panel fill.
        for(std::uint32_t threadM : std::array<std::uint32_t, 2>{1u, maxRegisterTileRows})
            for(std::uint32_t stages = 1u; stages <= 3u; ++stages)
                add(make(threadM, width, 1u, 1u, quantum * 8u, GemmStaging::cooperative, stages));

        for(auto const& [workerRows, workerCols] : reuseGrids)
        {
            for(std::uint32_t multiplier : std::array<std::uint32_t, 2>{8u, 16u})
                for(std::uint32_t stages : std::array<std::uint32_t, 2>{1u, 2u})
                    addWhenItFits(make(
                        maxRegisterTileRows,
                        2u * width,
                        workerRows,
                        workerCols,
                        quantum * multiplier,
                        GemmStaging::cooperative,
                        stages));
            // The largest K block uses one stage. A second 32-quantum panel pair already exceeds
            // the smallest block-shared limit the planner models, and keeping one variant leaves
            // room for every reuse grid under maxPlanCandidates.
            addWhenItFits(make(
                maxRegisterTileRows,
                2u * width,
                workerRows,
                workerCols,
                quantum * 32u,
                GemmStaging::cooperative,
                1u));
        }

        return family;
    }

    /** Rejection or acceptance of one plan. `reason` is empty exactly when the plan is legal. */
    struct PlanValidation
    {
        bool legal = false;
        std::string reason{};
    };

    constexpr std::uint64_t ceilDivU64(std::uint64_t numerator, std::uint64_t denominator)
    {
        return numerator / denominator + (numerator % denominator == 0u ? 0u : 1u);
    }

    constexpr std::uint64_t roundUpU64(std::uint64_t value, std::uint64_t multiple)
    {
        return ceilDivU64(value, multiple) * multiple;
    }

    /** Documented register-pressure estimate in bytes per worker.
     *
     * The model is TM*TN accumulators plus TM + TN operands and eight address/temporary scalars, each at the
     * accumulator element size, and it is always reported as an estimate. It is not a compiler resource report and
     * it does not claim register residency.
     */
    constexpr std::uint64_t estimatedRegisterBytes(GemmPlan const& plan, std::uint64_t accumulatorBytes)
    {
        std::uint64_t const elem = std::max<std::uint64_t>(accumulatorBytes, 1u);
        std::uint64_t const scalars
            = static_cast<std::uint64_t>(plan.threadM) * plan.threadN + plan.threadM + plan.threadN + 8u;
        return scalars * elem;
    }

    /** Shared storage a cooperative plan needs, including the documented metadata reservation. */
    constexpr std::uint64_t stagedBytes(GemmPlan const& plan, std::uint64_t accumulatorBytes)
    {
        if(plan.policy != GemmStaging::cooperative)
            return 0u;
        std::uint64_t const panelA = static_cast<std::uint64_t>(plan.tileM) * plan.kBlock * accumulatorBytes;
        std::uint64_t const panelB = static_cast<std::uint64_t>(plan.kBlock) * plan.tileN * accumulatorBytes;
        return static_cast<std::uint64_t>(plan.stages) * (panelA + panelB)
               + onAcc::staging::stageMetadataReserveBytes();
    }

    /** Deterministic score of one legal plan.
     *
     * The inputs are the section 6.3 models only: useful-work fraction, input arithmetic intensity, transfer cost
     * derived from the same panel bytes, and resource pressure from the staging bound and the register estimate.
     * Intensity and transfer cost use tile extents clipped to the problem, so a register block only scores its
     * full-tile advantage when the problem actually fills it. There is no timing, randomness or filesystem state.
     * The weights are frozen heuristic constants.
     */
    inline double scorePlan(GemmPlan const& plan, GemmProblem const& problem, GemmFacts const& facts)
    {
        double const bm = static_cast<double>(plan.tileM);
        double const bn = static_cast<double>(plan.tileN);
        double const bk = static_cast<double>(plan.kBlock);
        double const sA = static_cast<double>(problem.elementBytesA);
        double const sB = static_cast<double>(problem.elementBytesB);

        double usefulFraction = 0.0;
        if(problem.m != 0u && problem.n != 0u && problem.k != 0u)
        {
            double const paddedM = static_cast<double>(roundUpU64(problem.m, plan.tileM));
            double const paddedN = static_cast<double>(roundUpU64(problem.n, plan.tileN));
            double const paddedK = static_cast<double>(roundUpU64(problem.k, plan.kBlock));
            double const padded = paddedM * paddedN * paddedK;
            if(padded > 0.0)
                usefulFraction = (static_cast<double>(problem.m) * static_cast<double>(problem.n)
                                  * static_cast<double>(problem.k))
                                 / padded;
        }

        // The intensity and transfer advantage of a large tile exists only where the problem can
        // fill it. Clip the nominal tile extents to the problem: otherwise an 8x8 tile claims the
        // intensity of a full tile on a 1x1x1 or 1x1024x3 problem and outranks the small direct
        // fallback. For compute-heavy shapes every candidate tile is fully covered, so the larger
        // register block keeps its real arithmetic-intensity and operand-reuse advantage.
        double const effectiveM = std::min(bm, static_cast<double>(problem.m));
        double const effectiveN = std::min(bn, static_cast<double>(problem.n));
        double const effectiveK = std::min(bk, static_cast<double>(problem.k));
        double const panelBytes = effectiveM * effectiveK * sA + effectiveK * effectiveN * sB;
        double const macs = 2.0 * effectiveM * effectiveN * effectiveK;
        double const arithmeticIntensity = panelBytes > 0.0 ? macs / panelBytes : 0.0;
        double const aiNorm = arithmeticIntensity / (arithmeticIntensity + 8.0);
        // Transfer cost is the inverse of the input intensity: bytes of operand traffic per multiply-accumulate.
        double const transferCost = macs > 0.0 ? panelBytes / macs : 1.0;
        double const transferNorm = transferCost / (transferCost + 1.0);

        // Block-tile reuse only pays off when the grid still fills the machine. Without this term a
        // 128x128 tile wins every compute-heavy shape, but a 256^2 problem would launch only four
        // blocks onto dozens of processors. The schedule factor is a capability ratio built from the
        // reported processor count, not a backend identity, so a single-thread CPU host is
        // unaffected and a many-processor GPU keeps its larger tiles only where enough output tiles
        // remain. Saturation at three quarters of the processor count treats one partial wave as a
        // filled machine; a linear ratio otherwise punishes a 128x128 tile whose 64 blocks leave a
        // handful of processors idle even though the block tile reuses far more operand data.
        double parallelFraction = 1.0;
        if(problem.m != 0u && problem.n != 0u && facts.multiProcessorCount.known())
        {
            double const processors = std::max(1.0, static_cast<double>(facts.multiProcessorCount.get()));
            double const tiles = static_cast<double>(ceilDivU64(problem.m, plan.tileM))
                                 * static_cast<double>(ceilDivU64(problem.n, plan.tileN));
            parallelFraction = std::clamp(tiles / (0.75 * processors), 0.0, 1.0);
        }

        double stagingPressure = 0.0;
        if(plan.policy == GemmStaging::cooperative && facts.sharedMemPerBlockBytes.known())
        {
            double const limit = static_cast<double>(facts.sharedMemPerBlockBytes.get());
            if(limit > 0.0)
                stagingPressure = static_cast<double>(stagedBytes(plan, facts.accumulatorBytes)) / limit;
        }
        std::uint64_t const registerBudget
            = facts.registerBudgetBytes.known() ? facts.registerBudgetBytes.get() : estimatedRegisterBudgetBytes;
        double registerPressure = 0.0;
        if(registerBudget > 0u)
            registerPressure = static_cast<double>(estimatedRegisterBytes(plan, facts.accumulatorBytes))
                               / static_cast<double>(registerBudget);
        double const pressure = std::clamp(0.5 * stagingPressure + 0.5 * registerPressure, 0.0, 1.0);

        // Cooperative staging loads one A column panel and one B row panel per K step and shares them
        // across the worker grid; a direct plan reloads the per-worker operands in every worker. The
        // saving is therefore a capability ratio of the plan itself. The ramp suppresses the credit
        // for small worker grids, where the measured panel fill costs more than the sharing saves:
        // a 32x32 tile (16 workers) stays direct, while a 64x64 tile (64 workers) and larger keep
        // their measured cooperative advantage.
        double sharingFraction = 0.0;
        double refillSaving = 0.0;
        if(plan.policy == GemmStaging::cooperative)
        {
            double const workers = static_cast<double>(plan.workerRows) * static_cast<double>(plan.workerCols);
            double const directPanel = workers * (static_cast<double>(plan.threadM) + plan.threadN);
            double const stagedPanel = bm + bn;
            double const saving = directPanel > 0.0 ? std::max(0.0, 1.0 - stagedPanel / directPanel) : 0.0;
            double const ramp = std::clamp((workers - 16.0) / (64.0 - 16.0), 0.0, 1.0);
            // Only claim the saving where the problem fills the block tile and the output grid still
            // occupies the machine; a one-tile launch or a tiny problem keeps the direct path.
            double const coverage = (bm > 0.0 ? effectiveM / bm : 0.0) * (bn > 0.0 ? effectiveN / bn : 0.0);
            sharingFraction = ramp * saving * coverage * parallelFraction;

            // Every cooperative K block ends in one panel refill and one block-wide barrier. The
            // refill count per K element is 1/kBlock, so a plan that doubles the K block halves that
            // fixed cost. Normalise the saving to the smallest K block in the family (eight K
            // quanta). Small worker grids do not amortise a refill barrier, so the same ramp and
            // coverage gate as the sharing credit applies; a direct plan has no refill barrier.
            if(bk > 0.0)
            {
                double const smallestKBlock
                    = 8.0
                      * static_cast<double>(
                          std::max<std::uint32_t>(facts.kQuantum.known() ? facts.kQuantum.get() : 1u, 1u));
                refillSaving = std::clamp(1.0 - smallestKBlock / bk, 0.0, 1.0) * ramp * coverage * parallelFraction;
            }
        }

        return 0.36 * usefulFraction + 0.22 * aiNorm + 0.12 * (1.0 - transferNorm) + 0.12 * parallelFraction
               + 0.08 * (1.0 - pressure) + 0.10 * sharingFraction + 0.10 * refillSaving;
    }

    /** Validate one plan with the checks the launcher relies on.
     *
     * The same function handles automatic candidates and expert overrides. `family` is the compiled family the
     * launcher can dispatch; a plan outside it is legal in isolation but cannot be launched and is rejected.
     */
    inline PlanValidation validatePlan(
        GemmPlan const& plan,
        GemmProblem const& problem,
        GemmFacts const& facts,
        GemmCandidateFamily const& family)
    {
        auto reject = [](std::string reason) { return PlanValidation{false, std::move(reason)}; };

        if(plan.kernel != GemmKernel::registerBlocked)
            return reject("planner supports only the register-blocked FMA/SIMD kernel");
        if(plan.policy != GemmStaging::direct && plan.policy != GemmStaging::cooperative)
            return reject("operand-supply policy is not direct or cooperative");
        if(plan.stages < 1u || plan.stages > 3u)
            return reject("stage count must be in {1, 2, 3}");
        if(plan.policy == GemmStaging::direct && plan.stages != 1u)
            return reject("direct plans must use exactly one stage");

        auto const fieldOutOfRange = [](std::uint32_t value) { return value == 0u || value > maxPlanDimension; };
        if(fieldOutOfRange(plan.tileM) || fieldOutOfRange(plan.tileN) || fieldOutOfRange(plan.threadM)
           || fieldOutOfRange(plan.threadN) || fieldOutOfRange(plan.workerRows) || fieldOutOfRange(plan.workerCols)
           || fieldOutOfRange(plan.kBlock) || fieldOutOfRange(plan.threadCount))
            return reject("plan has a zero or above-bound tile, thread or K-block field");

        std::uint64_t const expectedTileM = static_cast<std::uint64_t>(plan.threadM) * plan.workerRows;
        std::uint64_t const expectedTileN = static_cast<std::uint64_t>(plan.threadN) * plan.workerCols;
        if(expectedTileM != plan.tileM || expectedTileN != plan.tileN)
            return reject("tile extent does not equal worker grid times per-worker sub-tile");
        if(static_cast<std::uint64_t>(plan.workerRows) * plan.workerCols != plan.threadCount)
            return reject("thread count does not equal worker rows times worker columns");

        if(!facts.simdWidth.known())
            return reject("SIMD width fact is unknown");
        std::uint32_t const width = std::max<std::uint32_t>(facts.simdWidth.get(), 1u);
        if(plan.threadN % width != 0u)
            return reject("per-worker N extent is not an integral number of SIMD packs");
        if(plan.simdWidth != width)
            return reject("plan SIMD width does not match the compiled SIMD width fact");
        if(!facts.kQuantum.known())
            return reject("K quantum fact is unknown");
        std::uint32_t const kQuantum = std::max<std::uint32_t>(facts.kQuantum.get(), 1u);
        if(plan.kBlock % kQuantum != 0u)
            return reject("K block is not a multiple of the operation K quantum");

        if(static_cast<std::uint64_t>(plan.threadM) * plan.threadN > maxUnrolledWork)
            return reject("estimated code size exceeds the documented unrolled-work budget");

        // Numerical mode and the storage signature. The planner does not change the numerical contract; it rejects a
        // mode whose required storage does not match the problem.
        switch(problem.mode)
        {
        case GemmNumericMode::fp32:
            if(problem.elementBytesA != 4u || problem.elementBytesB != 4u || problem.elementBytesC != 4u)
                return reject("fp32 mode requires four-byte storage for A, B and C");
            break;
        case GemmNumericMode::fp64:
            if(problem.elementBytesA != 8u || problem.elementBytesB != 8u || problem.elementBytesC != 8u)
                return reject("fp64 mode requires eight-byte storage for A, B and C");
            break;
        case GemmNumericMode::fp16ToFp32:
        case GemmNumericMode::bf16ToFp32:
            if(problem.elementBytesA != 2u || problem.elementBytesB != 2u || problem.elementBytesC != 4u)
                return reject("mixed 16-bit to FP32 mode requires two-byte A/B and four-byte C storage");
            break;
        default:
            return reject("numerical mode is not one of the four supported modes");
        }

        auto const layoutIsValid
            = [](GemmLayout layout) { return layout == GemmLayout::rowMajor || layout == GemmLayout::colMajor; };
        auto const opIsValid = [](GemmOp op) { return op == GemmOp::none || op == GemmOp::transpose; };
        if(!layoutIsValid(problem.aLayout) || !layoutIsValid(problem.bLayout) || !layoutIsValid(problem.cLayout)
           || !opIsValid(problem.aOp) || !opIsValid(problem.bOp) || !opIsValid(problem.cOp))
            return reject("layout or transpose tag is not a supported value");
        if(problem.cOp != GemmOp::none)
            return reject("the output view must not be transposed");

        auto const aligned = [](std::uint64_t alignment, std::uint64_t elementBytes)
        {
            if(alignment == 0u)
                return true;
            return alignment % std::max<std::uint64_t>(elementBytes, 1u) == 0u;
        };
        if(!aligned(problem.alignmentA, problem.elementBytesA) || !aligned(problem.alignmentB, problem.elementBytesB)
           || !aligned(problem.alignmentC, problem.elementBytesC))
            return reject("a known pointer alignment is not a multiple of the element size");

        if(!facts.maxThreadsPerBlock.known())
            return reject("block-thread limit fact is unknown");
        if(static_cast<std::uint64_t>(plan.threadCount) > facts.maxThreadsPerBlock.get())
            return reject(
                "thread count " + std::to_string(plan.threadCount) + " exceeds the block limit "
                + std::to_string(facts.maxThreadsPerBlock.get()));

        std::uint64_t const registerBudget
            = facts.registerBudgetBytes.known() ? facts.registerBudgetBytes.get() : estimatedRegisterBudgetBytes;
        if(registerBudget == 0u)
            return reject("register budget fact is zero, which cannot host any accumulator");
        if(estimatedRegisterBytes(plan, facts.accumulatorBytes) > registerBudget)
            return reject("estimated register pressure exceeds the available register budget");

        if(plan.policy == GemmStaging::cooperative)
        {
            if(!facts.sharedMemPerBlockBytes.known())
                return reject("cooperative staging requires a known block-shared limit");
            std::uint64_t const required = stagedBytes(plan, facts.accumulatorBytes);
            if(required > facts.sharedMemPerBlockBytes.get())
                return reject(
                    "staging requires " + std::to_string(required) + " bytes but the block-shared limit is "
                    + std::to_string(facts.sharedMemPerBlockBytes.get()));
        }

        if(!facts.maxBlocksPerGrid.known())
            return reject("grid-block limit fact is unknown");
        std::uint64_t const tiles = ceilDivU64(problem.m, plan.tileM) * ceilDivU64(problem.n, plan.tileN);
        if(tiles > facts.maxBlocksPerGrid.get())
            return reject(
                "output grid of " + std::to_string(tiles) + " tiles exceeds the grid limit "
                + std::to_string(facts.maxBlocksPerGrid.get()));
        if(tiles > std::numeric_limits<std::uint32_t>::max())
            return reject("output grid does not fit the 32-bit launch description");

        bool inFamily = false;
        for(std::size_t i = 0; i < family.size; ++i)
            if(family.candidates[i] == plan)
            {
                inFamily = true;
                break;
            }
        if(!inFamily)
            return reject("plan is not part of the compiled candidate family and cannot be dispatched");

        return PlanValidation{true, {}};
    }

    struct CandidateOutcome
    {
        GemmPlan plan{};
        bool legal = false;
        std::string reason{};
        double score = 0.0;
    };

    /** Everything a caller or a test needs in order to explain a selection without a device. */
    struct PlanDiagnostics
    {
        GemmFacts facts{};
        std::vector<std::string> factsUsed{};
        std::vector<std::string> missingFacts{};
        std::string selectedEngine{};
        GemmPlan selected{};
        GemmNumericMode mode = GemmNumericMode::fp32;
        std::uint64_t stagingBytes = 0;
        std::string registerPressureNote{
            "register pressure is a documented estimate (TM*TN accumulators plus operands and temporaries), not a "
            "compiler resource report"};
        std::vector<CandidateOutcome> candidates{};
        std::string fallbackReason{};
        bool overrideUsed = false;
        double selectedScore = 0.0;
    };

    struct PlanSelection
    {
        GemmPlan plan{};
        PlanDiagnostics diagnostics{};
    };

    /** Request constraints and the optional expert override. */
    struct GemmPlanRequest
    {
        /** Empty means no policy constraint; the caller resolves the shipped automatic default. */
        std::optional<GemmStaging> policy{};
        /** Cooperative stage count requested by the caller; ignored for the direct policy. */
        std::uint32_t stages = 1;
        /** Explicit plan; when set it is validated with the same checks as an automatic plan. */
        std::optional<GemmPlan> expertOverride{};
    };

    inline void fillFactsDiagnostics(PlanDiagnostics& diagnostics, GemmFacts const& facts)
    {
        diagnostics.facts = facts;
        auto const record = [&](auto const& fact)
        {
            if(fact.known())
                diagnostics.factsUsed.push_back(
                    std::string{fact.name} + "=" + std::to_string(fact.value.value()) + " " + fact.units);
            else
                diagnostics.missingFacts.push_back(
                    std::string{fact.name} + " ("
                    + (fact.unavailableReason != nullptr ? fact.unavailableReason : "unknown") + ")");
        };
        record(facts.simdWidth);
        record(facts.numPipelines);
        record(facts.cacheLineBytes);
        record(facts.mQuantum);
        record(facts.nQuantum);
        record(facts.kQuantum);
        record(facts.sharedMemPerBlockBytes);
        record(facts.maxThreadsPerBlock);
        record(facts.maxBlocksPerGrid);
        record(facts.multiProcessorCount);
        record(facts.warpSize);
        record(facts.globalMemCapacityBytes);
        record(facts.l1CacheBytes);
        record(facts.l2CacheBytes);
        record(facts.registerBudgetBytes);
        record(facts.activeBlocks);
        record(facts.nativeMatrixCombinations);
        record(facts.asyncCopyAlignmentBytes);
    }

    inline void fillPlanDiagnostics(PlanDiagnostics& diagnostics, GemmPlan const& plan, GemmFacts const& facts)
    {
        diagnostics.selected = plan;
        diagnostics.stagingBytes = stagedBytes(plan, facts.accumulatorBytes);
        diagnostics.selectedEngine
            = std::string{"register-blocked FMA/SIMD, "}
              + (plan.policy == GemmStaging::cooperative ? "cooperative staging" : "direct global loads");
    }

    /** Deterministic selection.
     *
     * No timing, randomness or filesystem state participates. The plan with the highest score wins; ties break on
     * tile M, tile N, K block, stage count, policy and finally the compiled family index, so identical inputs always
     * produce an identical plan. An expert override bypasses ranking but not validation: an illegal override throws
     * with the exact validation reason instead of launching.
     */
    inline PlanSelection selectPlan(
        GemmProblem const& problem,
        GemmFacts const& facts,
        GemmCandidateFamily const& family,
        GemmPlanRequest const& request = {})
    {
        PlanSelection selection;
        fillFactsDiagnostics(selection.diagnostics, facts);
        selection.diagnostics.mode = problem.mode;

        if(request.expertOverride.has_value())
        {
            PlanValidation const validation = validatePlan(*request.expertOverride, problem, facts, family);
            if(!validation.legal)
                throw std::invalid_argument("GEMM: expert plan override rejected: " + validation.reason);
            selection.plan = *request.expertOverride;
            selection.diagnostics.overrideUsed = true;
            selection.diagnostics.selectedScore = scorePlan(selection.plan, problem, facts);
            selection.diagnostics.fallbackReason = "expert override used; ranking skipped";
            fillPlanDiagnostics(selection.diagnostics, selection.plan, facts);
            return selection;
        }

        bool const stageRequestValid = request.stages >= 1u && request.stages <= 3u;

        for(std::size_t index = 0; index < family.size; ++index)
        {
            GemmPlan const& candidate = family.candidates[index];
            CandidateOutcome outcome;
            outcome.plan = candidate;

            if(request.policy.has_value() && candidate.policy != *request.policy)
            {
                outcome.reason = candidate.policy == GemmStaging::direct
                                     ? "direct policy rejected by the explicit cooperative request"
                                     : "cooperative policy rejected by the explicit direct request";
            }
            else if(
                candidate.policy == GemmStaging::cooperative && request.policy == GemmStaging::cooperative
                && !stageRequestValid)
            {
                outcome.reason = "requested stage count " + std::to_string(request.stages) + " is not in {1, 2, 3}";
            }
            else if(
                candidate.policy == GemmStaging::cooperative && request.policy == GemmStaging::cooperative
                && candidate.stages != request.stages)
            {
                outcome.reason = "stage count " + std::to_string(candidate.stages)
                                 + " differs from the explicit request " + std::to_string(request.stages);
            }
            else
            {
                PlanValidation const validation = validatePlan(candidate, problem, facts, family);
                outcome.legal = validation.legal;
                outcome.reason = validation.reason;
                if(outcome.legal)
                    outcome.score = scorePlan(candidate, problem, facts);
            }
            selection.diagnostics.candidates.push_back(std::move(outcome));
        }

        std::optional<std::size_t> best;
        for(std::size_t index = 0; index < selection.diagnostics.candidates.size(); ++index)
        {
            CandidateOutcome const& outcome = selection.diagnostics.candidates[index];
            if(!outcome.legal)
                continue;
            if(!best.has_value())
            {
                best = index;
                continue;
            }
            CandidateOutcome const& incumbent = selection.diagnostics.candidates[*best];
            bool const better = outcome.score > incumbent.score
                                || (outcome.score == incumbent.score
                                    && std::make_tuple(
                                           outcome.plan.tileM,
                                           outcome.plan.tileN,
                                           outcome.plan.kBlock,
                                           outcome.plan.stages,
                                           outcome.plan.policy)
                                           < std::make_tuple(
                                               incumbent.plan.tileM,
                                               incumbent.plan.tileN,
                                               incumbent.plan.kBlock,
                                               incumbent.plan.stages,
                                               incumbent.plan.policy));
            if(better)
                best = index;
        }

        if(!best.has_value())
        {
            // Report the rejections of the requested policy, not the expected
            // "wrong policy" rejection of every candidate of the other policy.
            std::string reason = "GEMM: no legal plan for the problem; first rejections: ";
            std::size_t reported = 0;
            auto const appendRejections = [&](bool onlyRequestedPolicy)
            {
                for(auto const& outcome : selection.diagnostics.candidates)
                {
                    if(outcome.legal || reported >= 3u)
                        continue;
                    if(onlyRequestedPolicy && request.policy.has_value() && outcome.plan.policy != *request.policy)
                        continue;
                    if(reported > 0u)
                        reason += "; ";
                    reason += outcome.reason;
                    ++reported;
                }
            };
            appendRejections(true);
            if(reported == 0u)
                appendRejections(false);
            throw std::invalid_argument(reason);
        }

        selection.plan = selection.diagnostics.candidates[*best].plan;
        selection.diagnostics.selectedScore = selection.diagnostics.candidates[*best].score;
        fillPlanDiagnostics(selection.diagnostics, selection.plan, facts);

        if(selection.plan == family.candidates[0])
            selection.diagnostics.fallbackReason
                = "small fallback selected: the minimal direct plan was the highest-ranked legal plan";
        else if(*best != 0u)
        {
            // Count legal survivors to report a single-survivor fallback explicitly.
            std::size_t legalCount = 0;
            for(auto const& outcome : selection.diagnostics.candidates)
                legalCount += outcome.legal ? 1u : 0u;
            if(legalCount == 1u)
                selection.diagnostics.fallbackReason = "only one legal candidate under the request";
        }
        return selection;
    }
} // namespace alpaka::onHost::internal
