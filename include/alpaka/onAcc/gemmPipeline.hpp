/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/staging.hpp"

#include <algorithm>
#include <cstdint>

// clang-format off
/** @file
 *
 * Explicit bounded stage pipeline for the common GEMM kernel.
 *
 * The pipeline is the schedule around one generation-tracked stage ring. It makes the prologue, the steady state and
 * the drain explicit instead of leaving the loop structure inside the GEMM body:
 *
 * | Phase       | Operations                                                         | Invariant |
 * | ---         | ---                                                                | --- |
 * | prologue    | acquire(i) -> fill(i) -> commit(i) for `i < min(S, L)`             | every tile is filled exactly once; slot `i % S` is free before its acquisition |
 * | steadyState | wait(i) -> consume(i) -> compute(i) -> complete(i) -> release(i)   | tile `i` reads generation `tokenOf(i)`; the slot is released before `i + S` is acquired |
 * |             | -> acquire(i+S) -> fill(i+S) -> commit(i+S) if `i + S < L`        | every refill has exactly one matching steady-state consume |
 * | drain       | wait/consume/compute/complete/release for the last `min(S, L)`     | no new fill is issued; all slots end free and `issued == L` |
 *
 * With `F` full K tiles, one optional partial tile and stage count `S`, the tile count is `L = F + (R ? 1 : 0)`.
 * Tokens are the existing `staging::StageTracker` tokens and carry `(stageIndex, generation)`. Every transition uses
 * the tracker's generation checks, so a completion for generation `g` can never release a slot refilled for `g + 1`
 * and a slot is never reused before all of its readers signalled `complete()`.
 *
 * Stage policy. `StagePolicy::asynchronous` is reserved. It follows the identical schedule and state transitions as
 * the synchronous policy today and becomes distinct only when a capable, tested asynchronous copy primitive is
 * available. No backend branch and no native copy is present in this component.
 */
// clang-format on

namespace alpaka::onAcc::gemm::pipeline
{
    /** Explicit phase of one schedule step. */
    enum class Phase : std::uint8_t
    {
        /** Initial fills before the first tile is consumed. */
        prologue,
        /** Consume-and-refill steps that keep the ring full. */
        steadyState,
        /** Final consumes after the last fill; no refill is issued. */
        drain
    };

    /** Operand-supply policy of the pipeline.
     *
     * `asynchronous` is reserved. It maps to the synchronous implementation until a capable copy primitive exists.
     */
    enum class StagePolicy : std::uint8_t
    {
        synchronous,
        asynchronous
    };

    constexpr char const* phaseName(Phase phase) noexcept
    {
        switch(phase)
        {
        case Phase::prologue:
            return "prologue";
        case Phase::steadyState:
            return "steady-state";
        case Phase::drain:
            return "drain";
        }
        return "unknown";
    }

    constexpr char const* stagePolicyName(StagePolicy policy) noexcept
    {
        switch(policy)
        {
        case StagePolicy::synchronous:
            return "synchronous";
        case StagePolicy::asynchronous:
            return "asynchronous(reserved, synchronous implementation)";
        }
        return "unknown";
    }

    /** Compile-time properties of a stage policy.
     *
     * `asyncRequested` records the caller's intent. `asyncActive` is false for both policies today; it becomes true
     * only when a tested native copy primitive is wired into this component.
     */
    template<StagePolicy T_policy>
    struct PolicyTraits
    {
        static constexpr StagePolicy policy = T_policy;
        static constexpr bool asyncRequested = T_policy == StagePolicy::asynchronous;
        static constexpr bool asyncActive = false;
    };

    /** Number of K tiles for `fullTiles` full blocks plus an optional partial block. */
    constexpr std::uint64_t tileCount(std::uint64_t fullTiles, bool partialTile) noexcept
    {
        return fullTiles + (partialTile ? 1u : 0u);
    }

    /** Closed-form operation counts of the bounded schedule for a tile count and a stage count. */
    struct PhaseCounts
    {
        std::uint64_t prologueFills{};
        std::uint64_t refills{};
        std::uint64_t waits{};
        std::uint64_t computes{};
        std::uint64_t completes{};
        std::uint64_t releases{};
        std::uint64_t steadyComputes{};
        std::uint64_t drainComputes{};

        constexpr std::uint64_t fills() const noexcept
        {
            return prologueFills + refills;
        }

        constexpr bool operator==(PhaseCounts const&) const = default;
    };

    /** Expected counts of the schedule for a tile count and stage count.
     *
     * Every tile is waited for, computed, completed and released exactly once. The prologue fills `min(S, L)` slots,
     * so exactly `max(0, L - S)` refills remain. Each refill belongs to one steady-state consume, and the final
     * `min(S, L)` consumes run in the drain phase.
     */
    constexpr PhaseCounts scheduleCounts(std::uint64_t tiles, std::uint32_t stages) noexcept
    {
        ALPAKA_ASSERT_ACC(stages > 0u);
        std::uint64_t const stageCount = static_cast<std::uint64_t>(stages);
        std::uint64_t const prologue = tiles < stageCount ? tiles : stageCount;
        std::uint64_t const refills = tiles > stageCount ? tiles - stageCount : 0u;
        PhaseCounts counts;
        counts.prologueFills = prologue;
        counts.refills = refills;
        counts.waits = tiles;
        counts.computes = tiles;
        counts.completes = tiles;
        counts.releases = tiles;
        counts.steadyComputes = refills;
        counts.drainComputes = tiles - refills;
        return counts;
    }

    /** Drive the bounded schedule over a generation-tracked stage ring.
     *
     * @tparam T_policy stage policy; both policies execute the same synchronous schedule today
     * @tparam T_stages ring depth
     * @tparam T_ring ring type with `tracker`, `acquire()`, `commit(acc, token)`, `waitReady(acc, token)`,
     *         `consume(token)`, `complete(acc, token)` and `release(token)`
     *
     * @param fill    called as `fill(batchIndex, token, phase)`; phase is prologue or steadyState
     * @param compute called as `compute(batchIndex, token, phase)`; phase is steadyState or drain
     */
    template<StagePolicy T_policy, std::uint32_t T_stages, typename T_ring>
    struct BoundedStagePipeline
    {
        static_assert(T_stages > 0u, "A stage pipeline needs at least one stage.");
        static_assert(
            T_policy == StagePolicy::synchronous || T_policy == StagePolicy::asynchronous,
            "Unknown stage policy.");

        using Traits = PolicyTraits<T_policy>;

        T_ring ring;

        template<typename T_Acc, typename T_Fill, typename T_Compute>
        ALPAKA_FN_ACC void run(
            T_Acc const& acc,
            std::uint64_t fullTiles,
            bool partialTile,
            T_Fill&& fill,
            T_Compute&& compute)
        {
            std::uint64_t const tiles = tileCount(fullTiles, partialTile);
            std::uint64_t const prologueTiles = std::min<std::uint64_t>(T_stages, tiles);

            // Prologue: fill min(S, L) distinct slots. No slot is acquired twice because none is released yet and
            // the ring has at least prologueTiles free slots.
            for(std::uint64_t i = 0u; i < prologueTiles; ++i)
            {
                auto const token = ring.acquire();
                ALPAKA_ASSERT_ACC(token == ring.tracker.tokenOf(i));
                fill(i, token, Phase::prologue);
                ring.commit(acc, token);
            }

            // Steady state and drain share one consume order. A refill is issued only while i + S < L, so the last
            // min(S, L) tiles never refill a slot and the drain leaves the ring empty.
            for(std::uint64_t i = 0u; i < tiles; ++i)
            {
                bool const refill = i + T_stages < tiles;
                Phase const phase = refill ? Phase::steadyState : Phase::drain;
                auto const token = ring.tracker.tokenOf(i);
                ring.waitReady(acc, token);
                ring.consume(token);
                compute(i, token, phase);
                ring.complete(acc, token);
                ring.release(token);
                if(refill)
                {
                    auto const next = ring.acquire();
                    ALPAKA_ASSERT_ACC(next == ring.tracker.tokenOf(i + T_stages));
                    fill(i + T_stages, next, Phase::steadyState);
                    ring.commit(acc, next);
                }
            }

            // Drain invariant: every slot returned to the free list, every tile acquired exactly once.
            ALPAKA_ASSERT_ACC(ring.tracker.allFree());
            ALPAKA_ASSERT_ACC(ring.tracker.issuedCount() == tiles);
        }
    };
} // namespace alpaka::onAcc::gemm::pipeline
