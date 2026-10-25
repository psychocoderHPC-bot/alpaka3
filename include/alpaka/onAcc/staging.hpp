/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Assert.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"

#include <array>
#include <cstdint>

/** @file
 *
 * Synchronous cooperative staging with generation tracking.
 *
 * A stage ring is a sequence of block-shared panel buffers. All workers of a block cooperate on one stage at a time:
 *
 * | Step    | Meaning |
 * | ---     | --- |
 * | acquire | claim a free stage for a new generation; its previous generation must have been released |
 * | fill    | every worker writes its share of the panel through the pointer returned by `panel()` |
 * | commit  | publish the filled panel with a block barrier (`onAcc::syncBlockThreads`) |
 * | wait    | observe that the committed generation is ready; in the synchronous policy the commit barrier already |
 * |         | established visibility, so `waitReady()` only validates the generation |
 * | consume | mark the tile as being read; the compute phase reads the panel |
 * | complete| publish that all readers of this worker are done with a block barrier |
 * | release | return the stage to the free list; only legal after `complete()` |
 *
 * Generation rules:
 * - Every stage carries a monotonically increasing generation of its own. A token identifies `(stage, generation)`;
 *   commit, wait, consume, complete and release reject a token whose generation is not the current one.
 * - A completion for generation `g` can therefore never release a stage that has already been refilled for `g+1`.
 * - `issued` counts all acquisitions in ring order, so `token(issueIndex)` is deterministic for the bounded schedule
 *   used by the GEMM kernel and by the state-machine test.
 *
 * The barriers are the existing alpaka block barriers. No collective operation is placed inside an element-wise
 * lockstep callback, and no blocking wait depends on an unscheduled virtual worker: `fill()` is a plain strided loop
 * over all workers, so the sequential executor, which runs one thread, completes the whole fill itself.
 *
 * Storage is provided by the caller, normally `onAcc::declareSharedMdArray`. `stageStorageBytes()` reports the
 * payload; `stageMetadataReserveBytes()` is the documented reservation for the static-shared allocation header and
 * alignment. A host must reject a request whose sum exceeds the device's `sharedMemPerBlockBytes` instead of
 * launching with a silently truncated allocation.
 */

namespace alpaka::onAcc::staging
{
    /** State of one stage slot. */
    enum class Phase : std::uint8_t
    {
        free,
        filling,
        ready,
        consuming,
        completed
    };

    /** Generation-tracked ring state for T_stages stage slots.
     *
     * This type owns no storage and performs no synchronization; it is usable on the host and the accelerator. The
     * `try*` operations return false on an illegal transition or a stale generation and leave the state untouched.
     * The non-`try` operations assert on failure and are the form the kernel uses.
     */
    template<std::uint32_t T_stages>
    struct StageTracker
    {
        static_assert(T_stages > 0u, "A stage ring needs at least one stage.");

        static constexpr std::uint32_t stageCount = T_stages;

        struct Token
        {
            std::uint32_t stage{};
            std::uint64_t generation{};

            constexpr bool operator==(Token const&) const = default;
        };

        struct Slot
        {
            Phase phase{Phase::free};
            std::uint64_t generation{};
        };

        std::array<Slot, T_stages> slots{};
        std::uint64_t issued{};

        static constexpr std::uint32_t stageOf(std::uint64_t issueIndex)
        {
            return static_cast<std::uint32_t>(issueIndex % T_stages);
        }

        static constexpr std::uint64_t generationOf(std::uint64_t issueIndex)
        {
            return issueIndex / T_stages + 1u;
        }

        /** Token for the acquisition at the given ring position. Valid for issued positions and the next one. */
        constexpr Token tokenOf(std::uint64_t issueIndex) const
        {
            return Token{stageOf(issueIndex), generationOf(issueIndex)};
        }

        constexpr std::uint64_t issuedCount() const
        {
            return issued;
        }

        constexpr std::uint64_t generation(std::uint32_t stageIndex) const
        {
            ALPAKA_ASSERT_ACC(stageIndex < T_stages);
            return slots[stageIndex].generation;
        }

        constexpr bool allFree() const
        {
            for(std::uint32_t i = 0u; i < T_stages; ++i)
                if(slots[i].phase != Phase::free)
                    return false;
            return true;
        }

        /** Claim the next stage. False when the ring is not in the expected state (overwrite before release). */
        constexpr bool tryAcquire(Token& out)
        {
            Token const next = tokenOf(issued);
            Slot& slot = slots[next.stage];
            if(slot.phase != Phase::free || slot.generation + 1u != next.generation)
                return false;
            slot = Slot{Phase::filling, next.generation};
            ++issued;
            out = next;
            return true;
        }

        constexpr Token acquire()
        {
            Token token{};
            bool const acquired = tryAcquire(token);
            ALPAKA_ASSERT_ACC(acquired);
            alpaka::unused(acquired);
            return token;
        }

        /** Publish the filled generation. The caller must issue the visibility barrier afterwards. */
        constexpr bool tryCommit(Token token)
        {
            Slot& slot = slots[token.stage];
            if(slot.phase != Phase::filling || slot.generation != token.generation)
                return false;
            slot.phase = Phase::ready;
            return true;
        }

        constexpr void commit(Token token)
        {
            bool const ok = tryCommit(token);
            ALPAKA_ASSERT_ACC(ok);
            alpaka::unused(ok);
        }

        /** Whether the given generation is committed and not stale. */
        constexpr bool tryWait(Token token) const
        {
            Slot const& slot = slots[token.stage];
            return slot.phase == Phase::ready && slot.generation == token.generation;
        }

        constexpr void waitReady(Token token) const
        {
            bool const ok = tryWait(token);
            ALPAKA_ASSERT_ACC(ok);
            alpaka::unused(ok);
        }

        /** Begin reading a ready generation. */
        constexpr bool tryConsume(Token token)
        {
            if(!tryWait(token))
                return false;
            slots[token.stage].phase = Phase::consuming;
            return true;
        }

        constexpr void consume(Token token)
        {
            bool const ok = tryConsume(token);
            ALPAKA_ASSERT_ACC(ok);
            alpaka::unused(ok);
        }

        /** Signal that this worker finished reading. The caller must issue the release barrier afterwards. */
        constexpr bool tryComplete(Token token)
        {
            Slot& slot = slots[token.stage];
            if(slot.phase != Phase::consuming || slot.generation != token.generation)
                return false;
            slot.phase = Phase::completed;
            return true;
        }

        constexpr void complete(Token token)
        {
            bool const ok = tryComplete(token);
            ALPAKA_ASSERT_ACC(ok);
            alpaka::unused(ok);
        }

        /** Return a completed generation to the free list. */
        constexpr bool tryRelease(Token token)
        {
            Slot& slot = slots[token.stage];
            if(slot.phase != Phase::completed || slot.generation != token.generation)
                return false;
            slot.phase = Phase::free;
            return true;
        }

        constexpr void release(Token token)
        {
            bool const ok = tryRelease(token);
            ALPAKA_ASSERT_ACC(ok);
            alpaka::unused(ok);
        }
    };

    /** Payload bytes of T_stages panels with T_elements values each. */
    template<typename T, std::uint32_t T_stages, std::uint32_t T_elements>
    constexpr std::uint64_t stageStorageBytes()
    {
        return static_cast<std::uint64_t>(T_stages) * static_cast<std::uint64_t>(T_elements) * sizeof(T);
    }

    /** Documented reservation for the static-shared allocation header and alignment. */
    constexpr std::uint64_t stageMetadataReserveBytes()
    {
        // One metadata header plus worst-case alignment padding of the single MdSpan allocation.
        return 64u;
    }

    /** Synchronous stage ring over caller-provided block-shared storage.
     *
     * `storage` points to a contiguous array of T_stages * T_elements values. `panel()` returns the base of the
     * requested stage's panel. `commit()` and `complete()` are the only functions that synchronize; both use the
     * block barrier of the accelerator and must be called by every worker of the block.
     */
    template<typename T, std::uint32_t T_stages, std::uint32_t T_elements>
    struct SyncStageRing
    {
        static_assert(T_stages > 0u);
        static_assert(T_elements > 0u);

        using Tracker = StageTracker<T_stages>;
        using Token = typename Tracker::Token;

        Tracker tracker{};
        T* storage{};

        constexpr T* panel(Token token) const
        {
            ALPAKA_ASSERT_ACC(token.stage < T_stages);
            return storage + static_cast<std::uint64_t>(token.stage) * T_elements;
        }

        Token acquire()
        {
            return tracker.acquire();
        }

        /** Publish a filled panel to every consumer in the block. */
        template<typename T_Acc>
        void commit(T_Acc const& acc, Token token)
        {
            tracker.commit(token);
            onAcc::syncBlockThreads(acc);
        }

        /** The synchronous policy publishes in commit(), so waiting only validates the generation. */
        template<typename T_Acc>
        void waitReady(T_Acc const& acc, Token token) const
        {
            alpaka::unused(acc);
            tracker.waitReady(token);
        }

        void consume(Token token)
        {
            tracker.consume(token);
        }

        /** Publish that this worker finished reading the panel before the stage may be refilled. */
        template<typename T_Acc>
        void complete(T_Acc const& acc, Token token)
        {
            tracker.complete(token);
            onAcc::syncBlockThreads(acc);
        }

        void release(Token token)
        {
            tracker.release(token);
        }
    };
} // namespace alpaka::onAcc::staging
