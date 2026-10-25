/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/lockstep.hpp"
#include "alpaka/trait.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <type_traits>

/** @file
 *
 * Non-owning fragment views over lockstep per-worker storage.
 *
 * A fragment adds a logical register-blocked shape, an ownership mapping and a layout identity to the storage that
 * `LockstepScope::var()` already provides. It does not allocate or own storage: the `LockstepVar` handle must outlive
 * every view created from it. The ownership mapping is a pure function of the worker stripe and the SIMD width:
 *
 *     (workerRow, workerCol, slot, lane) -> (m, n)
 *     m = workerRow * TM + slot / packsPerRow
 *     n = workerCol * TN + (slot % packsPerRow) * V + lane
 *
 * One slot is one SIMD register row segment: an M-row of the worker's TM x TN tile together with one V-wide run of
 * N-columns. Lanes beyond TN are inactive padding and are never exposed as owned coordinates.
 *
 * A broadcast only reads an element of the calling worker's own storage. Cross-worker redistribution is not part of
 * this facility and requires an explicit collective.
 */

namespace alpaka::onAcc::fragment
{
    namespace internal
    {
        constexpr uint32_t ceilDiv(uint32_t numerator, uint32_t denominator)
        {
            return (numerator + denominator - 1u) / denominator;
        }
    } // namespace internal

    /** SIMD value type used by a fragment of the given element type and width. */
    template<typename T, uint32_t T_V>
    using SimdValue = Simd<T, T_V>;

    /** Compile-time ownership and storage layout of a register-blocked per-worker tile.
     *
     * @tparam T_M number of M-rows owned by one worker (TM)
     * @tparam T_N number of N-columns owned by one worker (TN)
     * @tparam T_V SIMD width used for the N direction
     */
    template<uint32_t T_M, uint32_t T_N, uint32_t T_V>
    struct RegisterBlockedLayout
    {
        static_assert(T_M > 0u, "A register-blocked tile needs at least one M-row.");
        static_assert(T_N > 0u, "A register-blocked tile needs at least one N-column.");
        static_assert(T_V > 0u, "A register-blocked tile needs a positive SIMD width.");

        static constexpr uint32_t tileM = T_M;
        static constexpr uint32_t tileN = T_N;
        static constexpr uint32_t width = T_V;
        static constexpr uint32_t packsPerRow = internal::ceilDiv(T_N, T_V);
        /** number of register row segments owned by one worker */
        static constexpr uint32_t slotCount = T_M * packsPerRow;
        /** padded N extent of one register row, a multiple of the SIMD width */
        static constexpr uint32_t storageN = packsPerRow * T_V;
        /** number of scalar elements in the per-worker storage */
        static constexpr uint32_t storageElements = T_M * storageN;

        struct Coord
        {
            uint32_t m;
            uint32_t n;

            constexpr bool operator==(Coord const&) const = default;
        };

        struct SlotLane
        {
            uint32_t slot;
            uint32_t lane;

            constexpr bool operator==(SlotLane const&) const = default;
        };

        static constexpr uint32_t rowOf(uint32_t slot)
        {
            return slot / packsPerRow;
        }

        static constexpr uint32_t packOf(uint32_t slot)
        {
            return slot % packsPerRow;
        }

        /** Whether a lane holds an owned logical element. Padded lanes are inactive. */
        static constexpr bool laneOwned(uint32_t slot, uint32_t lane)
        {
            return lane < T_V && packOf(slot) * T_V + lane < T_N;
        }

        /** Local coordinate inside the worker tile for one owned lane. */
        static constexpr Coord localCoord(uint32_t slot, uint32_t lane)
        {
            return Coord{rowOf(slot), packOf(slot) * T_V + lane};
        }

        static constexpr bool ownsLocal(uint32_t localM, uint32_t localN)
        {
            return localM < T_M && localN < T_N;
        }

        /** Inverse mapping for one local coordinate. Requires ownsLocal(localM, localN). */
        static constexpr SlotLane localSlotLane(uint32_t localM, uint32_t localN)
        {
            return SlotLane{localM * packsPerRow + localN / T_V, localN % T_V};
        }

        /** Block coordinate owned by the given worker stripe. */
        static constexpr Coord blockCoord(uint32_t workerRow, uint32_t workerCol, uint32_t slot, uint32_t lane)
        {
            Coord const local = localCoord(slot, lane);
            return Coord{workerRow * T_M + local.m, workerCol * T_N + local.n};
        }

        static constexpr bool owns(uint32_t workerRow, uint32_t workerCol, uint32_t m, uint32_t n)
        {
            return m >= workerRow * T_M && m < (workerRow + 1u) * T_M && n >= workerCol * T_N
                   && n < (workerCol + 1u) * T_N;
        }

        /** Inverse of blockCoord for a block coordinate owned by the worker stripe. */
        static constexpr SlotLane blockSlotLane(uint32_t workerRow, uint32_t workerCol, uint32_t m, uint32_t n)
        {
            ALPAKA_ASSERT_ACC(owns(workerRow, workerCol, m, n));
            return localSlotLane(m - workerRow * T_M, n - workerCol * T_N);
        }
    };

    /** SIMD width of a worker-local register-blocked fragment.
     *
     * The fragment owns a contiguous N run of TN columns in one worker, so the worker-local width uses
     * divCeil(TN, 1). The result is the power-of-two floor of the minimum of the architecture SIMD width, the
     * cache-line element count and TN, clamped to at least one. This is the M3 measured model: V is 8 for float on
     * AVX2 and 1 where the architecture exposes no SIMD width.
     */
    template<typename T_Element, typename T_Api, typename T_DeviceKind, uint32_t T_N>
    consteval uint32_t registerBlockedWidth()
    {
        constexpr uint32_t arch = getArchSimdWidth<T_Element>(T_Api{}, T_DeviceKind{});
        constexpr uint32_t cachelineElements
            = static_cast<uint32_t>(getCachelineSize(T_Api{}, T_DeviceKind{}) / sizeof(T_Element));
        constexpr uint32_t allowed = arch < cachelineElements ? arch : cachelineElements;
        constexpr uint32_t bounded = allowed < T_N ? allowed : T_N;
        return std::max(std::bit_floor(bounded), 1u);
    }

    namespace internal
    {
        template<typename T_Var, typename T_Element, typename T_Layout>
        concept compatibleStorage = onAcc::internal::IsLockstepVar<std::remove_cvref_t<T_Var>>::value
                                    && std::same_as<typename std::remove_cvref_t<T_Var>::value_type, T_Element>
                                    && (std::remove_cvref_t<T_Var>::numSlotsPerWorker >= T_Layout::storageElements)
                                    && (T_Layout::width > 0u);

        /** Whether an operation argument is explicitly marked as host-only.
         *
         * The device overloads of the fragment operations are annotated for accelerator compilation. A host view or a
         * host-only functor must therefore never be instantiated for the device pass: their `at()` or call operator
         * may call host-only code. Such arguments opt out by carrying `static constexpr bool fragmentHostOnly = true`,
         * which @ref fragment::hostOnly() adds to a view or callable. Arguments without the marker are treated as
         * accessible from the accelerator.
         */
        template<typename T>
        concept hostOnlyOperation = requires { requires std::remove_cvref_t<T>::fragmentHostOnly; };
    } // namespace internal

    /** Explicitly mark a view or callable as host-only.
     *
     * The wrapper forwards `at()` and the call operator and carries the host-only marker read by the fragment
     * operations. Use it for host-side test views and report-only functors, which must not be instantiated for the
     * device compilation pass.
     */
    template<typename T>
    struct HostOnly
    {
        T value;

        static constexpr bool fragmentHostOnly = true;

        template<typename... T_Args>
        ALPAKA_FN_HOST constexpr decltype(auto) operator()(T_Args&&... args)
        {
            return value(ALPAKA_FORWARD(args)...);
        }

        template<typename... T_Args>
        ALPAKA_FN_HOST constexpr decltype(auto) operator()(T_Args&&... args) const
        {
            return value(ALPAKA_FORWARD(args)...);
        }

        template<typename... T_Args>
        ALPAKA_FN_HOST constexpr decltype(auto) at(T_Args&&... args)
        {
            return value.at(ALPAKA_FORWARD(args)...);
        }

        template<typename... T_Args>
        ALPAKA_FN_HOST constexpr decltype(auto) at(T_Args&&... args) const
        {
            return value.at(ALPAKA_FORWARD(args)...);
        }
    };

    template<typename T>
    HostOnly(T&&) -> HostOnly<std::decay_t<T>>;

    template<typename T>
    ALPAKA_FN_HOST constexpr auto hostOnly(T&& value)
    {
        return HostOnly<std::decay_t<T>>{ALPAKA_FORWARD(value)};
    }

    /** Ownership of a register-blocked tile inside a block tile of WR x WC worker stripes.
     *
     * @tparam T_M TM
     * @tparam T_N TN
     * @tparam T_V SIMD width
     * @tparam T_WR number of worker stripes in the M direction
     * @tparam T_WC number of worker stripes in the N direction
     */
    template<uint32_t T_M, uint32_t T_N, uint32_t T_V, uint32_t T_WR, uint32_t T_WC>
    struct RegisterBlockedOwnership
    {
        static_assert(T_WR > 0u && T_WC > 0u, "A worker grid needs at least one worker per dimension.");
        using Layout = RegisterBlockedLayout<T_M, T_N, T_V>;
        static constexpr uint32_t workerRows = T_WR;
        static constexpr uint32_t workerCols = T_WC;
        static constexpr uint32_t blockM = T_WR * T_M;
        static constexpr uint32_t blockN = T_WC * T_N;

        struct GlobalCoord
        {
            uint64_t m;
            uint64_t n;
        };

        uint64_t originM{};
        uint64_t originN{};
        uint32_t workerRow{};
        uint32_t workerCol{};

        constexpr GlobalCoord globalCoord(uint32_t slot, uint32_t lane) const
        {
            typename Layout::Coord const block = Layout::blockCoord(workerRow, workerCol, slot, lane);
            return GlobalCoord{originM + block.m, originN + block.n};
        }

        constexpr uint64_t workerOriginM() const
        {
            return originM + static_cast<uint64_t>(workerRow) * T_M;
        }

        constexpr uint64_t workerOriginN() const
        {
            return originN + static_cast<uint64_t>(workerCol) * T_N;
        }
    };

    /** Non-owning view of one SIMD register row of a fragment. */
    template<typename T, typename T_Layout>
    struct RowProjection
    {
        T* data{};
        uint32_t localM{};

        static constexpr uint32_t tileN = T_Layout::tileN;
        static constexpr uint32_t width = T_Layout::width;

        static constexpr bool owns(uint32_t localN)
        {
            return localN < tileN;
        }

        constexpr T& at(uint32_t localN) const
        {
            ALPAKA_ASSERT_ACC(owns(localN));
            return data[localN];
        }
    };

    /** Non-owning view of one M-column of a fragment. */
    template<typename T, typename T_Layout>
    struct ColumnProjection
    {
        T* data{};
        uint32_t localN{};

        static constexpr uint32_t tileM = T_Layout::tileM;
        static constexpr uint32_t width = T_Layout::width;

        static constexpr bool owns(uint32_t localM)
        {
            return localM < tileM;
        }

        constexpr T& at(uint32_t localM) const
        {
            ALPAKA_ASSERT_ACC(owns(localM));
            uint32_t const slot = localM * T_Layout::packsPerRow + localN / T_Layout::width;
            return data[slot * T_Layout::width + localN % T_Layout::width];
        }
    };

    /** Value broadcast from one owned element of the calling worker.
     *
     * The value is read through the fragment, so it can only originate from storage that belongs to this worker.
     */
    template<typename T, uint32_t T_V>
    struct Broadcast
    {
        T value{};

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr Simd<T, T_V> vector() const
        {
            return Simd<T, T_V>::fill(value);
        }
    };

    /** Non-owning fragment view over per-worker lockstep storage.
     *
     * @tparam T element type
     * @tparam T_Layout register-blocked layout type
     * @tparam T_Ownership ownership type providing the block origin and worker stripe
     */
    template<typename T, typename T_Layout, typename T_Ownership>
    struct Fragment
    {
        using value_type = T;
        using layout_type = T_Layout;
        using ownership_type = T_Ownership;
        static constexpr uint32_t width = T_Layout::width;
        static constexpr uint32_t tileM = T_Layout::tileM;
        static constexpr uint32_t tileN = T_Layout::tileN;

        T* data{};
        T_Ownership ownership{};

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void fill(T value) const
        {
            for(uint32_t i = 0u; i < T_Layout::storageElements; ++i)
                data[i] = value;
        }

        /** Run a functor for every owned element of this worker.
         *
         * The functor receives the global coordinate and a reference to this worker's storage. Padded lanes and
         * coordinates outside the logical tile are never visited.
         */
        template<typename T_Fn>
        requires(!internal::hostOnlyOperation<T_Fn>)
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void concurrent(T_Fn&& fn) const
        {
            for(uint32_t slot = 0u; slot < T_Layout::slotCount; ++slot)
                for(uint32_t lane = 0u; lane < T_Layout::width; ++lane)
                    if(T_Layout::laneOwned(slot, lane))
                    {
                        auto const global = ownership.globalCoord(slot, lane);
                        fn(global, data[slot * T_Layout::width + lane]);
                    }
        }

        /** Host-only overload of @ref concurrent for callables marked with @ref hostOnly().
         *
         * The body is identical but is a plain host function, so a host-only functor is never instantiated for the
         * device compilation pass.
         */
        template<typename T_Fn>
        requires internal::hostOnlyOperation<T_Fn>
        ALPAKA_FN_INLINE ALPAKA_FN_HOST constexpr void concurrent(T_Fn&& fn) const
        {
            for(uint32_t slot = 0u; slot < T_Layout::slotCount; ++slot)
                for(uint32_t lane = 0u; lane < T_Layout::width; ++lane)
                    if(T_Layout::laneOwned(slot, lane))
                    {
                        auto const global = ownership.globalCoord(slot, lane);
                        fn(global, data[slot * T_Layout::width + lane]);
                    }
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr Simd<T, T_Layout::width> pack(uint32_t slot) const
        {
            ALPAKA_ASSERT_ACC(slot < T_Layout::slotCount);
            T* base = data + slot * T_Layout::width;
            return Simd<T, T_Layout::width>{[base](auto laneIdx) constexpr
                                            { return base[static_cast<uint32_t>(laneIdx)]; }};
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void setPack(uint32_t slot, Simd<T, T_Layout::width> const& value)
            const
        {
            ALPAKA_ASSERT_ACC(slot < T_Layout::slotCount);
            T* base = data + slot * T_Layout::width;
            for(uint32_t lane = 0u; lane < T_Layout::width; ++lane)
                base[lane] = value[lane];
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr RowProjection<T, T_Layout> row(uint32_t localM) const
        {
            ALPAKA_ASSERT_ACC(localM < tileM);
            return RowProjection<T, T_Layout>{data + localM * T_Layout::packsPerRow * T_Layout::width, localM};
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr ColumnProjection<T, T_Layout> column(uint32_t localN) const
        {
            ALPAKA_ASSERT_ACC(localN < tileN);
            return ColumnProjection<T, T_Layout>{data, localN};
        }

        /** Broadcast one owned element of this worker. No cross-worker access exists. */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr Broadcast<T, T_Layout::width> broadcast(
            uint32_t localM,
            uint32_t localN) const
        {
            ALPAKA_ASSERT_ACC(T_Layout::ownsLocal(localM, localN));
            auto const slotLane = T_Layout::localSlotLane(localM, localN);
            return Broadcast<T, T_Layout::width>{data[slotLane.slot * T_Layout::width + slotLane.lane]};
        }

        /** Load the owned tile from a global view, zero-filling out-of-range coordinates.
         *
         * Coordinates are tested before the view is dereferenced.
         */
        template<typename T_View>
        requires(!internal::hostOnlyOperation<T_View>)
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void loadFrom(T_View const& view, uint64_t totalM, uint64_t totalN)
            const
        {
            for(uint32_t slot = 0u; slot < T_Layout::slotCount; ++slot)
                for(uint32_t lane = 0u; lane < T_Layout::width; ++lane)
                {
                    auto const global = ownership.globalCoord(slot, lane);
                    T value{0};
                    if(T_Layout::laneOwned(slot, lane) && global.m < totalM && global.n < totalN)
                        value = static_cast<T>(view.at(global.m, global.n));
                    data[slot * T_Layout::width + lane] = value;
                }
        }

        /** Host-only overload of @ref loadFrom for views marked with @ref hostOnly(). */
        template<typename T_View>
        requires internal::hostOnlyOperation<T_View>
        ALPAKA_FN_INLINE ALPAKA_FN_HOST constexpr void loadFrom(T_View const& view, uint64_t totalM, uint64_t totalN)
            const
        {
            for(uint32_t slot = 0u; slot < T_Layout::slotCount; ++slot)
                for(uint32_t lane = 0u; lane < T_Layout::width; ++lane)
                {
                    auto const global = ownership.globalCoord(slot, lane);
                    T value{0};
                    if(T_Layout::laneOwned(slot, lane) && global.m < totalM && global.n < totalN)
                        value = static_cast<T>(view.at(global.m, global.n));
                    data[slot * T_Layout::width + lane] = value;
                }
        }

        /** Store the owned tile to a global view through a transform.
         *
         * The transform receives the global coordinate and the accumulator value and returns the output value.
         * Out-of-range coordinates are not stored. Padded lanes are never stored.
         */
        template<typename T_View, typename T_Fn>
        requires(!internal::hostOnlyOperation<T_View> && !internal::hostOnlyOperation<T_Fn>)
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void storeTo(
            T_View& view,
            T_Fn&& fn,
            uint64_t totalM,
            uint64_t totalN) const
        {
            for(uint32_t slot = 0u; slot < T_Layout::slotCount; ++slot)
                for(uint32_t lane = 0u; lane < T_Layout::width; ++lane)
                    if(T_Layout::laneOwned(slot, lane))
                    {
                        auto const global = ownership.globalCoord(slot, lane);
                        if(global.m < totalM && global.n < totalN)
                            view.at(global.m, global.n) = fn(global, data[slot * T_Layout::width + lane]);
                    }
        }

        /** Host-only overload of @ref storeTo for views or callables marked with @ref hostOnly(). */
        template<typename T_View, typename T_Fn>
        requires(internal::hostOnlyOperation<T_View> || internal::hostOnlyOperation<T_Fn>)
        ALPAKA_FN_INLINE ALPAKA_FN_HOST constexpr void storeTo(
            T_View& view,
            T_Fn&& fn,
            uint64_t totalM,
            uint64_t totalN) const
        {
            for(uint32_t slot = 0u; slot < T_Layout::slotCount; ++slot)
                for(uint32_t lane = 0u; lane < T_Layout::width; ++lane)
                    if(T_Layout::laneOwned(slot, lane))
                    {
                        auto const global = ownership.globalCoord(slot, lane);
                        if(global.m < totalM && global.n < totalN)
                            view.at(global.m, global.n) = fn(global, data[slot * T_Layout::width + lane]);
                    }
        }
    };

    /** Create a non-owning fragment view over a lockstep variable.
     *
     * The static checks reject an element-type mismatch and storage too small for the layout. The variable must
     * outlive the returned view.
     */
    template<typename T_Element, typename T_Layout, typename T_Ownership, typename T_Var>
    requires internal::compatibleStorage<T_Var, T_Element, T_Layout>
    ALPAKA_FN_HOST_ACC constexpr Fragment<T_Element, T_Layout, T_Ownership> makeFragment(
        T_Var& var,
        T_Ownership const& ownership)
    {
        return Fragment<T_Element, T_Layout, T_Ownership>{var.storage.data(), ownership};
    }

    /** Storage layout of the transient A/B operand fragments.
     *
     * A occupies the first TM scalar slots, B follows in packsPerRow padded SIMD rows.
     */
    template<uint32_t T_M, uint32_t T_N, uint32_t T_V>
    struct OperandLayout
    {
        static constexpr uint32_t tileM = T_M;
        static constexpr uint32_t tileN = T_N;
        static constexpr uint32_t width = T_V;
        static constexpr uint32_t packsPerRow = internal::ceilDiv(T_N, T_V);
        static constexpr uint32_t bElements = packsPerRow * T_V;
        static constexpr uint32_t storageElements = T_M + bElements;
        static constexpr uint32_t bOffset = T_M;
    };

    /** Transient A column / B row register fragment loaded from global views.
     *
     * Unlike the persistent tile fragment this is a value type: A and B are read once per K step and live in
     * registers, so the compiler can keep them across the TM*TN FMAs without a memory round trip.
     */
    template<typename T, typename T_Layout>
    struct OperandFragment
    {
        using value_type = T;
        using layout_type = T_Layout;
        static constexpr uint32_t width = T_Layout::width;

        std::array<T, T_Layout::storageElements> data{};

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void clear()
        {
            for(uint32_t i = 0u; i < T_Layout::storageElements; ++i)
                data[i] = T{0};
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr T a(uint32_t localM) const
        {
            ALPAKA_ASSERT_ACC(localM < T_Layout::tileM);
            return data[localM];
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void setA(uint32_t localM, T value)
        {
            ALPAKA_ASSERT_ACC(localM < T_Layout::tileM);
            data[localM] = value;
        }

        /** One B lane of a register row. Scalar access keeps the hot loop free of SIMD constructor calls. */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr T bLane(uint32_t pack, uint32_t lane) const
        {
            ALPAKA_ASSERT_ACC(pack < T_Layout::packsPerRow && lane < T_Layout::width);
            return data[T_Layout::bOffset + pack * T_Layout::width + lane];
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr Simd<T, T_Layout::width> b(uint32_t pack) const
        {
            ALPAKA_ASSERT_ACC(pack < T_Layout::packsPerRow);
            T const* base = data.data() + T_Layout::bOffset + pack * T_Layout::width;
            return Simd<T, T_Layout::width>{[base](auto laneIdx) constexpr
                                            { return base[static_cast<uint32_t>(laneIdx)]; }};
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void setB(uint32_t pack, Simd<T, T_Layout::width>& value)
        {
            ALPAKA_ASSERT_ACC(pack < T_Layout::packsPerRow);
            T* base = data.data() + T_Layout::bOffset + pack * T_Layout::width;
            for(uint32_t lane = 0u; lane < T_Layout::width; ++lane)
                base[lane] = value[lane];
        }

        /** Broadcast one owned A element of this worker. */
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr Broadcast<T, T_Layout::width> broadcastA(uint32_t localM) const
        {
            return Broadcast<T, T_Layout::width>{a(localM)};
        }

        /** Load TM A values at column k, zero when the M or K coordinate is out of range. */
        template<typename T_View>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void loadColumn(
            T_View const& view,
            uint64_t rowBase,
            uint64_t k,
            uint64_t totalM,
            uint64_t totalK)
        {
            for(uint32_t r = 0u; r < T_Layout::tileM; ++r)
            {
                T value{0};
                if(rowBase + r < totalM && k < totalK)
                    value = static_cast<T>(view.at(rowBase + r, k));
                setA(r, value);
            }
        }

        /** Load TN B values at row k, zero when the K or N coordinate is out of range.
         *
         * Padded lanes and out-of-range N columns become an exact zero.
         */
        template<typename T_View>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void loadRow(
            T_View const& view,
            uint64_t k,
            uint64_t colBase,
            uint64_t totalK,
            uint64_t totalN)
        {
            for(uint32_t pack = 0u; pack < T_Layout::packsPerRow; ++pack)
                for(uint32_t lane = 0u; lane < T_Layout::width; ++lane)
                {
                    uint64_t const localN = static_cast<uint64_t>(pack) * T_Layout::width + lane;
                    T value{0};
                    if(k < totalK && localN < T_Layout::tileN && colBase + localN < totalN)
                        value = static_cast<T>(view.at(k, colBase + localN));
                    data[T_Layout::bOffset + pack * T_Layout::width + lane] = value;
                }
        }
    };

    template<typename T_Element, typename T_Layout>
    ALPAKA_FN_HOST_ACC constexpr OperandFragment<T_Element, T_Layout> makeOperandFragment()
    {
        return OperandFragment<T_Element, T_Layout>{};
    }

    /** Unroll a compile-time loop without a compiler pragma. */
    template<uint32_t T_Begin, uint32_t T_End, typename T_Fn>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void staticFor(T_Fn&& fn)
    {
        if constexpr(T_Begin < T_End)
        {
            fn(std::integral_constant<uint32_t, T_Begin>{});
            staticFor<T_Begin + 1u, T_End>(ALPAKA_FORWARD(fn));
        }
    }
} // namespace alpaka::onAcc::fragment
