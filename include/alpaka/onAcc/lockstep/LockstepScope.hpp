/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/ThreadSpace.hpp"
#include "alpaka/mem/concepts/IDataSource.hpp"
#include "alpaka/onAcc/WorkerGroup.hpp"
#include "alpaka/onAcc/lockstep/IndexedDataSimdRef.hpp"
#include "alpaka/onAcc/lockstep/LockstepIndex.hpp"
#include "alpaka/onAcc/lockstep/LockstepVar.hpp"
#include "alpaka/onAcc/lockstep/WorkerSpaceType.hpp"
#include "alpaka/trait.hpp"

#include <array>
#include <cstdint>
#include <type_traits>

namespace alpaka::onAcc
{
    namespace internal
    {
        template<uint32_t T_width, typename T_Arg, typename T_IdxVec>
        constexpr auto bindArg(T_Arg& arg, uint32_t slotBegin, std::array<T_IdxVec, T_width> const& idx)
        {
            if constexpr(requires { arg.template bind<T_width>(slotBegin, idx); })
                return arg.template bind<T_width>(slotBegin, idx);
            else
                return IndexedDataSimdRef<T_Arg, T_IdxVec, T_width>{&arg, idx};
        }

        template<typename T_Acc, typename T_ValueType>
        consteval uint32_t getLockstepSimdWidth()
        {
            using Api = ALPAKA_TYPEOF(std::declval<T_Acc>().getApi());
            using DeviceKind = ALPAKA_TYPEOF(std::declval<T_Acc>().getDeviceKind());

            return std::max(getArchSimdWidth<T_ValueType>(Api{}, DeviceKind{}), 1u);
        }

        template<typename T_ExplicitValueType, typename T_Arg>
        struct ConcurrentValueType
        {
            using type = T_ExplicitValueType;
        };

        template<typename T_Arg>
        struct ConcurrentValueType<void, T_Arg>
        {
            using type = BindValueType_t<T_Arg>;
        };

        template<typename T_ExplicitValueType, typename T_Arg>
        using ConcurrentValueType_t = typename ConcurrentValueType<T_ExplicitValueType, T_Arg>::type;

        template<typename T_Acc, typename T_ValueType, alpaka::concepts::CVector T_LogicalExtent>
        consteval uint32_t getLockstepSimdWidth(T_LogicalExtent const&)
        {
            using Api = ALPAKA_TYPEOF(std::declval<T_Acc>().getApi());
            using DeviceKind = ALPAKA_TYPEOF(std::declval<T_Acc>().getDeviceKind());
            constexpr uint32_t maxArchSimdWidth = getArchSimdWidth<T_ValueType>(Api{}, DeviceKind{});
            constexpr uint32_t cachelineBytes = getCachelineSize(Api{}, DeviceKind{});
            constexpr uint32_t maxWidthAllowed = cachelineBytes / sizeof(T_ValueType);
            constexpr uint32_t clampedWidth = std::max(std::min(maxArchSimdWidth, maxWidthAllowed), 1u);
            constexpr uint32_t simdWidth = std::bit_floor(clampedWidth);
            return std::max(std::min(simdWidth, T_LogicalExtent{}.product()), 1u);
        }

        template<
            typename T_Acc,
            typename T_ValueType,
            alpaka::concepts::CVector T_LogicalExtent,
            alpaka::concepts::CVector T_WorkerExtent>
        consteval uint32_t getLockstepSimdWidth(T_LogicalExtent const&, T_WorkerExtent const&)
        {
            constexpr uint32_t baseWidth = getLockstepSimdWidth<T_Acc, T_ValueType>(T_LogicalExtent{});
            constexpr uint32_t maxOwnedElements = divCeil(T_LogicalExtent{}.product(), T_WorkerExtent{}.product());
            return std::max(std::min(baseWidth, std::bit_floor(maxOwnedElements)), 1u);
        }
    } // namespace internal

    /** Lockstep execution scope over a compile-time known logical iteration space.
     *
     * A scope is created via onAcc::makeLockstep() and distributes the logical iteration space over the workers of the
     * work group. Iterations are processed in lockstep, i.e. all workers execute the same code path and only the
     * processed index differs. Within concurrent() the first functor argument is the lockstep index describing the
     * current (pack of) element(s).
     *
     * @tparam T_Acc accelerator type the scope is bound to
     * @tparam T_LogicalExtent compile-time known alpaka vector describing the number of logical elements per dimension
     * @tparam T_WorkGroup work group description defining which threads participate and how indices map to them
     */
    template<typename T_Acc, typename T_LogicalExtent, typename T_WorkGroup>
    struct LockstepScope
    {
        using LogicalExtent = T_LogicalExtent;
        using IdxType = typename T_LogicalExtent::value_type;
        using WorkerSpace = internal::WorkerSpace_t<T_Acc, T_WorkGroup>;
        using WorkerExtent = decltype(std::declval<WorkerSpace const&>().getThreadCount());
        static constexpr bool hasLazyGetThreadSpace
            = requires(T_WorkGroup const& workGroup, T_Acc const& acc) { workGroup.getThreadSpace(acc); };

        static_assert(
            alpaka::concepts::CVector<T_LogicalExtent>,
            "The lockstep logical extent must be compile time known.");

        constexpr LockstepScope(T_Acc const& acc, T_WorkGroup const& workGroup, T_LogicalExtent const& logicalExtent)
            : m_acc{acc}
            , m_workGroup{workGroup}
            , m_logicalExtent{logicalExtent}
        {
        }

        /** @return number of dimensions of the logical iteration space */
        static consteval uint32_t dim()
        {
            return T_LogicalExtent::dim();
        }

        /** @return the logical iteration space extents, i.e. the number of logical elements per dimension */
        constexpr auto getLogicalExtent() const
        {
            return m_logicalExtent;
        }

        /** @return the worker space, i.e. the participating thread index and thread count description */
        constexpr auto getWorkerSpace() const
        {
            if constexpr(hasLazyGetThreadSpace)
                return m_workGroup.getThreadSpace(m_acc);
            else
                return ThreadSpace{m_workGroup.getThreadIdx(m_acc), m_workGroup.getThreadCount(m_acc)};
        }

        /** Create distributed per-worker local storage usable within concurrent().
         *
         * The storage is private to each worker and partitioned over the workers of the group. Its size is derived
         * from the compile-time logical extent and the worker count, so a worker only allocates the slots it can own.
         * Passing the returned handle as an argument to concurrent() makes each worker access the
         * elements it is responsible for in the current lockstep iteration.
         *
         * @tparam T value type stored per element
         * @return a distributed storage handle to be passed as an argument to concurrent()
         */
        template<typename T>
        constexpr auto var() const
            requires(alpaka::concepts::CVector<T_LogicalExtent> && alpaka::concepts::CVector<WorkerExtent>)
        {
            return internal::LockstepVar<T, T_LogicalExtent, std::decay_t<WorkerExtent>>{};
        }

        /** Execute a functor concurrently over the logical iteration space without additional data arguments.
         *
         * @tparam T_ValueType value type used to select the SIMD width. If void (default) the scalar iteration
         *                     (width 1) is used; an explicit type selects the SIMD width best fitting that datatype
         *                     for the accelerator and logical extent.
         * @param fn functor invoked as fn(idx) where idx is the lockstep index of the processed element(s)
         */
        template<typename T_ValueType = void, typename T_Fn>
        ALPAKA_FN_ACC constexpr void concurrent(T_Fn&& fn) const
        {
            if constexpr(std::is_void_v<T_ValueType>)
                foreachImpl<1u>(ALPAKA_FORWARD(fn));
            else
                foreachImpl<calcSimdWidth<T_ValueType>()>(ALPAKA_FORWARD(fn));
        }

        /** Execute a functor concurrently over the logical iteration space with data arguments.
         *
         * Data arguments are bound to the current lockstep iteration, so each worker only accesses the elements it is
         * responsible for. Arguments such as the handle returned by var() or data wrapped via onAcc::map() are exposed
         * as SIMD references; all other sources are exposed as indexed references.
         *
         * @tparam T_ValueType value type used to select the SIMD width. If void (default) the value type is deduced
         *                     from T_Arg0; an explicit type selects the SIMD width best fitting that datatype for the
         *                     accelerator and logical extent.
         * @tparam T_Fn callable type
         * @tparam T_Arg0 type of the first data argument
         * @tparam T_Args types of the remaining data arguments
         * @param fn functor invoked as fn(idx, boundArgs...) where idx is the lockstep index of the processed
         *           element(s) and boundArgs are the per-iteration references for the data arguments
         * @param arg0 the first data argument to be bound
         * @param args the remaining data arguments to be bound
         */
        template<typename T_ValueType = void, typename T_Fn, typename T_Arg0, typename... T_Args>
        ALPAKA_FN_ACC constexpr void concurrent(T_Fn&& fn, T_Arg0&& arg0, T_Args&&... args) const
        {
            using ValueType = internal::ConcurrentValueType_t<T_ValueType, T_Arg0>;
            constexpr uint32_t simdWidth = calcSimdWidth<ValueType>();
            foreachImpl<simdWidth>(ALPAKA_FORWARD(fn), ALPAKA_FORWARD(arg0), ALPAKA_FORWARD(args)...);
        }

    private:
        template<typename T_ValueType>
        static consteval uint32_t calcSimdWidth()
        {
            if constexpr(alpaka::concepts::CVector<WorkerExtent>)
                return internal::getLockstepSimdWidth<T_Acc, T_ValueType>(T_LogicalExtent{}, WorkerExtent{});
            else
                return internal::getLockstepSimdWidth<T_Acc, T_ValueType>(T_LogicalExtent{});
        }

        template<uint32_t T_width, typename T_Fn, typename... T_Args>
        ALPAKA_FN_ACC constexpr void foreachImpl(T_Fn&& fn, T_Args&&... args) const
        {
            auto const logicalExtent = m_logicalExtent;
            auto const logicalSize = logicalExtent.product();
            auto const workerSpace = getWorkerSpace();
            auto const workerIdx = linearize(workerSpace.getThreadCount(), workerSpace.getThreadIdx());
            auto const workerCount = workerSpace.getThreadCount().product();

            uint32_t localSlot = 0u;
            for(IdxType linearIdx = workerIdx; linearIdx < logicalSize;)
            {
                if constexpr(T_width > 1u)
                {
                    auto const remainingAfterFirstLane = logicalSize - linearIdx - IdxType{1u};

                    if(remainingAfterFirstLane / workerCount >= static_cast<IdxType>(T_width - 1u))
                    {
                        auto laneIdx = internal::makeLaneIdx<T_width>(logicalExtent, linearIdx, workerCount);
                        auto idx = internal::makeLockstepIndex<T_width>(laneIdx, linearIdx, workerCount);
                        fn(idx, internal::bindArg<T_width>(args, localSlot, laneIdx)...);
                        linearIdx += static_cast<IdxType>(T_width) * static_cast<IdxType>(workerCount);
                        localSlot += T_width;
                        continue;
                    }
                }

                auto laneIdx = internal::makeLaneIdx<1u>(logicalExtent, linearIdx, workerCount);
                auto idx = internal::makeLockstepIndex<1u>(laneIdx, linearIdx, workerCount);
                fn(idx, internal::bindArg<1u>(args, localSlot, laneIdx)...);
                linearIdx += workerCount;
                ++localSlot;
            }
        }

        T_Acc const& m_acc;
        T_WorkGroup m_workGroup;
        T_LogicalExtent m_logicalExtent;
    };

    /** Create a lockstep execution scope for the given work group and logical extent.
     *
     * @tparam T_LogicalExtent compile-time known alpaka vector describing the logical iteration space extents
     * @param acc the accelerator the scope is executed on
     * @param workGroup the work group whose workers participate in the lockstep execution
     * @param logicalExtent number of logical elements per dimension
     * @return an onAcc::LockstepScope bound to acc, workGroup and logicalExtent
     */
    template<alpaka::concepts::CVector T_LogicalExtent>
    ALPAKA_FN_HOST_ACC constexpr auto makeLockstep(
        auto const& acc,
        auto const& workGroup,
        T_LogicalExtent const& logicalExtent)
    {
        return LockstepScope<
            std::decay_t<ALPAKA_TYPEOF(acc)>,
            T_LogicalExtent,
            std::decay_t<ALPAKA_TYPEOF(workGroup)>>{acc, workGroup, logicalExtent};
    }
} // namespace alpaka::onAcc
