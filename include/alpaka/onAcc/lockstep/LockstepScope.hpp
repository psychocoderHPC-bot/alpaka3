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

    template<typename T_Acc, typename T_LogicalExtent, typename T_WorkGroup>
    struct LockstepScope
    {
        using LogicalExtent = T_LogicalExtent;
        using IdxType = typename T_LogicalExtent::type;
        using WorkerSpace = internal::WorkerSpace_t<T_Acc, T_WorkGroup>;
        using WorkerExtent = decltype(std::declval<WorkerSpace const&>().size());
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

        static consteval uint32_t dim()
        {
            return T_LogicalExtent::dim();
        }

        constexpr auto getLogicalExtent() const
        {
            return m_logicalExtent;
        }

        constexpr auto getWorkerSpace() const
        {
            if constexpr(hasLazyGetThreadSpace)
                return m_workGroup.getThreadSpace(m_acc);
            else
                return ThreadSpace{m_workGroup.idx(m_acc), m_workGroup.size(m_acc)};
        }

        template<typename T>
        constexpr auto var() const
            requires(alpaka::concepts::CVector<T_LogicalExtent> && alpaka::concepts::CVector<WorkerExtent>)
        {
            return internal::LockstepVar<T, T_LogicalExtent, std::decay_t<WorkerExtent>>{};
        }

        template<typename T_ValueType = void, typename T_Fn>
        ALPAKA_FN_ACC constexpr void concurrent(T_Fn&& fn) const
        {
            if constexpr(std::is_void_v<T_ValueType>)
                foreachImpl<1u>(ALPAKA_FORWARD(fn));
            else
                foreachImpl<calcSimdWidth<T_ValueType>()>(ALPAKA_FORWARD(fn));
        }

        template<typename T_ValueType = void, typename T_Fn, typename T_Arg0, typename... T_Args>
        ALPAKA_FN_ACC constexpr void concurrent(T_Fn&& fn, T_Arg0&& arg0, T_Args&&... args) const
        {
            using ValueType = std::conditional_t<
                std::is_void_v<T_ValueType>,
                internal::BindValueType_t<T_Arg0>,
                T_ValueType>;
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
            auto const workerIdx = linearize(workerSpace.size(), workerSpace.idx());
            auto const workerCount = workerSpace.size().product();

            uint32_t localSlot = 0u;
            for(IdxType linearIdx = workerIdx; linearIdx < logicalSize;)
            {
                if constexpr(T_width > 1u)
                {
                    auto const lastLinearIdx
                        = linearIdx + static_cast<IdxType>(T_width - 1u) * static_cast<IdxType>(workerCount);

                    if(lastLinearIdx < logicalSize)
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

    template<alpaka::concepts::CVector T_LogicalExtent>
    ALPAKA_FN_HOST_ACC constexpr auto makeLockstep(
        auto const& acc,
        auto const& workGroup,
        T_LogicalExtent const& logicalExtent)
    {
        return LockstepScope<std::decay_t<ALPAKA_TYPEOF(acc)>, T_LogicalExtent, std::decay_t<ALPAKA_TYPEOF(workGroup)>>{
            acc,
            workGroup,
            logicalExtent};
    }
} // namespace alpaka::onAcc
