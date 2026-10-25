/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/lockstep/IsLockstepVar.hpp"
#include "alpaka/onAcc/lockstep/RegisterSimdRef.hpp"
#include "alpaka/trait.hpp"

#include <array>
#include <cstdint>
#include <type_traits>

namespace alpaka::onAcc::internal
{
    /** Return type of LockstepScope::var<T>(): distributed per-worker local storage.
     *
     * The storage is private to each worker and partitioned over the workers of the work group. Each worker owns
     * divCeil(logicalElements, workerCount) slots; its elements are addressed through the slot offset supplied when a
     * concurrent() functor is invoked. Users normally do not name this type directly but obtain a handle via
     * LockstepScope::var() and pass it to LockstepScope::concurrent().
     *
     * @tparam T value type stored per element
     * @tparam T_LogicalExtent compile-time known alpaka vector with the logical iteration space extents
     * @tparam T_WorkerExtent compile-time known alpaka vector with the number of participating workers
     */
    template<typename T, alpaka::concepts::CVector T_LogicalExtent, alpaka::concepts::CVector T_WorkerExtent>
    struct LockstepVar
    {
        using value_type = T;

        static constexpr uint32_t numElements = T_LogicalExtent{}.product();
        static constexpr uint32_t workerCount = T_WorkerExtent{}.product();
        static constexpr uint32_t numSlotsPerWorker = divCeil(numElements, workerCount);

        template<uint32_t T_width, typename T_IdxVec>
        constexpr auto bind(uint32_t slotBegin, std::array<T_IdxVec, T_width> const&)
        {
            return RegisterSimdRef<T, T_width>{storage.data() + slotBegin};
        }

        template<uint32_t T_width, typename T_IdxVec>
        constexpr auto bind(uint32_t slotBegin, std::array<T_IdxVec, T_width> const&) const
        {
            return RegisterSimdRef<T const, T_width>{storage.data() + slotBegin};
        }

        std::array<T, numSlotsPerWorker> storage;
    };

    template<typename T, alpaka::concepts::CVector T_LogicalExtent, alpaka::concepts::CVector T_WorkerExtent>
    struct IsLockstepVar<LockstepVar<T, T_LogicalExtent, T_WorkerExtent>> : std::true_type
    {
    };

    template<typename T>
    using BindValueType_t = std::conditional_t<
        isLockstepVar_v<T>,
        typename std::remove_cvref_t<T>::value_type,
        alpaka::GetValueType_t<std::decay_t<T>>>;
} // namespace alpaka::onAcc::internal
