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
        alpaka::trait::GetValueType_t<std::decay_t<T>>>;
} // namespace alpaka::onAcc::internal
