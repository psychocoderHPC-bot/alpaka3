/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/trait.hpp"

#include <array>
#include <cstdint>
#include <type_traits>

namespace alpaka::onAcc::internal
{
    template<typename T_Data, typename T_IdxVec, uint32_t T_width>
    struct IndexedDataSimdRef
    {
        using value_type = alpaka::trait::GetValueType_t<std::decay_t<T_Data>>;

        static consteval uint32_t width()
        {
            return T_width;
        }

        constexpr auto load() const
        {
            return Simd<value_type, T_width>{
                [&](auto laneIdx) constexpr { return (*data)[idx[static_cast<uint32_t>(laneIdx)]]; }};
        }

        template<typename T_Other, typename T_Storage>
        constexpr IndexedDataSimdRef& operator=(Simd<T_Other, T_width, T_Storage> const& rhs)
        {
            for(uint32_t lane = 0u; lane < T_width; ++lane)
                (*data)[idx[lane]] = static_cast<value_type>(rhs[lane]);

            return *this;
        }

        template<typename T_Rhs>
        requires(alpaka::concepts::Convertible<T_Rhs, value_type> && !alpaka::concepts::Simd<std::decay_t<T_Rhs>>)
        constexpr IndexedDataSimdRef& operator=(T_Rhs const& rhs)
        {
            for(uint32_t lane = 0u; lane < T_width; ++lane)
                (*data)[idx[lane]] = static_cast<value_type>(rhs);

            return *this;
        }

        T_Data* data;
        std::array<T_IdxVec, T_width> idx;
    };
} // namespace alpaka::onAcc::internal
