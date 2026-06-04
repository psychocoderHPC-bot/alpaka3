/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"

#include <cstdint>
#include <type_traits>

namespace alpaka::onAcc::internal
{
    template<typename T, uint32_t T_width>
    struct RegisterSimdRef
    {
        using value_type = T;

        static consteval uint32_t width()
        {
            return T_width;
        }

        constexpr auto load() const
        {
            return Simd<T, T_width>{[&](auto laneIdx) constexpr { return ptr[static_cast<uint32_t>(laneIdx)]; }};
        }

        template<typename T_Other, typename T_Storage>
        constexpr RegisterSimdRef& operator=(Simd<T_Other, T_width, T_Storage> const& rhs)
        {
            for(uint32_t lane = 0u; lane < T_width; ++lane)
                ptr[lane] = static_cast<T>(rhs[lane]);

            return *this;
        }

        template<typename T_Rhs>
        requires(alpaka::concepts::Convertible<T_Rhs, T> && !alpaka::concepts::Simd<std::decay_t<T_Rhs>>)
        constexpr RegisterSimdRef& operator=(T_Rhs const& rhs)
        {
            for(uint32_t lane = 0u; lane < T_width; ++lane)
                ptr[lane] = static_cast<T>(rhs);

            return *this;
        }

        T* ptr;
    };
} // namespace alpaka::onAcc::internal
