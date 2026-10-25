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
    /** SIMD reference to per-lane indexed elements of data.
     *
     * Bound for a lockstep iteration to a set of per-lane multi-dimensional indices. load() gathers the referenced
     * elements into a Simd value and assignment scatters a Simd value (or a broadcast scalar) back to the referenced
     * elements.
     *
     * @tparam T_Data type of the referenced data (may be const-qualified)
     * @tparam T_IdxVec per-lane multi-dimensional index vector type
     * @tparam T_width number of SIMD lanes
     */
    template<typename T_Data, typename T_IdxVec, uint32_t T_width>
    struct IndexedDataSimdRef
    {
        /** value type of the referenced elements */
        using value_type = alpaka::GetValueType_t<std::decay_t<T_Data>>;

        /** @return number of SIMD lanes of this reference */
        static consteval uint32_t width()
        {
            return T_width;
        }

        /** Gather the referenced per-lane elements.
         *
         * @return a Simd value holding the element referenced by each lane
         */
        constexpr auto load() const
        {
            return Simd<value_type, T_width>{[&](auto laneIdx) constexpr
                                             { return (*data)[idx[static_cast<uint32_t>(laneIdx)]]; }};
        }

        /** Scatter a Simd value to the referenced per-lane elements.
         *
         * @param rhs SIMD value whose lane i is written to the element referenced by lane i
         * @return reference to this object
         */
        template<typename T_Other, typename T_Storage>
        constexpr IndexedDataSimdRef& operator=(Simd<T_Other, T_width, T_Storage> const& rhs)
        {
            for(uint32_t lane = 0u; lane < T_width; ++lane)
                (*data)[idx[lane]] = static_cast<value_type>(rhs[lane]);

            return *this;
        }

        /** Broadcast-assign a scalar to all referenced elements.
         *
         * @param rhs scalar value written to every referenced element
         * @return reference to this object
         */
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
