/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"

#include <cstdint>
#include <type_traits>

namespace alpaka::onAcc::internal
{
    /** SIMD reference to a contiguous range of per-worker local (register) storage.
     *
     * Returned for lockstep data arguments such as the handle created by LockstepScope::var(). load() gathers the
     * stored elements into a Simd value and assignment writes a Simd value (or a broadcast scalar) back into the
     * local storage.
     *
     * @tparam T value type of the referenced storage (may be const-qualified)
     * @tparam T_width number of SIMD lanes
     */
    template<typename T, uint32_t T_width>
    struct RegisterSimdRef
    {
        /** value type of the referenced elements */
        using value_type = T;

        /** @return number of SIMD lanes of this reference */
        static consteval uint32_t width()
        {
            return T_width;
        }

        /** Gather the referenced local elements.
         *
         * @return a Simd value holding the element referenced by each lane
         */
        constexpr auto load() const
        {
            return Simd<T, T_width>{[&](auto laneIdx) constexpr { return ptr[static_cast<uint32_t>(laneIdx)]; }};
        }

        /** Store a Simd value into the referenced local elements.
         *
         * @param rhs SIMD value whose lane i is written to local element i
         * @return reference to this object
         */
        template<typename T_Other, typename T_Storage>
        constexpr RegisterSimdRef& operator=(Simd<T_Other, T_width, T_Storage> const& rhs)
        {
            for(uint32_t lane = 0u; lane < T_width; ++lane)
                ptr[lane] = static_cast<T>(rhs[lane]);

            return *this;
        }

        /** Broadcast-assign a scalar to all referenced local elements.
         *
         * @param rhs scalar value written to every referenced element
         * @return reference to this object
         */
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
