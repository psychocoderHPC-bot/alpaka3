/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/onAcc/lockstep/IndexedDataSimdRef.hpp"

#include <array>
#include <type_traits>

namespace alpaka::onAcc
{
    namespace internal
    {
        struct IdentityMap
        {
            template<uint32_t T_width, typename T_IdxVec>
            constexpr auto operator()(std::array<T_IdxVec, T_width> idx) const
            {
                return idx;
            }
        };

        template<typename T_Offset>
        struct OffsetMap
        {
            T_Offset offset;

            template<uint32_t T_width, typename T_IdxVec>
            constexpr auto operator()(std::array<T_IdxVec, T_width> idx) const
            {
                for(auto& laneIdx : idx)
                    laneIdx += offset;

                return idx;
            }
        };
    } // namespace internal

    /** Data wrapper applying an index mapping before lockstep access.
     *
     * Created via onAcc::map() and passed as a data argument to LockstepScope::concurrent(). When bound to an
     * iteration, the logical index of each SIMD lane is transformed by the mapper before the data is accessed, which
     * allows shifted or otherwise remapped access.
     *
     * @tparam T_Data wrapped data type
     * @tparam T_Mapper index mapping functor applied to the per-lane index vectors
     */
    template<typename T_Data, typename T_Mapper>
    struct MappedData
    {
        T_Data data;
        T_Mapper mapper;

        /** Bind the wrapped data to a lockstep iteration.
         *
         * @tparam T_width number of SIMD lanes of the current iteration
         * @tparam T_IdxVec per-lane multi-dimensional index vector type
         * @param slotBegin slot offset of this worker (unused, the mapping is index based)
         * @param idx per-lane logical indices before mapping
         * @return an indexed SIMD reference accessing data at the mapped indices
         */
        template<uint32_t T_width, typename T_IdxVec>
        constexpr auto bind([[maybe_unused]] uint32_t slotBegin, std::array<T_IdxVec, T_width> const& idx)
        {
            auto mappedIdx = mapper.template operator()<T_width>(idx);
            return internal::IndexedDataSimdRef<T_Data, T_IdxVec, T_width>{&data, mappedIdx};
        }

        template<uint32_t T_width, typename T_IdxVec>
        constexpr auto bind([[maybe_unused]] uint32_t slotBegin, std::array<T_IdxVec, T_width> const& idx) const
        {
            auto mappedIdx = mapper.template operator()<T_width>(idx);
            return internal::IndexedDataSimdRef<T_Data const, T_IdxVec, T_width>{&data, mappedIdx};
        }
    };

    /** Wrap data with an identity index mapping for lockstep concurrent() access.
     *
     * @param data the data to wrap
     * @return a MappedData wrapper exposing data at the unmodified logical indices
     */
    template<typename T_Data>
    constexpr auto map(T_Data&& data)
    {
        return MappedData<std::decay_t<T_Data>, internal::IdentityMap>{ALPAKA_FORWARD(data), internal::IdentityMap{}};
    }

    /** Wrap data with a constant index offset for lockstep concurrent() access.
     *
     * Each per-lane logical index is shifted by offset before data is accessed, enabling shifted/neighboring element
     * access within the same lockstep iteration.
     *
     * @param data the data to wrap
     * @param offset offset added to each per-lane logical index
     * @return a MappedData wrapper exposing data at shifted logical indices
     */
    template<typename T_Data, typename T_Offset>
    constexpr auto map(T_Data&& data, T_Offset&& offset)
    {
        return MappedData<std::decay_t<T_Data>, internal::OffsetMap<std::decay_t<T_Offset>>>{
            ALPAKA_FORWARD(data),
            internal::OffsetMap<std::decay_t<T_Offset>>{ALPAKA_FORWARD(offset)}};
    }
} // namespace alpaka::onAcc
