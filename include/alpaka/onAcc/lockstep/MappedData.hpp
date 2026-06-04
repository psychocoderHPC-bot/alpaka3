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

    template<typename T_Data, typename T_Mapper>
    struct MappedData
    {
        T_Data data;
        T_Mapper mapper;

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

    template<typename T_Data>
    constexpr auto map(T_Data&& data)
    {
        return MappedData<std::decay_t<T_Data>, internal::IdentityMap>{
            ALPAKA_FORWARD(data),
            internal::IdentityMap{}};
    }

    template<typename T_Data, typename T_Offset>
    constexpr auto map(T_Data&& data, T_Offset&& offset)
    {
        return MappedData<std::decay_t<T_Data>, internal::OffsetMap<std::decay_t<T_Offset>>>{
            ALPAKA_FORWARD(data),
            internal::OffsetMap<std::decay_t<T_Offset>>{ALPAKA_FORWARD(offset)}};
    }
} // namespace alpaka::onAcc
