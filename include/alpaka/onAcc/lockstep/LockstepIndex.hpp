/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/core/common.hpp"

#include <array>
#include <cstdint>

namespace alpaka::onAcc::internal
{
    template<typename T_LogicalExtent>
    ALPAKA_FN_HOST_ACC constexpr auto delinearize(T_LogicalExtent const& extents, std::integral auto linearIdx)
    {
        using VecType = typename T_LogicalExtent::UniVec;
        using IdxType = typename VecType::value_type;

        auto idx = VecType::fill(0u);
        auto remaining = static_cast<IdxType>(linearIdx);

        for(int32_t d = static_cast<int32_t>(VecType::dim()) - 1; d >= 0; --d)
        {
            idx[static_cast<uint32_t>(d)] = remaining % extents[static_cast<uint32_t>(d)];
            remaining /= extents[static_cast<uint32_t>(d)];
        }

        return idx;
    }

    /** Index of the element(s) processed by a worker in the current lockstep iteration.
     *
     * The index is passed as the first parameter of the functor given to LockstepScope::concurrent(). Its
     * representation depends on the SIMD width of the iteration: when width()==1 the index is scalar
     * (LockstepIndex<T_Idx, T_dim, 1u>), when width()>1 a pack of width() elements is processed and each index
     * component is a Simd value (LockstepIndex<T_Idx, T_dim, T_width>).
     *
     * @tparam T_Idx scalar index value type
     * @tparam T_dim number of dimensions of the logical iteration space
     * @tparam T_width number of SIMD lanes; 1 means scalar access
     */
    template<typename T_Idx, uint32_t T_dim, uint32_t T_width>
    struct LockstepIndex;

    /** Scalar lockstep index (SIMD width 1). */
    template<typename T_Idx, uint32_t T_dim>
    struct LockstepIndex<T_Idx, T_dim, 1u>
    {
        using value_type = T_Idx;
        using VecType = Vec<T_Idx, T_dim>;

        /** @return number of dimensions of the logical iteration space */
        static consteval uint32_t dim()
        {
            return T_dim;
        }

        /** @return SIMD width, always 1 for the scalar index */
        static consteval uint32_t width()
        {
            return 1u;
        }

        /** Access one dimension of the multi-dimensional index.
         *
         * @param d dimension to query
         * @return scalar index within dimension d
         */
        constexpr auto operator[](uint32_t d) const
        {
            return mdIdx[d];
        }

        /** @return scalar linear index of the processed element within the logical iteration space */
        constexpr auto linear() const
        {
            return linearIdx;
        }

        VecType mdIdx;
        T_Idx linearIdx;
    };

    /** SIMD lockstep index processing a pack of width() elements per worker. */
    template<typename T_Idx, uint32_t T_dim, uint32_t T_width>
    struct LockstepIndex
    {
        /** scalar index value type */
        using value_type = T_Idx;
        using SimdType = Simd<T_Idx, T_width>;
        using VecType = Vec<SimdType, T_dim>;

        /** @return number of dimensions of the logical iteration space */
        static consteval uint32_t dim()
        {
            return T_dim;
        }

        /** @return number of SIMD lanes processed by this index */
        static consteval uint32_t width()
        {
            return T_width;
        }

        /** Access one dimension of the multi-dimensional index.
         *
         * @param d dimension to query
         * @return Simd value holding the per-lane index within dimension d
         */
        constexpr auto operator[](uint32_t d) const
        {
            return mdIdx[d];
        }

        /** @return Simd value holding the per-lane linear indices within the logical iteration space */
        constexpr auto linear() const
        {
            return linearIdx;
        }

        VecType mdIdx;
        SimdType linearIdx;
    };

    template<uint32_t T_width, typename T_LogicalExtent>
    ALPAKA_FN_HOST_ACC constexpr auto makeLaneIdx(
        T_LogicalExtent const& logicalExtent,
        typename T_LogicalExtent::value_type linearIdxBegin,
        typename T_LogicalExtent::value_type linearStride)
    {
        using IdxVec = typename T_LogicalExtent::UniVec;
        std::array<IdxVec, T_width> idx{};

        for(uint32_t lane = 0u; lane < T_width; ++lane)
            idx[lane] = delinearize(
                logicalExtent,
                linearIdxBegin + linearStride * static_cast<typename T_LogicalExtent::value_type>(lane));

        return idx;
    }

    template<uint32_t T_width, typename T_IdxVec>
    ALPAKA_FN_HOST_ACC constexpr auto makeLockstepIndex(
        std::array<T_IdxVec, T_width> const& laneIdx,
        typename T_IdxVec::value_type linearIdxBegin,
        typename T_IdxVec::value_type linearStride)
    {
        using IdxType = typename T_IdxVec::value_type;

        if constexpr(T_width == 1u)
        {
            return LockstepIndex<IdxType, T_IdxVec::dim(), 1u>{laneIdx[0], linearIdxBegin};
        }
        else
        {
            using SimdType = Simd<IdxType, T_width>;
            auto mdIdx = Vec<SimdType, T_IdxVec::dim()>{
                [&](auto dimIdx) constexpr
                {
                    return SimdType{
                        [&](auto laneIdxConst) constexpr
                        { return laneIdx[static_cast<uint32_t>(laneIdxConst)][static_cast<uint32_t>(dimIdx)]; }};
                }};

            auto linearIdx = SimdType{[&](auto laneIdxConst) constexpr
                                      { return linearIdxBegin + linearStride * static_cast<IdxType>(laneIdxConst); }};

            return LockstepIndex<IdxType, T_IdxVec::dim(), T_width>{mdIdx, linearIdx};
        }
    }
} // namespace alpaka::onAcc::internal
