/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/lockstep.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>
#include <type_traits>

/** @file
 *
 * Common-source GEMM kernels for alpaka3 lockstep.
 *
 * One output tile of T_BM x T_BN elements is processed per block. Every worker owns a distributed accumulator
 * fragment (a per-worker std::array from the lockstep scope) and all workers execute the same code path. Operands are
 * loaded directly from global memory, without shared-memory staging, and the K loop runs in T_BK sized steps.
 *
 * All boundary handling tests coordinates before dereferencing a pointer: an invalid M/N coordinate suppresses the
 * output store and an invalid K coordinate contributes an exact zero. No collective operation is issued anywhere in
 * these kernels, so a divergent lockstep tail callback is harmless here.
 */

namespace alpaka::onAcc::gemm
{
    namespace internal
    {
        /** Return one lane of a lockstep index component, independent of the SIMD width. */
        template<typename T>
        ALPAKA_FN_ACC constexpr auto laneValue(T const& value, uint32_t lane)
        {
            if constexpr(alpaka::concepts::Simd<std::decay_t<T>>)
                return value[lane];
            else
            {
                alpaka::unused(lane);
                return value;
            }
        }

        /** Linear index of the block (frame) within the launched grid. */
        template<typename T_Acc>
        ALPAKA_FN_ACC constexpr uint32_t linearBlockIdx(T_Acc const& acc)
        {
            auto const blockCount = acc[layer::block].count();
            auto const blockIdx = acc[layer::block].idx();
            return static_cast<uint32_t>(linearize(blockCount, blockIdx));
        }
    } // namespace internal

    /** GEMM tile kernel: C = alpha * op(A) * op(B) + beta * C for one output tile per block.
     *
     * @tparam T_Accumulator accumulation value type
     * @tparam T_ViewC output view type
     * @tparam T_ViewA view type of operand A
     * @tparam T_ViewB view type of operand B
     * @tparam T_Alpha scalar type of alpha
     * @tparam T_Beta scalar type of beta
     * @tparam T_BM number of output rows per tile
     * @tparam T_BN number of output columns per tile
     * @tparam T_BK number of K elements processed per step
     */
    template<
        typename T_Accumulator,
        typename T_ViewC,
        typename T_ViewA,
        typename T_ViewB,
        typename T_Alpha,
        typename T_Beta,
        uint32_t T_BM,
        uint32_t T_BN,
        uint32_t T_BK>
    struct GemmTileKernel
    {
        using Accumulator = T_Accumulator;

        T_ViewC c;
        T_ViewA a;
        T_ViewB b;
        uint64_t m;
        uint64_t n;
        uint64_t k;
        T_Alpha alpha;
        T_Beta beta;
        uint32_t tileBaseM;
        uint32_t tileBaseN;
        uint32_t tilesN;
        uint32_t totalTiles;

        ALPAKA_FN_ACC void operator()(auto const& acc) const
        {
            uint32_t const linearBlock = internal::linearBlockIdx(acc);
            uint32_t const blockCount = static_cast<uint32_t>(acc[layer::block].count().product());
            // Executors are free to launch fewer blocks than tiles (CpuSerial launches one block), so every block
            // walks the tiles assigned to it in grid-stride order. One tile is still processed at a time.
            for(uint32_t tileLinear = linearBlock; tileLinear < totalTiles; tileLinear += blockCount)
            {
                uint64_t const tileRow = static_cast<uint64_t>(tileBaseM) + tileLinear / tilesN;
                uint64_t const tileCol = static_cast<uint64_t>(tileBaseN) + tileLinear % tilesN;
                uint64_t const originM = tileRow * static_cast<uint64_t>(T_BM);
                uint64_t const originN = tileCol * static_cast<uint64_t>(T_BN);

                auto scope = onAcc::makeLockstep(acc, onAcc::worker::threadsInBlock, CVec<uint32_t, T_BM, T_BN>{});

                // One distributed accumulator fragment per worker, one value per owned output element.
                auto sums = scope.template var<Accumulator>();
                scope.template concurrent<Accumulator>(
                    [](auto const&, auto sumRef) { sumRef = Accumulator{0}; },
                    sums);

                for(uint64_t k0 = 0u; k0 < k;)
                {
                    scope.template concurrent<Accumulator>(
                        [&](auto const& idx, auto sumRef)
                        {
                            constexpr uint32_t width = std::remove_cvref_t<decltype(idx)>::width();
                            auto values = sumRef.load();
                            for(uint32_t lane = 0u; lane < width; ++lane)
                            {
                                uint64_t const i = originM + static_cast<uint64_t>(internal::laneValue(idx[0], lane));
                                uint64_t const j = originN + static_cast<uint64_t>(internal::laneValue(idx[1], lane));

                                Accumulator sum = values[lane];
                                for(uint32_t kk = 0u; kk < T_BK; ++kk)
                                {
                                    uint64_t const kIndex = k0 + static_cast<uint64_t>(kk);
                                    Accumulator av = Accumulator{0};
                                    Accumulator bv = Accumulator{0};
                                    // Test the K and M/N coordinates before any load; invalid inputs stay zero.
                                    if(kIndex < k)
                                    {
                                        if(i < m)
                                            av = static_cast<Accumulator>(a.at(i, kIndex));
                                        if(j < n)
                                            bv = static_cast<Accumulator>(b.at(kIndex, j));
                                    }
                                    sum += av * bv;
                                }
                                values[lane] = sum;
                            }
                            sumRef = values;
                        },
                        sums);

                    if(k - k0 <= static_cast<uint64_t>(T_BK))
                        break;
                    k0 += static_cast<uint64_t>(T_BK);
                }

                scope.template concurrent<Accumulator>(
                    [&](auto const& idx, auto sumRef)
                    {
                        constexpr uint32_t width = std::remove_cvref_t<decltype(idx)>::width();
                        auto values = sumRef.load();
                        for(uint32_t lane = 0u; lane < width; ++lane)
                        {
                            uint64_t const i = originM + static_cast<uint64_t>(internal::laneValue(idx[0], lane));
                            uint64_t const j = originN + static_cast<uint64_t>(internal::laneValue(idx[1], lane));
                            if(i < m && j < n)
                            {
                                Accumulator value = static_cast<Accumulator>(alpha) * values[lane];
                                if(beta != T_Beta{0})
                                {
                                    Accumulator const old = static_cast<Accumulator>(c.at(i, j));
                                    if(beta == T_Beta{1})
                                        value += old;
                                    else
                                        value += static_cast<Accumulator>(beta) * old;
                                }
                                c.at(i, j) = static_cast<std::remove_cv_t<typename T_ViewC::value_type>>(value);
                            }
                        }
                    },
                    sums);
            }
        }
    };

    /** Scale kernel for the degenerate cases K == 0 or alpha == 0: C = beta * C, without touching A or B.
     *
     * The host entry point does not launch this kernel when beta == 1, because that case is a documented no-op.
     */
    template<typename T_ViewC, typename T_Beta, uint32_t T_BM, uint32_t T_BN>
    struct ScaleTileKernel
    {
        T_ViewC c;
        uint64_t m;
        uint64_t n;
        T_Beta beta;
        uint32_t tileBaseM;
        uint32_t tileBaseN;
        uint32_t tilesN;
        uint32_t totalTiles;

        ALPAKA_FN_ACC void operator()(auto const& acc) const
        {
            using ValueType = std::remove_cv_t<typename T_ViewC::value_type>;

            uint32_t const linearBlock = internal::linearBlockIdx(acc);
            uint32_t const blockCount = static_cast<uint32_t>(acc[layer::block].count().product());
            for(uint32_t tileLinear = linearBlock; tileLinear < totalTiles; tileLinear += blockCount)
            {
                uint64_t const tileRow = static_cast<uint64_t>(tileBaseM) + tileLinear / tilesN;
                uint64_t const tileCol = static_cast<uint64_t>(tileBaseN) + tileLinear % tilesN;
                uint64_t const originM = tileRow * static_cast<uint64_t>(T_BM);
                uint64_t const originN = tileCol * static_cast<uint64_t>(T_BN);

                auto scope = onAcc::makeLockstep(acc, onAcc::worker::threadsInBlock, CVec<uint32_t, T_BM, T_BN>{});
                scope.template concurrent<ValueType>(
                    [&](auto const& idx)
                    {
                        constexpr uint32_t width = std::remove_cvref_t<decltype(idx)>::width();
                        for(uint32_t lane = 0u; lane < width; ++lane)
                        {
                            uint64_t const i = originM + static_cast<uint64_t>(internal::laneValue(idx[0], lane));
                            uint64_t const j = originN + static_cast<uint64_t>(internal::laneValue(idx[1], lane));
                            if(i < m && j < n)
                            {
                                if(beta == T_Beta{0})
                                    c.at(i, j) = static_cast<ValueType>(0);
                                else
                                    c.at(i, j) = static_cast<ValueType>(beta * c.at(i, j));
                            }
                        }
                    });
            }
        }
    };
} // namespace alpaka::onAcc::gemm
