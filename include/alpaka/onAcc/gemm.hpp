/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/fragment.hpp"
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

    /** Register-blocked outer-product GEMM tile kernel.
     *
     * One block processes one T_TM*T_WR by T_TN*T_WC output tile. Each worker owns a T_TM x T_TN register tile,
     * stored as T_TM*ceil(T_TN/V) lockstep variable slots. For every K step the worker loads T_TM A values and
     * T_TN B values once and performs T_TM*T_TN FMAs that reuse them. The accumulator tile therefore costs
     * T_TM*T_TN element accumulators, never a whole large logical tile.
     *
     * @tparam T_Accumulator accumulation value type
     * @tparam T_ViewC output view type
     * @tparam T_ViewA view type of operand A
     * @tparam T_ViewB view type of operand B
     * @tparam T_Alpha scalar type of alpha
     * @tparam T_Beta scalar type of beta
     * @tparam T_TM M-rows per worker
     * @tparam T_TN N-columns per worker
     * @tparam T_WR worker stripes in the M direction
     * @tparam T_WC worker stripes in the N direction
     * @tparam T_BK number of K elements processed per step
     */
    template<
        typename T_Accumulator,
        typename T_ViewC,
        typename T_ViewA,
        typename T_ViewB,
        typename T_Alpha,
        typename T_Beta,
        uint32_t T_TM,
        uint32_t T_TN,
        uint32_t T_WR,
        uint32_t T_WC,
        uint32_t T_BK>
    struct RegisterBlockedGemmTileKernel
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
            using AccType = std::remove_cvref_t<decltype(acc)>;
            using Api = ALPAKA_TYPEOF(std::declval<AccType const&>().getApi());
            using DeviceKind = ALPAKA_TYPEOF(std::declval<AccType const&>().getDeviceKind());
            using AccumulatorValue = Accumulator;

            constexpr uint32_t width = fragment::registerBlockedWidth<AccumulatorValue, Api, DeviceKind, T_TN>();
            using Layout = fragment::RegisterBlockedLayout<T_TM, T_TN, width>;
            using OperandLayout = fragment::OperandLayout<T_TM, T_TN, width>;
            using Ownership = fragment::RegisterBlockedOwnership<T_TM, T_TN, width, T_WR, T_WC>;
            constexpr uint32_t blockM = Ownership::blockM;
            constexpr uint32_t blockN = Ownership::blockN;
            constexpr uint32_t packs = Layout::packsPerRow;

            uint32_t const linearBlock = internal::linearBlockIdx(acc);
            uint32_t const blockCount = static_cast<uint32_t>(acc[layer::block].count().product());
            for(uint32_t tileLinear = linearBlock; tileLinear < totalTiles; tileLinear += blockCount)
            {
                uint64_t const tileRow = static_cast<uint64_t>(tileBaseM) + tileLinear / tilesN;
                uint64_t const tileCol = static_cast<uint64_t>(tileBaseN) + tileLinear % tilesN;
                uint64_t const originM = tileRow * static_cast<uint64_t>(blockM);
                uint64_t const originN = tileCol * static_cast<uint64_t>(blockN);

                auto scope = onAcc::makeLockstep(
                    acc,
                    onAcc::worker::threadsInBlock,
                    CVec<uint32_t, blockM, T_WC * Layout::storageN>{});

                auto const workerSpace = scope.getWorkerSpace();
                uint32_t const workerCount = static_cast<uint32_t>(workerSpace.getThreadCount().product());
                alpaka::unused(workerCount);
                uint32_t const workerIdx
                    = static_cast<uint32_t>(linearize(workerSpace.getThreadCount(), workerSpace.getThreadIdx()));
                ALPAKA_ASSERT_ACC(workerCount == T_WR * T_WC);
                ALPAKA_ASSERT_ACC(workerIdx < T_WR * T_WC);

                Ownership const ownership{originM, originN, workerIdx / T_WC, workerIdx % T_WC};

                auto accumulatorStorage = scope.template var<AccumulatorValue>();
                auto accumulator = fragment::makeFragment<AccumulatorValue, Layout>(accumulatorStorage, ownership);
                auto operands = fragment::makeOperandFragment<AccumulatorValue, OperandLayout>();

                accumulator.fill(AccumulatorValue{0});

                // Keep the accumulator tile in registers for the K loop and publish it back to the fragment once.
                // This is the register hoisting the M3 milestone is about; the fragment remains the owner of the
                // logical tile, its ownership mapping and its global load/store operations.
                std::array<fragment::SimdValue<AccumulatorValue, width>, Layout::slotCount> accValues{};
                fragment::staticFor<0u, Layout::slotCount>(
                    [&](auto slotConst)
                    {
                        constexpr uint32_t slot = decltype(slotConst)::value;
                        accValues[slot] = accumulator.pack(slot);
                    });

                for(uint64_t k0 = 0u; k0 < k;)
                {
                    for(uint32_t kk = 0u; kk < T_BK; ++kk)
                    {
                        uint64_t const kIndex = k0 + static_cast<uint64_t>(kk);
                        // Test every coordinate before the load; invalid operands are exact zeros.
                        operands.loadColumn(a, ownership.workerOriginM(), kIndex, m, k);
                        operands.loadRow(b, kIndex, ownership.workerOriginN(), k, n);

                        // Hoist the B row into registers once: every M-row reuses the same T_TN values.
                        std::array<fragment::SimdValue<AccumulatorValue, width>, packs> bValues{};
                        fragment::staticFor<0u, packs>(
                            [&](auto packConst)
                            {
                                constexpr uint32_t pack = decltype(packConst)::value;
                                bValues[pack] = operands.b(pack);
                            });

                        fragment::staticFor<0u, T_TM>(
                            [&](auto rowConst)
                            {
                                constexpr uint32_t row = decltype(rowConst)::value;
                                AccumulatorValue const av = operands.broadcastA(row).value;
                                fragment::staticFor<0u, packs>(
                                    [&](auto packConst)
                                    {
                                        constexpr uint32_t pack = decltype(packConst)::value;
                                        constexpr uint32_t slot = row * packs + pack;
                                        accValues[slot] = accValues[slot] + av * bValues[pack];
                                    });
                            });
                    }

                    if(k - k0 <= static_cast<uint64_t>(T_BK))
                        break;
                    k0 += static_cast<uint64_t>(T_BK);
                }

                fragment::staticFor<0u, Layout::slotCount>(
                    [&](auto slotConst)
                    {
                        constexpr uint32_t slot = decltype(slotConst)::value;
                        accumulator.setPack(slot, accValues[slot]);
                    });

                accumulator.storeTo(
                    c,
                    [&](auto const& coord, AccumulatorValue value)
                    {
                        AccumulatorValue result = static_cast<AccumulatorValue>(alpha) * value;
                        if(beta != T_Beta{0})
                        {
                            AccumulatorValue const old = static_cast<AccumulatorValue>(c.at(coord.m, coord.n));
                            if(beta == T_Beta{1})
                                result += old;
                            else
                                result += static_cast<AccumulatorValue>(beta) * old;
                        }
                        return static_cast<std::remove_cv_t<typename T_ViewC::value_type>>(result);
                    },
                    m,
                    n);
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
