/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/KernelBundle.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Float16.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/gemm.hpp"
#include "alpaka/onHost/Queue.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"
#include "alpaka/onHost/algo/internal/gemmPlan.hpp"
#include "alpaka/onHost/algo/internal/gemmTypes.hpp"
#include "alpaka/onHost/trait.hpp"
#include "alpaka/trait.hpp"

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

/** @file
 *
 * Public asynchronous host-side GEMM entry point.
 *
 * The operation is
 *
 *     C := alpha * op(A) * op(B) + beta * C
 *
 * with A logically M x K, B logically K x N and C logically M x N. Row-major and column-major storage, padded
 * leading dimensions, subviews and the NN/NT/TN/TT transpose combinations are supported. The single kernel is
 * common source: one output tile per block, tile configuration derived from the compiled SIMD width, the executor
 * thread limit and the problem shape.
 *
 * Zero-size semantics match the design guide:
 * - M == 0 or N == 0: no-op.
 * - K == 0 or alpha == 0: compute beta * C without reading A or B.
 * - beta == 0: the old C value is not read.
 * - beta == 1 with K == 0 (or alpha == 0): no-op.
 *
 * Validation happens on the host before anything is enqueued. Invalid dimensions, leading dimensions, overlapping
 * output/input storage and overflowing size or launch arithmetic throw std::invalid_argument.
 */

namespace alpaka::onHost
{
    namespace internal
    {
        inline std::uint64_t checkedAdd(std::uint64_t lhs, std::uint64_t rhs, char const* what)
        {
            if(lhs > std::numeric_limits<std::uint64_t>::max() - rhs)
                throw std::invalid_argument(std::string{"GEMM: 64-bit overflow while computing "} + what);
            return lhs + rhs;
        }

        inline std::uint64_t checkedMul(std::uint64_t lhs, std::uint64_t rhs, char const* what)
        {
            if(lhs != 0u && rhs > std::numeric_limits<std::uint64_t>::max() / lhs)
                throw std::invalid_argument(std::string{"GEMM: 64-bit overflow while computing "} + what);
            return lhs * rhs;
        }

        inline std::uint64_t ceilDiv(std::uint64_t numerator, std::uint64_t denominator, char const* what)
        {
            std::uint64_t const quotient = numerator / denominator;
            return numerator % denominator == 0u ? quotient : checkedAdd(quotient, 1u, what);
        }

        inline std::uint64_t asU64(std::size_t value, char const* what)
        {
            if constexpr(sizeof(std::size_t) > sizeof(std::uint64_t))
            {
                if(value > std::numeric_limits<std::uint64_t>::max())
                    throw std::invalid_argument(std::string{"GEMM: value does not fit 64 bits: "} + what);
            }
            return static_cast<std::uint64_t>(value);
        }

        /** Largest power-of-two divisor of a pointer address, or zero when the pointer is null. */
        inline std::uint64_t pointerAlignment(void const* pointer)
        {
            if(pointer == nullptr)
                return 0u;
            auto const address = reinterpret_cast<std::uintptr_t>(pointer);
            if(address == 0u)
                return 0u;
            return static_cast<std::uint64_t>(address & (std::uintptr_t{0} - address));
        }

        /** Number of elements in the stored view, including the padding of the slow dimension. */
        inline std::uint64_t storedSpanElements(
            std::uint64_t storedRows,
            std::uint64_t storedCols,
            std::uint64_t ld,
            GemmLayout layout,
            char const* what)
        {
            if(storedRows == 0u || storedCols == 0u)
                return 0u;
            std::uint64_t const fast = layout == GemmLayout::rowMajor ? storedCols : storedRows;
            std::uint64_t const slow = layout == GemmLayout::rowMajor ? storedRows : storedCols;
            if(ld < fast)
                throw std::invalid_argument(
                    std::string{"GEMM: leading dimension of "} + what + " is smaller than its fast extent");
            return checkedAdd(checkedMul(slow - 1u, ld, what), fast, what);
        }

        template<typename T>
        std::uint64_t storedSpanBytes(GemmView<T> const& view, char const* what)
        {
            std::uint64_t const rows = asU64(view.storedRows(), what);
            std::uint64_t const cols = asU64(view.storedCols(), what);
            std::uint64_t const ld = asU64(view.ld, what);
            return checkedMul(storedSpanElements(rows, cols, ld, view.layout, what), sizeof(T), what);
        }

        struct ByteRange
        {
            std::uintptr_t begin;
            std::uintptr_t end;
        };

        template<typename T>
        ByteRange byteRange(GemmView<T> const& view, char const* what)
        {
            std::uint64_t const bytes = storedSpanBytes(view, what);
            auto const begin = reinterpret_cast<std::uintptr_t>(view.data);
            if(bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
                throw std::invalid_argument(std::string{"GEMM: address range overflow for "} + what);
            return ByteRange{begin, begin + static_cast<std::uintptr_t>(bytes)};
        }

        inline bool rangesOverlap(ByteRange const& lhs, ByteRange const& rhs) noexcept
        {
            return lhs.begin < rhs.end && rhs.begin < lhs.end;
        }

        /** Compile-time block thread count derived from the executor limit and the SIMD/ILP traits. */
        template<typename T_Acc, typename T_Api, typename T_DeviceKind, typename T_Exec>
        consteval std::uint32_t blockThreadCount()
        {
            constexpr std::uint32_t maxThreads
                = std::max(onHost::getMaxThreadsPerBlock(T_Api{}, T_DeviceKind{}, T_Exec{}), 1u);
            if constexpr(maxThreads == 1u)
                return 1u;
            else
            {
                constexpr std::uint32_t perThread = std::max(getNumElemPerThread<T_Acc>(T_Api{}, T_DeviceKind{}), 1u);
                constexpr std::uint32_t wanted = std::max(1u, perThread * 2u);
                return std::min(maxThreads, std::min(256u, std::bit_floor(wanted)));
            }
        }

        struct TileConfig
        {
            std::uint32_t tile;
            std::uint32_t bk;
            std::uint32_t threads;
        };

        /** Choose tile, K step and block size from compiled traits and the problem shape.
         *
         * The set of tile variants is a bounded compile-time family {4,8,16,32}; BK is {8,16}. No device name and no
         * backend identity participates in the decision. A problem smaller than four rows or columns keeps the 4x4
         * fallback, so width-one and tiny shapes always have a correct path.
         */
        template<typename T_Acc, typename T_Api, typename T_DeviceKind, typename T_Exec>
        TileConfig chooseTileConfig(std::uint64_t m, std::uint64_t n, T_Api, T_DeviceKind, T_Exec)
        {
            constexpr std::uint32_t archSimd = std::max(getArchSimdWidth<T_Acc>(T_Api{}, T_DeviceKind{}), 1u);
            constexpr std::uint32_t perThread = std::max(getNumElemPerThread<T_Acc>(T_Api{}, T_DeviceKind{}), 1u);

            std::uint64_t const minDim = std::min(m, n);
            std::uint32_t tile = std::clamp(std::bit_ceil(archSimd), 4u, 32u);
            if(minDim < 4u)
                tile = 4u;
            else if(minDim > tile && perThread >= tile)
                tile = std::min(tile * 2u, 32u);
            if(minDim > tile && perThread >= tile * 2u)
                tile = std::min(tile * 2u, 32u);

            std::uint32_t const bk = archSimd >= 8u ? 16u : 8u;

            constexpr std::uint32_t threads = blockThreadCount<T_Acc, T_Api, T_DeviceKind, T_Exec>();
            return TileConfig{tile, bk, threads};
        }

        /** The compiled candidate family for one operation, accumulator type, target and executor.
         *
         * The family is constexpr: its size and every plan field are known at compile time, so the launcher can
         * dispatch to a selected plan without making any problem dimension a template argument. On a target whose
         * block-thread limit is one the worker-column axis collapses and the family holds 32 entries; otherwise it
         * holds the full 64. `buildCandidateFamily` enforces the cap.
         */
        template<typename T_Acc, typename T_Api, typename T_DeviceKind, typename T_Exec>
        inline constexpr GemmCandidateFamily gemmCandidateFamily = buildCandidateFamily(
            std::max<std::uint32_t>(getArchSimdWidth<T_Acc>(T_Api{}, T_DeviceKind{}), 1u),
            blockThreadCount<T_Acc, T_Api, T_DeviceKind, T_Exec>(),
            1u);

        /** Compile-time register-blocked plan.
         *
         * The per-worker tile stays near 64 element accumulators: TM*TN is 64. The SIMD width is the M3 model,
         * min(architecture width, cache-line elements, TN) rounded down to a power of two, clamped to one. The block
         * tile is the worker grid times the per-worker tile; larger logical tiles come from more work groups, never
         * from a larger per-thread accumulator.
         */
        struct RegisterTileConfig
        {
            std::uint32_t tm;
            std::uint32_t tn;
            std::uint32_t wr;
            std::uint32_t wc;
            std::uint32_t bk;
        };

        template<typename T_Acc, typename T_Api, typename T_DeviceKind, typename T_Exec>
        consteval RegisterTileConfig registerTileConfig()
        {
            constexpr std::uint32_t arch = std::max(getArchSimdWidth<T_Acc>(T_Api{}, T_DeviceKind{}), 1u);
            constexpr std::uint32_t lineElements
                = static_cast<std::uint32_t>(getCachelineSize(T_Api{}, T_DeviceKind{}) / sizeof(T_Acc));
            constexpr std::uint32_t maxWidth = std::max(std::bit_floor(std::min(arch, lineElements)), 1u);
            constexpr std::uint32_t tn = maxWidth >= 8u ? 16u : (maxWidth >= 4u ? 8u : 4u);
            constexpr std::uint32_t tm = 64u / tn;
            constexpr std::uint32_t bk = arch >= 8u ? 16u : 8u;
            constexpr std::uint32_t threads = blockThreadCount<T_Acc, T_Api, T_DeviceKind, T_Exec>();
            return RegisterTileConfig{tm, tn, 1u, threads, bk};
        }

        /** Compile-time configuration value for the default operand supply policy.
         *
         * This is the single place that decides the shipped default. It stays `direct` until a planner milestone
         * selects staging from measured costs, so enabling cooperative staging is an explicit expert choice today.
         */
        inline constexpr GemmStaging defaultGemmStaging = GemmStaging::direct;

        /** Whether cooperative staging fits and is expected to amortise its panel traffic.
         *
         * Documented placeholder rule for the later planner: stage when the tile's arithmetic intensity
         * 2*bm*bn*bk / (bm*bk + bk*bn) reaches @p minIntensity and the stage ring fits the block-shared limit.
         * Both arguments are element counts, not bytes, except @p sharedBytes.
         */
        constexpr bool stagingRule(
            std::uint64_t bm,
            std::uint64_t bn,
            std::uint64_t bk,
            std::uint64_t elementBytes,
            std::uint64_t stages,
            std::uint64_t sharedBytes)
        {
            constexpr std::uint64_t minIntensity = 8u;
            std::uint64_t const panelElements = bm * bk + bk * bn;
            if(panelElements == 0u)
                return false;
            std::uint64_t const intensity = (2u * bm * bn * bk) / panelElements;
            std::uint64_t const required
                = stages * panelElements * elementBytes + onAcc::staging::stageMetadataReserveBytes();
            return intensity >= minIntensity && required <= sharedBytes;
        }

        /** Resolve an automatic request with the compile-time default, or apply the documented rule if the default
         * itself is automatic. Explicit requests are never overridden.
         */
        constexpr GemmStaging resolveStaging(
            GemmStaging requested,
            std::uint64_t bm,
            std::uint64_t bn,
            std::uint64_t bk,
            std::uint64_t elementBytes,
            std::uint64_t stages,
            std::uint64_t sharedBytes)
        {
            if(requested != GemmStaging::automatic)
                return requested;
            if(defaultGemmStaging != GemmStaging::automatic)
                return defaultGemmStaging;
            return stagingRule(bm, bn, bk, elementBytes, stages, sharedBytes) ? GemmStaging::cooperative
                                                                              : GemmStaging::direct;
        }

        struct TileRegion
        {
            std::uint64_t tileRowBegin;
            std::uint64_t tileRowEnd;
            std::uint64_t tileColBegin;
            std::uint64_t tileColEnd;
        };

        struct TileRegions
        {
            TileRegion full;
            TileRegion bottom;
            TileRegion right;
            TileRegion corner;
        };

        /** Split the output tiles into the four disjoint regions of the design guide. */
        inline TileRegions splitRegions(std::uint64_t m, std::uint64_t n, std::uint32_t bm, std::uint32_t bn)
        {
            ALPAKA_ASSERT(bm != 0u && bn != 0u);
            std::uint64_t const tileRows = ceilDiv(m, bm, "tile row count");
            std::uint64_t const tileCols = ceilDiv(n, bn, "tile column count");
            std::uint64_t const fullRows = m / bm;
            std::uint64_t const fullCols = n / bn;
            return TileRegions{
                TileRegion{0u, fullRows, 0u, fullCols},
                TileRegion{fullRows, tileRows, 0u, fullCols},
                TileRegion{0u, fullRows, fullCols, tileCols},
                TileRegion{fullRows, tileRows, fullCols, tileCols}};
        }

        template<typename T_View>
        void requireNonNull(T_View const& view, char const* what)
        {
            if(view.storedRows() != 0u && view.storedCols() != 0u && view.data == nullptr)
                throw std::invalid_argument(std::string{"GEMM: "} + what + " has elements but no data pointer");
        }

        /** Validate the complete problem before any work is enqueued.
         *
         * @return true when C is written, false for the beta==1 no-op paths.
         */
        template<typename T_C, typename T_A, typename T_B, typename T_Alpha, typename T_Beta>
        bool validateProblem(
            GemmView<T_C> const& c,
            GemmView<T_A> const& a,
            GemmView<T_B> const& b,
            std::uint64_t m,
            std::uint64_t n,
            std::uint64_t k,
            T_Alpha alpha,
            T_Beta beta)
        {
            if(c.op != GemmOp::none)
                throw std::invalid_argument("GEMM: the output view must not be transposed");
            if(asU64(a.rows, "A rows") != m || asU64(a.cols, "A cols") != k)
                throw std::invalid_argument("GEMM: A must be a logical M x K view");
            if(asU64(b.rows, "B rows") != k || asU64(b.cols, "B cols") != n)
                throw std::invalid_argument("GEMM: B must be a logical K x N view");
            if(asU64(c.rows, "C rows") != m || asU64(c.cols, "C cols") != n)
                throw std::invalid_argument("GEMM: C must be a logical M x N view");

            bool const readOperands = m != 0u && n != 0u && k != 0u && alpha != T_Alpha{0};
            bool const betaIsOne = beta == T_Beta{1};
            bool const writeOutput = m != 0u && n != 0u && !(!readOperands && betaIsOne);

            // Leading dimensions and address ranges are checked even when the view is currently unused, except for
            // empty views where no address is formed.
            ByteRange const cRange = byteRange(c, "C");
            if(readOperands)
            {
                ByteRange const aRange = byteRange(a, "A");
                ByteRange const bRange = byteRange(b, "B");
                if(rangesOverlap(cRange, aRange) || rangesOverlap(cRange, bRange))
                    throw std::invalid_argument("GEMM: the output C overlaps an input operand");
            }

            if(writeOutput)
                requireNonNull(c, "C");
            if(readOperands)
            {
                requireNonNull(a, "A");
                requireNonNull(b, "B");
            }

            // Worst-case launch count: the smallest tile variant produces the most tiles.
            std::uint64_t const maxTiles
                = checkedMul(ceilDiv(m, 4u, "launch tile rows"), ceilDiv(n, 4u, "launch tile columns"), "tile count");
            if(maxTiles > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("GEMM: tile count does not fit the 32-bit launch description");
            if(m != 0u && n != 0u && !readOperands && !betaIsOne)
            {
                // A scale-only launch uses the same grid; keep the K-step arithmetic in range explicitly.
                checkedAdd(k, 1u, "K extent");
            }
            return writeOutput;
        }
    } // namespace internal

    namespace internal
    {
        template<
            std::uint32_t T_BM,
            std::uint32_t T_BN,
            std::uint32_t T_BK,
            std::uint32_t T_Threads,
            typename T_Acc,
            typename T_Queue,
            typename T_Exec,
            typename T_C,
            typename T_A,
            typename T_B,
            typename T_Alpha,
            typename T_Beta>
        void launchGemmTiles(
            T_Queue const& queue,
            T_Exec const& exec,
            GemmView<T_C> c,
            GemmView<T_A> a,
            GemmView<T_B> b,
            std::uint64_t m,
            std::uint64_t n,
            std::uint64_t k,
            T_Alpha alpha,
            T_Beta beta,
            TileConfig const& config)
        {
            TileRegions const regions = splitRegions(m, n, T_BM, T_BN);
            auto const launch = [&](TileRegion const& region)
            {
                std::uint64_t const rows = region.tileRowEnd - region.tileRowBegin;
                std::uint64_t const cols = region.tileColEnd - region.tileColBegin;
                if(rows == 0u || cols == 0u)
                    return;
                auto const numTiles = static_cast<std::uint32_t>(checkedMul(rows, cols, "region tile count"));
                alpaka::unused(config.threads);
                queue.enqueue(
                    onHost::FrameSpec{Vec{numTiles}, CVec<std::uint32_t, T_Threads>{}, exec},
                    KernelBundle{onAcc::gemm::GemmTileKernel<
                        T_Acc,
                        GemmView<T_C>,
                        GemmView<T_A>,
                        GemmView<T_B>,
                        T_Alpha,
                        T_Beta,
                        T_BM,
                        T_BN,
                        T_BK>{
                        c,
                        a,
                        b,
                        m,
                        n,
                        k,
                        alpha,
                        beta,
                        static_cast<std::uint32_t>(region.tileRowBegin),
                        static_cast<std::uint32_t>(region.tileColBegin),
                        static_cast<std::uint32_t>(cols),
                        numTiles}});
            };
            launch(regions.full);
            launch(regions.bottom);
            launch(regions.right);
            launch(regions.corner);
        }

        template<
            std::uint32_t T_TM,
            std::uint32_t T_TN,
            std::uint32_t T_WR,
            std::uint32_t T_WC,
            std::uint32_t T_BK,
            std::uint32_t T_STAGES,
            std::uint32_t T_Threads,
            typename T_Acc,
            typename T_Queue,
            typename T_Exec,
            typename T_C,
            typename T_A,
            typename T_B,
            typename T_Alpha,
            typename T_Beta>
        void launchGemmTilesRegisterBlocked(
            T_Queue const& queue,
            T_Exec const& exec,
            GemmView<T_C> c,
            GemmView<T_A> a,
            GemmView<T_B> b,
            std::uint64_t m,
            std::uint64_t n,
            std::uint64_t k,
            T_Alpha alpha,
            T_Beta beta)
        {
            constexpr std::uint32_t bm = T_WR * T_TM;
            constexpr std::uint32_t bn = T_WC * T_TN;
            if constexpr(T_STAGES > 0u)
            {
                constexpr std::uint64_t panelElements
                    = static_cast<std::uint64_t>(bm) * T_BK + static_cast<std::uint64_t>(T_BK) * bn;
                static_assert(
                    panelElements <= std::numeric_limits<std::uint32_t>::max(),
                    "GEMM: one staging panel does not fit a 32-bit shared extent.");
                constexpr std::uint64_t required
                    = onAcc::staging::stageStorageBytes<T_Acc, T_STAGES, static_cast<std::uint32_t>(panelElements)>()
                      + onAcc::staging::stageMetadataReserveBytes();
                std::uint64_t const limit
                    = static_cast<std::uint64_t>(queue.getDevice().getDeviceProperties().sharedMemPerBlockBytes);
                if(required > limit)
                    throw std::invalid_argument(
                        std::string{"GEMM: cooperative staging with "} + std::to_string(T_STAGES) + " stages requires "
                        + std::to_string(required) + " bytes of block-shared memory but the " + "device exposes only "
                        + std::to_string(limit) + " bytes; reduce the stage count or use the direct path");
            }
            TileRegions const regions = splitRegions(m, n, bm, bn);
            auto const launch = [&](TileRegion const& region)
            {
                std::uint64_t const rows = region.tileRowEnd - region.tileRowBegin;
                std::uint64_t const cols = region.tileColEnd - region.tileColBegin;
                if(rows == 0u || cols == 0u)
                    return;
                auto const numTiles = static_cast<std::uint32_t>(checkedMul(rows, cols, "region tile count"));
                queue.enqueue(
                    onHost::FrameSpec{Vec{numTiles}, CVec<std::uint32_t, T_Threads>{}, exec},
                    KernelBundle{onAcc::gemm::RegisterBlockedGemmTileKernel<
                        T_Acc,
                        GemmView<T_C>,
                        GemmView<T_A>,
                        GemmView<T_B>,
                        T_Alpha,
                        T_Beta,
                        T_TM,
                        T_TN,
                        T_WR,
                        T_WC,
                        T_BK,
                        T_STAGES>{
                        c,
                        a,
                        b,
                        m,
                        n,
                        k,
                        alpha,
                        beta,
                        static_cast<std::uint32_t>(region.tileRowBegin),
                        static_cast<std::uint32_t>(region.tileColBegin),
                        static_cast<std::uint32_t>(cols),
                        numTiles}});
            };
            launch(regions.full);
            launch(regions.bottom);
            launch(regions.right);
            launch(regions.corner);
        }

        /** Try one compile-time family member and launch it when it matches the runtime plan.
         *
         * This is a named function template rather than a lambda because nvcc's CUDA front end does not expand a
         * function parameter pack that is captured by a lambda and then expanded inside a nested lambda; the pack
         * loses its pack identity and @c std::forward receives the whole captured pack.
         */
        template<
            std::size_t T_Index,
            typename T_Acc,
            typename T_Api,
            typename T_DeviceKind,
            typename T_Exec,
            typename... T_Args>
        bool tryDispatchGemmPlan(GemmPlan const& plan, T_Args&&... args)
        {
            constexpr GemmPlan candidate = gemmCandidateFamily<T_Acc, T_Api, T_DeviceKind, T_Exec>.candidates[T_Index];
            if(plan != candidate)
                return false;
            if constexpr(candidate.policy == GemmStaging::cooperative)
            {
                launchGemmTilesRegisterBlocked<
                    candidate.threadM,
                    candidate.threadN,
                    candidate.workerRows,
                    candidate.workerCols,
                    candidate.kBlock,
                    candidate.stages,
                    candidate.threadCount,
                    T_Acc>(ALPAKA_FORWARD(args)...);
            }
            else
            {
                launchGemmTilesRegisterBlocked<
                    candidate.threadM,
                    candidate.threadN,
                    candidate.workerRows,
                    candidate.workerCols,
                    candidate.kBlock,
                    0u,
                    candidate.threadCount,
                    T_Acc>(ALPAKA_FORWARD(args)...);
            }
            return true;
        }

        /** Visit the candidate family in order and stop at the first matching plan. */
        template<
            typename T_Acc,
            typename T_Api,
            typename T_DeviceKind,
            typename T_Exec,
            std::size_t... T_Index,
            typename... T_Args>
        bool tryDispatchGemmPlans(std::index_sequence<T_Index...>, GemmPlan const& plan, T_Args&&... args)
        {
            return (
                tryDispatchGemmPlan<T_Index, T_Acc, T_Api, T_DeviceKind, T_Exec>(plan, ALPAKA_FORWARD(args)...)
                || ...);
        }

        /** Dispatch the selected plan onto the compiled kernel family.
         *
         * Every family member is instantiated once per accumulator type, target and executor. The runtime plan is
         * compared against each compile-time member and only the matching kernel is launched; the plan fields remain
         * runtime values from the problem's point of view. A plan outside the family throws instead of silently
         * falling back to a different configuration.
         */
        template<typename T_Acc, typename T_Api, typename T_DeviceKind, typename T_Exec, typename... T_Args>
        void dispatchGemmPlan(GemmPlan const& plan, T_Args&&... args)
        {
            bool const dispatched = tryDispatchGemmPlans<T_Acc, T_Api, T_DeviceKind, T_Exec>(
                std::make_index_sequence<gemmCandidateFamily<T_Acc, T_Api, T_DeviceKind, T_Exec>.size>{},
                plan,
                ALPAKA_FORWARD(args)...);
            if(!dispatched)
                throw std::invalid_argument("GEMM: selected plan is not part of the compiled candidate family");
        }

        /** Dispatch the cooperative staging policy over the bounded stage-count family {1, 2, 3}. */
        template<
            std::uint32_t T_TM,
            std::uint32_t T_TN,
            std::uint32_t T_WR,
            std::uint32_t T_WC,
            std::uint32_t T_BK,
            std::uint32_t T_Threads,
            typename T_Acc,
            typename... T_Args>
        void dispatchStagedGemm(GemmStagingConfig const& staging, T_Args&&... args)
        {
            switch(staging.stages)
            {
            case 1u:
                launchGemmTilesRegisterBlocked<T_TM, T_TN, T_WR, T_WC, T_BK, 1u, T_Threads, T_Acc>(
                    ALPAKA_FORWARD(args)...);
                break;
            case 2u:
                launchGemmTilesRegisterBlocked<T_TM, T_TN, T_WR, T_WC, T_BK, 2u, T_Threads, T_Acc>(
                    ALPAKA_FORWARD(args)...);
                break;
            case 3u:
                launchGemmTilesRegisterBlocked<T_TM, T_TN, T_WR, T_WC, T_BK, 3u, T_Threads, T_Acc>(
                    ALPAKA_FORWARD(args)...);
                break;
            default:
                throw std::invalid_argument("GEMM: cooperative staging supports a stage count of 1, 2 or 3");
            }
        }

        template<
            std::uint32_t T_BM,
            std::uint32_t T_BN,
            std::uint32_t T_Threads,
            typename T_Queue,
            typename T_Exec,
            typename T_C,
            typename T_Beta>
        void launchScaleTiles(
            T_Queue const& queue,
            T_Exec const& exec,
            GemmView<T_C> c,
            std::uint64_t m,
            std::uint64_t n,
            T_Beta beta,
            TileConfig const& config)
        {
            TileRegions const regions = splitRegions(m, n, T_BM, T_BN);
            auto const launch = [&](TileRegion const& region)
            {
                std::uint64_t const rows = region.tileRowEnd - region.tileRowBegin;
                std::uint64_t const cols = region.tileColEnd - region.tileColBegin;
                if(rows == 0u || cols == 0u)
                    return;
                auto const numTiles = static_cast<std::uint32_t>(checkedMul(rows, cols, "region tile count"));
                alpaka::unused(config.threads);
                queue.enqueue(
                    onHost::FrameSpec{Vec{numTiles}, CVec<std::uint32_t, T_Threads>{}, exec},
                    KernelBundle{onAcc::gemm::ScaleTileKernel<GemmView<T_C>, T_Beta, T_BM, T_BN>{
                        c,
                        m,
                        n,
                        beta,
                        static_cast<std::uint32_t>(region.tileRowBegin),
                        static_cast<std::uint32_t>(region.tileColBegin),
                        static_cast<std::uint32_t>(cols),
                        numTiles}});
            };
            launch(regions.full);
            launch(regions.bottom);
            launch(regions.right);
            launch(regions.corner);
        }

        template<std::uint32_t T_BM, std::uint32_t T_Threads, typename T_Acc, typename... T_Args>
        void dispatchBk(TileConfig const& config, T_Args&&... args)
        {
            switch(config.bk)
            {
            case 8u:
                launchGemmTiles<T_BM, T_BM, 8u, T_Threads, T_Acc>(ALPAKA_FORWARD(args)...);
                break;
            case 16u:
            default:
                launchGemmTiles<T_BM, T_BM, 16u, T_Threads, T_Acc>(ALPAKA_FORWARD(args)...);
                break;
            }
        }

        template<typename T_Acc, std::uint32_t T_Threads, typename... T_Args>
        void dispatchTile(TileConfig const& config, T_Args&&... args)
        {
            switch(config.tile)
            {
            case 4u:
                dispatchBk<4u, T_Threads, T_Acc>(config, ALPAKA_FORWARD(args)...);
                break;
            case 8u:
                dispatchBk<8u, T_Threads, T_Acc>(config, ALPAKA_FORWARD(args)...);
                break;
            case 32u:
                dispatchBk<32u, T_Threads, T_Acc>(config, ALPAKA_FORWARD(args)...);
                break;
            case 16u:
            default:
                dispatchBk<16u, T_Threads, T_Acc>(config, ALPAKA_FORWARD(args)...);
                break;
            }
        }

        template<std::uint32_t T_Threads, typename... T_Args>
        void dispatchScaleTile(TileConfig const& config, T_Args&&... args)
        {
            switch(config.tile)
            {
            case 4u:
                launchScaleTiles<4u, 4u, T_Threads>(ALPAKA_FORWARD(args)...);
                break;
            case 8u:
                launchScaleTiles<8u, 8u, T_Threads>(ALPAKA_FORWARD(args)...);
                break;
            case 32u:
                launchScaleTiles<32u, 32u, T_Threads>(ALPAKA_FORWARD(args)...);
                break;
            case 16u:
            default:
                launchScaleTiles<16u, 16u, T_Threads>(ALPAKA_FORWARD(args)...);
                break;
            }
        }

        template<
            typename T_Acc,
            typename T_Queue,
            typename T_Exec,
            typename T_C,
            typename T_A,
            typename T_B,
            typename T_Alpha,
            typename T_Beta>
        void executeGemm(
            T_Queue const& queue,
            T_Exec const& exec,
            GemmView<T_C> c,
            GemmView<T_A> a,
            GemmView<T_B> b,
            T_Alpha alpha,
            T_Beta beta,
            GemmNumericMode mode,
            GemmKernel kernel,
            GemmStagingConfig staging,
            std::optional<GemmPlan> expertOverride,
            PlanDiagnostics* diagnosticsOut)
        {
            using Api = std::remove_cvref_t<decltype(queue.getApi())>;
            using DeviceKind = std::remove_cvref_t<decltype(queue.getDeviceKind())>;

            // The asynchronous policy is reserved: route it to the synchronous cooperative pipeline so it is
            // selectable today without changing the schedule, transitions or results.
            if(staging.policy == GemmStaging::asynchronous)
                staging.policy = GemmStaging::cooperative;

            std::uint64_t const m = asU64(c.rows, "M");
            std::uint64_t const n = asU64(c.cols, "N");
            std::uint64_t const k = asU64(a.cols, "K");

            bool const writeOutput = validateProblem(c, a, b, m, n, k, alpha, beta);
            if(m == 0u || n == 0u)
                return;

            TileConfig const config = chooseTileConfig<T_Acc>(m, n, Api{}, DeviceKind{}, exec);
            constexpr std::uint32_t blockThreads = blockThreadCount<T_Acc, Api, DeviceKind, T_Exec>();
            bool const readOperands = m != 0u && n != 0u && k != 0u && alpha != T_Alpha{0};

            if(expertOverride.has_value() && (kernel != GemmKernel::registerBlocked || !readOperands))
                throw std::invalid_argument(
                    "GEMM: an expert plan override requires the register-blocked path with operand reads");

            if(readOperands)
            {
                if(kernel == GemmKernel::registerBlocked)
                {
                    GemmFacts const facts
                        = makeGemmFacts<T_Acc>(Api{}, DeviceKind{}, queue.getDevice().getDeviceProperties());

                    GemmProblem problem;
                    problem.m = m;
                    problem.n = n;
                    problem.k = k;
                    problem.aLayout = a.layout;
                    problem.bLayout = b.layout;
                    problem.cLayout = c.layout;
                    problem.aOp = a.op;
                    problem.bOp = b.op;
                    problem.cOp = c.op;
                    problem.mode = mode;
                    problem.elementBytesA = sizeof(std::remove_cv_t<T_A>);
                    problem.elementBytesB = sizeof(std::remove_cv_t<T_B>);
                    problem.elementBytesC = sizeof(std::remove_cv_t<T_C>);
                    problem.alignmentA = pointerAlignment(a.data);
                    problem.alignmentB = pointerAlignment(b.data);
                    problem.alignmentC = pointerAlignment(c.data);
                    problem.readsOperands = readOperands;

                    // Keep the shipped automatic policy: an unconstrained request uses the compile-time default.
                    GemmPlanRequest request;
                    if(staging.policy != GemmStaging::automatic)
                        request.policy = staging.policy;
                    else if(defaultGemmStaging != GemmStaging::automatic)
                        request.policy = defaultGemmStaging;
                    request.stages = staging.stages;
                    request.expertOverride = expertOverride;

                    PlanSelection const selection
                        = selectPlan(problem, facts, gemmCandidateFamily<T_Acc, Api, DeviceKind, T_Exec>, request);
                    if(diagnosticsOut != nullptr)
                        *diagnosticsOut = selection.diagnostics;
                    dispatchGemmPlan<T_Acc, Api, DeviceKind, T_Exec>(
                        selection.plan,
                        queue,
                        exec,
                        c,
                        a,
                        b,
                        m,
                        n,
                        k,
                        alpha,
                        beta);
                }
                else
                {
                    dispatchTile<T_Acc, blockThreads>(config, queue, exec, c, a, b, m, n, k, alpha, beta, config);
                }
            }
            else if(writeOutput)
            {
                // beta == 1 is a no-op and was excluded by writeOutput.
                dispatchScaleTile<blockThreads>(config, queue, exec, c, m, n, beta, config);
            }
        }
    } // namespace internal
} // namespace alpaka::onHost
