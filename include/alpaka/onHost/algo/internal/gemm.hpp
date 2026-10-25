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
#include "alpaka/onHost/trait.hpp"
#include "alpaka/trait.hpp"

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

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
    /** Storage layout of a GEMM view. */
    enum class GemmLayout : std::uint8_t
    {
        rowMajor,
        colMajor
    };

    /** Whether a view is used as stored or transposed. */
    enum class GemmOp : std::uint8_t
    {
        none,
        transpose
    };

    /** Numerical mode of the GEMM operation.
     *
     * fp16ToFp32 requires @ref alpaka::Float16 storage for A and B and FP32 storage for C. bf16ToFp32 is the same
     * with @ref alpaka::BFloat16. In both modes the stored inputs are converted exactly to FP32 inside the common
     * kernel, the products and the accumulation are FP32, and the result is stored as FP32. The mode and the view
     * value types must agree: a mismatch, including passing already-converted FP32 operands with a mixed mode tag,
     * is rejected before launch. The two 16-bit storage types are never interchangeable.
     */
    enum class GemmNumericMode : std::uint8_t
    {
        fp32,
        fp64,
        fp16ToFp32,
        bf16ToFp32
    };

    /** Kernel implementation selected for the multiply-accumulate path.
     *
     * registerBlocked is the M3 default: one register-blocked outer-product per worker. simple keeps the M2 direct
     * path without operand reuse selectable for comparison and regression testing. Both paths share the same tile
     * regions, boundaries and numerical contract.
     */
    enum class GemmKernel : std::uint8_t
    {
        registerBlocked,
        simple
    };

    /** Non-owning GEMM matrix view.
     *
     * rows and cols are the *logical* extents of the view before applying op. The stored matrix therefore has
     * storedRows() x storedCols() elements and is addressed as
     *
     *     layout == rowMajor ? row * ld + col : col * ld + row
     *
     * A smaller logical extent with a larger ld, or a shifted data pointer, expresses a padded leading dimension and
     * a subview.
     */
    template<typename T>
    struct GemmView
    {
        using value_type = T;

        T* data = nullptr;
        std::size_t rows = 0;
        std::size_t cols = 0;
        std::size_t ld = 0;
        GemmLayout layout = GemmLayout::rowMajor;
        GemmOp op = GemmOp::none;

        [[nodiscard]] constexpr std::size_t storedRows() const noexcept
        {
            return op == GemmOp::none ? rows : cols;
        }

        [[nodiscard]] constexpr std::size_t storedCols() const noexcept
        {
            return op == GemmOp::none ? cols : rows;
        }

        [[nodiscard]] constexpr std::size_t storedElements() const noexcept
        {
            return storedRows() * storedCols();
        }

        [[nodiscard]] constexpr auto& stored(std::size_t row, std::size_t col) const
        {
            ALPAKA_ASSERT(data != nullptr);
            std::size_t const offset = layout == GemmLayout::rowMajor ? row * ld + col : col * ld + row;
            return data[offset];
        }

        /** Element (i, k) of the logical operand. */
        [[nodiscard]] ALPAKA_FN_INLINE constexpr auto& at(std::size_t i, std::size_t k) const
        {
            return op == GemmOp::none ? stored(i, k) : stored(k, i);
        }
    };

    template<typename T>
    GemmView(T*, std::size_t, std::size_t, std::size_t, GemmLayout, GemmOp) -> GemmView<T>;

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
            GemmKernel kernel)
        {
            using Api = std::remove_cvref_t<decltype(queue.getApi())>;
            using DeviceKind = std::remove_cvref_t<decltype(queue.getDeviceKind())>;

            std::uint64_t const m = asU64(c.rows, "M");
            std::uint64_t const n = asU64(c.cols, "N");
            std::uint64_t const k = asU64(a.cols, "K");

            bool const writeOutput = validateProblem(c, a, b, m, n, k, alpha, beta);
            if(m == 0u || n == 0u)
                return;

            TileConfig const config = chooseTileConfig<T_Acc>(m, n, Api{}, DeviceKind{}, exec);
            constexpr std::uint32_t blockThreads = blockThreadCount<T_Acc, Api, DeviceKind, T_Exec>();
            bool const readOperands = m != 0u && n != 0u && k != 0u && alpha != T_Alpha{0};

            if(readOperands)
            {
                if(kernel == GemmKernel::registerBlocked)
                {
                    constexpr RegisterTileConfig rb = registerTileConfig<T_Acc, Api, DeviceKind, T_Exec>();
                    launchGemmTilesRegisterBlocked<rb.tm, rb.tn, rb.wr, rb.wc, rb.bk, blockThreads, T_Acc>(
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
