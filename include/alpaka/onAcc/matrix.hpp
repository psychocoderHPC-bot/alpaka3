/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Float16.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/fn.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/warp.hpp"
#include "alpaka/tag.hpp"

#include <array>
#include <bit>
#include <concepts>
#include <cstdint>
#include <type_traits>

/** @file
 *
 * Common matrix-accumulate contract of alpaka3.
 *
 * The primitive computes one small multiply-accumulate step `C += A * B` on fragments that already live in registers.
 * It is not a GEMM, a tiling loop or a scheduler: callers own the loop over K, the fragment packing and the epilogue.
 *
 * Two fragment families share one function symbol, `MatrixAccumulate`:
 *
 * - **Local fragments** (`RegisterTile`, `ColumnFragment`, `RowFragment`) are per-worker register blocks. The
 *   accumulation is local: each worker updates only its own tile, no cross-worker communication and no barrier are
 *   involved. This is the portable register-blocked SIMD/FMA path and the default used by the GEMM kernel.
 * - **Warp fragments** (`WarpAccumulator`, `WarpOperandA`, `WarpOperandB`) follow the documented 32-lane layout of
 *   the `m16n8k16` matrix instruction. They are collective over exactly one 32-lane sub-warp. The fallback
 *   reconstructs the required operands with warp shuffles and accumulates with the same fp32 FMA arithmetic.
 *
 * Participation, completion and numerical mode are part of the contract:
 *
 * - Local fragments: every worker participates with its own fragments, updates only its own accumulator storage and
 *   returns synchronously. There is no cross-worker communication, no barrier and no visibility guarantee beyond the
 *   calling worker. Operands are register values; the only alignment requirement is the natural alignment of the
 *   element type.
 * - Warp fragments: exactly one 32-lane sub-warp participates. The call is synchronous and collective; all 32 lanes
 *   must reach it together. A value of lane `l` is observable only after the call returns. The documented load and
 *   store helpers accept a contiguous row-major 16x16 A, 16x8 B and 16x8 C tile.
 * - Numerical mode: fp16/bf16 inputs convert exactly to fp32 and accumulate in fp32. Strict fp32 and fp64 keep their
 *   precision; no path silently converts them to a 16-bit native instruction.
 *
 * Every specialized backend implementation is registered through the alpaka function-symbol mechanism in a backend
 * header. Common code never names a backend.
 */

namespace alpaka::onAcc::matrix
{
    /** Compile-time answer of a capability query.
     *
     * `unknown` means the information is not available at this point (for example on the host pass of a CUDA
     * compilation, where the target architecture is not known). It never means "supported".
     */
    enum class Availability : std::uint8_t
    {
        no = 0,
        unknown = 1,
        yes = 2
    };

    constexpr char const* toString(Availability const value) noexcept
    {
        switch(value)
        {
        case Availability::no:
            return "no";
        case Availability::unknown:
            return "unknown";
        case Availability::yes:
            return "yes";
        }
        return "unknown";
    }

    /** Instruction shape of one matrix-accumulate step.
     *
     * @tparam T_M rows of one operand/accumulator block
     * @tparam T_N columns of one operand/accumulator block
     * @tparam T_K contraction depth of one step
     */
    template<std::uint32_t T_M, std::uint32_t T_N, std::uint32_t T_K>
    struct Quantum
    {
        static constexpr std::uint32_t m = T_M;
        static constexpr std::uint32_t n = T_N;
        static constexpr std::uint32_t k = T_K;
    };

    /** Shape of the only native matrix instruction this revision implements. */
    using M16N8K16 = Quantum<16u, 8u, 16u>;

    // ---------------------------------------------------------------------------------------------------------------
    // Local, per-worker fragments
    // ---------------------------------------------------------------------------------------------------------------

    /** One column of operand A owned by one worker: T_M values for a fixed K coordinate. */
    template<typename T, std::uint32_t T_M>
    struct ColumnFragment
    {
        using value_type = T;
        static constexpr std::uint32_t rows = T_M;

        std::array<T, T_M> value{};
    };

    /** One row of operand B owned by one worker: T_N values for a fixed K coordinate. */
    template<typename T, std::uint32_t T_N>
    struct RowFragment
    {
        using value_type = T;
        static constexpr std::uint32_t columns = T_N;

        std::array<T, T_N> value{};
    };

    /** Row-major per-worker accumulator tile. */
    template<typename T, std::uint32_t T_M, std::uint32_t T_N>
    struct RegisterTile
    {
        using value_type = T;
        static constexpr std::uint32_t rows = T_M;
        static constexpr std::uint32_t columns = T_N;

        std::array<T, T_M * T_N> value{};

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr T& at(std::uint32_t m, std::uint32_t n)
        {
            ALPAKA_ASSERT_ACC(m < T_M && n < T_N);
            return value[static_cast<std::size_t>(m) * T_N + n];
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr T at(std::uint32_t m, std::uint32_t n) const
        {
            ALPAKA_ASSERT_ACC(m < T_M && n < T_N);
            return value[static_cast<std::size_t>(m) * T_N + n];
        }
    };

    namespace internal
    {
        template<typename T>
        constexpr bool isHalfStorage = std::same_as<T, Float16> || std::same_as<T, BFloat16>;

        /** Value conversion used by every portable implementation.
         *
         * 16-bit storage values are converted exactly to fp32 first; no other rounding is introduced.
         */
        template<typename T_Out, typename T_In>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr T_Out convert(T_In const value)
        {
            if constexpr(isHalfStorage<T_In>)
                return static_cast<T_Out>(value.toFloat());
            else
                return static_cast<T_Out>(value);
        }

        /** Largest power-of-two SIMD width not larger than the given count, capped at eight. */
        consteval std::uint32_t packWidth(std::uint32_t const count)
        {
            std::uint32_t width = 1u;
            while(width * 2u <= count && width < 8u)
                width *= 2u;
            return width;
        }
    } // namespace internal

    // ---------------------------------------------------------------------------------------------------------------
    // Warp fragment layout of m16n8k16
    // ---------------------------------------------------------------------------------------------------------------

    /** Per-lane register block of a warp fragment.
     *
     * The element order is the documented portable layout of @ref M16N8K16, not a backend-private register order.
     */
    template<typename T, std::uint32_t T_Count>
    struct WarpRegisters
    {
        using value_type = T;
        static constexpr std::uint32_t count = T_Count;
        static constexpr bool matrixWarpFragment = true;

        std::array<T, T_Count> value{};
    };

    /** Operand A of one m16n8k16 step: eight values per lane. */
    template<typename T>
    using WarpOperandA = WarpRegisters<T, 8u>;

    /** Operand B of one m16n8k16 step: four values per lane. */
    template<typename T>
    using WarpOperandB = WarpRegisters<T, 4u>;

    /** Accumulator of one m16n8k16 step: four fp32 values per lane. */
    template<typename T>
    using WarpAccumulator = WarpRegisters<T, 4u>;

    namespace internal
    {
        /** Logical (row, k) of lane-local A register `index`. */
        constexpr std::pair<std::uint32_t, std::uint32_t> warpACoordinate(
            std::uint32_t const lane,
            std::uint32_t const index)
        {
            std::uint32_t const row = lane / 4u + ((index / 2u) % 2u != 0u ? 8u : 0u);
            std::uint32_t const k = (lane % 4u) * 2u + (index >= 4u ? 8u : 0u) + index % 2u;
            return {row, k};
        }

        /** Logical (k, column) of lane-local B register `index`. */
        constexpr std::pair<std::uint32_t, std::uint32_t> warpBCoordinate(
            std::uint32_t const lane,
            std::uint32_t const index)
        {
            std::uint32_t const k = (lane % 4u) * 2u + (index >= 2u ? 8u : 0u) + index % 2u;
            std::uint32_t const column = lane / 4u;
            return {k, column};
        }

        /** Logical (row, column) of lane-local C register `index`. */
        constexpr std::pair<std::uint32_t, std::uint32_t> warpCCoordinate(
            std::uint32_t const lane,
            std::uint32_t const index)
        {
            std::uint32_t const row = lane / 4u + (index >= 2u ? 8u : 0u);
            std::uint32_t const column = (lane % 4u) * 2u + index % 2u;
            return {row, column};
        }

        /** Fill the lane-local A registers from a logical row-major 16x16 tile. */
        template<typename T>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void loadWarpA(
            WarpOperandA<T>& fragment,
            T const* tile,
            std::uint32_t const lane)
        {
            for(std::uint32_t i = 0u; i < 8u; ++i)
            {
                auto const [row, k] = warpACoordinate(lane, i);
                fragment.value[i] = tile[static_cast<std::size_t>(row) * 16u + k];
            }
        }

        /** Fill the lane-local B registers from a logical row-major 16x8 tile. */
        template<typename T>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void loadWarpB(
            WarpOperandB<T>& fragment,
            T const* tile,
            std::uint32_t const lane)
        {
            for(std::uint32_t i = 0u; i < 4u; ++i)
            {
                auto const [k, column] = warpBCoordinate(lane, i);
                fragment.value[i] = tile[static_cast<std::size_t>(k) * 8u + column];
            }
        }

        /** Store the lane-local accumulator registers into a logical row-major 16x8 tile. */
        template<typename T>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void storeWarpC(
            T* tile,
            WarpAccumulator<T> const& fragment,
            std::uint32_t const lane)
        {
            for(std::uint32_t i = 0u; i < 4u; ++i)
            {
                auto const [row, column] = warpCCoordinate(lane, i);
                tile[static_cast<std::size_t>(row) * 8u + column] = fragment.value[i];
            }
        }
    } // namespace internal

    /** Fill the lane-local operand A registers from a logical row-major 16x16 tile.
     *
     * This is the documented portable load path of the warp contract. The caller supplies the lane index of the
     * calling worker inside the 32-lane group.
     */
    template<typename T>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void loadWarpOperandA(
        WarpOperandA<T>& fragment,
        T const* tile,
        std::uint32_t const lane)
    {
        internal::loadWarpA(fragment, tile, lane);
    }

    /** Fill the lane-local operand B registers from a logical row-major 16x8 tile. */
    template<typename T>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void loadWarpOperandB(
        WarpOperandB<T>& fragment,
        T const* tile,
        std::uint32_t const lane)
    {
        internal::loadWarpB(fragment, tile, lane);
    }

    /** Store the lane-local accumulator registers into a logical row-major 16x8 tile.
     *
     * This is the documented conversion from the opaque native accumulator order to a portable epilogue layout.
     */
    template<typename T>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void storeWarpAccumulator(
        T* tile,
        WarpAccumulator<T> const& fragment,
        std::uint32_t const lane)
    {
        internal::storeWarpC(tile, fragment, lane);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Legality
    // ---------------------------------------------------------------------------------------------------------------

    /** Accumulation types the portable contract accepts.
     *
     * 16-bit inputs require an fp32 accumulator. Strict fp32 and fp64 never change precision: the accumulator type
     * must equal the operand type.
     */
    template<typename T_In, typename T_Acc>
    concept PortableTypes = (std::same_as<T_In, float> && std::same_as<T_Acc, float>)
                            || (std::same_as<T_In, double> && std::same_as<T_Acc, double>)
                            || (internal::isHalfStorage<T_In> && std::same_as<T_Acc, float>);

    /** Per-worker fragments of the local contract. */
    template<typename T>
    concept LocalTileFragment = requires { typename std::remove_cvref_t<T>::value_type; }
                                && !requires { typename std::remove_cvref_t<T>::matrixWarpFragment; } && requires {
                                       std::remove_cvref_t<T>::rows;
                                       std::remove_cvref_t<T>::columns;
                                   };

    template<typename T>
    concept LocalColumnFragment = requires { typename std::remove_cvref_t<T>::value_type; } && !requires {
        typename std::remove_cvref_t<T>::matrixWarpFragment;
    } && requires { std::remove_cvref_t<T>::rows; };

    template<typename T>
    concept LocalRowFragment = requires { typename std::remove_cvref_t<T>::value_type; } && !requires {
        typename std::remove_cvref_t<T>::matrixWarpFragment;
    } && requires { std::remove_cvref_t<T>::columns; };

    template<typename T>
    concept WarpFragment = requires { std::remove_cvref_t<T>::matrixWarpFragment; };

    /** Portable legality of one warp step. The fallback implements every combination that satisfies this concept. */
    template<typename T_In, typename T_Acc, typename T_Quantum>
    concept PortableWarpStep = std::same_as<T_Quantum, M16N8K16> && PortableTypes<T_In, T_Acc>;

    /** Combinations with a native `mma.sync.m16n8k16` implementation.
     *
     * Native code only exists for 16-bit inputs with fp32 accumulation. Everything else is portable-only and is
     * excluded from native template instantiation by this concept.
     */
    template<typename T_In, typename T_Acc, typename T_Quantum>
    concept NativeWarpStep
        = std::same_as<T_Quantum, M16N8K16> && internal::isHalfStorage<T_In> && std::same_as<T_Acc, float>;

    /** Compile-time native-support descriptor.
     *
     * This answers whether a native specialization is legal for the given type, shape, API and device kind. It does
     * not replace a runtime device query: the target architecture of a CUDA compilation is not part of the C++ type
     * system, so the host pass of a CUDA build answers `unknown`.
     *
     * @return `no` for anything without a native implementation, `yes` only for a provably legal compiled target,
     *         and `unknown` when the compiled target is not known at this point.
     */
    template<typename T_In, typename T_Acc, typename T_Quantum, typename T_Api, typename T_DeviceKind>
    consteval Availability matrixAccumulateSupported()
    {
        if constexpr(!NativeWarpStep<T_In, T_Acc, T_Quantum>)
            return Availability::no;
        else if constexpr(
            std::same_as<T_Api, alpaka::api::Cuda> && std::same_as<T_DeviceKind, alpaka::deviceKind::NvidiaGpu>)
        {
            // The architecture is a property of the current compiler pass, not of the requested API/device kind.
            if constexpr(ALPAKA_ARCH_PTX == ALPAKA_VERSION_NUMBER_NOT_AVAILABLE)
                return Availability::unknown;
            else if constexpr(ALPAKA_ARCH_PTX >= ALPAKA_VERSION_NUMBER(8, 0, 0))
                return Availability::yes;
            else
                return Availability::no;
        }
        else
            return Availability::no;
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Runtime availability
    // ---------------------------------------------------------------------------------------------------------------

    /** Runtime device query extension point.
     *
     * The primary template reports `no`, which is correct for every API/device kind without a native adapter. A
     * backend header specializes it. A query that cannot be answered must return `unknown`, never `yes`.
     */
    template<typename T_Device>
    struct GetAvailability
    {
        static Availability get(T_Device const&) noexcept
        {
            return Availability::no;
        }
    };

    /** Whether the native matrix operation is available on the given device.
     *
     * @param device an alpaka onHost device
     * @return `yes`, `no`, or `unknown` when the device cannot be queried
     */
    template<typename T_Device>
    Availability available(T_Device const& device) noexcept
    {
        return GetAvailability<std::remove_cvref_t<T_Device>>::get(device);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Function symbol
    // ---------------------------------------------------------------------------------------------------------------

    /** Matrix accumulate function symbol.
     *
     * Call as `MatrixAccumulate::call(acc, accumulator, operandA, operandB)`. The first argument is the accelerator,
     * because the operation runs inside a kernel. The overload is selected from the accelerator's API and device
     * kind, then constrained by fragment kind, element type and shape. When no native overload is viable the generic
     * alpaka fallback is used, so an unsupported native combination compiles and selects the fallback.
     */
    ALPAKA_FN_SYMBOL(MatrixAccumulate, alpaka::fn::Fallback::toAlpaka, alpaka::fn::Registration::alwaysTrue);

    namespace internal
    {
        /** Portable per-worker register-blocked outer product, the default of the local contract. */
        template<typename T_Acc, std::uint32_t T_M, std::uint32_t T_N, typename T_In>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void localFma(
            RegisterTile<T_Acc, T_M, T_N>& c,
            ColumnFragment<T_In, T_M> const& a,
            RowFragment<T_In, T_N> const& b)
        {
            constexpr std::uint32_t width = packWidth(T_N);
            for(std::uint32_t m = 0u; m < T_M; ++m)
            {
                T_Acc const av = convert<T_Acc>(a.value[m]);
                std::uint32_t n = 0u;
                for(; n + width <= T_N; n += width)
                {
                    Simd<T_Acc, width> bv{};
                    for(std::uint32_t lane = 0u; lane < width; ++lane)
                        bv[lane] = convert<T_Acc>(b.value[n + lane]);
                    for(std::uint32_t lane = 0u; lane < width; ++lane)
                        c.value[static_cast<std::size_t>(m) * T_N + n + lane] += av * bv[lane];
                }
                for(; n < T_N; ++n)
                    c.value[static_cast<std::size_t>(m) * T_N + n] += av * convert<T_Acc>(b.value[n]);
            }
        }

        /** Fetch one operand scalar from a distributed warp fragment by shuffling it from its owner lane. */
        template<typename T_Out, typename T_In, typename T_Acc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr T_Out shflOperand(
            T_Acc const& acc,
            T_In const value,
            std::uint32_t const source)
        {
            if constexpr(isHalfStorage<T_In>)
            {
                std::uint32_t bits = static_cast<std::uint32_t>(value.bits());
                bits = warp::shfl(acc, bits, source, 32u);
                return convert<T_Out>(T_In::fromBits(static_cast<std::uint16_t>(bits)));
            }
            else
                return convert<T_Out>(warp::shfl(acc, value, source, 32u));
        }

        /** Portable warp fallback: the same fp32 FMA sum, with operands reconstructed by warp shuffles. */
        template<typename T_Out, typename T_In, typename T_Acc>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void warpFma(
            T_Acc const& acc,
            WarpAccumulator<T_Out>& c,
            WarpOperandA<T_In> const& a,
            WarpOperandB<T_In> const& b)
        {
            auto const lane = warp::getLaneIdx(acc) % 32u;

            auto const rowLow = lane / 4u;
            auto const rowHigh = rowLow + 8u;
            auto const columnLow = (lane % 4u) * 2u;
            auto const columnHigh = columnLow + 1u;

            T_Out acc0{};
            T_Out acc1{};
            T_Out acc2{};
            T_Out acc3{};
            for(std::uint32_t k = 0u; k < 16u; ++k)
            {
                std::uint32_t const kHalf = k < 8u ? 0u : 1u;
                std::uint32_t const kPair = (k % 8u) / 2u;
                std::uint32_t const kOdd = k % 2u;
                std::uint32_t const aIndex = kHalf * 4u + kOdd;
                std::uint32_t const bIndex = kHalf * 2u + kOdd;

                T_Out const aLow
                    = shflOperand<T_Out>(acc, a.value[aIndex + (rowLow >= 8u ? 2u : 0u)], 4u * (rowLow % 8u) + kPair);
                T_Out const aHigh = shflOperand<T_Out>(
                    acc,
                    a.value[aIndex + (rowHigh >= 8u ? 2u : 0u)],
                    4u * (rowHigh % 8u) + kPair);
                T_Out const bLow = shflOperand<T_Out>(acc, b.value[bIndex], 4u * columnLow + kPair);
                T_Out const bHigh = shflOperand<T_Out>(acc, b.value[bIndex], 4u * columnHigh + kPair);

                acc0 += aLow * bLow;
                acc1 += aLow * bHigh;
                acc2 += aHigh * bLow;
                acc3 += aHigh * bHigh;
            }
            c.value[0] += acc0;
            c.value[1] += acc1;
            c.value[2] += acc2;
            c.value[3] += acc3;
        }
    } // namespace internal

    // ---------------------------------------------------------------------------------------------------------------
    // Generic fallback registrations
    // ---------------------------------------------------------------------------------------------------------------

    /** Generic local fallback for every device kind. */
    template<alpaka::concepts::DeviceKind T_DeviceKind, typename T_Acc, typename T_C, typename T_A, typename T_B>
    requires(
        LocalTileFragment<T_C> && LocalColumnFragment<T_A> && LocalRowFragment<T_B> && !LocalColumnFragment<T_B>
        && !LocalRowFragment<T_A>
        && PortableTypes<typename std::remove_cvref_t<T_A>::value_type, typename std::remove_cvref_t<T_C>::value_type>)
    constexpr void alpakaFnDispatch(
        MatrixAccumulate::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>,
        T_Acc const&,
        T_C& c,
        T_A const& a,
        T_B const& b)
    {
        static_assert(
            std::remove_cvref_t<T_A>::rows == std::remove_cvref_t<T_C>::rows
                && std::remove_cvref_t<T_B>::columns == std::remove_cvref_t<T_C>::columns,
            "matrixAccumulate: operand fragments do not match the accumulator tile.");
        internal::localFma(c, a, b);
    }

    /** Generic warp fallback. It requires a hardware warp of at least 32 lanes; a host warp has one lane. */
    template<alpaka::concepts::DeviceKind T_DeviceKind, typename T_Acc, typename T_C, typename T_A, typename T_B>
    requires(
        WarpFragment<T_C> && WarpFragment<T_A> && WarpFragment<T_B>
        && PortableTypes<typename std::remove_cvref_t<T_A>::value_type, typename std::remove_cvref_t<T_C>::value_type>)
    constexpr void alpakaFnDispatch(
        MatrixAccumulate::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>,
        T_Acc const& acc,
        T_C& c,
        T_A const& a,
        T_B const& b)
    {
        static_assert(
            std::remove_cvref_t<T_C>::count == 4u && std::remove_cvref_t<T_A>::count == 8u
                && std::remove_cvref_t<T_B>::count == 4u,
            "matrixAccumulate: warp fragment register counts do not match m16n8k16.");
        static_assert(
            T_Acc::getWarpSize() >= 32u,
            "matrixAccumulate: the warp contract needs a hardware warp of at least 32 lanes.");
        internal::warpFma(acc, c, a, b);
    }
} // namespace alpaka::onAcc::matrix
