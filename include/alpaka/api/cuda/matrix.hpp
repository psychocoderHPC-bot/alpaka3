/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/Api.hpp"
#include "alpaka/core/ApiCudaRt.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/matrix.hpp"
#include "alpaka/onHost/Device.hpp"

#if ALPAKA_LANG_CUDA

/** @file
 *
 * CUDA adapter of the common matrix-accumulate operation.
 *
 * The adapter is a single local collective operation: one `mma.sync.aligned.m16n8k16.row.col` instruction on the
 * warp fragments of @ref alpaka::onAcc::matrix. It contains no tiling loop, no scheduler and no GEMM.
 *
 * Architecture legality is a property of the compiler pass, so it is resolved here, in the backend header:
 *
 * - Device pass of sm_80 or newer: the native overload is defined and selected.
 * - Device pass below sm_80 (Volta sm_70, Turing sm_75): the overload does not exist, so the portable warp fallback
 *   is instantiated instead. Volta and Turing expose different `mma.sync` shapes (`m8n8k4`, `m16n8k8`); this
 *   revision does not implement them and does not pretend they are the Ampere shape.
 * - Host pass: the overload is defined so the call site parses; the inline PTX is compiled only for device passes.
 *
 * The physical register order is the documented portable `m16n8k16` fragment layout of `onAcc/matrix.hpp`. Values are
 * packed into `.f16x2` (or `.bf16x2`) registers in fragment order.
 */

namespace alpaka::onAcc::matrix
{
    namespace internal
    {
        /** One m16n8k16 instruction on the portable warp fragments.
         *
         * Requires that the whole 32-lane warp executes this call together (`.aligned` semantics). The accumulator
         * values are read and written in place.
         */
        template<typename T_In, typename T_Out>
        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr void mmaWarp(
            WarpAccumulator<T_Out>& c,
            WarpOperandA<T_In> const& a,
            WarpOperandB<T_In> const& b)
        {
            static_assert(
                isHalfStorage<T_In> && std::same_as<T_Out, float>,
                "The CUDA m16n8k16 adapter only implements 16-bit inputs with fp32 accumulation.");

            std::uint32_t aReg[4];
            std::uint32_t bReg[2];
            for(std::uint32_t i = 0u; i < 4u; ++i)
                aReg[i] = static_cast<std::uint32_t>(a.value[2u * i].bits())
                          | (static_cast<std::uint32_t>(a.value[2u * i + 1u].bits()) << 16u);
            for(std::uint32_t i = 0u; i < 2u; ++i)
                bReg[i] = static_cast<std::uint32_t>(b.value[2u * i].bits())
                          | (static_cast<std::uint32_t>(b.value[2u * i + 1u].bits()) << 16u);

#    if defined(__CUDA_ARCH__)
            float c0 = c.value[0];
            float c1 = c.value[1];
            float c2 = c.value[2];
            float c3 = c.value[3];
            if constexpr(std::same_as<T_In, Float16>)
            {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                             : "r"(aReg[0]), "r"(aReg[1]), "r"(aReg[2]), "r"(aReg[3]), "r"(bReg[0]), "r"(bReg[1]));
            }
            else
            {
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                             : "r"(aReg[0]), "r"(aReg[1]), "r"(aReg[2]), "r"(aReg[3]), "r"(bReg[0]), "r"(bReg[1]));
            }
            c.value[0] = c0;
            c.value[1] = c1;
            c.value[2] = c2;
            c.value[3] = c3;
#    else
            // Host pass: the kernel body is compiled but never executed on the host. Keep the arguments used so the
            // host compilation is warning-free and the call still type-checks.
            alpaka::unused(aReg, bReg);
#    endif
        }
    } // namespace internal

#    if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
    /** CUDA specialization for the m16n8k16 warp fragments.
     *
     * Constrained to the native type/shape combinations, so an unsupported combination is not instantiated here and
     * resolves to the generic fallback instead of failing to compile.
     */
    template<typename T_Acc, typename T_C, typename T_A, typename T_B>
    requires(
        WarpFragment<T_C> && WarpFragment<T_A> && WarpFragment<T_B>
        && NativeWarpStep<
            typename std::remove_cvref_t<T_A>::value_type,
            typename std::remove_cvref_t<T_C>::value_type,
            M16N8K16>)
    constexpr void alpakaFnDispatch(
        MatrixAccumulate::Spec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>,
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
        alpaka::unused(acc);
        internal::mmaWarp(c, a, b);
    }
#    endif

    /** Runtime CUDA availability: the native operation needs compute capability 8.0 or newer. */
    template<typename T_DeviceKind>
    struct GetAvailability<alpaka::onHost::Device<alpaka::api::Cuda, T_DeviceKind>>
    {
        static Availability get(alpaka::onHost::Device<alpaka::api::Cuda, T_DeviceKind> const& device) noexcept
        {
            int major = 0;
            int minor = 0;
            int const deviceIdx = device.getNativeHandle();
            if(::cudaDeviceGetAttribute(&major, ::cudaDevAttrComputeCapabilityMajor, deviceIdx) != ::cudaSuccess)
                return Availability::unknown;
            if(::cudaDeviceGetAttribute(&minor, ::cudaDevAttrComputeCapabilityMinor, deviceIdx) != ::cudaSuccess)
                return Availability::unknown;
            // m16n8k16 with fp16/bf16 inputs is defined from sm_80 (Ampere) onwards.
            return (major * 10 + minor) >= 80 ? Availability::yes : Availability::no;
        }
    };
} // namespace alpaka::onAcc::matrix

#endif // ALPAKA_LANG_CUDA
