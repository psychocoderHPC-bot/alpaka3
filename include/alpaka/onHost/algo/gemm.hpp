/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/onHost/algo/internal/gemm.hpp"

namespace alpaka::onHost
{
    /** Enqueue C := alpha * op(A) * op(B) + beta * C on the given queue.
     *
     * The operation follows the queue's asynchronous semantics: the call only enqueues work and returns. The caller
     * keeps the view storage alive until the queue has completed the operation.
     *
     * @tparam T_Executor executor type, constrained by @ref alpaka::concepts::Executor
     * @tparam T_Queue queue type
     * @param queue queue the work is enqueued on
     * @param exec executor used for the launch, @ref alpaka::exec::anyExecutor selects the default executor of the
     *        queue's device
     * @param c output view, logically M x N, op must be none
     * @param a operand A, logically M x K
     * @param b operand B, logically K x N
     * @param alpha scalar multiplier of op(A) * op(B)
     * @param beta scalar multiplier of the incoming C
     * @param mode numerical mode, defaults to strict FP32
     * @param kernel multiply-accumulate implementation, defaults to the M3 register-blocked outer-product path
     * @param staging operand supply policy of the register-blocked path, defaults to the compile-time configuration
     *        value (the direct global-load path). The cooperative policy stages each K block in block-shared memory
     *        and rejects a request whose bounded scratch exceeds the device's shared-memory limit.
     *
     * @throws std::invalid_argument for invalid dimensions, layouts, modes, overlapping storage or overflowing size
     *         and launch arithmetic
     *
     * @{
     */
    template<
        alpaka::concepts::Executor T_Executor,
        typename T_Queue,
        typename T_C,
        typename T_A,
        typename T_B,
        typename T_Alpha,
        typename T_Beta>
    inline void gemm(
        T_Queue const& queue,
        T_Executor const exec,
        GemmView<T_C> c,
        GemmView<T_A> a,
        GemmView<T_B> b,
        T_Alpha alpha,
        T_Beta beta,
        GemmNumericMode mode = GemmNumericMode::fp32,
        GemmKernel kernel = GemmKernel::registerBlocked,
        GemmStagingConfig staging = {})
    {
        static_assert(std::is_arithmetic_v<std::remove_cv_t<T_Alpha>>, "alpha must be an arithmetic scalar");
        static_assert(std::is_arithmetic_v<std::remove_cv_t<T_Beta>>, "beta must be an arithmetic scalar");

        if constexpr(exec == alpaka::exec::anyExecutor)
        {
            gemm(queue, defaultExecutor(queue.getDevice()), c, a, b, alpha, beta, mode, kernel, staging);
        }
        else
        {
            // The accumulator type is also the type the kernel uses to read alpha, beta and the old C value, so a
            // mismatch here would silently change the arithmetic rather than fail. Check the complete mode signature
            // before anything is enqueued and name the required storage in the error.
            auto const execute = [&]<typename T_Acc, typename T_AStorage, typename T_BStorage>(char const* requirement)
            {
                using CType = std::remove_cv_t<T_C>;
                using AType = std::remove_cv_t<T_A>;
                using BType = std::remove_cv_t<T_B>;
                if constexpr(
                    std::same_as<CType, T_Acc> && std::same_as<AType, T_AStorage> && std::same_as<BType, T_BStorage>)
                {
                    internal::executeGemm<T_Acc>(queue, exec, c, a, b, alpha, beta, kernel, staging);
                }
                else
                {
                    throw std::invalid_argument(
                        std::string{"GEMM: numerical mode does not match the view value types; "} + requirement);
                }
            };

            switch(mode)
            {
            case GemmNumericMode::fp32:
                execute.template operator()<float, float, float>("fp32 requires float storage for C, A and B");
                break;
            case GemmNumericMode::fp64:
                execute.template operator()<double, double, double>("fp64 requires double storage for C, A and B");
                break;
            case GemmNumericMode::fp16ToFp32:
                execute.template operator()<float, Float16, Float16>(
                    "fp16ToFp32 requires float C and Float16 A/B storage");
                break;
            case GemmNumericMode::bf16ToFp32:
                execute.template operator()<float, BFloat16, BFloat16>(
                    "bf16ToFp32 requires float C and BFloat16 A/B storage");
                break;
            }
        }
    }

    /** Enqueue GEMM with the device's default executor. */
    template<typename T_Queue, typename T_C, typename T_A, typename T_B, typename T_Alpha, typename T_Beta>
    inline void gemm(
        T_Queue const& queue,
        GemmView<T_C> c,
        GemmView<T_A> a,
        GemmView<T_B> b,
        T_Alpha alpha,
        T_Beta beta,
        GemmNumericMode mode = GemmNumericMode::fp32,
        GemmKernel kernel = GemmKernel::registerBlocked,
        GemmStagingConfig staging = {})
    {
        gemm(queue, defaultExecutor(queue.getDevice()), c, a, b, alpha, beta, mode, kernel, staging);
    }

    /** @} */
} // namespace alpaka::onHost
