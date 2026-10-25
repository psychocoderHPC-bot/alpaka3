/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

/** @file
 *
 * Value types shared by the GEMM public entry point and the M5 planner.
 *
 * The definitions live in their own header so that the planner does not have to include the host launch machinery and
 * the launch machinery does not have to include the planner first. Behaviour and validation live in
 * `internal/gemm.hpp`; this file only declares the descriptors.
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

    /** Operand supply policy of the register-blocked GEMM path.
     *
     * direct keeps the M3 path that reads A and B from global memory in the K loop. cooperative stages one A/B
     * panel per K block in block-shared memory and reads the register-blocked outer product from there. automatic
     * resolves to the compile-time configuration value @ref internal::defaultGemmStaging, so the shipped default is
     * the direct path and no measurement is required to select it.
     */
    enum class GemmStaging : std::uint8_t
    {
        automatic,
        direct,
        cooperative
    };

    /** Staging selection and depth for the register-blocked GEMM path.
     *
     * stages is the synchronous ring depth of the cooperative policy and must be in {1, 2, 3}. It is ignored by the
     * direct policy.
     */
    struct GemmStagingConfig
    {
        GemmStaging policy = GemmStaging::automatic;
        std::uint32_t stages = 1;
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

} // namespace alpaka::onHost
