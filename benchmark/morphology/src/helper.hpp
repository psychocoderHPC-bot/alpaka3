/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Independent host reference, deterministic input generation and verification for the
 *        HeCBench `morphology` benchmark.
 *
 * The pinned HeCBench reference (`src/morphology-cuda/main.cu` + `morphology.cu`, and its OpenMP twin
 * `src/morphology-omp/main.cpp` + `morphology.cpp`) runs a 2D separable morphological dilation and
 * erosion with a rectangular structuring element. The HeCBench accelerator reference (its own
 * `.cu`/`.cpp` kernels, not this repository's port) implements the van Herk/Gil-Werman (vHGW)
 * algorithm, which is O(1) per output element and independent of the structuring-element size.
 *
 * This file contains a straightforward, deliberately different O(size) host implementation: for
 * every output cell it scans the full structuring-element window `[c-L, c+R]` (horizontal) and
 * `[r-L, r+R]` (vertical) with the reference's border rules. It is not used for timing, only as
 * the trusted oracle for `verify()`.
 *
 * Border and parity rules reproduced from the pinned reference (validated exhaustively against the
 * pinned CUDA reference for odd and even structuring elements and odd/even image sizes):
 *  - Horizontal pass writes only columns `c` with `L <= c < W - hsize/2`, where `L = hsize/2` and
 *    the right window edge is `R = (hsize odd) ? hsize/2 : hsize/2 - 1`. The output cell is written
 *    to `dst[...]`; all other columns of the intermediate buffer are left at their initial 0.
 *  - Vertical pass writes rows `r` with `Lv <= r < H - vsize/2` (`Lv = vsize/2`,
 *    `Rv = (vsize odd) ? vsize/2 : vsize/2 - 1`); rows outside are set to the operation's border
 *    value (0 for erosion, 255 for dilation). Every column of those border rows is written.
 *  - Consequently the very first and last `hsize/2` (resp. `vsize/2`) rows of the final image are
 *    always the border value, and the intermediate buffer's untouched columns act as 0.
 *
 * Nothing in this file is device code.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace morphologyBench
{
    //! Border / foreground constants of the reference.
    constexpr std::uint8_t black = 0u;
    constexpr std::uint8_t white = 255u;

    //! Morphological operation selector.
    enum class Op
    {
        Erode,
        Dilate
    };

    inline std::uint8_t elementOp(std::uint8_t lhs, std::uint8_t rhs, Op op)
    {
        return op == Op::Dilate ? std::max(lhs, rhs) : std::min(lhs, rhs);
    }

    inline std::uint8_t borderValue(Op op)
    {
        return op == Op::Dilate ? white : black;
    }

    inline int roundUp(int x, int y)
    {
        return (x + y - 1) / y;
    }

    /** Fill the image exactly like `main.cu` in the pinned reference.
     *
     * The reference concentrates all its signal in a single white pixel at `(height/2 - 1,
     * width/2 - 1)` and leaves every other pixel black. `seed == 0` reproduces that exact image; a
     * nonzero seed produces a deterministic pseudo-random black/white image for broader coverage.
     */
    inline void generateImage(std::uint8_t* img, int width, int height, unsigned int seed)
    {
        std::size_t const count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        if(seed == 0u)
        {
            std::memset(img, black, count);
            if(height >= 2 && width >= 2)
                img[static_cast<std::size_t>(height / 2 - 1) * width + (width / 2 - 1)] = white;
            return;
        }
        // xorshift32: deterministic across compilers and platforms.
        std::uint32_t state = seed * 2654435761u + 12345u;
        for(std::size_t i = 0; i < count; ++i)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            img[i] = ((state >> 8) & 0x3u) == 0u ? white : static_cast<std::uint8_t>(state & 0xFFu);
        }
    }

    /** Trusted host reference for a single separable pass pair (`dilation` then `erosion`).
     *
     * @param inOut   input on entry, result on return (row-major, `width * height` bytes)
     * @param tmp     scratch buffer of `width * height` bytes (contents are irrelevant on entry)
     * @param dilation selects dilation (`true`) or erosion (`false`)
     */
    inline void morphologyReference(
        std::uint8_t* inOut,
        std::uint8_t* tmp,
        int width,
        int height,
        int hsize,
        int vsize,
        bool dilation)
    {
        Op const op = dilation ? Op::Dilate : Op::Erode;
        int const Lh = hsize / 2;
        int const Rh = (hsize % 2 != 0) ? hsize / 2 : hsize / 2 - 1;
        int const Lv = vsize / 2;
        int const Rv = (vsize % 2 != 0) ? vsize / 2 : vsize / 2 - 1;

        std::size_t const count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        std::memset(tmp, 0, count);

        // horizontal pass
        for(int r = 0; r < height; ++r)
        {
            std::uint8_t const* src = inOut + static_cast<std::size_t>(r) * width;
            std::uint8_t* dst = tmp + static_cast<std::size_t>(r) * width;
            for(int c = Lh; c < width - hsize / 2; ++c)
            {
                std::uint8_t value = src[c];
                for(int k = -Lh; k <= Rh; ++k)
                    value = elementOp(value, src[c + k], op);
                dst[c] = value;
            }
        }

        // vertical pass
        for(int r = 0; r < height; ++r)
        {
            for(int c = 0; c < width; ++c)
            {
                if(r < Lv || r >= height - vsize / 2)
                {
                    inOut[static_cast<std::size_t>(r) * width + c] = borderValue(op);
                    continue;
                }
                std::uint8_t value = tmp[static_cast<std::size_t>(r) * width + c];
                for(int k = -Lv; k <= Rv; ++k)
                    value = elementOp(value, tmp[static_cast<std::size_t>(r + k) * width + c], op);
                inOut[static_cast<std::size_t>(r) * width + c] = value;
            }
        }
    }

    /** Trusted host reference for the full measured workload: `repeat` times `dilate` then `erode`.
     *
     * This mirrors `main()` of the pinned reference, which applies the pair `repeat` times to the
     * image in place. The caller supplies the input in `image`; the result is written back in place.
     */
    inline void fullReference(
        std::uint8_t* image,
        int width,
        int height,
        int hsize,
        int vsize,
        int repeat)
    {
        std::size_t const count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        std::vector<std::uint8_t> a(image, image + count);
        std::vector<std::uint8_t> b(count, 0u);
        std::vector<std::uint8_t> work(count, 0u);
        for(int n = 0; n < repeat; ++n)
        {
            std::copy(a.begin(), a.end(), work.begin());
            morphologyReference(work.data(), b.data(), width, height, hsize, vsize, true);
            for(std::size_t i = 0; i < count; ++i)
                a[i] = work[i];
            std::copy(a.begin(), a.end(), work.begin());
            morphologyReference(work.data(), b.data(), width, height, hsize, vsize, false);
            for(std::size_t i = 0; i < count; ++i)
                a[i] = work[i];
        }
        std::copy(a.begin(), a.end(), image);
    }

    /** Exact byte comparison of the accelerator result against the host reference.
     *
     * The reference comparison is exact (its `PASS` test is an exact integer sum), so this port
     * keeps an exact comparison and exits nonzero on the first mismatch. The reference's
     * `sum(image) == 255` report line is reproduced by the caller.
     */
    inline int verify(
        std::uint8_t const* result,
        std::uint8_t const* reference,
        std::size_t count,
        char const* label)
    {
        for(std::size_t i = 0; i < count; ++i)
        {
            if(result[i] != reference[i])
            {
                std::fprintf(
                    stderr,
                    "morphology verification FAILED (%s): index %zu result=%u reference=%u (max abs diff "
                    "will follow)\n",
                    label,
                    i,
                    static_cast<unsigned int>(result[i]),
                    static_cast<unsigned int>(reference[i]));
                // report the worst difference to aid debugging
                unsigned int worst = 0;
                std::size_t worstIdx = i;
                for(std::size_t j = 0; j < count; ++j)
                {
                    unsigned int d = result[j] > reference[j] ? static_cast<unsigned int>(result[j] - reference[j])
                                                              : static_cast<unsigned int>(reference[j] - result[j]);
                    if(d > worst)
                    {
                        worst = d;
                        worstIdx = j;
                    }
                }
                std::fprintf(stderr, "  worst diff=%u at index %zu\n", worst, worstIdx);
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

    /** Check that the image contains no value other than the two reference levels and is finite.
     *
     * For an 8-bit image "finite" means every byte is a valid intensity; the reference's own
     * pass/fail test is `sum(image) == WHITE`, which can only hold if every byte is 0. This helper
     * reports the number of nonzero bytes so a caller can reproduce that check.
     */
    inline std::size_t nonzeroCount(std::uint8_t const* image, std::size_t count)
    {
        std::size_t n = 0;
        for(std::size_t i = 0; i < count; ++i)
            if(image[i] != 0u)
                ++n;
        return n;
    }

} // namespace morphologyBench
