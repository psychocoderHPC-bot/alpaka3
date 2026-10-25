/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * Standalone numpy cross-check exporter for reference_gemm.hpp.
 *
 * The reference must not certify itself. This tool exports its results for a
 * fixed, reproducible set of small shapes to CSV; ``verify_reference.py``
 * regenerates the same operands with numpy and compares. Neither the exporter
 * nor the checker links against the GEMM implementation.
 *
 * Build and run:
 *
 *   g++ -std=c++20 -O2 -I validation/numerics \
 *       validation/numerics/verify_reference.cpp -o /tmp/verify_reference
 *   /tmp/verify_reference > /tmp/gemm-reference.csv
 *   python3 validation/numerics/verify_reference.py /tmp/gemm-reference.csv
 *
 * Operand generation is a documented 64-bit LCG so both sides produce the
 * same float32 values:
 *
 *   state = 0x0123456789abcdef + 0x9e3779b97f4a7c15 * case_index
 *   next:  state = state * 6364136223846793005 + 1442695040888963407 (mod 2^64)
 *   value: ((state >> 40) / 2^24) * 2 - 1, rounded to float32
 *
 * Storage of A, then B, then C is filled linearly in memory order, including
 * leading-dimension padding. The exporter prints a double-accumulating
 * reference result as a hexadecimal double, which round-trips exactly.
 */

#include "reference_gemm.hpp"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace gemm = alpaka::validation::gemm;

namespace
{
    std::uint64_t nextRandom(std::uint64_t& state)
    {
        state = state * 6'364'136'223'846'793'005ULL + 1'442'695'040'888'963'407ULL;
        return state;
    }

    float nextFloat(std::uint64_t& state)
    {
        auto const value = static_cast<double>(nextRandom(state) >> 40) / 16777216.0;
        return static_cast<float>(value * 2.0 - 1.0);
    }

    struct Case
    {
        char const* id;
        std::size_t m;
        std::size_t n;
        std::size_t k;
        gemm::Layout layoutA;
        gemm::Layout layoutB;
        gemm::Layout layoutC;
        gemm::Op opA;
        gemm::Op opB;
        std::size_t padA;
        std::size_t padB;
        std::size_t padC;
        float alpha;
        float beta;
    };

    template<typename T>
    struct Buffer
    {
        std::size_t rows;
        std::size_t cols;
        std::size_t ld;
        gemm::Layout layout;
        gemm::Op op;
        std::vector<T> storage;

        Buffer(std::size_t rowsIn, std::size_t colsIn, gemm::Layout layoutIn, gemm::Op opIn, std::size_t pad)
            : rows(rowsIn)
            , cols(colsIn)
            , ld((layoutIn == gemm::Layout::rowMajor ? colsIn : rowsIn) + pad)
            , layout(layoutIn)
            , op(opIn)
        {
            auto const extent = layout == gemm::Layout::rowMajor ? rows : cols;
            storage.resize(ld * extent);
        }

        [[nodiscard]] gemm::MatrixView<T> view()
        {
            return {storage.data(), rows, cols, ld, layout, op};
        }

        [[nodiscard]] gemm::MatrixView<T const> view() const
        {
            return {storage.data(), rows, cols, ld, layout, op};
        }
    };

    template<typename T>
    void fill(std::vector<T>& storage, std::uint64_t& state)
    {
        for(auto& value : storage)
            value = static_cast<T>(nextFloat(state));
    }

    char const* layoutName(gemm::Layout layout)
    {
        return layout == gemm::Layout::rowMajor ? "row" : "col";
    }

    char const* opName(gemm::Op op)
    {
        return op == gemm::Op::none ? "N" : "T";
    }
} // namespace

int main()
{
    Case const cases[] = {
        {"tiny-1x1x1",
         1,
         1,
         1,
         gemm::Layout::rowMajor,
         gemm::Layout::rowMajor,
         gemm::Layout::rowMajor,
         gemm::Op::none,
         gemm::Op::none,
         0,
         0,
         0,
         1.0f,
         0.0f},
        {"square-8",
         8,
         8,
         8,
         gemm::Layout::rowMajor,
         gemm::Layout::rowMajor,
         gemm::Layout::rowMajor,
         gemm::Op::none,
         gemm::Op::none,
         2,
         3,
         1,
         1.0f,
         0.0f},
        {"rect-16x32x7-NT",
         16,
         32,
         7,
         gemm::Layout::rowMajor,
         gemm::Layout::colMajor,
         gemm::Layout::rowMajor,
         gemm::Op::none,
         gemm::Op::transpose,
         1,
         2,
         0,
         1.0f,
         0.0f},
        {"rect-31x17x33-TN",
         31,
         17,
         33,
         gemm::Layout::colMajor,
         gemm::Layout::rowMajor,
         gemm::Layout::colMajor,
         gemm::Op::transpose,
         gemm::Op::none,
         2,
         1,
         2,
         -1.25f,
         0.5f},
        {"rect-64x48x33-TT",
         64,
         48,
         33,
         gemm::Layout::rowMajor,
         gemm::Layout::rowMajor,
         gemm::Layout::rowMajor,
         gemm::Op::transpose,
         gemm::Op::transpose,
         3,
         3,
         2,
         0.5f,
         -2.0f},
        {"skinny-7x5x3-colA",
         7,
         5,
         3,
         gemm::Layout::colMajor,
         gemm::Layout::rowMajor,
         gemm::Layout::rowMajor,
         gemm::Op::none,
         gemm::Op::none,
         4,
         1,
         0,
         2.0f,
         1.0f},
        {"padded-33x9x17",
         33,
         9,
         17,
         gemm::Layout::colMajor,
         gemm::Layout::colMajor,
         gemm::Layout::colMajor,
         gemm::Op::none,
         gemm::Op::transpose,
         5,
         2,
         3,
         1.0f,
         0.25f},
        {"k-zero-5x4",
         5,
         4,
         0,
         gemm::Layout::rowMajor,
         gemm::Layout::rowMajor,
         gemm::Layout::rowMajor,
         gemm::Op::none,
         gemm::Op::none,
         0,
         0,
         0,
         1.0f,
         2.0f},
        {"m-zero-0x4",
         0,
         4,
         3,
         gemm::Layout::rowMajor,
         gemm::Layout::rowMajor,
         gemm::Layout::rowMajor,
         gemm::Op::none,
         gemm::Op::none,
         0,
         0,
         0,
         1.0f,
         2.0f},
    };

    std::printf("# alpaka3 gemm reference cross-check schema=1 generator=lcg-6364136223846793005\n");
    std::printf("case_id,M,N,K,layout_a,layout_b,layout_c,op_a,op_b,ld_a,ld_b,ld_c,alpha,beta,i,j,value\n");

    for(std::size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index)
    {
        Case const& item = cases[index];
        std::size_t const storedRowsA = item.opA == gemm::Op::none ? item.m : item.k;
        std::size_t const storedColsA = item.opA == gemm::Op::none ? item.k : item.m;
        std::size_t const storedRowsB = item.opB == gemm::Op::none ? item.k : item.n;
        std::size_t const storedColsB = item.opB == gemm::Op::none ? item.n : item.k;

        Buffer<float> a(storedRowsA, storedColsA, item.layoutA, item.opA, item.padA);
        Buffer<float> b(storedRowsB, storedColsB, item.layoutB, item.opB, item.padB);
        Buffer<double> c(item.m, item.n, item.layoutC, gemm::Op::none, item.padC);

        std::uint64_t state = 0x0123'4567'89ab'cdefULL + 0x9e37'79b9'7f4a'7c15ULL * index;
        fill(a.storage, state);
        fill(b.storage, state);
        fill(c.storage, state);

        gemm::gemm_reference<double>(c.view(), a.view(), b.view(), item.alpha, item.beta);

        for(std::size_t i = 0; i < item.m; ++i)
        {
            for(std::size_t j = 0; j < item.n; ++j)
            {
                std::printf(
                    "%s,%zu,%zu,%zu,%s,%s,%s,%s,%s,%zu,%zu,%zu,%.9g,%.9g,%zu,%zu,%a\n",
                    item.id,
                    item.m,
                    item.n,
                    item.k,
                    layoutName(item.layoutA),
                    layoutName(item.layoutB),
                    layoutName(item.layoutC),
                    opName(item.opA),
                    opName(item.opB),
                    a.ld,
                    b.ld,
                    c.ld,
                    static_cast<double>(item.alpha),
                    static_cast<double>(item.beta),
                    i,
                    j,
                    c.view().stored(i, j));
            }
        }
    }
    return 0;
}
