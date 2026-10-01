/* Copyright 2026 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "helpers.hpp"

#include <alpaka/alpaka.hpp>

//! alpaka version of explicit finite-difference 2D heat equation solver using a lockstep execution scope.
//!
//! This is the lockstep counterpart of StencilKernel. It processes a compile-time known core tile with
//! onAcc::LockstepScope::concurrent() while loading the shared memory tile exactly like the original kernel.
//! The logical iteration space per block is the compile-time tile; the physical block extent (number of workers)
//! is derived from the launch frame specification and may be smaller than the tile.
//!
//! \param uCurrBuf Current buffer with grid values of u for each x, y pair and the current value of t:
//!                 u(x, y, t) | t = t_current
//! \param uNextBuf resulting grid values of u for each x, y pair and the next value of t:
//!              u(x, y, t) | t = t_current + dt
//! \param numNodes number of nodes per dimension of the simulation domain, used to derive the block grid
//! \param tile compile-time core tile size (CVector), the logical iteration space of the lockstep scope
//! \param sharedMemExtents size of the shared memory box (tile + halo per dimension)
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t
struct StencilKernelLockstep
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::concepts::IMdSpan auto const& uCurrBuf,
        alpaka::concepts::IMdSpan auto uNextBuf,
        alpaka::concepts::Vector auto numNodes,
        alpaka::concepts::CVector auto tile,
        alpaka::concepts::CVector auto sharedMemExtents,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        using namespace alpaka;

        for(auto blockStartIdx :
            onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{0u, 0u}, numNodes, tile}))
        {
            // CUDA 13.1/13.2 bug workaround: concepts can not be used in a range based for loop
            static_assert(concepts::Dim<ALPAKA_TYPEOF(blockStartIdx), 2u>);
            auto sdata = onAcc::declareSharedMdArray<double, uniqueId()>(acc, sharedMemExtents);

            // avoid data race with the stencil calculation at the end
            onAcc::syncBlockThreads(acc);

            // Load the shared memory tile exactly like the original StencilKernel: every worker loads whole
            // (tile + halo) box at the block origin.
            for(auto idx2d : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{sharedMemExtents}))
            {
                // CUDA 13.1/13.2 bug workaround: concepts can not be used in a range based for loop
                static_assert(concepts::Dim<ALPAKA_TYPEOF(idx2d), 2u>);
                auto bufIdx = idx2d + blockStartIdx;
                sdata[idx2d] = uCurrBuf[bufIdx];
            }

            onAcc::syncBlockThreads(acc);

            double const rX = dt / (dx * dx);
            double const rY = dt / (dy * dy);

            // The lockstep scope iterates the compile-time core tile in lockstep over the block workers.
            // xDir is dim1 and yDir is dim0, matching the original kernel. Shared memory offsets are expressed
            // as non-negative absolute offsets from the core element to avoid unsigned underflow.
            auto scope = onAcc::makeLockstep(acc, onAcc::worker::threadsInBlock, tile);
            scope.template concurrent<double>(
                [rX, rY](
                    [[maybe_unused]] auto const& idx,
                    auto uNext,
                    auto center,
                    auto left,
                    auto right,
                    auto down,
                    auto up)
                {
                    // same arithmetic order as the original kernel
                    uNext = center.load() * (1.0 - 2.0 * rX - 2.0 * rY) + (left.load() + right.load()) * rX
                            + (down.load() + up.load()) * rY;
                },
                // output core cell, shifted by the block origin; the original kernel computes only core cells
                // `IdxRange{chunkSize} >> 1u`, i.e. the core is offset by the halo {1,1} from the block origin
                onAcc::map(uNextBuf, blockStartIdx + Vec{1u, 1u}),
                // center = (1, 1)
                onAcc::map(sdata, Vec{1u, 1u}),
                // x- direction (left)
                onAcc::map(sdata, Vec{1u, 0u}),
                // x+ direction (right)
                onAcc::map(sdata, Vec{1u, 2u}),
                // y- direction (down)
                onAcc::map(sdata, Vec{0u, 1u}),
                // y+ direction (up)
                onAcc::map(sdata, Vec{2u, 1u}));
        }
    }
};
