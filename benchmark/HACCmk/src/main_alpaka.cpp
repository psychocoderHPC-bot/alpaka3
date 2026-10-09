/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "helper.hpp"
#include "misc.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iostream>

namespace haccmkAlpaka
{
    template<typename T>
    struct DeltaMove
    {
        T x;
        T y;
        T z;

        constexpr DeltaMove operator+(DeltaMove const& other)
        {
            return DeltaMove{x + other.x, y + other.y, z + other.z};
        }
    };

    constexpr void alpakaSimdizedInvoke(auto&& f, alpaka::concepts::SpecializationOf<DeltaMove> auto&&... args)
    {
        alpakaSimdizedInvoke(ALPAKA_FORWARD(f), ALPAKA_FORWARD(args).x...);
        alpakaSimdizedInvoke(ALPAKA_FORWARD(f), ALPAKA_FORWARD(args).y...);
        alpakaSimdizedInvoke(ALPAKA_FORWARD(f), ALPAKA_FORWARD(args).z...);
    }

    template<uint32_t T_width, typename T>
    constexpr auto makeSimdized(DeltaMove<T> const& value)
    {
        using SimdMemberType = ALPAKA_TYPEOF(alpaka::makeSimdized<T_width>(std::declval<T>()));
        DeltaMove<SimdMemberType> result;
        alpakaSimdizedInvoke([](alpaka::concepts::Simd auto& lhs, auto const& rhs) { lhs = rhs; }, result, value);
        return result;
    }

    /** Hybrid variant of the HACCmk Step10 kernel: SIMD inner loop spread across a thread block.
     *
     * One thread block is responsible for one active particle at a time (the `blocksInGrid` map lets a block
     * process several particles in a grid-stride loop). The inner `n2` reduction is split across the block's
     * threads via `SimdAlgo{worker::threadsInBlock}` so that each thread gets a SIMD-vectorized partial. The
     * per-thread `DeltaMove` partials are staged into shared memory and combined with a converging tree
     * reduction; thread 0 finally scales the block sum by `fcoeff`.
     *
     * The kernel is written with backend independent `onAcc` facilities only, so it compiles unchanged for the
     * CPU, CUDA, HIP and SYCL backends.
     */
    struct Step10KernelSimdCoop
    {
        /** Logical number of threads per frame. `FrameSpec` frame extent and shared memory sizing both use it. */
        static constexpr uint32_t blockSize = 128u;

        ALPAKA_FN_ACC auto operator()(
            auto const& acc,
            int n1,
            int n2,
            alpaka::concepts::IMdSpan auto xx,
            alpaka::concepts::IMdSpan auto yy,
            alpaka::concepts::IMdSpan auto zz,
            alpaka::concepts::IMdSpan auto mass,
            alpaka::concepts::IMdSpan auto vx2,
            alpaka::concepts::IMdSpan auto vy2,
            alpaka::concepts::IMdSpan auto vz2,
            float fsrrmax2,
            float mp_rsm2,
            float fcoeff) const
        {
            using namespace alpaka;

            constexpr float const ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, ma3 = -0.00109313,
                                  ma4 = 0.0000605491, ma5 = -0.00000147177;

            // Shared staging buffers for the per-thread partial sums, one per component.
            auto sx = onAcc::declareSharedMdArray<float, uniqueId()>(acc, CVec<uint32_t, blockSize>{});
            auto sy = onAcc::declareSharedMdArray<float, uniqueId()>(acc, CVec<uint32_t, blockSize>{});
            auto sz = onAcc::declareSharedMdArray<float, uniqueId()>(acc, CVec<uint32_t, blockSize>{});

            // Runtime block geometry. The physical thread count is not guaranteed to equal `blockSize`; on host
            // backends the frame may be executed with fewer threads and for a frame extent below the warp size the
            // backend may use more. All reduction code below therefore depends on the runtime values only.
            uint32_t const threads = static_cast<uint32_t>(acc[layer::thread].count().product());
            uint32_t const tid
                = static_cast<uint32_t>(linearize(acc[layer::thread].count(), acc[layer::thread].idx()));
            // Smallest power of two that covers all participating threads, bounded by the shared array size.
            // (A ternary avoids binding a reference to the static constexpr member, which nvcc cannot
            //  materialize in device code.)
            uint32_t const nextPow2Candidate = std::bit_ceil(threads > 0u ? threads : 1u);
            uint32_t const nextPow2 = nextPow2Candidate < blockSize ? nextPow2Candidate : blockSize;

            // Outer loop over active particles: each block handles a grid-stride subset of [0, n1).
            for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{n1}))
            {
                float xxi = xx[i];
                float yyi = yy[i];
                float zzi = zz[i];

                // Per-thread SIMD-vectorized partial of the inner n2 reduction.
                DeltaMove<float> move{0, 0, 0};
                auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInBlock};
                move = simdGrid.transformReduce(
                    acc,
                    Vec{n2},
                    DeltaMove<float>{0, 0, 0},
                    std::plus{},
                    [&](auto const&,
                        concepts::Simd auto const& simd_xx1,
                        concepts::Simd auto const& simd_yy1,
                        concepts::Simd auto const& simd_zz1,
                        concepts::Simd auto const& simd_mass1) constexpr
                    {
                        concepts::Simd auto dxc = simd_xx1 - xxi;
                        concepts::Simd auto dyc = simd_yy1 - yyi;
                        concepts::Simd auto dzc = simd_zz1 - zzi;

                        concepts::Simd auto r2 = dxc * dxc + dyc * dyc + dzc * dzc;

                        using SimdType = ALPAKA_TYPEOF(simd_mass1);
                        concepts::Simd auto m = SimdType::fill(0.f);

                        where(r2 < fsrrmax2, m) = simd_mass1;

                        alpaka::concepts::Simd auto tmp = r2 + mp_rsm2;
                        concepts::Simd auto bar = alpaka::math::sqrt(tmp);

                        concepts::Simd auto p = float{1.0} / (tmp * bar);

                        concepts::Simd auto f
                            = p - (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5)))));
                        concepts::Simd auto fac = SimdType::fill(0.f);
                        where(r2 > 0.0f, fac) = m * f;

                        using SimdizedType = decltype(makeSimdized<SimdType::width()>(move));
                        return SimdizedType{(fac * dxc), (fac * dyc), (fac * dzc)};
                    },
                    xx,
                    yy,
                    zz,
                    mass);

                /* Stage the partials into shared memory. Every index in [0, blockSize) is assigned exactly once:
                 * indices [0, threads) receive the owning thread's partial and indices [threads, blockSize) are
                 * explicitly zero-filled. The strided walk also covers the case threads < blockSize, where a thread
                 * owns more than one shared slot, and leaves unused slots zeroed for the next particle iteration.
                 */
                for(uint32_t k = tid; k < blockSize; k += threads)
                {
                    if(k < threads)
                    {
                        sx[k] = move.x;
                        sy[k] = move.y;
                        sz[k] = move.z;
                    }else
                    {
                        sx[k] = 0.f;
                        sy[k] = 0.f;
                        sz[k] = 0.f;
                    }
                }

                // Make the staged partials visible to the whole block before the reduction reads them.
                onAcc::syncBlockThreads(acc);

                /* Converging tree reduction over the power-of-two range [0, nextPow2). Writes touch [0, s) while
                 * reads touch [s, 2s), i.e. the two sets are disjoint within one step; the sync at the end of each
                 * step orders the next step. `nextPow2 <= blockSize` guarantees in-bounds access.
                 */
                for(uint32_t s = nextPow2 / 2u; s > 0u; s >>= 1u)
                {
                    if(tid < s)
                    {
                        sx[tid] += sx[tid + s];
                        sy[tid] += sy[tid + s];
                        sz[tid] += sz[tid + s];
                    }
                    onAcc::syncBlockThreads(acc);
                }

                // A single thread publishes the block result. The arithmetic matches the reference: sum the
                // per-particle contributions and multiply once by fcoeff.
                for([[maybe_unused]] auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{1u}))
                {
                    vx2[i] = sx[0] * fcoeff;
                    vy2[i] = sy[0] * fcoeff;
                    vz2[i] = sz[0] * fcoeff;
                }

                // Do not let a block start the next particle before every thread finished reading the shared sums.
                onAcc::syncBlockThreads(acc);
            }
        }
    };

    void haccmk(
        auto devAcc,
        auto exec,
        int const repeat,
        int const n1,
        int const n2,
        auto& xx,
        auto& yy,
        auto& zz,
        auto& mass,
        auto& vx2,
        auto& vy2,
        auto& vz2,
        float const fsrmax,
        float const mp_rsm,
        float const fcoeff)
    {
        using namespace alpaka;

        // a blocking queue is used to reduce the starting overhead for CPU devices
        onHost::Queue queue = devAcc.makeQueue(queueKind::blocking);

        auto d_xx = onHost::alloc<float>(devAcc, n2);
        auto d_yy = onHost::alloc<float>(devAcc, n2);
        auto d_zz = onHost::alloc<float>(devAcc, n2);
        auto d_mass = onHost::alloc<float>(devAcc, n2);
        auto d_vx2 = onHost::alloc<float>(devAcc, n1);
        auto d_vy2 = onHost::alloc<float>(devAcc, n1);
        auto d_vz2 = onHost::alloc<float>(devAcc, n1);

        onHost::memcpy(queue, d_xx, xx);
        onHost::memcpy(queue, d_yy, yy);
        onHost::memcpy(queue, d_zz, zz);
        onHost::memcpy(queue, d_mass, mass);

        // One block per active particle gives the most parallelism and hides the reduction latency best;
        // the kernel's grid-stride loop keeps this correct if fewer blocks are used.
        int const numFrames = std::max(1, n1);
        // The cooperative kernel relies on a block-per-frame mapping; the shared tree reduction uses blockSize slots.
        auto frameSpec = onHost::FrameSpec{
            numFrames,
            static_cast<int>(Step10KernelSimdCoop::blockSize),
            exec};

        std::cout << "FrameSpec " << frameSpec << std::endl;

        float total_time = 0.f;

        for(int i = 0; i < repeat; i++)
        {
            // reset output
            onHost::memcpy(queue, d_vx2, vx2, n1);
            onHost::memcpy(queue, d_vy2, vy2, n1);
            onHost::memcpy(queue, d_vz2, vz2, n1);

            onHost::wait(devAcc);
            auto start = std::chrono::steady_clock::now();

            auto const haccKernel = KernelBundle{
                Step10KernelSimdCoop{},
                n1,
                n2,
                d_xx,
                d_yy,
                d_zz,
                d_mass,
                d_vx2,
                d_vy2,
                d_vz2,
                fsrmax,
                mp_rsm,
                fcoeff};

            queue.enqueue(frameSpec, haccKernel);
            onHost::wait(devAcc);
            auto end = std::chrono::steady_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            // first round is a warmup round
            if(i != 0)
                total_time += time;
        }

        printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / (repeat - 1));
        onHost::memcpy(queue, vx2, d_vx2, n1);
        onHost::memcpy(queue, vy2, d_vy2, n1);
        onHost::memcpy(queue, vz2, d_vz2, n1);
        onHost::wait(devAcc);
    }

    auto example(auto const deviceSpec, auto const exec, int n2, int n1, int repeat) -> int
    {
        using namespace alpaka;

        using IdxVec = Vec<int, 1u>;

        // Define problem size
        IdxVec const extent(n2);

        std::cout << "Outer loop count is set " << n1 << std::endl;
        std::cout << "Inner loop count is set " << n2 << std::endl;
        std::cout << "Element type float" << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device devAcc = devSelector.makeDevice(0);


        auto xx = onHost::allocHost<float>(extent);
        auto yy = onHost::allocHost<float>(extent);
        auto zz = onHost::allocHost<float>(extent);
        auto mass = onHost::allocHost<float>(extent);
        auto vx2 = onHost::allocHost<float>(extent);
        auto vy2 = onHost::allocHost<float>(extent);
        auto vz2 = onHost::allocHost<float>(extent);
        auto vx2_hw = onHost::allocHost<float>(extent);
        auto vy2_hw = onHost::allocHost<float>(extent);
        auto vz2_hw = onHost::allocHost<float>(extent);

        float fsrrmax2, mp_rsm2, fcoeff, dx1, dy1, dz1, dx2, dy2, dz2;
        int i = 0;


        /* Initial data preparation */
        fcoeff = 0.23f;
        fsrrmax2 = 0.5f;
        mp_rsm2 = 0.03f;
        dx1 = 1.0f / static_cast<float>(n2);
        dy1 = 2.0f / static_cast<float>(n2);
        dz1 = 3.0f / static_cast<float>(n2);
        xx[0] = 0.f;
        yy[0] = 0.f;
        zz[0] = 0.f;
        mass[0] = 2.f;

        for(i = 1; i < n2; i++)
        {
            xx[i] = xx[i - 1] + dx1;
            yy[i] = yy[i - 1] + dy1;
            zz[i] = zz[i - 1] + dz1;
            mass[i] = static_cast<float>(i) * 0.01f + xx[i];
        }

        for(i = 0; i < n2; i++)
        {
            vx2[i] = 0.f;
            vy2[i] = 0.f;
            vz2[i] = 0.f;
            vx2_hw[i] = 0.f;
            vy2_hw[i] = 0.f;
            vz2_hw[i] = 0.f;
        }

        for(i = 0; i < n1; ++i)
        {
            hacc::haccmk_gold(n2, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass, &dx2, &dy2, &dz2);
            vx2[i] = vx2[i] + dx2 * fcoeff;
            vy2[i] = vy2[i] + dy2 * fcoeff;
            vz2[i] = vz2[i] + dz2 * fcoeff;
        }

        haccmk(devAcc, exec, repeat, n1, n2, xx, yy, zz, mass, vx2_hw, vy2_hw, vz2_hw, fsrrmax2, mp_rsm2, fcoeff);

        return hacc::verify(n2, vx2, vy2, vz2, vx2_hw, vy2_hw, vz2_hw);
    }
} // namespace haccmkAlpaka

auto main(int argc, char* argv[]) -> int
{
    // naming is taken from the original code
    int n2 = 1;
    int n1 = 1;
    int repeat = 1;

    if(int const ret = hacc::parseCmd(argc, argv, n2, n1, repeat))
        return ret;

    using namespace alpaka;

    /* Execute the example once for each backend (device specification + executor)
     *
     * If you would like to execute it for a single accelerator only you can use the following code.
     *  @code{.cpp}
     *  auto deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
     *  auto executor = exec::gpuCuda;
     *  return example(deviceSpec, executor, n2, n1, repeat);
     *  @endcode
     *
     * Some examples for device specifications (depending on the active dependencies).
     *
     *   onHost::DeviceSpec{api::host, deviceKind::cpu}
     *   onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu}
     *   onHost::DeviceSpec{api::hip, deviceKind::amdGpu}
     *   onHost::DeviceSpec{api::oneApi, deviceKind::intelGpu}
     *
     * A list of api's and device kinds can be found
     * https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html#available-apis
     * A list of executors can be found
     * https://alpaka3.readthedocs.io/en/latest/basic/cheatsheet.html#executors
     */
    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return haccmkAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), n2, n1, repeat); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
