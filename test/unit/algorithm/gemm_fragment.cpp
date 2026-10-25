/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace alpaka;
namespace fragment = alpaka::onAcc::fragment;

namespace
{
    template<typename T>
    struct HostView
    {
        T* data{};
        std::size_t ld{};

        T& at(std::size_t row, std::size_t col) const
        {
            return data[row * ld + col];
        }
    };

    template<typename T_Layout>
    std::vector<std::pair<uint32_t, uint32_t>> ownedCoordinates(
        uint32_t workerRow,
        uint32_t workerCol)
    {
        std::vector<std::pair<uint32_t, uint32_t>> result;
        for(uint32_t slot = 0u; slot < T_Layout::slotCount; ++slot)
            for(uint32_t lane = 0u; lane < T_Layout::width; ++lane)
                if(T_Layout::laneOwned(slot, lane))
                {
                    auto const local = T_Layout::blockCoord(workerRow, workerCol, slot, lane);
                    result.emplace_back(local.m, local.n);
                }
        return result;
    }

    template<typename T_Layout>
    void checkOwnership(uint32_t workerRows, uint32_t workerCols)
    {
        std::vector<uint32_t> owners(static_cast<std::size_t>(workerRows * T_Layout::tileM)
                                     * workerCols * T_Layout::tileN, 0u);
        for(uint32_t wr = 0u; wr < workerRows; ++wr)
            for(uint32_t wc = 0u; wc < workerCols; ++wc)
                for(uint32_t slot = 0u; slot < T_Layout::slotCount; ++slot)
                    for(uint32_t lane = 0u; lane < T_Layout::width; ++lane)
                    {
                        if(!T_Layout::laneOwned(slot, lane))
                            continue;
                        auto const block = T_Layout::blockCoord(wr, wc, slot, lane);
                        CHECK(block.m < workerRows * T_Layout::tileM);
                        CHECK(block.n < workerCols * T_Layout::tileN);
                        CHECK(T_Layout::owns(wr, wc, block.m, block.n));
                        ++owners[static_cast<std::size_t>(block.m) * workerCols * T_Layout::tileN + block.n];

                        auto const slotLane = T_Layout::blockSlotLane(wr, wc, block.m, block.n);
                        CHECK(slotLane.slot == slot);
                        CHECK(slotLane.lane == lane);
                    }
        for(auto const count : owners)
            CHECK(count == 1u);

        // The owned coordinate set of a worker is a rectangle and does not depend on V.
        auto const coordinates = ownedCoordinates<T_Layout>(workerRows - 1u, workerCols - 1u);
        CHECK(coordinates.size() == static_cast<std::size_t>(T_Layout::tileM) * T_Layout::tileN);
    }
} // namespace

TEST_CASE("gemm fragment ownership mapping is a bijection", "[gemm][fragment][mapping]")
{
    checkOwnership<fragment::RegisterBlockedLayout<3, 5, 1>>(1, 1);
    checkOwnership<fragment::RegisterBlockedLayout<4, 8, 1>>(2, 3);
    checkOwnership<fragment::RegisterBlockedLayout<4, 16, 8>>(2, 3);
    checkOwnership<fragment::RegisterBlockedLayout<5, 7, 8>>(3, 2);
    checkOwnership<fragment::RegisterBlockedLayout<7, 3, 2>>(2, 2);
    checkOwnership<fragment::RegisterBlockedLayout<8, 8, 4>>(4, 6);
    checkOwnership<fragment::RegisterBlockedLayout<16, 4, 1>>(2, 2);
}

TEST_CASE("gemm fragment mapping is independent of V and worker count", "[gemm][fragment][mapping]")
{
    using Narrow = fragment::RegisterBlockedLayout<5, 7, 1>;
    using Wide = fragment::RegisterBlockedLayout<5, 7, 8>;

    for(uint32_t wr = 0u; wr < 2u; ++wr)
        for(uint32_t wc = 0u; wc < 3u; ++wc)
        {
            auto narrow = ownedCoordinates<Narrow>(wr, wc);
            auto wide = ownedCoordinates<Wide>(wr, wc);
            std::sort(narrow.begin(), narrow.end());
            std::sort(wide.begin(), wide.end());
            CHECK(narrow == wide);
        }

    // The same worker stripe has the same global coordinates in a larger worker grid.
    using Small = fragment::RegisterBlockedOwnership<4, 8, 4, 2, 3>;
    using Large = fragment::RegisterBlockedOwnership<4, 8, 4, 4, 6>;
    Small const small{100u, 200u, 1u, 2u};
    Large const large{100u, 200u, 1u, 2u};
    for(uint32_t slot = 0u; slot < Small::Layout::slotCount; ++slot)
        for(uint32_t lane = 0u; lane < Small::Layout::width; ++lane)
        {
            auto const a = small.globalCoord(slot, lane);
            auto const b = large.globalCoord(slot, lane);
            CHECK(a.m == b.m);
            CHECK(a.n == b.n);
        }
}

TEST_CASE("gemm fragment operations stay inside worker storage", "[gemm][fragment][operations]")
{
    using Layout = fragment::RegisterBlockedLayout<3, 5, 4>;
    using Ownership = fragment::RegisterBlockedOwnership<3, 5, 4, 2, 3>;
    using Var = onAcc::internal::LockstepVar<
        float,
        CVec<std::uint32_t, 2 * 3, 3 * Layout::storageN>,
        CVec<std::uint32_t, 2 * 3>>;

    static_assert(Var::numSlotsPerWorker == Layout::storageElements);
    static_assert(fragment::internal::compatibleStorage<Var, float, Layout>);
    static_assert(!fragment::internal::compatibleStorage<Var, double, Layout>);
    static_assert(!fragment::internal::compatibleStorage<Var, float, fragment::RegisterBlockedLayout<9, 5, 4>>);

    Var var{};
    Ownership const ownership{10u, 20u, 1u, 2u};
    auto tile = fragment::makeFragment<float, Layout>(var, ownership);

    tile.fill(1.5f);
    std::size_t visited = 0u;
    tile.concurrent(
        [&](auto const& global, float& value)
        {
            CHECK((global.m >= 10u + 3u && global.m < 10u + 6u));
            CHECK((global.n >= 20u + 10u && global.n < 20u + 15u));
            CHECK(value == 1.5f);
            ++visited;
        });
    CHECK(visited == static_cast<std::size_t>(Layout::tileM) * Layout::tileN);

    tile.row(1).at(4) = 7.0f;
    CHECK(tile.pack(1 * Layout::packsPerRow + 1)[0] == 7.0f);
    CHECK(tile.column(4).at(1) == 7.0f);
    auto const broadcast = tile.broadcast(1u, 4u);
    CHECK(broadcast.value == 7.0f);
    auto const vector = broadcast.vector();
    for(uint32_t lane = 0u; lane < Layout::width; ++lane)
        CHECK(vector[lane] == 7.0f);

    // loadFrom/storeTo exercise guarded global access.
    std::vector<float> global(32u * 32u, 0.0f);
    HostView<float> view{global.data(), 32u};
    tile.loadFrom(view, 32u, 32u);
    CHECK(tile.column(0).at(0) == 0.0f);
    tile.fill(2.0f);
    tile.storeTo(view, [](auto const&, float value) { return value + 1.0f; }, 32u, 32u);
    CHECK(global[(10u + 3u) * 32u + (20u + 10u)] == 3.0f);

    // Coordinates outside the total view are neither read nor written.
    tile.fill(9.0f);
    tile.storeTo(view, [](auto const&, float value) { return value; }, 11u, 31u);
    CHECK(global[(10u + 3u) * 32u + (20u + 10u)] == 3.0f);
}

namespace
{
    template<typename T_Element>
    struct ManualTileKernel
    {
        onHost::GemmView<T_Element> c;
        onHost::GemmView<T_Element> a;
        onHost::GemmView<T_Element> b;
        std::uint64_t m;
        std::uint64_t n;
        std::uint64_t k;
        T_Element alpha;
        T_Element beta;
        std::uint32_t tileBaseM;
        std::uint32_t tileBaseN;
        std::uint32_t tilesN;
        std::uint32_t totalTiles;

        template<
            std::uint32_t TM,
            std::uint32_t TN,
            std::uint32_t V,
            std::uint32_t BK,
            typename T_Acc>
        ALPAKA_FN_ACC void run(T_Acc const& acc) const
        {
            constexpr std::uint32_t packs = (TN + V - 1u) / V;
            std::uint32_t const linearBlock = static_cast<std::uint32_t>(
                alpaka::linearize(acc[layer::block].count(), acc[layer::block].idx()));
            std::uint32_t const blockCount = static_cast<std::uint32_t>(acc[layer::block].count().product());

            for(std::uint32_t linear = linearBlock; linear < totalTiles; linear += blockCount)
            {
                std::uint64_t const originM
                    = (static_cast<std::uint64_t>(tileBaseM) + linear / tilesN) * TM;
                std::uint64_t const originN
                    = (static_cast<std::uint64_t>(tileBaseN) + linear % tilesN) * TN;

                std::array<Simd<T_Element, V>, TM * packs> sum;
                for(auto& value : sum)
                    value = Simd<T_Element, V>::fill(T_Element{0});

                for(std::uint64_t k0 = 0u; k0 < k;)
                {
                    for(std::uint32_t kk = 0u; kk < BK; ++kk)
                    {
                        std::uint64_t const kIndex = k0 + kk;
                        std::array<T_Element, TM> av{};
                        std::array<Simd<T_Element, V>, packs> bv;
                        for(std::uint32_t r = 0u; r < TM; ++r)
                        {
                            av[r] = T_Element{0};
                            if(originM + r < m && kIndex < k)
                                av[r] = a.at(originM + r, kIndex);
                        }
                        for(std::uint32_t p = 0u; p < packs; ++p)
                        {
                            bv[p] = Simd<T_Element, V>::fill(T_Element{0});
                            for(std::uint32_t lane = 0u; lane < V; ++lane)
                            {
                                std::uint64_t const localN = p * V + lane;
                                if(localN < TN && kIndex < k && originN + localN < n)
                                    bv[p][lane] = b.at(kIndex, originN + localN);
                            }
                        }
                        for(std::uint32_t r = 0u; r < TM; ++r)
                            for(std::uint32_t p = 0u; p < packs; ++p)
                                sum[r * packs + p] = sum[r * packs + p] + av[r] * bv[p];
                    }
                    if(k - k0 <= BK)
                        break;
                    k0 += BK;
                }

                for(std::uint32_t r = 0u; r < TM; ++r)
                {
                    std::uint64_t const i = originM + r;
                    if(i >= m)
                        continue;
                    for(std::uint32_t p = 0u; p < packs; ++p)
                        for(std::uint32_t lane = 0u; lane < V; ++lane)
                        {
                            std::uint64_t const localN = p * V + lane;
                            std::uint64_t const j = originN + localN;
                            if(localN >= TN || j >= n)
                                continue;
                            T_Element value = alpha * sum[r * packs + p][lane];
                            if(beta != T_Element{0})
                            {
                                T_Element const old = c.at(i, j);
                                value += beta == T_Element{1} ? old : beta * old;
                            }
                            c.at(i, j) = value;
                        }
                }
            }
        }

        ALPAKA_FN_ACC void operator()(auto const& acc) const
        {
            using AccType = std::remove_cvref_t<decltype(acc)>;
            using Api = ALPAKA_TYPEOF(std::declval<AccType const&>().getApi());
            using DeviceKind = ALPAKA_TYPEOF(std::declval<AccType const&>().getDeviceKind());
            // The benchmark only covers the host CPU plan, whose block has one worker and one worker stripe.
            constexpr std::uint32_t tm = 4u;
            constexpr std::uint32_t tn = 16u;
            constexpr std::uint32_t v = fragment::registerBlockedWidth<T_Element, Api, DeviceKind, tn>();
            constexpr std::uint32_t bk = getArchSimdWidth<T_Element>(Api{}, DeviceKind{}) >= 8u ? 16u : 8u;
            run<tm, tn, v, bk>(acc);
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "gemm register-blocked and simple paths agree",
    "[gemm][fragment][paths]",
    std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    auto device = test::getDevice(deviceExec);
    auto exec = test::getExecutor(deviceExec);
    auto queue = device.makeQueue();

    constexpr std::size_t m = 37u;
    constexpr std::size_t n = 29u;
    constexpr std::size_t k = 19u;
    std::vector<float> aHost(m * k), bHost(k * n), cHost(m * n, 0.0f);
    for(std::size_t i = 0u; i < aHost.size(); ++i)
        aHost[i] = static_cast<float>((i * 7u) % 11u) * 0.125f - 0.5f;
    for(std::size_t i = 0u; i < bHost.size(); ++i)
        bHost[i] = static_cast<float>((i * 5u) % 13u) * 0.0625f - 0.25f;

    auto dA = onHost::alloc<float>(device, Vec{aHost.size()});
    auto dB = onHost::alloc<float>(device, Vec{bHost.size()});
    auto dC = onHost::alloc<float>(device, Vec{cHost.size()});
    onHost::memcpy(queue, dA, aHost);
    onHost::memcpy(queue, dB, bHost);

    onHost::GemmView<float> const viewA{onHost::data(dA), m, k, k, onHost::GemmLayout::rowMajor, onHost::GemmOp::none};
    onHost::GemmView<float> const viewB{onHost::data(dB), k, n, n, onHost::GemmLayout::rowMajor, onHost::GemmOp::none};
    onHost::GemmView<float> const viewC{onHost::data(dC), m, n, n, onHost::GemmLayout::rowMajor, onHost::GemmOp::none};

    onHost::memcpy(queue, dC, cHost);
    onHost::gemm(
        queue,
        exec,
        viewC,
        viewA,
        viewB,
        0.75f,
        0.5f,
        onHost::GemmNumericMode::fp32,
        onHost::GemmKernel::registerBlocked);
    std::vector<float> blocked(cHost.size());
    onHost::memcpy(queue, blocked, dC);

    onHost::memcpy(queue, dC, cHost);
    onHost::gemm(
        queue,
        exec,
        viewC,
        viewA,
        viewB,
        0.75f,
        0.5f,
        onHost::GemmNumericMode::fp32,
        onHost::GemmKernel::simple);
    std::vector<float> simple(cHost.size());
    onHost::memcpy(queue, simple, dC);
    onHost::wait(queue);

    double maxDifference = 0.0;
    for(std::size_t i = 0u; i < blocked.size(); ++i)
        maxDifference = std::max(maxDifference, static_cast<double>(std::abs(blocked[i] - simple[i])));
    INFO("max path difference " << maxDifference);
    CHECK(maxDifference <= 1.0e-5);
}

TEMPLATE_LIST_TEST_CASE(
    "gemm register-blocked vs manual indexed throughput",
    "[.][gemm][fragment][benchmark]",
    std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>)
{
    using Exec = std::remove_cvref_t<decltype(alpaka::getExecutor(TestType::makeDict()))>;
    if constexpr(!std::same_as<Exec, exec::CpuOmpBlocks>)
    {
        SKIP("throughput comparison only runs on the host OpenMP executor");
    }
    else
    {
        auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
        auto device = test::getDevice(deviceExec);
        auto exec = test::getExecutor(deviceExec);
        auto queue = device.makeQueue();

        using Api = std::remove_cvref_t<decltype(queue.getApi())>;
        using DeviceKind = std::remove_cvref_t<decltype(queue.getDeviceKind())>;
        constexpr auto plan = onHost::internal::registerTileConfig<float, Api, DeviceKind, Exec>();
        static_assert(plan.wr == 1u);
        static_assert(plan.wc == 1u, "the manual benchmark covers the one-worker-per-block host plan");
        constexpr std::uint32_t tm = plan.tm;
        constexpr std::uint32_t tn = plan.tn;
        constexpr std::uint32_t bk = plan.bk;
        constexpr std::uint32_t threads = plan.wc;
        constexpr std::uint32_t bm = tm;
        constexpr std::uint32_t bn = tn;

        std::vector<std::array<std::size_t, 3>> const shapes{
            {256u, 256u, 256u},
            {512u, 512u, 512u},
            {1024u, 1024u, 512u}};

        std::printf("[fragment benchmark] plan TM=%u TN=%u BK=%u threads=%u\n", tm, tn, bk, threads);
        std::printf("%-18s %12s %12s %10s\n", "shape", "register[ms]", "manual[ms]", "ratio");

        for(auto const& shape : shapes)
        {
            std::size_t const m = shape[0];
            std::size_t const n = shape[1];
            std::size_t const k = shape[2];
            std::vector<float> aHost(m * k), bHost(k * n);
            for(std::size_t i = 0u; i < aHost.size(); ++i)
                aHost[i] = static_cast<float>((i * 13u) % 17u) * 0.03125f - 0.25f;
            for(std::size_t i = 0u; i < bHost.size(); ++i)
                bHost[i] = static_cast<float>((i * 11u) % 19u) * 0.03125f - 0.25f;

            auto dA = onHost::alloc<float>(device, Vec{aHost.size()});
            auto dB = onHost::alloc<float>(device, Vec{bHost.size()});
            auto dC = onHost::alloc<float>(device, Vec{m * n});
            onHost::memcpy(queue, dA, aHost);
            onHost::memcpy(queue, dB, bHost);

            onHost::GemmView<float> const viewA{
                onHost::data(dA),
                m,
                k,
                k,
                onHost::GemmLayout::rowMajor,
                onHost::GemmOp::none};
            onHost::GemmView<float> const viewB{
                onHost::data(dB),
                k,
                n,
                n,
                onHost::GemmLayout::rowMajor,
                onHost::GemmOp::none};
            onHost::GemmView<float> const viewC{
                onHost::data(dC),
                m,
                n,
                n,
                onHost::GemmLayout::rowMajor,
                onHost::GemmOp::none};

            auto const regions = onHost::internal::splitRegions(m, n, bm, bn);

            auto const runManual = [&]
            {
                auto const launchRegion = [&](onHost::internal::TileRegion const& region)
                {
                    std::uint64_t const rows = region.tileRowEnd - region.tileRowBegin;
                    std::uint64_t const cols = region.tileColEnd - region.tileColBegin;
                    if(rows == 0u || cols == 0u)
                        return;
                    auto const regionTiles = static_cast<std::uint32_t>(rows * cols);
                    queue.enqueue(
                        onHost::FrameSpec{Vec{regionTiles}, CVec<std::uint32_t, threads>{}, exec},
                        KernelBundle{ManualTileKernel<float>{
                            viewC,
                            viewA,
                            viewB,
                            m,
                            n,
                            k,
                            1.0f,
                            0.0f,
                            static_cast<std::uint32_t>(region.tileRowBegin),
                            static_cast<std::uint32_t>(region.tileColBegin),
                            static_cast<std::uint32_t>(cols),
                            regionTiles}});
                };
                launchRegion(regions.full);
                launchRegion(regions.bottom);
                launchRegion(regions.right);
                launchRegion(regions.corner);
                onHost::wait(queue);
            };
            auto const runRegister = [&]
            {
                onHost::gemm(
                    queue,
                    exec,
                    viewC,
                    viewA,
                    viewB,
                    1.0f,
                    0.0f,
                    onHost::GemmNumericMode::fp32,
                    onHost::GemmKernel::registerBlocked);
                onHost::wait(queue);
            };

            // Correctness of the manual reference against the library path.
            std::vector<float> reference(m * n), manualResult(m * n);
            runRegister();
            onHost::memcpy(queue, reference, dC);
            runManual();
            onHost::memcpy(queue, manualResult, dC);
            onHost::wait(queue);
            CHECK(std::equal(reference.begin(), reference.end(), manualResult.begin()));

            for(int warmup = 0; warmup < 3; ++warmup)
            {
                runRegister();
                runManual();
            }

            constexpr int rounds = 3;
            constexpr int samples = 10;
            std::vector<double> registerTimes;
            std::vector<double> manualTimes;
            for(int round = 0; round < rounds; ++round)
                for(int sample = 0; sample < samples; ++sample)
                {
                    auto const time = [&](auto&& op)
                    {
                        auto const begin = std::chrono::steady_clock::now();
                        op();
                        auto const end = std::chrono::steady_clock::now();
                        return std::chrono::duration<double, std::milli>(end - begin).count();
                    };
                    if((round + sample) % 2 == 0)
                    {
                        registerTimes.push_back(time(runRegister));
                        manualTimes.push_back(time(runManual));
                    }
                    else
                    {
                        manualTimes.push_back(time(runManual));
                        registerTimes.push_back(time(runRegister));
                    }
                }

            auto const median = [](std::vector<double> values)
            {
                std::sort(values.begin(), values.end());
                return values[values.size() / 2u];
            };
            auto const spread = [](std::vector<double> values)
            {
                std::sort(values.begin(), values.end());
                return values[values.size() - 1u] - values.front();
            };
            double const rMedian = median(registerTimes);
            double const mMedian = median(manualTimes);
            double const ratio = mMedian / rMedian;
            std::printf(
                "%-18s %12.4f %12.4f %10.3f\n",
                (std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k)).c_str(),
                rMedian,
                mMedian,
                ratio);
            std::printf(
                "  spread: register %.4f ms, manual %.4f ms, throughput %.3f TFLOP/s (2*M*N*K)\n",
                spread(registerTimes),
                spread(manualTimes),
                2.0 * static_cast<double>(m) * n * k / (rMedian * 1.0e9));
            CHECK(ratio >= 0.95);
        }
    }
}
