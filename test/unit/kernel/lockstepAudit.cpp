/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

using namespace alpaka;

namespace
{
    /** Host accelerator placeholder carrying the current worker index.
     *
     * The lockstep scope only needs the API and device kind (for compile-time SIMD width selection) and the worker
     * index (through the work group). A real queue is not required to exercise the iteration and slot assignment
     * logic, so this test drives the same foreachImpl() code path on the host with a deterministic worker count.
     */
    struct FakeHostAcc
    {
        uint32_t workerIdx = 0u;

        constexpr auto getApi() const
        {
            return alpaka::api::Host{};
        }

        constexpr auto getDeviceKind() const
        {
            return alpaka::deviceKind::Cpu{};
        }
    };

    /** Work group with a compile-time worker count and a worker index taken from the fake accelerator. */
    template<uint32_t T_numWorkers>
    struct StaticWorkerGroup
    {
        constexpr auto getThreadSpace(auto const& acc) const
        {
            return ThreadSpace{Vec{acc.workerIdx}, CVec<uint32_t, T_numWorkers>{}};
        }
    };

    template<typename T_ValueType, typename T_Device>
    consteval uint32_t archSimdWidthFor()
    {
        using Api = std::remove_cvref_t<decltype(std::declval<T_Device const&>().getApi())>;
        using DeviceKind = std::remove_cvref_t<decltype(std::declval<T_Device const&>().getDeviceKind())>;
        return getArchSimdWidth<T_ValueType>(Api{}, DeviceKind{});
    }

    ALPAKA_FN_HOST_ACC constexpr auto linearValue(auto const& value)
    {
        if constexpr(alpaka::concepts::Simd<std::decay_t<decltype(value)>>)
        {
            using SimdType = std::decay_t<decltype(value)>;
            return Simd<int32_t, SimdType::width()>{
                [&](auto laneIdx) constexpr { return static_cast<int32_t>(value[static_cast<uint32_t>(laneIdx)]); }};
        }
        else
        {
            return static_cast<int32_t>(value);
        }
    }

    /** Visit every logical element once, per worker, and check exact-once coverage.
     *
     * @return the maximum SIMD width taken by any iteration
     */
    template<uint32_t T_logical, uint32_t T_workers>
    int32_t runCoverageCase()
    {
        using LogicalExtent = CVec<uint32_t, T_logical>;
        std::array<int32_t, T_logical> count{};
        std::array<int32_t, T_logical> out{};
        int32_t maxWidth = 0;

        auto outSpan = makeMdSpan(out.data(), Vec{T_logical});

        for(uint32_t worker = 0u; worker < T_workers; ++worker)
        {
            FakeHostAcc acc{worker};
            auto scope = onAcc::makeLockstep(acc, StaticWorkerGroup<T_workers>{}, LogicalExtent{});
            scope.concurrent(
                [&](auto const& idx, auto outRef)
                {
                    using IdxType = std::remove_cvref_t<decltype(idx)>;
                    constexpr uint32_t width = IdxType::width();
                    maxWidth = std::max(maxWidth, static_cast<int32_t>(width));

                    auto const linear = idx.linear();
                    if constexpr(width == 1u)
                        ++count[static_cast<uint32_t>(linear)];
                    else
                        for(uint32_t lane = 0u; lane < width; ++lane)
                            ++count[static_cast<uint32_t>(linear[lane])];

                    outRef = linearValue(linear) + int32_t{1};
                },
                onAcc::map(outSpan));
        }

        for(uint32_t i = 0u; i < T_logical; ++i)
        {
            CAPTURE(i);
            CHECK(count[i] == int32_t{1});
            CHECK(out[i] == static_cast<int32_t>(i) + int32_t{1});
        }

        INFO("logical size : " << T_logical);
        INFO("workers      : " << T_workers);
        INFO("max width    : " << maxWidth);
        return maxWidth;
    }

    /** Keep one logical element in the same distributed slot while the SIMD width changes between passes.
     *
     * Pass 1 uses the width of int32_t, pass 2 the width of uint8_t, and pass 3 the width deduced from the value type
     * of the distributed variable. Every worker owns a private distributed variable, exactly as in a kernel. If the
     * divCeil-based slot assignment changed with the width, element i would read the value of another element and
     * outA/outB would not equal i + 2.
     */
    template<uint32_t T_logical, uint32_t T_workers>
    void runSlotStabilityCase()
    {
        using LogicalExtent = CVec<uint32_t, T_logical>;
        std::array<int32_t, T_logical> outA{};
        std::array<int32_t, T_logical> outB{};
        std::array<int32_t, 3u> widths{};
        auto outASpan = makeMdSpan(outA.data(), Vec{T_logical});
        auto outBSpan = makeMdSpan(outB.data(), Vec{T_logical});

        for(uint32_t worker = 0u; worker < T_workers; ++worker)
        {
            FakeHostAcc acc{worker};
            auto scope = onAcc::makeLockstep(acc, StaticWorkerGroup<T_workers>{}, LogicalExtent{});
            auto distributed = scope.template var<int32_t>();

            scope.template concurrent<int32_t>(
                [&](auto const& idx, auto varRef)
                {
                    using IdxType = std::remove_cvref_t<decltype(idx)>;
                    widths[0] = std::max(widths[0], static_cast<int32_t>(IdxType::width()));
                    varRef = linearValue(idx.linear()) + int32_t{1};
                },
                distributed);

            scope.template concurrent<std::uint8_t>(
                [&](auto const& idx, auto varRef, auto outRef)
                {
                    using IdxType = std::remove_cvref_t<decltype(idx)>;
                    widths[1] = std::max(widths[1], static_cast<int32_t>(IdxType::width()));
                    auto value = varRef.load() + int32_t{1};
                    outRef = value;
                    varRef = value;
                },
                distributed,
                onAcc::map(outASpan));

            scope.concurrent(
                [&](auto const& idx, auto varRef, auto outRef)
                {
                    using IdxType = std::remove_cvref_t<decltype(idx)>;
                    widths[2] = std::max(widths[2], static_cast<int32_t>(IdxType::width()));
                    outRef = varRef.load();
                },
                distributed,
                onAcc::map(outBSpan));
        }

        for(uint32_t i = 0u; i < T_logical; ++i)
        {
            CAPTURE(i);
            CHECK(outA[i] == static_cast<int32_t>(i) + int32_t{2});
            CHECK(outB[i] == static_cast<int32_t>(i) + int32_t{2});
        }

        INFO("logical size : " << T_logical);
        INFO("workers      : " << T_workers);
        INFO("width pass1  : " << widths[0]);
        INFO("width pass2  : " << widths[1]);
        INFO("width pass3  : " << widths[2]);

        if constexpr(archSimdWidthFor<std::uint8_t, FakeHostAcc>() != archSimdWidthFor<int32_t, FakeHostAcc>())
        {
            // The passes must really use different widths, otherwise this case would not check cross-width storage
            // stability.
            CHECK(widths[0] != widths[1]);
        }
    }
} // namespace

TEST_CASE("lockstep audit compile-time invariants", "[kernel][lockstep][audit]")
{
    // --- BindValueType_t is well-formed for an ordinary data argument, a local variable, and a mapped argument ---
    int32_t storage[4]{};
    auto dataSource = makeMdSpan(storage, Vec{4u});
    using DataSource = ALPAKA_TYPEOF(dataSource);
    using LockstepVar = onAcc::internal::LockstepVar<int32_t, CVec<uint32_t, 8u>, CVec<uint32_t, 4u>>;

    static_assert(std::is_same_v<onAcc::internal::BindValueType_t<DataSource>, int32_t>);
    static_assert(std::is_same_v<onAcc::internal::BindValueType_t<DataSource&>, int32_t>);
    static_assert(std::is_same_v<onAcc::internal::BindValueType_t<DataSource const&>, int32_t>);
    static_assert(std::is_same_v<onAcc::internal::BindValueType_t<LockstepVar>, int32_t>);
    static_assert(std::is_same_v<onAcc::internal::BindValueType_t<LockstepVar&>, int32_t>);
    static_assert(std::is_same_v<onAcc::internal::BindValueType_t<ALPAKA_TYPEOF(onAcc::map(dataSource))>, int32_t>);
    static_assert(std::is_same_v<onAcc::internal::BindValueType_t<ALPAKA_TYPEOF(onAcc::map(dataSource, 1))>, int32_t>);

    // --- The scope stores the accelerator by reference, so the scope cannot detach from its lifetime ---
    using Scope = decltype(onAcc::makeLockstep(
        std::declval<FakeHostAcc const&>(),
        StaticWorkerGroup<4u>{},
        CVec<uint32_t, 16u>{}));
    static_assert(std::is_same_v<typename Scope::AcceleratorStorage, FakeHostAcc const&>);
    static_assert(std::is_reference_v<typename Scope::AcceleratorStorage>);

    // --- Zero worker counts are rejected by a compile-time guard (see the static_assert in LockstepScope) ---
    static_assert(onAcc::internal::lockstepWorkerCountIsValid<CVec<uint32_t, 1u>>());
    static_assert(onAcc::internal::lockstepWorkerCountIsValid<CVec<uint32_t, 4u, 2u>>());
    static_assert(!onAcc::internal::lockstepWorkerCountIsValid<CVec<uint32_t, 0u>>());
    static_assert(!onAcc::internal::lockstepWorkerCountIsValid<CVec<uint32_t, 0u, 3u>>());

    // --- divCeil slot assignment ---
    // Any change to the divCeil-based slot partition breaks these values; the runtime storage-layout case below then
    // observes mismatched elements.
    static_assert(
        onAcc::internal::LockstepVar<int32_t, CVec<uint32_t, 97u>, CVec<uint32_t, 4u>>::numSlotsPerWorker == 25u);
    static_assert(
        onAcc::internal::LockstepVar<int32_t, CVec<uint32_t, 8u>, CVec<uint32_t, 4u>>::numSlotsPerWorker == 2u);
    static_assert(
        onAcc::internal::LockstepVar<int32_t, CVec<uint32_t, 5u>, CVec<uint32_t, 16u>>::numSlotsPerWorker == 1u);
    static_assert(
        onAcc::internal::LockstepVar<int32_t, CVec<uint32_t, 4u>, CVec<uint32_t, 4u>>::numSlotsPerWorker == 1u);
    static_assert(
        onAcc::internal::LockstepVar<int32_t, CVec<uint32_t, 1u>, CVec<uint32_t, 4u>>::numSlotsPerWorker == 1u);

    // --- SIMD width clamping ---
    // workers > logical elements: at most one element per worker, so the width must clamp to 1.
    static_assert(
        onAcc::internal::getLockstepSimdWidth<FakeHostAcc, int32_t>(CVec<uint32_t, 5u>{}, CVec<uint32_t, 16u>{})
        == 1u);
    // A single logical element always uses width 1, independent of the architecture width.
    static_assert(
        onAcc::internal::getLockstepSimdWidth<FakeHostAcc, int32_t>(CVec<uint32_t, 1u>{}, CVec<uint32_t, 1u>{}) == 1u);
    static_assert(onAcc::internal::getLockstepSimdWidth<FakeHostAcc, int32_t>(CVec<uint32_t, 1u>{}) == 1u);
    // With enough elements the width is the architecture width clamped to the number of owned elements.
    constexpr uint32_t width97x4
        = onAcc::internal::getLockstepSimdWidth<FakeHostAcc, int32_t>(CVec<uint32_t, 97u>{}, CVec<uint32_t, 4u>{});
    constexpr uint32_t archWidth = getArchSimdWidth<int32_t>(alpaka::api::Host{}, alpaka::deviceKind::Cpu{});
    static_assert(width97x4 >= 1u && width97x4 <= archWidth);

    // --- Const local-variable binding loads ordinary value types; the store operators are unavailable ---
    constexpr LockstepVar constVar{{1, 2}};
    using ConstRef = decltype(constVar.template bind<2u>(0u, std::array<Vec<uint32_t, 1u>, 2u>{}));
    static_assert(std::is_same_v<typename ConstRef::value_type, int32_t>);
    static_assert(std::is_same_v<ALPAKA_TYPEOF(std::declval<ConstRef>().load()), Simd<int32_t, 2u>>);
    // A read-only binding keeps a pointer to const storage. Writing through it is a compile error (verified with a
    // standalone probe, because a positive translation unit cannot contain the failing expression).
    static_assert(std::is_const_v<std::remove_pointer_t<decltype(std::declval<ConstRef>().ptr)>>);

    auto const constRef = constVar.template bind<2u>(0u, std::array<Vec<uint32_t, 1u>, 2u>{});
    auto const loaded = constRef.load();
    CHECK(loaded[0u] == int32_t{1});
    CHECK(loaded[1u] == int32_t{2});

    // A mutable binding keeps its store operators and round-trips a value.
    LockstepVar mutableVar{{0, 0}};
    auto mutableRef = mutableVar.template bind<2u>(0u, std::array<Vec<uint32_t, 1u>, 2u>{});
    mutableRef = Simd<int32_t, 2u>{int32_t{7}, int32_t{9}};
    auto const roundTrip = mutableRef.load();
    CHECK(roundTrip[0u] == int32_t{7});
    CHECK(roundTrip[1u] == int32_t{9});
}

TEST_CASE("lockstep audit storage layout", "[kernel][lockstep][audit]")
{
    // Odd logical extent (prime), workers < logical: mixed wide and scalar tail iterations.
    runSlotStabilityCase<97u, 4u>();
    // Odd extent with three workers, so the wide packs and the tail are not aligned to the worker count.
    runSlotStabilityCase<67u, 3u>();
    // One worker: every element is owned by the same worker.
    runSlotStabilityCase<31u, 1u>();
}

TEST_CASE("lockstep audit coverage edge cases", "[kernel][lockstep][audit]")
{
    // workers > logical elements: surplus workers own no element and the width clamps to 1.
    int32_t const clampedWidth = runCoverageCase<5u, 16u>();
    CHECK(clampedWidth == 1);

    // Odd extent with several owned elements per worker.
    runCoverageCase<97u, 4u>();

    // Minimal iteration space.
    runCoverageCase<1u, 1u>();
}
