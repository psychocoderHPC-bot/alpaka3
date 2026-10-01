/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace alpaka::onAcc::internal
{
    template<typename T>
    struct IsLockstepVar : std::false_type
    {
    };

    template<typename T>
    constexpr bool isLockstepVar_v = IsLockstepVar<std::remove_cvref_t<T>>::value;
} // namespace alpaka::onAcc::internal
