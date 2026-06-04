/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/ThreadSpace.hpp"

#include <type_traits>

namespace alpaka::onAcc::internal
{
    template<typename T_Acc, typename T_WorkGroup>
    struct WorkerSpaceType
    {
        using type = decltype(ThreadSpace{
            std::declval<T_WorkGroup const&>().idx(std::declval<T_Acc const&>()),
            std::declval<T_WorkGroup const&>().size(std::declval<T_Acc const&>())});
    };

    template<typename T_Acc, typename T_WorkGroup>
    requires(requires { std::declval<T_WorkGroup const&>().getThreadSpace(std::declval<T_Acc const&>()); })
    struct WorkerSpaceType<T_Acc, T_WorkGroup>
    {
        using type = decltype(std::declval<T_WorkGroup const&>().getThreadSpace(std::declval<T_Acc const&>()));
    };

    template<typename T_Acc, typename T_WorkGroup>
    using WorkerSpace_t = typename WorkerSpaceType<T_Acc, T_WorkGroup>::type;
} // namespace alpaka::onAcc::internal
