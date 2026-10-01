/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** @file
 * @brief Lockstep execution and distributed per-worker local storage for kernels.
 *
 * Lockstep execution lets all workers of a group cooperatively process a logical iteration space while executing the
 * same instructions (SIMT). This header bundles the user-facing lockstep API:
 * - onAcc::makeLockstep() creates an onAcc::LockstepScope from an accelerator, a work group and a compile-time known
 *   logical extent.
 * - onAcc::LockstepScope::concurrent() invokes a functor once per logical index distributed over the workers; the
 *   functor's first parameter is the lockstep index of the processed element(s).
 * - onAcc::LockstepScope::var() creates distributed per-worker local storage which can be passed to concurrent().
 * - onAcc::map() wraps data with an index mapping for shifted or otherwise remapped lockstep access.
 *
 * @see onAcc::LockstepScope
 */

#include "alpaka/onAcc/lockstep/IndexedDataSimdRef.hpp"
#include "alpaka/onAcc/lockstep/IsLockstepVar.hpp"
#include "alpaka/onAcc/lockstep/LockstepIndex.hpp"
#include "alpaka/onAcc/lockstep/LockstepScope.hpp"
#include "alpaka/onAcc/lockstep/LockstepVar.hpp"
#include "alpaka/onAcc/lockstep/MappedData.hpp"
#include "alpaka/onAcc/lockstep/RegisterSimdRef.hpp"
#include "alpaka/onAcc/lockstep/WorkerSpaceType.hpp"
