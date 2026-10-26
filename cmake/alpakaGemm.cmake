#
# Copyright 2026 alpaka authors
# SPDX-License-Identifier: MPL-2.0
#
include_guard(GLOBAL)

# Opt-in host-native ISA flags for the GEMM benchmark and GEMM test targets. The
# library itself stays header-only and baseline-portable: it never adds these
# flags to consumers, and no other target is affected. The planner and the
# register-blocked kernel evaluate the compiled SIMD width in the consuming
# translation unit, so a baseline build plans and executes with the baseline
# vector width. Performance numbers therefore require this option.
option(ALPAKA_GEMM_NATIVE_ISA "Compile the GEMM benchmark and test targets with host-native ISA flags" OFF)

# Apply the opt-in flags to the host C++ translation units of one GEMM target.
# CUDA and HIP passes keep their own device flags, so enabling this for a test
# target that is also built for an accelerator does not change the device
# compilation. Unknown compilers are left untouched instead of guessing a flag.
function(alpaka_gemm_apply_native_isa target)
    if(NOT ALPAKA_GEMM_NATIVE_ISA)
        return()
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
        set(isa_flags "-march=native")
    elseif(MSVC)
        set(isa_flags "/arch:AVX2")
    else()
        return()
    endif()
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${isa_flags}>")
endfunction()
