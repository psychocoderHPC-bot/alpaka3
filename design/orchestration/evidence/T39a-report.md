# T39a: GEMM test suite on real GPUs (correctness only)

Source: canonical tip d6ea524015b5852dec968375d400b2cf9d37eb91 (branch gemm/common-source), tarball transfer.
sha256 /tmp/t39a-src.tar.gz: 8661940b259f2c54479106cd05ab5c2c3c93843fc2e105a770d3e897895f85a1
Key-file sha256 identical on local and on hal + fwk394 (CMakeLists.txt, gemm.cpp, gemm_fragment.cpp, gemm_numerics.cpp).
Transfer method: computeMCP_file_upload, single tarball, no retry needed.

| target | executor | device | build | gemm | fragment | numerics | lockstep | audit | verdict |
|--------|----------|--------|-------|------|----------|----------|----------|-------|---------|
| hal    | GpuCuda  | A30 (sm_80) | CUDA 13.4 | 30 cases / 63012 assertions PASS | build fail (project defect) | 14/4243 PASS | 15/4041 PASS | 3/605 PASS | PARTIAL |
| hal    | GpuHip   | RX 7900 XT (gfx1100) | HIP/AMDclang 7.2.0 | build fail (toolchain) | 6/16198 PASS | 14/4243 PASS | 15/4041 PASS | 3/605 PASS | PARTIAL |
| hal    | OneApiGpu | Arc A770 (dg2) (+ CpuSerial on EPYC) | SYCL 2026.2 icpx | build fail (toolchain) | build fail (toolchain) | 14/4243 PASS | 15/4041 PASS | 3/605 PASS | PARTIAL |
| fwk394 | GpuCuda  | 2080 Ti sm_75 + Quadro GV100 sm_70 | CUDA 12.5 | build fail (toolchain) | build fail (toolchain) | build fail | build fail | build fail | FAIL (build) |

Worst scaled errors (unit_test_algorithm_gemm on hal / A30 CUDA sm_80):
- fp64 2.31738e-16
- fp32 2.78752e-07
- fp16ToFp32 adv 1.59862e-07, base 1.31429e-07
- bf16ToFp32 adv 9.04462e-08, base 7.91276e-08
No other platform prints these (gemm.cpp does not build there).
gemm_fragment / gemm_numerics / lockstep do not print a single "worst scaled error" number per run
(only per-case INFO lines), so no worst-scale value to report for them.

Classified failures:
1) unit_test_algorithm_gemm_fragment on hal, CUDA sm_80, nvcc 13.4 -- project defect:
   - include/alpaka/onAcc/fragment.hpp:318, :375, :398 nvcc error #20011-D:
     "calling a __host__ function from a __host__ __device__ function is not allowed".
     Touched via concurrent(lambda), loadFrom(HostView<T> view, ...), storeTo(HostView<T> view, lambda, ...).
   - AMD clang / HIP compiles and runs the same test (see hal HIP).
   - Fix direction: make concurrent/loadFrom/storeTo ALPAKA_FN_ACC for CUDA, or add
     HostView / lambda overloads qualified __device__; do not mark the host path usable.

2) unit_test_algorithm_gemm build under HIP/AMD 7.2.0 AND oneAPI/icpx 2026.2 -- toolchain limitation:
   - validation/numerics/reference_gemm.hpp:326 static_assert hasExtendedPrecisionReference failed:
     "gemm_reference_extended needs a long double wider than double".
     amdclang default: long double == double bits (53); icpx 2026.2 build reports digits==53 as well.
   - Triggered by test/unit/algorithm/gemm.cpp:724 (unit_test_algorithm_gemm "gemm FP64 mode")
     that calls gemm_reference_extended regardless of the toolchain capability gate.
   - Builds / runs fine on CUDA 13.4 host gcc where long double is 80-bit extended.
   - Note: gemm_numerics (no long double reference call) builds and passes on all
     three platforms that compile gemm.py functions at all.

3) unit_test_algorithm_gemm build on fwk394, CUDA 12.5 (nvcc 12.5), CMAKE_CUDA_ARCHITECTURES="70;75"
   -- toolchain limitation:
   - include/alpaka/onHost/algo/gemm.hpp:34:88: error: expected identifier before '>' token
   - nvcc 12.5 cannot parse the C++20 class-type deduced wrapper on the
     alpaka::concepts::Executor auto const exec parameter in gemm.hpp:36-37.
   - nvcc 13.4 (same host, different arch) parses it fine.

4) unit_test_algorithm_gemm_fragment build under oneAPI/icpx 2026.2 -- toolchain limitation:
   - test/unit/algorithm/gemm_fragment.cpp:383, :384, :512, :518: error: expected unqualified-id
   - icpx rejects std::printf in this TU; the code compiles on g++ and clang.

Remaining unvalidated:
- unit_test_algorithm_gemm on HIP gfx1100 (all four modes incl. fp64).
- unit_test_algorithm_gemm on oneAPI Arc A770 (all four modes incl. fp64).
- unit_test_algorithm_gemm on fwk394 multi-arch (CUDA 12.5 cannot build the gemm API).
- unit_test_algorithm_gemm_fragment on CUDA (nvcc) and on oneAPI/bits.
- device-name printout inside tests (none produced); device choice evidenced
  via nvidia-smi cap + arch, sycl-ls, and build-time architecture flags.
- The fwk394 -Dalpaka_HOST_NumaCpu=OFF switch (only exercised once at configure;
  never reached a test build).
- LOCKSTEP on CUDA sm_75 / sm_70 on the same target (blocked by build failure #3).
- fp32 / fp64 / fp16 / bf16 verification on any AMD or Intel GPU.
