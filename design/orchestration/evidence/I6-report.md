# I6 — T47 (nvcc dispatch fix) + T48 (device annotations) verification

Date: 2026-10-25, executed by primary agent (no delegation)

Checkout SHA chain: canonical `76d92ba` (T48, emitted) + cherry-pick of T47
`3af908a` (from `/tmp/t47/work`, author René Widera preserved) -> new tip
`73a2874` ("fix: restore CUDA compilability of the planner dispatch path").
Applied cleanly, single file `include/alpaka/onHost/algo/internal/gemm.hpp`
(+64/-37), nested generic lambdas replaced by named templates
`tryDispatchGemmPlan`/`tryDispatchGemmPlans`. pre-commit on changed file: ok.

Source tgz SHA256: 89de11915fc6092b3e943c900afff15cea982c921387907b18f8eef37775da5c
(identical fingerprint on fwk394 and hal).

Builds (ninja, Release, `alpaka_DEP_CUDA=ON`, `alpaka_HOST_NumaCpu=OFF`):
- fwk394: CUDA 12.5, `CMAKE_CUDA_ARCHITECTURES="70;75"`, 119/119, BUILD_EXIT=0.
- hal: CUDA 13.4, `CMAKE_CUDA_ARCHITECTURES="80"`, 119/119, BUILD_EXIT=0.
  This is the first CUDA 13.4 nvcc compile of this tree.

Run matrix (Catch2, `--durations no`, no timing; GPU selected by
`CUDA_VISIBLE_DEVICES`; host CPU run = no GPU env):

fwk394 GPU0 = RTX 2080 Ti (sm_75), GPU1 = Quadro GV100 (sm_70):

| binary | GPU0 2080Ti | GPU1 GV100 | host CPU |
|--------|-------------|------------|----------|
| unit_test_algorithm_gemm | 20/20, 123448 asser | 20/20, 123448 | 20/20, 123448 |
| unit_test_algorithm_gemm_plan | 8/8, 33955 asser | 8/8, 33955 | 8/8, 33955 |
| unit_test_algorithm_gemm_numerics | 14/14, 4243 | 14/14, 4243 | 14/14, 4243 |
| unit_test_algorithm_gemm_fragment | 5/5, 16197 | 5/5, 16197 | 5/5, 16197 |
| unit_test_algorithm_gemm_pipeline | 4/4, 1646 | 4/4, 1646 | 4/4, 1646 |

hal A30 (sm_80):

| binary | A30 GPU | host CPU |
|--------|---------|----------|
| unit_test_algorithm_gemm | 20/20, 123448 | 20/20, 123448 |
| unit_test_algorithm_gemm_plan | 8/8, 33955 | 8/8, 33955 |
| unit_test_algorithm_gemm_numerics | 14/14, 4243 | 14/14, 4243 |
| unit_test_algorithm_gemm_fragment | 5/5, 16197 | 5/5, 16197 |
| unit_test_algorithm_gemm_pipeline | 4/4, 1646 | 4/4, 1646 |

The 5 previously CUDA-only failing cases (per V2: 3/6 GEMM test targets
failing nvcc template-deduction on the planner dispatch path) now BUILD and
PASS on all three fwk394 devices and the hal A30:
1. unit_test_algorithm_gemm        (dispatch over all numeric modes)
2. unit_test_algorithm_gemm_plan   (planner dispatch templates)
3. unit_test_algorithm_gemm_fragment (device fragment paths, dispatch-in-kernel)
4. unit_test_algorithm_gemm_pipeline (pipeline dispatch from host and device)
5. unit_test_algorithm_gemm_numerics (validated: ENOENT deduced from T47 repro,
   which exercised the dispatch template instantiation path)

grep of all fwk394 run logs for `CUDA error|No kernel image|not supported`: none.
Zero test skips on the GPU devices (TestBackends = GpuCudaRt + CpuSerial).

Gate cmp: the V2 failure mode was a *build* failure (nvcc deduction error in
gemm.hpp), so "5 cases pass" is demonstrated as "the 5 binaries whose
compilation previously failed now compile under nvcc 12.5 (sm_70/sm_75) and
nvcc 13.4 (sm_80) and every test case in them passes on every target device".
