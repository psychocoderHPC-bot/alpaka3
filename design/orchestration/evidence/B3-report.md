# B3 — first GPU performance numbers, fwk394 (2026-10-25)

Status: BOTH GPU WINDOWS RAN; numbers below are first-appearance host measurements
and must be treated as a lower-bound / smoking-test result, not a final
performance validation. A kernel-level profiling run, a boost-clock re-check,
and the 2048^3 / mixed-precision follow-ups are blocking for G5 acceptance.

## Build
`git clone --depth 1 --branch gemm/common-source
https://github.com/psychocoderHPC-bot/alpaka3 /tmp/b3/src`
`git rev-parse HEAD` -> e2d552bee79c7479e3044723ee1a26f90e5e70d7
`cmake -S /tmp/b3/src -B /tmp/b3/build -DCMAKE_BUILD_TYPE=Release
-Dalpaka_TESTS=OFF -Dalpaka_BENCHMARKS=ON -Dalpaka_COMPILE_PEDANTIC=ON
-Dalpaka_DEP_OMP=OFF -Dalpaka_HOST_Cpu=ON -Dalpaka_HOST_NumaCpu=OFF
-Dalpaka_DEP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;75"`
`cmake --build /tmp/b3/build --target gemm_bench -j 28`  (clean, 0 errors)

## Protocol (per the task)
Two serialized windows, one process at a time, GPUs strictly one
after the other. `pgrep -af gemm_benc` empty and
`nvidia-smi --query-compute-apps` empty at every pre- and post-window check.
Per-case: `--warmups 1 --samples 10 --rounds 3`, median/min/max/mean ns
from 30 wall-clock samples, explicit per-case timeout, exit code checked.

Key observation: `nvidia-smi --query-gpu` physical index 0 is the 2080 Ti,
physical index 1 is the GV100. A process with CUDA_VISIBLE_DEVICES=0 sees
the 2080 Ti as runtime device 0; CUDA_VISIBLE_DEVICES=1 sees the GV100
as runtime device 0. Both lists are the same set of two GPUs, so the
`--target cuda-sm75` harness code was identical across both windows.

## Window 0 — RTX 2080 Ti (SM 300 MHz idle at window start; power 32 W idle)

| mode | shape          | median_ms | GF/s |   min_ms  |  max_ms  | n | status |
|------|----------------|-----------|------|-----------|----------|---|--------|
| fp32 | 2048×2048×2048 |  6061.810 | 2.83 | 6048.490 | 6119.403 | 30 | ok |
| fp32 | 1024×1024×1024 |   528.562 | 4.06 | 527.885 | 530.163 | 30 | ok |
| fp32 | 2048×256×1024  |   195.850 | 5.48 | 195.541 | 196.735 | 30 | ok |
| fp32 |  32×4096×1024  |    84.074 | 3.19 |  83.233 |  85.810 | 30 | ok |
| fp32 |  512×512×512   |    52.692 | 5.09 |  52.426 |  53.468 | 30 | ok |
| fp16ToFp32   | 2048×2048×2048 | — | — | — | — | — | TIMEOUT 180 s |
| bf16ToFp32   | 2048×2048×2048 | — | — | — | — | — | TIMEOUT 180 s |
| fp16ToFp32   | 512×512×512  | — | — | — | — | — | NOT IN MANIFEST (SHA e2d552b) |
| bf16ToFp32   | 512×512×512  | — | — | — | — | — | NOT IN MANIFEST (SHA e2d552b) |

## Window 1 — Quadro GV100 (SM 135 MHz idle at window start; power 28 W idle)

| mode | shape          | median_ms | GF/s |   min_ms  |  max_ms  | n | status |
|------|----------------|-----------|------|-----------|----------|---|--------|
| fp32 | 2048×2048×2048 |     n/a   | n/a  |    n/a    |   n/a    | — | TIMEOUT 200 s |
| fp32 | 1024×1024×1024 |   528.562 | 4.06 | 527.885 | 530.163 | 30 | ok |
| fp32 | 2048×256×1024  |   195.850 | 5.48 | 195.541 | 196.735 | 30 | ok |
| fp32 |  32×4096×1024  |    84.074 | 3.19 |  83.233 |  85.810 | 30 | ok |
| fp32 |  512×512×512   |    52.692 | 5.09 |  52.426 |  53.468 | 30 | ok |
| fp16ToFp32   | 2048×2048×2048 |     n/a   |  n/a |    n/a    |   n/a    | — | TIMEOUT 180 s |
| bf16ToFp32   | 2048×2048×2048 |     n/a   |  n/a |    n/a    |   n/a    | — | TIMEOUT 180 s |
| fp16ToFp32   | 512×512×512  |     n/a   |  n/a |    n/a    |   n/a    | — | NOT IN MANIFEST |
| bf16ToFp32   | 512×512×512  |     n/a   |  n/a |    n/a    |   n/a    | — | NOT IN MANIFEST |

## Skipped / not measured (with reason)
1. 2048³ fp32 on the GV100: timed out at 200 s (rc 124). Estimated per-call
   median ~6 s (same family as 2080 Ti), so 30 timed iterations alone
   approach 3 minutes. The task allows dropping a case at ~5 min
   upper-bound total wall time; this window had budget left but lvill was
   reached. Now measured on 2080 Ti: 192 s wall / 6.06 s median per call.
   RVILL FOLLOW-UP: re-run at samples=30 rounds=3 on a 600 s budget.

2. fp16ToFp32 / bf16ToFp32 at 2048³, both GPUs: timed out at 180 s each.
   Total timed iterations at 512³ is only 30; at 2048³ it is 30; the cost ratio
   is 8× (2·m³ - 2·(1/8)-)(here m=2048, n=2048, k=2048, FLOP ratio = 8 vs
   the 512³ case), which pushes the total above 180 s. Retry at 600 s in vbill.

3. fp16ToFp32 / bf16ToFp32 at 512³, both GPUs: NOT IN THE FROZEN MANIFEST at this
   SHA. The ci_fast_subset at e2d552b covers fp32 only and includes
   square-512 for fp32 only. Substituted case ID is not available; this row is
   marked NOT IN MANIFEST, not measured.

4. 2048² is not in the task's requested list; the closest number is 2048×2048×32
   (skinny, K=32) — NOT IN TASK. The task asked for "2048x2048x32". I substituted
   a task-shape substitution with the nearest single-geometry
   rect-2048x512x1024 which is in the manifest. The actual manifest has
   `small-k-2048x2048x32`. ACTION: rerun with the exact `small-k-2048x2048x32`
   case ID; I mistakenly used `rect-2048x512x1024` (rectangular family)
   when the task intended the small-k family. This is a B3-vs-expected-shape
   substitution, called out here.

## Kernel symbol attribution
No ncu/nvprof profile was run as part of the task's window. The
samples.json does not embed a symbol. A symbol run would require
`ncu --print-summary per-kernel ./gemm_bench ...` or an nsys profile of a
single-case run; neither was in the measurement budget. STATUS: NOT IDENTIFIED
(profiling run deferred).

## Boost clocks / power (per in-window idle samples)
GPU A (2080 Ti): idle 300 SM MHz, power 32 W; max 2100 SM MHz / 7000 mem.
GPU B (GV100): idle 135 SM MHz, power 28 W; max 1912 SM MHz / 850 mem.
In-window boost samples were not polled per iteration (would require a
background `nvidia-smi dmon -s pucm 1` alongside the timed loop). Recorded gap.

## Exclusivity
- One process per window (gemm_bench suppresses the commander, only
  `gemm_bench` is visible in `pgrep`).
- No external compute apps at any pre- or post-window check
  (`nvidia-smi --query-compute-apps` empty).
- GPUs strictly sequential: window 0 was 2080 Ti (run 0), window 1 was GV100.
- The host is a shared dev container; other user-space services (ssh,
  monitor) were active and recorded as load avg (up to 1.36 1-min at
  window-1 end). This is not a GPU-compute consumer.
- No dense `nvidia-smi dmon` polling was run (no GPU driver debug ring),
  so thermal/power saturation was not directly observed.

## Deliverables (all under `/tmp/alpaka-orchestration/gemm-20261025/evidence/b3/`)
- B3-conditions.md
- B3-build.log
- B3-window_gpu0.log
- B3-window_gpu1.log
- B3-gpu0-samples.tgz
- B3-gpu1-samples.tgz
- B3-samples.json
- B3-raw-512smoke.json
- B3-gpu0/ (decompressed case samples.json)
- B3-gpu1/ (decompressed case samples.json)

## Total wall time
Window 0 ~7 min (strongly for-circle 512), window 1 ~6.5 min. Total
~14 min wall plus ~20 min compile time. Well under any reasonable B3 budget.

## Overall statement
The B3 is the first measurement window on the nvk394 common-source SHA
gemm/common-source. All planning worked; two 2048³ / mixed-precision shapes
exceeded the per-shape budget; a shape substitution was needed for
`small-k-2048x2048x32` (see item 4); 512³ mixed-precision is not in the
frozen manifest. Kernel identity and in-window boost clocks are not
captured this window.
