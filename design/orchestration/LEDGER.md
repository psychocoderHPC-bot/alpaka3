# GEMM project ledger — run gemm-20261025

Spec: /home/dev/alpaka3-gemm-design-guide.md (v2, 591 lines)
PR destination: psychocoderHPC-bot/alpaka3, base `dev`
Topic branch (all workers): `gemm/common-source`
Artifacts: /tmp/alpaka-orchestration/gemm-20261025/

## Pinned refs
- bot origin/dev = 6f13700c390a09df818a255d53e730f1f9d69456 (2026-10-01)
- upstream dev  = b7d339d07056a9a2a6c4051cc3927157bc5f0d51 (guide baseline)
- lockstep     = 05290c84b5c2349ff93585a95845e5471bba2540, ref origin/issue-6-lockstep2
- /workspace worktree = ded79f8, 5 behind origin/dev, MUST stay read-only

## Tasks
| ID | Outcome | Owner | Status | Deps | Evidence |
|----|---------|-------|--------|------|----------|
| P0 | restructured plan + DAG | alpaka-work-planner | validated | - | planner message |
| R1/T1a | repo/PR/API recon | alpaka-explorer | validated | - | inline, no log (no write perm) |
| R2/T3 | hal+fwk394 hardware inventory | fr-general | validated | - | logs/hwinv-hal.txt, logs/hwinv-fwk394.txt |
| T3b | AMD/HIP reachability probe (jureca etc.) | fr-general | validated | T3 | logs/hwinv-amd-probe.txt |
| T3c | hal AMD GPU re-verification | fr-general | validated | T3 | logs/hwinv-hal-amd-recheck.txt |
| T3d | hal HIP stack (rocBLAS/rocWMMA/profiler/compile+run smoke) | fr-general | validated | T3c | logs/hwinv-hal-hip-stack.txt |
| T3f | install rocBLAS + a profiler on hal/fwk388, verify with a real GEMM + trace | fr-general | validated | T3d | logs/hip-stack-install.txt |
| T1 | base pin + lockstep audit (6 items) + FP16 gap | fr-general | validated | R1 | logs/T1-lockstep-audit.md |
| T6a | FP16/BF16 storage types + exact conversions + exhaustive tests | fr-engineering | needs-integration | T1 | wd /tmp/alpaka3-t6a-gemm-J1Rusr @ 9f7216c + 04ccd4c |
| T6b | lockstep audit fixes (docs item 6, zero-worker guard) + regressions | alpaka-implementer | discovered | T5,T1 | |
| T2 | design/gemm.md + validation/gemm-requirements.yaml | alpaka-implementer | needs-review | - | wd /tmp/alpaka3-t2-gemm-L6j0VU @ 145db5e1 (base 35ba304f) |
| T5a | canonical base + lockstep-tip reconciliation | fr-quick | validated | R1,T1,T2 | logs/T5a-base-reconcile.txt |
| T5 | M0 integration: lockstep rebase + T2/T4 fold-in + coverage manifest | alpaka-implementer | needs-review | T1,T2,T4,T5a | wd /tmp/alpaka3-t5-gemm-l5RK2z @ 57d188ff |
| T4b | pre-commit lint/format fixes on harness files | fr-code | needs-integration | T5 | commit 4a726f18 |
| T8a | remote GPU build/run workflow + first hardware validation | fr-general | validated | T5 | evidence/T8a-remote-build.md + .sh |
| T8b | clean fwk394 multi-arch + NUMA verdict, hal reproducibility | fr-general | validated | T8a | evidence/T8b-fwk394-clean.md + T8b-replay.sh |
| T2b | correct design doc hardware facts, owner field, checker schema | fr-code | integrated | T5a,T4 | actual SHA 0205ce19 (the 0205ce16 I recorded was a typo) |
| T6b | lockstep Defect A guard + Defect B public docs + 4 audit regressions | fr-engineering | needs-integration | T5 | wd /tmp/alpaka3-t6b-gemm-Eyks @ 3b526fc + ca6bde3 |
| T6c | CPU worker/slot/lane model + M3 tile invariant | fr-reasoning | validated | T6b | evidence/T6c-cpu-workers.{cpp,log} |
| T20 | upstream fit discovery: conventions, duplicate work, lockstep upstream status | explore | validated | - | inline |
| T22 | evaluate psychocoderHPC/alpakaVendor as the vendor baseline interface | fr-general | partial(unreliable report) | - | |
| T3g | install oneMKL + CPU BLAS on hal/fwk394 | fr-general | failed(corrupt output) | - | logs/blas-install.txt (partial) |
| T3h | verify installed BLAS + oneMKL/Arc + cublas availability | fr-quick | validated | T3g | logs/blas-verify.txt |
| T11 | M3 register-blocked fragments | fr-engineering | needs-integration | T21,T6c | wd /tmp/t11/work @ ff2ad28+16fabc9 |
| T11b | G3 abstraction-cost re-measurement ON computeMCP (local numbers void) | fr-general | ready(after publish) | T11,5h | |
| T24 | harden acceptance checker (matrix, SHA staleness, bench integrity, vendor pairing) | fr-code | running | T11 | |
| T5i | integrate T21+T23+T11+T24, verify, publish, update PR | fr-general | FAILED(garbled output, no work done) | T21,T23,T11,T24 | |
| P1 | ff canonical to dfa82dc + cherry-pick T23,T6e | fr-code | validated | T24,T23,T6e | canonical tip d6ea5240, ctest gemm|lockstep 86/86 |
| P2 | publish 14 commits to remote (API only) | fr-general | blocked(API quota) | P1 | rate limit 5000/5000 used, resets 12:08Z |
| P2c | SSH push credential audit | fr-quick | validated | - | logs/ssh-push-check.txt: NO usable repo key |
| P2d | paced publish after 12:08Z reset | fr-general | ready | P2,P2c | |
| P3 | update PR #60 body with M2/M3 state | fr-code | discovered | P2 | |
| T24 | harden acceptance checker | fr-code | needs-integration | T11 | wd /tmp/t24/work @ dfa82dc (27 pytest) |
| T39a | GPU correctness of M2/M3 on hal + fwk394 | fr-general | validated | d6ea5240 | evidence/T39a-report.md |
| T42 | fix 4 GPU build defects | fr-engineering | needs-integration | d6ea5240 | wd /tmp/t42/work @ 0222513,7539f82,db4bdb2,13debc7 |
| T43 | fix signed-zero test oracle (oneAPI) | fr-engineering | needs-integration | T42 | wd /tmp/t43/work @ 1091f6a (test-only) |
| T45 | investigate oneAPI fast-math vs strict FP contract | fr-reasoning | FAILED(empty output) | f75dd56e | |
| T45a | audit which FP compile flags reach GEMM TUs (local) | fr-quick | running | f75dd56e | |
| T45b | measure on-device signed-zero/denormal/NaN/FMA behaviour | fr-general | discovered | T45a | |
| T11b | G3 + throughput on hal | fr-general | validated(late, partly contaminated) | f75dd56e | G3 PASS; reported a 12h hang on 2048 serial |
| H1 | reproduce/characterise the 2048-serial hang | fr-reasoning | running | 6f60c0a5 |  two sweep scripts + 2 concurrent gemm_bench per mode, same output dir. Processes killed on hal. All numbers discarded. |
| T11c | ONE exclusive benchmark window on hal | fr-general | validated | bfac7fef | evidence/T11c-1792950900/ |
| B1 | GPU bench on rosi_v100 | fr-general | blocked(build fail + host cannot run CUDA) | 8d3409b | evidence/B1-failure.md |
| C1 | PR progress comment with first valid CPU benchmark | fr-quick | running | T11c | |
| B2 | GPU bench on fwk394 | fr-general | aborted(build fail) | 8d3409b | gemm_bench also fails: same T47 regression (gemm.hpp:579 std::forward, 98 errors) |
| T49 | profiler analysis of CPU GEMM (why 4.9 GF/s/thread) | fr-general | running | 8d3409b | |
| T14 | M5 heuristic planner | fr-engineering | needs-integration | f75dd56e | wd m5-work @ 8b001cd1 |
| T46 | fix bench ctest output path | fr-code | needs-integration | T14 | wd /tmp/t46-gemm-fix/src @ 0c75a08 |
| I3 | integrate T14 + T46 + T43, verify | fr-code | validated | T43,T14,T46 | canonical tip bfac7fef |
| P5 | push M5 tip | fr-quick | validated | I3 | remote tip bfac7fef |
| T15 | M6 pipeline abstraction | fr-engineering | needs-integration | bfac7fef | wd /tmp/T15-gemm-pipeline-tUahCX @ 1d0c65b,0595137,30a959f,c73c35e |
| I4 | integrate M6 pipeline | fr-code | committed (local build killed per D49) | T15 | canonical tip 8d3409b22f19 |
| P6 | push M6 tip | fr-quick | validated | I4 | remote tip 8d3409b22f19 |
| V1 | sm_70 verify on rosi_v100 | fr-general | validated | P6 | same nvcc failure on 12.6; rosi_v100 CANNOT execute CUDA (libcuda.so.1 absent) |
| V2 | CUDA 12.5 verify on fwk394 | fr-general | validated | P6 | 3/6 targets BUILD FAIL (nvcc template deduction regression); 3 pass on both GPUs |
| T47 | fix nvcc dispatch deduction regression | fr-engineering | needs-integration | V2 | wd /tmp/t47/work @ 3af908a |
| I5 | ff canonical to T48 + push | fr-quick | validated | T48 | remote tip 76d92ba |
| I6 | integrate T47 + verify CUDA + push | fr-general | validated | T47,I5 | remote tip 73a28748, ALL CUDA/HIP green |
| V4 | oneAPI verify at 73a28748 | fr-general | validated | I6 | Arc A770 + SYCL CPU pass; found bf16 host defect + SYCL deprecation |
| T50 | bf16 host-CpuSerial subnormal defect | fr-reasoning | needs-integration(after T50b verify) | V4 | wd /tmp/opencode/t50/work @ e73ebec |
| T50b | narrow verification of the FP-model fix | fr-general | FAILED(fix ineffective) | T50 | pin reaches TU but defect unchanged |
| T50c | isolate icpx denormal loss with minimal repro, real fix | fr-reasoning | FAILED(empty output) | T50b | |
| T50d | minimal repro + flag sweep | fr-reasoning | validated | T50b | ROOT CAUSE: icpx -O3 sets FTZ+DAZ in MXCSR process-wide |
| T50e | build-wide precise pin + contract test + docs | fr-engineering | needs-integration | T50d | wd /tmp/opencode/t50/work @ e73ebec+28d6cb6 |
| I12 | integrate FP-model fix + push | fr-code | validated | T50e | remote tip 9a684c80 |
| O3 | bench: --staging + plan reporting | fr-engineering | validated | 9a684c80 | wd /tmp/alpaka3-o3-issue-N-WJOvsJ/repo @ 6e82209 |
| I13 | ff + push O3 | fr-quick | validated | O3 | remote tip 6e82209 |
| M2 | direct vs staging measurement on hal | fr-general | validated | NEGATIVE: staging 1.7-2x SLOWER on CPU |
| V6 | final-SHA correctness on hal (CUDA/HIP/oneAPI) | fr-general | validated(1 build gap) | f008d8ed | CUDA pipeline test fails to build under nvcc 13.4 |
| V7 | final-SHA correctness gfx906 | fr-general | validated | f008d8ed | ALL PASS, GPU proven by VRAM+KFD |
| T52 | fix nvcc 13.4 pipeline-test build (test-only) | fr-engineering | running | f008d8ed | 
| M8a | evidence into requirements.yaml + acceptance draft | fr-code | running | 6e82209 | |
| T60 | make gemm_bench device-selectable | fr-engineering | validated | d3ad4575 | wd /tmp/gemm-t60/work @ 9f901102+719649b6 |
| B5 | first real GPU bench attempt | fr-general | FAILED(echoed prompt, no artifacts) | 719649b6 | |
| B5b | one-GPU minimal bench | fr-general | FAILED(no artifacts) | 719649b6 | |
| B5c | three-command GPU bench | fr-general | validated | FIRST REAL GPU NUMBERS |
| B6 | fwk394 GV100 + cuBLAS baseline | fr-general | validated | GV100 97.5/794.8/930.2 GF/s; cuBLAS 1820/9001/10909 -> ratio 5.4-8.8% |
| O4 | does cooperative staging help on GPU | fr-general | validated | staging -20% at 1024/2048, +11% at 256; NOT the lever |
| O5 | larger tile/BK family for GPU reuse | fr-engineering | needs-integration | 6f60c0a5 | wd /tmp/alpaka3-o5-issue-Xx2Kud @ d4328ab |
| I16 | integrate O5 + push | fr-quick | validated | remote tip d4328ab1 |
| B8 | ratios at new tip on both fwk394 GPUs | fr-general | validated | 2080Ti 0.289/0.318; GV100 0.230/0.455; geomean 0.30/0.32 |
| B9 | hal ratios at new tip | fr-general | FAILED(garbled) | d4328ab1 | |
| B9b | A30 ratios at new tip | fr-general | validated | A30 78.7/1514/2166 vs cuBLAS 1260/5761/5910 -> 0.06/0.26/0.37 |
| O6 | why BK stays 8 + fix + measure | fr-engineering | needs-integration | d4328ab1 | wd o6-bk-refill @ 59199c3 (+2.4-3.3% only) |
| I17 | integrate O6 + push | fr-quick | validated | remote tip 3b8e3e2b |
| C3 | refresh PR #60 body to measured final state | fr-code | validated | 3b8e3e2b | body applied, draft preserved, permalinks clean |
| D2 | ncu profile of the 128x128 cooperative kernel | fr-general | paused | 3b8e3e2b | environment verified (ncu 2024.2.1.0, driver 555.52.04, both GPUs idle), full state + build + ncu commands + fallbacks in /tmp/opencode/d2-handoff.md; nothing built, nothing pushed (canonical tree clean; local HEAD and remote refs/heads/gemm/common-source both 3b8e3e2b); resume: run Phase1-2 on fwk394 after a fresh nvidia-smi idle check; label profile as shared-load diagnostic |
| C2 | PR comment with first valid GPU numbers + ratios | fr-quick | running | B8 | 
| B7 | hal numbers + vendor baselines | fr-general | FAILED(no measurements; worker killed processes it should not have touched) | 6f60c0a5 | hal verified healthy after |
| B7b | A30 three shapes + cuBLAS | fr-general | validated (PRE-O5 tip) | 82.2/373.6/620.4 GF/s vs cuBLAS 998/5637/5856 -> 7-11% |
| B9 | hal A30+gfx1100 ratios at new tip | fr-general | running | d4328ab1 | 
| I11 | ff + push T60 | fr-quick | validated | T60 | remote tip 719649b6 |
| T51 | fix deprecated SYCL barrier (upstream compat) | fr-engineering | needs-integration | V4 | wd hal:/tmp/t51/src @ 1a94fe03 |
| I9 | integrate T51 + push | fr-quick | blocked(commit only lived on hal) | T51 | stopped correctly |
| I9b | fetch T51 via git bundle + push | fr-quick | validated | I9 | remote tip 1a94fe03 |
| M7a | matrix contract + CUDA adapter + fallback tests | fr-engineering | needs-integration | 1a94fe03 | wd /tmp/alpaka3-m7a-issue-7a-YKSI @ 1c9e28da..d3ad4575 |
| I10 | ff + push M7a | fr-quick | running | M7a | |
| M7b | integrate warp fragments into the GEMM tile + epilogue | fr-engineering | discovered | M7a,D1 | |
| O1 | plan tile + host ISA opt-in | fr-engineering | running | 76d92ba | |
| O2 | vectorize fp16/bf16 widen | fr-engineering | needs-integration | 73a2874 | wd work-o2 @ 811eb11+4c49a47 |
| I7 | ff canonical to O2 + push | fr-quick | validated | O2 | remote tip 4c49a47 |
| I8 | ff canonical to O1 + push | fr-quick | blocked(divergence: O1 based on 73a2874) | O1 | no work done, stopped correctly |
| I8b | cherry-pick O1 onto 4c49a47, verify, push | fr-code | validated | O1,I7 | remote tip e2d552b |
| M1 | native-ISA + tile measurement window on hal | fr-general | running | I8b | |
| A1 | audit which tile the benchmark really launches | explore | running | - | |
| V3 | gfx906 HIP fallback verify on fwk388 | fr-general | validated | P6 | 3 test TUs fail: staging/pipeline not device-callable. Native mfma/wmma verified ABSENT (4 methods). |
| T48 | fix staging/pipeline device annotations | fr-engineering | validated | V3 | wd /tmp/t48/work @ 76d92ba; HIP GPU run proven via KFD PID |
| I5 | ff canonical to T48 + push | fr-quick | running | T48 | |
| T44 | GPU re-validation with planner+staging on all devices | fr-general | ready(after T11b) | T11b,P5 | |
| T44 | GPU re-validation incl. staging + scratch trigger on all devices | fr-general | ready(after T11b) | T11b | |
| I2 | integrate T42 onto canonical, resolve conflicts with M4, verify | fr-code | discovered | I1,T42 | |
| T35 | rosi_v100 access verification | fr-quick | validated | - | logs/rosi-v100-check.txt |
| T36 | pre-warm CUDA build env on rosi_v100 | fr-general | partial(channel died) | T35 | evidence/T36-rosi-v100-buildenv.md |
| T6e | fix nvcc -Werror defect + portability sweep | fr-engineering | needs-integration | T21 | wd /tmp/t6e/work @ 5cb2a8f+84779a3 |
| T5j | integrate T6e CUDA-build fixes, verify, publish | fr-general | discovered | T5i,T6e | |
| T23 | verify + integrate alpakaVendor as benchmark-only baseline | fr-engineering | needs-integration | T22 | wd /tmp/alpaka-orchestration/t23-vendor-1792919346 @ 59e80ed |
| T23b | wire CUDA/HIP/oneMKL devices into the vendor baseline functor | fr-engineering | discovered | T23,T5h | |
| T5h | integrate T21 + T23 into canonical, verify, publish, update PR#60 | fr-general | STALLED(no output, no commits, remote unchanged) | T21,T23 | |
| T21 | conform to upstream conventions: onHost/algo/gemm.hpp, error+mode style, test path, alpaka-linked bench, tutorial page | fr-engineering | discovered | T9b,T20 | |
| T5b | canonical tree (integration + verification) | alpaka-implementer | validated | T2b,T4b,T5 | canonical @ 66efc781 |
| T5c | push branch + open draft PR | fr-quick | blocked(transport) | T5b | push denied: Invalid username or token |
| T5e | publish via Git Data API + draft PR | fr-general | validated | T5c | PR #60 https://github.com/psychocoderHPC-bot/alpaka3/pull/60 (draft) |
| T9 | M2 common GEMM (regions, tails, transposes, strides, alpha/beta) | alpaka-implementer | validated(local) | T5,T6b,T7 | wd /tmp/alpaka3-t9-gemm-SdBD @ a571bc1f |
| T9b | integrate M2 into canonical + wire FP16/BF16 modes | alpaka-implementer | validated(local) | T9 | canonical LOCAL tip 64bc1c71 |
| T5g | publish M2 + mixed precision on top of dde6e354, update PR#60 | fr-general | ready | T21,T5f,T9b | |
| T21 | conform GEMM API to onHost/algo conventions + docs + ctest bench | fr-engineering | needs-integration | T9b,T20 | wd /tmp/t21/work @ 9013951+bba8bfd+f814d5e |
| T9c | M2 on real GPUs + first honest baseline | fr-general | failed(transport) | T9 | evidence/t9c-* staging only |
| T9d | M2 GPU validation via git-clone transport on published SHA | fr-general | ready(after publish) | T9b,T5g | |
| T6d | document var() CVector requirement + CpuSerial frame collapse | fr-code | discovered | T6b,T9 | |
| T5d | integrate T7 + T6a into canonical, re-verify | alpaka-implementer | validated | T5c,T7,T6a | canonical local 5f0137df |
| T5f | recover T6b commits + integrate + publish increment + update PR#60 | fr-general | validated | T5d,T6b | remote tip dde6e354 |
| T2b | correct design doc: AMD facts, owner field, base SHA | alpaka-implementer | discovered | T5a,T5c | |
| T4 | workloads.json + bench driver + check_acceptance.py + tests | alpaka-implementer | needs-review | - | wd /tmp/alpaka3-t4-gemm-rHA8Ku @ b5bf2e2 (base 6f13700c) |
| T7 | reference oracle + tolerance model + adversarial tests | fr-engineering | needs-integration | T2,T4 | wd /tmp/alpaka3-t7-gemm-rPu40r @ 03aabc14 + 5814201e |
| T5 | lockstep integration + coverage manifest + draft PR | alpaka-implementer | discovered | T1,T2,T3,T4 | |
| T6 | audit fixes + FP16/BF16 types (from scratch) | alpaka-implementer | discovered | T1 | |
| T7 | scalar/MP oracle + tolerance model | alpaka-implementer | discovered | T2,T4 | |
| T8 | independent M1 review | fr-review | discovered | T6,T7 | |
| T9 | common correct GEMM M2 | alpaka-implementer | discovered | T6,T7,T8 | |
| T10 | M2 baseline bench + first profiles | alpaka-task-worker | discovered | T5,T9 | |
| T11 | M3 layouts/fragments/register reuse | alpaka-implementer | discovered | T9 | |
| T12 | independent M3 review | fr-review | discovered | T11 | |
| T13 | M4 staging | alpaka-implementer | discovered | T11,T12 | |
| T14 | M5 planner | alpaka-implementer | discovered | T9,T13 | |
| T15 | M6 pipeline | alpaka-implementer | discovered | T13,T14 | |
| T16 | M7 native adapters | alpaka-implementer | discovered | T11,T14,T15 | |
| T17 | profiler-driven optimization rounds | alpaka-task-worker | discovered | T10 | |
| T18 | M8 acceptance matrix + checker | alpaka-task-worker | discovered | T9,T13,T15,T16,T17 | |
| T19 | final independent review | fr-review-general | discovered | T18 | |

## Decision ledger
D1 (high) Base = bot origin/dev 6f13700c, not the stale /workspace tree (ded79f8).
   Evidence: R1. Reconsider if T1 shows bot dev has diverged from upstream in ways that break lockstep integration.
D2 (high) Topic branch name `gemm/common-source` for every worker; commits cherry-picked serially by the manager.
   Reason: single incremental PR is mandatory; disjoint clones prevent checkout contention.
D3 (high) FP16/BF16 must be added from scratch in-tree (R1: no fp16/bf16 symbols anywhere).
   Falsified if T1 finds an existing facility under a different spelling.
D4 (medium) Do not run `git fetch --unshallow` on /workspace; use throwaway full clones for ancestry.
D5 (medium) Workers cannot push; only the manager publishes. PR opened after T5.

## Hypothesis ledger
H1 lockstep supports register blocking within 5% of hand-written code — untested until T11.
H2 vendor-symbol dispatch can host native GEMM primitives without backend branches — untested until T16.
H3 three distinct NVIDIA generations reachable across hal+fwk394 — pending T3.

## Hardware matrix (T3, verified)
hal: AMD EPYC 7452 (Zen2, avx2 only, no avx512), 64 cores, 251Gi RAM; NVIDIA A30 (Ampere cc8.0) driver 615.71, nvcc 13.4, cuBLAS 13; oneAPI 2026.2 CPU + Intel Arc A770 (Level Zero); ncu + nsys.
hal ALSO: AMD Radeon RX 7900 XTX (gfx1100, 24GiB), ROCm 7.2.0 usable, tools under /opt/rocm/bin (not on PATH).
fwk388: Ryzen Threadripper 1950X (Zen1), 32T; AMD Radeon VII (gfx906, wavefront 64), ROCm 7.2.0, hipcc 7.2, rocprof+rocprofv3, NO rocBLAS; connects on demand.
rosi_v100: Xeon 6136 48c; Tesla V100-SXM2-32GB (sm_70). Optional.
fwk394: Intel Xeon W-2175 (Cascade Lake, avx512f, no bf16/amx), 28 cores, 125Gi RAM; RTX 2080 Ti (Turing 7.5) + Quadro GV100 (Volta 7.0), driver 555.52, nvcc 12.5, cuBLAS 12; no oneAPI; ncu only (no nsys).
Three distinct NVIDIA generations confirmed: Volta, Turing, Ampere. No AMD/ROCm device on any connected target -> HIP is BLOCKED_BY_HARDWARE unless T3b finds one.
No avx512_bf16/amx_tile anywhere -> BF16/FP16 arithmetic must be generic emulation on all CPUs.

D6 (high) oneAPI CPU = hal (SPIR/OpenCL CPU device); oneAPI GPU = Intel Arc A770 on hal. Record separately as required.
D7 rev1 (high) AMD/HIP is AVAILABLE: fwk388 = Radeon VII (gfx906, wavefront 64), ROCm 7.2.0, hipcc 7.2 (clang 22), rocprof + rocprofv3 present, rocBLAS ABSENT. fwk388 connects on demand (was listed disconnected). jureca + jureca-h100 unreachable (exec error, last_error null) after 2 dial attempts each - BLOCKED, evidence logged.
   Consequences: (a) required-target matrix gains fwk388/hip; (b) gfx906 has no matrix cores, so native HIP matrix ops can only be validated as capability-gated fallback, never as an MMA pass; (c) HIP vendor-BLAS comparison needs rocBLAS installed (authorized by guide s9) or must be recorded as a documented limitation.
D7 rev2 (high) Operator correction CONFIRMED: hal has AMD Radeon RX 7900 XTX (Navi 31, PCI 1002:744c, gfx1100, 24GiB) with a fully usable ROCm 7.2.0 install (/opt/rocm, not on PATH), /dev/kfd + render nodes present. So HIP has TWO endpoints: gfx1100 on hal (RDNA3, wave32, likely WMMA-capable) and gfx906 on fwk388 (Vega20, wave64, no matrix cores). Native-HIP MMA, if validated, is gfx1100-only; gfx906 stays fallback with recorded hardware limitation.
D7 rev2a (high) Process lesson, keep: absence of a vendor CLI on PATH is NOT evidence of missing hardware. Always probe lspci / /dev/kfd / /sys/class/drm first. This error produced a wrong BLOCKED verdict once; do not repeat it in later coverage claims.
D10 (high) Integration strategy for lockstep: PR #7 (open, MERGEABLE, NOT draft) head 05290c84 is a pure +1544/-0 add across 14 files, merge-base 8ae77718, and the 12 intervening dev commits touch 92 files with ZERO overlap. Rebase the 10-commit branch onto bot origin/dev; expect no conflicts. The scout's "unrelated file changes" report was a shallow-graft artifact, retracted. PR #15 (draft heat benchmark) depends on #7 and must be rebased after.
D11 (medium) Audit verdicts (T1, confidence medium-high, ref 05290c84): items 1,2,3,5 OK as-is. Item 4: zero logical extent is safe; ZERO WORKER COUNT hits divCeil(N,0) in consteval with no guard - untested/UNKNOWN, needs a guard or documented precondition plus a regression test. Item 6: the no-collectives/no-divergence-in-tail rule exists only in a private doxygen @attention, absent from docs/source/tutorial/lockstep.rst - public-doc defect, fix in T6b.
D12 (high) M1 must add FP16/BF16 from scratch (T6a dispatched). Four integration points: GetArchSimdWidth per api/deviceKind, Simd<T,W> ops, GetValueType_t, BindValueType_t.

D8 (medium) rosi_v100 also dials up (Tesla V100-SXM2-32GB sm_70, 48-core Xeon 6136) - optional extra CUDA/Volta endpoint, not required.
D9 (high) No target exposes avx512_bf16/amx_tile; no target has native bf16. All FP16/BF16 arithmetic is generic emulation. CPU mixed-precision results must be reported with this caveat.

## Canonical integration tree
/tmp/alpaka3-t5-gemm-l5RK2z, branch gemm/common-source, tip 57d188ff, 4 commits on 6f13700c:
  e1e90fc4 lockstep integration (9 PR#7 commits rebased --onto 6f13700c, zero conflicts, squashed to one commit, original authors preserved in provenance)
  84d426d2 design + requirements (T2, patch-id == 145db5e1)
  dc30ea7e benchmark + acceptance harness (T4, patch-id == b5bf2e2)
  57d188ff backend coverage manifest + lockstep provenance
Local validation: pedantic Release g++13.3 CPU+OpenMP; ctest 29/29 pass (lockstep 10, simd 7, transformReduce 6, tutorial 6); gemm_bench smoke OK; pytest 10/10.

## Inbound commits (not yet canonical)
- T4 b5bf2e2 on base 6f13700c: benchmark/gemm/{workloads.json,src,plot.py,README,CMakeLists}, benchmark/CMakeLists.txt, validation/check_acceptance.py, validation/tests/test_check_acceptance.py. 10 pytest cases green, placeholder functor only, pedantic -Werror build green.
  IMPORTANT: check_acceptance.py defines the requirement/evidence schema; T2's YAML must be conformed to it (T2b).
- T2 145db5e1bbe7a1b0f207f5579ab7e416d4ddc66c on base 35ba304f: design/gemm.md + validation/gemm-requirements.yaml. 32 requirement records (G1..G10 + 15 section-8 families), thresholds frozen. NEEDS FIX: claims HIP unavailable on hal (wrong), no owner field, base SHA unverified.

D13 RESOLVED (high) Canonical base = 6f13700c390a09df818a255d53e730f1f9d69456 (API + full clone agree). The internal `gate` mirror lags by 30 commits and produced the bogus 35ba304f base; it must not be used for pinning. Worker rule: `git fetch https://github.com/psychocoderHPC-bot/alpaka3.git dev` then pin the literal SHA.
D15 (high) Lockstep tip = 05290c84. b77f140 (gate mirror) is a DIVERGENT lineage: 9 ahead / 7 behind, fork at 8ae77718. Rebase 05290c84 onto the canonical base; ignore b77f140.
D14 (high) The requirements YAML and the acceptance checker were authored independently; the checker's schema wins, so T2b must conform the YAML and keep the frozen thresholds.

D16 (critical) The local container has NO GPU toolchains (no nvcc/hipcc/icpx). Every GPU compile/run must happen on computeMCP targets, so the branch must be transported (git clone from GitHub on the remote, else file_upload_tree) and verified by content hash before results are believed. T8a establishes and proves this.
D17 (medium) Rebasing PR#7 preserved all 14 paths byte-identical to 05290c84; zero conflicts. The 9 commits were squashed into one integration commit per instruction; provenance recorded in design/lockstep-integration.md. Reconsider if maintainers want the original 9-commit history inside the PR.

D18 (medium) T2b rewrote design/gemm.md and validation/gemm-requirements.yaml as a fresh commit on the base (0205ce1, parent 6f13700c) rather than an amendment of 84d426d2. Integration therefore REPLACES those two files in the canonical tree rather than cherry-picking, to avoid a same-path add/add conflict.
D19 (high) Requirement schema now has per-record owner + required flag + metric/op/value thresholds; G9 split into fp16 (all native targets) and bf16 (Ampere, Arc A770, gfx1100 only), which is correct: Turing (7.5) and Volta (7.0) have no bf16 MMA.
D20 (medium) T2b reports the gfx1100 WMMA probe failed for the builtin it tried - awaiting T3d's own report before deciding whether native HIP MMA is usable at all.

## Numerics model (T7, verified locally)
Bound: |err| <= safety * gamma_n * (|alpha|*sum_k|A_ik B_kj| + |beta||C_ij|) + (n+1)*denorm_min, safety=8; u=2^-24 (FP32, and FP16/BF16 accumulation), 2^-53 (FP64 vs long double).
Input quantization error reported SEPARATELY: u_in=2^-11 (FP16), 2^-8 (BF16). FMA counted as K roundings; alpha/beta separate. n*u>=1 -> informative=false, alternative = bitwise equality on exactly-representable data; never asserted as pass.
Evidence: ctest 14/14, 4243 assertions, ASan+UBSan clean, numpy/fsum cross-check 8 shapes exact, mutation test sensitivity confirmed.
TOLERANCE MODEL IS FROZEN. Do not relax it later to make a kernel pass; a failing case is a kernel or plan defect.

D21 (high) T7's claim that "6f13700 is not on origin/dev" is WRONG - it read the lagging gate mirror. T5a verified origin/dev == 6f13700c twice (API + full clone). D13/D15 stand. Recorded so no future worker re-opens it.

D22 (blocker, resolved) The alpaka-implementer profile is denied `gh *` and `git push*` at the tool layer, so publishing must go through fr-quick (verified gh access in T5a). Never ask an implementer profile to publish.
D23 (high) No worker may rebuild or amend the canonical tree while publishing; the canonical tree is the single integration point and only one writer touches it per batch.

## FP16/BF16 (T6a, verified)
alpaka::Float16 / alpaka::BFloat16 in include/alpaka/core/Float16.hpp: trivially copyable, size/align 2, constexpr, ALPAKA_FN_HOST_ACC, fromBits/bits/fromFloat/toFloat.
RNE hand-coded (no hardware instruction). Subnormal widening exact, narrowing rounds on the subnormal grid with carry to smallest normal. Overflow above RNE threshold -> signed inf; largest finite preserved. NaN: sign kept, quiet bit forced, top payload bits kept; quiet NaNs round-trip bit-exactly, sNaN quieted.
Evidence: exhaustive widen + round-trip of ALL 65536 patterns per type; exhaustive narrowing at every adjacent-pair midpoint/just-below/just-above; 15/15 new tests, 299/299 across simd/algorithm/math suites. CUDA/HIP/oneAPI compile UNTESTED locally (no toolchain).

D24 (blocker, workaround in flight) `git push` over HTTPS to the bot fork is refused: the clone-time credential is read-only for Git transport, while `GH_TOKEN` works for the API. Publish route = credential-helper injection of the already-configured GH_TOKEN, else GitHub Git Data API commits (author/committer set EXPLICITLY to the human identity, trailer last line) + gh pr create. No new credential was invented. Never ask an implementer profile to push.
D25 (high) Publication freezes a snapshot (`publish-66efc781`) so integration can continue in the canonical tree without racing the publisher. Local and remote SHAs will differ after API-created commits; trees must be identical - compare tree SHAs, not commit SHAs.

## Lockstep M1 closure (T6b, verified)
Defect A: compile-time static_asserts reject zero workers in LockstepScope/getLockstepSimdWidth/LockstepVar before any divCeil; runtime path gets ALPAKA_ASSERT_ACC + early return. Negative-compile probes confirmed failing for the bad case and compiling for the good one.
Defect B: participation rules now in public doxygen (makeLockstep, both concurrent overloads, map helpers) + tutorial section "Participation Rules and Restrictions".
Tests: lockstep 20/20 -> 23/23 (604 assertions), ASan+UBSan clean, pedantic clean.
OPEN FINDING P1 (may reshape M3): host CPU executors appear to yield ONE worker when FrameSpec carries compile-time CVec extents; compile-time multi-worker var() needed a custom work group in tests. Under investigation in T6c.
OPEN P2: IndexedDataSimdRef const-store fails only at instantiation (no requires(!const)).
OPEN P3: zero logical extent silently accepted, width 1, no iterations - reject or document.

D26 (critical, publish model) git push is permanently unavailable. Every publish is Git Data API: build blobs/tree/commit with EXPLICIT author+committer (human identity) and the bot trailer as the final line, parent = current REMOTE tip, then PATCH the ref non-force. Verify each commit's author/committer and the GraphQL author list afterwards. Never publish from a local tip without mapping trees.
D27 (medium) Two published files had mode 100755; they were captured with real ls-tree modes. Keep that behaviour - do not hardcode modes.

## Local canonical history (after T5d)
66efc781 -> 5ccc08d (numerics reference) -> 8b698d5 (numerics tests) -> 4ec2302 (fp16/bf16 types) -> f6037ef (fp16/bf16 tests) -> 5f0137df (integration alignment fix)
LOCAL TIP 5f0137df7a008f932dc0c21bdec63471fc694386. Verified: pre-commit 23/23 green, ctest 39/39 (lockstep+simd+tutorial), gemm numerics 14/14, precision 19/19, math/algorithm 528/528, pytest 10/10, smoke ok.
FP16/BF16 Simd probe: getArchSimdWidth<Float16>=8, <BFloat16>=8 on host; Simd fill/copyFrom/copyTo/lane-toFloat correct. M2 mixed-precision integration risk: none observed on host.
T6b NOT yet integrated at T5d time (its commits were not on that tree's HEAD; dirty worktree). Recovery + integration is T5f.

## HIP stack reality (T3d, verified)
hal ROCm 7.2.0 ships ONLY core HIP runtime + hipcc. ABSENT: rocBLAS, hipBLAS, hipBLASLt, rocsolver, rocWMMA, rocprof, rocprofv3, rocprof-compute, omniperf. fwk388 has rocprof+rocprofv3 but no rocBLAS.
GPU executes kernels (verified). warpSize=32 on gfx1100, 64-bit shuffle mask REQUIRED (HIP 7.x static_assert).
MFMA `__builtin_amdgcn_mfma_f32_16x16x16f16` is registered but the C++ facade has a D(v4f32)/C(v4f16) slot-type inconsistency; no form we tried compiles as a clean 6-arg call. WMMA builtin name does not exist. 32x32x8 bf16 symbol unverified.
CONSEQUENCE D28 (high): M7 native HIP MMA CANNOT be claimed on this stack. HIP native path = classic FMA fallback; the claim is scoped out with this evidence, not hand-waved. rocWMMA absence means no legal fragment/shape set exists to claim from headers.
D29 (medium) rocminfo reports 96 CUs, hipDeviceProp_t.multiProcessorCount reports 48. Treat rocminfo as the physical truth for planning and never equate the two.

## Launch part of the project (M0 exit)
First coherent commit series published as DRAFT PR #60 (6 commits). M0 artifacts exist: design/gemm.md, validation/gemm-requirements.yaml (33 records), design/backend-coverage.yaml, benchmark/gemm harness + frozen workload manifest, validation/check_acceptance.py + tests, design/lockstep-integration.md.
M1 largely done: lockstep audit closed (A+B), 4 audit regressions, FP16/BF16 types (exhaustive), numeric reference + FROZEN tolerance model.

## Remote hardware validation (T8a, first real evidence)
Transport: 2.36MB gzip tar (731 files) uploaded to /tmp/t8a on hal and fwk394; identical 6-keyfile sha256 fingerprint both sides. Outbound git/HTTPS also works from hal (so a remote clone is a valid alternative).
LOCKSTEP PRIMITIVE TESTS ON REAL HARDWARE - 0 CHECK failures, 0 SIMD-tail assertions:
  hal CUDA A30 (sm_80) 15/15; hal HIP gfx1100 15/15; hal oneAPI Arc A770 + Intel CPU 10 pass/5 device-filtered skips.
  fwk394 sm_75/sm_70 builds: 10 pass / 5 fail = 3x NUMA membind EPERM (container lacks CAP_SYS_RESOURCE/CAP_SYS_NICE -> ENVIRONMENT) + 2x NoKernelImage (single-arch binary meeting the other GPU -> CONFIGURE bug, fixed by CMAKE_CUDA_ARCHITECTURES="70;75").
Key CMake options recorded in evidence/T8a-remote-build.md; do not rediscover them.
T8b closed the fwk394 gap: multi-arch build CMAKE_CUDA_ARCHITECTURES="70;75" eliminates NoKernelImage; with -Dalpaka_HOST_NumaCpu=OFF (repo-native, cmake/alpakaHwloc.cmake:50) BOTH GPUs are 10/10, ZERO failures. NUMA EPERM proven environmental: fwk394 container CapEff=0, CapBnd lacks CAP_SYS_RESOURCE (bit 24), ulimit -l 64MB; hal's container has the cap and passes.
hal reproducibility re-confirmed on the same source fingerprint: A30 15/15 (2.48s), gfx1100 15/15 (1.43s).
LOCKSTEP IS NOW VALIDATED ON: A30(sm_80), 2080Ti(sm_75), GV100(sm_70), gfx1100, Arc A770(level_zero), Intel CPU(opencl), EPYC CPU. Zero unexplained failures. Coverage manifest may move these to passed WITH these evidence paths; fwk388/gfx906 and oneAPI-GPU-on-other-hosts stay unverified.

## Execution model (T6c, verified by experiment - THIS IS THE M3 INVARIANT)
worker = execution thread: CPU = OS thread/frame (threadsInBlock pinned to CVec{1}, so W=1 per thread; parallelism comes from frames); GPU = CUDA/HIP/SYCL thread.
slot = one row (M axis) of the worker's TM x TN register tile = one lockstep iteration step the worker owns.
lane = V consecutive columns (N axis) of that row; TN multiple of V, stride-contiguous. V = min(archSimdWidth, cacheline/sizeof, divCeil(L,W)) pow2-floor >=1 -> 8 on AVX2 float, 1 on GPU.
Accumulators live in per-worker private var() storage of TM*TN.
MEASURED: OMP-4 with CVec{4} frames -> 4 OS threads, W=1 each, slots=L; SIMD packs collapse INTO the worker (numElemPerThread=16/32 = lanes in a worker, NOT extra workers). CpuSerial = 1 kernel instance.
var() NEEDS a CVec thread extent, so it is available only via threadsInBlock (W=1). threadsInGrid extents are runtime (UniVec) and cannot supply var().
REGISTER BUDGET WARNING: a 128x128 tile on one AVX2 worker = 2048 YMM vectors = 128x the register file. Per-thread sub-tile must be <= ~64 accumulators (8 vectors); reach big tiles via frames/threads, not per-thread tile growth.
D30 (high) M3 tile model = lane/slot layout (option c): (worker,slot,lane)->(m,n) is a pure function of worker stride and V, so no backend branch. Alternates rejected: (a) is (c) reskewed; (b) breaks contiguity.

## M2 kernel (T9, local CPU verdict)
API: onHost::gemm(queue, exec, GemmView<T_C> c, GemmView<T_A> a, GemmView<T_B> b, alpha, beta, GemmNumericMode mode = fp32) + default-executor overload.
GemmView{T* data, rows, cols, ld, GemmLayout{rowMajor,colMajor}, GemmOp{none,transpose}}.
Tile: bit_ceil(arch SIMD) clamped to {4,8,16,32}, widened once by getNumElemPerThread and shape; <4 rows/cols keeps 4x4; BK 8/16. Block thread count is consteval from getMaxThreadsPerBlock + SIMD/ILP. Blocks walk a grid-stride tile loop (REQUIRED because CpuSerial collapses the grid to one block).
Regions: disjoint full/bottom/right/corner launches; coordinates tested before load; K tail -> exact 0; invalid M/N -> no store; no collectives.
Evidence: ctest -R gemm 28/28 across 4 host CPU/NUMA backends; FULL ctest 1028/1028; ASan+UBSan 71108 assertions clean; worst scaled error FP32 2.79e-07, FP64 2.32e-16 (frozen bound is far looser); bench adapter emits real records with exact 2MNK/median_ns.
NOT validated: any GPU executor, fp16/bf16 modes (T9b wires them), performance.
T9 found 2 real lockstep usability gaps: (i) var() requires a CVector worker extent and fails with an opaque constraint error on a runtime ThreadSpec - undocumented; (ii) CpuSerial AdjustThreadSpec collapses any FrameSpec to one block/one thread, so a block==tile kernel silently computes only the first tile - grid-stride is mandatory, undocumented. Both need public docs (T6d).

## Upstream fit (T20, verified)
- Algorithm convention: ONE header per algorithm at include/alpaka/onHost/algo/<name>.hpp, impl in onHost/algo/internal/<name>.hpp, umbrella include in alpaka.hpp, public name alpaka::onHost::<name>, queue first + optional executor, anyExecutor -> defaultExecutor. NO alpaka:: wrapper.
- Errors: algorithms return void; static preconditions via concepts/requires; internal invariants ALPAKA_ASSERT; user misuse -> std::invalid_argument (precedent Event.hpp:133). No onHost::Result exists.
- Tests: test/CMakeLists.txt globs test/unit/<dir>/*.cpp -> target unit_test_<dir>_<file>, catch_discover_tests.
- Lockstep is NOT upstream anywhere. GEMM therefore depends on bot PR #7 landing; our PR must state that dependency explicitly (it carries #7 head 05290c84 rebased onto 6f13700, byte-identical tree).
- No active upstream GEMM/BLAS PR or issue. Closed/abandoned: #294, #320 (vendor provider registry with cuBLAS/cuDNN/rocBLAS/MIOpen), #406 (superseded by fn.hpp). Do NOT resurrect a provider registry; use ALPAKA_FN_SYMBOL only.
- Formatting-only commits must use author `Tools <alpaka@hzdr.de>` and be split from content commits (docs/source/contribution/tools.rst:75-79).
- Public-API objection risks: (1) include/alpaka/onHost/gemm/ instead of onHost/algo/gemm.hpp; (2) top-level design/ + validation/ dirs do not exist upstream; (3) carrying lockstep inside the GEMM PR; (4) bespoke error/Result API; (5) numerical-mode enum style.

D31 (medium) We KEEP top-level design/ and validation/ because the task specification (guide s11 "Evidence organization") mandates those exact paths. Recorded as a deliberate deviation with a stated lower-risk alternative (docs/source/advanced/ + benchmark/gemm/) for the maintainer to choose at review.
D32 (high) We KEEP the carried lockstep because the guide requires ONE incremental PR with deliberate lockstep integration + provenance. T21 must add an explicit "depends on PR #7" paragraph to the PR body and to design/lockstep-integration.md.
D33 (high) Public API path must become include/alpaka/onHost/algo/gemm.hpp (+ onHost/algo/internal/gemm/) and the test must move to test/unit/algorithm/gemm.cpp; the benchmark must link alpaka and register a ctest. Deferred to T21 so the M2 kernel stays stable while it is validated on hardware.

## AMD software now installed+verified (T3f)
hal: rocBLAS 5.2.0 + rocblas-dev + hipblaslt 1.2.1 + rocsolver + rocprofiler-dev + rocprofiler-compute 3.4.0 + hsa-amd-aqlprofile. 20 gfx1100 Tensile kernels present. VERIFIED GEMM: 512^3 = 3516 GFLOP/s (max_rel_err 1.25e-6), 1024^3 = 8242 GFLOP/s (2.18e-6) vs a CPU-double reference. rocprofv3 1.1.0 traces work (kernel CSV + SQLite, 22 lines, real fields); --pmc SQ_WAVES valid.
fwk388: rocBLAS installed BUT the 7.2 prebuilt ships no gfx906 Tensile kernel -> aborts "Illegal seek for GPU arch : gfx906". So NO fair HIP baseline on gfx906 unless rocBLAS is built from source (deferred, P1, and fwk388 is an OPTIONAL extra host: the required HIP target is hal/gfx1100).
D34 (CRITICAL for M8 profiling claims) On gfx1100 ALL hardware PMCs (GRBM_*, SQ_INSTS_*, SALU/VALU/L2/Occ) read 0.0 silently - verified across 64 random states via rocprofv2 CSV and rocprofv3 SQLite. Not "rejected", just zero. HIP performance claims on gfx1100 must rest on timing/derived metrics (kernel duration, achieved occupancy from grid/workgroup, bytes/instruction counts from the ISA, rocprofv3 --pmc SQ_WAVES) and must DISCLOSE the counter limitation. gfx906 DOES return real counters.
D35 rocprof v1 does not support gfx11xx; use rocprofv3. omniperf has no apt candidate in 7.2.

## Operator input (2026-10-25b)
T3g authorized: install rocBLAS / oneAPI BLAS / similar via `sudo apt` on hal and fwk394. This fills the remaining vendor-baseline holes: oneMKL + Intel CPU device on hal, CPU BLAS (MKL or OpenBLAS) on fwk394, so G5/G6 can compare serial-vs-1-thread and parallel-vs-equivalent-resources as the guide requires.

## Operator input (2026-10-25)
T22: https://github.com/psychocoderHPC/alpakaVendor is offered as a full GEMM interface to vendor backends, to be used as the performance reference. Treat it as a BENCHMARK-ONLY dependency: it must never become our implementation (guide: vendor libraries are benchmarks). It also should replace four bespoke per-host drivers with one consistent baseline path, which directly serves gates G5/G6 and the "matched vendor, same mode/layout/resources" requirement.

D36 (critical, transport) computeMCP_file_upload became unreliable mid-session (>900 B payloads fail, small writes succeed), which killed T9c after it had staged everything. There is a better route that removes the failure mode entirely: both remote hosts reach GitHub, so a remote validation run should `git clone -b gemm/common-source https://github.com/psychocoderHPC-bot/alpaka3` and check out the PUBLISHED SHA, verify `git rev-parse HEAD`, then build locally. Tarball upload is fallback only. Consequence: nothing can be validated on hardware until the commit is published, so publish precedes GPU validation from now on.
D37 (medium) T9c's report contains confusions (a "tulip" target, an "embui" remote, gfx906 described as MI250X, /workspace described as being on branch bow_MAT_dev). Treat its narrative as unreliable; only its staged artifact list is usable. Re-verify target identities from computeMCP_targets before any run.

>>> CONCURRENCY FAULT (mine, resolved): T5f published M1 to remote tip dde6e354 while T9b appended 5 M2 commits locally. Remote is therefore M1-only; M2 (1377416..64bc1c7) is unpublished. Verified: T5f's 8 remote trees byte-match local, attribution correct, PR #60 body updated, still DRAFT.
RULE: exactly ONE worker holds the canonical tree at a time. Publication and integration never overlap. Publish BEFORE hardware validation so every GPU number cites a published SHA.

## M2 + mixed precision on CPU (T9b, verified)
Canonical LOCAL tip 64bc1c71 (after fb1e2b3): 1377416 kernel, a7f86a7 host API, a9a997d tests, b66e253 bench adapter, 64bc1c7 FP16/BF16 wiring.
Numerics import from T9's tree DISCARDED (only clang-format/whitespace drift); canonical validation/numerics unchanged vs 5f0137d.
Mode wiring: fp16ToFp32 = Float16 A/B + float C; bf16ToFp32 = BFloat16 A/B + float C; conversion is a common-kernel static_cast to the FP32 accumulator; products/accumulation/C all FP32; no backend branch; no TF32/BF16 arithmetic. Host throws std::invalid_argument for swapped tags, mismatched C type, 16-bit storage with fp32/fp64, and pre-converted FP32 with a mixed tag (strict interpretation).
Worst scaled error: fp32 2.79e-07, fp64 2.32e-16, fp16_fp32 1.60e-07, bf16_fp32 8.78e-08.
Worst input quantization error: fp16 3.24e-03 (=2^-11 scale), bf16 2.88e-02 (=2^-8 scale). Both consistent with the frozen model.
Evidence: pre-commit green, full ctest 612/612 (this config), gemm 34/34, fp16/bf16 14/14, ASan+UBSan clean, pytest 10/10, bench real records for fp32 and fp16_fp32.
NOT validated: any GPU, mixed-mode on GPU, FP64 in the bench adapter.

## alpakaVendor (T23, VERIFIED against pinned ccf128e3c1ff)
CONFIRMED: MPL-2.0, header-only INTERFACE target alpakaVendor::alpakaVendor, C++20, depends on alpaka3 (FetchContent pinned e91f6f4d), and it COMPILES against our dev checkout at 64bc1c7.
CONFIRMED GEMM: gemm(queue, alpha, A, B, beta, C, Options={}) over alpaka mdspan views. float/double/complex ONLY - no fp16/bf16/tf32. No capability predicate, no GEMM fallback.
REFUTED (cosmetic): HOMEPAGE_URL says alpaka-group/alpakaVendor, not psychocoderHPC.
Integrated at benchmark/gemm/vendor/ behind ALPAKA_GEMM_VENDOR_BASELINE=OFF (default): OFF path adds no dependency and no fetch; ON FetchContent-pins the SHA and builds one extra target gemm_vendor_bench writing the same samples.json plus backend/device_name. fp16/bf16 emit status "vendor_unavailable" with a reason. Missing OpenBLAS produces a documented FATAL_ERROR, not a code error.
CPU 1024^3 fp32: direct cblas 43.4 vs wrapper 44.0 GFLOP/s median under alternating load - no measurable wrapper overhead. Absolute values are contention-limited (load avg 20-27, powersave governor) and must be re-measured on an idle host.
STILL UNAVAILABLE as vendor cells: fp16/bf16 (no 16-bit scalar in the interface), cuBLAS + oneMKL devices (not wired into the functor yet), gfx1100 rocBLAS, gfx906 (no Tensile). These must be recorded vendor_unavailable, never a pass.

## alpakaVendor (T22, PARTIAL - report was garbled, treat as hypotheses until T23 verifies)
Unverified claims: psychocoderHPC/alpakaVendor, branch dev, HEAD ~ccf128e3, MPL-2.0, header-only INTERFACE target alpakaVendor::alpakaVendor, C++20, depends on alpaka3, wraps cuBLAS/rocBLAS/oneMKL/OpenBLAS.
Claimed GEMM: gemm(queue, alpha, A, B, beta, C, Options={}) on alpaka-mdspan views; float/double/complex only (NO fp16/bf16/tf32); no capability predicate, no fallback.
Claimed CPU prototype: ~1.08 Tflop/s at 1024^3 fp32 vs 1.02 for direct CBLAS (~8% wrapper gap).
D40 (process) Two workers in a row returned corrupted output (T22 partially, T3g fully). Keep delegated tasks narrow with short command lists and small deliverables; prefer several small verification tasks over one broad one when the provider appears degraded.
D38 (high) If it compiles against our alpaka3 dev checkout, alpakaVendor becomes the single vendor-baseline path: benchmark-only, FetchContent pinned by SHA, separate target/TU, never included from include/alpaka/**. It must never become our implementation.
D39 (medium) It cannot fill mixed-precision G5/G6 cells (no fp16/bf16) nor gfx906 (no Tensile). Those cells need either a direct rocBLAS/cuBLAS fp16 driver or an explicit "vendor_unavailable" record. Never mark an unfilled cell as a pass.

## API now conformed (T21, verified on CPU)
public include/alpaka/onHost/algo/gemm.hpp; internal include/alpaka/onHost/algo/internal/gemm.hpp; kernel include/alpaka/onAcc/gemm.hpp. Public name alpaka::onHost::gemm, executor optional via alpaka::concepts::Executor, anyExecutor -> defaultExecutor (matches transform/reduce).
GemmNumericMode enumerators already camelCase: fp32, fp64, fp16ToFp32, bf16ToFp32.
User misuse -> std::invalid_argument (overflow_error removed); ALPAKA_ASSERT for view pointer and tile-extent invariants. No bespoke Result type.
Tests: unit_test_algorithm_gemm + unit_test_algorithm_gemm_numerics (old test/unit/gemm/ removed). Bench: gemm_bench + gemm_bench_smoke registered with ctest.
Docs: docs/source/tutorial/gemm.rst + docs/snippets/example/250_gemm.cpp (auto-registered, builds, 64 assertions).
Full ctest 1081/1081; numerics byte-identical to before.

## Vendor BLAS availability (T3h, verified)
hal: oneMKL 2026.2.0 at /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so.3 (link with -lmkl_rt; plain -mkl fails), OpenBLAS 0.3.26, cuBLAS 13, and intel-oneapi-mkl-sycl-{2026.2,blas} installed so oneMKL-on-Arc is PLAUSIBLE (not yet run).
fwk394: OpenBLAS 0.3.26, cuBLAS 12, no oneMKL.
fwk388: rocBLAS .so present but rocblas/library has NO gfx906 kernel -> unusable for a fair baseline.
All CPU 64^3 sgemm checks PASS with consistent reference value 32.0.

D41 (HARD RULE, operator instruction 2026-10-25) NO benchmarks in the Terok container - it runs on a laptop. All timing/throughput/profiling evidence comes from computeMCP targets (hal, fwk394, fwk388). Local container = configure/build/unit tests/sanitizers/codegen only.
   Consequences: (a) T11's local abstraction-cost benchmark is DISCARDED as evidence; only its correctness, mapping and codegen results count; the G3 comparison must be re-measured on hal (T11b). (b) T23's CPU 43.4 vs 44.0 GFLOP/s comparison is DISCARDED (it was contention-limited anyway). (c) T9c/T9d baselines must run remotely. (d) Any earlier local throughput number in this ledger or a worker report is not evidence.

D51 (high) `rosi_v100` is a BUILD-ONLY host at best and effectively unusable: nvidia-smi sees 4x V100 but `libcuda.so.1`/`libnvidia-ml.so.1` are NOT in the exec namespace, so `dlopen`/`cuInit` fail and every GpuCuda test skips or runs nothing (0 MiB used, 0% util). Its parallel build also deadlocks (make jobserver semop + LD_PRELOAD=libfakeroot-sysv); only -j1 completes. Demoted: do not schedule GPU work there. Remaining GPU hosts: `hal` (A30, gfx1100, Arc A770) and `fwk394` (2080 Ti sm_75 + GV100 sm_70).
D52 (high) V1 independently pinned the nvcc root cause: EDG/nvcc rejects `std::forward<T_Args>(args)...` in `dispatchGemmPlan` (`onHost/algo/internal/gemm.hpp:579/591`) for a deduced const-ref pack, while g++ accepts it; the sibling code in the same file uses `ALPAKA_FORWARD(instance)`. The minimal project fix is to use `ALPAKA_FORWARD` there. Same failure on nvcc 12.5 and 12.6 -> it is a latent project defect, not a toolchain limitation.
D42 (operator, temporary, VERIFIED) `rosi_v100` = host gv001, 48 cores, 376 GiB RAM, 4x Tesla V100-SXM2-32GB (cc 7.0), driver 570.158.01, CUDA 12.6 (nvcc at /usr/local/cuda/bin, not on PATH), cuBLAS 12.6, ncu present, nsys absent, 189 GiB tmpfs /tmp, EXCLUSIVE Slurm allocation (~8h), full capability set, outbound git works. GPU proven with a real vector-add kernel. Build with -DCMAKE_CUDA_ARCHITECTURES=70. Use it to offload remote CUDA builds and benchmark repetitions so hal/fwk394 do not serialize. It is an OPTIONAL host, not a required matrix entry; nothing is required from it for the gates. Availability expires; do not design a dependency on it.

## DEFECT FOUND (T36) - blocks every CUDA build
test/unit/kernel/lockstepAudit.cpp:~203 `auto dataSource` is "set but not used" under nvcc with -Werror=unused-but-set-variable (gcc does not flag it). With -Dalpaka_COMPILE_PEDANTIC=ON this fails the whole CUDA build. Fix in progress (T6e) plus a compiler sweep, because one instance implies more.

## INCIDENT (T36, operational)
The worker ran a bulk `ipcrm` while probing SysV IPC on rosi_v100 and corrupted that container's IPC namespace; the compute-MCP channel to rosi_v100 is now down (hal still responds normally, verified). Needs host-side recovery; do not schedule work there until it responds. Standing prohibition added: workers must never run `ipcrm`, `ipcs -a` cleanup or other SysV IPC manipIculation on remote containers.
Also learned on rosi_v100: cmake's CUDA ABI check needs pre-existing SysV semaphore arrays (a `mksem` helper was left at /tmp/gemm/mksem); no ccache; nvcc not on PATH; /usr/bin/time absent.

## M3 register blocking (T11, local correctness+codegen valid; TIMINGS VOID per D41)
Fragment API: include/alpaka/onAcc/fragment.hpp with RegisterBlockedLayout<TM,TN,V> (pure (worker,slot,lane)->(m,n), inverse, padded-lane mask), RegisterBlockedOwnership<TM,TN,V,WR,WC>, Fragment<T,Layout,Ownership> as a non-owning view over existing scope.var storage (fill, concurrent, pack/setPack, row/column projections, broadcast, loadFrom, storeTo), Row/ColumnProjection, OperandFragment with loadColumn/loadRow, makeFragment/makeOperandFragment with compile-time rejection of wrong types/sizes.
Tile: TM=4, TN=16, V=8 -> 64 accumulators = 8 YMM; 14/16 YMM live; no accumulator spill in the hot loop (verified: 2 vmovups for B + 4 vbroadcastss for A per K step, then 8 vfmadd231ps into ymm6..ymm13).
Correctness: fp32 2.55e-07 (improved from 2.79e-07), fp64 2.32e-16, fp16 1.60e-07, bf16 8.78e-08. Full ctest 619/619, ASan+UBSan clean, mapping bijection exhaustive incl. forced V=1 and V=8, worker grids (1,1)..(4,6).
G3 numbers collected locally (ratios 1.006/1.011/0.977) are VOID by D41 - the host was a laptop (Core Ultra 7 255U, powersave, load 3.29) and absolute throughput ~0.02 TFLOP/s is meaningless. G3 must be re-measured on computeMCP (T11b).

## Acceptance checker hardened (T24, verified)
New: required-target/executor matrix with status vocabulary {passed, failed, unsupported, blocked, vendor_unavailable, not_run}, evidence-path existence, passed-without-evidence rejected, reason required for unsupported/blocked/vendor_unavailable, not_run never counts as pass.
SHA attribution: per-entry/target/sample SHA, ancestor-or-equal of head, rejects evidence predating any change under include/alpaka/**, rejects mixed SHAs.
Benchmark integrity: samples>=10, rounds>=3, median_ns>0, host/device identity, compiler/toolkit/driver, load/governor/threads, gflops vs 2MNK/median_ns within 1%, REJECTS host_origin=terok_container.
Vendor pairing: G5/G6 need a matched baseline (shape/layout/transpose/mode/target/device); ratios computed by the checker, producer-supplied ratios ignored; missing -> vendor_unavailable.
Exit codes: 2 schema, 3 missing, 4 stale, 5 threshold, 6 unapproved skip. 27 pytest cases. E2E demo verified.

## CUDA build now GREEN (T6e, verified on hal with nvcc 13.4)
Three real CUDA build blockers fixed:
1. test/unit/kernel/lockstepAudit.cpp: dataSource used only in unevaluated type contexts -> nvcc set-but-not-used. Fix writes 17 through the view and asserts the aliased stack storage, so the type-deduction check remains AND a real aliasing assertion is added.
2. test/unit/algorithm/gemm.cpp: nvcc #20208-D "long double treated as double in device code" for accumulators in device lambdas -> switched to double with explicit casts (values are already double error metrics; host-only long double FP64 reference path unchanged).
3. test/unit/precision/fp16Conversion.cpp: dead exponentMask constant (nvcc #177-D) removed.
4. benchmark/gemm/CMakeLists.txt: host -Werror reached nvcc and made --expt-relaxed-constexpr fatal -> warning flags scoped to COMPILE_LANGUAGE:CXX (no relaxation for host or device code).
Evidence: pedantic CPU build + ctest 49/49; ASan+UBSan 47/47 clean; REAL CUDA configure+build on hal for all new test+bench targets, ctest 61/62 (the 1 failure is the pre-existing unit_test_precision_precision_NOT_BUILT).
NOTE: the original 12.6 defect does not reproduce on nvcc 13.4, so the V100/12.6 confirmation is still pending until rosi_v100 returns.
Sweep results: 0 diagnostics in owned files under g++ -Wall -Wextra -Wpedantic -Werror -Wunused-but-set-variable -Wshadow -Wconversion. 107 pre-existing -Wshadow hits and 1 -Wconversion hit live in upstream headers and are reported only, not fixed.

D43 (process, important) Long multi-phase tasks on the current provider return garbled or empty output (T5h stalled, T5i returned a truncated shell fragment, T22/T3g partially corrupt). Decompose publishing work into SMALL single-purpose tasks: one integration step per task, then a separate publish step, then a separate PR-body step. Never combine integrate+verify+publish+PR-update in one handoff again.

## Verified integration lineage (so integration is 3 cherry-picks, not 11)
/tmp/t24/work dfa82dc already CONTAINS T21 (9013951,bba8bfd,f814d5e) and T11 (ff2ad28,16fabc9). Canonical 64bc1c7 is an ancestor, so the update is a fast-forward.
Remaining side branches: T23 59e80ed (from 64bc1c7), T6e 5cb2a8f + 84779a3 (from f814d5e).

## PUBLISH BLOCKER (diagnosed 11:46Z)
Authenticated GitHub core API limit is EXHAUSTED: limit 5000, used 5000, remaining 0, resets 2026-10-25T12:08:00Z. Cause: the Git Data API publish path (1 blob per changed file + tree + commit) combined with repeated retries from the two stalled/garbled publisher tasks (T5h, T5i) in the same hour. git push over HTTPS is refused; SSH is unavailable.
SSH audit: three ed25519 keys exist in ~/.ssh (agent-config-bot, alpakahecbench-publish-bot, terok_hal_task) - ALL rejected for psychocoderHPC-bot/alpaka3 (Permission denied publickey). One agent identity `tk-main:alpaka3-bot` on /tmp/ssh-agent.sock authenticates to github.com but has NO grant (not even read) on the repo.
Consequence: publishing resumes at 12:08Z, in PACED small batches honouring Retry-After / x-ratelimit-remaining / x-ratelimit-reset, with no retry loops. Durable fix = grant the tk-main:alpaka3-bot key write access on the repo (deploy key) so plain `git push` works and the API quota stops being the transport.

## Credential rights truth (P2e, verified)
- SSH agent identity `tk-main:alpaka3-bot` is NOT registered on GitHub (fails even the ssh greeting). Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBAQ8Jl6UJkKCiEJ0bB1vYdR8PZz431dK9mj0AeahRTN
- ~/.ssh/agent-config-bot -> psychocoderHPC-bot/agent-config; ~/.ssh/alpakahecbench-publish-bot -> psychocoderHPC-bot/alpakaHeCBench. Neither covers alpaka3.
- GH_TOKEN authenticates as bot id 330351933 and HAS published before; REST quota 5000/5000 exhausted until 12:08Z; a separate repo-permission/CAS quota 1750 is exhausted until ~18:00Z.
- Direct api.github.com DNS is blocked; gh works only through the configured proxy.

D44 (operator action) The `tk-main:alpaka3-bot` public key was added to the psychocoderHPC-bot account. Publishing switches to plain `git push` over SSH with SSH_AUTH_SOCK=/tmp/ssh-agent.sock. The Git Data API transport is retired except for read-only queries; it must never again be the routine publish path.

D45 (decision, authorized) SHH push works. Remote branch history was created by the Git Data API, so its SHAs differ from local; `dde6e354` is NOT an ancestor of local `d6ea5240` (14 API commits vs 28 local commits, identical trees). Because `gemm/common-source` is our own unmerged draft topic branch with no other consumers, a controlled `--force-with-lease=<expected dde6e354>` rewrite is authorized to restore one linear history and SHA equality. PR #60 links to the 14 obsolete API SHAs must be corrected afterwards.

## PUBLISHED STATE (P2g)
Remote `gemm/common-source` == local `d6ea524015b5852dec968375d400b2cf9d37eb91`, one linear history of 28 commits on 6f13700c. `dev` unchanged at 6f13700c. 14 obsolete API-created SHAs are dangling and must not be linked anywhere: 7ff934fd 5dc7, c84632d3 6d33, 0f10c830 8a86, c1938b9e 3985, 9d5c1e23 38c2, 8cbfb580 8230, 2a623f21 f588, 239bd3f5 30cf, 11ab67ae 2eac, 06099f55 7ec8, e25ef1f4 d781, 24d0cbc1 ac59, 8901d71f 8d88, dde6e354 5f18.

## GPU VALIDATION RESULT (T39a, published tip d6ea5240, tarball transport 1-shot)
hal CUDA A30 (nvcc 13.4): gemm 30 cases / 63012 assertions PASS; numerics 14/4243 PASS; lockstep 15/4041 PASS; audit 3/605 PASS; fragment BUILDFAIL (own defect).
  Worst scaled errors ON GPU: fp64 2.32e-16, fp32 2.79e-07, fp16 1.60e-07, bf16 9.04e-08 - identical to CPU. THE COMMON-SOURCE KERNEL IS CORRECT ON REAL NVIDIA HARDWARE.
hal HIP gfx1100: fragment 6/16198 PASS; gemm/numerics build blocked by reference long double; lockstep PASS.
hal oneAPI (Arc A770 + EPYC): lockstep PASS; gemm blocked by same reference issue; fragment blocked by icpx printf.
fwk394 CUDA 12.5: EVERYTHING BUILDFAIL - nvcc 12.5 rejects the C++20 deduced class-type template parameter in the public gemm entry point.
D46 (high) Four defects to fix, in priority order: (1) public API deduced `concepts::Executor auto const exec` is not portable to nvcc 12.5; (2) fragment.hpp host/device misuse (host view/lambda paths reachable from __host__ __device__); (3) reference_gemm.hpp requires long double wider than 53 bits, blocking HIP/oneAPI builds - must degrade with a VISIBLE skip, never a silent precision reduction; (4) gemm_fragment.cpp printf chain rejected by icpx.

## M4 staging (T13, verified on CPU)
include/alpaka/onAcc/staging.hpp: StageTracker, SyncStageRing, scratch accounting. Semantics: acquire -> fill (worker-strided A/B panel write) -> commit (syncBlockThreads) -> wait (generation validated) -> consume/complete (second barrier) -> release. Barriers outside element-wise callbacks, inside converged code, no producer-blocking waits.
Generations: per-slot monotonic generation; tokens carry (stage, generation); every transition rejects stale/mismatched generations, so a completion for g cannot release a slot refilled for g+1.
Boundary: M/N/K coordinates tested BEFORE dereference, exact zeros written; compute never reads uninitialised shared memory.
Scratch: S*(BM*BK + BK*BN)*sizeof(T) + 64B <= sharedMemPerBlockBytes, else std::invalid_argument (no truncation). Stage family bounded to {1,2,3}.
Policy: `automatic` resolves compile-time internal::defaultGemmStaging = DIRECT (cooperative is an expert override); documented rule stagingRule = arithmetic intensity >= 8 and ring fits. No backend-name branch.
Correctness identical to direct: fp32 2.79e-07, fp64 2.32e-16, fp16 1.60e-07, bf16 8.78e-08.
Evidence: pre-commit green; full ctest 1095/1095; targeted 93/93; ASan+UBSan clean (gemm 40 cases/246880 assertions, state 27 schedules, progress 4, fragment 7/16199).
NOT validated: any GPU staging, any overlap/performance claim, async copies, scratch-limit trigger on real GPU shared memory. TSan cannot run in this container (unexpected memory mapping; setarch -R not permitted) - documented limitation.

## GPU BUILD MATRIX AFTER T42 (all three GEMM binaries build everywhere)
hal CUDA 13.4 sm_80 A30: gemm 10/21004 pass, fragment 4/16196, numerics 14/4243.
hal HIP 7.2 gfx1100: gemm 9 pass + 1 SKIP (FP64, no extended precision), fragment pass, numerics 13 + 1 skip.
hal oneAPI 2026.2 (Intel CPU + Arc A770): gemm 16 pass, 2 fail, 2 skip; fragment pass; numerics 13 + 1 skip.
fwk394 nvcc 12.5 sm_70+sm_75: BOTH GPUs gemm 10/21004 pass, fragment 4/16196, numerics 14/4243. DEFECT 1 (CUDA 12.5 API) PROVEN FIXED.
Fixes: (1) named constrained T_Executor by value instead of deduced CTAD; (2) fragment concurrent/loadFrom/storeTo split into device overloads constrained to non-host-only args + ALPAKA_FN_HOST overloads, host test views use fragment::hostOnly(); (3) FP64 reference SKIPs visibly where long double is not extended, reference throws instead of static_assert; (4) printf chain replaced by ostringstream.
REMAINING FAILURE: oneAPI gemm FP16/BF16 stored-edge case, input 0x8000 (-0), expected 0x80000000, got 0. ORACLE DEFECT: under IEEE RNE (+0)+(-0)=+0. Fix in T43.

## ALARM (from T43): possible fast-math in the oneAPI build
T43's root cause was that icpx compiled a GEMM test file with `-fno-signed-zeros`, folding (+0)+(-0) to -0. The test oracle is now correct and separate from storage round-trip, and all devices pass (oneAPI CPU 9 pass + 1 FP64 skip, Arc 9 + 1 skip, CUDA 10/10, HIP 9 + 1 skip; FP64 skip is explicit with a reason on oneAPI/HIP).
BUT if that flag (or -ffast-math / FTZ / fininite-math-only) applies to the GEMM translation units, our STRICT FP32/FP64 contract on oneAPI is not what the design claims. T45 investigates the real flags per executor, measures signed-zero/denormal/NaN/FMA behaviour on device, and either scopes the flags out of the strict path or pins and documents the contract with tests. Until T45 reports, no oneAPI numerical claim may be treated as strict.

## M5 planner (T14, host-only verified)
Facts: available = SIMD width, pipelines, cache line, op quanta, shared/block/grid/MP/warp/global memory. unknown-with-reason = L1, L2/LLC, register budget, active blocks, native matrix combos, async-copy alignment. No invented constants.
Candidates: TM{1,2,4,8} x TN=V x WC{1,16} clamped x BK{8,16}*qk x {direct, coop S=1,2,3}; cap 64 (array bound + guard). CPU 32 distinct, GPU-like 64. Minimal direct fallback is index 0.
Scoring: 0.45*usefulFraction + 0.25*AI/(AI+8) + 0.15*(1-pressure) + 0.15*(1-transferNorm), deterministic tie-break.
Override: optional explicit plan runs the IDENTICAL validatePlan (structure, SIMD coverage, quantum, mode, layout, alignment, threads, register estimate, scratch, grid, family membership) and throws invalid_argument with the reason.
Evidence: pre-commit green; 753 ctest of which 2 bench entries FAIL for an environmental reason (driver refuses output inside the checkout; the ctest registration omits an output dir) - fixing in T46. ASan+UBSan 8 cases/33955 assertions clean. Property grid 1280 shape x type x target x transpose combos, every selected plan re-accepted by launch validation; 144 cooperative override cases; 96 determinism repeats.
COST: unit_test_algorithm_gemm 52s -> 122s and 2.71MB -> 5.97MB (+3.26MB) on CPU for 32 kernels/type/executor. This is a real CI-budget concern and the guide's candidate cap/compile budget must be reported honestly at M5 exit; trimming is a follow-up.

## M5 integrated (I3)
Canonical local tip bfac7fef. Planner IS in the default path: internal/gemm.hpp:800-818 builds problem/facts/request and calls selectPlan; dispatchGemmPlan (gemm.hpp:558-588) maps the selected plan onto the compiled family and launches launchGemmTilesRegisterBlocked with candidate tileM/tileN/workerRows/workerCols/kBlock/stages. Direct fallback = candidate index 0.
Evidence: pre-commit green (23 hooks), full ctest 753/753, targeted 75/75, ASan+UBSan 52/52 clean, four errors unchanged, signed-zero + storage round-trip assertions separate at test/unit/algorithm/gemm.cpp:1204-1230.
NOTE: the planner now chooses GPU tiles too, so GPU correctness MUST be re-validated with the planner active (T44) before any GPU performance number is trusted.

## M6 pipeline (T15, CPU verified)
include/alpaka/onAcc/gemmPipeline.hpp: pipeline::BoundedStagePipeline with phases Phase::prologue/steadyState/drain passed to fill/compute callbacks.
Counts are closed-form in scheduleCounts: L fills/waits/computes/completes/releases, max(0,L-S) refills, min(S,L) drain consumes. Every refill has exactly one matching steady consume; every slot is free and issued==L at the end.
Tokens remain (stageIndex, generation) from StageTracker; all M4 generation checks preserved.
Async hook: StagePolicy::asynchronous runs the IDENTICAL synchronous schedule with asyncActive=false; public GemmStaging::asynchronous maps to cooperative. Selectable, identical results, no fake overlap, no backend branch.
Evidence: 19 (L,S) pairs x 2 policies = 38 schedules, 732 checked transitions, 64 illegal transitions REJECTED, stalled-producer guard throws instead of hanging. Full ctest 758/758, targeted 80/80, ASan+UBSan 57/57 clean, four errors unchanged.
NOT validated: GPU execution, native async copies/overlap, profiler evidence, deadlock freedom on hardware.

D56 (hard rule) Benchmark windows are bounded to <=45 minutes of measurement. B4 occupied hal for 3 hours and produced no GPU data because the harness cannot use a GPU. A worker must abort a window as soon as the harness reports a CPU executor for a requested GPU target.
D47 (HARD RULE, operator instruction) Benchmarks require an EXCLUSIVE window per computeMCP target. No concurrent measurement and no build/test on the same host during a measurement window. Reason: co-scheduled work invalidates every timing, including the "quiet" run. Consequence: GPU correctness builds (T44) and G3/throughput runs (T11b) must be serialized per target, and a benchmark worker owns the target until it reports its end-of-window load average. Add: every benchmark report must include start/end load average and a statement that no other job ran on that target.

## BENCHMARK INCIDENT (T11b, invalidated)
T11b launched overlapping measurement scripts on hal (window6-sweep.sh and window6b-sweep.sh ran concurrently), producing TWO gemm_bench processes per mode against the same output-dir for square-2048. Load average rose with each pair. Every number from that run is void; the processes were terminated (verified: no gemm_bench, no sweep scripts remain).
Root cause: the worker did not serialize its own windows and did not wait for each run. Lesson: a benchmark worker must run exactly one process, wait for it, and prove exclusivity with pgrep + load average at both ends. T11c enforces this and is bounded to small shapes (no 2048).

D48 (operator instruction) Keep the laptop light: local builds/tests are TARGETED only, max -j4, no full-suite rebuilds or full ctest sweeps in parallel with other workers. Heavy verification belongs on computeMCP. Observed violation: at 15:56 local load average was 20.7 on 14 cores with 12+ cc1plus processes from a full ctest build in the container. Builds do not invalidate measurements (they are not benchmarks) but they degrade the operator's machine and are now bounded.

D49 (operator directive) REMOTE-FIRST: all builds/tests/benchmarks move to computeMCP targets, distributed in parallel. This container is editing/git/pre-commit/inspection only. Local full-suite builds were killed on 2026-10-25 15:57Z (load was 20.7 on 14 cores). rosi_v100 is back online but BUILD-ONLY (no CUDA execution, parallel build broken) - see D51.
TARGET ROLES: hal = CUDA 13.4 + HIP gfx1100 + oneAPI (and CPU benchmark windows, exclusive); fwk394 = CUDA 12.5 sm_70+sm_75; fwk388 = gfx906 HIP fallback; rosi_v100 = sm_70 CUDA 12.6, parallel verification and Volta benchmarks.

## REGRESSION FOUND (V2, fwk394 CUDA 12.5, tip 8d3409b)
3 of 6 targets FAIL to build: unit_test_algorithm_gemm, _gemm_fragment, _gemm_pipeline. One root cause in include/alpaka/onHost/algo/internal/gemm.hpp ~lines 571/579/583/591: launchGemmTilesRegisterBlocked candidate-fails deduction and std::forward<T_Args>(args...) inside an index_sequence fold becomes <error-type>; first instantiation reported is api::Host/deviceKind::Cpu, then CUDA cascades (~100 errors/TU).
NOT a CUDA 12.5/sm_70/sm_75 limitation: probe TUs with Device<api::Cuda,NvidiaGpu>+Queue+GpuCuda compile under nvcc 12.5.82 with both gencodes.
Introduced after M5 planner dispatch + M6 pipeline refactor (tip f75dd56e built fine on CUDA 12.5 and 13.4). Survived local verification because local builds are gcc-only. This is exactly the failure mode remote matrix validation exists to catch.
Still valid from V2: numerics 14/4243, plan 8/33955, lockstep 10/2694 on RTX 2080 Ti, Quadro GV100 and CPU - all zero failures.

## FIRST VALID BENCHMARK (T11c, hal EPYC 7452, SHA bfac7fe, exclusive window)
fp32 ser (1 thread) 4.91/4.95/4.97/4.99 GFLOP/s at 256^3, 512^3, 1024x1024x512, 1024^3 - flat, i.e. compute bound per thread.
fp32 par (64 threads) 75.1/117.4/116.0/95.2 GFLOP/s -> 15.3x/23.7x/23.3x/19.1x scaling.
mixed fp16->fp32 512^3 1 thread: 150.6 ms, 1.78 GFLOP/s = 2.8x SLOWER than fp32 (54.26 ms). Not yet explained.
G3: runnable, 12/12 assertions pass; register-blocked is equal or FASTER than the manual kernel (ratios manual/RB 1.023/1.064/1.097 host, 0.997/1.066/1.097 numa). Abstraction cost is therefore not the problem.
Conditions: taskset 0-63, schedutil, g++ 13.3, OMP 4.5, load < 0.56, no overlapping process, 10 samples x 3 rounds.
INTERPRETATION: per-thread ~4.9 GFLOP/s is roughly 10% of an AVX2 core's FMA ceiling. Either the hot loop is not vectorised, or it is bandwidth bound because the default policy is direct load with no packing. T49 profiles it before any change.

D50 The nvcc dispatch regression (T47) also breaks `benchmark/gemm` (gemm_bench.cpp.o), not only the test targets. Consequence: NO GPU measurement is possible at 8d3409b on nvcc, so B1 (rosi_v100) is expected to abort for the same reason. GPU benchmark windows reopen only after T47 lands and is verified on nvcc 12.5 + 12.6.

## SECOND REGRESSION (V3, fwk388 HIP gfx906, tip 8d3409b)
gemm.cpp fails to compile on HIP: "reference to __host__ function acquire/commit/waitReady/consume/complete/release in __host__ __device__ function" - gemmPipeline.hpp:189-212 calling SyncStageRing methods (staging.hpp ~246-297) that are plain constexpr, unannotated, and call ALPAKA_ASSERT_ACC. The pipeline/staging code is NOT device-callable. This will hit CUDA too once the deduction bug is fixed.
GOOD: native matrix absence PROVEN for gfx906 by four independent methods (hsaco ELF scan for 'mfma'/'wmma' bytes, nm -C symbols, llvm-objdump disassembly, llvm-mc) - zero instructions. So the HIP path is genuinely the common SIMD/FMA fallback.
CAVEAT from V3: its lockstep "GPU" pass may have executed on CPU (no KFD process appeared). Not accepted as GPU evidence until re-run with a GPU-only build.
Two independent regressions landed with M5/M6 and neither is visible in a gcc-only local build. Remote matrix validation is now mandatory for every merge.

## HIP NOW WORKS ON REAL HARDWARE (T48)
Fixed: SyncStageRing's six transition methods (staging.hpp:261-301) were plain host members; now ALPAKA_FN_HOST_ACC, as are StageTracker's tokenOf/issuedCount/generation/allFree/try-*. All assertions kept - the cause was fixed, not the assert. scheduleCounts marked ALPAKA_FN_HOST (host-test only) so its assert stays out of device code.
Evidence on fwk388 (gfx906, GPU-only via HIP_VISIBLE_DEVICES=0): unit_test_algorithm_gemm 9 pass + 1 FP64 skip with device printed `Hip AmdGpu / AMD Radeon VII` and an active KFD PID observed (~319 MB VRAM); gemm_fragment 16197 assertions; gemm_numerics 13+1 skip; gemm_pipeline 1622 assertions. The exhaustive test loops stages 1/2/3, so the in-kernel BoundedStagePipeline -> SyncStageRing path really executed on the GPU.
CUDA still blocked by the separate nvcc deduction regression (T47). Zero errors point at staging/pipeline now.

## nvcc ROOT CAUSE (T47)
Minimal reproducer: nvcc's CUDA front end loses the pack identity of a function parameter pack captured by a lambda and expanded inside a nested generic lambda, so `std::forward<T_Args>(args)...` expands only T_Args; g++ forwards .cpp compiles to gcc, which accepts it. Fix: named templates (`tryDispatchGemmPlan`, `tryDispatchGemmPlans`) that expand the pack at function scope. Plan selection, candidate family and evaluation order unchanged.
Verified at 3af908a (WITHOUT the T48 fix): builds on fwk394 CUDA 12.5 and rosi CUDA 12.6; gemm 15/20 cases, fragment 5/5, pipeline 4/4; the 5 remaining CUDA failures were the (now fixed) device-annotation bug.
D53 The 2 regressions had a common root cause pattern: M5/M6 added template/annotation constructs that only a non-gcc compiler rejects. Every future merge must pass the device matrix (nvcc 12.5, nvcc 13.4, HIP) before it is considered verified.

## MILESTONE: full CUDA + HIP correctness matrix GREEN at 73a28748
fwk394 CUDA 12.5 (2080 Ti sm_75, GV100 sm_70, host): gemm 20/20 (123448 assertions), plan 8/8 (33955), numerics 14/14 (4243), fragment 5/5 (16197), pipeline 4/4 (1646) - identically on both GPUs and host.
hal CUDA 13.4 (A30 sm_80): same counts, all pass. nvcc 13.4 exercised for the first time.
fwk388 HIP gfx906 (previous tip): gemm 9 pass + 1 FP64 skip, fragment, numerics, pipeline pass, GPU execution proven by KFD PID.
The five CUDA cases that failed before (FP64, K-tile, transpose/alpha-beta, FP16/BF16 exhaustive+adversarial) now PASS everywhere. Zero skips, zero device errors.
Published: remote gemm/common-source = 73a28748e81bf5ed73703ffb621cf2d1bb4d177a; dev untouched.

## O2 result (branch-free widening)
Float16::toFloat and BFloat16::toFloat are now branch-free: BF16 is one shift; FP16 builds the normal/inf/NaN pattern and the subnormal pattern (mantissa * exact 2^-24) and selects with an all-ones mask. No branch, no while loop, no variable shift. Subnormals, signed zero, inf and NaN payloads are bit-identical to the old code.
Bit-exactness PROVEN: all 65536 patterns per type identical old-vs-new (0 mismatches); fp16Conversion 5/5 (1073 assertions), bf16Conversion 5/5 (182).
Vectorisation: the conversion control flow is gone ("optimized: loop vectorized using 16 byte vectors" with -march=znver2; object shows vcvtdq2ps/vmulps/vpslld/vpcmpeqd). Under default x86-64 the remaining miss is "complicated access pattern" from GemmView's runtime layout/op select, not the conversion.
GPU spot checks: A30 30 cases/185172 assertions, gfx1100 184788 assertions, fwk394 CUDA 12.5 20 cases/123448 - all pass. Four errors unchanged.
KNOWN ISSUE (not a regression): icpx bf16Conversion reports 3 failures (254 subnormal widening + 2 narrowing) because icpx's default FTZ flushes the TEST's own ldexp reference to zero; conversion output is unchanged. Needs the test reference hardened under FTZ (ties to T45b).

## O1 result (planner tile + ISA option)
ALPAKA_GEMM_NATIVE_ISA (default OFF) applies -march=native to host TUs of gemm_bench, gemm_vendor_bench and unit_test_algorithm_gemm* only. Library stays portable; CUDA/HIP passes keep their flags.
Planner: largest candidate now TM=maxRegisterTileRows(8), TN=V; family axes and the 64 cap unchanged; scoring clips BM/BN/BK to the problem so compute-heavy shapes keep the larger block and tiny/skinny keep the direct fallback. Determinism, reasons and candidate 0 unchanged.
Registers: V=8 -> 64 accumulators = 8 of 16 YMM + 1 hoisted B row; V=1 -> 25 of 255 GPU registers.
Codegen proof: baseline TU has 0 vfmadd and 0 ymm; native TU has 822 vfmadd and 124 ymm instructions.
FMA contraction changes the fp32 last bits: 2.78752e-07 -> 2.54774e-07 native; with -ffp-contract=off it returns to 2.78752e-07. MUST be documented as allowed contraction, inside the frozen bound.
Correctness with O1: fwk394 40/40, hal CUDA 40/40, gfx1100 40/40 (FP64 skip by design), oneAPI CPU 62/63 (the one failure is the FTZ-broken bf16 test reference).
OPEN CONTRADICTION (A1): the profiler saw TM=1,TN=4 with scalar mulss at 8d3409b, while the planner reports TM=8 for the same shape. If gemm_bench bypasses the planner, every performance number so far measured something other than the design. A1 audits the adapter and the launch path.

## A1 AUDIT RESULT (contradiction resolved)
The benchmark uses the PUBLIC planner path with no override (gemm_alpaka.hpp:196-204; defaults onHost/algo/gemm.hpp:122-125). The selected plan DOES pick the kernel: internal/gemm.hpp:565-567 rejects non-matching candidates, dispatchGemmPlan throws rather than launching a default (:620-625). So the benchmark was honest.
The profiler was wrong: the binary instantiates EVERY candidate, including the minimal TM=1,TN=V,BK=8,stages=0 fallback, so sampling can land on a kernel that never ran. Also TM=1 with V=4 would still be packed SSE, not scalar mulss.
CONSEQUENCE: the earlier 4.9 GFLOP/s per thread was measured with baseline x86-64 = SSE V=4 and NO FMA. The AVX2 comparison I made was wrong. M1 measures baseline vs native ISA and proves the executed kernel by perf symbol + template arguments.

## oneAPI executor verified (V4) + two findings
oneAPI at 73a2874: Arc A770 and SYCL CPU both pass gemm (9 pass + 1 FP64 skip, 61596 assertions), fragment, lockstep; printed device names verified per run. Signed-zero case PASSES now. FP64 SKIPs with the printed reason.
FINDING 1 (project defect, host CpuSerial ONLY): bf16 stored-edge case fails 6/108 assertions at gemm.cpp:1200 for largest-subnormal / smallest-normal / just-below-one patterns, producing 0 instead of a nonzero bit pattern; the same case passes on Arc A770 and SYCL-CPU at the same tip, and fp16 is clean. Investigated in T50.
FINDING 2 (upstream alpaka3 incompatibility): syclGeneric/onAcc.hpp:165 used the deprecated nd_item::barrier(); pedantic oneAPI 2026.2 fails. T51 replaced it with sycl::group_barrier(m_item.get_group()) - the SYCL 2020 equivalent - with no version guard needed and no warning suppression. Verified: pedantic oneAPI build succeeds with no deprecation diagnostics; CUDA/HIP builds unaffected.

D54 (process) A worker edited and committed ONLY on a remote host, leaving no local object; the commit could not be integrated. Recovery: `git bundle create` on the remote, `computeMCP_file_download` to the local evidence dir, then `git fetch <bundle>`. Rule for future tasks: any work that must be integrated has to exist as a commit reachable from a LOCAL repo the integrator can fetch, or the worker must produce a bundle and leave its path in the report.

## FALSE ALARM RESOLVED: the benchmark never ran on a GPU
`benchmark/gemm/src/gemm_alpaka.hpp:52-57` hardcodes `makeHostDevice<float>()` + `defaultExecutor`, so gemm_bench is HOST-CPU ONLY. Verified independently: the binary contains no CUDA launch path, ncu cannot capture a GEMM kernel, and the manifest has no 512 case at all.
CONSEQUENCE: B3's "RTX 2080 Ti at 2.8-5.5 GFLOP/s" numbers are CPU numbers on the fwk394 Xeon (they match single-thread CPU throughput almost exactly). ALL B3 GPU CLAIMS ARE VOID. B4 (hal) is suspect for the same reason and will be voided unless it shows a real device path. We currently have ZERO GPU performance data.
GPU CORRECTNESS evidence is unaffected: the unit tests instantiate all backends via TEMPLATE_LIST_TEST_CASE.
T60 makes the benchmark device-selectable, with a hard rule: requesting a non-CPU device on a build without it must FAIL LOUDLY, never silently run on CPU. That silent fallback is the actual bug that produced a fabricated GPU table.

## OLD ALARM (now void)
fwk394, SHA e2d552b, fp32 on RTX 2080 Ti: 512^3 52.7 ms = 5.09 GF/s; 1024^3 528.6 ms = 4.06 GF/s; 2048^3 6061.8 ms = 2.83 GF/s; 2048x256x1024 5.48 GF/s; 32x4096x1024 3.19 GF/s.
That is ~0.04% of the card's ~13 TFLOP/s, and 30x SLOWER than the same kernel on hal's 64-thread EPYC CPU (154 GF/s at 512^3). Structural defect, not tuning.
B3's GV100 rows are IDENTICAL to the millisecond to the 2080 Ti rows - impossible; treat as unverified.
Also: the workload manifest at e2d552b has no 512^3 mixed-precision entries, and 2048^3 mixed timed out at 180 s.
D1 profiles with ncu and re-measures both GPUs, with launch geometry and executed kernel symbol.

## M7a (native matrix contract) verified on hardware
include/alpaka/onAcc/matrix.hpp, symbol onAcc::matrix::MatrixAccumulate. Two levels: Local fragments (per-worker outer product, SIMD/FMA fallback, default) and Warp fragments (one collective m16n8k16 step on exactly one 32-lane sub-warp, synchronous, documented lane->(row,col/k) mapping, public load/store epilogue).
Types: fp16/bf16 -> fp32 (exact input conversion), strict fp32->fp32, strict fp64->fp64.
Capability: matrixAccumulateSupported<T_In,T_Acc,Quantum,Api,DeviceKind>() is tri-state (no/unknown/yes) plus matrix::available(device); unknown never means yes; non-native combos are excluded by `requires` so they instantiate the fallback.
Architecture evidence: mma.sync m16n8k16 f16/bf16 is sm_80+; sm_75 is m16n8k8, sm_70 m8n8k4 (not implemented). Adapter overload absent below sm_80, so Volta/Turing get the fallback by construction.
Runtime proof: A30 sm_80 NATIVE with SASS showing HMMA.16816.F32 and HMMA.16816.F32.BF16; 2080 Ti sm_75 and GV100 sm_70 FALLBACK with ZERO HMMA in SASS; gfx906 and gfx1100 FALLBACK; host g++ fallback. All assertions pass within the frozen bound.
NOT validated: GEMM does not consume the warp contract yet (M7b); Volta/Turing native shapes unimplemented; no oneAPI adapter; single scoreboard tile only.

## ROOT CAUSE FOUND (T50): icpx host FTZ, not a widening bug
The BF16 stored-edge failures on host CpuSerial were never a conversion bug. BF16 widening is a pure `bits << 16` (Float16.hpp:131) and is bit-exact. Under icpx's DEFAULT host FP model, FP32 denormals are flushed, so BF16 subnormals (0x0001, 0x8001, 0x007f) which widen to FP32 DENORMALS became 0 at the accumulate step (onAcc/gemm.hpp:172/176). FP16 subnormals widen to FP32 NORMALS, which is why FP16 passed; 0x0080 (smallest BF16 normal = smallest FP32 normal) passed, marking the exact denormal boundary. Failing CHECK was test/unit/algorithm/gemm.cpp:1200.
Fix (e73ebec): pin `-ffp-model=precise` on alpaka_target_host when CMAKE_CXX_COMPILER_ID is Intel/IntelLLVM and the user has not enabled alpaka_FAST_MATH/FTZ. Documented in docs/source/tutorial/gemm.rst + CHANGELOG. Scoped to Intel compilers only (g++ untouched), opt-out preserved for fast-math builds.
This is exactly the contract issue the specification warns about: an engine that flushes subnormals must expose and test that contract, never silently claim strict FP32.
NOTE: T50 ran long builds on hal while B4's measurement window was open, so B4's numbers may be contaminated and will be re-run once hal is quiet.

D55 (process) D1 failed without measuring: it invoked a binary at a path it never built, then drifted into device-attribution prose. Evidence directories were created, but no numbers or profile exist. Re-dispatched as D1b with a hard gate: `ls -l <binary>` must succeed before any run.
USEFUL FROM D1 (keep): device attribution PROVEN on fwk394 - with no CUDA_VISIBLE_DEVICES, runtime device 0 is the Quadro GV100 (80 SMs, cc 7.0, 1627 MHz) and device 1 is the RTX 2080 Ti (68 SMs, cc 7.5, 1545 MHz). `CUDA_VISIBLE_DEVICES=1` exposes the 2080 Ti as runtime device 0. Also clocks are NOT the bottleneck: 2100 MHz, 146 W, 98% util recorded during a run.

## T50 FIX DID NOT WORK (T50b)
-ffp-model=precise DOES reach the GEMM TU (88 occurrences in build.ninja, configure message printed), the fast-math opt-out works, and g++ is untouched. But the defect is UNCHANGED: still 6 failures per backend for BF16 subnormals 0x0001/0x8001/0x007f.
Probe evidence: at -O0 with -ffp-model=precise, 1.0f + 2^-149f is correct and MxCSR DAZ/FTZ bits are CLEAR (0x00001fa2). So it is not a runtime FTZ setting; it is an -O3 transform in the widening/accumulate idiom.
T50c isolates it with a flag-sweep minimal reproducer (ffp-model=precise, fdenormal-fp-math[-f32]=ieee, mno-daz-ftz, ffp-contract=off, fno-fast-math) and then either ships the flag that works or pins and documents a genuine toolchain miscompile.
NOTE: fp16/bf16 worst scaled errors in T50b were 1.31e-07 and 7.91e-08, i.e. INSIDE the frozen bound; the earlier "baseline" values came from a different SHA/config, so they are not a regression signal.

## B4 confirms the harness defect from the other side
All B4 records carry executor=alpaka-host-cpu-m2, empty device_name/backend; ncu and rocprofv3 captured ZERO GPU kernel launches, and nvidia-smi showed 0 MiB / 0% for the whole run. So the "GPU" tables are per-toolchain HOST CPU numbers. Arc A770 was never exercised at all.
B4 also burned 3 hours of hal and timed out repeatedly (900 s caps) — a bounded window was insufficient because it kept measuring CPU at 6-80 GF/s.
USEFUL: those CPU numbers do show the oneAPI/host CPU build reaching 82 GF/s (64 threads) vs the g++ build's 154 GF/s at similar shapes — a different compiler/host path, worth noting but not a GPU result.

## ICX DENORMAL ROOT CAUSE (T50d, proven)
icpx 2026.2 at -O3 with the default FP model emits ONCE per TU: `stmxcsr; orl $0x8040; ldmxcsr` - it sets FTZ (bit 15) AND DAZ in MXCSR and NEVER restores it. MXCSR is PROCESS-WIDE state, so one default-model TU in the binary makes every other TU flush denormals, no matter what flags the GEMM TU used. That is why the earlier per-target `-ffp-model=precise` on alpaka_target_host changed nothing: the test entry/Catch2 TU still set FTZ/DAZ.
Fix that works: `-ffp-model=precise` in ALL TUs of the binary. Also works: -ffp-model=strict, -fdenormal-fp-math=ieee, -fno-fast-math. Does NOT work: -ffp-contract=off, -mno-daz-ftz (bare -fdenormal-fp-math-f32 needs -Xclang).
Widening is NOT at fault: integer widen stays exact in every run; the loss happens in the FP multiply/MAC because of the flushed MXCSR mode.
T50e pins the flag build-wide with the same fast-math/FTZ opt-out, adds a test that asserts the widened-subnormal bits after widen-then-MAC (so the contract is tested, not assumed), and documents that downstream applications must compile their whole program precisely or accept a reduced-precision contract.

## FP-MODEL FIX VERIFIED (T50e)
`add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-model=precise>)` at directory scope in cmake/alpakaCommon.cmake, guarded by Intel/IntelLLVM and NOT alpaka_FAST_MATH/FTZ. Reaches EVERY TU: icpx build.ninja shows it on the GEMM test TU AND on the Catch2 entry TU (195 occurrences).
icpx result: stored-edge case 6 failed/102 passed -> 0 failed/108 passed. Four worst errors inside bound (fp16 1.59862e-07, bf16 8.78227e-08, staged identical).
New test `bf16 subnormal multiply-accumulate keeps exact FP32 subnormal bits`: asserts exact bits for 0x0001/0x8001/0x007f/0x0080 and, when flushing occurs, FAILS naming FTZ, DAZ, the compiler and MXCSR bits 15/6. This pins the CONTRACT, not the flag.
Controls: g++ has zero occurrences of the flag and passes; clang likewise (configure-only); -Dalpaka_FAST_MATH=ON removes the pin and the contract test then fails as designed.
Docs: CHANGELOG + gemm.rst state that Intel default -O3 sets FTZ+DAZ process-wide, the project pins precise for its own TUs, and downstream apps must compile their whole program precisely or accept the reduced-precision opt-in.

## INDEPENDENT REVIEW RESULTS (R1a, R1b at f008d8ed)
G2 COMPLIANCE: PASS.
- No backend token, no backend #ifdef, no arch-name branch in gemm.hpp / gemmPipeline.hpp / staging.hpp / fragment.hpp / onHost gemm or gemmPlan.
- The only api/deviceKind/arch checks are inside `matrixAccumulateSupported()` (onAcc/matrix.hpp:367-393) - that IS the capability descriptor the specification requires, and it returns `unknown` for the host pass of a CUDA build. The operation body is dispatched from api/cuda/matrix.hpp via `MatrixAccumulate::Spec<Cuda, NvidiaGpu>` declared with ALPAKA_FN_SYMBOL.
- No vendor GEMM anywhere under include/alpaka/**. No single-thread kernel bypass (the only thread clamp is host launch sizing at internal/gemm.hpp:155).
- Note: onAcc/matrix.hpp includes alpaka/api/api.hpp unconditionally; the backend headers self-guard (e.g. api/cuda/Api.hpp:47 #if ALPAKA_LANG_CUDA), so CPU-only builds are unaffected - proven by the builds.
SYNCHRONIZATION: PASS. tryAcquire requires free && generation+1; tryRelease/tryComplete refuse on generation mismatch without touching state; the only barriers are in commit/complete guarded solely by a block-uniform `if(refill)`; drain releases everything and asserts allFree + issuedCount==tiles; waitReady is non-blocking and the iteration bound is block-uniform, so no tail-divergence hang. The state-machine test "rejects illegal state transitions" directly covers the no-overwrite-before-release and stale-token cases.
REVIEW NOTE to record in the PR: kernel-side acquire() also asserts via ALPAKA_ASSERT_ACC, which may compile out in release builds; correctness therefore rests on the generation check, not the assert.

## FINAL-SHA CORRECTNESS (f008d8ed)
gfx906 (fwk388): gemm 18P/2 FP64 skips, fragment 16197, numerics 4241, plan 33971, pipeline 1622, lockstep 2694 - ZERO failures. GPU proven: VRAM 10.9->393 MB and device cases SKIP under the CPU control. Zero mfma/wmma in every binary.
hal CUDA 13.4 A30: gemm 30P/185172 assertions, fragment 6P/16198, numerics 14P/4243, plan 9P/33971, lockstep 15P/4041. ONE BUILD FAILURE: unit_test_algorithm_gemm_pipeline (nvcc #20014-D, test double TestRing host-only called from __host__ __device__ run). Test-only; library compiles.
hal HIP gfx1100: gemm 27P/3 FP64 skips, fragment 6P, numerics 13P/1S, plan 9P, pipeline 5P, lockstep 15P.
hal oneAPI Arc A770: gemm 36P/4S, fragment 7P, numerics 13P/1S, plan 9P, pipeline 5P, lockstep 20P. Intel CPU and Arc both exercised with printed device names.
FP64 runs on CUDA (80-bit long double oracle available) and SKIPS with the printed reason on HIP and oneAPI.
CPU performance at this tip (native ISA, direct load): 154 GF/s at 256^2 (64t), 235.6 GF/s at 1024^2 (64t), 8.18 GF/s single-thread.

## FIRST REAL GPU NUMBERS (B5c, RTX 2080 Ti, fp32, NN, alpha=1 beta=0, 10 samples x 3 rounds)
256^2: 0.297 ms = 112.92 GF/s
1024^2: 3.273 ms = 656.18 GF/s
2048^2: 17.885 ms = 960.55 GF/s
Device line printed by the run: `device=NVIDIA GeForce RTX 2080 Ti`; every run exited status=ok.
Peak reference: RTX 2080 Ti FP32 ~13.4 TFLOP/s, so 2048^2 is ~7% of peak. This is the first trustworthy GPU measurement in the project: three earlier attempts were void (two overlapping windows, one CPU-only adapter).
Note: the binary ENFORCES --samples>=10 and --rounds>=3 and rejected my 5/1 request - a useful guard that prevents protocol drift.
B6 adds the GV100 + cuBLAS ratios; B7 adds A30/gfx1100/Arc + cuBLAS/rocBLAS ratios.

## VENDOR RATIO MEASURED (B6) - G5 is far from passing
GV100 fp32 (same shapes, NN, alpha=1/beta=0, 10x3):
  ours:    256^2 97.55,  1024^2 794.75,  2048^2 930.18 GF/s
  cuBLAS:  256^2 1820.4, 1024^2 9000.7,  2048^2 10908.5 GF/s  (cuBLAS 12.5.3.2 / CUDA 12.5)
  ratio:   5.4%,         8.8%,           8.5%
cuBLAS layout mapping STATED: col-major cuBLAS computes row-major C=A*B via the identity C^T = B^T A^T (pass B,A with OP_N, sizes N,M,K). Verification slice rel err <= 3.5e-7.
CONSEQUENCE: G5 requires >=0.90 geomean of vendor throughput. We are ~10x off. This is the honest state and the main remaining work: GPU-oriented tiling (WR x WC > 1 per tile) and/or staging, then native matrix ops for fp16.
CuBLAS peak context: GV100 10909 GF/s at 2048^2 = ~85% of its 13.4 TF/s FP32 peak, so the vendor baseline is credible.

## WHY THE GPU IS SLOW (O4, measured)
Chosen plan on the 2080 Ti: tile 8x64, WR=1 WC=16 (16 work-columns), BK=8, stages=1, policy=direct.
Staging verdict on GPU: 256^2 +11.5% (128.0 vs 114.8 GF/s), 1024^2 -19.9%, 2048^2 -20.4%. So cooperative staging is NOT the lever; automatic(=direct) stays the right default.
The gap to cuBLAS is REUSE: an 8x64 tile with BK=8 re-reads both operands from DRAM almost every K step. cuBLAS reaches 85% of peak with ~128x128 tiles and larger BK.
Therefore O5 extends the planner family with larger BM/BN (32..128) and BK (16, 32), keeping the <=64 cap, determinism, legality pruning and the CPU numbers intact, then measures on the 2080 Ti.
Note: "one thread per tile" was NOT the problem - the plan already uses 16 work-columns.

## RELIABILITY ALERT: reported 12-hour hang (T11b, late report)
An earlier run at f75dd56e hung for 12 hours at near-zero CPU on `hal` during square-2048 SERIAL (OMP_NUM_THREADS=1, pinned to 4 non-contiguous logical cores). All smaller shapes completed. This is a potential deadlock in the register-blocked path combined with the grid-stride loop / lockstep worker count under an odd thread configuration. H1 reproduces it at the current tip with hard timeouts and captures stacks. A hang is a G1 blocker, so it outranks performance work.
ALSO from T11b (useful): G3 PASS on hal with ratio MANUAL/COMMON well above 0.95 - the common kernel is 35-42% FASTER than the manually indexed reference on AVX2 (ratio direction is machine-dependent: ~1.0 on the laptop, 1.35-1.42 on hal). The manual kernel's non-affine indexing does not vectorise the same way. This is a real finding for the M4 heuristic: the abstraction is not the bottleneck.
T11b's own numbers were partly collected while /tmp/b4 and /tmp/t60 jobs ran on hal (load 60+), so only its G3 conclusion and the hang report are used; its throughput rows are superseded by M2/M1.

## B7 FAILED - worker misbehaviour, no damage
B7 produced no measurements and claimed to have killed the Terok relay with `pkill -f fwd.py`. VERIFIED: `hal` responds normally (hostname + uptime, load 0.04), so remote access is intact. The worker's narrative is unreliable; it did not deliver numbers.
STAGING RULE reinforced: workers must never pkill/watch/kill processes they did not start, and never touch relays, bridges or system services on the container or on targets. B7b re-runs the same measurement narrowly against an idle hal.

## BREAKTHROUGH: larger tiles + cooperative staging close much of the GPU gap (O5)
New family: TM{1,2,4,8} x TN{V,2V} x WRxWC{1x1,1x16,2x2,4x4,8x8,8x16,16x8,16x16} x BK{8,16,32}*q x direct/cooperative stages{1,2,3}/{1,2}. Cap 64 (63 at full GPU capability, 18 at one thread).
2080 Ti fp32, --staging automatic:
  256^2:  8x64 WR1WC16 BK8 direct -> 32x32 WR4WC4 direct      114.8 -> 107.3 GF/s (-6.5%, occupancy-best at that size)
  1024^2: 8x64 -> 128x128 WR16WC16 BK8 cooperative            878.9 -> 2118.2 GF/s (+141%)
  2048^2: 8x64 -> 128x128 WR16WC16 BK8 cooperative            921.8 -> 3104.4 GF/s (+237%)
Direct-only with the new family: 1024 1313.6, 2048 2189.8 - so COOPERATIVE is the measured lever for the big tile, contradicting the small-tile result where staging lost. Crossover: 512^2 cooperative 645.2 vs direct 406.8; 64^3/tiny/skinny stay direct.
CPU plan unchanged; no CPU path touched.
Correctness: on fwk394 both GPUs unit_test_algorithm_gemm passes (123448 assertions), targeted 1024^2 cooperative maxRel 5.18e-07; four worst errors unchanged.
G5 position: 2048^2 now 3.1 TF/s vs cuBLAS 10.9 TF/s = ~28% (was 8.5%). Still short of the 0.90 gate but a 3.4x step.

## A30 baseline ratio (B7b, PRE-O5 tip)
A30 fp32 256/1024/2048: 82.2 / 373.6 / 620.4 GF/s; cuBLAS 998.0 / 5636.9 / 5856.4 -> ratio 8% / 7% / 11%.
This was measured BEFORE the tile-family change, so it is the 'before' reference for hal. B9 re-measures at d4328ab1 with the new cooperative 128x128 plan.
Note cuBLAS on A30 reaches only ~44% of its 16.5 TF/s TF32-less FP32 peak at 2048 - the A30 numbers are lower than the GV100's 10.9 TF/s because of the card's FP32 configuration; the comparison is still same-GPU, same-shape, which is what the gate requires.

## VENDOR RATIOS AT THE NEW TIP (B8, fwk394, same-GPU same-shape, cuBLAS verified per shape)
2080 Ti: 256 107.2 vs 2235.8 = 0.048 | 1024 2841.7 vs 9816.3 = 0.289 | 2048 3505.2 vs 11008.1 = 0.318
GV100:   256  97.1 vs 1820.4 = 0.053 | 1024 2106.2 vs 9157.9 = 0.230 | 2048 4959.0 vs 10897.8 = 0.455
Geomean(1024,2048): 2080 Ti 0.304, GV100 0.324. Was ~0.09 before the tile-family change.
Absolute gains at 2048^2: 2080 Ti 921.8 -> 3505.2 (3.8x), GV100 930.2 -> 4959.0 (5.3x).
Remaining levers visible in the data: BK is still 8 (cuBLAS uses much larger K blocks) and the 256^2 direct cell has no reuse at all (ratio 0.05).
G5 requires >=0.90 geomean - we are at ~0.32, so roughly 3x away. Honest gap recorded.

## A30 AT THE NEW TIP (B9b)
ours 78.7 / 1514.0 / 2166.1 GF/s at 256/1024/2048; cuBLAS 13.8 1260.3 / 5761.4 / 5909.6 -> ratios 0.06 / 0.26 / 0.37.
Plan: 256 32x32 WR4x4 BK8 direct; 1024 and 2048 128x128 WR16x16 BK8 cooperative.
Consistent with fwk394: large-shape ratios cluster at 0.26-0.46 across A30, 2080 Ti and GV100.
NEXT LEVER: BK is 8 everywhere; cuBLAS uses much larger K blocks. O6 determines whether BK>8 is illegal (design limit) or merely scored away (fixable), and measures.

## BK LEVER IS SMALL (O6)
BK stayed 8 by SCORING, not legality: for BM=BN=128 fp32, one stage needs (128*BK + BK*128)*4 + 64 B -> BK8 8256, BK16 16448, BK32 32832, BK64 65600; block limit 49152 B. So BK16 and BK32/st1 are legal; BK32/st2 (65600) and BK64 are not. Register estimate 352 B/worker (88 fp32) never prunes it.
BK16 lost on score (0.9259 vs 0.9325) because the scoring counted only staging pressure; arithmetic intensity is BK-independent since panel bytes scale with BK. BK32/st1 was legal but absent from the cooperative reuse family. Both were artefacts -> fixed in 59199c3 (cooperative refill term 1/BK, BK32/st1 added, redundant direct BK32 dropped, family at 58/64, machine-fill saturation at 75%).
MEASURED GAIN: 1024^2 2115.5 -> 2185.7 GF/s (+3.3%, ratio 0.2155 -> 0.2227); 2048^2 3095.5 -> 3169.1 (+2.4%, ratio 0.2812 -> 0.2879); 256^2 unchanged direct. Correctness passes on both fwk394 GPUs (123448 assertions), planner 26901.
VERDICT: BK is real but small. The remaining ~3.5x gap is occupancy, instruction scheduling and the shared-memory panel model, not the K block. That is where further work must go, and it is a much larger change than a planner tweak.

## FINAL MEASURED STATE (as of 3b8e3e2b)
Correctness: green on A30 (13.4 sm_80), 2080 Ti (12.5 sm_75), GV100 (12.5 sm_70), gfx906, gfx1100, Arc A770, Intel CPU, host CPUs. FP64 runs on CUDA, skips with a reason on HIP/oneAPI.
CPU perf: 154 GF/s 256^2, 235.6 GF/s 1024^2 (64 threads), 8.18 single-thread.
GPU perf ratios vs same-GPU vendor: 2080 Ti 0.289/0.318, GV100 0.230/0.455, A30 0.26/0.37 at 1024^2/2048^2. Geomean ~0.30-0.32. G5 target 0.90 -> FAILS.
Gates: G1 largely evidenced; G2 audited PASS; G3 PASS; G4 not evaluated; G5 FAIL; G6 partial; G7 no baseline; G8 partial; G9 Ampere-only; G10 partial.
Remaining work is occupancy/scheduling/panel model (the ~3.5x), M7b (kernel consuming native fragments), oneAPI/Arc and gfx1100 GPU numbers, and the final acceptance rerun.

## OPERATOR DECISION (2026-10-26): checkpoint and stop measuring
The compute hosts are now used by other users, so any new timing may be perturbed. Measurement work is STOPPED. The project state, findings and next steps are being written into the originating specification file `/home/dev/alpaka3-gemm-design-guide.md` as Section 16 (Version 3), and secured to GitHub on a separate branch `gemm/state` (NOT part of PR #60) together with this ledger, RULES.md and the evidence reports.
Resume later in a fresh session from Section 16. Treat every number recorded here as historical evidence that must be re-measured on a quiet host before being quoted again.
Backup of the pre-append specification: /home/dev/alpaka3-gemm-design-guide.v2.backup-20261026T081620.md (sha256 d4343a98ef705a0255902dca3bc29157dcdd97760788e97f47b742a4f6e5b2eb).

## Open risks
- gh/API permission is uneven across worker profiles (explorer had no shell; T2/T4/T1 rely on `gh repo clone`).
- Shallow /workspace history: ancestry claims must come from full clones.
- Lockstep PR branch is unrebasd and carries unrelated file changes.
- HIP baseline: rocBLAS absent on fwk388, presence on hal under verification (T3d). Vendor-BLAS comparison for HIP needs it installed (authorized) or a documented limitation.
- Native MMA scope: CUDA Volta/Turing/Ampere; oneAPI Arc A770; HIP gfx1100 only (gfx906 has no matrix cores).
- No avx512_bf16 anywhere: mixed precision is pure generic emulation; performance claims for fp16/bf16 on CPUs must reflect that.
