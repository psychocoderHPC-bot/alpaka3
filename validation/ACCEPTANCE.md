# GEMM acceptance draft (M8a)

Status: draft, not accepted.
Source under test: `6e822093be2c4799d8da09d2d509bfd90c828199` (`gemm/common-source`).
Machine-readable records: `validation/gemm-requirements.yaml` (33 records).
Evidence root: `/tmp/alpaka-orchestration/gemm-20261025/evidence/` and the run ledger
`/tmp/alpaka-orchestration/gemm-20261025/LEDGER.md`.
This draft changes records only. No code, build, benchmark or toolchain run was done for M8a.

## Headline verdicts

| Verdict | Result | Basis |
| --- | --- | --- |
| FUNCTIONALLY_COMPLETE | NO | Correctness passes on CUDA sm_70/75/80, HIP gfx906 and oneAPI for fp32, fp16 and bf16, but FP64 is an explicit SKIP on HIP and oneAPI (no extended `long double`). No record reaches `passed`: two required review findings are open, no run exists at the final include/alpaka change, and per-case samples are absent. |
| PERFORMANCE_ACCEPTED | NO | The only valid performance data is CPU-only (M1, G3). There is no GPU performance evidence and no matched vendor baseline, so G4, G5, G6 and G7 are not run. |
| NATIVE_FEATURE_COMPLETE | NO | A30 SASS shows `HMMA.16816.F32` and `HMMA.16816.F32.BF16`, but there is no native-mode performance evidence and GEMM does not consume the warp matrix contract yet (M7b pending). |

## Record counts

| passed | failed | unsupported | blocked | not_run |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 23 | 10 |

`blocked` means accepting evidence exists but at least one required target, mode, review or
final-SHA rerun is missing. `not_run` means no accepting evidence exists; G5 and G6 list
the invalidated B3/B4 reports only to record why no GPU performance result is admissible.
FP64 on HIP and oneAPI is a recorded capability limitation inside the blocked correctness
records, not an unsupported status for a whole record.

## Gate summary

| Gate | Status | Evidence | Why it is not passed |
| --- | --- | --- | --- |
| G1 Correctness | blocked | `I6-report.md`, `T39a-report.md`, `remote-hal/*` logs, `LEDGER.md` | CUDA sm_70/75/80 and both host CPUs pass every suite (gemm 20/20, plan 8/8, numerics 14/14, fragment 5/5, pipeline 4/4). HIP gfx906 and oneAPI pass fp32/fp16/bf16. FP64 SKIPs on HIP and oneAPI because `long double` is 53-bit. No rerun at 6e822093. |
| G2 Common source | not_run | none | No source-policy review or preprocessor-guard scan was performed; the reviews directory is empty. |
| G3 Abstraction overhead | blocked | `T11c-g3.md`, `T11c-1792950900/summary.json`, `T11c-raw/g3.*`, `T11-codegen/abstraction-bench.log` | 12/12 assertions pass and ratios are 1.023/1.064/1.097 (host) and 0.997/1.066/1.097 (numa), all at or above 0.95. The run used the hal host CPU under `CpuOmpBlocks`, which is not a named required target. Six required targets and the small-case latency analysis are missing. |
| G4 Heuristic quality | not_run | none | No held-out candidate sweep against a bounded-family oracle exists, so neither the 0.95 geomean nor the 0.85 per-case threshold can be evaluated. |
| G5 Competitive large GEMM | not_run | `B3-report.md`, `B4-conditions/summary.txt`, `M1-OUTPUT.md` | No GPU performance evidence. B3 and B4 used a CPU-only adapter (`executor=alpaka-host-cpu-m2`) and are invalid. rocBLAS gfx1100 numbers used a separate harness. cuBLAS and oneMKL were never measured, so no matched-vendor ratio exists. |
| G6 Broad workload | not_run | `B3-report.md`, `B4-conditions/summary.txt` | No per-target, per-mode, per-family candidate-to-vendor ratios exist. The 0.50 blocking floor cannot be checked. |
| G7 Regression | not_run | none | No regression benchmark against a previously accepted common implementation exists. |
| G8 Resources | not_run | none | No launch-legality, scratch-bound, packing/workspace-cost or spill report exists. |
| G9 Native claims | blocked | `LEDGER.md` (M7a), `I6-report.md` | A30 instruction evidence exists for fp16 and bf16. Native performance evidence is absent, GEMM does not consume the warp contract yet, and sm_70/75, gfx906 and gfx1100 select the fallback. |
| G10 Maintainability | blocked | `LEDGER.md`, `I6-report.md`, `T11c-conditions.md` | Plan diagnostics, the frozen tolerance document, the 64-candidate cap and reproduction commands exist. Independent review and an end-to-end reproduction record at the final SHA are missing. |

Section 8 families follow the same split. `S8-STRUCT-ownership`, `S8-STRUCT-boundaries`,
`S8-STRUCT-planner-facts`, `S8-STRUCT-async-lifetime`, all five `S8-NUM-*` records,
`S8-MEM-canaries` and `S8-PIPE-device` are blocked by the FP64 skip, missing final-SHA
rerun or missing per-target evidence. `S8-STRUCT-simd-slots` also carries the open
zero-worker guard finding. `S8-PIPE-state-machine` carries the open public-documentation
finding. `S8-MEM-sanitizers` has clean host ASan/UBSan runs, but no required target, no
TSan and no GPU sanitizer. `S8-MEM-stress` is not run.

## Acceptance checker

Run:

```text
python3 validation/check_acceptance.py \
  --requirements validation/gemm-requirements.yaml \
  --run-root /tmp/m8a-work/acceptance-run \
  --head-sha 6e822093be2c4799d8da09d2d509bfd90c828199 \
  --repo <checkout> \
  --json-report /tmp/m8a-work/acceptance-report.json
```

Result: exit 4, 1270 findings. Categories: 875 `missing-sample`, 179 `target-not-run`,
169 `stale-invalidate-sha`, 33 `not-passing`, 9 `missing-threshold-metric`,
3 `vendor-unavailable`, 2 `unresolved-review-finding`. There are no schema errors,
failing thresholds or unapproved skips. The checker's per-requirement summary is
30 `not_run` and 3 `vendor_unavailable` (G5 and both G6 records); no requirement is passed.

The external run root contains `acceptance.json` generated from the records. It holds no
`samples.json`, because the correctness evidence exists as aggregate Catch2 logs and
reports, not as per-case sample records. The 875 `missing-sample` findings are the checker
stating that gap.

Exit 4 comes from `stale-invalidate-sha`. The newest `include/alpaka` change is the tested
head `6e822093` itself, so every earlier evidence SHA predates it. The checker requires a
final-SHA rerun; that run does not exist. This is a real acceptance gap, not a schema
error. Where one record mixes SHAs, `allow_mixed_shas: true` is set and the individual
SHA is still recorded.

## Evidence limits and unmapped items

- No GPU performance evidence exists. Every earlier GPU performance table is invalid because
  the benchmark adapter selected a host CPU device. The adapter has since been fixed, but no
  new GPU measurement was made.
- No matched vendor ratio exists. The rocBLAS gfx1100 measurement (3516 GFLOP/s at 512^3,
  8242 at 1024^3) used its own harness. cuBLAS and oneMKL baselines were not measured.
- The hal EPYC 7452 host CPU used for M1 and G3 has no name in `required_targets`.
  `hal/oneapi-cpu` is the Intel OpenCL CPU device, a different executor, so the CPU
  measurements cannot be mapped to that target without overstating them.
- B3 and B4 are retained as invalidation evidence only. Their numbers must not be quoted as
  GPU results.
- The reviews directory is empty. All review verdicts remain pending, and the two recorded
  major findings stay open.
- FP64 SKIPs on HIP and oneAPI are a limitation with a printed reason. They are not passes,
  and they are not counted toward any gate.
