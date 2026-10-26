# Standing rules for every worker in run gemm-20261025

0. **REMOTE-FIRST EXECUTION (operator directive).** Run as much as possible on the
   computeMCP targets, in parallel across different targets. This container is for
   editing, git, pre-commit, and light inspection only. Every build, every test run
   and every measurement belongs on a remote target. Distribute work: `hal` for the
   CUDA+HIP+oneAPI correctness matrix, `fwk394` for CUDA 12.5 / Turing / Volta,
   `fwk388` for gfx906 HIP fallback, `rosi_v100` for parallel sm_70 work while the
   grant lasts. Never put two measurements on one target; builds may run on
   different targets concurrently.
1. **KEEP THE TEROK CONTAINER LIGHT.** It is a laptop. Local work is limited to
   targeted builds and targeted tests. Use `-j4` maximum for compiles and ctest,
   build only the targets under test (never `all`, never the full suite unless a
   milestone exit demands it), and prefer computeMCP for anything heavy. Do not
   run full ctest sweeps locally in parallel with other workers.
1b. **NO PERFORMANCE MEASUREMENT IN THE TEROK CONTAINER.** Terok runs on a laptop.
   Timings, throughput, GFLOP/s, profiler runs and benchmark comparisons run ONLY on
   computeMCP targets (`hal`, `fwk394`, `fwk388`). Local container work is limited to
   configure, build, unit tests, sanitizers, static analysis and codegen reports.
   Any number produced locally is INVALID as evidence and must not reach the PR.
2. `/workspace` is read-only. Never edit it, never unshallow it, never push from it.
3. Editors work in their own `/tmp` clone or copy. Exactly ONE worker holds the
   canonical tree at a time. Publication never overlaps integration.
4. Publish before hardware validation: every GPU number cites a published SHA.
5. `git push` is refused. Publish via the GitHub Git Data API with EXPLICIT
   author/committer = human identity and the bot `Co-authored-by` trailer as the
   final message line. Verify trees and authors after publishing.
6. **EXCLUSIVE BENCHMARK WINDOWS.** Never run two measurements at once on the same
   target, and never build/test on a target while a benchmark runs there. One worker
   owns a target's measurement window; everyone else waits. Record the start/end and
   the load average at both ends. Parallel jobs make every number wrong, including the
   ones from the quiet run.
7. Benchmark truth: useful `2*M*N*K`; >=10 samples over >=3 rounds; median plus
   spread; report load average, governor, clocks, thread count, toolkit versions.
   Never fabricate, never average across devices or precision modes, never mark an
   unavailable cell as a pass.
8. Tolerances are frozen (validation/numerics/TOLERANCE.md). A failing case is a
   kernel or plan defect, never a tolerance to relax.
9. Vendor libraries are benchmarks, never the implementation. No backend GEMM body,
   no architecture-name branch, no single-thread special path in the kernel.
10. Common source only. Specialized intrinsics go behind `ALPAKA_FN_SYMBOL` dispatch
   with a tested fallback.
11. No measured data, plots or profiler output in the repo. Evidence lives under
    /tmp/alpaka-orchestration/gemm-20261025/.
12. Delegation is flat: workers never delegate; new work returns to the orchestrator.
13. Prose for humans passes stop-slop; GitHub links use full 40-character SHAs.
