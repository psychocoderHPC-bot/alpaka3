# Task D2 — ncu profile of the 2080 Ti GEMM kernel, tip 3b8e3e2b
# State: PAUSED, pre-build. Nothing built, nothing profiled, nothing pushed.
# Budget: 30 min once resumed. Profiling only: no code or policy changes,
# no killed processes.
# Shared-host rule (2026-10-26): fwk394 is now used by others too. Every step
# starts with an nvidia-smi idle check on the 2080 Ti (CUDA index 1 on that
# box); if it is occupied, stop, report, do NOT kill, do NOT wait for it.

## 1. Goal and hypotheses (in priority order)
Profile ONE case: `square-2048` fp32, plan 128x128 WR 16x16 BK 32/st1 cooperative
(best 2080 Ti kernel at tip 3b8e3e2b: 3169 GF/s vs cuBLAS 11008, ratio 0.288).
ncu shall separate, and the report shall name ONE dominant limiter:
  (a) occupancy limited by shared memory or registers,
  (b) shared-memory bank conflicts in the panel layout,
  (c) not enough independent FMA chains per thread (dependency stalls),
  (d) global load inefficiency on the panel fill,
  (e) synchronization stalls at the stage boundaries.

## 2. Environment (verified 2026-10-26, all read-only, nothing started)
- Host `fwk394` (only box with the 2080 Ti; now shared by other users):
  GPU0 = RTX 2080 Ti (sm_75), GPU1 = Quadro GV100 (sm_70).
  `which ncu` = /usr/local/cuda/bin/ncu, Version 2024.2.1.0 (build 34372528)
  public-release; driver 555.52.04. `ncu --query-metrics` lists 388 lines matching
  smem/bank patterns. nvidia-smi: both GPUs 0 % util, no compute apps at check time.
- Host `hal`: A30 + RX 7900 XTX + Arc A770, no 2080 Ti. No gemm_bench build left;
  an old `b8_all.sh` script lives at /tmp/b8local/b8_all.sh (local container path,
  see Notes).
- Code: parent repo `/workspace` (bot base, branch `dev`, HEAD ded79f8, unmodified
  in this session). The GEMM factorial lives under
  `/tmp/alpaka-orchestration/gemm-20261025/canonical`, clean worktree, HEAD
  3b8e3e2b5072d10824bd9c0434df6c314b66240b, branch gemm/common-source.
  Remote ref refs/heads/gemm/common-source = 3b8e3e2b... (ls-remote confirmed
  2026-10-26): NO push needed; `git push` from canonical would be a no-op.
- tip 3b8e3e2b = successor of d4328ab: "perf(gemm): score cooperative K-block
  refills and allow BK 32" (refill scoring 1/BK, BK32/st1 added to the cooperative
  reuse family, redundant direct BK32 dropped; tile family 58/64, machine-fill
  saturation 75%).
- State at this tip (measured O6/B8, 2080 Ti fp32, --staging automatic):
  256^2 107.3 / 1024^2 2185.7 / 2048^2 3169.1 GF/s -> ratios 0.048 / 0.289 / 0.288
  vs cuBLAS 2235.8 / 9816.3 / 11008.1. Canonical goal is geomean >= 0.90 of vendor
  (G5); 2048^2 at 0.288 is the gap D2 interrogates.
Plan rows: 256 = 32x32 WR4x4 BK8 st1 DIRECT; 1024/2048 = 128x128 /16x16 BK8/st1
  cooperative (pre-BK32 family). The BK32/st1 row is the one D2 must profile
  (the spec names it explicitly).

## 3. Phase 1 — build at tip on fwk394 (target <= 20 min)
1.0  mkdir -p /tmp/d2  (computeMCP_file_mkdir, target fwk394).
1.1  Get the 3b8e3e2b tree onto fwk394. PREFERRED (works because the branch is
     published to the bot fork): `git clone https://github.com/psychocoderHPC-bot/alpaka3.git /tmp/d2/src && git -C /tmp/d2/src checkout 3b8e3e2b5072d10824bd9c0434df6c314b66240b && git -C /tmp/d2/src rev-parse HEAD`.
     If clone is blocked: tar the canonical tree WITHOUT .git, exclude build
     dirs, and computeMCP_file_upload the SINGLE tarball; after either path
     verify with `git rev-parse 3b8e3e2b...` on the receiving side.
     Do NOT reuse /tmp/opencode/B9/src or /tmp/b9/src (or any /tmp/src-*/
     STRIP tree from prior MMA tasks): those are at 3ad5448f / 6f13700, not
     the D2 tip 3b8e3e2b. Re-clone or re-uploading the correct SHA is mandatory.
1.2  Configure, benches only, both arches (use the default Unix Makefiles
     generator unless Ninja is actually installed on the box; check first):
     cmake -S /tmp/d2/src -B /tmp/d2/build \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_CUDA_ARCHITECTURES="70;75" \
       -DALPAKA_BENCHMARKS=ON -DALPAKA_DEP_CUDA=ON -DALPAKA_DEP_OMP=OFF \
       -DALPAKA_HOST_Cpu=ON -DALPAKA_EXEC_CpuSerial=ON \
       -DALPAKA_GEMM_NATIVE_ISA=ON
     (B8's comparable run used -G Ninja; add it back if `which ninja` is
     non-empty. B8 also used alpaka_COMPILE_PEDANTIC=ON, which makes warnings
     errors for the bench TU; only add it if the build is clean, otherwise drop
     it and note the deviation. GEMM_NATIVE_ISA=ON keeps host -march=native for
     simd=4 and leaves the device pass untouched.)
1.3  cmake --build /tmp/d2/build --parallel <cores>  (fwk394 has 28 usable
     cores; B8 used -j32 on a 32-core box and took 6:18 wall; expect the same
     order. Do NOT exceed the box width or the shared host will be pushed.)
1.4  Verify: /tmp/d2/build/benchmark/gemm/gemm_bench exists; cuobjdump -res-usage
     shows .target sm_70 AND .target sm_75; no OpenMP runtime links.
     No ctest at this step: profiling is the objective.

## 4. Phase 2 — ncu on ONE case (target <= 8 min)
2.0  nvidia-smi idle check (mandatory before every run on this shared host):
     nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used
               --format=csv,noheader
     Proceed only if the 2080 Ti (CUDA index 1 = name RTX 2080 Ti under
     CUDA_VISIBLE_DEVICES=1; GPU 0 on this box is a different card) shows 0 % util
     and ~0 MiB. Record the snapshot (index, name, util, memGiB) in prof.txt.
2.1  Profile the one case, the one GPU. Verified harness CLI (source: in-tree at
     3b8e3e2b, benchmark/gemm/src/gemm_bench.cpp; flags: --manifest, --output-dir,
     --revision, --target, --device-kind, --executor, --staging
     automatic|direct|cooperative, --subset all|ci-fast, --filter TEXT (substring
     match; pass `square-2048` so only that case id is kept), --samples N,
     --rounds N, --warmups N, --max-cases N, --smoke, --list-cases).
     The non-smoke guard (line 708) is `if(!args.smoke && (args.rounds < 3u ||
     args.samplesPerRound < 10u)) usageError(...)`. So the spec's
     `--samples 1 --rounds 1` IS rejected; the minimum valid non-smoke probe is:
        CUDA_VISIBLE_DEVICES=1 /tmp/d2/build/benchmark/gemm/gemm_bench \
          --filter square-2048 --samples 10 --rounds 3 --warmups 1 --staging automatic
     For the recorded plan (128x128 WR 16x16 BK 32/st1 cooperative) pass
     --staging cooperative to force it and remove any ambiguity. ncu replays each
     kernel launch internally; the 30 launches (3x10) are NOT 30 wall runs, they
     are 30 kernel replays the ncu will capture identically. OK to reduce for
     profile brevity: use --max-cases 1 to select exactly one case and rely on the
     single-GPU path, because the rest of the measurement wall time will not
     affect ncu metric capture (same kernel, same launch config, same 30
     replays). But DO NOT drop below rounds=3 x samples=10 unless --smoke, which
     the spec does not want (smoke runs a placeholder functor, not the real kernel).
     Run exactly this:
        CUDA_VISIBLE_DEVICES=1 ncu --target-processes all \
          --section SpeedOfLight --section Occupancy \
          --section MemoryWorkloadAnalysis \
          --section SchedulerStats --section WarpStateStats \
          --section LaunchStats \
          -f -o /tmp/d2/prof \
          /tmp/d2/build/benchmark/gemm/gemm_bench --filter square-2048 \
            --samples 10 --rounds 3 --staging cooperative
     --max-cases 1
     Note CUDA_VISIBLE_DEVICES=1 (2080 Ti on this host is index 1; GPU 0 on this
     host is a different card; the prior B8/O6 runs both used index 1 for the
     2080 Ti, so keep it for reproducibility; re-check the nvidia-smi snapshot
     in 2.0 before running to confirm the name is right for index 1).
2.2  Import:
     ncu --import /tmp/d2/prof.ncu-rep
         --page details > /tmp/d2/prof.txt
     ncu --import /tmp/d2/prof.ncu-rep --page raw   > /tmp/d2/prof_raw.txt
     The raw page is for quoteable full metric names (e.g.
     l1tex__data_bank_conflicts_pipe_lsu) if the details page summarises them
     differently.
2.3  Read /tmp/d2/prof.txt against the checklist in Section 5; write the report.

## 5. Metric checklist -> hypothesis mapping
The report must contain, and be able to point to, EVERY one of these:
- Occupancy section (hypotheses a, e):
    Achieved Occupancy %; Theoretical Occupancy (SM %) and its limiting factor
    (ncu names it: registers / shared memory per thread / block size / barrier
    count / grid size).
- LaunchStats (hypotheses a, b, e):
    Registers per thread; shared memory per block (bytes); block dimensions
    (x,y,z) and grid dimensions actually launched; total blocks.
- SpeedOfLight:
    SM throughput %; DRAM throughput %; compute (SM) vs memory-bound classification.
- MemoryWorkloadAnalysis (hypotheses b, d):
    L2 hit rate %; DRAM read %; L1TEX throughput % and L1TEX hit rate;
    bank conflicts: l1tex__data_bank_conflicts_pipe_lsu.sum (or "shared memory
    bank conflicts" if the details page prints a count).
- SchedulerStats (hypotheses a, c):
    Issued Warp Per Scheduler per cycle; No Eligible Warps %;
    (high + low achieved occupancy => (a); low + high occupancy => (c)).
- WarpStateStats (hypotheses c, d, e):
    top 3 warp stall reasons with their %-share (barrier, long scoreboard,
    wait, mio/lg throttle, math pipe throttle, not selected);
    instruction mix if present: FMA instruction % of issued warp instructions
    (direct signal for (c); if absent, combine SchedulerStats "No Eligible"
    with a high SM throughput).
- FMA / instruction share: instructions per warp; FMA/warp-instruction share if the
  section exposes it; otherwise mark n/c (not computable) and do not fabricate.

## 6. Verdict (FINAL output, exactly 3 sentences)
- S1: which hypothesis dominates, with the ONE number that proves it
  (e.g. "occupancy is limited to 12.5 % by registers, theoretical maximum 25 %").
- S2: the concrete mechanism behind that number (which stall line, which metric).
- S3: what a fix must change (occupancy, per-thread accumulator count, panel
  layout / padding, K block / stage depth) named as a SINGLE lever.
- If the evidence does not cleanly separate two hypotheses (both within the same
  1-2 % band), state that; do not invent a winner.

## 7. Fallbacks
- ncu permission error (expected pattern on a multi-user NVIDIA driver):
  capture the exact stderr, then run
  nsys profile --stats=true  [same invocation]
  for launch geometry (grid/block/smem/regs per kernel) and state in the report:
  "counters unavailable; launch geometry only".
- If ncu is not present or broken (which ncu returns nothing, segfault, non-zero
  exit on a small probe), state the exact error and fall back the same way.
- If another job occupies the 2080 Ti at the start of phase 2, STOP; report the
  nvidia-smi snapshot (which job, util %, mem MiB) and finish with: "GPU occupied
  at check time; profile not run". Do not kill; do not wait for it to finish.

## 8. Report format (<= 26 lines, the whole task output)
- tip + build (full SHA, both targets, release)
- executed kernel symbol with template args (from the ncu import)
- launch geometry (block, grid, smem bytes, regs)
- occupancy numbers and limiter (achieved, theoretical, named factor)
- bandwidth numbers (DRAM %, L2 hit, L1TEX %, bank conflict count)
- top 3 warp stall reasons with %
- instructions per warp and FMA share if available
- the three-sentence verdict
- tooling limitations (ncu version, driver, GPUs, samples/rounds, shared-host caveats)

## 9. Session-local context (paths that only exist in THIS container)
- /tmp/alpaka-orchestration/gemm-20261025/canonical  (clean tree, this task's source)
- /tmp/alpaka-orchestration/gemm-20261025/LEDGER.md  (project ledger: O5/O6, B8, B9, G5)
- /tmp/alpaka-orchestration/gemm-20261025/evidence/ (per-task measurement evidence)
- /tmp/opencode/b8-report.md  (B8 cuBLAS methodology, full 2048^2 context)
- /tmp/opencode/c2/comment.md (prior PR table with d4328ab numbers)
- /tmp/b8local/b8_all.sh                                            (old probe, local only)
- /tmp/opencode/d2-handoff.md                                        (this file)
Do not recreate, prune, or re-clone the canonical tree: it is the source of truth
for 3b8e3e2b and is clean.

## 10. Current state at pause
- Nothing was built. /tmp/d2/ does not exist yet on fwk394.
- The only thing this session did was read-only verification of: host GPUs
  (fwk394 = 2080 Ti + GV100; ncu 2024.2.1.0 present, CUDA 12.5), the canonical
  tree being clean at 3b8e3e2b, and the remote ref already at 3b8e3e2b
  (no push needed). No files were modified in /workspace, no PTY was left open,
  no process was started on the GPU and no process was killed.
- The unrelated prior *.ncu-rep files visible in /tmp on fwk394 (B8-1, B8-2, BB-*)
  are from a previous T8a/IPvX run; do not mix them up with the new D2 profile.

## TODO on resume (in this exact order)
R1. nvidia-smi idle check on fwk394; the 2080 Ti is at CUDA index 1. If the 2080
     Ti is busy, STOP and report; otherwise proceed.
R2. Phase 1: clone 3b8e3e2b onto fwk394 (or upload a tar), build benches-only
     with the exact CMake flags in 3. (Add -G Ninja only if `which ninja`
     returns a path; otherwise Use the default generator.)
R3. Verify: the binary exists at /tmp/d2/build/benchmark/gemm/gemm_bench;
     cuobjdump shows targets sm_70 + sm_75.
R4. Phase 2: one ncu profile run. Use CUDA_VISIBLE_DEVICES=1 (matching B8/O6
     reproducibility), --filter square-2048, --staging cooperative, --samples 10
     --rounds 3 (the harness refuses anything below those thresholds).
R5. ncu --import the report to /tmp/d2/prof.txt and /tmp/d2/prof_raw.txt.
R6. Fill in the Section 5 checklist from those two files.
R7. Write the 3-sentence Section 6 verdict.
R8. Preserve /tmp/d2/prof.ncu-rep if the user wants it for re-opening; the
     Section 8 report format (<= 26 lines) is the deliverable.
R9. Mark D2 DONE in the project LEDGER.
