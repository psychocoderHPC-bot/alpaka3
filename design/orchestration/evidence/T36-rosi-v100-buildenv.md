# T36 — pre-warmed CUDA build environment on `rosi_v100` (partial)

**Status:**PARTIAL — tree + Ninja configure present; full build and smoke NOT completed
(the compute-MCP data channel to the container died at ~10:56 UTC 2026-10-25 after an
`ipcrm` sweep on a broken System V IPC stack, ~2 min before the build should have
finished). Everything below is verified on the target at the moment of loss.

## Source
- repo: `https://github.com/psychocoderHPC-bot/alpaka3` branch `gemm/common-source`
- path: `/tmp/gemm/src` (cloned --depth 1, 2026-10-25 ~10:28 UTC)
- SHA built: **`dde6e3545f18afb0f3ce9ef7a9ce5acadafa5c4b`**
- LOCAL MODIFICATION (uncommitted, workspace-only):
  `test/unit/kernel/lockstepAudit.cpp:203` — `auto dataSource` → `auto const dataSource`
  (one line; silences `-Werror=unused-but-set-variable` in the CUDA-TU compile of that
  test). Backup of the pristine file: `/tmp/gemm/src.lockstepAudit.bak`.
  This is a project defect on the GEMM branch: the variable is used only inside
  type-contexts (`ALPAKA_TYPEOF`, `static_assert`), which nvcc's -W flags still count
  as "set but not used".

## Build dir
- `/tmp/gemm/build-cuda70` — **Ninja** generator, CMake 3.28.3, configured 10:50:20 UTC
  (configure wall 8.3 s). Objects from an earlier Makefiles attempt were wiped
  (`rm -rf` before the Ninja reconfig), so this is a cold tree: a clean build is
  ~6-8 min at -j 40 (measured: a Makefiles clean build reached 73% in 4.5 min).

## Exact configure command (as used, accepted in full)
```
export PATH=/usr/local/cuda/bin:$PATH
cmake -S /tmp/gemm/src -B /tmp/gemm/build-cuda70 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dalpaka_TESTS=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_COMPILE_PEDANTIC=ON \
  -Dalpaka_DEP_OMP=OFF -Dalpaka_HOST_Cpu=ON -Dalpaka_DEP_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```
All 9 options accepted, none rejected. Extras observed: hwloc 2.10 (AUTO),
`std::atomic_ref` + `std::simd` found, Catch2 fetched into `build/_deps`.

## Build
- First full build (Unix Makefiles, `-j 40`, 10:41:30→10:46:08 UTC): failed at 73% on the
  single `lockstepAudit` warning-as-error above. At that point: 157 .o, 46 test executables,
  `gemm_bench` already linked (path `build-cuda70/benchmark/gemm/gemm_bench`).
  Error (60-line capture truncated; the unique failure):
  ```
  lockstepAudit.cpp:203:6: error: variable 'dataSource' set but not used
      [-Werror=unused-but-set-variable]
  gmake: *** [Makefile:146: all] Error 2
  ```
  Classification: **project defect** (not toolchain, not missing dependency).
- After the patch, Ninja reconfigure + fresh build was started; the control link died
  ~2 min in, so no clean build artifact is left.

## Environment traps (these are what will bite later workers)
1. **System V IPC is broken in this cgroup.** Every `semop` currently fails with
   `Invalid argument` (verified via strace on a live build: `semtimedop` → forever, or `semctl/semop` → EINVAL). Confirmed consequences:
   - `cmake` CUDA-ABI **configure** fails abortingly with
     `semop(1): encountered an error: Invalid argument` **unless** a set of empty
     semaphore arrays pre-exists (perm 0666). Helper that creates 16 of them is left
     at `/tmp/gemm/mksem` (source `/tmp/gemm/mksem.c`, keys 0x3f1..0x49b step 0x11).
     Post-sweep they may be gone again — re-run `gcc -o /tmp/gemm/mksem /tmp/gemm/mksem.c && /tmp/gemm/mksem`.
   - **never** `ipcrm -s <all ids>` here while the gateway holds IPC; it permanently
     broke this job's IPC and every subsequent `cmake`/`cmake --build`/`ninja` that
     touches semas, and took the MCP channel down with it. The correct recovery is
     host-side (kernel IPC ns), not in-container.
   - `cmake --build` (Makefiles AND Ninja) hangs in `__do_semtimedop` in this state;
     invoking the generator binary directly (`ninja -C <dir> -j 40`) or `gmake` was
     the only path observed to start compiling at all. `ctest` is safe (no IPC).
2. `nvcc` not on PATH → keep `export PATH=/usr/local/cuda/bin:$PATH` in every shell.
3. **No ccache / sccache** on the target → full clean rebuild each time the cache is
   blown (~6-8 min at -j 40); no install performed (per task).
4. `/usr/bin/time` absent → wall time via `DATE=$(date +%s%N)` before/after, or
   the timestamped build.log; peak RSS via the sampler at `/tmp/gemm/peakmem.sh`
   (samples VmRSS of the build process + children every 2 s while a PID file exists).
5. MCP exec calls on this gateway have a short ceiling (~30 s); long builds must run
   as `setsid nohup` detached scripts with marker files + short poll loops. The
   `computeMCP_session_*` PTY channel was additionally flaky for 4+ sessions at the
   start of this task.

## Reuse protocol for the next worker
```
# 0. If the target's compute-MCP channel is dead: request a gateway re-dial first.
# 1. Verify environment sanity:
ls /tmp/gemm/src/.git /tmp/gemm/build-cuda70/build.ninja    # CMakeCache.txt may be Makefiles; if so: rm -rf /tmp/gemm/build-cuda70
# 2. IPC workaround (only if cmake still emits the semop(1) error):
gcc -o /tmp/gemm/mksem /tmp/gemm/mksem.c && /tmp/gemm/mksem
# 3. Incremental update + rebuild (STABLE path; preserves objects):
export PATH=/usr/local/cuda/bin:$PATH
git -C /tmp/gemm/src fetch origin gemm/common-source && git -C /tmp/gemm/src checkout FETCH_HEAD   # re-apply the 1-line const patch if the file changed!
cmake -S /tmp/gemm/src -B /tmp/gemm/build-cuda70 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dalpaka_TESTS=ON -Dalpaka_BENCHMARKS=ON -Dalpaka_COMPILE_PEDANTIC=ON \
  -Dalpaka_DEP_OMP=OFF -Dalpaka_HOST_Cpu=ON -Dalpaka_DEP_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
ninja -C /tmp/gemm/build-cuda70 -j 40
# 4. Smoke:
ctest --test-dir /tmp/gemm/build-cuda70 -R 'gemm|lockstep' --output-on-failure
/tmp/gemm/build-cuda70/benchmark/gemm/gemm_bench --smoke
```
Benchmarks (throughput) remain a separately scheduled task — do NOT run them here.

## Artifacts left on the target
- `/tmp/gemm/src` (fixed source), `/tmp/gemm/build-cuda70` (Ninja, cold)
- helpers: `/tmp/gemm/{wrapper.sh,build.sh,mksem,mksem.c,peakmem.sh,configure.log,build.out,build.done.rc,watch.out}`
- this note: `/tmp/alpaka-orchestration/gemm-20261025/evidence/T36-rosi-v100-buildenv.md`
  (also to be copied to the target's same path once the channel is restored)
