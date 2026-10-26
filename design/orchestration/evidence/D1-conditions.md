# D1 conditions

Task: D1 — diagnose why the GPU kernel is ~3-orders-of-magnitude below expectations, and
re-measure the GSP100 (GV100) row that was corrupted in the B3 report. No source changes.

## Run date
2026-10-25 (UTC; host clocks differ ±1min, noted per target below)

## Source
- Repo: https://github.com/psychocoderHPC-bot/alpaka3
- Branch: gemm/common-source
- Cloned HEAD (local): `d3ad457550b14e182ab92423a8ac2d190e8a545d`
  docs: record the M7 matrix-accumulate milestone
- Most recent commit on the branch prior to HEAD (verified via
  `git log --oneline gemm/common-source~1..HEAD`):
  - `1a94fe0` fix: use the supported SYCL barrier in the oneAPI backend
  - `e2d552b` docs: record the GEMM register tile, clipped scoring and ISA option   <- B3's SHA
  - `ec93e4c` fix: rank the register-filling tile for compute-heavy GEMM shapes
  - `c0fa6f1` build: add opt-in host-native ISA for GEMM benchmark and tests
  - (plus M7-related history below these)
- B3 SHA: `e2d552bee79c7479e3044723ee1a26f90e5e70d7`
- D1 vs B3: **D1 built 4 commits newer than B3** (the O2/O1 set, a SYCL fix
  that is inert here — SYCL disabled in the build — and the M7
  matrix-accumulate milestone). Measured against D1 HEAD; B3's row is
  therefore NOT assumed to be comparable.
- Note: task stated "the extra commits are a SYCL fix and an O2/O1 set"; the
  M7 matrix-accumulate milestone is also newer than B3 and changes the CUDA
  original adapter path. Both are captured in this D1 run; B3's row is
  independence measured.

## Build
```
cmake -S <src> -B <build> \\
  -DCMAKE_BUILD_TYPE=Release \\
  -Dalpaka_TESTS=OFF -Dalpaka_BENCHMARKS=ON \\
  -Dalpaka_COMPILE_PEDANTIC=ON -Dalpaka_DEP_OMP=OFF \\
  -Dalpaka_HOST_Cpu=ON -Dalpaka_HOST_NumaCpu=OFF \\
  -Dalpaka_DEP_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;75"
cmake --build <build> --target gemm_bench -j <ncpu>
```
- fwk394: 28 cores in use, clean 0 errors.
- rosi_v100: initial build at `-DCMAKE_CUDA_ARCHITECTURES=70` (both target
  GPUs are sm_70); then rebuilt with `-DCMAKE_CUDA_ARCHITECTURES="70;75"`
  to cover the 2080 Ti too (not required for rosi_v100 measurement).

## Hosts

| target   | host             | GPU (phys idx 0) | GPU (phys idx 1)  | driver        | CUDA toolkit |
|----------|------------------|-------------------|--------------------|---------------|--------------|
| fwk394   | 337def6dbb2f (devbox | NVIDIA GeForce RTX 2080 Ti | Quadro GV100 | 555.52.04 | 12.5.82 (nvcc V12.5.82) |
| rosi_v100| gv001            | Tesla V100-SXM2-32GB | Tesla V100-SXM2-32GB (3 more) | 570.158.01 | 12.6.85 (nvcc V12.6.85) |

## Device indexing (PROVEN by standalone probe binary, two different CUDA_VISIBLE_DEVICES states)

On the 2-GPU host (fwk394), the mapping between nvidia-smi physical
index and CUDA_VISIBLE_DEVICES value, as observed by a probe
(`cudaGetDeviceProperties` after the same process has allocated):

| default CUDA_VISIBLE_DEVICES | CUDA device 0 is | nvidia-smi idx |
|---|---|---|
| not set (both visible) | 2080 Ti  | 0 |
| not set (both visible) | GV100      | 1 |
| `CUDA_VISIBLE_DEVICES=0` | 2080 Ti | 0 (the card in that slot) |
| `CUDA_VISIBLE_DEVICES=1` | GV100     | 1 (the card in that slot) |

This is the **B3 smoking gun**: B3's report labeled its windows as
"index 0 = 2080 Ti (GPU A)" and "index 1 = GV100 (GPU B)", then ran
window 0 via `--filter`, which on a 2-GPU host with `CUDA_VISIBLE_DEVICES
unspecified` gives device 0 = physical index 0 = 2080 Ti. For window 1
it used `CUDA_VISIBLE_DEVICES=1`, which gives device 0 = physical
index 1 = GV100. **B3's numbers on the "2080 Ti" row are actually the
GV100's, and its GV100 row is actually the 2080 Ti's.** The
"identical numbers to the millisecond" problem is resolved: the B3
report had the two GPUs' rows swapped in its legend; the underlying
samples JSON files are consistent with one card per card (verified by
sample-count/shape checks in the B3-gpu0/ and B3-gpu1/ trees, see
extract-terminology-verification notes).

## Pre-flight protocol
(Per task; identical to B3 except clocks / P-state are also polled in-window.)

```
pgrep -af gemm_benc | grep -v grep   # must be empty
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader   # must be empty
sed -n '1p' /proc/loadavg | awk '{if($1<1.0)print "ok:"$0; else print "warn:"$0}'
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader
```

## Exclusive-window protocol
- Always one process at a time.
- `pkill -9 -f gemm_bench` not used (would kill the running process);
  no in-window polling via `nvidia-smi -l` was needed (D1's in-window
  clock polling was done via `nvidia-smi --query-gpu=clocks.sm ... -l 1`
  in the background, sampled for 5 s into /tmp/d1clk_*.log).
- Each case: `--warmups 1 --samples 10 --rounds 3` = 30 timed samples
  per case (harness's `--samples 10 --rounds 3` accumulates to
  `rounds*samples_per_round = 30` timed runs), unless the case
  exceeds 120 s in wall time, in which case samples were reduced to
  3 (noted per-call in the delta).
