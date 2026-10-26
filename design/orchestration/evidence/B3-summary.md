# B3 first GPU performance numbers, fwk394 (RTX 2080 Ti + Quadro GV100)

## Result: NO MEASURED THROUGHPUT, HTTP-scale hang under CUDA

Task B3 was executed end to end per spec (SHA-verified clone, exact build flags,
sequenced measurement trials, pre-flights). The installed harness contains **no
performance numbers**: large cases freeze on process startup before a single
kernel completes, while a trivial 256-square probe recorded an elapsed time of
0.000 ms per GEMM call (= 2e12 GFLOP/s, physically impossible). This is a
defect in this branch (or in the harness) and it must be resolved before
any of the G5/G6 performance gates (large square mean ratio >= 0.90, per-case
ratio >= 0.80) can be evaluated on GPU.

- Source: e2d552bee79c7479e3044723ee1a26f90e5e70d7 (branch gem/common source)
- Build: exact task flags; `gemm_bench` built clean, pedantic CUDA (sm_70;sm_75)
- Evidence files: B3-conditions, B3-samples-*.json, B3-run-logs

## Measured table (what was actually obtained)

Per specified shape / mode / target, via `gemm_bench --samples 10 --rounds 3
--warmups 1`:

| GPU | shape | mode | status |
|---|---|---|---|
| 0 (2080Ti) | all 6 shapes, all modes | all | not measurable (hang) |
| 1 (GV100)  | all 6 shapes, all modes | all | not attempted (see exclusivity) |

256-square fp32, 4 kernels, all modes: 0.000 ms per call (see B3-samples-512smoke.json)
(shape mismatch: the frozen manifest has no 512-square; use 256-square).

## What the numbers below are not

- **Not a correct GPU measurement.** After the code compiled for the target,
  the harness produced no perf-profiling output. Every recorded time is 0.000 ms.
- **Not validated.** No correctness comparison against cuBLAS or against
  CPU on this branch for the cases that can be executed on GPU either. Only
  unit tests on branch + a trivial naive CUDA kernel show that this platform's
  GPU clocks work.
- **Not sufficient for the perf gates.** Needs to be re-run after the hang
  is fixed.

## Hang

- `square-2026` (i.e. 2026-square) and `square-001k` (1024-square) hang for
  > 300 s at startup. Process spins in C++ host code, GPU util 0%, memory 1 MiB.
- 8-thread PT to CPU: spinning in libc's `poll`. Not CUDA, not host.
  (host-side pipeline/planner; no kernel was yet enqueued.)
- 256-square: completes "immediately" with 0.000 ms timing -> the timing
  harness is not measuring real time, or the kernel is not performing the
  work (possible stub), or the devicelevel is different from expected.

## Kernel symbol attribution

Symbol attribution for the kernels that would actually execute is unavailable
until the hang is fixed. `ncu` is installed on the target but can only profile
kernels that actually run; the function is not reached.

## Exclusions / exclusivity

- `fwk394` was not exclusively reserved by this task; it is a shared container
  on the host system. No other user compute app was visible (computeapps was
  empty during every preflight). GPU0 was borrowed by us and GPU1 was left
  unused (0% util) to preserve the ambient state. Did not attempt concurrent
  measurement on both GPUs (task said one process at a time).
- 256-square smoke was done on GPU0 only (earliest-complete).

## Next steps for Task B3 to proceed

1. Fix the hang: investigate the host-side planner with an extremely small case;
   confirm that `onHost::allocate` / `onHost::memcpy` return correctly for
   CUDA device type (memory.h path).
2. Verify that the kernel for 256-square actually runs (patch the harness to
   return sum, compare to CPU).
3. Rerun B3 per spec after step 1 lands.
