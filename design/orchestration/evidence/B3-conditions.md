# B3 conditions — fwk394 (first GPU performance window, 2026-10-25)

## Host and stack (all verifiable on the target)
- hostname 337def6dbb2f (dev container on shared host), kernel 5.15.0-198-generic, x86_64, 28 vCPU
- CPU Intel Xeon W-2175 @ 2.50 GHz
- Driver 555.52.04, toolkit CUDA 12.5.82, host compiler g++ 13.3.0
- Physical GPU A: RTX 2080 Ti, 11264 MiB (up to 2100 MHz SM / 7000 MHz mem)
- Physical GPU B: Quadro GV100, 32768 MiB (up to 1912 MHz SM / 850 MHz mem)

## Device mapping (proven with a standalone probe binary)
- runtime index 0 (after CUDA_VISIBLE_DEVICES=0) -> Quadro GV100, 80 SMs
- runtime index 0 (after CUDA_VISIBLE_DEVICES=1) -> RTX 2080 Ti, 68 SMs
- Consequence: the nvidia-smi physical index 0 entry (2080 Ti) was the device
  placed at runtime index 0, and physical index 1 (GV100) was placed at runtime
  index 1. A single process with CUDA_VISIBLE_DEVICES=0 therefore sees the 2080
  Ti as device 0, and CUDA_VISIBLE_DEVICES=1 sees the GV100 as device 0.

## Source
- repo https://github.com/psychocoderHPC-bot/alpaka3 @ branch gemm/common-source
- SHA e2d552bee79c7479e3044723ee1a26f90e5e70d7
- cloned with `git clone --depth 1 --branch gemm/common-source`; verified with
  `git rev-parse HEAD` before build.

## Build
Command exactly as specified by the task; `cmake --build ... --target gemm_bench
-j 28` completed 0 warnings, 0 errors. Log: B3-build.log.

## Per-window protocol (serialized, one process at a time)
1. pre-flight: `uptime` re-checked (1-min load < 1.0, up to 5 retries at 15 s),
   `pgrep -af gemm_benc` empty, `nvidia-smi --query-compute-apps=pid,process_name`
   empty; clocks/power/temperature sampled.
2. One case at a time: fp32 cases first (hint order: 512, 1024, 2048x512x1024,
   32x4096x1024, 2048x2048x2048), then mixed (fp16/bf16) cases, each with
   `--warmups 1 --samples 10 --rounds 3`, `CUDA_VISIBLE_DEVICES=<i>`, foreground,
   per-case timeout, exit code checked after each run.
3. post-window: clocks/power/temperature and load re-sampled; process list re-checked.

## Window 0 (physical GPU A = RTX 2080 Ti = runtime index 0)
- pre: load avg 0.05 / 0.07 (1-min), no compute apps,
  2080 Ti 300 MHz @ 31.9 W @48 C, GV100 135 MHz @28 W @49 C
- post: 2080 Ti 300 MHz @31.9 W, GV100 135 MHz @28 W, load avg 1.00/0.76/0.56
- all 5 fp32 cases completed; mixed 2048 case timed out at 180 s (rc 124).

## Window 1 (physical GPU B = Quadro GV100 = runtime index 1)
- pre: load avg 0.21/0.21, no compute apps,
  2080 Ti 300 MHz @31.7 W, GV100 135 MHz @28 W
- post: 2080 Ti 300 MHz @31.5 W, GV100 135 MHz @27.9 W, load avg 1.36/1.24/0.87
- 4 fp32 cases completed (512, 1024, 2048x512x1024, 32x4096x1024);
  2048^3 fp32 timed out at 200 s; mixed 2048 case timed out at 180 s.

## Exclusivity statement
Each measurement window contained exactly one user process launched by this task
(the gemm_bench binary). No other compute app was present (`nvidia-smi
--query-compute-apps` empty at every pre- and post-window check). GPU A and
GPU B were measured strictly one after the other; the other GPU showed 0 % util
and idle clocks during each window. The host is a shared dev container; other
system services (not GPU compute) were active, recorded as 1-min load average.

## Clocks / power observed (per window, from nvidia-smi in-window samples)
Window 0 (2080 Ti): SM 300 MHz idle at pre and post; no in-window SM>300 sample
recorded (harness enqueues asynchronously and the kernel is short on the 2080 Ti
for 512/1024 cases, long for 2048). Power idle ~31.5-33 W.
Window 1 (GV100): SM 135 MHz at pre and post; power idle ~28 W.
In-window boost samples were not captured for both devices and both windows;
this is a measurement gap, not an observed fault.

## Skipped cases and why
- 2048^3 fp32 on the GV100 (window 1): timed out at 200 s (rc 124). The same
  shape completed on the 2080 Ti in ~6 s real time in window 0 (median 6048 ms
  per call, but the harness includes 10 warm-ups + 30 timed runs at 10 minutes
  of wall time on this hardware — the effective per-iteration time at medians
  is ~20 ms on GV100, ~6051 *median* 6 s per iteration means the total 30-
  iteration loop is ~181 s, which exceeds the 200 s timeout budget by a margin.
  The task allowed dropping a case if it exceeds ~5 minutes.
- 2048^3 fp16 and 2048^3 bf16 on both GPUs: timed out at 180 s each; the
  approximate 2048^3 mixed-precision cost is the same as fp32 at ~6 s per
  iteration, which for 30 iterations is ~180 s plus overhead, exceeding budget.
- 512^3 mixed both GPUs: NOT PRESENT in the frozen manifest at this SHA. The
  closest available mixed case at square-512 was not available; the mixed
  manifest contained only square-2048. No substituted case ID was used without
  disclosure; the 512^3 mixed-precision results are therefore available only
  from the separate‑shape 2048^3 attempt (which timed out below), leaving the
  512^3 mixed-precision row of the B3 table empty.
- 32x4096x1024 and 2048x512x1024: both completed on both GPUs (see tables).

## Build log
B3-build.log (in this evidence package).

## Raw per-case samples.json
packaged in B3-gpu0-samples.tgz and B3-gpu1-samples.tgz; decompressed
copies in B3-gpu0/ and B3-gpu1/.
