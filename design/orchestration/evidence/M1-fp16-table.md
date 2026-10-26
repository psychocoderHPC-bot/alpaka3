# M1 mixed fp16→fp32 (widening) table

At 512³, 1 thread, both builds. This is the branch-free fp16→fp32 widening path.

| build    | mode         | median ms | GFLOP/s (2*M*N*K) |
|----------|--------------|-----------|-------------------|
| baseline | fp16_fp32    | 179.601   |      1.495        |
| native   | fp16_fp32    | 116.957   |      2.295        |

- **native / baseline = 1.535×** on the widening path (1 thread, 512³).
- **vs the earlier "1.78 GFLOP/s" reference at the same shape**: the native build now
  measures **2.295 GFLOP/s** = **1.29×** the earlier figure. That earlier number is no
  longer reproducible on this binary; the current build is strictly faster.
- Per-element fbias: the widening pass is memory-bound (fp16 input -> fp32 output), so
  its raw "GFLOPs" is low; the ~1.5× native gain here is the same AVX2+FMA + wider-TN
  per kernel, applied to a convert+dot kernel.

At 1024³ (for completeness, 1 thread):
| build    | median ms | GFLOP/s |
|----------|-----------|---------|
| baseline | 1441.05   | 1.490   |
| native   |  936.83   | 2.292   |
(speedup 1.538×, consistent with 512³.)
