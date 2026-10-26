# M1 FP32 table

Shape × threads (median ms / GFLOP/s). Speedup = native / baseline.
Metric: 2*M*N*K useful FLOPs / median wall time.
fp32 only; 10 samples × 3 rounds, 1 warm-up, pinned to cores 0-63.

| shape  | threads | baseline median ms | baseline GFLOP/s | native median ms | native GFLOP/s | speedup |
|--------|---------|--------------------|------------------|------------------|----------------|---------|
| 512   ³ | 1       | 39.407             |   6.812          |  23.775          |   11.291       | 1.658   |
| 512   ³ | 64      |  2.140             | 125.431          |   1.744          |  153.939       | 1.227   |
| 1024  ³ | 1       | 313.613            |   6.848          | 188.184          |   11.412       | 1.666   |
| 1024  ³ | 64      | 18.912             | 113.553          |  11.558          |  185.794       | 1.636   |

Notes:
- Single-thread speedup is uniformly ~1.66× from 512³ to 1024³.
- At 64 threads, 1024³ keeps a ~1.64× native advantage while 512³ already saturates:
  the 512³ 64-thread time is bound by tile-parallelism and small panels, not by FLOP rate.
- Baseline 1 t ≈ 6.8 GFLOP/s confirms the earlier ~4.9–7 GFLOP/s reference for this build.
