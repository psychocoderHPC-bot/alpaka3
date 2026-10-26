# M1 executed-kernel proof (perf, by symbol)

Mandatory Phase 3. Run: `perf record -g` on the **native** build, 1 thread, 512³, `--subset all`.
19 136 samples captured (native) / 29 391 (baseline). No perf filter; default cycle
event. `--no-children` (self-only) so we see the leaf kernel that owns the cycles.

## Hottest kernel — native build, 1 thread, 512³

Two `RegisterBlockedGemmTileKernel` instantiations as leaves (fp16 path first, fp32 second):

```
82.01%  [.] void alpaka::onAcc::gemm::RegisterBlockedGemmTileKernel<
           float,                       /* T_Acc  */
           alpaka::onHost::GemmView<float>,          /* T_C    */
           alpaka::onHost::GemmView<alpaka::Float16>,/* T_A    */
           alpaka::onHost::GemmView<alpaka::Float16>,/* T_B    */
           float, float,               /* T_Alpha, T_Beta */
           8u, 8u,                     /* T_TM = 8, T_TN = 8 */
           1u, 1u,                     /* T_WR = 1, T_WC = 1 */
           8u,                         /* T_BK  = 8 */
           0u                          /* T_STAGES = 0 (direct, no cooperative staging) */
         >::operator()< ... CpuOmpBlocks ... > const

17.04%  [.] void alpaka::onAcc::gemm::RegisterBlockedGemmTileKernel<
           float, GemmView<float>, GemmView<float>, GemmView<float>,
           float, float, 8u, 8u, 1u, 1u, 8u, 0u  > ::operator()< ... > const
```

**Executed tile (native): T_TM=8, T_TN=8, T_WR=1, T_WC=1, T_BK=8, T_STAGES=0, executor=CpuOmpBlocks.**
This is exactly the "TM=8, TN=V" with V=8 (the AVX2 float vector width) the task asks to locate.
There is ONE tile/kernel family member that carries all the cycles; the block thread vector
dims (`WarpSize=integral_constant<1>`, `OneLayer<Vec<CVec<1>>`) and `SingleThreadStaticShared<32ul>`
(single 32-element shared step) are consistent with 1 OMP thread per block.

## Same query on the BASELINE build, 1 thread, 512³

```
81.21%  [.] RegisterBlockedGemmTileKernel<
           float, GemmView<float>, GemmView<Float16>, GemmView<Float16>,
           float, float,
           8u, 4u,      /* T_TM=8, T_TN=4  <-- different from native's T_TN=8 */
           1u, 1u,
           8u,          /* T_BK=8 (same) */
           0u           /* T_STAGES=0 */
         >::operator()< ... SingleThreadStaticShared<16ul> ... > const

17.97%  [.] RegisterBlockedGemmTileKernel<
           float, GemmView<float>, GemmView<float>, GemmView<float>,
           float, float, 8u, 4u, 1u, 1u, 8u, 0u  > ::operator()< ... > const
```

**Executed tile (baseline): T_TM=8, T_TN=4, T_WR=1, T_WC=1, T_BK=8, T_STAGES=0, CpuOmpBlocks.**

## Conclusion / confounder exposed

The planner's `T_TN` tracks the *compiled* float SIMD width:
- baseline compiles to SSE2 width (4 float)  -> `T_TN = 4`, shared step 16
- native   compiles to AVX2 width  (8 float) -> `T_TN = 8`, shared step 32

So baseline and native do NOT run the same kernel instantiation: **the ISA flag and the
planner's tile are coupled** (both are derived from `getArchSimdWidth<float>` in the
consuming translation unit). The 1-thread speedup of ~1.66× therefore bundles two changes:
(1) SSE2 -> AVX2+FMA codegen, and (2) the per-worker tile widening T_TN 4 -> 8.
They cannot be separated from this A/B alone; see `M1-caveats.md` for what would isolate them.
