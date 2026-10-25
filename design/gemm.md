# Portable GEMM design of record

Status: M0 design artifact, revision 2, 25 October 2026.
Binding specification: `alpaka3-gemm-design-guide.md`, Version 2, 25 October 2026.
Machine-readable companion: `validation/gemm-requirements.yaml`.
External evidence root: `/tmp/alpaka-orchestration/gemm-20261025/logs`.
This document records design decisions and evidence obligations. It does not claim measured performance, correctness on hardware, or native validation.

## 1 Scope and non-goals

The first release implements real dense GEMM:

`C := alpha * op(A) * op(B) + beta * C`

with A logically M x K, B K x N and C M x N. The implementation supports row-major and column-major positive-stride views, padded leading dimensions, subviews, and the NN, NT, TN and TT transpose combinations. It rejects invalid dimensions, unsupported strides, overlapping C with A or B, and overflowing address calculations before launch. A and B may alias each other because both are read-only.

Zero-size contract:

- M = 0 or N = 0 is a no-op.
- K = 0 or alpha = 0 computes `beta * C` without reading A or B.
- beta = 0 does not read old C.
- beta = 1 with K = 0 is a no-op.
- Unused views still pass the same shape and stride validation. A null pointer is illegal unless the public API explicitly allows it.

Non-goals: negative strides, complex arithmetic, grouped GEMM, batched GEMM, fused activations, extended floating-point types, and internal vendor GEMM calls. Small single matrices remain in benchmark scope.

## 2 Source baseline and API inventory

Verified revisions. Both provenance records live in `logs/T5a-base-reconcile.txt` and `logs/T1-lockstep-audit.md`.

| Revision | Identity | Verification | Status |
| --- | --- | --- | --- |
| `6f13700c390a09df818a255d53e730f1f9d69456` | `psychocoderHPC-bot/alpaka3` branch `dev` | GitHub API ref read and full-clone `git rev-parse`; canonical base | VERIFIED base |
| `05290c84b5c2349ff93585a95845e5471bba2540` | lockstep PR 7 head, ref `origin/issue-6-lockstep2` | `gh pr view 7` reports OPEN, not draft, mergeable=MERGEABLE; full-clone diff is 14 files, +1544/-0 | VERIFIED, not merged |
| `b7d339d07056a9a2a6c4051cc3927157bc5f0d51` | upstream `dev` snapshot inspected by the guide | full-clone `git rev-parse`; 3 commits ahead of the canonical base | VERIFIED, not merged |
| `35ba304fd2215117cd1179830d2cbe468ced3285` | internal `gate` mirror branch `dev` | mirror ref read | STALE: 30 commits behind the canonical base; not a base |
| `b77f140fde4c5e94ff2a5ef6ddbc5a647a952f64` | internal `gate` mirror lockstep branch | full-clone ancestry | DIVERGENT lineage (9 ahead, 7 behind PR head, fork point `8ae77718a57c38954bd5c51e49fb1194d8850085`); ignore |

PR 7 sits on merge-base `8ae77718a57c38954bd5c51e49fb1194d8850085`. None of the 12 commits between that merge-base and the canonical base touches a file in the PR, so the rebase is a pure addition without expected content conflicts. The `gate` mirror is stale and must not be used to pin any revision.

The canonical base has no lockstep header and no lockstep test. `include/alpaka/onAcc/lockstep.hpp`, `include/alpaka/onAcc/lockstep/` and `test/unit/kernel/lockstep.cpp` appear only on the PR head. The lockstep integration commit must land before kernel work.

Existing APIs verified at the base SHA:

| API | Location | Use |
| --- | --- | --- |
| `onAcc::syncBlockThreads(acc)` | `include/alpaka/onAcc/Acc.hpp` | block barrier for staging |
| `onAcc::memFence(acc, scope, order)` | `include/alpaka/onAcc/memFence.hpp` | memory ordering without a barrier |
| `onAcc::declareSharedVar`, `onAcc::declareSharedMdArray` | `include/alpaka/onAcc/Acc.hpp` | static shared storage |
| `onAcc::getDynSharedMem<T>(acc)` | `include/alpaka/onAcc/Acc.hpp` | dynamic shared storage |
| `onAcc::worker::threadsInBlock`, `linearThreadsInBlock` | `include/alpaka/onAcc/WorkerGroup.hpp` | participant groups |
| `onAcc::layout::Strided`, `Contiguous`, `Optimized` | `include/alpaka/onAcc/layout.hpp` | existing iteration layouts |
| `onAcc::SimdAlgo`, `onAcc::worker::threadsInGrid` | `include/alpaka/onAcc/SimdAlgo.hpp` | host-side SIMD algorithm pattern |
| `getArchSimdWidth`, `getNumPipelines`, `getNumElemPerThread`, `getCachelineSize` | `include/alpaka/api/trait.hpp` | compile-time planning facts |
| `DeviceProperties` | `include/alpaka/onHost/DeviceProperties.hpp` | global memory, shared memory per block, multiprocessor count, warp size, block and grid limits |
| `ALPAKA_FN_SYMBOL`, `alpakaFnDispatch`, `Symbol::Spec<api, deviceKind>` | `include/alpaka/fn.hpp`, `docs/source/tutorial/vendorInterop.rst` | vendor symbol dispatch |
| `getSimdFrameSpec`, host `transform` | `include/alpaka/onHost/internal/interface.hpp`, `include/alpaka/onHost/algo/internal/transform.hpp` | host frame selection pattern |

Lockstep APIs verified on PR 7 head only, absent from the canonical base: `onAcc::makeLockstep`, `LockstepScope::var()`, `LockstepScope::concurrent()`, `LockstepVar`, `MappedData`, `RegisterSimdRef`, `onAcc::map()`.

### 2.1 Lockstep audit result

The audit at `05290c84` (`logs/T1-lockstep-audit.md`) covers the six items the guide requires. Four items are OK as written:

| Item | Verdict | Evidence |
| --- | --- | --- |
| 1 `BindValueType_t` lazy selection | OK | `std::conditional_t` selects one branch, so the unselected `SetValueType_t` is never instantiated; the explicit value-type requirement for a first `map()` argument is documented |
| 2 accelerator reference lifetime | OK | `LockstepScope` stores `T_Acc const&`; all in-tree uses are inside the kernel functor where `acc` is live; no compile-time guard prevents misuse outside a kernel |
| 3 const local binding | OK | the const `bind` overload yields `RegisterSimdRef<T const, W>`; load is valid and a store is a compile error |
| 5 ownership stability across passes | OK | `var<T>()` fixes the slot count at construction; successive `concurrent()` calls see the same per-worker slot range; the docstring wording needs clarification only |

Two items remain open defects:

- Item 4. Zero logical extent is safe and workers greater than elements is bounded, but a zero worker extent reaches `divCeil(N, 0)` in the consteval `getLockstepSimdWidth` two-argument overload with no guard. No test covers it. Closing regression: `zero_worker_rejection` (plus `zero_extent`), tracked by `S8-STRUCT-simd-slots`.
- Item 6. The no-collectives and no-tail-divergence rule exists only in a private doxygen `@attention` on `foreachImpl`. Public `concurrent()` documentation and `docs/source/tutorial/lockstep.rst` do not state it. Closing regression: `tail_callback_collective_contract`, tracked by `S8-PIPE-state-machine`.

Proposed new APIs, not present at either verified revision:

- Layout policy with a validated ownership map, compose, slice, transpose, and contiguous/broadcast classification.
- Fragment view over `scope.var()` storage, carrying shape, ownership, value type, layout identity and projection operations.
- `matrixAccumulate(group, c, a, b)` as a reusable vendor symbol with a common SIMD/FMA fallback.
- Stage controller implementing acquire, fill, commit, wait, consume, complete and release with generations.
- Planner facts model, bounded candidate family, inspectable plan object, diagnostics and expert override.
- Additional facts for cache capacity, register budget, active blocks, async-copy capability and native matrix combinations.
- FP16 and BF16 value types and conversion tests. A search of `include/` at the base SHA found no such value type; treat them as an M1 prerequisite.

Names in the guide pseudocode such as `makeLockstepWithLayout`, `Plan::layout.partition`, `Plan::matrixOperation.kQuantum`, `copyOperands`, `loadRowFragment` and `matrixAccumulate` are design vocabulary. They become real API only after review.

## 3 The five composable parts

### 3.1 Views

Views describe logical matrix coordinates, shapes and strides. The kernel consumes a validated problem description, never raw backend pointers. View normalization handles row-major and column-major storage, leading-dimension padding, subviews and all transpose combinations. Validation runs on the host with checked wide arithmetic for dimensions, byte sizes, pitches and launch counts. Device indexing uses a type wide enough for the supported range. A 64-bit host shape with a truncated 32-bit kernel index is not supported.

The existing alpaka view and data-storage facilities carry the raw storage. The GEMM layer adds an explicit problem descriptor so that planning, validation and diagnostics share one interpretation.

### 3.2 Ownership layouts

An ownership layout defines the mapping `(worker, localSlot) -> logicalCoordinate` and the inverse, or an equivalent validated ownership relation. SIMD packing groups local slots and stays separate from ownership. A worker group identifies participants and is not a memory level.

Required operations: compose, slice, transpose, and classify contiguous or broadcast access. Copy layouts and compute layouts may differ. Physical shared-memory padding and swizzling stay separate from logical ownership, so a storage change cannot change mathematical coordinates.

Validation rules:

- Every valid output coordinate has exactly one owner.
- Every owned valid slot is in bounds.
- Holes are marked inactive.
- Round trips are proven where an inverse is supplied.
- Multiple readers of A and B are allowed; duplicate C owners are rejected.

The default layout keeps the existing flattened cyclic ownership. The fallback supports serial execution and any supported SIMD width. Warp size and vector width remain distinct concepts.

### 3.3 Lockstep storage and iteration

The kernel uses `onAcc::makeLockstep` for its logical scope and `scope.var<T>()` for per-worker storage. A fragment view wraps that storage and adds logical shape, ownership, value type and layout identity. No second local-storage system is introduced.

Fragments provide fill, compatible element-wise concurrent access, row and column projections, broadcast, and load/store bindings. Static subfragment traversal allows unrolling without forcing unbounded code growth. Broadcast reads values owned by the same worker. Cross-worker redistribution is an explicit named collective with a stated group and synchronization contract.

The same logical value refers to the same storage at every SIMD pass width. Compile-time checks reject incompatible scopes and layouts. Documented lifetimes prevent a fragment view from outliving its owning variable unnoticed.

### 3.4 Matrix operations

The first implementation is a common register-blocked outer-product/FMA operation. It reuses row and column fragments across several accumulator entries. Independent unrolling is shared, including width-one execution.

The optional collective `matrixAccumulate(group, c, a, b)` contract declares legal operand and accumulator types, instruction shapes, group membership, layouts, alignment, completion semantics and numerical mode. Its fallback is common SIMD/FMA arithmetic. Native fragments may be opaque and require an explicit conversion to a portable epilogue layout. Portable element-wise epilogues never apply arbitrary local-slot indexing to an opaque native fragment.

Every specialized implementation uses the alpaka vendor/function-symbol interface described in section 5.

### 3.5 Staging

Staging uses verified alpaka shared storage: `declareSharedVar`, `declareSharedMdArray` or `getDynSharedMem`. Barriers use `onAcc::syncBlockThreads`. The synchronous policy completes each cooperative phase before the next phase starts. A sequential executor runs a cooperative phase to completion without waiting for unscheduled virtual workers. A producer/consumer group is never emulated with blocking waits that prevent the producer from running.

The async policy preserves the same contract and uses native acceleration only where the capability is present and tested. Commit and wait participation stays converged for the native primitive. Per-worker remainder branches do not alter collective stage counts. The baseline needs no cooperative grid launch and no grid-wide synchronization.

Implemented synchronous policy (M4). `onAcc::staging::StageTracker<S>` validates the acquire, fill, commit, wait, consume, complete and release transitions and tracks a per-slot generation. `onAcc::staging::SyncStageRing<T, S, E>` wraps block-shared storage from `declareSharedMdArray`; commit and complete use `onAcc::syncBlockThreads`. The common kernel fills each K-block panel cooperatively with a worker-strided loop, tests boundary coordinates before the load and writes exact zeros into the invalid halo. The register-blocked outer product then reads the panel, so the compute phase never reads uninitialised shared memory.

Scratch bound. One stage holds `BM*BK + BK*BN` accumulator values. The host requires `S*(BM*BK + BK*BN)*sizeof(accumulator) + 64` bytes to fit `DeviceProperties::sharedMemPerBlockBytes` and throws `std::invalid_argument` otherwise; it never truncates the allocation. The stage count family is bounded to {1, 2, 3}.

Policy selection. `GemmStaging::cooperative` is an explicit expert override. `automatic` resolves to the compile-time configuration value `internal::defaultGemmStaging`, currently `direct`, so the shipped default keeps the existing global-load path. The documented rule in `internal::stagingRule` (arithmetic intensity at least 8 and the ring fits shared memory) is the input for the M5 planner.

## 4 Numerical modes and contracts

| Mode | Arithmetic contract | Forbidden shortcut |
| --- | --- | --- |
| FP32 | FP32 operands and accumulation; FMA contraction and reordering are allowed and documented | Silent TF32, FP16 or BF16 input conversion |
| FP64 | FP64 operands and accumulation | FP32 partial accumulation |
| FP16 to FP32 | Stored FP16 inputs convert exactly to FP32 for generic arithmetic; FP32 accumulation | Undisclosed change of input conversion or rounding |
| BF16 to FP32 | Stored BF16 inputs convert exactly to FP32 for generic arithmetic; FP32 accumulation | Treating BF16 as FP16 |
| Optional reduced FP32 | Explicit opt-in, separately named and benchmarked | Reporting it as strict FP32 |

Contract details:

- Alpha and beta convert to the accumulation type. Alpha multiplies the product in the accumulation type. Beta multiplies the converted previous C value in the accumulation type.
- The epilogue computes in the accumulation type and rounds once to the output type, unless a named mode specifies otherwise.
- Required input and output conversions use round-to-nearest, ties-to-even.
- Mixed modes always support FP32 output. A lower-precision output requires a separately named mode with explicit conversion semantics.
- Strict mode preserves input precision. It does not promise a fixed summation order or cross-device bitwise equality.
- FP32 may contract multiply-add and reorder operations; the mode documentation records both.
- FP64 never uses FP32 partial accumulation.
- Default validation covers finite values without fast-math assumptions.
- An engine that flushes subnormals or differs on exceptional inputs must expose and test that contract, or stay outside strict mode.
- An FP64-incapable device is never reported as an FP64 pass.
- The implementation never lowers precision, calls vendor GEMM internally, or weakens the source constraint to pass a benchmark.

Hardware consequence, from the section 11 inventory: no reachable target exposes `avx512_bf16` or `amx_tile`, so FP16 and BF16 arithmetic is generic emulation on every CPU target. The only native matrix scope is CUDA Volta/Turing/Ampere, oneAPI Arc A770 and HIP gfx1100. The gfx906 target has no matrix cores. G9 native records cover those targets; every other target passes on the tested fallback.

## 5 Common source boundary and vendor dispatch

The GEMM kernel and its control flow stay common. Common code contains no backend-specific GEMM body, no architecture-name branch, no backend-selection `#ifdef`, and no special single-thread execution path.

New reusable alpaka primitives for MMA, fragment operations and asynchronous copies are in scope. A native adapter is a general alpaka operation, not a backend GEMM hidden behind a common name.

Dispatch rule:

1. Define a symbol with `ALPAKA_FN_SYMBOL`.
2. Provide a generic `alpakaFnDispatch` fallback that works on every supported executor.
3. Specialize constrained overloads with `Symbol::Spec<api, deviceKind>`.
4. Keep backend headers and compilation guards in backend registration files.
5. Combine API and device-kind dispatch with legal capability, type and shape checks plus a tested fallback. Device-kind dispatch alone does not prove that a GPU generation supports an instruction.
6. Add a small non-GEMM example or unit kernel for each new public primitive, with fallback and specialized dispatch tests.
7. Document participation, memory ordering and numerical guarantees.
8. Keep unsupported native combinations out of template instantiation where possible.

Common GEMM code invokes only the portable symbol. A backend primitive implements one local or collective operation; it never contains a tiling loop, scheduler or complete GEMM.

Unsupported native capability selects the tested fallback. Unavailable hardware is recorded as BLOCKED_BY_HARDWARE and does not satisfy native validation.

## 6 Output regions, boundaries and K tails

The full output splits into disjoint regions: full rectangle, bottom strip, right strip and corner. The kernel launches nonempty regions as four launches. A tiny-problem policy may use one boundary-aware launch when dimensions and costs justify it. That choice depends on dimensions and costs, not backend identity. Total submission overhead is measured.

Boundary rules:

- Full and partial regions use fixed-size logical fragments; buffer dimensions need not divide the tile.
- Invalid M, N and K operands are tested before the load and become zero.
- Invalid output stores are suppressed.
- The implementation never performs an out-of-range load and masks the result afterward.
- No inactive output thread returns early before a collective.
- A remainder is never handled merely by a range-based `for` loop.

K-tail schedule. Let F be the number of full K tiles, R indicate one final partial tile, S the stage count, and L = F + (R ? 1 : 0). Each stage carries a generation counter in addition to its ring index.

```text
for i in [0, min(S, L)):
    acquire(i % S); fillBatch(i); commit(i % S)
for i in [0, L):
    waitReady(i % S, generation(i)); compute(stage(i % S))
    completeReaders(i % S); release(i % S)
    if i + S < L:
        acquire(i % S); fillBatch(i + S); commit(i % S)
finish()
```

`fillBatch(i)` calls the full-tile body for `i < F` and the explicit partial operation otherwise. It never passes a smaller tile to an unchanged fixed-size body. The schedule is tested for S = 1, zero full tiles, only a partial tile, fewer tiles than stages, exact stage multiples and repeated wraparound. A synchronous backend satisfies the same contract without promising transfer/compute overlap.

For finite operands, zero-filled K tails preserve the mathematical result. Exceptional-value tests remain separate because padded rows can still meet infinities in valid counterpart operands.

## 7 Stage contract

Every stage follows the same contract on every executor:

| Step | Meaning |
| --- | --- |
| acquire | Claim a free slot for a new generation and establish that prior readers have released it |
| fill | Write producer data for the current generation |
| commit | Publish the generation to consumers with the declared visibility |
| wait | Wait until the required generation is ready |
| consume | Read the stage in the compute phase |
| complete | Signal that all readers, including asynchronous matrix operations, finished the stage |
| release | Return the slot for reuse after completion and visibility are established |

Rules:

- Generation tracking prevents a stale completion signal from releasing a reused slot.
- Releasing means all readers, including asynchronous matrix operations, have completed.
- Waiting for a copy is not enough to publish every worker's copy to every consumer; the group visibility contract is explicit.
- Commit and wait participation stays converged.
- Per-worker remainder branches cannot change collective stage counts.
- The sequential executor completes a cooperative phase without waiting for unscheduled virtual workers.
- The hot path contains no hidden device-wide synchronization.

## 8 Planning facts, candidate family and pruning

Planning separates compile-time operation properties, runtime device facts and measured estimates. Every fact records provenance and units. Unknown is not zero, not unlimited, and not an invented hardware constant.

Facts:

- Existing: SIMD width per type, instruction parallelism estimate, cache-line bytes, shared memory per block, thread and grid limits.
- From the executor or device: group and subgroup widths, collective legality, CPU concurrency and topology.
- Proposed optional: L1, L2 and last-level capacity and sharing, register budget, compiled resource usage, active blocks, native matrix combinations, async-copy capability and alignment. The native matrix fact is never inferred from a backend name: section 11 records which hardware actually carries matrix instructions.

Missing optional facts fall back to conservative defaults. Global memory capacity is not free memory. Temporary allocations use a checked size, a bounded workspace policy and allocation failure handling.

Candidate construction:

1. Choose a legal numerical operation. Always keep common FMA/SIMD. An optional native operation must satisfy the requested numerical contract.
2. Let V be the usable SIMD width for arithmetic and accumulation, P the instruction parallelism estimate, and (qm, qn, qk) the operation shape quantum. Mixed inputs do not make the input SIMD width the right accumulation width.
3. Seed register-column candidates with small multiples of V, for example V, 2V and 4V. Seed register-row candidates with a small set, for example 1, 2, 4 and 8. These are search seeds, not optimality claims.
4. Build subgroup and block shapes from legal compute partitions with integral fragment coverage. Worker count is a tuning variable, separate from logical tile size.
5. Choose BK from small multiples of qk and transfer quanta. Evaluate S in {1, 2, 3} only when memory and progress constraints allow it.
6. Include generic direct-load and staged or packed policies, plus optional cache macrotiles. All policies are callable wherever their capabilities hold; no backend-name branch. Direct loading is an important small-problem candidate.
7. Prune by numerical validity, layout legality, address alignment, thread limits, scratch capacity, estimated register pressure, grid limits and code-size budget.
8. Rank survivors using problem shape, useful work, resource pressure and transfer cost. Return the reasons with the selected plan.

Resource models, with element sizes sA and sB and padding included:

```text
stageBytes = S * (BM*BK*sA + BK*BN*sB + paddingBytesPerStage)
AI_inputs  = 2*BM*BN*BK / (BM*BK*sA + BK*BN*sB)
usefulFraction = M*N*K / (roundUp(M,BM)*roundUp(N,BN)*roundUp(K,BK))
```

Add barrier state, epilogue scratch and other shared storage before checking the per-block bound. Per-SM occupancy does not follow from that bound alone. The generic per-worker TM x TN tile needs TM*TN accumulators plus A/B operands, prefetched fragments and temporaries. CPU vector-register demand for the output is roughly `TM * ceil(TN/V)` vectors, which differs from GPU scalar register demand. Replace estimates with compiled resource evidence where possible.

For CPU cache macrotiles, constrain packed panels to a conservative fraction of cache capacity available per sharing group. Tune that fraction empirically. Cache-line bytes do not give cache capacity. Avoid duplicating full packed panels per thread when common scheduling and queue-managed storage achieve the reuse.

The default selector is deterministic, needs no timing run and explains its choice. An optional bounded autotuner benchmarks the best few legal candidates, stays off by default, and has a trial, time and workspace budget. Its cache key contains device identity, compiled target, executor, compiler and library revision, numerical mode, shape class, strides and alignment class. Correctness inputs and outputs are never tuning scratch.

The offline candidate sweep is a diagnostic oracle inside the common-source family. Training shapes and held-out evaluation shapes stay separate. Heuristic constants freeze before held-out evaluation.

## 9 Public interface and diagnostics

The basic call exposes views, alpha and beta, the numerical mode and an optional tuning policy. The advanced API exposes an inspectable plan object and a validated expert override. Overrides pass the same legality checks as automatic plans. The advanced plan API stays stable enough for reproducible benchmarks.

Diagnostics report facts used, missing facts, selected engine, tile shapes, thread count, SIMD width, staging bytes, numerical mode, rejection reasons and fallback reason.

### 9.1 Benchmark workload and reporting

The frozen workload manifest covers square, rectangular, small-K, skinny, boundary and tiny shapes. The release suite covers all numerical modes where supported, all four transpose combinations, compact and padded strides, and representative alpha and beta combinations. CI runs a bounded fast subset. Capacity-skipped cases stay visible and never count as wins.

References are the M2 baseline, a manually indexed same-plan implementation, the best legal bounded-family candidate, and the installed vendor BLAS (CPU BLAS, cuBLAS, rocBLAS, oneMKL). Vendor libraries are references, never the implementation. Availability differs by target: cuBLAS 12/13 exists on both CUDA hosts, rocBLAS is absent on fwk388 and needs installation or an explicit documented omission, and oneMKL is not in the loader path on hal and needs verification. A missing vendor baseline is recorded with a reason, never treated as a pass. Precision and math mode, layout, dimensions, thread count, device and workspace limits match. Serial CPU compares with single-thread BLAS; parallel CPU compares with equivalent resources.

Measurements separate device time from end-to-end queue latency, cold planning from warm execution, and report median plus variability over at least 10 samples in at least 3 rounds. Throughput uses useful `2*M*N*K` operations, not padded work. Measured tables, plots, profiler outputs and acceptance reports stay outside the checkout and appear in the incremental PR description and dated progress comments.

## 10 Maintainability rules

- One common GEMM source and one vendor-symbol dispatch path.
- New public primitives are reusable outside GEMM and ship with a non-GEMM test.
- No second local-storage system. Reuse `scope.var()` and the lockstep concurrent path.
- No backend-specific tiling loop, scheduler or GEMM body behind a generic name.
- Every requirement in `validation/gemm-requirements.yaml` has an ID, one status and one owner role. The owner roles are api-ownership, numerics, sync-progress, cpu-codegen, gpu-native, benchmark-methodology and maintainability.
- Bounded template instantiation family, at most 64 candidate plans per operation, type and compiled target initially. Prune before expensive instantiation.
- Record compile time and binary growth. A budget increase needs review and measured benefit.
- Reproducible commands, plan diagnostics, numerical and boundary semantics, and reviewed tests ship in the repository.
- Measured data, plots, profiler outputs and performance reports stay outside the checkout. The repository carries tools, workload definitions and the acceptance checker only.
- One incremental draft PR stays open in `psychocoderHPC-bot/alpaka3`, receives coherent validated commits, and shows current concepts, correctness coverage, per-target tables and dated plots. It is never merged by the implementation agent.
- Review records contain source SHA, scope, inspected evidence, findings with severity, reproduction, resolution commit and verification status.
- The implementing author never labels self-review as independent.

## 11 Hardware and environment matrix

Inventory date: 25 October 2026, external run `gemm-20261025`. Every row cites the raw probe log under `/tmp/alpaka-orchestration/gemm-20261025/logs/`.

| Target | Required | Observed | Toolchain and profilers | Status | Evidence |
| --- | --- | --- | --- | --- | --- |
| hal/cuda | yes | NVIDIA A30 (Ampere, compute capability 8.0), 24576 MiB, driver 615.71.09 | nvcc 13.4, cuBLAS 13, ncu + nsys | available | `hwinv-hal.txt` |
| hal/hip | yes | AMD Radeon RX 7900 XTX (Navi 31, gfx1100, wave32, 48 CUs, ~24 GiB); `/dev/kfd` and renderD129 present; ROCm 7.2.0 at `/opt/rocm`, not on PATH | hipcc 7.2.26015, clang 22.0.0git; a compile+run properties probe passed | available | `hwinv-hal-amd-recheck.txt`, `hwinv-hal-hip-stack.txt` |
| hal/oneapi-cpu | yes | AMD EPYC 7452 32-core (avx2 only; no avx512f, avx512_bf16 or amx_tile) through Intel OpenCL CPU | oneAPI 2026.2, icpx 2026.2.0 | available | `hwinv-hal.txt`, `hwinv-amd-probe.txt` |
| hal/oneapi-gpu | yes | Intel Arc A770 (DG2) through Level Zero and OpenCL | oneAPI 2026.2, icpx 2026.2.0 | available | `hwinv-hal.txt`, `hwinv-amd-probe.txt` |
| fwk394/cpu | yes | Intel Xeon W-2175 (28 threads, avx512f; no avx512_bf16 or amx_tile) | g++ 13.3.0 | available | `hwinv-fwk394.txt` |
| fwk394/cuda | yes | GeForce RTX 2080 Ti (Turing, 7.5) and Quadro GV100 (Volta, 7.0) | nvcc 12.5, cuBLAS 12, ncu only (nsys absent) | available | `hwinv-fwk394.txt` |
| fwk388/hip | yes | AMD Radeon VII (Vega 20, gfx906, wave64); ROCm 7.2.0; connects on demand | hipcc 7.2.26015, rocprof + rocprofv3 1.1.0; rocBLAS absent | available, vendor BLAS absent | `hwinv-amd-probe.txt` |
| rosi_v100/cuda | optional | Tesla V100-SXM2-32GB (Volta, sm_70); no ROCm; connects on demand | NVIDIA driver visible; toolkit not inventoried | optional | `hwinv-amd-probe.txt` |
| jureca | no | unreachable after two dial attempts | not probed | BLOCKED_BY_HARDWARE | `hwinv-amd-probe.txt` |
| jureca-h100 | no | unreachable after two dial attempts | not probed | BLOCKED_BY_HARDWARE | `hwinv-amd-probe.txt` |

Facts that change coverage:

- The three distinct NVIDIA generations are 7.0 (GV100), 7.5 (RTX 2080 Ti) and 8.0 (A30).
- HIP has two endpoints: gfx1100 on hal (RDNA3, wave32) and gfx906 on fwk388 (Vega 20, wave64). Both are required for correctness and portability. Native matrix evidence is gfx1100-only; gfx906 has no matrix cores and must select the tested fallback.
- The oneAPI GPU device exists only on hal (Arc A770). It is not a required target on fwk394 or fwk388.
- No target exposes `avx512_bf16` or `amx_tile`; CPU FP16 and BF16 arithmetic is generic emulation everywhere.
- Native MMA scope is CUDA Volta/Turing/Ampere, oneAPI Arc A770 and HIP gfx1100. The record `G9-native-evidence` covers fp16 on all of them; `G9-native-evidence-bf16` covers bf16 on Ampere, Arc A770 and gfx1100, because Volta and Turing have no bf16 matrix instructions.
- ROCm sits at `/opt/rocm` on hal and no profile exports it. The inventory probed `lspci`, `/dev/kfd` and `/sys/class/drm`, which found the AMD device despite the missing PATH entry.
- jureca and jureca-h100 stay BLOCKED_BY_HARDWARE. They never count as a pass, and they are not required targets for this release.
- rocBLAS is absent on fwk388, so the HIP vendor baseline needs installation or a documented omission. A missing baseline fails G5 and G6 coverage for that target. oneMKL is not in the loader path on hal and remains unverified.

## 12 Milestone to evidence mapping

Every milestone records its source SHA, commands, external evidence links and reviewer decision.

| Milestone | Exit evidence | Requirement records | External evidence path |
| --- | --- | --- | --- |
| M0 Baseline | this design, the requirements file, API and hardware inventory, harness skeleton, draft PR | all records exist with IDs, status and a per-requirement owner; no record claims a pass | run root: inventory and baseline logs |
| M1 Lockstep and numerical foundations | audit regressions for items 4 and 6, FP16/BF16 conversion tests, all four modes, no duplicate local API | S8-STRUCT-simd-slots, S8-PIPE-state-machine, S8-NUM-exceptional, G1-numerical | run root: unit test logs and conversion vectors |
| M2 Correct common GEMM | exhaustive small and adversarial shapes against an independent reference on every available target | G1-bounds, G1-ownership, G1-queue-lifetime, S8-STRUCT-ownership, S8-MEM-canaries | run root: correctness logs per target and mode |
| M3 Layouts and register reuse | mapping bijection, generated-code evidence, operand reuse, spill check | G1-ownership, G3-abstraction-overhead, G8-resources | run root: disassembly, resource reports, explicit-plan comparison |
| M4 Shared storage and cache reuse | correctness across stage and tile boundaries, deterministic writes, bounded scratch, reuse evidence, sequential progress | G1-bounds, G8-resources, S8-PIPE-device | run root: stage matrix, scratch accounting, reuse measurements |
| M5 Heuristic planner | legal defaults without timing, minimal-fact fallback, held-out comparison, template and binary budgets | G4-heuristic-geomean, G4-heuristic-per-case, G10-maintainability | run root: selection logs, candidate sweep, build-size report |
| M6 Pipeline abstraction | prologue, steady-state, drain and generation tests for every stage count, no deadlock or premature reuse | S8-PIPE-state-machine, S8-PIPE-device, G1-queue-lifetime | run root: state-machine traces and device runs |
| M7 Native matrix operations | legal combinations on actual hardware, native instruction evidence, boundary GEMMs, correct completion and epilogue conversion | G2-vendor-symbol, G9-native-evidence, G9-native-evidence-bf16, G9-fallback | run root: per-target native correctness and instruction evidence |
| M8 Final portability and performance review | full suite on the final SHA and the acceptance report | G1 through G10, threshold block in the YAML | run root: acceptance.json and review records |

The M0 exit also needs a reference GEMM smoke test, a timing smoke test, a benchmark harness skeleton and a backend coverage manifest. This commit delivers only the design and requirement artifacts. M0 stays open until those items and the lockstep reconciliation land.

## 13 Gate to evidence mapping

All gates are pending. The required target list is hal/cuda, hal/hip, hal/oneapi-cpu, hal/oneapi-gpu, fwk394/cpu, fwk394/cuda and fwk388/hip. rosi_v100/cuda is optional coverage, not a required target. jureca and jureca-h100 are BLOCKED_BY_HARDWARE. All four numerical modes are required for correctness; native gates are scoped to native-capable targets and the modes those targets actually support.

| Gate | Record IDs | Passing evidence on the final SHA | Status |
| --- | --- | --- | --- |
| G1 Correctness | G1-numerical, G1-ownership, G1-bounds, G1-queue-lifetime | numerical, ownership, bounds and lifetime logs for every required target and mode | pending |
| G2 Common source | G2-common-source, G2-vendor-symbol | source-policy review, macro scan, dispatch tests | pending |
| G3 Abstraction overhead | G3-abstraction-overhead | matched explicit-plan benchmark within 5 percent median or latency analysis | pending |
| G4 Heuristic quality | G4-heuristic-geomean, G4-heuristic-per-case | held-out comparison to the best bounded-family candidate | pending |
| G5 Competitive large GEMM | G5-large-gemm | matched vendor ratio on large aligned square cases | pending |
| G6 Broad workload | G6-broad-workload, G6-floor-case | per-family matched vendor ratio and per-case floor check | pending |
| G7 Regression | G7-regression | statistically supported regression analysis against the accepted baseline | pending |
| G8 Resources | G8-resources | launch legality, scratch bound, resource and spill report | pending |
| G9 Native claims | G9-native-evidence, G9-native-evidence-bf16, G9-fallback | actual-hardware native correctness, instruction and performance evidence on native-capable targets, plus fallback tests | pending |
| G10 Maintainability | G10-maintainability | reproduction check, diagnostics snapshot, template family bound | pending |

The acceptance checker reads manifests and external results, rejects missing required entries, stale SHAs, unsupported substitutions, unresolved findings, failing thresholds and unapproved skips, then generates the final report. Reported completion separates FUNCTIONALLY_COMPLETE, PERFORMANCE_ACCEPTED and NATIVE_FEATURE_COMPLETE.

## 14 Frozen thresholds

The YAML `thresholds:` block is the canonical form. Summary:

- G3: median throughput ratio at least 0.95 on compute-heavy cases, 5 percent tolerance; small cases use confidence and absolute latency analysis.
- G4: held-out geometric mean ratio at least 0.95 to the best bounded-family candidate, each case at least 0.85.
- G5: large aligned square geometric mean vendor ratio at least 0.90, each case at least 0.80.
- G6: per target, mode and family geometric mean vendor ratio at least 0.80; any case below 0.50 is a blocking investigation.
- G7: no statistically supported regression above 5 percent.
- Planning: at most 64 candidate plans per operation, type and compiled target initially, with pruning before expensive instantiation.
- Build: compile time and binary growth are measured. The numeric absolute CI budget is UNVERIFIED and must be set from a measured base-revision build with the CI flags. The rule is frozen; the value is not. Measurement protocol: build base revision `6f13700c` with the project CI configuration and flags, record CI build wall time and the produced binary size, then freeze budget = measured baseline times a margin chosen before the measurement. No number is invented here.
- Latency-limited cases are marked and given a predeclared practical tolerance during M0. A case is never reclassified as latency-limited after it fails.

## 15 Unverified items and gaps

- The canonical base `6f13700c` and lockstep PR 7 head `05290c84` are verified. The PR is not merged and the open defect fixes are not implemented. The PR merge and the defect fixes precede kernel work.
- HIP is available on hal/gfx1100 and fwk388/gfx906. rocBLAS is absent on fwk388, so the HIP vendor baseline is unverified. oneMKL is not in the loader path on hal and is unverified.
- The gfx1100 WMMA compile probe failed for the exact builtin it tried (`__builtin_amdgcn_wmma_f32_16x16x16_f16` was rejected, with `__builtin_amdgcn_mfma_f32_16x16x16f16` suggested). The properties probe compiled and ran. Native HIP matrix evidence remains unvalidated: the failed probe reflects the builtin it tried, so record it as a tooling limitation.
- The numeric CI build budget and binary-size budget need a measured base-revision build. They remain UNVERIFIED in this commit; the measurement protocol is stated in section 14.
- The lockstep audit regressions for items 4 and 6 do not exist yet. No reference GEMM, timing smoke test, benchmark harness or backend coverage manifest exists either. M0 exit stays open.
- No independent review of this design has occurred.
- jureca and jureca-h100 stay unreachable and BLOCKED_BY_HARDWARE.

## 16 Gaps found in the guide

These are recorded as discovered work. This document does not change the guide.

1. Section 7 M0 exit requires every requirement to have an owner. The section 11 minimal record has no owner field. This file adds a per-requirement `owner` field and drops the earlier top-level default; the checker ignores it but preserves it.
2. Section 10 requires an absolute CI build budget frozen at M0 from the existing build baseline, but names no baseline measurement protocol, CI job or preset. Section 14 now states the protocol; the numeric value stays UNVERIFIED.
3. Section 10 requires a predeclared practical latency tolerance at M0 but gives no value or derivation rule.
4. Section 7 M0 lists a benchmark harness skeleton and a reference GEMM/timing smoke test, while the design artifacts alone cannot satisfy them. M0 remains open after this commit.
5. Section 5 pseudocode mixes existing lockstep names with proposed helpers. The boundary between verified API and design vocabulary needs an explicit inventory, which this file provides.
6. Section 13 asks the implementation to discover the current bot-repository base branch. The worker must read the live repository or a fresh clone, never the stale `gate` mirror that lags the canonical base by 30 commits.

## 17 Review record

Reviewer: none yet. Verdict: pending. No independent review of this design has occurred. Each requirement names its own owner role; the review record itself is owned by maintainability.

## 18 M2 as implemented

M2 adds the first correct common-source GEMM. It is a direct-load implementation with no shared-memory staging, no layout engine, no planner and no native matrix instruction. The public host header is `include/alpaka/onHost/algo/gemm.hpp`, the host-side implementation lives in `include/alpaka/onHost/algo/internal/gemm.hpp`, and the common-source kernels are in `include/alpaka/onAcc/gemm.hpp`. The public host entry point is

```cpp
template<typename T_Queue, typename T_C, typename T_A, typename T_B, typename T_Alpha, typename T_Beta>
void alpaka::onHost::gemm(
    T_Queue const& queue,
    alpaka::concepts::Executor auto const exec,
    GemmView<T_C> c,
    GemmView<T_A> a,
    GemmView<T_B> b,
    T_Alpha alpha,
    T_Beta beta,
    GemmNumericMode mode = GemmNumericMode::fp32);
```

The executor is optional. An overload without the executor, and the value `alpaka::exec::anyExecutor`, select the queue device's default executor. User-misuse checks throw `std::invalid_argument`, and internal invariants use `ALPAKA_ASSERT`.

`GemmView<T>` carries the logical rows/cols, leading dimension, row-major or column-major layout and the none/transpose flag. The call validates dimensions, leading dimensions, overlapping C/A or C/B storage, null pointers of views that are actually read or written, and checked 64-bit size and launch arithmetic before it enqueues anything. All four numeric modes are implemented. `fp16ToFp32` requires `alpaka::Float16` A/B storage and `bf16ToFp32` requires `alpaka::BFloat16`; both store C as FP32, convert the stored operand to FP32 exactly inside the common kernel, and accumulate in FP32. The mode and the view value types must agree. A mismatch, including FP32 operands passed with a mixed-mode tag or one 16-bit storage type used for the other mode's tag, throws before launch. The two 16-bit types are never treated as interchangeable and no backend branch performs the conversion.

Tile selection is a bounded compile-time family. The base tile size is `bit_ceil(getArchSimdWidth<Accumulator>)` clamped to {4,8,16,32}; it grows by one step when `getNumElemPerThread` and the problem shape justify it, and problems smaller than four rows or columns keep the 4x4 fallback. `BK` is 8 or 16 from the compiled SIMD width. The block thread count is a compile-time value from `getMaxThreadsPerBlock` and the SIMD/ILP traits, so host CPUs get one worker per block and a GPU would get a bounded power-of-two block. The executor may launch fewer blocks than tiles; each block then walks its assigned tiles in grid-stride order, and one tile is still processed at a time by a lockstep scope with one distributed accumulator fragment per worker.

Output tiles are split into the four disjoint regions of section 6: the full rectangle, the bottom strip, the right strip and the corner. Each region is a separate launch with its own tile origin; all four kernels are the same boundary-aware code. Coordinates are tested before any load. Invalid K contributes an exact zero, and invalid M/N suppresses the store. No collective exists in the kernel, so the lockstep tail callback can diverge freely.

Known limitations for M2: direct global loads only, one accumulator fragment per worker without register blocking or operand reuse, no packed panels or cache macrotiles, no plan diagnostics or expert override, no native matrix path, and no GPU measurement in this change. The benchmark adapter covers strict FP32 and the FP16-to-FP32 mixed mode on the host CPU.

## 19 M5 as implemented

M5 adds a host-only planner in `include/alpaka/onHost/algo/internal/gemmPlan.hpp`. The public descriptors (`GemmLayout`, `GemmOp`, `GemmNumericMode`, `GemmKernel`, `GemmStaging`, `GemmStagingConfig`, `GemmView`, `GemmPlan`) live in `include/alpaka/onHost/algo/internal/gemmTypes.hpp`, so the planner does not depend on the launch machinery. The register-blocked launch path now selects a plan through the planner and dispatches to the compiled family; the simple M2 path, the numerical contract and every kernel arithmetic statement are unchanged.

### Facts

Every fact carries a value, a provenance (`compiled-trait`, `device-property`, `estimate`, `unavailable`) and units. Unknown is an empty value with an explicit reason, never zero or a guessed constant.

| Fact | Source | Status |
| --- | --- | --- |
| `simdWidth`, `numPipelines`, `cacheLineBytes`, operation quanta | compiled traits | available |
| `sharedMemPerBlockBytes`, `maxThreadsPerBlock`, `maxBlocksPerGrid`, `multiProcessorCount`, `warpSize`, `globalMemCapacityBytes` | `DeviceProperties` | available |
| L1 capacity, L2/last-level capacity | no query in the project | unknown, reason recorded |
| register budget and compiled resource usage | no query in the project | unknown, reason recorded |
| active blocks per processor | no query in the project | unknown, reason recorded |
| native matrix combinations | no capability enumeration | unknown, reason recorded |
| async-copy capability and alignment | no query in the project | unknown, reason recorded |

The register-pressure model is a planner estimate, not a fact: `(TM*TN + TM + TN + 8) * accumulatorBytes` per worker. It uses `registerBudgetBytes` when a device reports one and the documented `estimatedRegisterBudgetBytes = 2048` otherwise. Both are labelled as estimates in the diagnostics.

### Candidate family and cap

The family is generated from dimensionless multiples of the compiled quanta: per-worker rows `TM` in {1,2,4,8}, per-worker columns `TN = V`, worker columns {1,16} clamped to the block-thread limit, `BK` in {8,16} multiples of the K quantum, and the policy direct or cooperative with stage counts {1,2,3}. The product is at most 64; `maxPlanCandidates = 64` is enforced by the array bound, a runtime guard and a `static_assert`. On a target whose block-thread limit is one, the worker-column axis collapses and the family has 32 distinct entries. The first entry is the minimal direct fallback (`TM=1`, `TN=V`, one worker, `BK=8` quanta), so width-one and tiny shapes always have a legal plan. Runtime problem dimensions never become template arguments.

Pruning checks, in order: structural tile/worker identities and ranges, SIMD pack coverage, K-quantum multiple, documented unrolled-work budget, numerical mode/storage signature, layout and transpose validity, known pointer alignment, block-thread limit, estimated register pressure, cooperative scratch (`stages * (BM*BK + BK*BN) * accumulatorBytes + 64` bytes, the M4 rule), grid-block limit and membership in the compiled family. Every rejection carries a reason.

### Ranking

Survivors are ranked by the section 6.3 models only: useful-work fraction, input arithmetic intensity `2*BM*BN*BK / (BM*BK*sA + BK*BN*sB)`, transfer cost derived from the same panel bytes, and resource pressure from the staging bound and the register estimate. Weights are frozen heuristic constants; ties break on tile M, tile N, K block, stage count, policy and compiled family index. No clock, random engine or filesystem state participates.

### Overrides and defaults

The advanced entry point accepts `std::optional<GemmPlan>` after the staging configuration. An explicit plan runs the same `validatePlan` as an automatic candidate and throws `std::invalid_argument` with the exact reason when it is illegal, outside the compiled family, or inappropriate for the problem. An explicit `GemmStagingConfig` restricts selection to that policy and stage count; the shipped automatic default remains the compile-time `defaultGemmStaging` (direct), so existing behaviour and the explicit cooperative override are preserved. Diagnostics (`PlanDiagnostics`) report the facts used, missing facts, engine, selected plan, tile and thread shape, staging bytes, numerical mode, per-candidate rejection reasons, fallback reason and override use. The planner was validated host-only; heuristic quality (G4) is measured later on hardware, not in this change.
