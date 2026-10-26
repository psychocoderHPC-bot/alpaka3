# Portable GEMM design and implementation guide for alpaka3

Version 2 — 25 October 2026

## 1 Purpose and binding decisions

Build a readable, reusable, high-performance GEMM implementation around alpaka3 lockstep. Preserve one common GEMM source across executors, derive configuration from capabilities and problem properties, and verify performance rather than assuming abstractions compile away. Implement strict FP32/FP64 and mixed precision in the project from its first correctness milestones. Evaluate a broad workload, not just aligned square matrices.

This is an implementation specification, not a claim that the proposed APIs exist or that performance targets have been achieved. All pseudocode is illustrative. Existing APIs are identified explicitly in section 2. No implementation, compilation, hardware benchmark, or independent review was performed as part of writing this guide.

### Confirmed requirements

- Use entirely common kernel code. No CUDA/HIP/SYCL/CPU-specific GEMM bodies, architecture-name branches, or special single-thread execution path in the kernel.
- Use the existing lockstep feature rather than inventing a competing register-variable/concurrent abstraction.
- Include FP32 and FP64 multiplication with same-precision accumulation, and FP16/BF16 input with FP32 accumulation. FP32 output is mandatory for mixed modes; optional lower-precision outputs must have explicit conversion semantics.
- Use a broad benchmark suite: square, rectangular, skinny, small-K, small matrices, and non-divisible sizes.
- Derive tile selection from device/executor capabilities and problem properties. A hand-picked tile for every GPU model is not acceptable.
- Specify intermediate validation, review, and machine-checkable final acceptance.
- Supported GEMM arithmetic is FP16/BF16, FP32 and FP64 as specified below; extended floating-point types are outside scope.
- Route every newly integrated specialized intrinsic through the alpaka3 vendor/function-symbol interface. No backend-selection `#ifdef` or native intrinsic calls in common GEMM code.
- Use computeMCP: hal for NVIDIA, AMD, and oneAPI CPU/GPU; fwk394 for CPU and NVIDIA. Inventory and test all three NVIDIA GPU generations available across these systems.
- Maintain one incremental PR in `psychocoderHPC-bot/alpaka3`, starting as a draft early and ending ready for review. Publish performance tables, concepts and plots in the PR description and progress responses; add fresh plots in chronological GitHub comments.
- Keep measured performance data, generated plots and profiler outputs out of the code base. Commit benchmark/test tools and workload definitions, not their generated results.
- The implementation agent must not ask questions or voluntarily stop before all completion criteria are fulfilled. Resolve routine choices from source, tests and measurements and continue until completion.

### Confirmed common source boundary

The user clarified that new reusable alpaka backend primitives for MMA and asynchronous copies are allowed, while the GEMM kernel and its control flow remain common. CUDA/HIP/oneAPI native adapters therefore belong in the implementation scope where hardware and toolchains support them. Backend-specific GEMM microkernels, tiling loops or schedulers remain prohibited.

Implement the common SIMD/FMA operation and synchronous staging first, then native adapters behind the same contracts. Reuse equivalent primitives already available at the implementation revision. New adapters must be general alpaka operations, not backend-specific GEMM implementations hidden behind a common name. Unsupported native capabilities select a tested fallback; unavailable hardware is BLOCKED_BY_HARDWARE and cannot count as native validation.

“Absolute top performance on every backend” cannot be guaranteed by a design or finite benchmark set. Operationally, maximize measured performance under the common-source and numerical constraints, meet the proposed gates below, and report every remaining gap. Do not lower numerical precision, switch to vendor GEMM internally, or weaken the source constraint to pass a benchmark.

## 2 Source baseline and existing functionality

Source inspection used `alpaka-group/alpaka3` dev commit `b7d339d07056a9a2a6c4051cc3927157bc5f0d51` and lockstep PR 7 head `05290c84b5c2349ff93585a95845e5471bba2540`. These are source snapshots, not a claim that either is the latest development location. Before implementation, resolve the authoritative upstream repository/branch, inspect AGENTS.md, record its SHA, and compare lockstep against the selected base. Do not blindly transplant the entire fork history. Keep integration fixes distinct from algorithm changes.

| Existing component | Observed behavior | Consequence for this project |
| --- | --- | --- |
| `onAcc::makeLockstep` | Creates a scope with compile-time logical extent and a worker group | Reuse it as the logical execution scope |
| `scope.var<T>()` | Per-worker `std::array`, requiring compile-time worker extent | Already provides persistent local values; register residency is not guaranteed |
| `scope.concurrent()` | Binds data/local slots to SIMD references and runs a generic callback | Already supports computation on local variables |
| Lockstep iteration | Flattened cyclic ownership: worker + slot × workerCount | Add layout flexibility only where a real GEMM requires it |
| Lockstep tail | Full SIMD packs followed by width-one invocations | Handles the declared logical extent, not arbitrary external-buffer boundaries |
| `MappedData` | Custom mapper machinery; public `map` helpers for identity and offset | Reuse and extend rather than duplicate; current mapped index type is constrained |
| `SimdAlgo::concurrent` | Explicitly unrolled pack callbacks when SIMD width exceeds one | Exposes ILP; does not promise all packs are loaded before compute |
| `SimdAlgo::calcSimdPackConfig` | Uses architecture width, cache-line bytes, and concurrency bytes | Reuse facts and design principles, not the element-wise model unchanged |
| Host `transform` | Calls `getSimdFrameSpec`, then `SimdAlgo` over `threadsInGrid` | Example of host selection plus a common kernel |

`getNumElemPerThread<T>()` is the product of `getArchSimdWidth<T>()` and `getNumPipelines()`. Here “pipelines” means an instruction-issue/ILP trait; it is not the number of asynchronous copy buffers. `getSimdFrameSpec` is explicitly intended for flat non-hierarchical execution. GEMM needs its own planner, sharing trustworthy facts with these functions.

The inspected `DeviceProperties` exposes global capacity, shared memory per block, multiprocessor count, warp size, and block/grid thread limits. It does not expose all the cache, register, occupancy, and matrix-operation facts needed below. New facts are proposed extensions, not existing fields. Compile-time SIMD traits describe the compiled target; do not assume they automatically detect every runtime CPU ISA.

The snapshot has CPU serial, OpenMP-block, TBB-block, CUDA, HIP, and oneAPI executor families. Re-enumerate the supported matrix at the implementation revision, including optional build configurations. “Unavailable in this environment” must never become “passed.”

### Lockstep audit before extension

Verify these issues from the actual code, not from contradictory historical bot reviews:

1. `BindValueType_t` uses `std::conditional_t`; check both alternatives are well-formed for ordinary data and local variables. Use lazy selection if necessary and add a regression test.
2. Check accelerator reference lifetime; reject accidental temporaries or document and enforce safe ownership.
3. Check const local-variable binding and `Simd<const T, W>` behavior. Read-only references should load ordinary value types and reject stores.
4. Test zero logical extent, zero worker count rejection, workers exceeding elements, odd extents, and width clamping. Do not assume `bit_floor(0)` itself is undefined; inspect how its result is used.
5. Verify variable ownership remains stable when successive passes choose different SIMD widths or first argument types.
6. Move callback participation restrictions into public documentation. A tail callback may execute a different number of times on different workers; block/warp collectives cannot be placed there arbitrarily.

## 3 Mathematical and API contract

The initial public operation is real dense GEMM:

`C := alpha * op(A) * op(B) + beta * C`, with A logically M×K, B K×N, and C M×N.

Support row-major and column-major positive-stride views, padded leading dimensions, subviews, and NN/NT/TN/TT transpose combinations. Reject invalid dimensions, overlapping output/input storage, unsupported strides, and overflowing address calculations before launch. Inputs A and B may alias each other because they are read-only. Negative strides, complex arithmetic, grouped GEMM, and batched GEMM are outside the first release unless added explicitly; small individual matrices remain in the benchmark scope.

Define zero-size behavior: M=0 or N=0 is a no-op; K=0 or alpha=0 computes beta*C without reading A/B. beta=0 does not read old C. beta=1 with K=0 is a no-op. Document validation requirements for unused views; do not infer that a null pointer is legal unless that API contract explicitly allows it.

Use sufficiently wide, checked host arithmetic for dimensions, byte sizes, pitches and launch counts. Device indexing must also represent the supported ranges. A 64-bit host shape with a silently truncated 32-bit kernel index is not support for large matrices.

The queue API is asynchronous in the normal alpaka sense. Planning, packing, scratch allocation, kernel launches, epilogue and scratch lifetime must respect queue dependencies. Never free temporary panels before all queued uses complete. No hidden device-wide synchronization in the hot path.

### Numerical modes

| Mode | Required arithmetic contract | Forbidden shortcut |
| --- | --- | --- |
| FP32 | FP32 operands and accumulation; FMA/reordering allowed and documented | Silent TF32 or FP16/BF16 conversion |
| FP64 | FP64 operands and accumulation | FP32 partial accumulation |
| FP16→FP32 | Stored FP16 inputs converted exactly to FP32 for generic arithmetic; FP32 accumulation | Changing input conversion/rounding without disclosure |
| BF16→FP32 | Stored BF16 inputs converted exactly to FP32 for generic arithmetic; FP32 accumulation | Treating BF16 as FP16 |
| Optional reduced FP32 | Explicit opt-in, separately named and benchmarked | Reporting it as strict FP32 |

Strict here preserves input precision, not a globally fixed summation order or cross-device bitwise equality. Specify alpha/beta arithmetic, output conversion, FMA contraction, denormals and exceptional-value behavior. Use round-to-nearest ties-to-even for required input/output conversions unless a separately named mode specifies otherwise. Default validation covers finite values without fast-math assumptions; if an engine flushes subnormals or differs for exceptional inputs, expose and test that contract or exclude it from the strict mode. Never label an unsupported FP64 device as an FP64 pass.

Audit existing FP16/BF16 types and conversions before adding anything. If missing, implement/review common value types and conversion tests as a prerequisite; do not make native matrix instructions a prerequisite for mixed-precision correctness. Test signed zero, ties, subnormals, overflow, infinities and NaNs according to the selected storage format.

## 4 Architecture and minimal extensions

Keep the design in five composable parts: views, ownership layouts, lockstep storage/iteration, matrix operations, and staging. The GEMM kernel composes them without querying backend names.

### 4.1 Ownership and memory layouts

Introduce a layout policy only after a first lockstep GEMM demonstrates the missing operation. It must define the mapping `(worker, localSlot) -> logicalCoordinate` and the inverse or equivalent validated ownership relation. SIMD packing is a separate grouping of local slots. A worker group identifies participants; it is not a memory level.

Required operations are compose, slice, transpose, and identify contiguous/broadcast access. Copy and compute layouts may differ. Physical shared-memory padding/swizzling is separate from logical ownership. A change in physical storage must not change the mathematical coordinates.

Provide compile-time validation for static layouts and exhaustive small-domain host tests. Every valid output coordinate must have exactly one owner, all owned valid slots must be in bounds, and holes must be marked inactive. Prove round trips where an inverse is supplied. Multiple readers of A/B are allowed; duplicate C owners are not.

The fallback layout must support serial execution and arbitrary supported SIMD width. Never equate warp size with vector width, or assume a warp is 32 everywhere. Keep the current cyclic layout as a compatible default.

### 4.2 Local fragments using lockstep variables

Extend `scope.var` or wrap it with non-owning fragment views rather than creating another local-storage system. A fragment carries logical shape, ownership, value type and layout identity. Provide fill, compatible element-wise concurrent access, row/column projections, broadcast, and load/store bindings. Static subfragment traversal should permit unrolling without forcing huge code growth.

A local broadcast may only read values owned by that worker. Cross-worker redistribution must be an explicitly named collective with a valid group and synchronization contract; it cannot be hidden inside an ordinary local view. Do not expose undefined native fragment element order as a portable indexing API.

At every pass, the same logical value must refer to the same storage regardless of SIMD pack width. Compile-time checks should reject incompatible scopes/layouts. Design lifetimes so no fragment view can outlive its owning variable unnoticed in documented usage.

### 4.3 Matrix operation contract

Implement a common register-blocked outer-product/FMA operation first. It explicitly reuses row and column fragments across several accumulator entries. Use a shared implementation of independent unrolling where beneficial, including width-one execution; evaluate an ILP extension instead of assuming SimdAlgo already unrolls that path.

An optional collective `matrixAccumulate(group, c, a, b)` contract declares legal operand/accumulator types, instruction shapes, group membership, layouts, alignment, completion semantics and numerical mode. Its fallback is common SIMD/FMA arithmetic. Native fragments may be opaque and require explicit conversion to a portable epilogue layout.

Native primitive implementations are authorized within the boundary in section 1 and MUST use alpaka3's vendor/function-symbol interface. The inspected tutorial defines `ALPAKA_FN_SYMBOL`, generic `alpakaFnDispatch` fallbacks, and API/device-kind `Symbol::Spec<api, deviceKind>` overloads. It explicitly supports onAcc calls with the required device-callable annotations. Verify the exact current accelerator dispatch syntax in the implementation/tests before coding.

Define reusable symbols for matrix accumulation, fragment operations and asynchronous staging operations where specialization is needed. Provide common fallbacks and constrained API/device implementations behind those symbols. Keep backend headers and necessary compilation guards in backend implementation/registration files. The common GEMM body invokes only the portable symbols and uses no backend-selection `#ifdef`. API/device-kind dispatch alone does not establish that a particular GPU generation supports an instruction: combine it with legal capability/type/shape checks and a tested fallback.

These functions must be reusable by unrelated alpaka applications. Add a small non-GEMM example or unit kernel for each new public primitive, test fallback and specialized dispatch, and document participation, memory ordering and numerical guarantees. Do not create a parallel private dispatch system just for GEMM. A backend primitive may implement one local/collective operation; it must not contain an entire backend-specific GEMM, tiling loop or scheduler hidden behind a generic name. A capability-based common implementation is acceptable; architecture-name branches are not.

Native adapter implementation checklist:

- CUDA: evaluate the installed toolkit's supported matrix primitives and supported datatype/shape combinations. Start with a synchronous operation whose participation contract is easy to verify; add asynchronous or larger-group operations only with explicit completion and ownership support. Do not assume all CUDA GPUs have the same matrix capabilities.
- HIP: evaluate the installed compiler intrinsics or a suitably licensed/versioned matrix library such as rocWMMA. Query/encode actual supported wave sizes and type/shape combinations; never infer support from a marketing family name alone.
- oneAPI: evaluate the installed joint-matrix facility and query legal combinations. Unsupported compiler extensions must not break builds of the generic implementation. Keep experimental APIs behind a narrow version-tested adapter.
- CPU: retain native SIMD/FMA as the required arithmetic path. Optional CPU matrix primitives require explicit capability/OS-state checks, legal packing and verified numerical semantics. They are not necessary for every CPU target.
- Each adapter supplies a capability descriptor, compile-time legality checks, a deterministic fallback, fragment load/accumulate/store or conversion, and unit tests independent of GEMM. The descriptor must distinguish per-thread, subgroup, workgroup and larger collective participation.
- A native accumulator is opaque where its physical element order is unspecified. Portable element-wise epilogues require a documented conversion/store path; benchmark its cost. Never apply arbitrary local-slot indexing to an opaque fragment.
- Native operand packing/conversion and synchronization are included in end-to-end timing. Record optional dependencies and ensure builds without them remain supported.

### 4.4 Staging and synchronization

Start with a synchronous stage policy. Define acquire, fill, commit, wait, consume, completion and release precisely. Releasing means all readers, including asynchronous matrix operations, have completed. Waiting for a copy is not necessarily enough to publish all workers' copies to every consumer; the group visibility contract must be explicit.

Shared/local storage and block barriers must use verified existing alpaka interfaces. A sequential executor executes a cooperative phase to completion without waiting for unscheduled virtual workers. Do not emulate a producer/consumer group with blocking waits that prevent the producer from running.

The optional asynchronous policy preserves the same stage contract and uses native acceleration only where permitted and available. Keep commit/wait participation converged as required by the native primitive; per-worker remainder branches must not alter collective stage counts. Ordinary block-local CUDA pipelines do not require a cooperative grid launch. No grid-wide synchronization is needed by this baseline GEMM. Split producer/consumer roles, cluster cooperation, and asynchronous matrix engines are later extensions requiring their own progress and completion tests.

### 4.5 Public interface and diagnostics

Provide automatic plan selection, an inspectable plan object, and a validated expert override. Diagnostics include facts used, missing facts, selected engine, tile shapes, thread count, SIMD width, staging bytes, numerical mode, rejection reasons and fallback reason. Overrides must undergo the same legality checks as automatic plans.

Do not expose implementation details in the basic call beyond views, alpha/beta, numerical mode and optional tuning policy. Keep the advanced plan API stable enough for reproducible benchmarks.

## 5 Common kernel and explicit boundaries

The following pseudocode defines responsibilities, not exact signatures. All collective operations are outside per-element lockstep tail callbacks.

```cpp
executeGemm(queue, problem, numericalMode)
{
    validate(problem, numericalMode);
    if(problem.M == 0 || problem.N == 0) return;
    if(problem.K == 0 || problem.alpha == 0) {
        enqueueScaleCWithoutReadingAB(queue, problem);
        return;
    }

    auto plan = choosePlan(queryFacts(queue), problem, numericalMode);
    dispatchCompiledPlan(plan, [&](auto staticPlan) {
        auto regions = splitOutputIntoDisjointRegions(
            problem.M, problem.N, staticPlan.BM, staticPlan.BN);
        // Full rectangle, bottom strip, right strip, corner.
        enqueueNonempty<FullMN>(regions.full, staticPlan, problem);
        enqueueNonempty<PartialM>(regions.bottom, staticPlan, problem);
        enqueueNonempty<PartialN>(regions.right, staticPlan, problem);
        enqueueNonempty<PartialMN>(regions.corner, staticPlan, problem);
    });
}
```

For tiny problems, a common dispatcher can use one boundary-aware launch instead of four. It must choose based on dimensions/costs, not backend identity. Record total submission overhead in performance results.

```cpp
template<class Plan, class Boundary>
kernel gemmTile(acc, problem, outputTileOrigin)
{
    auto team = Plan::team(acc);
    auto ownership = Plan::layout.partition(team, outputTileOrigin);
    auto scope = makeLockstepWithLayout(acc, team, Plan::outputShape, ownership);
    auto sums = makeAccumulatorFromScope(scope, Plan::matrixOperation);
    sums.fill(0);

    auto stages = allocateStages(acc, Plan::storageDescription);
    auto pipe = makeStageController(team, stages, Plan::stageCount);
    auto fullCount = problem.K / Plan::BK;
    auto tailK = problem.K % Plan::BK;

    auto fillFull = [&](auto stage, auto batch) {
        copyOperands<Boundary, FullK>(problem, outputTileOrigin,
                                     batch * Plan::BK, stage);
    };
    auto fillTail = [&](auto stage) {
        // Test coordinates BEFORE load; invalid operands become zero.
        copyOperands<Boundary, PartialK>(problem, outputTileOrigin,
                                        fullCount * Plan::BK, tailK, stage);
    };
    auto compute = [&](auto ready) {
        static_assert(Plan::BK % Plan::matrixOperation.kQuantum == 0);
        static_for<0, Plan::BK, Plan::matrixOperation.kQuantum>([&](auto k) {
            auto a = loadRowFragment(ready.A, ownership, k);
            auto b = loadColumnFragment(ready.B, ownership, k);
            matrixAccumulate(team, sums, a, b);
        });
        completeMatrixReadsBeforeStageRelease(team, sums);
    };

    runStages(pipe, fullCount, tailK != 0, fillFull, fillTail, compute);
    storeEpilogue<Boundary>(sums, ownership, problem);
}
```

Use fixed-size logical fragments for full and boundary tiles. Buffer dimensions do not need to divide tile dimensions. Boundary operations zero-fill invalid M/N/K inputs and suppress invalid output stores. Never compute an out-of-range load and mask its result afterward. Do not early-return inactive output threads before a collective. No remainder is handled merely by writing a range-based `for` loop.

### Explicit bounded stage schedule

Let `F` be the number of full K tiles, `R` indicate one final partial tile, and `S>=1` the stage count. `fillBatch(i)` selects `fillFull(i)` when `i<F`, otherwise invokes the explicit partial operation; it never passes a smaller tile to an unchanged fixed-size body.

```cpp
L = F + (R ? 1 : 0);
for(i = 0; i < min(S, L); ++i) {
    acquire(i % S); fillBatch(i); commit(i % S);
}
for(i = 0; i < L; ++i) {
    waitReady(i % S);
    compute(stage(i % S));
    completeReaders(i % S);
    release(i % S);
    if(i + S < L) {
        acquire(i % S); fillBatch(i + S); commit(i % S);
    }
}
finish();
```

Specify acquire/commit/visibility consistently for all participants. Track generations, not only ring-buffer indices, to prevent reusing a stale completion signal. Validate S=1, zero full tiles, only a partial tile, fewer tiles than stages, exact stage multiples and repeated wraparound. A synchronous backend satisfies the same contract without promising transfer/compute overlap.

For finite operands, zero-filled K tails preserve the mathematical result. Exceptional-value semantics need separate testing: padded rows/columns can still encounter infinities in valid counterpart operands; do not assume zero multiplication makes every exceptional path harmless.

## 6 Device-derived planning and heuristics

### 6.1 Facts and uncertainty

Separate compile-time operation properties from runtime device facts and measured estimates. Record provenance and units. Unknown is not zero, unlimited, or an invented hardware constant.

| Fact | Source or action | Use |
| --- | --- | --- |
| SIMD width per type | Existing compiled-target trait | Vector packing and register columns |
| Instruction parallelism estimate | Existing `getNumPipelines` | Initial independent accumulator chains |
| Cache-line bytes | Existing trait | Alignment and contiguous transfer quanta |
| Shared memory and thread/grid limits | Existing device properties | Hard legality checks |
| Group/subgroup widths and collective legality | Verified executor/device capabilities | Ownership and native operation participation |
| L1/L2/last-level capacity and sharing | New optional queries where available | Cache panel budgets, never hard correctness assumptions |
| Register budget and compiled resource usage | Optional hardware facts plus compiler reports | Candidate rejection and spill diagnosis |
| Active blocks/workgroups | Kernel-specific query where available | Refine occupancy estimate |
| Native matrix combinations | Optional capability enumeration | Filter shapes, types and numerical modes |
| Async copy capability/alignment | Optional capability enumeration | Stage policy selection |
| CPU concurrency/topology | Executor facts and optional topology queries | Avoid oversubscription and unsuitable packing reuse |

Do not treat global memory capacity as free memory. Temporary allocations need checked size, a bounded workspace policy and allocation failure handling. Planning must work with conservative defaults when optional facts are unavailable.

### 6.2 Candidate construction

Runtime dimensions cannot become arbitrary C++ template arguments. Compile a bounded family of generic configurations and dispatch to it. Generate candidates from dimensionless multipliers of SIMD/group/operation quanta, not a table keyed by GPU model. The family must include a correct small fallback for every supported type/executor.

1. Choose the legal numerical operation. Always retain common FMA/SIMD. Optional native operations must satisfy the requested numerical contract.
2. Let `V` be usable SIMD width for arithmetic/accumulation, `P` the ILP estimate, and `(qm,qn,qk)` the operation shape quantum. Mixed inputs do not imply the input SIMD width is the right accumulation width.
3. Seed register-column candidates with small multiples of V (for example V, 2V, 4V) and register-row candidates with a small set (for example 1, 2, 4, 8). These are search seeds, not optimality claims. Jointly consider A/B fragments and address temporaries.
4. Build subgroup and block shapes from legal compute partitions. Ensure integral fragment coverage. Worker count is a tuning variable distinct from logical tile size.
5. Choose BK from small multiples of qk and transfer quanta. Evaluate stages in {1,2,3} only if memory and progress constraints permit. Larger BK increases reuse opportunities but storage and tails also grow.
6. Consider generic direct-load and staged/packed policies, and optional cache macrotiles. All policies must be callable on every backend where their capabilities hold; no backend-name branches. Direct loading is an important small-problem candidate.
7. Prune by numerical validity, layout legality, address alignment, thread limits, scratch capacity, estimated register pressure, grid limits and code-size budget.
8. Rank survivors using problem shape, useful work, resource pressure and transfer cost. Return reasons alongside the selected plan.

### 6.3 Resource models

With element sizes `sA` and `sB`, staging count S and padding included explicitly:

`stageBytes = S * (BM * BK * sA + BK * BN * sB + paddingBytesPerStage)`.

Add barrier state, epilogue scratch and other shared storage before checking the per-block bound. Per-SM occupancy cannot be inferred from that bound alone.

For a generic per-worker TM×TN FMA tile, estimate scalar storage as `TM*TN` accumulators plus A/B operands, prefetched fragments and temporaries. CPU vector-register demand for the output is roughly `TM * ceil(TN/V)` vectors; it is not the same as GPU scalar register demand. Native fragment storage needs engine-specific resource information. Replace estimates with compiled resource evidence when possible.

Estimate block input arithmetic intensity as:

`AI_inputs = 2*BM*BN*BK / (BM*BK*sA + BK*BN*sB)`.

This excludes output traffic, cache reuse, packing and synchronization. Do not optimize AI alone. Account for the output grid `ceil(M/BM)*ceil(N/BN)` and insufficient parallel work, especially with skinny matrices. High occupancy is not itself the objective.

Padding efficiency is:

`usefulFraction = M*N*K / (roundUp(M,BM)*roundUp(N,BN)*roundUp(K,BK))`.

Compute it safely and only for nonzero dimensions. Penalize wasted work; for tiny/skinny shapes, prefer smaller tiles or direct-load policies. Compare kernel plus packing/dispatch cost, not just inner-loop throughput.

For CPU cache macrotiles, constrain packed panels using a conservative fraction of cache capacity available per sharing group. Tune that fraction as an empirical parameter. Cache-line bytes do not tell you cache capacity. Avoid duplicating full packed panels per thread when reuse can be achieved by common scheduling and queue-managed storage.

### 6.4 Determinism and tuning

The default selector must be deterministic, require no timing runs and explain its choice. An optional bounded autotuner can benchmark the best few legal candidates, but is off by default. Provide a maximum trial/time/workspace budget and an explicit cache key containing device identity, compiled target, executor, compiler/library revision, numerical mode, shape class, strides and alignment class. Never use correctness inputs/output as unprotected tuning scratch.

The offline candidate sweep is a diagnostic oracle within the common-source family, not the default implementation. Separate training shapes from held-out evaluation shapes. Freeze heuristic constants before evaluating held-out data. A fast tuned path does not excuse a poor default selector.

## 7 Implementation milestones and exit criteria

Commit reusable source, concepts, tests, benchmark drivers and workload definitions in the implementation repository. Generated measurements, plots, profiler outputs and performance-bearing review/acceptance reports belong outside the checkout and are published through the PR description, comments and attachments/artifact links. Every milestone records its source SHA, commands, external evidence links and reviewer decision.

### M0 Baseline and executable requirements

Deliver `design/gemm.md`, `validation/gemm-requirements.yaml`, a source/API inventory, backend coverage manifest and benchmark harness skeleton. Reconcile the lockstep PR with the selected base. Record the confirmed native-primitive boundary and hardware availability.

Exit: every requirement has an ID, owner, evidence link and status; a reference GEMM and timing smoke test run; the required computeMCP hardware inventory is recorded, including all three NVIDIA generations; unsupported/unavailable cases are explicit. Open or reuse the incremental draft PR with its initial concepts, milestones and measured baseline as soon as there is a coherent first commit. Do not wait for final performance before publishing progress.

### M1 Lockstep and numerical foundations

Resolve the section 2 audit with focused regressions. Add missing FP16/BF16 storage/conversion support only after checking existing facilities. Test all four numerical modes from this milestone, not only at the end.

Exit: mapping/width/const/lifetime tests and conversion tests pass; existing lockstep and SIMD tests still pass; no new duplicate local-variable API. Review is complete for numerical and ownership semantics.

### M2 Correct common GEMM

Implement a simple common kernel using lockstep accumulators, explicit output regions and K tail handling. Add transpose, strides, alpha/beta, queue lifetime and zero-size handling. Establish a reproducible baseline; optimize only after correctness.

Exit: exhaustive small matrices and randomized adversarial shapes pass against an independent reference on every available target; address/race checks pass where supported. Missing targets remain blocked in the coverage manifest.

### M3 Layouts and register reuse

Add only the layouts and local projections required to implement register-blocked outer products. Keep default lockstep behavior compatible. Hoist reusable A/B fragments and expose independent accumulator chains. Verify whether the compiler already eliminates repeated loads before expanding the API.

Exit: mapping bijection/coverage tests pass; generated-code evidence shows expected SIMD/FMA and intended operand reuse; compare against a manually indexed equivalent with identical tiles. No unexplained spills in selected large-tile configurations; explicit code must not win materially merely because abstractions prevented unrolling or alias analysis.

### M4 Shared storage and cache reuse

Add cooperative staging, transpose/padding layouts, and optional common packed-panel/cache-macrotile policies. Keep barriers outside divergent callbacks. Include packing/temporary storage in API timing. Test sequential executor progress explicitly.

Exit: correct across stage and tile boundaries, deterministic absence of duplicate writes, bounded scratch, and measurable reuse evidence. Retain direct-load plans when staging loses. A storage feature is not automatically selected on every device.

### M5 Heuristic planner

Implement the facts model, bounded compiled candidate family, legality pruning, cost ranking, diagnostics and expert overrides. Mock unknown and contradictory optional facts. Add held-out workloads and compare the default against the offline candidate sweep.

Exit: every default selection is legal, defaults need no timing run, fallback works with minimal facts, selection meets the heuristic gate in section 10, and the family meets compile-time/binary-size budgets.

### M6 Pipeline abstraction

Implement and test the synchronous stage contract first. Add native acceleration through reusable backend primitives where supported. Use state-machine tests plus real execution; do not rely only on timing to detect races. Keep producer/consumer partitioning out of the first implementation unless needed and reviewed.

Exit: prologue/steady-state/drain/generation tests pass for all stage counts; no deadlocks or premature reuse; profiler evidence supports any claimed overlap; the selector disables staging depth that regresses performance. Missing required native support is explicitly incomplete; synchronous correctness alone does not demonstrate acceleration.

### M7 Native matrix operations

Define and review the common operation/fragment contract. Add narrowly scoped adapters and compare native and fallback paths under the same numerical mode. Validate hardware, compiler and SDK support before instantiating a native combination. Keep unsupported paths out of template instantiation, not merely out of runtime execution.

Exit: legal group/layout/type combinations tested on actual hardware; native instruction use verified; full and boundary GEMMs pass; no collectives inside divergent tail callbacks; synchronous/asynchronous completion and epilogue conversion are correct. If a required target cannot be exercised, preserve the fallback and report the native validation milestone blocked; do not claim native completeness.

### M8 Final portability and performance review

Run the complete suite on the final SHA, rerun all affected checks after fixes, and produce the acceptance report. Provide minimal usage, advanced tuning, numerical guarantees, fallback diagnostics and reproduction instructions.

Exit: section 10 gates pass and independent review closes all required findings. If any gate is incomplete, publish a progress update with exact blockers, continue remediation and independent work, and do not announce completion or voluntarily end the task.

## 8 Correctness and intermediate tests

### Structural tests

- Exhaustively enumerate small shapes and worker counts; verify exactly-once C ownership and correct A/B coordinates for all layouts/transposes.
- Fill local slots with unique coordinate IDs; read them after passes at different SIMD widths and through compatible views. Test broadcast reuse and reject cross-worker access disguised as local access.
- Test lengths at width/worker/tile boundaries: 0, 1, q−1, q, q+1, 2q−1, 2q, 2q+1 for every relevant quantum q.
- Test planner facts with missing caches/register data, small memory, subgroup widths other than 32, width one, large accumulator types and maximum index values. Use rejection tests rather than actually allocating impossible matrices.
- Check async lifetimes with back-to-back operations, distinct queues where supported, scratch reuse and destruction of temporary host handles.

### Numerical tests

Use a separately implemented scalar reference and higher precision where feasible. For FP32 and converted FP16/BF16 inputs use FP64 reference accumulation on small/medium cases. For FP64 use a verified higher-precision library on bounded cases, or compensated/reference methods with documented limits. Use a test-only multiprecision reference where needed; do not add an extended-precision GEMM type or kernel. Large tests can use vendor comparison plus sampled high-precision elements and independent residual checks; sampled tests alone are insufficient.

Compare against the actual quantized FP16/BF16 inputs for accumulation correctness. Report input-quantization error against original FP32 separately. Test random signs, cancellation, zeros, identity, diagonal, rank-one, scale-separated magnitudes and reproducible seeds. Add exceptional values according to the declared numerical mode.

Define tolerances before benchmarking. Use a forward-error scale involving `sum_k abs(A_ik*B_kj)`, alpha/beta and output conversion, rather than relative error against a nearly zero output alone. For a classical accumulation model use `gamma_n = n*u/(1-n*u)` with a justified operation count and safety factor; when n*u>=1 the bound is uninformative and requires a different validation approach. Native engines need bounds appropriate to their documented arithmetic, not an assumed scalar FMA order. Record max absolute/scaled errors and worst coordinates; never loosen tolerances to make a failing candidate pass.

### Memory and concurrency tests

Use canaries and deliberately short/non-divisible views. Include misaligned subviews and padded leading dimensions. Run available host sanitizers and backend memory/race/synchronization diagnostics, with tool limitations documented. Add stress tests with many tiles, stage wraparound, low block residency and repeated launches. A green compiler build is not race validation.

For pipelines, simulate legal participant interleavings in a host state machine: no read before ready, no overwrite before release, matched commits/consumes, correct generations, and all initialized data visible. Then validate on device; the simulator is not a proof of hardware memory ordering.

## 9 Benchmark and profiling protocol

Freeze a versioned workload manifest before optimization. Suggested initial shapes (M,N,K) are below; generate boundary variations around every selected tile as additional tests.

| Family | Shapes |
| --- | --- |
| Square | (64,64,64), (256,256,256), (1024,1024,1024), (2048,2048,2048), (4096,4096,4096) |
| Rectangular | (4096,256,1024), (256,4096,1024), (2048,512,4096), (512,2048,4096) |
| Small K | (2048,2048,8), (2048,2048,32), (2048,2048,64) |
| Skinny | (1,4096,1024), (32,4096,1024), (4096,1,1024), (4096,32,1024) |
| Boundaries | (127,129,65), (513,511,257), (1023,1025,1027) |
| Tiny | (1,1,1), (8,8,8), (16,32,7), (31,17,33) |

Run all numerical modes where supported, NN/NT/TN/TT, compact and padded strides, and representative alpha/beta combinations. Use a bounded fast subset for CI and the full manifest for release validation. Larger sizes are optional capacity-qualified extensions; skipped capacity cases remain visible and are not included as wins.

Compare against: M2 baseline; a manually indexed same-plan implementation for abstraction cost; best legal candidate in the bounded family; and appropriate installed vendor BLAS (CPU BLAS, cuBLAS, rocBLAS, oneMKL). Vendor libraries are benchmarks, not the implementation. Match precision/math mode, layout, dimensions, thread count, device and workspace limits. Compare serial CPU with single-thread BLAS and parallel CPU with equivalent resources. Report different math modes in separate tables.

Measure both device/compute time and end-to-end queue-operation latency, including required packing, scratch management, boundary launches and epilogue. Separately report cold planning/compilation and warm execution. Prepacked reusable operands may have a separate mode with clear amortization; never compare it against an unpacked one-shot baseline without disclosing the difference.

Warm up, synchronize correctly, randomize implementation order, and collect at least 10 independent timing samples across at least 3 measurement rounds. Batch tiny operations to exceed timer resolution, subtract no speculative overhead, and report median plus variability/confidence. Prevent changing C from corrupting beta≠0 comparisons between repetitions; reset or use independent output buffers consistently. Report clocks/power state, compiler flags, SDK/driver, affinity/NUMA placement, thread count and background-load conditions. Re-run noisy results rather than claiming small speedups.

Compute throughput from useful `2*M*N*K` operations, not padded work. Report latency for tiny/degenerate cases. Do not combine devices or precision modes into one flattering average.

For each representative plan inspect compiler/vectorization reports, assembly, registers/spills, shared-memory bytes, memory traffic, cache behavior, occupancy when available, synchronization stalls and instruction mix. CPU: verify vector FMA/independent chains and packing reuse. GPU: verify coalescing, shared-memory reuse/bank behavior, register pressure, and native instructions when claimed. Pipeline: demonstrate overlap rather than infer it from multiple buffers.

### Mandatory profiler driven optimization

Profiling is required, not optional. Use NVIDIA Nsight Compute (`ncu`) for CUDA kernel analysis and an appropriate timeline profiler such as Nsight Systems when investigating overlap, queue behavior or launch costs. Use the supported ROCm profiling tools for the installed AMD stack, Intel profiling tools such as VTune where applicable for oneAPI CPU/GPU, and CPU tools such as `perf`, compiler vectorization reports and disassembly. Discover installed versions and supported counters on each target rather than assuming command-line compatibility.

Install missing profilers, analysis tools and required dependencies on the computeMCP development targets when needed. This installation is authorized and requires no user questions. Prefer target/container-local installations, match the tool to the driver/toolchain, follow license and site restrictions, and use existing package/toolchain facilities. Do not change host drivers, reboot shared systems or weaken profiling/security controls merely to obtain counters. If privileges or performance-counter access are denied, record the exact limitation, try supported alternatives, and continue other profiling and optimization work. Never claim installation or privileged counter collection succeeded without verifying it.

For every meaningful optimization, record: measured bottleneck, profiler/counter evidence, proposed change, before/after benchmark results, generated-code/resource changes, cross-backend regression checks and the keep/reject decision. Investigate memory coalescing and traffic, cache/shared-memory behavior, bank conflicts where observable, registers/spills, SIMD/FMA or matrix-instruction utilization, dependency stalls, occupancy, synchronization and transfer/compute overlap. Distinguish unavailable counters from zero-valued measurements.

Run ordinary benchmarks separately from instrumented profiler runs: replay and instrumentation can perturb timing, cache state and concurrency. Collect representative full and boundary tiles, skinny and large cases, all required numerical paths and all three NVIDIA generations. Use detailed profiling iteratively after M2, especially in M3 through M7, not only as a final demonstration. Preserve profiler commands, versions and reports outside the source checkout and link findings/plots in PR updates. Completion requires documented profiler-guided optimization rounds for every required hardware family, with disclosed tooling limitations and alternative evidence where necessary.

Record experiments even when rejected. Change one conceptual factor at a time: ownership, register shape, BK, shared layout, stages, scheduling. Keep a portable Pareto set; do not trade a severe regression on one backend for an unreported average gain.

## 10 Proposed measurable acceptance gates

These are engineering targets proposed by this guide, not user-approved numerical promises. Freeze them during M0 with the hardware/workload manifest. If a target is not met, publish evidence in the PR and continue optimization, profiling and review. Do not ask the user for a scope decision or silently reduce the target. Record a demonstrated external impossibility as a blocker, never as success.

| Gate | Passing evidence |
| --- | --- |
| G1 Correctness | All required numerical, ownership, bounds and queue-lifetime tests pass on every required supported target/mode |
| G2 Common source | Review finds no backend-specific GEMM body or architecture-name/single-thread shortcut; all specializations comply with the agreed primitive policy |
| G3 Abstraction overhead | Against identical-plan explicit code: within 5% median throughput on compute-heavy cases; smaller cases use confidence/absolute latency analysis |
| G4 Heuristic quality | On held-out shapes per target/mode: geometric mean throughput ratio to best bounded-family candidate >=0.95, and each case >=0.85 |
| G5 Competitive large GEMM | On large aligned square cases per target/mode: >=0.90 geometric mean of matched vendor throughput, with every case >=0.80 |
| G6 Broad workload | Per target/mode/family: >=0.80 geometric mean matched vendor throughput; any individual case below 0.50 is a blocking investigation, not hidden in an aggregate |
| G7 Regression | No statistically supported >5% regression versus the previously accepted common implementation; investigate exceptions rather than average them away |
| G8 Resources | No illegal launch or unbounded scratch; selected hot plans have no unexplained spills; report all packing and workspace costs |
| G9 Native claims | Each claimed native mode has actual-hardware correctness, instruction and performance evidence; unsupported modes must select fallback; untested supported modes cannot be counted complete |
| G10 Maintainability | Reproducible commands, plan diagnostics, documented numerical/boundary semantics, reviewed tests and bounded template instantiation family |

For very short operations, percentage thresholds become unstable: mark such cases latency-limited, report confidence intervals and absolute time, and use a predeclared practical tolerance during M0. Do not classify a slow case as latency-limited only after it fails. If vendor behavior differs numerically, omit that ratio with a reason and use matched comparable evidence; no missing baseline counts as a pass.

Proposed planning/build budget: <=64 candidate plans per operation/type/compiled target initially, with pruning before expensive instantiation where possible. Measure and record compile time and binary growth; set an absolute CI budget at M0 using the existing build baseline. Budget increases require review and measured benefit.

The final report must separate FUNCTIONALLY_COMPLETE, PERFORMANCE_ACCEPTED, and NATIVE_FEATURE_COMPLETE. Overall COMPLETE requires all applicable gates, all required hardware, closed review findings, and compliance with the confirmed native-primitive boundary. “Best so far,” “CI green,” and “all available machines passed” are not substitutes.

## 11 Review and evidence requirements

Review every milestone, including the baseline harness, numerical tolerance model and heuristic scoring. Use independent reviewers where available; the implementing author must not label self-review independent. This guide specifies review work for the implementation project and does not claim it has occurred.

Required review perspectives are API/ownership, numerical analysis, synchronization/progress, CPU SIMD/code generation, GPU memory/native operations, benchmark methodology, and maintainability/source-policy compliance. One person may cover multiple areas if qualified, but name the coverage and limitations.

A review record contains source SHA, scope, inspected evidence, findings with severity, reproduction, resolution commit and verification status. Resolve correctness, race, numerical-contract and unsupported-native-use findings before proceeding. After a fix, rerun affected tests and re-review the changed assumptions. At final acceptance rerun the full required matrix on the exact final SHA.

Evidence organization:

```text
# Committed reusable project content
 design/gemm.md
 validation/gemm-requirements.yaml
 benchmark/gemm/workloads.json
 benchmark/gemm/<drivers-and-plotting-scripts>
 validation/<acceptance-checker-and-tests>

# External run directory, outside every tracked source checkout
 <run-root>/<revision>/<target>/<mode>/samples.json
 <run-root>/<revision>/<target>/<mode>/profiles/
 <run-root>/<revision>/plots/
 <run-root>/<revision>/reviews/
 <run-root>/<revision>/acceptance.json
```

Publish measured results as Markdown tables in the GitHub PR description and user-facing progress/final responses. Publish plots as rendered images in the description and new dated progress comments. Attach raw data and detailed reports or use accessible run artifacts linked from those comments. Do not commit measurements, profiler traces, generated figures, benchmark numbers in READMEs, or performance reports to the source branch. The acceptance checker takes external evidence paths; its implementation belongs in the repo. Preserve raw samples durably before remote containers or job allocations disappear.

Each result row should contain revision, target, executor, toolchain, numerical mode, M/N/K, transpose/strides, alpha/beta, plan ID, actual resource usage, timings and variability, correctness error, baseline identity, and status. Store raw samples as well as aggregates.

A minimal requirement record is:

```yaml
id: G1-boundaries
requirement: Full and partial M/N/K regions compute every valid output once
status: pending
source_sha: null
required_targets: []  # populated and frozen at M0; empty is not accepted
modes: [fp32, fp64, fp16_fp32, bf16_fp32]
tests: [ownership, boundary_gemm, guarded_memory]
evidence: []
review: {reviewer: null, verdict: pending, findings: []}
```

Implement an acceptance checker that reads manifests and results and fails for missing required entries, stale SHAs, unsupported substitutions, unresolved required findings, failing numeric thresholds or unapproved skips. Generate the final report from those records. Evidence should be inspectable without trusting the agent's summary.

## 12 Agent execution rules and completion checklist

Start by reading the selected repo and lockstep code. Verify every API against source/tests; never invent a callable API because its name appears in this guide. Preserve user changes, use an isolated branch/worktree when appropriate, and open/update the implementation PR and its progress comments as authorized here. Do not merge the PR or publish a release; final completion means a reviewed, validated PR ready for the maintainer.

Proceed through M0–M8, keeping a working correct version. Do not ask the user questions, request routine confirmation, or stop at a plan, a partial kernel, a draft PR or a performance shortfall. Inspect source and available tools, choose defensible defaults, document decisions, and keep implementing, testing, profiling, reviewing and correcting until all gates pass.

If computeMCP disconnects, inspect session/job state, reconnect or resume through computeMCP, and continue without creating duplicate jobs. Diagnose environment failures and use the other required target for independent work while recovering. Preserve checkpoints and external evidence. Do not abandon the task because one route failed, and do not replace the common kernel with vendor GEMM to meet performance goals. Checkpoint the exact branch, commits, commands, pending tests and next actions so execution can resume after context or infrastructure interruptions.

No autonomous agent can guarantee access to unavailable hardware or override denied permissions. An unavoidable external interruption must be recorded truthfully in the draft PR/checkpoint; it is not task completion and does not waive any gate. Continue all feasible work and recovery. Never fabricate a passing result or run an unbounded destructive/retry loop to satisfy the persistence instruction.

Before declaring completion, the agent must answer yes with evidence to each item:

- [ ] Authoritative base and lockstep integration revisions recorded.
- [ ] Common-source boundary obeyed; all specialized intrinsics use the alpaka vendor interface with no backend-selection macros in GEMM.
- [ ] hal and fwk394 tested through computeMCP as assigned; all three NVIDIA generations inventoried and validated; supported executors and numerical modes enumerated.
- [ ] Strict and mixed-precision contracts implemented and tested.
- [ ] Lockstep variables/concurrent reused; duplicate abstraction avoided.
- [ ] Ownership, local reuse and boundary operations verified independently.
- [ ] Full M/N/K regions and every partial region correct, including zero sizes.
- [ ] Queue lifetime, synchronization and sequential progress verified.
- [ ] Deterministic heuristic selects only legal plans and passes held-out evaluation.
- [ ] CPU/GPU generated-code and resource evidence collected; ncu and appropriate AMD/Intel/CPU profilers used, missing tools installed where needed and feasible, and optimization decisions backed by before/after measurements.
- [ ] All required performance families measured against fair references.
- [ ] Applicable G1–G10 gates passed on the final SHA.
- [ ] Native functionality either fully validated or explicitly blocks the native milestone.
- [ ] Every required independent review finding closed and retested.
- [ ] Reproduction instructions and acceptance checker run successfully from clean checkout.
- [ ] One incremental bot-repo PR is ready for review with current concepts, final performance tables/plots and a chronological series of progress plot comments.
- [ ] Performance evidence appears in GitHub and responses, with no measured datasets, generated figures or profiler outputs committed to the code base.

## 13 Required execution environment and reporting workflow

### Required machines through computeMCP

| Target | Required validation |
| --- | --- |
| hal | NVIDIA/CUDA, AMD/HIP, oneAPI CPU and oneAPI GPU |
| fwk394 | CPU executors and NVIDIA/CUDA |

Use computeMCP for both systems. Discover available targets, sessions, compiler environments and device selectors; do not invent connection commands or assume previous sessions remain alive. Inventory GPU model, UUID/device ID, generation/compute capability, driver/toolkit, available memory, supported numerical modes and native primitives. Inventory CPU ISA, topology, SIMD target, affinity and compiler versions. Record oneAPI CPU and GPU results separately.

The user reports three NVIDIA generations available across the two systems. Discover the actual distribution and test each distinct generation; do not assume all three exist on each host or substitute one CUDA result for the whole family. Test all relevant NVIDIA devices on both machines, distinguishing repeated hardware from distinct architectures. Never guess model names. Maintain an external coverage matrix with passed, failed, unsupported and temporarily unavailable states. Missing access does not remove a required target.

### Repository and autonomous defaults

The PR destination is `psychocoderHPC-bot/alpaka3`. Discover its current development/base branch and the authoritative upstream alpaka3 lineage using remotes, branch history, existing PRs and project instructions. Use the current compatible development base and record the choice; do not ask the user. Check for overlapping GEMM work and reuse the task's existing PR if present. Integrate lockstep deliberately, with provenance and a separately reviewable integration commit when required.

Keep the specified strict and mixed modes. No extended-precision GEMM support is requested. FP32 output remains the default mixed-mode output; reduced FP32 modes must remain explicit opt-ins. Batched/complex GEMM and fused activations stay outside the initial scope. Use section 10 targets as working completion gates and select conservative bounded workspace/build defaults from available resources, documenting decisions without clarification questions. Default tile selection remains deterministic and heuristic; any diagnostic tuning is reported separately.

### Incremental PR lifecycle

1. Open a draft PR early, after the initial coherent commit and baseline harness. Do not defer visibility until all milestones pass. If the task already has a PR, update it instead of creating a duplicate.
2. Keep the same PR and branch throughout implementation. Push coherent incremental commits after each validated change/milestone; rerun affected checks and record the new SHA.
3. Update the description after each milestone and each meaningful benchmark round. During sustained work, post a concise progress update at least every two hours of active work, even when a long run has not finished. Avoid noise from individual test-case updates.
4. Each completed benchmark round adds a NEW chronological GitHub comment with its plots, timestamp, measured SHA, hardware/compiler identity, numerical modes, comparison baseline, workload and interpretation. Preserve prior comments/plots to show improvement and regression history. Do not repost old figures as new measurements.
5. The description always reflects the latest measured state: concepts/design, completed versus pending milestones, correctness coverage, per-target/mode performance tables, current plots, limitations, open findings and next steps. Label stale measurements with their actual SHA rather than implying they describe the current head.
6. Tables include useful GFLOP/s, latency, matched vendor ratio, baseline ratio, variability and verification status. Plots show throughput versus shape/size, ratios to fair references and progress across revisions; keep numerical modes and hardware distinct. Include boundary and small/skinny workloads, not just favorable square cases.
7. Upload figures using available GitHub attachment/artifact facilities and render them in comments/description through durable accessible URLs. Verify the links. Do not commit image files or result tables into the source branch to host them. If uploads are temporarily unavailable, publish the measured Markdown tables immediately and repair figure publication; plots remain an incomplete reporting item until visible.
8. Publish performance tables in user-facing progress/final answers too. Include the PR link, measured revision and evidence links. No claims of measured speedup without actual data.
9. At completion, refresh the description around the final implementation, concepts, final reviewed SHA, full performance tables and plots, and exact remaining unsupported hardware capabilities. Mark the PR ready for review only when all applicable gates pass. Do not merge it.

The publication actions in this workflow are authorized for the implementation agent. They do not mean that the guide-editing task itself has created or updated a GitHub PR.

## 14 Source references

Repository links below are pinned where source behavior was inspected. External native APIs are moving dependencies; the implementation must pin and re-check its installed SDK versions.

1. [Lockstep PR 7](https://github.com/psychocoderHPC-bot/alpaka3/pull/7).
2. [LockstepScope at inspected head](https://github.com/psychocoderHPC-bot/alpaka3/blob/05290c84b5c2349ff93585a95845e5471bba2540/include/alpaka/onAcc/lockstep/LockstepScope.hpp).
3. [LockstepVar](https://github.com/psychocoderHPC-bot/alpaka3/blob/05290c84b5c2349ff93585a95845e5471bba2540/include/alpaka/onAcc/lockstep/LockstepVar.hpp), [RegisterSimdRef](https://github.com/psychocoderHPC-bot/alpaka3/blob/05290c84b5c2349ff93585a95845e5471bba2540/include/alpaka/onAcc/lockstep/RegisterSimdRef.hpp), and [MappedData](https://github.com/psychocoderHPC-bot/alpaka3/blob/05290c84b5c2349ff93585a95845e5471bba2540/include/alpaka/onAcc/lockstep/MappedData.hpp).
4. [Host transform implementation](https://github.com/alpaka-group/alpaka3/blob/b7d339d07056a9a2a6c4051cc3927157bc5f0d51/include/alpaka/onHost/algo/internal/transform.hpp).
5. [SIMD frame selection](https://github.com/alpaka-group/alpaka3/blob/b7d339d07056a9a2a6c4051cc3927157bc5f0d51/include/alpaka/onHost/internal/interface.hpp).
6. [SIMD and ILP traits](https://github.com/alpaka-group/alpaka3/blob/b7d339d07056a9a2a6c4051cc3927157bc5f0d51/include/alpaka/api/trait.hpp).
7. [SimdAlgo configuration](https://github.com/alpaka-group/alpaka3/blob/b7d339d07056a9a2a6c4051cc3927157bc5f0d51/include/alpaka/onAcc/SimdAlgo.hpp) and [SimdConcurrent](https://github.com/alpaka-group/alpaka3/blob/b7d339d07056a9a2a6c4051cc3927157bc5f0d51/include/alpaka/onAcc/internal/SimdConcurrent.hpp).
8. [DeviceProperties](https://github.com/alpaka-group/alpaka3/blob/b7d339d07056a9a2a6c4051cc3927157bc5f0d51/include/alpaka/onHost/DeviceProperties.hpp) and [executor includes](https://github.com/alpaka-group/alpaka3/blob/b7d339d07056a9a2a6c4051cc3927157bc5f0d51/include/alpaka/api/executor.hpp).
9. [CUDA GEMM optimization worklog](https://siboehm.com/articles/22/CUDA-MMM): motivating example for register/shared-memory reuse, not a portable-performance guarantee or complete boundary implementation.
10. [NVIDIA pipeline documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html) and [CCCL memcpy_async](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html): staging and completion semantics.
11. [rocWMMA API reference](https://rocm.docs.amd.com/projects/rocWMMA/en/docs-7.14.0/api-reference/api-reference-guide.html): optional native adapter research; verify actual installed version and supported combinations.
12. [oneAPI joint_matrix extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc): optional native adapter research; combinations are device-specific and the extension is experimental.

13. [Vendor function interface tutorial](https://github.com/alpaka-group/alpaka3/blob/b7d339d07056a9a2a6c4051cc3927157bc5f0d51/docs/source/tutorial/vendorInterop.rst) and [dispatch example](https://github.com/alpaka-group/alpaka3/blob/b7d339d07056a9a2a6c4051cc3927157bc5f0d51/docs/snippets/example/200_vendorInterop.cpp): required dispatch mechanism for reusable specialized primitives.

## 15 Design review record

This version received an author consistency review against the inspected source and the selected requirements. Document checks verified the milestone/gate structure and balanced code fences. A small host model exercised 160 combinations of full-stage count, partial-stage presence and ring depth for the illustrative schedule, checking exactly-once consumption and safe modeled slot reuse. That model does not validate device synchronization, compiler behavior or GEMM correctness. This guide has not received an independent code or numerical review. No benchmark result is asserted.

Corrections incorporated into the design: lockstep already provides register-oriented concurrent operations; SIMD width differs from subgroup width; instruction-issue pipelines differ from asynchronous stages; transform's flat launch heuristic is not a GEMM tile heuristic; scalar SIMD tails do not solve external matrix boundaries; fixed native collectives cannot run inside divergent callbacks; runtime heuristics need bounded compile-time variants; and native adapters are authorized only as reusable primitives beneath common GEMM control flow.

Remaining risks requiring implementation evidence are compiler preservation of register reuse, layout/SIMD ownership stability, CPU progress and packed-panel reuse, resource estimates versus actual compiled usage, precision contracts of native operations, and performance portability across the final hardware matrix. Section 11 defines the independent review required before implementation acceptance.

Version 2 incorporates mandatory profiler-driven optimization and authorized tool installation, the user-specified vendor dispatch mechanism, computeMCP targets and three-generation NVIDIA coverage, external-only performance evidence, incremental bot-repo PR and plot comments, and autonomous execution without questions. The supported GEMM type set excludes extended floating-point types. The design revision does not assert that remote hardware has been inspected or that a PR has been opened.

## 16 Execution state and continuation (Version 3, 2026-10-26)

Self-contained continuation for this project. Read this section, the ledger and
RULES, and the cited evidence before acting. It records measured state, not new
claims.

### 16.1 Resume in five lines

1. Clone `psychocoderHPC-bot/alpaka3`, branch `gemm/common-source`; the head is
   `3b8e3e2b5072d10824bd9c0434df6c314b66240b` on base `dev`
   `6f13700c390a09df818a255d53e730f1f9d69456`; upstream is
   `alpaka-group/alpaka3`.
2. Ledger and decision log: `/tmp/alpaka-orchestration/gemm-20261025/LEDGER.md`;
   standing rules: `.../RULES.md`; measured evidence: `.../evidence/` (M1, T11c, T11b,
   B5c, B6, B8, B9b, O4, O5, O6, D1b, T39a, V6, V7, T50d, T51 via bundle, R1a, R1b). All
   evidence lives outside the checkout.
3. PR #60 on `psychocoderHPC-bot/alpaka3` is DRAFT, refreshed to the measured final
   state (permalinks clean, still draft). Do not mark it ready until the gates pass.
4. The measurement hosts (`hal`, `fwk394`, `fwk388`) may now be shared with other
   users, so every performance number in this section must be re-measured in an
   exclusive window before it is trusted; the exact counts and correctness below
   remain authoritative.
5. No benchmark, build or profile work may run in the Terok container (it is a
   laptop); all such work goes to the computeMCP targets.

### 16.2 Repository and publication mechanics

- Pinned refs: branch `gemm/common-source` head
  `3b8e3e2b5072d10824bd9c0434df6c314b66240b`; bot `dev`
  `6f13700c390a09df818a255d53e730f1f9d69456`; lockstep carried as
  `05290c84b5c2349ff93585a95845e5471bba2540`; PR #60 is draft. Remote branch history
  was once created via the Git Data API, so local and remote SHAs can differ while
  the trees are identical; compare tree SHAs, and the head above is the current
  single linear history on `6f13700c390a09df818a255d53e730f1f9d69456`.
- `git push` over HTTPS to the bot fork is refused (the clone-time credential is
  read-only for Git transport; `GH_TOKEN` authenticates API-only). Publishing uses SSH
  with `SSH_AUTH_SOCK=/tmp/ssh-agent.sock`, identity `tk-main:alpaka3-bot`
  (fingerprint `SHA256:K7qavOVpyBUtFnUTzf4E38iA/t0XZTE2TOlxbickq+M`). The Git Data
  API is retired except for read-only queries and must not be the routine publish
  path.
- Commit attribution, every commit in every repository: author and committer are the
  human identity `René Widera <r.widera@hzdr.de>` (never `opencode <noreply@opencode.ai>`),
  and the bot trailer is the final line:
  `Co-authored-by: psychocoderHPC-bot <330351933+psychocoderHPC-bot@users.noreply.github.com>`.
  A `Co-authored-by` followed by a blank line is prose, not a trailer, so strip
  trailing newlines and append exactly one: `msg.rstrip("\n") + "\n"`. Verify the
  GraphQL authors after publishing.
- Recovery when a worker committed only on a remote host: run
  `git bundle create X.bundle <base>..<head>` on the remote, download it with
  `computeMCP_file_download`, then `git fetch <bundle>`. Work that must be integrated
  has to exist as a commit reachable from a local repo the integrator can fetch, or the
  worker must leave a bundle and its path in the report.
- Canonical integration tree: `/tmp/alpaka-orchestration/gemm-20261025/canonical`,
  branch `gemm/common-source`. Exactly one writer on it at a time; publication never
  overlaps integration.

### 16.3 What is implemented, per milestone

- M0 Baseline, requirements and harness: `design/gemm.md`,
  `validation/gemm-requirements.yaml` (33 records), `design/backend-coverage.yaml`,
  `benchmark/gemm` harness plus the frozen workload manifest,
  `validation/check_acceptance.py`. Draft PR #60 opened at M0. Evidence:
  `T8a-remote-build.md`.
- M1 Lockstep and numerics: lockstep audit closed with the Defect A zero-worker
  guard and Defect B public participation docs, four audit regressions; FP16/BF16
  storage types and exhaustive conversion tests; numeric reference with the frozen
  tolerance model. Files: `include/alpaka/core/Float16.hpp`, `validation/numerics`.
  Evidence: `T1-lockstep-audit.md`, `T6c-cpu-workers.log`.
- M2 Correct common GEMM: one common kernel with output regions, K tail,
  transposes, strides, alpha/beta, zero sizes; mixed-precision modes
  fp16ToFp32/bf16ToFp32. Files: `onHost/algo/gemm.hpp`, `onAcc/gemm.hpp`. Evidence:
  `T39a-report.md` (CPU and first GPU).
- M3 Layouts and register reuse: the lane/slot invariant
  (worker, slot, lane) with the register-blocked fragment outer product, timed
  4x16xV=8 tile at 8 YMM live, no spill. File: `onAcc/fragment.hpp`. Evidence:
  `T11b-codegen/`, `T11c-1792950900/`.
- M4 Staging: cooperative stage contract with per-slot generations and bounded
  scratch. File: `onAcc/staging.hpp`. Evidence: `T11c-1792950900/` (staging effect
  measured in O4).
- M5 Planner: facts model, bounded compiled candidate family (cap 64), scoring,
  diagnostics and expert override; default resolution of the staging policy. Files:
  `onAcc/gemmPlan.hpp`, `onHost/algo/internal/gemm.hpp`. Integrated tip
  `bfac7feff886d5dc8e791b5f97a069c9275aabac` (recorded in the ledger as the M5
  integration tip).
- M6 Pipeline: `BoundedStagePipeline` with closed-form schedule counts and a selected
  async hook that runs the identical synchronous schedule. File:
  `onAcc/gemmPipeline.hpp`. Evidence: T15 and T42 (state machine only; GPU overlap
  unproven).
- M7 Native matrix operations: PARTIAL. M7a ships the matrix contract plus a CUDA
  m16n8k16 warp-fragment adapter proven by SASS on Ampere (A30 sm_80 shows
  HMMA.16816; smaller gens fall back by construction); fp16/bf16-to-fp32, strict
  fp32/fp64, tri-state capability fallback. Files: `onAcc/matrix.hpp`,
  `api/cuda/matrix.hpp`. The head `3b8e3e2b5072d10824bd9c0434df6c314b66240b` carries
  M7a. M7b (the kernel consuming the warp-fragment contract) is open: the GEMM tile
  does not yet use native fragments, so native matrix throughput is not in the final
  numbers.
- M8 Final portability and performance review: acceptance machinery and a draft
  report exist (`validation/check_acceptance.py` hardened, `M8a` requirements
  populated). The final single-SHA acceptance rerun (all G1 through G6 cells on one
  published tip, independent review closure) is pending.

### 16.4 Measured results

All GPU numbers below were collected in exclusive windows under an exclusive policy.
The hosts are shared again, so re-measure before trusting any of them.

(i) Correctness matrix (counts from the final-SHA validation V6/V7 and
`T39a-report.md`). FP64 runs on CUDA (80-bit `long double` reference) and skips with
the printed reason on HIP and oneAPI.

| Target (device) | Executor | Mode | Count at `f008d8ed` (draft SHA in ledger) |
| --- | --- | --- | --- |
| A30 | cuda | fp32 | 30 pass / 185172 assertions |
| A30 | cuda | fp64 | pass (CUDA only) |
| A30 | cuda | fp16, bf16 | pass |
| 2080 Ti | cuda | fp32, fp64, fp16, bf16 | 20 pass / 123448 assertions |
| GV100 | cuda | fp32, fp64, fp16, bf16 | 20 pass / 123448 assertions |
| gfx1100 | hip | fp32, fp16, bf16 | 27 pass, 3 fp64 skip |
| gfx1100 | hip | fp64 | skip (no extended `long double`) |
| gfx906 | hip | fp32, fp16, bf16 | 18 pass, 2 fp64 skip |
| Arc A770 | sycl | fp32, fp16, bf16 | 36 pass, 4 fp64 skip |
| Intel CPU | sycl | fp32, fp16, bf16 | pass, fp64 skip |
| host CPUs | openmp, omp, tbb, serial | all four modes | 612 to 1028 ctest pass |

Worst scaled errors (within the frozen bound): fp32 2.79e-07, fp64 2.32e-16,
fp16 1.60e-07, bf16 9.04e-08 on GPU and CPU.

(ii) CPU performance (hal EPYC 7452, AVX2, `-march=native`, g++ 13.3, OMP 4.5,
schedutil governor, taskset 0-63, load below 0.56, 10 samples x 3 rounds;
direct-load policy; exclusive window): 154 GF/s at 256^2 and 235.6 GF/s at 1024^2
(64 threads); 8.18 GF/s single-thread. The flat per-thread ~4.9 GF/s figure from the
baseline x86-64 build (SSE V=4, no FMA) is superseded and must not be quoted.

(iii) GPU performance, vendor ratio (ratio = ours / matched same-GPU vendor, fp32,
NN, alpha=1 beta=0, 10 samples x 3 rounds).

| GPU | SHA of tips | 256^2 | 1024^2 | 2048^2 | geomean (1024,2048) |
| --- | --- | --- | --- | --- | --- |
| 2080 Ti (Turing) | O5 `d4328ab1`, O6 `3b8e3e2b5072d10824bd9c0434df6c314b66240b` | 0.048 (B8) | 0.289 (B8) | 0.318 (B8) | 0.304 |
| GV100 (Volta) | same | 0.053 (B8) | 0.230 (B8) | 0.455 (B8) | 0.324 |
| A30 (Ampere) | O5 `d4328ab1` | 0.06 (B9b) | 0.26 (B9b) | 0.37 (B9b) | - |

(iv) Staging/tile crossover and BK (all at tip `d4328ab1` unless noted; 2080 Ti
fp32): cooperative staging loses for small tiles (256^2 -6.5%, and -20% at 1024/2048
for the 8x64 tile in O4) but wins for the 128x128 cooperative plan (O5), which closes
much of the large-GEMM gap; crossover 512^2 cooperative 645.2 vs direct 406.8.
O6 at `3b8e3e2b5072d10824bd9c0434df6c314b66240b`: BK>8 is scored, not illegal; adding
BK32/st1 to the cooperative family gave +3.3% at 1024^2 and +2.4% at 2048^2.
Remaining gap is occupancy, instruction scheduling and the shared-memory panel model
(roughly 3.5x), not the K block.

### 16.5 Key findings that must not be rediscovered

- Execution model: worker = execution thread (CPU = OS thread/frame with a fixed
  thread extent of 1, so parallelism comes from frames/blocks, not W; GPU = CUDA/HIP/
  SYCL thread). lane is V consecutive columns of the worker row; V = 8 on AVX2 float,
  V = 1 on GPU. `scope.var` needs a compile-time CVec worker extent, so it only
  supplies threadsInBlock (W=1); threadsInGrid extents are runtime. A 128x128 tile on
  one AVX2 worker would need 2048 YMM, 128x the register file, so per-thread sub-tile
  stays about 64 accumulators; reach big tiles with frames/threads.
- The six lockstep audit items (section 2 above), outcome at D11: items 1, 2, 3, 5 OK
  as-is; item 4 zero-logical-extent is safe but zero worker count hit an
  unguarded `divCeil(N, 0)` in consteval (fixed by the Defect A guard); item 6 the
  no-collectives rule lived only in a private doxygen note (fixed by Defect B
  public docs).
- `CpuOmpBlocks` cannot grow worker columns, so tile size is the only CPU lever that
  scales a worker's register tile; CPU parallelism comes from frames (OS threads), not
  from a larger per-thread W.
- Cooperative staging measurement (O4/O5): staging is slower than direct for small
  tiles (256^2 -6.5%, and -20% at 1024/2048 for the 8x64 tile in O4) but wins for the
  128x128 cooperative plan (O5), which closes much of the large-GEMM gap.
- icpx -O3 with the default FP model sets FTZ and DAZ in MXCSR once per entry TU and
  never restores it, a process-wide effect that flushes FP32 subnormals; this is the
  root cause of the BF16 subnormal defect on host CpuSerial (subnormals widen to FP32
  denormals, then multiply/accumulate to 0). The fix (T50e) pins `-ffp-model=precise`
  build-wide in `cmake/alpakaCommon.cmake` (Intel/IntelLLVM, not fast-math), with a
  contract test that pins the exact subnormal bits.
- The benchmark adapter originally hardcoded a host device and silently produced
  fake GPU numbers; it is fixed so requesting a non-CPU device on a build without it
  fails loudly and records device identity (T60).
- gfx906 has no matrix cores and the ROCm 7.2 prebuilt ships no rocBLAS gfx906
  Tensile kernel, so HIP on gfx906 is the generic FMA fallback with no fair vendor
  baseline.
- gfx1100 hardware PMCs (GRBM, SQ_INSTS, SALU/VALU/L2, OCC) read zero silently;
  gfx1100 performance claims must rest on timing and derived metrics and disclose this
  counter limitation (gfx906 returns real counters).
- `rosi_v100` cannot execute CUDA (`libcuda.so.1`/`libnvidia-ml.so.1` are not in the
  exec namespace, so cuInit fails and only -j1 builds; its parallel build deadlocks);
  treat it as a demoted/build-only host.
- The `fwk394` container has zero effective capabilities (`CapEff=0`), so NUMA binding
  fails with EPERM; build there with `-Dalpaka_HOST_NumaCpu=OFF`.
- `symbol=perf` attribution can point at a non-executed kernel instantiation because
  the binary instantiates every candidate; prove the executed kernel by template
  arguments and the layout, not by sampled symbol names.

### 16.6 Environment and target inventory

- hal: AMD EPYC 7452 (Zen2, AVX2, no AVX-512/BF16, 64c, 251Gi); NVIDIA A30 Ampere
  sm_80, driver 615.71, CUDA 13.4, cuBLAS 13; AMD RX 7900 XTX gfx1100 24GiB, ROCm
  7.2.0 (tools under /opt/rocm, not on PATH; rocBLAS 5.2.0 plus rocprof installed and
  verified); Intel oneAPI 2026.2 CPU and Arc A770 (level zero), oneMKL 2026.2,
  OpenBLAS; ncu + nsys. Roles: CUDA 13.4 A30, HIP gfx1100, oneAPI Arc and CPU, plus
  exclusive CPU/GPU benchmark windows.
- fwk394: Intel Xeon W-2175 (Cascade Lake, AVX-512F, no bf16/AMX, 28c, 125Gi);
  RTX 2080 Ti Turing sm_75 and Quadro GV100 Volta sm_70, driver 555.52, CUDA 12.5,
  cuBLAS 12.5, ncu only, no nsys, no oneAPI; OpenBLAS. Build with
  `-DCMAKE_CUDA_ARCHITECTURES=70;75` and `-Dalpaka_HOST_NumaCpu=OFF`. Roles: CUDA 12.5
  Turing and Volta correctness and GPU benchmarking.
- fwk388: Ryzen Threadripper 1950X (Zen1, 32T); AMD Radeon VII gfx906 wavefront 64,
  ROCm 7.2.0, hipcc 7.2, rocprof + rocprofv3, no rocBLAS for gfx906. Connects on
  demand. Role: gfx906 HIP fallback correctness (optional host).
- rosi_v100: Xeon 6136 48c, 4x Tesla V100 SXM2 sm_70, CUDA 12.6, ncu only, nsys
  absent. BLOCKED for GPU execution (missing `libcuda.so.1`) and parallel builds
  deadlock; kept only for parallel LLVM builds if it returns.
- Build options recorded: CUDA arch flags per host as above, `-march=native` opt-in
  via `ALPAKA_GEMM_NATIVE_ISA` (host TUs of the GEMM tests and benches only), Intel
  `-ffp-model=precise` build-wide pin, oneAPI sourced via `source
  /opt/intel/oneapi/setvars.sh`. No target exposes AVX-512 BF16 or AMX, so all
  CPU FP16/BF16 arithmetic is generic emulation and must be reported with that
  caveat.

### 16.7 Gate status G1 through G10

- G1 Correctness: LARGELY EVIDENCED. GPU and CPU matrix green at `f008d8ed` (V6: one
  nvcc-13.4 build gap in the test-only pipeline target, T52; V7: gfx906 all pass).
  Not declared complete until the final single-SHA rerun on the final tip with all
  target/mode cells (including Arc and gfx1100 full counts) re-confirmed.
- G2 Common source: PASS by independent review (R1a/R1b). No backend `#ifdef` or
  architecture-name branch outside capability checks; specializations use
  `ALPAKA_FN_SYMBOL`.
- G3 Abstraction overhead: PASS (T11c on hal, exclusive window). The
  register-blocked common kernel is equal or faster than the manually indexed
  reference; abstraction cost is not the problem.
- G4 Heuristic quality: NOT EVALUATED. Held-out shape sweep versus the best
  bounded-family candidate has not been run on any target.
- G5 Competitive large GEMM: FAILED. Measured geomean of matched-vendor throughput is
  about 0.30-0.32 (2080 Ti, GV100, A30) against the 0.90 target, with large-shape
  per-case ratios in 0.26-0.46. Valid at `d4328ab1` (O5) and
  `3b8e3e2b5072d10824bd9c0434df6c314b66240b` (O6); hosts now shared, so the ratio
  must be re-measured, but the honest verdict is failure.
- G6 Broad workload: PARTIAL. GPU numbers cover square 256/1024/2048 fp32 only;
  remaining families (rectangular, small-K, skinny, boundaries, tiny, mixed
  precision) and other devices are not measured.
- G7 Regression: NO PREVIOUS BASELINE. There is no previous accepted common GEMM to
  regress against, so the gate is vacuous until a baseline exists.
- G8 Resources: PARTIAL. No illegal launch or unbounded scratch observed; the
  pipeline state machine is covered by tests; full GPU resource evidence (registers,
  spills, occupancy) is incomplete.
- G9 Native claims: PARTIAL, Ampere-only. The CUDA m16n8k16 warp-fragment contract is
  proven by SASS on A30 sm_80; the kernel does not consume native fragments yet (M7b
  open), so native throughput is not reflected, and HIP/oneAPI native matrix is not
  claimed (gfx906 has no matrix cores; gfx1100 adapter unverified).
- G10 Maintainability: PARTIAL. Reproducible commands, plan diagnostics and reviewed
  tests exist; the documentation of FP model contracts and boundary semantics and the
  final bounded-family size are not yet complete.

### 16.8 Ordered next steps to finish

1. Close the remaining GPU performance gap from an `ncu` profile of the 128x128
   cooperative kernel: occupancy, instruction scheduling and the shared-memory panel
   model (the roughly 3.5x, per O6). Subsystem: `onAcc/staging.hpp`, planner scoring,
   and the register-blocked kernel in `onAcc/gemm.hpp`. Proof: A30 and 2080 Ti
   2048^2 fp32 ratio above 0.90 on re-measurement (G5).
2. M7b: make the GEMM tile and epilogue consume the warp-fragment contract
   (`onAcc/matrix.hpp`) for fp16/bf16-to-fp32 on Ampere. Subsystem: `onAcc/gemm.hpp`,
   `onAcc/matrix.hpp`. Proof: full and boundary fp16/bf16 GEMM correct and at least
   competitive with the vendor on A30, with HMMA still present in SASS (G9).
3. GPU numbers for the remaining devices: oneAPI Arc A770 (hal) and gfx1100 (hal).
   Subsystem: `benchmark/gemm` device driver. Proof: full G1 counts plus the fp32
   256/1024/2048 ratios versus matched vendor (G1, G5/G6 completion).
4. Matched vendor baselines for oneAPI (oneMKL on Arc and CPU). Subsystem:
   `benchmark/gemm/vendor` functor wiring. Proof: geom-mean matched-ratio table per
   target/mode (G5/G6, oneAPI cells no longer vendor_unavailable).
5. G4 held-out sweep. Subsystem: `benchmark/gemm` sweep plus an offline candidate
   oracle in the common-family. Proof: held-out geometric-mean ratio at least 0.95
   and every case at least 0.85 per target/mode.
6. Final-SHA full acceptance rerun. Subsystem: `validation/check_acceptance.py` and
   the frozen manifest. Proof: `acceptance.json` where every required cell passes on
   one published tip with evidence paths that verify.
7. Close the open review findings (for example the nvcc-13.4 pipeline-test build per
   T52; the FTZ-broken bf16 test reference under FTZ; the stale-SHA link fixes in PR
   #60). Subsystem: test and header files named in the trace above. Proof: each
   finding has a resolution commit, retested in the final matrix.
8. Mark PR #60 ready for review only after the gates pass. Subsystem: PR #60.
   Proof: all applicable G1-G10 gates pass on the final SHA; otherwise keep it draft
   with the exact blockers published.

### 16.9 Standing execution rules

- Remote-first and parallel: run builds, tests and benchmarks on the computeMCP
  targets (`hal`, `fwk394`, `fwk388`), not in the Terok container, which is a laptop
  limited to editing, git, pre-commit and light inspection; never put two
  measurements on one target.
- No performance measurement in the Terok container; timings, throughput, GFLOP/s and
  profiler runs are produced only on remote targets and anything local is invalid as
  evidence.
- Exclusive measurement windows bounded at 45 minutes, with a device gate that aborts
  as soon as the harness reports a CPU executor for a requested GPU target; record the
  start/end load average and that no other job ran.
- One writer on the canonical tree at a time; publication never overlaps integration,
  and publication precedes hardware validation so every GPU number cites a published
  SHA.
- Never kill, `pkill` or watch a process you did not start, and never touch relays or
  container services on any target; never run SysV IPC manipulation such as
  `ipcrm`.
- Never fabricate a number; unavailable cells are recorded vendor_unavailable or
  not_run, never a pass, and never average across devices or precision modes.
- Tolerances are frozen (`validation/numerics/TOLERANCE.md`): a failing case is a
  kernel or plan defect, never a tolerance to relax.
- Vendor libraries are benchmarks and never the implementation: no backend GEMM body,
  no architecture-name branch and no single-thread special path in the common kernel
  (specialized intrinsics go behind `ALPAKA_FN_SYMBOL` with a tested fallback).
- Evidence lives outside the checkout and does not commit measured data, plots or
  profiler output.

### 16.10 Known-bad and void results (do not reuse)

- B3: the two "GPU" tables (RTX 2080 Ti and GV100) are host CPU numbers on the
  fwk394 Xeon from the hardcoded-host-device adapter; the GV100 rows are identical to
  the 2080 Ti rows to the millisecond, which is impossible. Void.
- B4: its records carry `executor=alpaka-host-cpu-m2` with an empty device, and ncu
  and rocprofv3 captured zero GPU kernel launches; these are per-toolchain host CPU
  numbers and burned hal for three hours. Void.
- T11b overlapped-window run: two concurrent `gemm_bench` processes per mode wrote the
  same output dir under rising load; every number from that run is void.
- T11b throughput rows collected under load above 60: void, except its G3
  conclusion and its hang report, which are kept.
- The reported 12-hour hang during 2048-serial at `f008d8ed` was a slow kernel on
  odd thread pinning rather than a deadlock; the 12-hour number must not be cited as a
  G1 blocker, although the hang report itself is kept as a reliability signal.
- The `perf` attribution to a non-executed kernel instantiation (A1): every candidate
  is instantiated, so a sampled kernel symbol is not proof of the executed plan;
  attribute the executed kernel by template arguments, not by `symbol=perf` names.
