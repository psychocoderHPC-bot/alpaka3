# T50d NARROW diagnostic — BF16 subnormal -> FP32 becomes 0 under icpx -O3

**Target:** `hal`, x86_64, `icpx 2026.2.0 (2026.2.0.20261001)`
**Artifacts:** `/tmp/t50d/repro.cpp` (single 30-line reproducer; also on `hal` at same path)
**Reference (correct = g++ -O3):** `0x0001->0x00008000  0x8001->0x80008000  0x007f->0x003f8000  0x0080->0x00400000`

## 1) Verdict

- **Path that loses the subnormal:** the FP **multiply / multiply-accumulate** (`widen -> x*0.5f` and `widen -> fmaf(x,0.5f,0.0)`). NOT the integer widening, NOT the bit->float conversion: `widen` (int `b<<16`) and `widenRT` (no FP op) are bit-correct in **every** configuration.
- **Mechanism (a compile-time transform):** default-model `-O3` icpx emits
  ```
  stmxcsr   ...
  orl  $32832   # 0x8040  -> MXCSR bit 15 (FTZ) + companion bit
  ldmxcsr ...
  ```
  in the process **entry** function, *and never restores it* (grep: exactly one poke, no restore; precise emits **none**). This enables process-wide hardware Flush-To-Zero / Denormals-Are-Zero, so the plain `mulss`/`vmulps`/`vfmadd*` then replace denormal inputs with 0 (or -0) and round underflow results to 0. It is *not* a pre-existing runtime mode: the thread-default MXCSR is clean (the team's 0x1fa2 reading is the pre-poke state, i.e. before the entry sets the bits in each optimized function).
- **Which flag restores it (at -O3):** YES — the FP-clarity flags. icpx **accepts** all of the following and they all restore correct bits:
  `-ffp-model=precise`, `-fp-model=precise` (single-f), `-ffp-model=strict`, `-fdenormal-fp-math=ieee`, `-fno-fast-math`.
  The flag-hits that do **NOT** fix it (still flushed): bare `-O3`, `-ffp-contract=off`, `-mno-daz-ftz`.
  icpx **rejects** the bare `-fdenormal-fp-math-f32=ieee` spelling; it works only as `-Xclang -fdenormal-fp-math-f32=ieee`.
- **The per-TU-vs-whole-build gap (why "reaches the TU but does not fix"):**
  The FTZ is set **process-wide** by the entry TUs. Compiling **only the GEMM TU** with `-ffp-model=precise` removes that TU's own poke but cannot undo the MXCSR state the entry already established. Demonstrated (two-TU build):
  | build | 0x0001 0x8001 0x007f 0x0080 |
  |---|---|
  | both default -O3 | 0 / -0 / 0 / 0  (flushed) |
  | both precise -O3 | 0x00008000 / 0x80008000 / 0x003f8000 / 0x00400000  (correct) |
  | **gemm precise + entry default** | **0 / 0 / 0 / 0  (STILL flushed)** |
  So `-ffp-model=precise` must be applied **build-wide (all TUs including the entry)** to have any effect.

## 2) Bits table (from `/tmp/t50d/repro.cpp`, icpx 2026.2, `hal` at -O3 unless noted)

| b  | widen (int) | widenRT (no op) | mul (x*0.5f) | acc (fma) | config        |
|----|-------------|-----------------|--------------|-----------|---------------|
|---|-------------|-----------------|--------------|-----------|-----|
|0x0001|`0x00010000`|`0x00010000`|`0x00000000`|`0x00000000`| icpx -O3 (default)  **[FLUSHED]** |
|0x8001|`0x80010000`|`0x80010000`|`0x80000000`|`0x80000000`| icpx -O3 (default)  **[FLUSHED=-0.0]** |
|0x007f|`0x007f0000`|`0x007f0000`|`0x00000000`|`0x00000000`| icpx -O3 (default)  **[FLUSHED]** |
|0x0080|`0x00800000`|`0x00800000`|`0x00000000`|`0x00000000`| icpx -O3 (default)  **[FLUSHED: product underflows to denormal]** |
|0x0001|`0x00010000`|`0x00010000`|`0x00008000`|`0x00008000`| icpx -O3 -ffp-model=precise  **[OK]** |
|0x8001|`0x80010000`|`0x80010000`|`0x80008000`|`0x80008000`| icpx -O3 -ffp-model=precise  **[OK]** |
|0x007f|`0x007f0000`|`0x007f0000`|`0x003f8000`|`0x003f8000`| icpx -O3 -ffp-model=precise  **[OK]** |
|0x0080|`0x00800000`|`0x00800000`|`0x00400000`|`0x00400000`| icpx -O3 -ffp-model=precise  **[OK]** |
|0x0001/8001/007f/0080| n/a | n/a | correct / correct / correct / correct | same | g++ -O3  **[OK — control]** |

Flag sweep (icpx -O3) for the `acc` (=fma) path:
| flag (besides -O3)     | 0x0001    | 0x8001      | 0x007f     | 0x0080     | result |
|-------------------------|-----------|-------------|------------|------------|--------|
| (default)               | 0x0       | 0x80000000  | 0x0        | 0x0        | FLUSHED |
| -ffp-model=precise      | 0x8000    | 0x80008000  | 0x3f8000   | 0x400000   | OK |
| -fp-model=precise       | 0x8000    | 0x80008000  | 0x3f8000   | 0x400000   | OK |
| -ffp-model=strict       | 0x8000    | 0x80008000  | 0x3f8000   | 0x400000   | OK |
| -fdenormal-fp-math=ieee | 0x8000    | 0x80008000  | 0x3f8000   | 0x400000   | OK |
| -fno-fast-math          | 0x8000    | 0x80008000  | 0x3f8000   | 0x400000   | OK |
| -Xclang -fdenormal-fp-math-f32=ieee | 0x8000 | 0x80008000 | 0x3f8000 | 0x400000   | OK |
| -ffp-contract=off       | 0x0       | 0x80000000  | 0x0        | 0x0        | FLUSHED |
| -mno-daz-ftz            | 0x0       | 0x80000000  | 0x0        | 0x0        | FLUSHED |
| -fdenormal-fp-math-f32=ieee (bare) | — | — | — | — | **REJECTED (compile error)** |

## 3) Assembly evidence (the transform)

`icpx -O3 -S` on `fl.cpp` (default): `main` entry (lines 10-12 / 18-20):
```
    subq    $24, %rsp
    stmxcsr 12(%rsp)
    orl   $32832, 12(%rsp)        # imm = 0x8040  <== enables FTZ (bit 15) + DAZ companion
    ldmxcsr 12(%rsp)
    ... callq pm / pa ...         # every FP op in the process then runs with FTZ|DAZ
pm:
    mulss   %xmm1, %xmm0          # plain scalar mul — under the set MXCSR, denormal -> 0
    retq
```
`icpx -O3 -ffp-model=precise -S` on the same file: **no `stmxcsr`/`or`/`ldmxcsr` anywhere**, and `pa` (the fmaf) becomes a tail call to the IEEE `fmaf@PLT`. That single difference is what restores denormals.

The "turn the value into zero" instructions are: the compiler-inserted `stmxcsr; orl $0x8040; ldmxcsr` (sets FTZ/DAZ process-wide) combined with the ordinary `mulss`/`vmulps`/`vfmadd` that then flush. Negative denormal preserves sign: `0x8001` -> `0x80000000` = -0.0. Smallest-normal `0x0080` is not an input denormal but its `*0.5` underflows to a denormal and its result is flushed to `0x0`.

## 4) Recommendation

Rebuild **all** targets (every TU, including the executable/entry) for icpx with **`-ffp-model=precise`** (equivalently `-ffp-model=strict`, or `-fdenormal-fp-math=ieee`). In CMake:
```cmake
# for every target so the entry TU does not leave FTZ|DAZ on process-wide
target_compile_options(<target> PRIVATE $<$<COMPILE_COMPILER:ICPX>:-ffp-model=precise>)
# CMAKE_CXX_FLAGS works too if the flag is IC-free:  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffp-model=precise")
```
`-ffp-model=precise` on **one** TU (the GEMM) is *not sufficient*; it only removes that TU's own poke and cannot undo the FTZ the entry TU set (see the two-TU "gemm precise + entry default ⇒ still flushed" row). Apply it build-wide.

If any entry/runtime TU cannot be switched (e.g. a third-party default-model TU, a dependency that sets FTZ in a constructor, or explicit `-ffast-math` in the entry), then **no per-TU FP flag can guarantee the result** — the behaviour is an icpx-specific, entry-set, process-global denormal flush, and it must be (a) **documented** and (b) **pinned by a unit test** that asserts the exact bits of known widened subnormals through the widen-then-mac path.

## 5) Support (header-only, no scale measured)
- Simplified to scalar `mul` + scalar `fma`; icpx's autovectorisation may issue `vmulps`/`vfmadd231ps` for the GEMM inner loop, but the MXCSR poke (not the whole SIMD form) is the one-and-only trigger (proven: two-TU test flushes with and without vectorisation because the FTZ is global).
- The two-TU link test proves the process-wide scope is the actual failure mode, not just a per-TU vector form.

## 6) Standing assumptions
- No `-ffast-math`, no `-fp-model=fast`, no target-specific denormal flag already set by the app; default `-O3` clean TU (default model = fast class the same class of FP semantics that opting into `-ffp-model=precise` overrides).
- The two-TU argument depends on shared MXSCR visibility across the process; no inter-thread/port context was in force.
