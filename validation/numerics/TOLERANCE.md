# GEMM numerical tolerance model

This note derives the forward-error bound implemented in `tolerance.hpp` and
states the operation counts, the safety factor, the separate input-quantization
report for the storage-only modes, and the limits of the reference. It is a
validation artifact for M1. It does not add a public library type or promise a
backend-specific error model.

## Contract being tested

For real dense GEMM,

```
C_ij = alpha * sum_k A_ik B_kj + beta * C_ij(old)
```

the strict numeric modes are FP32 (FP32 operands and accumulation), FP64 (FP64
operands and accumulation), and FP16/BF16 storage converted exactly to FP32
before FP32 arithmetic. FMA contraction and reordering within the accumulation
are allowed; a silent conversion to TF32 or reduced storage is not.

The reference in `reference_gemm.hpp` computes the same expression in a wider
format with a naive ascending-`k` loop. It shares no kernel code with the GEMM
under test.

## Forward-error derivation

Let `u` be the unit roundoff of the accumulation format and `n` the number of
rounding operations in the candidate's evaluation of one output element. A
standard rounding lemma gives, for `n*u < 1`,

```
|fl(x) - x| <= gamma_n * (sum of the exact term magnitudes)
gamma_n = n*u / (1 - n*u).
```

For an FMA dot product each of the `K` terms costs one rounding, so `n`
contains `K` accumulation steps. A separate multiply and add path costs `2K`;
that variant is selectable in the model and must be selected for a backend that
does not contract.

The magnitude scale of the measured expression is

```
S_ij = sum_k |A_ik * B_kj|
scale_ij = |alpha| * S_ij + |beta| * |C_ij(old)|.
```

The implemented per-element bound is

```
|candidate_ij - reference_ij| <= safety * gamma_n * scale_ij
                                 + (n + 1) * denorm_min(output).
```

The scale uses the exact values of the operands the candidate consumes, so S_ij
uses the quantized FP16/BF16 values in those modes. The bound is componentwise
and is asserted per element. Tests report the maximum absolute error, the
maximum `error/scale` ratio and the coordinate of the worst value.

Using `S_ij` rather than `|C_ij|` is what makes cancellation and scale-separated
inputs testable. A near-zero output with large `sum |A B|` has an uninformative
relative error but a meaningful forward bound. Tolerances are never relaxed to
make a case pass.

### Operation count

`ToleranceModel::effectiveOperationCount` returns:

| Path | Roundings |
| --- | --- |
| `K > 0`, `alpha != 0`, beta zero | `K + 1` (dot, alpha multiply) |
| `K > 0`, `alpha != 0`, beta one | `K + 2` (dot, alpha multiply, add) |
| `K > 0`, `alpha != 0`, beta neither zero nor one | `K + 3` (dot, alpha multiply, beta multiply, add) |
| `K == 0` or `alpha == 0`, beta one | `0` |
| `K == 0` or `alpha == 0`, beta zero | `0` |
| `K == 0` or `alpha == 0`, beta neither | `1` (beta multiply) |
| any of the above with output conversion | `+1` |

The non-FMA variant doubles `K`. The count is conservative for `alpha == 1` and
`beta == 1` because those multiplies are still counted. A larger count only
enlarges the bound.

### FMA contraction

The products `A_ik * B_kj` are contracted into the accumulation in the strict
FP32 and FP64 modes. One FMA has one rounding, which is exactly the `K`
roundings the model charges. Contracting is therefore allowed and is covered.
A separate multiply followed by an add has two roundings and must set
`separateMultiplyAdd = true`, giving `2K` accumulation roundings. The
alpha/beta multiplies are separate multiplies in both cases. Reordering the FMA
chain changes which terms are rounded together but not the count, so the same
bound applies.

### Safety factor

The default safety factor is 8. It covers the reference's own representation
error, a different but legal epilogue schedule (for example a fused
multiply-add in the epilogue that the count charges twice), and the fact that
the rounding lemma is a worst case that no real schedule reaches. The factor is
fixed before any measurement. It is not adjusted per case, shape or backend. A
failure stays a failure.

### Denormal floor

Each of the `n` roundings can turn a term smaller than the output's smallest
subnormal into zero, with absolute error below `0.5 * denorm_min`. The bound
adds `(n + 1) * denorm_min` to cover those terms and the final store. This
assumes gradual underflow. A build with flush-to-zero or fast math does not
satisfy the strict contract and needs a separate, explicitly named mode; the
tests do not claim to cover it.

## Per mode

Let `n` be the count from the table above.

| Mode | Accumulation `u` | Bound | Safety |
| --- | --- | --- | --- |
| FP32 | `2^-24` | `8 * gamma_n * scale + (n+1) * denorm_min(float)` | 8 |
| FP64 | `2^-53` | `8 * gamma_n * scale + (n+1) * denorm_min(double)` | 8 |
| FP16 -> FP32 | `2^-24` (accumulation) | same as FP32, against the quantized inputs | 8 |
| BF16 -> FP32 | `2^-24` (accumulation) | same as FP32, against the quantized inputs | 8 |

`gamma_n` is evaluated once per comparison with the candidate's K and epilogue
configuration. For FP64 the reference is `long double`, not a subtraction
against a rounded FP64 result.

### FP16 and BF16 input quantization

Input quantization error is reported separately and never mixed into `gamma_n`:

```
|reference_original - reference_quantized|
    <= |alpha| * sum_k ( |A_ik| |B_kj - B_quant,kj|
                       + |B_quant,kj| |A_ik - A_quant,ik| ) * 4
       + denorm_min(output)
```

The two references both accumulate the exact stored values in a wider format,
so the beta term cancels. `compareInputQuantization` implements this check with
safety factor 4. The first-order propagation is exact for the tested values
because every FP16/BF16 value and difference is representable in double. The
accumulation bound is checked against the quantized inputs only. Comparing a
candidate against the original FP32 inputs with the accumulation bound would
conflate two different errors, and the tests do not do that.

Input unit roundoffs are `2^-11` for FP16 and `2^-8` for BF16. They label the
modes and the report, not the accumulation bound.

## The `n*u >= 1` case

When `n*u >= 1`, `gamma_n` is infinite and the forward bound is uninformative.
The model reports `informative == false` and an infinite bound instead of
treating a vacuous comparison as a pass. In practice this needs
`K >= 2^24` in FP32 or `K >= 2^53` in FP64, so unit tests reach the branch with
`forcedOperationCount`, which is documented as a model test and not evidence.

The alternative validation for an uninformative bound is:

1. Use operands whose exact result is representable, for example small
   integers, powers of two, identity or diagonal matrices, and require bitwise
   equality with the higher-precision reference. The `n*u >= 1` test does this.
2. Require deterministic output for a fixed seed.
3. For production evidence at such K, compare sampled elements against a
   compensated or multiprecision reference and report the residual. That
   evidence is separate from this bound and is not claimed by these tests.

No test asserts an infinite bound as a pass.

## Reference precision and limits

* FP32 and converted FP16/BF16: reference accumulates in `double`. Every FP32
  operand and every FP16/BF16 value is exactly representable in double, so the
  reference contributes no input error. Double rounding of the accumulation is
  about `K*2^-53`, negligible against the FP32 bound.
* FP64: `gemm_reference_extended` accumulates in `long double` and is compiled
  only when `long double` has more mantissa bits than `double` (x86-64: 64
  bits, `u_ref = 2^-64`). Its error term is about `2^-11` of the candidate FP64
  bound for the tested sizes, so it is negligible. On platforms where
  `long double` equals `double`, the FP64 test skips explicitly
  (`hasExtendedPrecisionReference`) and a multiprecision reference is required.
  The tested FP64 sizes are bounded (`K <= 33`, matrices at most a few hundred
  elements) to keep the naive loop cheap. The same limit applies to any manual
  spot check.
* All reference paths are host-only and header-only. No extended-precision type
  enters the library API.

## What this model does not cover

* Fast math, flush-to-zero, or any mode that changes the declared arithmetic.
* NaN and infinity propagation, signed-zero results and exceptional-value
  ordering. Those belong to the separate exceptional-value requirement.
* Engine-specific native accumulators whose documented arithmetic is not
  FMA-like. They need an engine-specific bound and must not be labelled strict.
* Absolute performance and benchmark gates. Those use section 10 thresholds.

## Reproduction commands

```
g++ -std=c++20 -O2 -I validation/numerics \
    validation/numerics/verify_reference.cpp -o /tmp/verify_reference
/tmp/verify_reference > /tmp/gemm-reference.csv
python3 validation/numerics/verify_reference.py /tmp/gemm-reference.csv
```

The cross-check rebuilds the operands from the documented LCG, evaluates the
product with `math.fsum` (a correctly rounded, independent summation order) in
float64, and compares each exported element. It covers nine declared cases; the
`M == 0` case exports no elements by definition.
