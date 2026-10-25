# Lockstep integration provenance

This branch carries the lockstep scope primitives that the GEMM work depends on.
The section records where they came from and why they live here.

## Source

- Pull request: <https://github.com/psychocoderHPC-bot/alpaka3/pull/7>
- PR head: `05290c84b5c2349ff93585a95845e5471bba2540`
- PR base at the time of rebase: `8ae77718a57c38954bd5c51e49fb1194d8850085`
- Integration branch: `gemm/common-source`
- Integration base: `6f13700c390a09df818a255d53e730f1f9d69456`

All carried commits were authored by René Widera. The rebase kept those
authors. The committer is the workflow committer.

## Carried commits

| Original SHA | Subject |
| --- | --- |
| `eaba06b165e1e3b26db3f5c587f56a4409f1b571` | Refine lockstep API and kernel test |
| `a99e9feee0dde4fe9abf0ddc162ec2802d789776` | rebase lockstep on dev and fix interface changes |
| `2c3256ec4ea3c75e8c7392114acdeabd7a2d6dd9` | Fix lockstep SIMD tail bounds |
| `235d953fdb9292422663a415c36c37da8a1d21b7` | Add Doxygen documentation to lockstep public interfaces |
| `dd95acb0da44d1eeea1cdcd460ed38f6550a9a97` | Add lockstep tutorial and portable SIMD snippet |
| `31cc35fb4cb64eacc51e46858acf491018008763` | Add lockstep SIMD tail regression coverage |
| `0c1b9e3f695d70ab7ab19384753edf7de584cb63` | Document lockstep warp divergence and add direct includes |
| `d7abd2bd2432877763bdabba4dc687eb3c69fe65` | Fix lockstep device-code portability by taking the work group by value |
| `6dbcabb22cddf1b669a90391effd26b265f36fae` | Apply clang-format to lockstep changes |

## Rebase result

```text
git rebase --onto 6f13700c 8ae77718 05290c84
```

The rebase completed with no conflicts. The rebased series ended at
`6dbcabb22cddf1b669a90391effd26b265f36fae`. The series is squashed into the
integration commit `e1e90fc419f910f953e2f91af236e84a0850cd0d` so the branch
history states the provenance once and keeps the carried file set reviewable
as one change.

The rebased change set is 14 files: the eight headers under
`include/alpaka/onAcc/lockstep/`, `include/alpaka/onAcc/lockstep.hpp`,
`include/alpaka/alpaka.hpp`, `test/unit/kernel/lockstep.cpp`,
`docs/source/tutorial/lockstep.rst`, `docs/source/index.rst`, and
`docs/snippets/example/240_lockstep.cpp`. No file outside the lockstep scope
changed.

The rebase did not change lockstep behaviour. The tree is byte-identical to the
PR head for every carried path.

## Why the prerequisite is carried here

The GEMM milestones need the lockstep scope and SIMD primitives to build the
portable kernel. Waiting for PR #7 to merge would leave this branch either
blocked or stacked on an unmerged PR, and every later GEMM milestone would
inherit that stack.

The prerequisite is a pure addition: +1544 lines across 14 files with no
deletions. Its merge base with `dev` is `8ae77718`, and the dev commits after
that point touch a disjoint file set. The rebase therefore applies cleanly and
does not modify any dev behaviour. Carrying it here keeps `gemm/common-source`
independently buildable and lets the eventual PR #7 merge reduce to deleting
the duplicate commit from this branch.

## Related integration commits

| Purpose | SHA |
| --- | --- |
| Lockstep scope integration | `e1e90fc419f910f953e2f91af236e84a0850cd0d` (subject to the commit that adds this file) |
| GEMM design and requirements (T2) | `84d426d2c9cbbacb9735cb3c06492c9340a21385` |
| GEMM benchmark and acceptance harness (T4) | `dc30ea7e929bcce2069053aada90c741ee7c9628` |

The hardware and backend inventory that accompanies the design lives in
`design/backend-coverage.yaml`. No executor is marked passed there; the
manifest records the state after hardware discovery only.
