# FFT121 row-prologue evaluation

Decision: reject production retention. Two AVX2 row-prologue implementations
preserve the incumbent arithmetic and memory contracts and pass native tests.
The four-stream refinement improves N=121 on two AMD models, but repeated
full-family comparisons report unrelated regressions. The production kernel
therefore remains identical to main; this directory preserves the evidence.

## Specification and implementation

For stream s and pair m, input positions are `(m+1, 10-m)`, for m in 0..5.
Pair sums, signed differences, and DC accumulation retain the scalar expression
tree. All input reads precede output writes. Loads read exactly four complex
slots at positions 1..=4 and 7..=10, and stores stay within typed scratch rows.
Pair five, streams 8..11, unsupported-ISA declines, and other-stride fallbacks
remain unchanged. No FMA is introduced in the prologue.

The initial eight-row transpose is evaluated at `5da2d87b`. The four-stream
refinement at `834f2c0e` uses half-register component stores and removes
cross-128-bit exchanges. These implementations remain in Git history, not in
the final production tree. Both pass 5/5 native debug and 5/5 native release
tests, including finite-component scratch properties. Retained tests cover
25 input/output layouts, both directions, six signal families, offset/gap
guards, independent direct DFTs, and no-mutation unsupported-host declines.

## Measurement contract

The baseline is main `aa5adba1df3f318d47cb5e41770dff9d01099603`, after PR #649.
Benchmark sources, optimization profiles, toolchain, dependencies, address
lanes, timed closures, 26 lengths, and 100 samples stay unchanged. The package
patch version differs. Each campaign discards one full warm run per image,
then executes ABBA and BAAB on one available OS CPU. Processor selection in
the campaign and benchmark agree. Later campaigns use one executable launch
path to equalize argv[0] and AT_EXECFN. Images are hash-verified; observations
are always computed afresh. The native replicated comparator evaluates all
104 cases with its simultaneous median intervals and spread rule.

The table contains descriptive medians of the four run medians for each role.
Ratios are medians of per-run Apollo/RustFFT ratios. These summaries exclude
all warm runs and are not the statistical retention decision.

| Candidate | Hosted CPU | Main ns | Candidate ns | Main ratio | Candidate ratio | Supported N=121 gain |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Eight-row | Xeon 6973P-C | 330.709 | 301.368 | 1.876 | 1.643 | Yes |
| Eight-row | EPYC 7763 | 418.872 | 408.471 | 2.007 | 2.010 | No |
| Four-stream | EPYC 7763 | 425.633 | 395.976 | 2.041 | 1.924 | Yes |
| Four-stream | EPYC 9V74 | 366.612 | 345.709 | 2.404 | 2.239 | Yes |

The earlier issue result of approximately 291 ns / 1.62x is historical.
These hosted results do not revalidate the original Arrow Lake efficiency
core. The corrected main ratio depends on the runner; no universal ratio is
inferred.

## Retention and remaining bound

Every completed campaign reports supported movements in unrelated cases,
including RustFFT identity and short transforms. In the four-stream EPYC 7763
campaign, Apollo N=3, f64 N=11/67, and RustFFT f64 N=1/2 regress. The subsequent
EPYC 9V74 campaign still reports Apollo N=1/3 and RustFFT f64 N=1 regressions.
These findings are preserved, not removed from the comparator's family.

Disassembly of the validated images shows identical normalized timing-loop
instructions at code addresses shifted 32 bytes relative to 64-byte boundaries.
This establishes a placement confound; it does not prove the complete cause
of each regression. Engine-internal buffer addresses also remain uncontrolled.
The target gain alone does not close the declared broad performance-preservation
gate, so the production prototypes are removed.

The incumbent scratch occupies 1,936 bytes per eleven-stream call. Its k-dot
tail compiles to separate packed multiply/add plus a scalar final stream,
while the main eight-stream region uses FMA. Dense pair dots require
`4 * 5 * 5 = 100` real coefficient products per stream. RustFFT 6.4.1 also
uses dense pair symmetry in its [AVX column butterfly](https://github.com/ejmahler/RustFFT/blob/4758ab0dd6f256c50ac8987c75c9cb96152dc2ca/src/avx/avx_vector.rs)
and [packed 11-point leaf](https://github.com/ejmahler/RustFFT/blob/4758ab0dd6f256c50ac8987c75c9cb96152dc2ca/src/avx/avx32_butterflies.rs).
Rader or a minimal-multiplication formulation may change arithmetic cost,
but this study does not establish their throughput advantage. Resolve binary
layout attribution, then compare register-resident pair dots and exact-rounding
tail packing before choosing a mathematical replacement.

## Evidence

- [Eight-row Intel campaign](https://github.com/ryancinsight/apollo/actions/runs/38457429306): `eight-row-intel/`.
- [Eight-row AMD replication](https://github.com/ryancinsight/apollo/actions/runs/38461461780): `eight-row-amd/`.
- [Four-stream native evaluation](https://github.com/ryancinsight/apollo/actions/runs/38462868681): `four-stream-amd/`.
- [Four-stream replication](https://github.com/ryancinsight/apollo/actions/runs/38465965088): `four-stream-replication/`.

Each directory contains eight complete CSV reports, CPU and image identities,
exact source revisions, both comparison directions, and provenance.
`aggregate.json` records the descriptive summaries above. Binaries and discarded
warm runs remain in CI artifacts rather than the repository.
