# ADR 0049: Fused twiddle multiplication and register transpose

- **Status:** Rejected
- **Date:** 2026-09-05
- **Class:** [patch]
- **Item:** [APOLLO-FOUR-STEP-PROFILE](../../backlog.md#apollo-four-step-profile)

## Context and hypothesis

The generic FourStep decomposition multiplies intermediate samples by cached
complex twiddles while transposing them. Its 16-by-16 scalar tiles read rows
contiguously and store at a matrix-row stride. At lengths 65536 and 262144,
rows have 256 and 512 elements. Routing and workspace ownership remain governed
by [ADR 0039](0039-one-dimensional-power-of-two-routing.md) and
[ADR 0048](0048-worker-scratch-lifetime.md).

A test-only phase diagnostic at `b7d52d45` places the three layout phases at
80–92% of N=262144 latency across f32/f64 and queried core classes. The fused
multiply-transpose accounts for 34–40%. Row measurements include submission,
execution and join; separate direct Stockham measurements do not isolate join
cost. Caller affinity does not control Moirai worker placement.

The experiment replaces the fused scalar tile with a Hermes lane kernel:
contiguous source and twiddle register loads, native complex multiplication,
pair-preserving register transpose, and contiguous destination stores. Apollo
owns the twiddle mathematics; value-preserving layout remains Leto's concern
under [ADR 0040](0040-leto-fft-layout-ownership.md). Existing Hermes `Simd` views
and `ComplexReg` provide the operations without a new public seam.

The operation has six real arithmetic operations and at least three complex
transfers per element: 0.125 FLOP/byte for f64 and 0.25 for f32. The hypothesis
therefore concerns memory movement; vector arithmetic alone does not establish
benefit. Separate multiplication and transpose would add a full-buffer pass.

## Acceptance and measurements

Acceptance requires a supported complete-transform improvement in the unchanged
engine census, no supported regression, unchanged warm allocation bounds, and
no executable-size growth. Analytical complex-product bounds, independent
spectra and normalized inverses establish numerical behavior. Native tests and
census runs retain their committed 60-second termination bounds.

The corrected candidate passes Clippy and 546 release library tests, including
existing analytical FFT suites. The diagnostic's conservative L1 roundtrip
bound is only a sanity check and is not independent numerical evidence. Its f64
AVX2 and AVX512 assembly contains complex FMA and permutations with no repeated
capability probes. Carrying the dispatcher's proof token removes those probes;
exact const-generic tile lengths remove unused register slots. Bounds-check
failure paths and live tile stack traffic remain, so this is not a zero-cost
abstraction claim.

Sixteen retained-binary census runs use two counterbalanced replications on
queried performance CPU 1 and efficiency CPU 3 of an Intel Core Ultra 9 285K.
All runs finish within the committed bound. The performance-core replicated
comparison establishes neither improvement nor regression across 39 cases.
Two real-transform 4096 cases remain undecidable because between-run spread
covers the within-run effect. Phase-only changes are descriptive and cannot
establish complete-engine speedup.

The efficiency-core set is invalid for performance conclusions: an independent
review finds active Cargo PID 57408 at both endpoints of its first two runs,
consuming 0.625 and 0.609375 CPU seconds. The snapshots do not establish that
process's command or lifetime. An explicit benchmark lease did not prevent
that overlap. No replacement run is needed to decide this candidate, because
its size gate already fails. Endpoint CPU totals were also recomputed with
explicit floating-point overloads after the report's integer-overload defect;
raw snapshots remain unchanged. Surviving-process utilization ranges from
0.63% to 1.93% of 24 logical processors and misses short-lived processes.

The matched census executable grows from 6,862,336 to 6,878,720 bytes:
16,384 bytes. PE sections attribute 14,528 bytes to code, 1,216 to read-only
data, 84 to unwind data and 40 to relocations; file alignment accounts for the
remaining difference. This localizes growth to emitted code and associated
metadata, but does not assign every byte to individual monomorphizations.

## Decision

Reject the register-tile candidate: the valid complete-engine comparison
establishes no benefit and the artifact-size acceptance fails. Remove its
production module and private tests; retain the original fused operation and
the allocation-free, test-only phase recorder. Production builds exclude the
recorder module and all five call sites through `cfg(test)`.

The retained profiler passes Clippy, 545 release library tests and its ignored
phase-contract test (3.921 seconds). Its rebuilt census returns to 6,862,336
bytes. The raw hash changes at exactly nine bytes: COFF/debug timestamps and
CodeView PDBAge. Every byte outside those linker metadata fields is identical
to the retained baseline, confirming production payload preservation.

Evidence remains under the Atlas ignored output root
`output/apollo-four-step-profile`: `census-manifest.json` records the runner,
comparison mapping and validity; `binary-sizes.json`, `binary-sections.txt`,
`candidate-source-hashes.json` and `candidate-source/` identify the rejected
candidate; the core-class directories preserve raw CSV samples and endpoint
load snapshots. `phase-medians.csv` is descriptive, not a significance test.
The existing census workload and comparator remain unchanged.

A later experiment must target an attributed layout cost and retain the same
acceptance. This result does not justify a provider API change, an ISA-routing
change, a wider tolerance, or a larger binary budget.

## Revision

2026-09-05: changed Proposed to Rejected after numerical/codegen validation,
replicated census comparison, independent contention review and PE size
attribution. The profiler remains as the diagnostic deliverable.
