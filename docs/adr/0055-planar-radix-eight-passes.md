# ADR 0055: Planar stage-set driver and the radix-8 trial

- **Status:** Accepted
- **Date:** 2026-09-08
- **Class:** [patch] [perf]
- **Item:** [APOLLO-PLANAR-RADIX-DEPTH](../../backlog.md#apollo-planar-radix-depth); parent [ATLAS-APOLLO-BEAT-THE-REFERENCES](../../backlog.md#atlas-apollo-beat-the-references)

## Context

ADR 0054 established that the batched planar stage sets are bound by
instruction issue, not by plane traffic, and its codegen count put the fused
radix-4 pass at 88 to 114 instructions per four-lane quad: 40 to 48 vector
arithmetic, 28 to 40 vector loads and stores, 19 to 34 scalar address
instructions (each of the eight row pointers reloaded from the stack before
its vector load) and 4 to 5 spill reloads of the hoisted twiddles. Apollo
trails RustFFT by 1.2 times in `f64` from 16384 through 65536 after the
domain and seam changes.

Each stage set applied two stages per pass, so a 256-point row set took
four passes; every pass streams the whole `len * batch` array once and
pays its row addressing and loop control once per row set. Three stages
per pass takes the same 256-point set in three passes and the 512-point
set in three instead of five, cutting loads, stores and addressing by a
third at the cost of holding eight complex rows, sixteen vectors, across
the butterfly.

## Decision

Both stage sets keep two stages per pass, and both run through one shared
driver. The radix-8 grouping was built and measured first and is rejected
on this host; the driver it was built on is retained because it measures
faster than the hand-written pair bodies it replaces.

The trial grouped stages three at a time while three remained, then two,
then one; the time-decimated set ascending from `l = 2` and the
frequency-decimated set descending from `l = len`. The per-element
operation order is the single-stage order regardless of grouping, so
results are bitwise those of the previous pairing, which the existing
suites confirmed on the candidate.

The butterfly arithmetic moves to one place: a `Lane` trait implemented by
the SIMD vector (fused multiply-add) and by the scalar element (the
two-operation product the remainder always used), six `Radix` types naming
the stage groupings and their twiddle order, and one `butterfly_rows`
driver that loads a row set from the planes or the interleaved source,
applies the four-step fold, runs the radix, and stores to the planes or the
interleaved sink, for the vector loop and the scalar remainder alike. The
two kernels are now their pass loops and twiddle indexing only; the six
hand-written vector and scalar butterfly bodies they carried are deleted.

Sixteen live vectors are the whole AVX2 file, so in the trial the twiddles
were splatted once per twiddle index and reloaded as the allocator decided.
That was the measured trade the stop criterion below decided.

## Verification

The batched suite (direct-transform agreement at 4 through 4096 in `f64`
and 16 through 256 in `f32`, round trips, the plan cache), the four-step
workspace suite (exact impulse spectrum through the selected route at 4
through 262144, both precisions and directions, both workspace extents),
the RustFFT differential at 65536 and 262144, the dimension-1d plan suite
and the crate's API tests. Lengths 4, 8 and 16 exercise the radix-2-only,
radix-8-only and radix-8-then-radix-2 groupings; 32 and 64 the
radix-8-then-radix-4 and double radix-8 groupings; 256 and 512 the
three-pass forms.

## Measurement and stop criterion

`pinned_sections` for the per-pass attribution and `engine_census`,
`twiddless_comparison` and `phastft_comparison` for the verdict on the
pinned performance core. Accept when the whole-transform intervals at
16384 and 65536 sit below ADR 0054's in the census and the warm instrument
does not regress; reject and restore the pair grouping otherwise, keeping
the shared driver if it is neutral.

## Result and decision

Reject the radix-8 grouping on AVX2; accept the shared driver with radix-4
pairs. Evidence is `output/apollo-planar-radix8/` with its manifest; the
candidate source is retained there as `candidate-*.rs`.

Radix-8 (all 110 affected tests passing) against the seam-fused pairs,
apollo `f64` medians in microseconds on the pinned performance core:

| N | instrument | pairs (three runs) | radix-8 |
|---|---|---|---|
| 4096 | census | 8.2 / 8.3 / 9.0 | 10.7 |
| 16384 | census | 40.8 / 41.5 / 40.8 | 42.7 |
| 65536 | census | 210 / 204 / 205 | 228 |
| 4096 | warm | 9.2 / 9.1 / 9.0 | 11.2 |
| 16384 | warm | 42.1 / 42.7 / 44.1 | 45.8 |
| 65536 | warm | 212 / 201 / 201 | 256 |

The per-pass attribution put the loss in the frequency-decimated set
(stage set 2 at 65536 `f64`: 409k to 537k cycles) with the time-decimated
set flat: eight complex rows plus seven twiddles spill on a sixteen-register
file, and the spills cost more than the third pass they save. The
reference arms held within 2% in every row above, so the regression is
supported at every length in both instruments.

Shared driver with radix-4 pairs (121 tests passing), three runs each,
against the seam-fused pairs:

| N | instrument | before (three runs) | after (three runs) | rustfft after |
|---|---|---|---|---|
| 4096 | warm f64 | 9.2 / 9.1 / 9.0 | 8.3 / 8.5 / 8.5 | 7.8 to 8.4 |
| 16384 | warm f64 | 42.1 / 42.7 / 44.1 | 36.3 / 38.3 / 38.8 | 33.8 to 34.4 |
| 16384 | warm f32 | 20.6 / 21.1 / 20.5 | 18.9 / 18.5 / 19.7 | 18.0 to 18.2 |
| 65536 | warm f64 | 212 / 201 / 201 | 213 / 211 / 212 | 170 |
| 65536 | warm f32 | 100.7 / 100.1 / 101.5 | 100.2 / 98.0 / 98.0 | 108 to 109 |
| 16384 | census f64 | 40.8 / 41.5 / 40.8 | 45.3 / 41.4 / 40.8 | 33.5 to 35.2 |
| 65536 | census f64 | 210 / 204 / 205 | 227 / 238 / 206 | 171 to 196 |

The warm instrument, whose reference arms held within 1%, separates at
4096 and 16384 in `f64` (8 to 14% faster) and at 16384 in `f32` (7%), and
is neutral at 65536 within 5%. The census's first driver run and its
65536 rows carry a reference-arm shift of 17% and are not counted; its
clean rows are neutral. No supported regression at any length. The gain is
the driver's tighter row loop, not the grouping; the six hand-written
vector and scalar butterfly bodies are deleted for a net removal of code.

What remains against RustFFT in `f64`: 1.1 times at 16384 and 1.24 times
at 65536. The codegen count in the parent item still names the levers the
driver has not yet pulled, the stack-reloaded row pointers and the fold
branch inside the row loop, and those are the next increment; radix-8 is
re-measured only on a 32-register file (AVX-512), recorded here as the
trigger.

## Revision

2026-09-08: changed Proposed to Accepted with the radix-8 grouping rejected
and the shared driver retained, after the replicated measurements above.

2026-09-08, second increment: the driver's seams became compile-time. Each
pass now runs a monomorphization carrying only the seams it uses (planes,
interleaved source, fold, interleaved sink, fold with sink), and rows are
addressed as first plus step rather than as an offset table. The plain
radix-4 pass fell from 88 to 114 instructions per four-lane quad to 53
(8 FMA, 16 loads, 8 stores, no vector spills, no general-register reloads,
5 scalar); the folded pass to 82 with 4 reloads and the sourced pass to 71.
Whole-transform timings moved within noise in both instruments (census
`f64` 16384: 45.3 / 41.4 / 40.8 to 42.4 / 37.6; 65536: 227 / 238 / 206 to
203 / 202; warm neutral within 3%). With the overhead gone the pass sits at
the vector-arithmetic floor, 32 vector operations per quad per pass at
about 3.6 per cycle, so the remaining lever is the arithmetic itself: the
first time-decimated pass, the last frequency-decimated pass and the `j = 0`
row set of every pass multiply by `1` and `∓i` with full complex multiplies.
Listings under `output/apollo-planar-radix8/asm_spec_*.s`.
