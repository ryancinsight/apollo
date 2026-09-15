# Apollo gap audit

Open risks with their re-open triggers, and the findings the board, the ADRs
and the session notes cite, one anchored section each. Compacted by
`scripts/compact_gap_audit.py`: a section keeps its heading, its lead and its
re-open lines; the narrative behind a finding is the commit that recorded it.

## Per-thread scratch holds its high-water mark for the thread's life (2026-09-02, corrected 2026-09-03) <a id="scratch-pool-retention"></a>
Every transform crate reaches `mnemosyne::scratch::ScratchPool` through a
`thread_local!`. The pool holds `MAX_POOL_SLOTS` = 4 `AlignedVec` slots and
`with_scratch` grows a slot with `ensure_len` when the request exceeds its
length. Nothing in `pool.rs` shrinks, truncates, or resets a slot, so each slot
stays at the high-water mark of whatever that thread ever ran.

## A whole module class compiles only off-CI (2026-09-02) <a id="windows-gated-modules"></a>
`base128::pinned_probe` is declared
`#[cfg(all(test, windows, target_arch = "x86_64"))]`, and every CI job runs
Linux. The module therefore never compiles in CI, so no CI run can see any
diagnostic it emits. Eleven imports left behind by the probe split sat unused on
`main` while the `ci` workflow reported green on that very commit; the same
command CI runs — `cargo clippy --locked --workspace --all-targets
--all-features -- -D warnings` — fails immediately on a Windows checkout.

## A fix measured on one length class regressed another (2026-09-02) <a id="length-class-split"></a>
Routing the one-shot `FftPrecision` entry through the cached plan was measured
on power-of-two lengths — 128, 256, 512 — where it wins by 18 to 46%. It
merged. The composite lengths were never in the probe, and there the same
substitution *loses* by 21 to 29%: n = 100 goes 101.40 ns to 130.65, n = 384
281.79 to 354.34.

## The fused half codelet is falsified, and the instrument that said otherwise (2026-09-02) <a id="half-fusion-not-established"></a>
`ATLAS-APOLLO-F16-FUSED-SMALL-BASES-2026-09-02`, **built end to end across
three repositories and then falsified**. Two findings, and the second is the
more valuable one.

## What fusing the half conversion into the codelets would take (2026-09-02) <a id="half-fusion-blocked-upstream"></a>
`ATLAS-APOLLO-F16-FUSED-SMALL-BASES-2026-09-02`, attempted and **blocked on an
upstream API-design decision, with the ceiling measured**.

## The half bridge's residual is conversion, not call overhead (2026-09-02) <a id="half-bridge-residual"></a>
After the bulk bridge and the stack path, a half-storage transform still
costs about 2x the f32 plan route at n = 8 and 16. The probe attributes the
gap: conversion alone (both directions) is 3.15 ns at n = 8, 3.34 at 16,
4.09 at 32 and 7.62 at 64 — a fixed ~3 ns plus a small per-lane term, which
is call-shaped rather than work-shaped, and eunomia's `widen_f16`/`narrow_f16`
dispatch wrappers carry no `#[inline]`, so a consumer pays two cross-crate
calls per transform.

## A freshly built probe binary's first run is not measurable (2026-09-02) <a id="first-run-after-build"></a>
**Correction, and a measurement rule for every pinned probe in this

## Correction: the storage route's gap to the plan is 1.2-2.0x (2026-09-02) <a id="half-storage-routing-corrected"></a>
Supersedes the table in `#half-storage-small-and-routing`, whose n >= 64 rows
were taken on freshly built binaries and inflated by
`#first-run-after-build`. Re-measured with a discarded warm-up pass, both
core classes, medians in nanoseconds:

## Resolution: compact storage uses the cached plan route (2026-09-02) <a id="half-storage-plan-route"></a>
The compact `Complex<F16>` entry now resolves the existing cached `f32`
`FftPlan1D` for lengths above the register-resident range, then performs one
bulk widen/plan execution/narrow sequence. Lengths 2, 4, 8, 16, and 32 retain
their measured stack-resident codelet route. The implementation is in the API
layer, so execution remains independent of orchestration and no second plan
cache is introduced.

## Correction: the bulk bridge's gain is uniform across sizes (2026-09-02) <a id="half-storage-bulk-bridge-corrected"></a>
Supersedes the table in `#half-storage-bulk-bridge`. Those arms were also
measured one run after each build; re-measured under the warm protocol, the
element-wise to bulk comparison is both larger and flatter than recorded —
the earlier numbers understated the gain at n >= 64, where the inflated
kernel time diluted the conversion share:

## Small half transforms need no pool, and the storage route misses the plan (2026-09-02) <a id="half-storage-small-and-routing"></a>
Two findings from the same probe, after the bulk bridge removed the conversion
cost (`#half-storage-bulk-bridge`).

## Half-storage transforms paid more to convert than to transform (2026-09-02) <a id="half-storage-bulk-bridge"></a>
`Complex<F16>` has no native kernel: `precision_bridge` promotes the buffer to
`Complex32`, runs the f32 route, and demotes it back. That contract is right —
x86-64 without AVX512-FP16 has no half arithmetic, so the alternative is
emulation, and the promotion is where the precision decision belongs. What was
wrong is that both halves of it ran one lane at a time, `f16::to_f32()` per
element, while eunomia — the stack's owner of the `binary16` conversion
vocabulary — has carried F16C bulk converters (`F16::widen_slice` /
`narrow_slice`, one `vcvtph2ps`/`vcvtps2ph` per eight lanes, runtime-dispatched,
verified bit-for-bit against `half`) as public API the whole time.

## Runtime vector arms for the small transforms (2026-09-01) <a id="runtime-vector-arms"></a>
`ATLAS-APOLLO-COMPILE-TIME-FEATURE-GATES-2026-09-01`. Five files selected
their SSE/AVX bodies with compile-time `cfg(target_feature = "avx", "fma")`
and no runtime fallback; nothing sets `target-cpu`, so every default build
ran the scalar arm. The vector bodies now live in local
`#[target_feature(enable = "avx,fma")]` fns selected by a once-per-process
probe (`avx_fma_available`, the crate's `OnceLock` idiom), scalar arm as
fallback; the twelve shared SIMD helpers carry the attribute instead of the
cfg, the four combine-twiddle tables only the arch gate.

## Unchecked indexing hid what the array types already proved (2026-09-01) <a id="ratchet-slice-safe-indexing"></a>
The first SAFETY-ratchet burn-down slices deleted `unsafe` rather than
commenting it, and one of them measured as a win. `winograd/radix/
odd_prime_pair.rs` indexed its const-sized arrays (`[Complex<F>; N]`,
`[F; H]`) through `get_unchecked` with loop indices already bounded by the
same consts; plain indexing lets every bounds check fold in each
monomorphization -- and the f32 odd-prime route got faster: pinned paired
runs, f32 n = 7/11/13 -29%/-28%/-25%, 19 -12%, 31 -9%, RustFFT f32 controls
within 1%, f64 flat at every size. The slice-through-unchecked-pointer form
had been hiding the array length from LLVM, not saving a check.

## The compose arena aligned its offset, not its address (2026-09-01) <a id="compose-arena-alignment"></a>
`ATLAS-APOLLO-COMPOSE-ARENA-MIRI` asked for miri coverage of the fused
composite's thread-local bump arena. The first miri run of the new arena
tests found a real defect in the production pattern the tests replicate:
`alloc` aligned the bump *offset* inside a `Vec<u8>`, whose base pointer
promises only byte alignment, so the `&mut [Complex<F>]` handed out through
`from_raw_parts_mut` could be misaligned — "constructing invalid value ...
encountered an unaligned reference (required 8 byte alignment but found 1)".

## Every pre-`core_class` core label is corrected in place (2026-09-01) <a id="core-label-sweep"></a>
Revision note for `ATLAS-APOLLO-INVERTED-CORE-CLAIMS-2026-09-01`. Every
`P-core`/`E-core` label in a record dated before the `core_class.rs`
landing (`b21fc172`, 2026-09-01 13:09) — including this file's
`#column-pass-consolidation` and `#fused-split-loads` entries written that
morning — was measured by the hardcoded `[2, 12]` probe, whose two arms were
both inverted (`#core-class-inversion`). Those labels are swapped in place:

## Restarted real-half split twiddles (2026-09-01) <a id="real-half-split-twiddles"></a>
The allocation-free f64 real-half route spent more time untangling the retained
half-length complex transform than executing it at every stable measured size.
The exact phase probe attributed that bound to one f64 `sin_cos` evaluation per
output bin. Apollo source `5a8b90d3` now evaluates sine and cosine once per
eight-bin block and advances the remaining twiddles with native-precision
complex multiplication. Restarting after at most seven advances bounds rounding
drift while avoiding an O(N) retained twiddle table and preserving one generic
f32/f64 implementation.

## The split gathers at its native width (2026-09-01) <a id="wide-gather"></a>
The split's gather ran a hardcoded four-lane blend network; for a four-byte
scalar the four-lane request lands in the scalar-emulated frame, so the f32
route gathered without its vector unit. Two hermes provider rounds closed
it (`ATLAS-APOLLO-WIDE-STRIDED-LOADS-2026-09-01`):

## Fused split loads are the third boundary-fusion falsification (2026-09-01) <a id="fused-split-loads"></a>
Hypothesis (filed as the small-size split's next lead): the gather and the
base kernel's first loads are the same traffic paid twice, so a `LoadSource`
strategy — the load-side mirror of the `StoreSink` family — should absorb
the gather's blend network into phase one at zero marginal cost and delete
the pass (~38 ns measured at two blocks, 60-80 ns estimated with scratch
effects).

## The column pass is one function at every width (2026-09-01) <a id="column-pass-consolidation"></a>
Independent review of the eight-lane base kernel (`ATLAS-APOLLO-BASE-KERNEL-
LANE-WIDTH`) found its column pass a line-for-line sibling of the four-lane
kernel's phase 3 — same mixed-radix twiddle, same `ROWS`-point DIF network,
same `rev` store — differing only in the chunk geometry factor (eight groups
of two complex samples against four groups of four). The duplicate carried a
real cost beyond drift risk: the sink family (`CombineSink`,
`FinalCombineSink`) lived only in the four-lane copy, so an eight-lane f32
host ran the 256/512 splits through the two-pass fallback.

## Mellin real-input complex reduction (2026-09-01) <a id="mellin-real-complex-dot"></a>
Resolved by `ATLAS-APOLLO-MELLIN-REAL-COMPLEX-DOT-2026-09-01`. Threshold-sized
forward log-frequency rows previously retained a 2N-lane f64 buffer solely to
materialize each real sample as `[sample, 0]` before Hermes' complex dot. Hermes
now owns a real-by-interleaved-complex reduction, so Apollo passes the borrowed
N-lane real input directly and retains only the 2N interleaved-weight buffer.
The removed role is 16N bytes per active worker: 2,048 bytes at N = 128 and
4,096 bytes at N = 256.

## CPU STFT forward-frame fusion is control-confounded (2026-09-01) <a id="stft-window-fusion"></a>
The reusable CPU STFT forward path copies one real frame into retained scratch,
multiplies it by the Hann window into a second retained scratch, then
materializes interleaved complex input. At frame length 1,024 those two buffers
retain 16,384 bytes per active worker. A source-equivalent candidate replaced
the three passes for interior frames with one Hermes multiply-and-interleave
kernel, preserved the scalar boundary path, and recorded zero warmed global,
reallocation, and direct-Mnemosyne counts after plan warmup.
The retained benchmark is the reopening oracle. A later candidate must improve

## The f32 reinterleave sink was using half the native width (2026-09-01) <a id="reinterleave-native-width"></a>
`InterleaveRows` was the last batched boundary kernel still hardcoding four
lanes. The planar transpose and the half combine already take their width from
the scalar plan contract, so on AVX2 the f32 sink alone moved 16 bytes per
vector while its eight-lane width was available. The tile body now indexes in
whole `lanes`-wide chunks and one body serves both widths; dispatch stays
outside the loops. `TRANSPOSE_LANES` became `BOUNDARY_LANES`, which is what it
had already grown into — it governed the transpose and the combine before this
change and governs all three passes after it.

## Leto/Hermes multidimensional complex transpose (2026-09-01) <a id="leto-hermes-complex-transpose"></a>
Apollo's private 2-D/3-D axis helper previously reconstructed Leto C-from-F
views for every adjacent matrix and called the generic assignment kernel. Leto
Ops PR #135, merged as `060eb7eb`, now owns the complete allocation-free
batched-complex contract: it validates the full batch before mutation, selects
the widest exact Hermes hardware width among 16/8/4 scalar lanes for at least
256 matrices with both sides at most 16, handles ragged edges, and retains the
canonical Leto assignment path for other shapes or unsupported targets.

## N = 96 Good-Thomas columns use a compile-time CRT schedule (2026-09-01) <a id="n96-column-unroll"></a>
After the three DFT-32 rows moved into one Hermes target frame, the generated
N = 96 codelet still spent 46.07 ns f32 and 53.98 ns f64 in its 32 DFT-3
columns and irregular scatter. The generic generated loop recomputed each
column's CRT base, advanced three destinations with wrap checks, and executed
32 backedges even though the `(3,32)` factorization fixes every address at
macro-expansion time.

## The N = 96 Good-Thomas row batch uses one Hermes target frame (2026-09-01) <a id="f32-n96-codelet"></a>
The f32 N = 96 fixed codelet entered this increment at
222.863/222.935/223.045 ns against RustFFT at
94.133/94.087/94.280 ns on logical processor 2. Both scalar plans select the
generated `(3,32)` Good-Thomas codelet, so the adjacent composite cache and
flat Stockham route are absent from this execution path.

## Base splits now borrow their retained complete twiddle table (2026-08-31) <a id="base-split-twiddle-reuse"></a>
The selected f64 base route left `FftPlan1D::twiddle_fwd` empty even though
N = 256/512 combines still need the final stage of the corresponding power-of-
two table. Each N = 256 execution therefore entered the process cache and
temporarily cloned one `Arc`; N = 512 did that independently for the N = 256
and N = 512 tables. The complete N table is stage-major and already contains
the smaller split stage, so the second lookup and owner were redundant.

## Portable exact-width fallback was linked behind a hardware-only probe (2026-08-31) <a id="hardware-lane-link-footprint"></a>
Apollo's planar combine tries exact eight lanes, then exact four lanes, before
running its own scalar loop. Hermes' existing `vectorize_lanes` contract treats
its portable scalar backend as an exact-width capability, so the second probe
could instantiate a portable four-lane kernel even though Apollo immediately
owns the same fallback decision. The exact Linux PR #219 artifact retained the
unused `call_scalar_in_avx2_frame` specialization at 5,080 bytes.

## Quarter-turn twiddle reuse regresses the final base-128 sink (2026-08-31) <a id="base128-quarter-turn"></a>
Three exact-processor comparisons on the merged base-128 and fused-sink route
place f64 N = 256 at 1.2533--1.2564x RustFFT and N = 512 at
1.2403--1.2475x. Three release phase runs attribute N = 512 to
116.2--116.5 ns of gather, 725.9--726.9 ns of base transforms, and
450.7--459.2 ns of fused-combine residual. The final sink is therefore the
largest non-base phase.

## Exact-processor, retained-state FFT comparison (2026-08-31) <a id="comparison-sweep-coverage"></a>
Apollo's default RustFFT comparison stopped at 512 even though production has
distinct routes at 1,024 and 32,768. A direct extension initially produced
multiple latency bands. Two independent causes were present: Windows moved the
process between heterogeneous processor classes, and RustFFT 6.4.1's
convenience `process` method allocated and zero-filled a scratch vector inside
every timed call. Apollo already retained its execution scratch in the plan.
The old comparison therefore measured different lifecycles and its large-size
rows are not valid throughput evidence.

## Reusable QFT plans use Apollo FFT (2026-08-31) <a id="qft-fft-route"></a>
The reusable CPU QFT plan previously called the public dense kernel for every
length. That path performs N² complex products, schedules rows through Moirai
from N=128, and fills one `2N`-f64 twiddle-lane scratch on every participating
worker before calling Hermes. The plan also retained its own N-element
`Complex64` twiddle vector.

## The combine rides the column pass out (2026-08-31) <a id="combine-sink"></a>
Fresh phase attribution for the shipping 128 kernel — 397 TSC for the
fused redistribute-and-rows, 331 for the columns, 728 total, with a body
of 321 vector instructions, no calls and no compares — put it near its
throughput floor and within ~10% of the reference at its own sizes. The
remaining structural waste was at the split: the odd block's column pass
stored its spectrum to scratch, and the combine pass immediately reloaded
it, rotated it, and wrote the real output.

## Small non-smooth Rader routing (2026-08-31) <a id="small-nonsmooth-rader"></a>
The first dynamic primes whose `m = n - 1` convolution is not smooth over
Apollo's supported radix set are 59, 83, and 107. The prior selector sent all
three to Bluestein. A same-process 100-sample instrument measured forced
Bluestein versus the corrected automatic half-cyclic route (microseconds):

## The split's boundary: gather vectorized, combine fused, combine-SIMD falsified twice (2026-08-31) <a id="split-boundary"></a>
Piece attribution for the split, pinned (scalar pieces, production plan):

## Four-byte base transforms use the native AVX2 width (2026-08-30) <a id="base-native-width"></a>
The base plan previously requested exactly four scalar lanes for every
precision. On AVX2 that is native for an eight-byte scalar, but a four-byte
scalar has eight lanes. Hermes therefore reached its four-lane scalar fallback
for f32: the base route still beat the generic route, but it executed at half
the available register width and did not satisfy the module's native-capability
contract.

## The instance-major construction generalizes to n = 64 (2026-08-30) <a id="instance-major-64"></a>
n = 64 was the worst ratio on the ladder — 1.88 against RustFFT, against
1.10 at 128 — and it was still served by the sample-major kernel. The
instance-major construction turns out to fit it with almost no new
machinery: 64 is **four** stride-4 subsequences of sixteen where 128 is
eight of them, so phase 1's radix-4×4 sixteen-sample row transform is
reused verbatim with the pair loop halved (`p in 0..ROWS/2`, load stride
`ROWS/2·b + p`), and the column pass is the same DIF net truncated — it
starts at distance 2, where the only twiddle is a rotation, and stores
through the two-bit reversal instead of the three-bit one.

## The instance-major kernel zero-filled a buffer it fully overwrites (2026-08-29) <a id="base-kernel-memset"></a>
Disassembling the kernel that [shipped yesterday](#across-instance-outlining)
showed two defects it had inherited from being shelved before this
repository's own fixes landed — the shelf preserved the code and also
preserved everything the mainline learned afterward *not* being applied to
it.
**Runtime-sized types re-opened the bounds checks.** `data: &mut [T]` and

## The across-instance layout was blocked by a closure, not by registers (2026-08-29) <a id="across-instance-outlining"></a>
The across-instance row layout — registers holding two FFT *instances* rather
than two samples, so every row twiddle becomes a broadcast and the row
multiplies fall from 64 to 16 — was
[built, verified against every oracle, and shelved](#base128-across-instance).
Its rows were faster even then. What sank it was that the **untouched column
pass degraded fourfold**, which the record named "the partial-outlining
signature" and attributed to the layout outgrowing a single kernel body.

## What the base kernel spends, instruction by instruction (2026-08-29) <a id="base-kernel-ops"></a>
With [the CPU re-probing removed](#base-kernel-probes) the kernel is worth
modelling properly, because every earlier measurement of it was taken
against a body inflated 70% by spill traffic. Read out of the emitted
assembly rather than estimated.

## The base kernel was not arithmetic-bound; it was re-probing the CPU (2026-08-29) <a id="base-kernel-probes"></a>
The base kernel was [recorded as measured to the end of its
arithmetic](#base128-root2), 136 complex multiplies against the reference's
72, with the gap declared unconvertible in the AVX2 layout and blocked on
register width. That conclusion was wrong, and the way it was wrong is worth
keeping: every experiment had varied the *arrangement* and none had looked at
what the compiler actually emitted.

## Two small-size hypotheses, both falsified (2026-08-28) <a id="small-size-falsified"></a>
With [the flat split](#flat-base-split) landed, n = 256 became the worst
size on the ladder at 1.73 against RustFFT. Two candidate causes were
measured, and neither survived.

## The small-size split gathered once per level (2026-08-28) <a id="flat-base-split"></a>
n = 512 was the worst size on the ladder, 1.83 against RustFFT where 1024
sat at 1.22. The route decimates down to the 128-point base, and it did so
by recursion: 512 halved into two 256s, each of which halved again. Every
level gathered, so 512 paid three gathers and two nested scratch
acquisitions.

## The sink's permutation is cheaper on the write side (2026-08-28) <a id="sink-permutation"></a>
[Decimating stage set two in frequency](#dif-stage-set) moved the route's
bit-reversal into the sink, and the sink got dearer for it — the split
route's combine most of all. What was not separated was *which* part of the
sink's work the increase belonged to: the reversed read, or the fact that
the combine reads two planes where the square route's sink reads one.

## Apollo is faster than RustFFT at n = 16384 (2026-08-28) <a id="dif-stage-set"></a>
The [pass attribution](#planar-pass-attribution) found one pass whose entire
content was undoing the pass before it: the deinterleave earns bit-reversed
rows for free, the transpose destroys that order, and a repair pass restored
it before stage set two. [Reading how the swap list is built](#planar-pass-attribution)
showed the pass is plain bit-reversal, which ruled out folding it into the
transpose and left one answer — run stage set two decimated in frequency,
which consumes natural order and emits bit-reversed, so the sink absorbs the
permutation by reading `rev(row)`.

## The benchmark gate flagged a benchmark the change cannot reach (2026-08-28) <a id="benchmark-gate-noise"></a>
The [paired decimation pass](#split-single-pass) was held at merge by the
CI benchmark regression check, which reported two benchmarks slower in all
four counterbalanced comparisons: `bluestein_f64/257`, by 1.3% to 7.8%
across the four, and `half_cyclic_f64/67`.

## Where the planar route's time actually goes (2026-08-28) <a id="planar-pass-attribution"></a>
The [standing measurement](#reference-standing) said the even powers are
converging on parity and left it there, because nothing said *why* they are
behind at all. The driver had a per-pass instrument, but it printed a line
per pass per call, so a size worth measuring — thousands of calls — could
not be measured with it without the printing becoming the measurement. It
now accumulates per label and a probe drains the totals.

## The paired decimation pass, and why it bought less than the model said (2026-08-28) <a id="split-single-pass"></a>
[Fusing the decimation](#odd-power-fusion) left half its cost, and the
shape looked obvious: each half deinterleaved from `data` at stride two, so
each traversed every cache line of the input and the pair read the array
twice. One pass taking the adjacent pair together would read each line once
and halve the remainder.

## Alignment is the hidden variable of a placement measurement (2026-09-09) <a id="alignment-hidden-variable"></a>
The planar route read 3.9 or 4.4 to 4.7 µs at 2048 `f64` across runs of one
binary, and the pinned ladder's direct-driver arm ran 36 to 48% slower
than the plan arm on the same route. Two page-offset sweeps contradicted
each other until the buffers' addresses were tabulated: every fast case
had the scratch at 0 mod 64 and every slow case at 16 or 48, whatever the
page offset. The allocator returns sixteen-byte alignment, so a per-process
allocation is one alignment sample and the pattern reads as placement.

## A green landing proves the ISA it ran on (2026-09-09) <a id="runner-isa-varies"></a>
The hosted runner has AVX-512 on some runs and not others. PR 357 landed
green on a run without it; the next two landings drew AVX-512 hosts and
every planar-route test failed, because the plane column order is an
involution at the AVX2 widths and not at the AVX-512 ones, and one map
had served both directions. The pattern: a width-dependent property
verified only under the dispatched backend is verified on one width.
verified only by the runs that draw such a host; re-open if hermes gains

## Fusing the odd-power decimation into the planar boundary (2026-08-28) <a id="odd-power-fusion"></a>
The [standing measurement](#reference-standing) isolated the odd-power
deficit to movement rather than arithmetic: the decimation materialized the
even and odd subsequences into scratch, each half then deinterleaved *again*
into its own planes, each reinterleaved its result, and only then did the
combine run. Three passes over `n` existed solely because the halves were
transformed as if they were free-standing inputs.

## Where we actually stand against the references (2026-08-28) <a id="reference-standing"></a>
The ladder began at n = 256, so it could not see the two sizes the base
kernel serves directly, and no single record stated the standing across the
whole range. Both are fixed: the instrument now starts at n = 64, and this
is what it reports, pinned, E-core, min-of-twelve-blocks, all three engines
in the same run on the same core.

## What a wider ISA would buy, and what stands in its way (2026-08-28) <a id="wider-isa"></a>
**Revision 2026-08-30.** The [base-native-width increment](#base-native-width)
closes this partition for f32 on AVX2 by adding a four-complex eight-lane map.
The analysis below remains current for f64 AVX-512 and the other fixed-width
kernel families; no local AVX-512 timing is claimed.

## The 128-base multiply count cannot be converted into time (2026-08-28) <a id="base128-root2"></a>
[The arithmetic comparison](#base128-arithmetic-count) put Apollo at 136
complex multiplies against the reference's 72 and named three levers. All
three have now been measured, and the count gap does not convert.

## The base kernel generalizes over row length; `reverse_bits` does not (2026-08-28) <a id="base-row-length"></a>
The 128-point base is eight rows of sixteen. A 64-point transform is the
same construction at eight rows of eight — its decimation-in-time row
transform is the sixteen-sample one **without the last stage**, and its
column pass is identical — so the kernel was made generic over the row
length rather than copied, and `n = 64` now has a base where it previously
ran six ping-pong Stockham passes over 1 KB.

## Eliminating the split's gather does not pay (2026-08-28) <a id="strided-base-source"></a>
The small-size split gathers each 128-sample subsequence into scratch before
transforming it. The gather looked removable rather than tunable: a power of
two `128 * 2^d` decimates into `2^d` subsequences at stride `2^d`, and in
register terms a subsequence chunk is **one blend of two parent chunks** —
the same network phase 1 already runs — so the base kernel can read the
parent directly.

## Small sizes paid the four-step's fixed passes (2026-08-28) <a id="small-size-splitting"></a>
After the odd-power fix the worst sizes on the ladder were n = 256 and
n = 512, both about 2x RustFFT while their neighbours ran 1.05x to 1.35x.
The cause shows without a reference: **n = 256 cost 2.96x n = 128** where
the arithmetic asks for about 2.3x. The four-step route pays six passes over
the array whatever the size, and at 4 KB those passes are the transform.

## Odd powers of two were routed off the fast path entirely (2026-08-28) <a id="odd-power-routing"></a>
`FourStep::admits` required `trailing_zeros() % 2 == 0`, its comment
recording the reason: "an odd `log2` would need an asymmetric split, whose
cost this route has never been measured at." Extending the pinned ladder
through the odd powers measured it, and the exclusion was expensive well
past a tuning margin — pinned, E-core, against RustFFT:

## Checked view access, not register pressure, dominated the 128-base (2026-08-28) <a id="base128-bounds"></a>
The assembly diagnostic named on the item was run. It answers the question
it was asked and finds something larger.

## Splitting the 128-base kernel works; the layout still cannot be cashed (2026-08-28) <a id="base128-split"></a>
The restore trigger from [the across-instance validation](#base128-across-instance)
was executed: the transform was split into two `LaneKernel`s — an
across-instance row pass and the column pass — each entering its own
target-feature scope through its own `vectorize_lanes` call, with the phase
stamps moved out of the kernels into the driver so the attributed and
comparison builds share identical bodies. All nine oracles pass in every
variant below.

## The across-instance row layout pays, and outgrows one kernel body (2026-08-28) <a id="base128-across-instance"></a>
Lever 3 of [the arithmetic comparison](#base128-arithmetic-count) was built:
the 16-point row transforms rewritten to hold two FFT *instances* per
register instead of two samples of one. The derivation is clean and the
implementation is correct on the first attempt — all nine oracles pass,
including the direct-DFT and incumbent-differential checks.

## The 128-base gap is a multiply count, set by register layout (2026-08-28) <a id="base128-arithmetic-count"></a>
Three schedule changes refused to move the base-128 row pass
([above](#base128-row-ilp)), so the remaining 1.62x against RustFFT
(294 against 182 ns pinned) was taken to the arithmetic. Both kernels were
counted from source: Apollo's `components/base128/butterfly.rs` and
RustFFT 6.4.1's `Butterfly128Avx64` with the `avx_vector` helpers it calls.

## The 128-base row pass is at a structural plateau (2026-08-27, extended 2026-08-28) <a id="base128-row-ilp"></a>
The 128-point base's row pass (`components/base128`) is its largest phase, and
the standing hypothesis was chain latency: four strictly dependent DIT-16
stages whose four butterflies cannot fill the FMA and shuffle pipelines while
each waits on the one before. The prescribed cure was two-row interleaving —
process staging rows in pairs so every stage carries eight independent
butterflies instead of four, sharing each stage's twiddle loads.

## Native SIMD width is a capability partition (2026-08-27) <a id="native-width-partition"></a>
**Revision 2026-08-30.** Fixed-four-lane dispatch remains the correct contract
for kernels with only that address map. The base transform now implements a
second eight-lane map and uses a native-capability query that does not mistake
Hermes' scalar fallback for hardware SIMD; see
[Four-byte base transforms use the native AVX2 width](#base-native-width).

## Retained-footprint attribution: duplicate twiddle tables and worker retention (2026-08-27) <a id="retained-attribution"></a>
Evidence tier: allocation-call accounting — `kernel/retained_footprint.rs`
uses pointer-identity ledgers for every successful global-allocator call and
every direct Mnemosyne call, then lists every live requested block. Run it as
the only selected test in its process:
`cargo nextest run --offline -p apollo-fft --release --lib --run-ignored
ignored-only -E "test(retained_footprint_attribution)" --no-capture`.

## Global-allocator peak working set (2026-08-28) <a id="peak-working-set"></a>
Evidence tier: allocation-call accounting — the census's wrapping global
allocator extended with a live-bytes balance and high-water mark, measured in
explicit windows (`engine_census::peak_working_set_census`, run
`APOLLO_PEAK_WORKING_SET_ONLY=1 cargo bench -p apollo-fft --bench
engine_census`). Counts are exact for calls reaching that allocator, but do not
include direct Mnemosyne allocation or resident memory. This section, unlike
the timing census, needs no quiet machine.

## Every pinned table had its core-class axis inverted (2026-09-01) <a id="core-class-inversion"></a>
Evidence tier: OS query (`GetLogicalProcessorInformationEx`,
`RelationProcessorCore`), corroborated by an independent per-processor timing
sweep and by reproduction of a superseded table on the opposite core class.

## The AVX-Stockham retirement premise holds on performance cores (corrected 2026-09-01) <a id="stockham-backend-matrix"></a>
Evidence tier: same-binary pinned measurement (`stockham::backend_matrix`,
release profile), both backends instantiated under `cfg(test)`, interleaved
per size, thread pinned to a processor whose class is queried from
`GetLogicalProcessorInformationEx`; numerical agreement between backends
guarded to rounding-difference growth over `log2 n` stages. Three runs
2026-09-01; medians below, per-run ranges in
[ADR 0042](docs/adr/0042-avx-stockham-backend-retained.md).
is correcting. First step of the reopened item.

## Stage fusion pays on the kernel, and the end-to-end census cannot see it (2026-08-26) <a id="stage-fusion"></a>
Evidence tier: kernel measured in isolation across five working-set sizes, both
builds in the same session on the same host; correctness from the direct-DFT
oracle and the existing suite. End-to-end figures from the committed census with
the cache flushed between arms.

## The one-dimensional crossover contradicts ADR 0039 (2026-08-26) <a id="crossover-contradiction"></a>
Raised as a contradiction to resolve, not as a correction to apply. The decision
is recorded and Accepted; this is the evidence that disagrees with it.

## Where the remaining gap to RustFFT and PhastFT lives (2026-08-26) <a id="register-residency"></a>
Evidence tier: arithmetic rates measured in one process with the cache flushed
between arms; structural claims derived by reading the reference sources at the
locked versions (`rustfft 6.4.1`, `phastft 0.4.1`). Absolute timings are from a
host known to move Apollo's own figure by 2x, so ratios within a single run are
used and cross-run absolutes are not.

## An exactly-representable oracle can be exactly blind (2026-08-26) <a id="blind-oracle"></a>
Evidence tier: measured, with each ladder verified by reintroducing the real
recurrence and confirming the failure. Delivered as
`crates/apollo-fft/tests/twiddle_accuracy_gate.rs` in PR #121.

## The batched layout breaks the throughput ceiling (2026-08-25) <a id="batched-layout"></a>
Evidence tier: measured at kernel level, reproducible across four shapes and
several runs; correctness verified against a direct DFT and against RustFFT.
The initial end-to-end result was withheld because the instrument allowed
cross-arm cache state to move Apollo's result; the cache-flushing census above
now provides the end-to-end comparison while retaining that historical caveat.
### Parallel extension rejected (2026-08-31) <a id="batched-parallel-rejection"></a>
can reopen this item.

## The power-of-two gap is per-lane throughput, not algorithm (2026-08-25) <a id="lane-throughput"></a>
Evidence tier: measured, seven kernel variants, all engines interleaved in one
process, every variant correctness-gated against Apollo before its timing was
read.

## Cost census against RustFFT, PhastFT, and RealFFT (2026-08-25) <a id="engine-census"></a>
Evidence tier: measured, all engines interleaved inside one process, with
transient allocation counted by a wrapping global allocator rather than
estimated. Correctness gated by analytical, symmetry, and differential oracles
before any timing was read.

## Planar rewrite: hypothesis falsified before the rewrite (2026-08-25) <a id="planar-hypothesis-falsified"></a>
Evidence tier: measured, four prototype variants, all three engines interleaved
in one process, with a correctness assertion against Apollo gating every timing.

## Power-of-two f64 throughput profile (2026-08-25) <a id="pot-f64-profile"></a>
Evidence tier: measured, interleaved within a single process, with a
change-reverted control and per-size isolation runs. Host: x86-64 with AVX2 and
FMA, no AVX-512; no `target-cpu` or `RUSTFLAGS` set anywhere in Apollo or the
Atlas root config.

## FWHT migration onto the Hermes entry — negative result (2026-08-25) <a id="fwht-vectorize-negative"></a>
Evidence tier: measured, three structures, min-of-15 per size on one host.

## PhastFT reference audit (2026-08-25) <a id="phastft-2026-08-25"></a>
Reference: `https://github.com/QuState/PhastFT` at commit
`7bbbfa5bbac8681af7d1abf6fb02990d8eacb552`; crates.io `phastft` 0.4.1,
published 2026-07-31, `#![forbid(unsafe_code)]` at its crate root. It depends on
`fearless_simd` 0.5.0 for its lane kernels.

## Remaining Gaps
### Hephaestus 0.12 fallible device construction (2026-07-13)
<a id="audit-2026-06-10"></a>

## Residual findings from workspace performance/consolidation audit (2026-06-10)
Open items from the parallel duplication/allocation/dispatch audit; each is a candidate micro-sprint. Evidence tier: source inspection only unless noted.
<a id="audit-2026-07-22-bench-resolution"></a>

## Slop patterns recorded 2026-09-01

- **PM pushes cancelled merge-gate runs.** `ci.yml` ran on every push to main
  under a ref-keyed `cancel-in-progress` group with no `paths-ignore`; board
  and ADR writes via the API started full runs and cancelled the verification
  of the merge before them (#243, #244 both ended cancelled). Root cause: PM
  artifacts not excluded from the push trigger; gate cancellable. Check that
  would have caught it: a merge-gate run ending `cancelled` is a defect, not
  a status. Fixed in #246 (paths-ignore + cancel only on pull_request);
  proven by a board push starting no run while the gate completed.
- **Whole-file identity never short-circuits.** The benchmark identity gate
  compared executables with `cmp`; two build directories guarantee
  different symbol hashes and build id, so the gate always fell through and
  the pair jobs timed identical code, reporting +10% on #242. Check that
  would have caught it: the gate had never once reported
  `measurements_required=false` — a check that never passes is not a check.
  Fixed in #250 (compare code sections). The candidate-role bias did not
  reproduce under #253's forced/swapped diagnostics (runs 33574856108,
  33575247861: no supported regression either way); recorded as a one-off,
  cause unestablished.
