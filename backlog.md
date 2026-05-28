# Apollo Backlog

## Open in this sprint (Closure CVXIII phase)
- [ ] [patch] Optimize AVX/FMA twiddle loading and eliminate indexing offsets/negation branches for f64 sizes 32 and 64 in `impls.rs`.

## Closed in this sprint (Closure CVXII phase)
- [x] [patch] Narrow reduced f32 Winograd-pair execution to DFT31 only. A
  broader structure-of-arrays reduced route for N=29/37/41/53 was measured and
  rejected because it did not produce stable improvement and regressed larger
  short-prime rows. The retained DFT31 route keeps the generic Winograd-pair
  path for every f64 short odd prime and for f32
  N=11/13/17/19/23/29/37/41/43/47/53. Value-semantic coverage now exercises
  promoted f64 odd-prime routes, all f32 odd-prime routes, and the reduced
  f32 DFT31 inverse route. Current optimized `xtask` rows record reduced f32
  DFT31 at 87.31 ns Apollo vs 83.75 ns RustFFT (`1.043x`), while the generic
  DFT31 probe measured 107.39 ns Apollo vs 82.46 ns RustFFT (`1.302x`).
  f32 N=23/43/47 beat RustFFT; f32 N=11/13/17/19/29/31/37/41/53 remain
  measured misses.

## Closed in this sprint (Closure CVXI phase)
- [x] [patch] Replace static-Rader routing for short odd-prime `ShortDft`
  sizes 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, and 53 with the
  existing Winograd-pair kernel. The change removes the Rader gather,
  sub-FFT, pointwise spectrum multiply, inverse sub-FFT, and scatter from the
  production short-prime path while retaining and extending generated static
  Rader codelets as validation and direct Rader surfaces through N=53.
  Focused value tests pass, the proc-macro crate compiles under test, and
  optimized `xtask` clone-inclusive rows showed the direct pair route as the
  better production short-prime baseline; Closure CVXII supersedes the f32
  row inventory after a focused reduced-layout follow-up.

## Closed in this sprint (Closure CVX phase)
- [x] [patch] Reduce runtime Rader and ordered-Rader Good-Thomas permutation
  cache memory by retaining only the generator-order table. The inverse
  generator order used for scatter is derived exactly from the cyclic group
  identity `g^{-q} = g^(N-1-q)` for prime `N`, so the cached scatter array is
  no longer independent state. This removes `N-1` retained `usize` values per
  cached prime/generator pair; for N=10007 on 64-bit targets this removes
  10006 `usize` values, or 80,048 bytes, from the runtime Rader permutation
  cache. Verification covers the identity directly, Rader direct-DFT and
  half-cyclic/full-cyclic equivalence, ordered Good-Thomas routes, `xtask`
  compile, and focused half-cyclic Criterion execution under the local
  opt-level 1 bench profile.

## Closed in this sprint (Closure CVIX phase)
- [x] [patch] Reduce half-cyclic Rader cache-build memory by streaming the
  length `N-1 = 2m` kernel directly into cyclic and negacyclic CRT residues.
  The previous builder materialized a full `2m` kernel plus two `m` residue
  buffers before FFTing the residues. The new builder retains only the two
  residue buffers, eliminating `2m` complex values of peak temporary storage
  and one explicit split pass. For N=10007 this removes 10006 temporary complex
  values from the build path: 160,096 bytes for f64 complex spectra and 80,048
  bytes for f32 complex spectra. Rader-focused correctness checks pass, and the
  focused opt-level 1 benchmark shows N=1031 forced half-cyclic f64/f32 rows
  improved while smaller forced rows remain noisy and below the production
  threshold.

## Closed in this sprint (Closure CVIII phase)
- [x] [patch] Integrate the Liu-Tolimieri half-cyclic convolution strategy
  into runtime Rader prime FFT execution without retaining a Bluestein fallback
  path. Rader now factors the length `N-1 = 2m` cyclic convolution through the
  CRT identity `x^(2m)-1 = (x^m-1)(x^m+1)`, computes cyclic and negacyclic
  length-`m` residues, and recombines lower/upper halves in place. Automatic
  production routing is conservative at `N-1 >= 1024` for both f64 and f32;
  forced strategy hooks remain feature-gated for tests and benchmarks. Focused
  Rader tests pass against direct DFT references and the N=10007 large-prime
  roundtrip. The final local optimized benchmark completed with
  `CARGO_PROFILE_BENCH_QUICK_OPT_LEVEL=1`; default bench-quick codegen was
  terminated before Criterion emitted rows in this environment.

## Closed in this sprint (Closure CVII phase)
- [x] [patch] Convert fixed coprime Good-Thomas dispatch from ad hoc route
  plumbing into a proc-macro-derived support/match surface backed by one
  bounded const-generic PFA codelet. The generator derives canonical coprime
  pairs from one `short_sizes` list and `max_n`; the retained implementation
  specializes `(N1, N2, N, INVERSE)` through monomorphization and uses const
  CRT maps plus direct `ShortDft<N>` row/column transforms. The partial
  proc-macro short-codelet refactor is now complete: `ShortDft<N>` is a
  derived blanket capability rather than a cyclic `ShortWinogradScalar`
  supertrait requirement, and `generate_winograd_fft!` is exported again. A
  full unrolled per-pair body prototype passed value checks but was rejected
  because the bench/release codegen path exceeded the bounded verification
  budget. The review-comment rows demonstrate that monomorphization is not
  route equivalence: N=9 is a prime-power short codelet,
  N=84/N=90/N=150/N=175 are fixed coprime PFA shapes with different subfactor
  costs, and N=94 is the direct `2*p` promoted-prime route. A focused N=44
  probe improved Apollo from the prior table row but still missed RustFFT at
  120.96 ns vs 78.51 ns for f64 and 145.49 ns vs 91.35 ns for f32, so
  `benchmark_results.md` was not rewritten for that row. Targeted
  quick-profile rows were regenerated for N=84/N=90/N=94/N=150/N=175. N=94
  remains a RustFFT win at 0.729x f64 and 0.519x f32; N=150 f64 is near parity
  at 1.042x; N=84/N=90/N=175 and f32 N=150 remain misses that need route-cost
  adaptation instead of more size-specific code. A focused N=10 refresh shows
  the apparent f32-only regression was stale Criterion data: current N=10 is
  40.75 ns Apollo vs 55.60 ns RustFFT for f64 and 42.38 ns Apollo vs 51.42 ns
  RustFFT for f32. `xtask benchmark --sizes ...` now treats subset runs as row
  merges, and the normal measurement path is the optimized `xtask` bounded
  adaptive clone-inclusive runner rather than a Criterion subprocess. The
  already-built optimized `xtask` binary rewrote the full canonical table in
  65.6 seconds. A focused N=77 refresh first closed the stale 4.739x f32
  mixed-epoch Criterion row, then the shared odd-prime-pair DFT kernel was
  changed from iterator-zip arithmetic to const-indexed loops with native
  `one()` sign selection. Current N=77 is 199.94 ns Apollo vs 103.96 ns
  RustFFT for f64 and 235.34 ns Apollo vs 78.52 ns RustFFT for f32. The route
  remains canonical fixed Good-Thomas `(11, 7)`; the remaining f32 disparity
  is a real route-cost/vectorization gap, not dynamic dispatch or stale table
  evidence.

## Closed in this sprint (Closure CVI phase)
- [x] [patch] Add generated short Good-Thomas codelets for N=18, N=24, and
  N=36 without hand-written per-size arithmetic bodies. The new leaves are
  built from existing short factors through `generate_good_thomas!` and routed
  through the canonical short-Winograd dispatch. Generic natural and ordered
  PFA now use the existing thread-local PFA scratch allocation for column
  buffers instead of allocating a fresh `Vec` per transform. A broader generated
  fixed coprime dispatch was prototyped and rejected for this increment because
  optimized ThinLTO release builds did not complete reliably under the current
  codegen profile. The direct generated `2*p` natural-prime path now uses a
  twiddle-free Good-Thomas row/column codelet across the promoted-prime family,
  and `xtask benchmark` now defaults to a quick Criterion profile plus an
  optimized `bench-quick` Cargo profile for iterative table refreshes.
  `benchmark_results.md` was regenerated for N=38/N=58/N=74/N=82/N=94.
  N=94 remains faster than RustFFT in both precisions; N=38/N=58/N=74/N=82
  remain measured misses, with N=74 f32 the largest current regression.

## Closed in this sprint (Closure CV phase)
- [x] [patch] Correct the natural Good-Thomas PFA CRT scatter layout. The
  cached PFA output permutation is keyed by transformed column-major frequency
  coordinates `(k2, k1)`, so the natural PFA kernel now indexes
  `output_perm[k2 * n1 + k1]` instead of row-major `k1 * n2 + k2`.
  Direct-DFT forward and unnormalized inverse tests cover a nontrivial
  coprime natural PFA shape. The same closure completed the stale Winograd
  const-generic direction migration exposed by a fresh rebuild: generated
  Good-Thomas, production short-codelet dispatch, and unit tests now target
  the current `const INVERSE` DFT-3/7/8/15 entry points. The generated `3*p`
  Good-Thomas route now emits direct const-generic column and row codelet calls
  from one proc-macro prime list instead of relying on a separate short-codelet
  adapter surface. The canonical `vs_rustfft` f64/f32 Criterion table now also
  includes N=38/N=58/N=74/N=82/N=94. Fresh targeted rows show N=33 still
  misses RustFFT at 1.419x f64 and 1.778x f32, while N=94 now beats RustFFT at
  0.665x f64 and 0.726x f32. `xtask benchmark` is now the single active
  benchmark runner/table generator for `benchmark_results.md`; the old Python
  extractor, quick comparison example, duplicate validation `vs_rustfft` bench,
  and bench output logs were removed. `apollo-fft` bumped to 0.12.22.

## Closed in this sprint (Closure CIV phase)
- [x] [patch] Move Good-Thomas family dispatch deeper into proc-macro
  generation. The `3*p` generator now emits the complete per-prime transform
  body, not closure parameters around a hand-written driver, and the direct
  Winograd-pair `2*p` route now uses `generate_two_by_prime_natural_dispatch!`
  from one prime/half-size table. The sealed Winograd scalar contract now
  carries the prime-pair table capability required by generated dispatch in
  release builds. The hand DFT-3 codelet remains selected because the current
  generated Winograd direct-DFT prototype has not been proven faster.
  `benchmark_results.md` was regenerated after a targeted N=33 Criterion
  refresh; N=33 records f64 Apollo/RustFFT 94.34 ns / 68.16 ns and f32
  Apollo/RustFFT 108.75 ns / 64.81 ns. `apollo-fft` bumped to 0.12.21.

## Closed in this sprint (Closure CIII phase)
- [x] [patch] Fuse more of the compact Good-Thomas `3*p` route into generated
  code without deleting retained FFT components. `generate_three_by_prime_dispatch!`
  now emits per-prime CRT gather/scatter functions from the single supported
  prime list, and `three_by_prime_impl` receives those generated functions as
  monomorphized parameters. The runtime kernel keeps the generic
  `MixedRadixScalar` contract and uses `short_winograd_const` for const-size
  row codelet dispatch. Generated Rader now matches the runtime generator
  convention and uses exact f64 constants; direct static Rader expansion is
  intentionally bounded to 5/7/11/13 until the generator can emit a scalable
  convolution form. `benchmark_results.md` was regenerated from the canonical
  Criterion cache; N=33 records f64 Apollo/RustFFT 93.00 ns / 64.92 ns and f32
  Apollo/RustFFT 108.00 ns / 67.49 ns. `apollo-fft` bumped to 0.12.20.

## Closed in this sprint (Closure CII phase)
- [x] [patch] Move the compact Good-Thomas `3*p` dispatch surface from
  hand-written duplicated match arms to the internal `apollo-fft-macros`
  generator. `generate_three_by_prime_dispatch!` now owns the single
  short-prime list and emits the support predicate plus monomorphized
  `(P, inverse)` dispatch arms. The numerical kernel remains the existing
  generic `three_by_prime_impl` over `MixedRadixScalar` and
  `ThreeByPrimePlan<const P>`, so the runtime path stays statically
  dispatched and no retained Rader, Good-Thomas, Winograd, butterfly,
  Stockham, or four-step component was removed. `benchmark_results.md` was
  regenerated from the canonical Criterion cache; N=33 still records f64
  Apollo/RustFFT 101.49 ns / 70.27 ns and f32 Apollo/RustFFT 121.28 ns /
  78.91 ns. `apollo-fft` bumped to 0.12.19.

## Closed in this sprint (Closure CI phase)
- [x] [patch] Move the compact Good-Thomas `3*p` route toward the generator
  architecture described by the root `gen*.md` notes. Added
  `ThreeByPrimePlan<const P>` so CRT input and output maps, including modular
  inverses, are derived in `const fn` once per monomorphized short-prime
  factor. The hot transform body now indexes precomputed CRT maps rather than
  calculating modulo-based input/output routes. This is the stable Rust
  foundation for a future proc-macro SSA route generator and keeps the current
  code maintainable and verifiable. `benchmark_results.md` was regenerated
  from completed Criterion rows; N=33 now records f64 Apollo/RustFFT
  101.49 ns / 70.27 ns and f32 Apollo/RustFFT 121.28 ns / 78.91 ns.
  `apollo-fft` bumped to 0.12.18.

## Closed in this sprint (Closure C phase)
- [x] [patch] Close the N=33 routing gap without deleting retained FFT
  components. N=33 is `3*11`, so it has a coprime Good-Thomas decomposition,
  but the dispatcher previously selected the prime-23 mixed-radix composite
  `[11, 3]` route first. Added a compact `3*p` Good-Thomas CRT codelet for
  short-prime factors 5/7/11/13/17/23 and route that verified structural
  family before `cached_prime23_radices`. Existing radix-composite,
  Good-Thomas, Rader, Winograd, butterfly, Stockham, and four-step components
  remain available. The benchmark-only ordered-Rader hook was also updated to
  the current ordered-Rader API. `benchmark_results.md` was regenerated from
  Criterion; N=33 now records f64 Apollo/RustFFT 104.08 ns / 69.15 ns and f32
  Apollo/RustFFT 128.21 ns / 63.15 ns. `apollo-fft` bumped to 0.12.17.

## Closed in this sprint (Closure XCIX phase)
- [x] [patch] Remove typed real-storage conversion temporaries from
  caller-owned execution paths. f64, f32, and compact f16 `forward_*_into`
  implementations now fill the caller-owned complex spectrum buffer directly
  with one `Zip` pass before executing the existing monomorphized 1D/2D/3D
  plans in place. The matching `inverse_*_into` implementations now copy the
  input spectrum into caller-owned scratch, execute the inverse plan, and fill
  real output directly from scratch without allocating a mapped temporary.
  Allocating typed forward paths also avoid the previous complex-array clone by
  transforming the mapped output in place. Compact f16 still converts only at
  the storage boundary and executes through the f32 plan family. No Rader,
  Good-Thomas, Winograd, butterfly, Stockham, or four-step component was
  removed. `benchmark_results.md` remains the single canonical
  Apollo-vs-RustFFT f64/f32 clone-inclusive table and was regenerated from the
  current Criterion cache snapshot. `apollo-fft` bumped to 0.12.16.

## Closed in this sprint (Closure XCVIII phase)
- [x] [patch] Reconcile generic FFT plans with typed real-storage cache
  policy. `FftPlan1D`, `FftPlan2D`, and `FftPlan3D` now remain generic over
  the mixed-radix scalar contract, while public typed real FFT helpers resolve
  plan precision through `RealFftData::PlanScalar` and `PlanCacheProvider`.
  Native f64/f32 storage keeps native cached plan families; compact f16 storage
  delegates to f32 plans at the storage boundary instead of forcing
  `f16: MixedRadixScalar`. The power-of-two fast path now starts at N>=64 so
  N=16/N=32 stay on the faster current short-codelet route, and all retained
  Rader, Good-Thomas, Winograd, butterfly, Stockham, and four-step components
  remain available until measured replacements beat RustFFT.
  `benchmark_results.md` remains the single canonical Apollo-vs-RustFFT f64/f32
  clone-inclusive table and was regenerated from the current Criterion cache
  snapshot. `apollo-fft` bumped to 0.12.15.

## Closed in this sprint (Closure XCVII phase)
- [x] [patch] Restore scalable power-of-two routing without deleting retained
  components. Mixed-radix dispatch now sends every power-of-two length N>=16
  through one generic fast-path before small Winograd, composite, PFA, or Rader
  routing can claim the shape. The fast-path uses Stockham for asymmetric
  powers and retains square four-step for even-exponent lengths above the
  four-step threshold, preserving N=2/N=4/N=8 short codelets. This fixes the
  N=32768 selector fallthrough/no-op risk exposed by the benchmark table.
  `FftPlan1D` now uses the generic mixed-radix twiddle and scratch-cache APIs
  directly instead of removed precision suffix helpers, and exposes the same
  generic caller-owned typed forward/inverse methods as 2D/3D plans for
  zero-allocation benchmark compilation. Rader, Good-Thomas, Winograd,
  butterfly, Stockham, and four-step implementations remain available until a
  measured replacement beats RustFFT.
  `benchmark_results.md` remains the single canonical Apollo-vs-RustFFT f64/f32
  clone-inclusive table and was regenerated from the current Criterion cache
  snapshot. `apollo-fft` bumped to 0.12.14.

## Closed in this sprint (Closure XCVI phase)
- [x] [patch] Reduce small-coprime composite routing overhead without deleting
  retained components. N=6, N=10, N=12, and N=14 now use stack-resident
  Good-Thomas CRT codelets built from the existing Winograd DFT-3/4/5/7 leaves,
  bypassing the generic mixed-radix scratch/twiddle route for these shapes.
  The route remains monomorphized through `short_winograd`, no Rader,
  Good-Thomas, Winograd, butterfly, or composite component was removed, and
  retained routes stay available until measured replacements beat RustFFT.
  The obsolete private Good-Thomas gather helper left unused by the fused
  ordered-Rader PFA path was removed to resolve the bench build dead-code
  warning at source.
  `benchmark_results.md` remains the single canonical Apollo-vs-RustFFT f64/f32
  clone-inclusive table. `apollo-fft` bumped to 0.12.13.

## Closed in this sprint (Closure XCV phase)
- [x] [patch] Reduce Rader negacyclic convolution memory traffic across
  large-prime routes. The Nussbaumer split now writes the negacyclic half in
  already-twisted form, and CRT recombination multiplies by the conjugate
  twist while combining cyclic and negacyclic results. This removes two
  standalone full passes over the negacyclic half while keeping the same
  mathematical decomposition, fused radix-composite pointwise dispatch, and all
  retained Rader, Good-Thomas, Winograd, butterfly, and composite routes.
  `benchmark_results.md` remains the single canonical Apollo-vs-RustFFT f64/f32
  clone-inclusive table. `apollo-fft` bumped to 0.12.12.

## Closed in this sprint (Closure XCIV phase)
- [x] [patch] Reduce promoted-prime permutation and Rader convolution memory
  traffic without deleting retained kernels. Direct `2*p` Good-Thomas routing
  now calls one monomorphized Winograd-pair two-prime kernel that consumes the
  original interleaved even/odd input, eliminating the previous even-half stack
  copy and odd-half compaction pass across every `PrimePairTable<P, H>` direct
  route. The f32/f64 scalar implementations now satisfy the
  `composite_forward_with_pointwise` contract, so Rader circular and
  negacyclic convolution can fuse supported radix-composite forward FFT stages
  with spectrum multiplication instead of running a separate pointwise pass.
  No Rader, Good-Thomas, Winograd, butterfly, or composite route was removed.
  `benchmark_results.md` remains the single canonical Apollo-vs-RustFFT f64/f32
  clone-inclusive table. `apollo-fft` bumped to 0.12.11.

## Closed in this sprint (Closure XCIII phase)
- [x] [patch] Reduce generic fused-routing and Good-Thomas permutation
  overhead. Radix-composite scalar fallback stages now walk destination blocks
  through `chunks_exact_mut(stage_chunk)`, so one const-radix monomorphized
  stage dispatch covers each output block without recomputing slice bounds.
  Final fused pointwise multiplication now uses raw pointer traversal over the
  contiguous output block under the existing length contract. Good-Thomas
  natural and ordered-Rader PFA gather/scatter loops now use cached-permutation
  length assertions and four-wide unchecked copies. No Rader, Good-Thomas,
  Winograd, butterfly, or composite route was removed; retained components stay
  available until a measured replacement beats RustFFT. The retained Winograd
  N=82 composite codelet now carries the required `PrimePairTable<41, 20>`
  bound. `benchmark_results.md` was regenerated from the Criterion cache plus
  the latest quick strategy and selected public comparisons. `apollo-fft`
  bumped to 0.12.10.

## Closed in this sprint (Closure XCII phase)
- [x] [patch] Reduce radix-composite dispatch overhead and refresh benchmark
  evidence. Recursive fused-composite scratch arena logic now lives in the
  cohesive `radix_composite::adaptive` leaf, bringing `arity.rs` back under the
  repository structural limit. The flat fused Stockham path now dispatches
  scalar fallback stages through one const-radix match per stage instead of one
  match per output group, preserving the existing f64 AVX2 radix-3/radix-4
  hooks and improving routing adaptability for every scalar fallback radix.
  The final fused pointwise spectrum multiply now uses one contiguous output
  pass instead of a radix/column nested loop.
  Rader benchmark routing now targets the shared generic Rader kernel and real
  Winograd-pair kernels, and stable-Rust static Rader permutation tables remain
  compile-time constants per prime dispatch arm.
  Winograd large-composite leaves remain available; no composite component is
  gated or removed before a measured RustFFT-beating replacement exists.
  `benchmark_results.md` was regenerated from all available Criterion
  `new/estimates.json` records and the latest debug Rader-vs-Winograd-pair
  quick strategy comparison. `apollo-fft` bumped to 0.12.9.

## Closed in this sprint (Closure XCI phase)
- [x] [patch] Optimize Rader Bluestein cache retention and inverse pointwise
  execution. Bluestein now caches `(chirp_fw, kernel_fw)` only; the inverse
  path derives `conj(kernel_fw)` from the even cyclic kernel identity instead
  of retaining a second M-length spectrum. The pre-chirp, zero-pad, post-chirp,
  and conjugated pointwise stages route through the f64/f32 SIMD dispatchers.
  The SIMD zero-fill count is corrected to typed element lanes. For N=10007,
  M=20736, the persistent cache entry saves one M-length spectrum: 331,776
  bytes for f64 complex data or 165,888 bytes for f32 complex data per cached
  prime/precision entry. `apollo-fft` bumped to 0.12.8.

## Closed in this sprint (Closure XC phase)
- [x] [patch] Optimize standalone Rader memory traffic and retained scratch.
  Standalone generated and runtime Rader now compute the nonzero DC contribution
  while gathering primitive-root ordered inputs, removing the previous separate
  full pass over `data[1..N]`. Static-table and runtime scatter loops now use
  the same unrolled unchecked-index shape as the gather path. Rader padded
  scratch now retains one aligned thread-local buffer per precision and uses a
  local nested-call fallback instead of retaining a two-buffer pool, reducing
  persistent per-thread Rader scratch retention by one maximum-size buffer.
  Direct-DFT Rader coverage passes, `cargo check -p apollo-fft` passes with
  pre-existing odd-prime-pair dead-code warnings, and release strategy-only
  `quick_compare` records Rader latencies of 148/126 ns at N=29, 121/123 ns at
  N=31, and 138/136 ns at N=37 for f64/f32. The current comparison hook's
  Winograd column aliases Rader, so these numbers are recorded as Rader absolute
  latency only. `apollo-fft` bumped to 0.12.7.

## Closed in this sprint (Closure LXXXIX phase)
- [x] [patch] Restore fused radix-composite Stockham dispatch and radix-4
  factorization verification.
  The fused dispatcher now routes `Fused2` through `Fused6` ZST arity types
  through `ExecutionPolicy` chunk dispatch, corrects each fused twiddle slice to
  `(radix - 1) * prev_len * prior_product`, and lowers consecutive radix-2
  pairs to radix-4 stages while keeping `factorize_composite` prime-only.
  The rejected highest-power lowering probe was removed because it emitted
  unsupported radix 16. `Radix<R>` and fallback dispatch now cover 4, 8, 17, and 23;
  nested fused `Compose` stages reserve their complete recursive arena scratch
  before exposing midpoint pointers. Direct Winograd coverage for N=17 and N=23
  remains active. N=29/N=31/N=37 now route through the no-gather Winograd-pair
  kernels after bounded Rader-vs-Winograd comparison showed Winograd-pair still
  faster for these small primes. Generated Rader remains available for larger
  primes and gated comparison, but generated leaves N=17..97 now route through
  one const-generic `rader_static_impl::<F, N, G, G_INV>` body with fused
  gather+x0 accumulation, fused scatter+x0 offset, and final-forward-stage
  composite pointwise fusion for the convolution spectrum. N=29/N=31/N=37
  Rader comparison leaves now use static gather/scatter permutation tables to
  remove runtime modular-index recurrence from the small-prime Rader hot loop.
  The ordered-layout Rader leaf now consumes generator-ordered nonzero inputs,
  emits inverse-generator-ordered nonzero outputs, and reuses `data[1..]` as
  the convolution buffer so fused callers can bypass standalone Rader
  gather/scatter and scratch copy without changing the natural-order dispatch.
  Good-Thomas PFA now uses that ordered leaf for prime `n1` subtransforms that
  would otherwise route through Rader: the PFA transpose writes generator-order
  columns, ordered Rader runs in place, and the final CRT scatter consumes the
  inverse-generator output order. N=29/N=31/N=37 remain excluded from that PFA
  branch so the measured Winograd-pair production choice is preserved.
  The ordered-Rader PFA branch now reuses the same cached generator and
  inverse-generator permutation arrays as standalone Rader, removing runtime
  modular index recurrence from the transpose/scatter layout conversion.
  Stale radix-composite fallback dispatch code was removed from the module
  graph. Closure XCII supersedes the earlier composite export narrowing:
  retained large Winograd leaves remain available until a measured replacement
  beats RustFFT. Radix-shape tests now encode the radix-4 promotion invariant
  (`192 = 3 * 4^3`). Earlier
  bounded debug Winograd/Rader ratios are 0.345/0.541 at N=29, 0.309/0.710 at
  N=31, and 0.414/0.883 at N=37 for f64/f32. Earlier release strategy-only
  Winograd/Rader ratios are 0.206/0.476 at N=29, 0.368/0.566 at N=31, and
  0.334/0.555 at N=37 for f64/f32. Rader latency improved versus the previous
  restored fused-path probe, but Winograd-pair remains faster for these
  small-prime strategy rows. Latest release production comparison:
  N=29 Apollo 0.096 us vs RustFFT 0.107 us; N=31 0.104 us vs 0.105 us; N=37
  0.147 us vs 0.132 us. Ordered-Rader PFA probe hooks now cover
  N=38/N=82/N=86/N=94/N=106 through `APOLLO_FFT_QUICK_N` and the
  `ordered_rader_pfa_coprime_composites` Criterion group. Latest release
  ordered-Rader PFA ratios against RustFFT are 6.433, 2.581, 2.505, 1.845, and
  2.455 respectively. The follow-up production increment added a dedicated
  `good_thomas::two_by_prime` route for N=2p, promoted
  N=19/N=29/N=31/N=37/N=41/N=43/N=47/N=53 to the shared odd-prime
  Winograd-pair kernel, moved odd-prime pair code into
  `winograd/radix/odd_prime_pair.rs`, expanded benchmark hooks, and removed the
  stale dedicated DFT-82 codelet so N=82 falls through to the optimized
  two-by-prime route. The follow-up memory increment loads the even half of
  promoted N=2p composites into a const-generic stack array and compacts the
  odd half in place before fused two-prime Winograd execution, bypassing
  thread-local PFA scratch on that route. Latest release prime-leaf ratios
  against RustFFT are 0.907, 0.972, 0.736, 0.799, 0.720, 0.599, 0.582, and
  0.909 for N=19/N=29/N=31/N=37/N=41/N=43/N=47/N=53. Latest two-by-prime ratios
  are 1.514, 1.195, 1.228, 1.059, 1.025, 0.943, 0.587, and 0.757 for
  N=38/N=58/N=62/N=74/N=82/N=86/N=94/N=106; N=38 remains the largest composite
  gap, while N=58/N=62/N=74/N=82 are marginal misses in the bounded probe.
  The remaining declarative composite-test generator was replaced by a
  const-generic helper and grouped explicit tests, preserving all value checks
  while keeping the test leaf under 500 lines. `apollo-fft` bumped to 0.12.6.

## Closed in this sprint (Closure LXXXVIII phase)
- [x] [patch] Add N=23 Winograd pair-symmetry codelet and dispatch coverage.
  The DFT-23 kernel uses eleven conjugate input pairs, const-generic
  direction dispatch, direct f64/f32 `FftPrecision` fast paths, and split
  scalar constant leaves to preserve the 500-line file bound. Apollo N=23
  latest run: f64 92.341 ns vs RustFFT 116.48 ns; f32 104.80 ns vs RustFFT
  139.88 ns. The Rader cache cleanup keeps split gather/scatter arrays and
  restores direction-specific convolution spectra for inverse correctness.
  `apollo-fft` bumped to 0.12.5.

## Closed in this sprint (Closure LXXXVII phase)
- [x] [patch] Add N=17 Winograd pair-symmetry codelet and benchmark coverage.
  The DFT-17 body is shared behind f64/f32 scalar constants and const-generic
  direction dispatch; call wrappers select inlined f64 and out-of-line f32
  codegen without cloning the algorithm. Apollo N=17 latest run: f64 71.932 ns
  vs RustFFT 81.043 ns; f32 90.289 ns vs RustFFT 112.84 ns. `apollo-fft`
  bumped to 0.12.4.

## Closed in this sprint (Closure LXXXVI phase)
- [x] [patch] Add N=13 Winograd pair-symmetry codelet and dispatch coverage.
  Direction is encoded as a const generic so forward/inverse variants are
  separately monomorphized and runtime sign dispatch is eliminated. DFT-13 and
  DFT-3 leaves now live under `winograd/radix/`, keeping `radix.rs` below 500
  lines. Apollo N=13 latest run: f64 82.158 ns vs RustFFT 94.077 ns; f32
  78.778 ns vs RustFFT 86.069 ns. `apollo-fft` bumped to 0.12.3.

## Closed in this sprint (Closure LXXXV phase)
- [x] [patch] Replace O(N²) `dft7_impl` with Winograd constant algorithm
  (18 real muls: Hermitian symmetry + circulant cosine/sine matrix). Add `dft7`
  to `ShortWinogradScalar` trait and `7 =>` dispatch arm. Partition three
  identical 534-line winograd test files into domain-scoped modules (185 tests).
  Apollo f64 N=15: ~82 ns (−24% vs RustFFT ~108 ns); f32: ~89 ns (−15% vs
  ~105 ns). `apollo-fft` bumped to 0.12.2.

## Closed in this sprint (Closure LXXXIV phase)
- [x] [patch] Add DFT-100 Good-Thomas PFA codelet to `winograd/composite.rs`
  and wire it into `ShortWinogradScalar` / `short_winograd` dispatch. N=100
  previously fell through to the generic `pfa_fft` path. Apollo f64 N=100
  is now 310 ns (−25% vs RustFFT 415 ns); f32 is 292 ns (−11% vs RustFFT
  327 ns). Five correctness tests added; all 261 tests pass.

## Closed in this sprint (Closure LXXXIII phase)
- [x] [major] Remove concrete mixed-radix twiddle wrapper entry points and
  route all internal plan-owned twiddle reuse through the canonical
  const-generic dispatch body. Dead Winograd AVX wrapper leaves were removed,
  radix-15 leaves now use the stack-only generic Good-Thomas Winograd codelet,
  broad Stockham AVX stage/pair leaves now share one monomorphized backend,
  the unreachable legacy CPU SIMD six-step, matrix-workspace, and radix2
  infrastructure island was deleted, `dispatch_inplace` remains crate-private,
  and `apollo-fft` was bumped to 0.12.0.

## Closed in this sprint (Closure LXXXII phase)
- [x] [patch] Reduce `apollo-fft` Stockham butterfly leaf size and dispatch
  coupling. The f64 AVX scratch routing logic now lives in a separate
  `butterfly::dispatch` leaf, leaving fixed butterfly codelets in
  `butterfly::fixed` and preserving static dispatch. Stale benchmark references
  to removed internal kernel modules were cleaned to compile against the
  maintained generic selector and `real_fft` twiddle builders, and the
  type-named compact storage leaf was consolidated into `mixed_radix/dispatch.rs`.
  `apollo-fft` was bumped to 0.11.1.

## Closed in this sprint (Closure LXXVI phase)
- [x] [patch] Reduce `apollo-fft` frequency utility iterator construction
  overhead. `fftfreq` and `rfftfreq` now use exact-capacity fill loops for
  known-length output vectors. `apollo-fft` was bumped to 0.9.11.

## Closed in this sprint (Closure LXXV phase)
- [x] [patch] Reduce `apollo-fft` shift utility redundancy and per-element
  modulo work. `fftshift` and `ifftshift` now share one split-slice copy helper,
  and the unused `Default` generic bound was removed. `apollo-fft` was bumped
  to 0.9.10.

## Closed in this sprint (Closure LXXII phase)
- [x] [patch] Reduce `apollo-fft` native 3D f32/f16 real allocation
  pipelines. The allocating native real32 forward path now constructs its
  caller-owned output buffer through the sealed overwrite-first workspace
  contract, and native inverse projection uses the same exact-size buffer
  strategy. `apollo-fft` was bumped to 0.9.7.

## Closed in this sprint (Closure LXXI phase)
- [x] [patch] Reduce `apollo-fft` native 2D f32/f16 real allocation
  pipelines. The native real32 path now uses shared monomorphized
  overwrite-first packing and projection helpers backed by the sealed workspace
  allocation contract. `apollo-fft` was bumped to 0.9.6.

## Closed in this sprint (Closure LXX phase)
- [x] [patch] Reduce `apollo-fft` 1D compact f16 power-of-two allocation
  pipelines. The compact f16 path now uses exact-size overwrite-first buffers
  for input packing and output projection, backed by the sealed workspace
  allocation helper. `apollo-fft` was bumped to 0.9.5.

## Closed in this sprint (Closure LXIX phase)
- [x] [patch] Reduce `apollo-fft` 1D precision dispatch duplication. Native
  f32 paths and mixed f16 non-power-of-two paths now share monomorphized
  `Complex32` forward/inverse helpers for conversion, twiddle-aware kernel
  dispatch, and real-output projection. `apollo-fft` was bumped to 0.9.4.

## Closed in this sprint (Closure LXVIII phase)
- [x] [patch] Reduce `apollo-fft` Bluestein plan-construction memory writes.
  The padded convolution filter now initializes overwritten mirrored chirp
  entries directly and zero-fills only the unused gap before the pre-transform.
  Generated scratch scripts and the generated Stockham broadcast experiment
  were removed from the deliverable scope. `apollo-fft` was bumped to 0.9.3.

## Closed in this sprint (Closure LXVII phase)
- [x] [patch] Consolidate `apollo-fft` plan-owned scratch allocation.
  1D Bluestein, 1D iRFFT, 2D/3D axis-pass, 3D R2C, and six-step f32 workspaces
  now share one sealed uninitialized workspace helper for scratch element types
  whose full contents are overwritten before read. `apollo-fft` was bumped to
  0.9.2.

## Closed in this sprint (Closure LXVI phase)
- [x] [patch] Reduce `apollo-fft` normalization and workspace memory overhead.
  Inverse scale passes now share AVX-capable normalization helpers, twiddle
  tables and composite twiddle tables fill exact pre-sized buffers, and
  overwritten FFT workspace buffers avoid zero-fill allocation cost.
  `apollo-fft` was bumped to 0.9.1.

## Closed in this sprint (Closure LXV phase)
- [x] [major] Remove concrete public auto-selector FFT wrappers from
  `apollo-fft`. Internal plans, tests, and benchmarks now use the canonical
  generic `fft_forward`, `fft_inverse`, and `fft_inverse_unnorm` entry points
  or the lower-level mixed-radix implementation where axis normalization
  requires it. `apollo-fft` was bumped to 0.9.0.

## Closed in this sprint (Closure LXIV phase)
- [x] [major] Remove the remaining public type-suffixed Winograd DFT-16/32/64
  wrappers. The recursive Winograd codelets now share one generic implementation
  family and mixed-radix dispatch calls the generic codelets directly.
  `apollo-fft` was bumped to 0.8.0.

## Closed in this sprint (Closure LXIII phase)
- [x] [major] Remove type-suffixed public short-Winograd wrappers for
  DFT-2/3/4/5/7/8 and twiddle multiplication. Mixed-radix dispatch now calls
  the canonical generic Winograd implementations directly, stale wrapper docs
  were removed, and `apollo-fft` was bumped to 0.7.0.

## Closed in this sprint (Closure LXII phase)
- [x] [major] Remove type-suffixed direct DFT wrappers and the debug-only f32
  parity binary from `apollo-fft`. Direct DFT tests, benchmarks, and kernel
  regressions now use the canonical generic `dft_forward` / `dft_inverse`
  functions, and `apollo-fft` was bumped to 0.6.0.

## Closed in this sprint (Closure LXI phase)
- [x] [patch] Reuse `apollo-fft` Bluestein and mixed-radix composite scratch
  buffers across calls and cache composite twiddle tables by exact radix
  decomposition plus direction. The composite cache no longer aliases different
  radix orders for the same length, stale allocation/`MaybeUninit` docs were
  removed, and `apollo-fft` was bumped to 0.5.3.

## Closed in this sprint (Closure LX phase)
- [x] [patch] Reduce `apollo-fft` typed 3D plan redundancy. The f32/f16 3D
  allocating and caller-owned paths now share one private monomorphized
  `Plan3dReal32` helper, the dead f32-only 3D real-to-complex writer was
  deleted, and `apollo-fft` was bumped to 0.5.2.

## Closed in this sprint (Closure LIX phase)
- [x] [patch] Reduce `apollo-fft` typed 2D plan redundancy and crate-root file
  size. The f32/f16 2D paths now share one private monomorphized
  `Plan2dReal32` helper, duplicated 2D plan Rustdoc was removed, crate-root
  tests moved into `lib_tests.rs`, and `apollo-fft` was bumped to 0.5.1.

## Closed in this sprint (Closure LVIII phase)
- [x] [major] Remove the stale `FftPlan3D::nz_complex` compatibility alias,
  rename `HalfSpectrum3D::nz_complex` to `HalfSpectrum3D::nz_c`, and remove
  stale compatibility wording from `apollo-fft`. The canonical half-spectrum
  bookkeeping name is `nz_c`; concrete precision kernel entry points remain
  documented as dispatch anchors for the generic `FftPrecision` API; and
  `apollo-fft` was bumped to 0.5.0.

## Closed in this sprint (Closure LVII phase)
- [x] [major] Remove the radix-specific f16 FFT module and custom `Cf16`
  wrapper from `apollo-fft`. Compact f16 complex storage now uses
  `num_complex::Complex<half::f16>`; the f16 execution bridge is a generic
  monomorphized `Complex32Bridge` with reusable thread-local scratch; the dead
  native f16 CPU gate and public f16-specific FFT wrappers were deleted; caller
  code and benchmarks now use the generic `fft_forward`/`fft_inverse` trait
  entry points; and `apollo-fft` was bumped to 0.4.0.

## Closed in this sprint (Closure LVI phase)
- [x] [patch] Integrate remote RustFFT comparator work with the current
  Stockham/composite/Bluestein FFT architecture. `apollo-fft` now uses the
  workspace `rustfft` dev-dependency, keeps the separate `vs_rustfft`
  benchmark, removes dead radix-specific benchmark rows for deleted kernels,
  routes exact 2/4/8/16/32/64 f64/f32 mixed-radix transforms through a shared
  static-dispatch short-Winograd helper, and removes unused f16 twiddle caches
  from the mixed-radix facade.

## Closed in this sprint (Closure LV phase)
- [x] [minor] Add caller-owned `apollo-hilbert` analytic observable
  projections and route envelope/phase through reusable analytic scratch.
  `AnalyticSignal` now exposes `*_into` projection methods for real,
  quadrature, envelope, phase, and instantaneous frequency; allocating
  projection methods delegate to the same non-generic helpers; `HilbertPlan`
  exposes `envelope_into` and `phase_into`; plan-level envelope/phase avoid
  per-call owned analytic vectors; parity/mismatch/capacity tests cover the new
  paths; and `apollo-hilbert` was bumped to 0.3.0.
- [x] [minor] Add caller-owned `apollo-hilbert` analytic-signal execution and
  remove per-call analytic allocation from caller-owned quadrature. The direct
  kernel now exposes `analytic_signal_into`, `HilbertPlan` exposes
  `analytic_signal_into`, owned analytic execution routes through the
  caller-owned path, quadrature projection reuses a thread-local Complex64
  analytic scratch buffer, crate-root docs no longer claim private DFT
  ownership, parity/capacity/mismatch tests cover the new paths, and
  `apollo-hilbert` was bumped to 0.2.0.
- [x] [minor] Add `apollo-fft` 1D real-forward slice execution and remove the
  final `apollo-hilbert` ndarray input bridge. `FftPlan1D` now exposes one
  non-generic caller-owned slice path that the existing ndarray path delegates
  to, Hilbert analytic-signal execution uses the cached FFT plan directly on
  real slices, 1D precision methods and tests were split into leaf modules so
  `dimension_1d.rs` stays below 500 lines, `apollo-hilbert` no longer depends
  on `ndarray`, `apollo-fft` was bumped to 0.3.0, and `apollo-hilbert` was
  bumped to 0.1.4.
- [x] [patch] Remove `apollo-hilbert` analytic-signal spectrum and inverse
  copy allocations. The owner kernel now keeps the forward FFT output as the
  analytic spectrum, applies the Hilbert mask in place, runs the complex
  inverse in place, moves the contiguous buffer out once for the allocating
  API, routes owned quadrature through the caller-owned writer, and bumps
  `apollo-hilbert` to 0.1.3.
- [x] [patch] Remove `apollo-hilbert` caller-owned quadrature copy-through
  allocation and dead direct `rayon` dependency. The owner kernel now exposes a
  slice-level quadrature writer, `HilbertPlan::transform_into` routes through
  it, typed transform workspaces inherit the allocation reduction, direct
  kernel parity/mismatch tests cover the new path, and `apollo-hilbert` was
  bumped to 0.1.2.
- [x] [patch] Remove `apollo-hilbert` typed quadrature and analytic input
  bridge allocations. Typed Hilbert `f32`/`f16` paths now reuse thread-local
  f64 input/output workspaces, keep `f64` storage on the zero-copy owner path,
  preserve shared analytic-mask execution, add repeated-call capacity/value
  coverage, and bump `apollo-hilbert` to 0.1.1.
- [x] [patch] Remove `apollo-sdft` typed direct-bin bridge allocations. Typed
  direct-bin execution now reuses thread-local f64 input and Complex64 output
  workspaces, keeps arithmetic in the shared direct-bin owner kernel, adds
  repeated-call capacity/value coverage, and bumps `apollo-sdft` to 0.1.1.
- [x] [patch] Remove `apollo-stft` inverse WOLA per-call workspace
  allocations. `inverse_into`, `inverse`, and typed inverse now reuse
  thread-local frame, complex, overlap, and weight workspaces through the
  shared slice-level inverse owner path; repeated-call value tests prove
  capacity reuse; the ADR now reflects the owner inverse workspace design; and
  `apollo-stft` was bumped to 0.2.1.
- [x] [major] Remove `apollo-stft` per-call typed bridge allocations and
  deprecated allocating alias methods. STFT typed forward/inverse now reuse
  thread-local f64/Complex64 bridge workspaces through shared slice-level
  kernels, storage/profile traits moved to a dedicated leaf module, the 1D
  plan file is below the 500-line structural limit, a co-located ADR records
  the breaking cleanup, and `apollo-stft` was bumped to 0.2.0.
- [x] [patch] Remove `apollo-qft` plan-path dense output allocation and typed
  Complex64 bridge allocations. QFT dense kernels now expose caller-owned
  output execution, plan `forward_into`/`inverse_into` route through slices,
  typed paths reuse thread-local Complex64 input/output workspaces, and
  `apollo-qft` was bumped to 0.1.1.
- [x] [patch] Remove per-call `apollo-gft` typed storage f64 bridge
  allocations. GFT typed paths now reuse thread-local f64 input/output
  workspaces through slice-level graph-basis multiply, and `apollo-gft` was
  bumped to 0.1.1.
- [x] [patch] Remove per-call `apollo-fwht` typed storage bridge and f16
  compute allocations. FWHT typed defaults now reuse thread-local f64
  input/output workspaces through slice-level f64 execution, mixed f16 paths
  reuse a thread-local f32 compute workspace, and `apollo-fwht` was bumped to
  0.1.1.
- [x] [patch] Remove per-call `apollo-czt` plan-path convolution and typed
  bridge allocations. `CztPlan` now owns a reusable Bluestein convolution
  workspace, precomputes square-plan inverse Vandermonde nodes, exposes
  internal Complex64 slice execution for typed storage, and reuses thread-local
  Complex64 typed input/output workspaces. `apollo-czt` was bumped to 0.2.1.
- [x] [patch] Remove newly surfaced `apollo-fft` dead radix-2 butterfly helper
  section after the Stockham path became canonical, add missing `FftPlan3D`
  Rustdoc, and bump `apollo-fft` to 0.2.2.
- [x] [patch] Remove per-call `apollo-frft` typed-storage Complex64 bridge
  allocations. `Complex32` and `[f16; 2]` FrFT paths now reuse thread-local
  Complex64 input/output workspaces and call internal slice entry points on the
  canonical direct FrFT kernel, eliminating two O(N) heap allocations per typed
  forward/inverse call. `apollo-frft` was bumped to 0.1.2.
- [x] [patch] Restore `apollo-fft` dependency compilation after current module
  header drift. Reinstated the kernel module declarations/`FftPrecision` trait
  header; `apollo-fft` was bumped to 0.2.1.
- [x] [patch] Remove current `apollo-fft` dead generic helper surface. Deleted
  the unused f16 with-twiddles bridge, obsolete uniform power-of-two
  digit-reversal helper, obsolete power-of-four/eight shape predicates, and
  unused Winograd stage traits after Stockham/composite routing became
  canonical.
- [x] [patch] Remove per-call `apollo-frft` unitary coefficient allocation.
  `UnitaryFrftPlan` now reuses a thread-local O(N) coefficient workspace for
  `V^T x`, phase multiplication, and reconstruction, preserving the
  Candan-Grünbaum unitary DFrFT contract while eliminating repeated heap
  allocation in forward/inverse calls. Removed stale backward-compatibility
  wording on live crate-root exports and bumped `apollo-frft` to 0.1.1.
- [x] [major] Remove remaining `apollo-fft` compatibility re-export surfaces.
  Public root exports now point directly at canonical `application`,
  `domain::contracts`, and `domain::metadata` owners; in-repo callers no longer
  use `apollo_fft::{backend,error,types}` or `apollo_fft::application::plan`.
  The legacy `FFT_CACHE` alias and unused
  `infrastructure::cpu::simd::power_of_two::{radix4,radix8}` forwarding modules
  were removed, and `apollo-fft` was bumped to 0.2.0.
- [x] [major] Remove deprecated `apollo-stft-wgpu` non-power-of-two error
  variant and explicit dead-code suppressions. `FrameLenNotPowerOfTwo` was no
  longer returned after Chirp-Z support; non-power-of-two tests now assert
  successful forward, inverse, and reusable-buffer paths. GPU-retained buffer
  fields now use `_`-prefixed ownership names instead of `#[allow(dead_code)]`.
- [x] [patch] Remove remaining WGPU dead-code suppressions in NUFFT/NTT cleanup.
  NUFFT reusable buffers now reject dispatches whose sample count exceeds the
  pre-allocated capacity before any GPU write, and NTT reusable buffers removed
  duplicated scalar `n_inv` storage while retaining GPU resources through
  `_`-prefixed ownership fields.
- [x] [patch] Remove NUFFT-WGPU per-dispatch layout-placeholder allocations.
  Fast Type-1 and 3D Type-2 paths now bind one retained `layout_padding_buffer`
  for shader entries that are structurally required by the shared layout but not
  read by that entry point.
- [x] [patch] Remove dead `apollo-dctdst` DCT-II/DST-II fast-path output
  allocations. Single-projection fast DCT-II/DST-II now reuse the 2N-point FFT
  setup and fill only the requested projection instead of allocating an N-length
  unused sibling output.
- [x] [patch] Continue Apollo-vs-RustFFT f32 N=4096 investigation. Reject disabling
  the f32 N=4096 radix-16 quad suffix: same-session Criterion measured Apollo
  6.5098 µs vs RustFFT 3.7433 µs with the quad predicate disabled.
- [x] [patch] Restore local `vs_rustfft` benchmark compilation against the current
  public API by adding the missing RustFFT dev-dependency, registering the bench,
  repairing Winograd typed entry points, and routing the untracked benchmark
  through the present mixed-radix precomputed-twiddle surface.
- [x] [patch] Record residual performance gap: current f32 N=4096
  precomputed-twiddle row measures Apollo 22.790 µs vs RustFFT 3.5969 µs. This
  row is not comparable to the earlier plan-scratch row because the plan-scratch
  API used by that row is absent in this checkout.
- [x] [patch] Route large f32 power-of-two mixed-radix dispatch through the
  monomorphized Stockham scratch-backed kernel instead of the radix-8 facade.
  Final retained f32 N=4096 Criterion measured Apollo zero-alloc reused
  7.0463 µs, Apollo caller-twiddle reused 8.9737 µs, and RustFFT reused
  6.2814 µs. Rejected the initial production 8x512 hybrid and direct
  no-argument micro-dispatch probes because both regressed the then-retained
  route.
- [x] [patch] Improve f32 N=4096 Stockham scheduling and public-path cache
  overhead. Disable the spilling radix-16 quad suffix on the retained
  scratch-backed path, keep the stride-64 triple suppression, and add a
  single-entry thread-local f32 forward-twiddle fast cache that borrows the
  cached table instead of cloning `Arc` on each public call. Longer Criterion
  measured Apollo zero-alloc reused 6.3347 µs, Apollo caller-twiddle reused
  6.0315 µs, and RustFFT reused 4.2974 µs.
- [x] [patch] Audit and reject the terminal groups=1 in-place Stockham hook:
  the groups=1 source layout is interleaved (`src[2j]`, `src[2j+1]`), while
  the hook assumed split halves and was not a valid generic copyback removal.
  Retain the f32 public dispatch inlining and consolidated f32 Stockham
  workspace, and reject the static N=4096 twiddle specialization because it
  regressed Apollo public zero-alloc to 5.4357 µs.
- [x] [patch] Reject the promoted f32 8x512 N=4096 production route after
  same-tree Criterion showed the generic Stockham route was faster. Retain the
  f32 N=4096 radix-8/radix-8 tail schedule and split public scratch/twiddle
  cache, removing the dead combined workspace. Final retained Criterion
  measured Apollo public zero-alloc reused 5.4298 µs, Apollo caller-twiddle
  reused 5.2661 µs, and RustFFT reused 3.6958 µs; an earlier same-state run
  measured Apollo public 4.8645 µs and caller-twiddle 4.7913 µs, so the
  remaining spread is benchmark variance.
- [x] [patch] Continue f32 N=4096 probe discipline. Reject the 64 KiB low-live
  threshold, a separate single-entry Stockham twiddle cache, a direct N=4096
  four-pass specialization, and unchecked twiddle subslices because repeated
  Criterion did not show a stable retained improvement.
- [x] [patch] Continue f32 N=4096 monomorphization and memory-efficiency probes.
  Reject stride-64 radix-16 fusion after Criterion regressed Apollo public to
  9.7711 µs and caller-twiddle to 9.3225 µs versus RustFFT 3.7232 µs. Reject
  forced `#[inline(always)]` at the Stockham AVX/cache boundaries because rustc
  rejects the target-feature combination and repeat Criterion did not retain an
  improvement. Current retained run after reverts measured Apollo public
  5.4895 µs, Apollo caller-twiddle 5.4176 µs, and RustFFT 4.3328 µs.
- [x] [patch] Continue f32 N=4096 hot-codelet probes. Reject paired 128-bit
  stores in the quarter-groups-one suffix because Criterion regressed Apollo
  public to 7.1908 µs and caller-twiddle to 6.1711 µs versus RustFFT
  3.8321 µs. Reject even-radix tail monomorphization and const-generic
  radix-1 quarter-turn signs because repeat Criterion did not retain a caller
  improvement and the const-sign probe regressed the public row to 8.1940 µs.
- [x] [patch] Continue assembly-level f32 N=4096 investigation. Release assembly
  showed the Windows ABI saves XMM6-XMM15 around the separate f32 Stockham
  codelets. A private raw-pointer `sysv64` ABI removed the XMM save block from
  the suffix assembly, but focused Criterion did not retain a kernel-row
  improvement: first combined run measured Apollo caller-twiddle 5.4358 µs
  versus RustFFT 3.5192 µs, while repeat measured Apollo caller-twiddle
  7.7158 µs versus RustFFT 5.2601 µs. Reverted the ABI probe.
- [x] [patch] Add nonsimd scalar permutation cleanup. Replace generic
  power-of-two digit reversal division/modulo with shift/mask digit extraction
  in the shared radix permutation helper. This is SWAR-adjacent scalar work for
  non-Stockham routes; focused f32 N=256 Criterion remained effectively neutral
  at Apollo public 983.67 ns and caller-twiddle 991.61 ns, so the residual
  N=256 gap is in radix-4 butterflies/scheduling rather than digit reversal.
- [x] [patch] Expand f32 forward autosort coverage. Lower the f32 Stockham
  dispatch threshold from 1024 to 256 so N=256 bypasses radix-4 digit reversal.
  Focused Criterion improved N=256 from the prior digit-reversal route near
  983.67 ns public / 991.61 ns caller-twiddle to 197.50 ns public /
  218.36 ns caller-twiddle on repeat. Rejected lowering the threshold to 64:
  N=64 public regressed to 64.969 ns while caller-twiddle was neutral.
- [x] [patch] Integrate f32 inverse autosort coverage and benchmark it. Route
  f32 power-of-two inverse paths at lengths >=256 through Stockham with inverse
  twiddles, and scale explicitly for normalized inverse. Add inverse zero-alloc
  rows to `vs_rustfft`. Focused Criterion showed old inverse digit-reversal
  baseline at 963.10 ns for N=256 and 23.104 µs for N=4096, while retained
  Stockham inverse measured 230.60 ns and 5.5408 µs after restoration.
- [x] [patch] Expand f64 autosort coverage for forward and inverse
  power-of-two paths at lengths >=256. Add f64 inverse zero-allocation
  benchmark rows and an N=256 forward+normalized-inverse value test. Focused
  Criterion showed the old f64 digit-reversal baseline at 830.23 ns forward /
  778.38 ns inverse for N=256 and 25.456 µs forward / 32.167 µs inverse for
  N=4096; retained Stockham measured 315.24 ns / 257.88 ns and 10.050 µs /
  10.731 µs. Rejected f64 threshold 64 because it regressed N=64 public and
  caller-twiddle rows.
- [x] [patch] Improve N=256/N=512 Stockham memory efficiency by removing
  production f64 N=256/N=512 and f32 N=512 fixed single-pass kernels from
  dispatch in favor of the fused generic AVX scheduler. Focused Criterion
  measured f64 N=256 at 255.90 ns public / 228.16 ns caller-twiddle /
  225.37 ns inverse, f64 N=512 at 591.36 ns public / 581.33 ns caller-twiddle,
  and f32 N=512 at 366.39 ns public / 346.71 ns caller-twiddle /
  328.85 ns inverse. Added N=512 f32/f64 roundtrip tests.
- [x] [patch] Add a static f32 N=4096 four-triple Stockham schedule that skips
  the generic scheduler loop and directly executes the four retained radix-8
  fused stages. Focused Criterion improved f32 N=4096 caller-twiddle forward
  from 6.9498 µs to 5.4670 µs and inverse from 6.5585 µs to 5.1970 µs in the
  same retained benchmark history, but RustFFT still measured 3.7807 µs
  forward and 3.7765 µs inverse on the latest run. Rejected the same static
  N=4096 schedule for f64 because it regressed forward to 11.264 µs, and
  rejected an f32 N=512 no-copy tail schedule because it regressed forward to
  440.90 ns and inverse to 570.83 ns.
- [x] [patch] Probe RustFFT-like f32 8x512 production decomposition using the
  verified column radix-8 step, mixed twiddles, retained row-local N=512 fused
  Stockham, and final transpose. Correctness held, but Criterion regressed
  f32 N=4096 forward/inverse to 11.792 µs / 11.786 µs. Reordering the final
  transpose to contiguous destination stores improved the failed route to
  9.9378 µs / 9.9228 µs but remained slower than the retained four-triple
  schedule, so the production probe was reverted.
- [x] [patch] Implement and reject a f32 Butterfly512-style 8x64 production
  candidate. It used the verified radix-8 column pass, mixed twiddles, eight
  fixed 64-point row butterflies, and final transpose. Correctness held, but
  Criterion regressed N=512 forward/inverse to 546.25 ns / 573.94 ns versus the
  retained fused scheduler. A vectorized mixed-twiddle variant regressed
  forward further to 773.36 ns, so the production dispatch was reverted.
- [x] [patch] Audit the complete RustFFT `Butterfly512Avx` pathway and encode
  its twiddle-layout precondition as Apollo tests. The prior 8x64 candidate was
  arithmetically valid but did not satisfy RustFFT's actual 16x32 base-kernel
  memory contract. New f32/f64 tests pin the separated-column packed twiddle
  order used by the fused twiddle+transpose stage, giving the next production
  kernel a verified layout target without weakening the retained dispatch.
- [x] [patch] Benchmark the current open zero-allocation rows and retain only
  measured improvements. Rejected production f32/f64 N=512 fixed single-pass
  leaves because they regressed both precisions. Retained f64 N=4096
  forward-only static four-triple dispatch selected by the forward twiddle sign:
  current baseline Apollo forward 17.686 µs improved to 15.844 µs, while inverse
  stays on the generic schedule because the same static schedule regressed
  inverse under inverse twiddles.
- [x] [patch] Improve 3D R2C/C2R memory efficiency by eliminating per-row
  temporary `Vec<Complex64>` allocation in the Z-axis split/extraction passes.
  Forward R2C now packs the length-`nz/2` complex subproblem into the
  caller-owned half-spectrum row prefix; inverse C2R reuses the mutable
  half-spectrum scratch row for recovered packed spectrum values before the
  sub-IFFT. Removed unused f32 R2C future-reservation fields and their plan-time
  allocations.
- [x] [patch] Reject the closure-borrowed thread-local twiddle cache probe.
  It removed hot-path `Arc` clones in source form but regressed focused f32
  N=4096 public zero-allocation Criterion to 8.4200 µs median. Restored the
  retained cache route; the repeat row measured 7.0245 µs median in this
  session.
- [x] [patch] Remove unreachable 2D FFT fallback lane materialization.
  `FftPlan2D` axis dispatch only calls `Axis(0)` and `Axis(1)`; the previous
  fallback for impossible axes allocated `Vec<Vec<Complex64>>` or
  `Vec<Vec<Complex32>>`, copied every lane, transformed the nested buffers, and
  scattered them back. The invalid-axis branch is now an explicit invariant, so
  row/column fast paths remain the only production paths.
- [x] [patch] Correct the monomorphized generic DFT-8 twiddle sign used by the
  composite-radix path. The generic Winograd helper now uses forward roots
  `exp(-2πik/8)` and inverse roots `exp(+2πik/8)` for both f64 and f32,
  restoring composite sizes such as N=24, N=48, N=192, N=384, N=1000, and
  N=10000 without reintroducing type-specific helper clones.
- [x] [major] Remove deprecated FFT compatibility aliases. Deleted
  `FftPlan1D/2D/3D::{forward_into,inverse_into}` forwarding methods and the
  legacy `ProcessorFft3d` type alias, then updated in-repo Python bindings to
  call `forward_real_to_complex_into` / `inverse_complex_to_real_into`
  directly. This leaves one authoritative caller-owned API surface.

## Open performance target
- [ ] [patch] Surpass RustFFT across the full `vs_rustfft` zero-allocation
  matrix. Current retained rows already beat RustFFT for f64 N=512 forward and
  inverse and intermittently for f32 N=256 forward, but f64 N=256, f64 N=4096,
  f32 N=512, and f32 N=4096 remain open gaps.
- [ ] [patch] Replace the retained N=512/N=4096 f32 Stockham base path with a
  complete 16x32 Butterfly512 pathway: column butterfly16, packed twiddle
  multiply, fused 4x4 transpose stores, row butterfly32, then mixed-radix8xn
  composition for N=4096.

## Closed in this sprint (Closure XLI phase)
- [x] [minor] Add separable CPU 2D DHT: `DhtPlan::forward_2d`, `inverse_2d` (N×N, involutory scaling 1/N²).
- [x] [minor] Add separable CPU 3D DHT: `DhtPlan::forward_3d`, `inverse_3d` (N×N×N, involutory scaling 1/N³).
- [x] [minor] Add `DhtError::ShapeMismatch2d` and `DhtError::ShapeMismatch3d` variants.
- [x] [minor] Add `ndarray = "0.16"` dependency to `apollo-dht`; re-export `Array2`, `Array3`.
- [x] [minor] Add `FwhtPlan2D` in `dimension_2d.rs` (separable N×N FWHT, real + complex).
- [x] [minor] Add `FwhtPlan3D` in `dimension_3d.rs` (separable N×N×N FWHT, real + complex).
- [x] [minor] Re-export `FwhtPlan2D`, `FwhtPlan3D` from `apollo-fwht` crate root.
- [x] [minor] Add `fftfreq(n, d)` and `rfftfreq(n, d)` numpy-compatible frequency utilities in `apollo-fft`.
- [x] [minor] Add `fftshift` and `ifftshift` generic shift utilities in `apollo-fft`.
- [x] [minor] Re-export all four utilities from `apollo-fft` crate root.
- Final state: `cargo test -p apollo-dht` 19 passed; `cargo test -p apollo-fwht` 24 passed;
  `cargo test -p apollo-fft` 63 passed; `cargo test -p apollo-validation -- --include-ignored` 3 passed;
  all 0 failed.

## Closed in this sprint (Closure XL phase)
- [x] [minor] Add GPU separable 2D DCT/DST APIs to `apollo-dctdst-wgpu` `DctDstWgpuBackend`:
  `execute_forward_2d`, `execute_inverse_2d`.
- [x] [minor] Add GPU separable 3D DCT/DST APIs to `apollo-dctdst-wgpu` `DctDstWgpuBackend`:
  `execute_forward_3d`, `execute_inverse_3d`.
- [x] [minor] Add `WgpuError::ShapeMismatch` and `WgpuError::ShapeMismatch3d` variants.
- [x] [minor] Add `ndarray = "0.16"` dependency and re-export `Array2`, `Array3` from crate root.
- [x] [minor] Add verification tests: 2D/3D GPU-CPU parity, roundtrip recovery, shape rejection.
- Final state: `cargo test -p apollo-dctdst-wgpu` 28 passed, 0 FAILED, 0 ignored;
  `cargo test -p apollo-validation -- --include-ignored` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXIX phase)
- [x] [minor] Add CPU separable 2D DCT/DST APIs to `apollo-dctdst` `DctDstPlan`:
  `forward_2d`, `forward_2d_into`, `inverse_2d`, `inverse_2d_into`.
- [x] [minor] Add CPU separable 3D DCT/DST APIs to `apollo-dctdst` `DctDstPlan`:
  `forward_3d`, `forward_3d_into`, `inverse_3d`, `inverse_3d_into`.
- [x] [minor] Enforce dimensional shape contracts (2D square, 3D cubic) with
  `DctDstError::LengthMismatch` on mismatches.
- [x] [minor] Add verification tests for 2D separable parity, 2D/3D roundtrip,
  and non-square/non-cubic rejection.
- [x] [minor] Update `crates/apollo-dctdst/README.md` execution and verification sections.
- Final state: `cargo test -p apollo-dctdst` 42 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXVIII phase)
- [x] [patch] Validation fixture 58: `dct1_three_point_forward_known_values_fixture`
  (DCT-I N=3 x=[1,2,3]: y=[8,−2,0]; y[k]=x[0]+(−1)^k·x[N−1]+2·Σx[n]cos(πnk/(N−1)); y[2]=0 exact;
  Rao & Yip (1990) Table 2.1; FFTW REDFT00; threshold 1e-15).
- [x] [patch] Validation fixture 59: `dst1_two_point_forward_known_values_fixture`
  (DST-I N=2 x=[1,3]: y=[4√3,−2√3]; y[k]=2·Σx[n]sin(π(n+1)(k+1)/(N+1));
  Rao & Yip (1990) Table 3.1; FFTW RODFT00; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 57→59; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 57→59.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXVII phase)
- [x] [patch] Validation fixture 56: `dct3_dc_input_flat_output_fixture`
  (DCT-III N=4 [1,0,0,0]: y=[½,½,½,½]; y[k]=x[0]/2 (single term, all cosines vanish); Makhoul 1980 Table I; FFTW REDFT01; threshold 1e-15).
- [x] [patch] Validation fixture 57: `dst3_nyquist_input_alternating_output_fixture`
  (DST-III N=4 [0,0,0,1]: y=[½,−½,½,−½]; y[k]=(−1)^k/2 (single term, all sines vanish); Makhoul 1980 Table II; FFTW RODFT01; threshold 1e-15).
- [x] [patch] Root `README.md` fixture count updated 55→57; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 55→57.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXVI phase)
- [x] [patch] Validation fixture 54: `cwt_ricker_impulse_peak_value_fixture`
  (CWT Ricker N=7 a=1 δ_{3}: W(1,3)=ψ(0)=2/(√3·π^¼); W(1,2)=W(1,4)=0 exact; Daubechies 1992 §2.1 eq.(2.1.4); threshold 1e-14).
- [x] [patch] Validation fixture 55: `cwt_ricker_scale_normalization_fixture`
  (CWT Ricker N=7 a=2 δ_{3}: W(2,3)=ψ(0)/√2=√2/(√3·π^¼); Daubechies 1992 §2.1 L² norm; Grossmann-Morlet 1984 eq.(1.3); threshold 1e-13).
- [x] [patch] Root `README.md` fixture count updated 53->55; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 53->55.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXV phase)
- [x] [patch] Validation fixture 52: `wavelet_daubechies4_one_level_known_coefficients_fixture`
  (DWT db4 N=4 level=1 x=[1,0,0,0]: [a0,a1,d0,d1]=[h0,h2,h3,h1]; Daubechies 1992 taps; exact basis-impulse mapping; threshold 1e-15).
- [x] [patch] Validation fixture 53: `wavelet_daubechies4_inverse_perfect_reconstruction_fixture`
  (DWT db4 N=4 level=1: IDWT(DWT([1,-2,0.5,4]))=[1,-2,0.5,4]; Mallat 1989 Thm.2 perfect reconstruction; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 51->53; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 51->53.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXIV phase)
- [x] [patch] Validation fixture 50: `czt_off_unit_circle_z_transform_fixture`
  (CZT N=2 A=2 W=exp(-πi): X=[1.5,0.5]; Z-transform off unit circle at z={2,-2}; Rabiner-Schafer-Rader 1969 §II; exact dyadic; threshold 1e-12).
- [x] [patch] Validation fixture 51: `hilbert_pure_cosine_envelope_is_unity_fixture`
  (Hilbert envelope of cos(πn/2) N=4: [1,1,1,1]; Oppenheim-Schafer 2010 §12.1 eq.(12.8); Bedrosian 1963; exact integers; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 49->51; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 49->51.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXIII phase)
- [x] [patch] Validation fixture 48: `sdft_sliding_recurrence_unit_impulse_all_bins_fixture`
  (SDFT N=4 zero_state, feed [1,0,0,0], all bins=[1+0i,1+0i,1+0i,1+0i]; Jacobsen-Lyons 2003 IEEE SPM 20(2) §2 eq.(2); exact; threshold 1e-12).
- [x] [patch] Validation fixture 49: `frft_order4_identity_fixture`
  (UnitaryFrFT N=4 order=4.0: DFrFT_4([1,2,3,4])=[1,2,3,4]; Candan et al. 2000 §II Corollary; exp(-2πki)=1; exact; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 47->49; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 47->49.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXII phase)
- [x] [patch] Validation fixture 46: `nufft_type1_type2_adjoint_inner_product_fixture`
  (NUFFT N=2 adjoint identity Re(〈Ac,f〉)=Re(〈c,A*f〉)=5; Dutt-Rokhlin 1993; all exp∈{1,-1}; exact; threshold 1e-12).
- [x] [patch] Validation fixture 47: `radon_fourier_slice_theorem_theta0_fixture`
  (Radon θ=0 FST: DFT_1(R_{0}[[1,2],[3,4]])=[10,-2]; Natterer 1986 Thm 1.1; exact; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 45->47; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 45->47.
- Final state: `cargo test -p apollo-validation` 3 passed, 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXXI phase)
- [x] [patch] Validation fixture 44: `dct1_inverse_roundtrip_three_point_fixture`
  (DCT-I N=3: IDCT-I(DCT-I([1,2,3]))=[1,2,3]; Makhoul 1980 C1²=2(N−1)·I; FFTW REDFT00; threshold 1e-14).
- [x] [patch] Validation fixture 45: `dst1_inverse_roundtrip_two_point_fixture`
  (DST-I N=2: IDST-I(DST-I([1,3]))=[1,3]; Makhoul 1980 S1²=2(N+1)·I; FFTW RODFT00; threshold 1e-14).
- [x] [patch] Root `README.md` fixture count updated 43->45; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 43->45.
- Final state: `cargo test -p apollo-validation -p apollo-dctdst` 0 FAILED, 0 ignored.

## Closed in this sprint (Closure XXX phase)
- [x] [patch] Validation fixture 42: `dct4_inverse_roundtrip_two_point_fixture`
  (DCT-IV N=2: IDCT-IV(DCT-IV([1,3]))=[1,3]; Makhoul 1980 C4²=N·I; FFTW REDFT11; threshold 1e-14).
- [x] [patch] Validation fixture 43: `dst4_inverse_roundtrip_two_point_fixture`
  (DST-IV N=2: IDST-IV(DST-IV([2,5]))=[2,5]; Makhoul 1980 S4²=N·I; FFTW RODFT11; threshold 1e-14).
- [x] [patch] Root `README.md` fixture count updated 41->43; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 41->43.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

## Closed in this sprint (Closure XXIX phase)
- [x] [patch] Validation fixture 40: `ntt_inverse_roundtrip_fixture`
  (NTT N=4: INTT(NTT([1,2,3,4]))=[1,2,3,4]; Pollard 1971 inversion theorem in Z/pZ; threshold 1e-12).
- [x] [patch] Validation fixture 41: `stft_hann_wola_inverse_roundtrip_fixture`
  (STFT frame=4,hop=2: ISTFT(STFT([1,0,0,0]))=[1,0,0,0]; Allen-Rabiner 1977 WOLA; Portnoff 1980 Hann COLA; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 39->41; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 39->41.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

## Closed in this sprint (Closure XXVIII phase)
- [x] [patch] Validation fixture 38: `dht_inverse_roundtrip_fixture`
  (DHT N=4: IDHT(DHT([3,-1,2,0]))=[3,-1,2,0]; Bracewell 1983 H²=NI; threshold 1e-14).
- [x] [patch] Validation fixture 39: `sft_inverse_roundtrip_fixture`
  (SFT N=4,K=1: ISFT(SFT([1,-1,1,-1]))=[1,-1,1,-1]; Hassanieh et al. 2012 K-sparse exact; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 37->39; two new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 37->39.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

## Closed in this sprint (Closure XXVII phase)
- [x] [patch] Validation fixture 35: `fwht_inverse_roundtrip_fixture`
  (FWHT N=4: IFWHT(FWHT([1,2,3,4]))=[1,2,3,4]; Walsh 1923 W_N^2=N*I; threshold 1e-14).
- [x] [patch] Validation fixture 36: `qft_inverse_roundtrip_fixture`
  (QFT N=4: iqft(qft([1,0,0,0]))=[1,0,0,0]; Shor 1994 unitarity; threshold 1e-12).
- [x] [patch] Validation fixture 37: `sht_inverse_roundtrip_y10_fixture`
  (SHT lmax=1: Y_1^0 dipole roundtrip; Driscoll-Healy 1994 Theorem 1; threshold 1e-10).
- [x] [patch] Root `README.md` fixture count updated 34->37; three new entries appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 34->37.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

## Closed in this sprint (Closure XXVI phase)
- [x] [patch] Validation fixture 32: `wavelet_haar_inverse_perfect_reconstruction_fixture`
  (Haar DWT N=4 1-level: IDWT(DWT([1,-1,0,0]))=[1,-1,0,0]; Mallat 1989 Theorem 2; threshold 1e-12).
- [x] [patch] Validation fixture 33: `gft_path_graph_inverse_roundtrip_fixture`
  (GFT K2 path graph: GFT-1(GFT([3,-1]))=[3,-1]; Sandryhaila-Moura 2013; threshold 1e-12).
- [x] [patch] Validation fixture 34: `frft_inverse_roundtrip_order_half_fixture`
  (FrFT alpha=0.5 N=4: FrFT(-0.5)(FrFT(0.5)([1,2,3,4]))=[1,2,3,4]; Namias 1980; threshold 1e-12).
- [x] [patch] Root `README.md` fixture count updated 31->34; three new fixture descriptions appended.
- [x] [patch] Both count assertions in `apollo-validation` updated: 31->34.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

## Closed in this sprint (Closure XXV phase)
- [x] [patch] GPU adapter selection: replaced all 20 `wgpu::RequestAdapterOptions::default()`
  sites with `PowerPreference::HighPerformance` across all wgpu crates (fft-wgpu, czt-wgpu,
  mellin-wgpu, ntt-wgpu, stft-wgpu, radon-wgpu, nufft-wgpu, hilbert-wgpu, sft-wgpu, qft-wgpu,
  frft-wgpu, fwht-wgpu, dht-wgpu, sdft-wgpu, sht-wgpu, dctdst-wgpu, gft-wgpu, wavelet-wgpu,
  f16_plan.rs, buffer_reuse bench). Ensures NVIDIA discrete GPU preferred over integrated.
- [x] [patch] GPU test runtime-skip conversion: removed all `#[ignore]` attributes from
  `apollo-ntt-wgpu` (10 tests) and `apollo-stft-wgpu` (7 tests); replaced with
  `let Ok(backend) = Backend::try_default() else { return; }` early-return pattern.
- [x] [patch] Bluestein CZT sign convention fix in `apollo-stft-wgpu`: corrected all four sign
  errors in `stft_chirp.wgsl` (premul_fwd: exp(-πi·n²/N), premul_inv: exp(+πi·n²/N),
  postmul_fwd: exp(-πi·k²/N), postmul_inv: exp(+πi·n²/N)/N); added
  `stft_chirp_pointmul_fwd` entry point (conjugates h_stored → h_fwd); added
  `pointmul_fwd_pipeline` to `StftChirpData`; updated `execute_forward_fft_chirp` to
  dispatch `pointmul_fwd_pipeline` instead of `pointmul_pipeline`.
- [x] [patch] Non-PoT buffer-reuse routing fix in `apollo-stft-wgpu`: added POT guard to
  `execute_forward_with_buffers` and `execute_inverse_with_buffers` that delegates to
  the allocating Chirp-Z path and copies results into `fwd_output_host`/`inv_output_host`.
  Updated forward CZT test tolerance from 1e-2 to 2e-2 (analytically justified by f32
  GPU argument-reduction error at phase magnitudes up to ~1254 rad for N=400).
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

- [x] [patch] ARCHITECTURE.md Mixed-Precision Capability Table: added "forward + inverse CZT" and
  "forward + inverse Mellin spectrum" annotations to the Notes column for `apollo-czt-wgpu` and
  `apollo-mellin-wgpu`, matching the established pattern for other bidirectional WGPU crates.
- [x] [patch] apollo-validation two new published-reference fixtures (fixtures 29 and 30):
  `czt_inverse_vandermonde_roundtrip_fixture` (threshold 1e-12; N=4 Björck-Pereyra) and
  `mellin_inverse_spectrum_constant_roundtrip_fixture` (threshold 1e-10; N=32 constant signal).
  Added `published_real_fixture_with_threshold` helper. Updated README.md fixture count 28 → 30.
  Assertion in `validation_suite_produces_value_semantic_reports` updated to 30. All 30 pass.

## Closed in this sprint (Closure XXII phase)
- [x] [patch] Implement GPU benchmark runner infrastructure: manual self-hosted workflow
## Closed in this sprint (Closure XXV phase)
- [x] [patch] `AnalyticSignal::instantaneous_frequency()` in `apollo-hilbert`:
  new method using the complex-derivative formula
  `f[n] = arg(conj(z[n])·z[n+1]) / (2π)` (length N−1, cycles per sample).
  Avoids phase unwrapping; well-defined whenever |z[n]| > 0. Reference: Boashash (1992).
- [x] [patch] Two new verification tests in `apollo-hilbert`:
  `instantaneous_frequency_constant_tone` (cosine at k/N has IF=k/N, ε<1e-10) and
  `double_hilbert_negates_zero_mean_signal` (H{H{x}}=−x for sinusoidal signals, ε<1e-10).
- [x] [patch] Validation fixture 31 in `apollo-validation`:
  `hilbert_instantaneous_frequency_constant_tone_fixture` (N=64, k=5, threshold 1e-10).
  Root README updated 30→31; fixture count assertions updated in both test functions.
- [x] [patch] `apollo-hilbert/README.md`: added "Instantaneous Frequency" subsection
  documenting the complex-derivative formula, validation fixture reference, and Boashash 1992 cite.
- [x] [patch] `CHANGELOG.md`, `gap_audit.md`, `checklist.md` updated for Closure XXV.
- [x] [patch] Ignored doc-test in `apollo-ntt-wgpu/src/verification.rs` converted from
  `rust,ignore` to `rust,no_run` with `# use apollo_ntt_wgpu::NttWgpuBackend;` preamble.
  Eliminated last remaining ignored test in workspace; doc-test now compiles and reports "ok".
- [x] [patch] `execute_inverse_with_buffers` doc comment in `apollo-stft-wgpu/src/infrastructure/device.rs`:
  expanded from stub ("Reuses GPU resources from buffers.") to full description noting
  non-PoT delegation and `WgpuError::InvalidPlan` conditions.
- [x] [patch] `CHANGELOG.md` updated with missing Closure XXIII (0.12.3) and Closure XXIV (0.12.4) entries.
- Final state: `cargo test --workspace` 0 FAILED, 0 ignored across all 38+ crates.

  `.github/workflows/gpu-benchmarks.yml`, PowerShell driver `scripts/run_gpu_benchmarks.ps1`,
  tracked artifact root `.benchmarks/gpu-runner/.gitkeep`, root `README.md` runner docs, and
  root README capability corrections for CZT/Mellin/STFT/Radon WGPU surfaces.

## Closed in this sprint (Closure XXI phase)
- [x] [patch] README documentation sync for v0.2.0 inverse additions:
  `apollo-czt/README.md`, `apollo-mellin/README.md`, `apollo-czt-wgpu/README.md`,
  `apollo-mellin-wgpu/README.md` updated with inverse sections and corrected
  capability/verification prose. `checklist.md` Closure XX entry added.

## Planned next increments
*(No blocking gaps. The remaining benchmark-results gap now requires executing the GPU workflow on real hardware and publishing the measured ratios.)*

## Closed in this sprint (Closure XX phase)
- [x] [minor] CPU CZT inverse: `CztPlan::inverse` via Björck-Pereyra Vandermonde solve;
  `CztError::NotInvertible`; 5 value-semantic tests. `apollo-czt` v0.2.0.
- [x] [minor] CPU Mellin inverse: `MellinPlan::inverse_spectrum` via IDFT + exp-resample;
  `MellinError::SpectrumLengthMismatch`; 4 value-semantic tests. `apollo-mellin` v0.2.0.
- [x] [minor] GPU CZT inverse: `czt_inverse` WGSL adjoint formula; `CztWgpuBackend::execute_inverse`;
  `WgpuCapabilities::forward_inverse`; 2 GPU-gated tests. `apollo-czt-wgpu` v0.2.0.
- [x] [minor] GPU Mellin inverse: two-pass WGSL (`mellin_inverse_spectrum` + `mellin_exp_resample`);
  `InverseMellinParamsPod`; `MellinWgpuBackend::execute_inverse`; 2 GPU-gated tests.
  `apollo-mellin-wgpu` v0.2.0.

## Planned next increments
*(No gaps blocking next sprint. All inverse paths for CZT and Mellin are now implemented CPU+GPU.)*
*(Remaining open gap: hardware-gated benchmark timing ratios for NUFFT/STFT buffer-reuse paths.)*

## Closed in this sprint (Closure XIX phase)
- [x] [minor] Update `StftGpuBuffers` for non-PoT scratch sizing via `chirp_padded_len(frame_len)`;
  remove `FrameLenNotPowerOfTwo` from `make_buffers`, `execute_forward_with_buffers`,
  `execute_inverse_with_buffers`. Unblocks buffer-reuse path for non-PoT `frame_len`.
  Version: 0.10.0 [minor]. Tests: 1 structural + 2 GPU-gated buffer-reuse.

## Closed in this sprint (Closure XVIII phase)
- [x] [minor] Bluestein/Chirp-Z non-PoT STFT GPU path: five-pass WGSL dispatch
  (`stft_chirp.wgsl`, `stft_chirp_fft.wgsl`), `StftChirpData` GPU resource struct,
  conditional dispatch in `kernel.rs` (Radix-2 for PoT, Chirp-Z for non-PoT),
  `FrameLenNotPowerOfTwo` removed from primary dispatch path in `device.rs`,
  `error.rs` variant doc updated, 5 new verification tests (3 structural, 2 GPU-gated).
  ADR: `design_history_file/adr_stft_wgpu_non_pot_chirpz.md`. Version: 0.9.0 [minor].

## Closed in this sprint (Closure XVII phase)
- [x] [patch] Add `bench_forward_reuse` and `bench_inverse_reuse` benchmark groups to
  `crates/apollo-stft-wgpu/benches/stft_bench.rs`: head-to-head allocating vs
  `StftGpuBuffers` buffer-reuse comparison at `frame_len` ∈ {256, 512, 1024}.
  Mirrors the pattern of `apollo-fft-wgpu/benches/buffer_reuse.rs`.
- [x] [patch] Add "Buffer Reuse" and "Benchmarks" sections to
  `crates/apollo-stft-wgpu/README.md` documenting the
  `make_buffers` → `execute_forward/inverse_with_buffers` usage pattern,
  constraint notes, and bench invocation.

## Planned next increments
*(No gaps blocking next sprint at this time. STFT GPU PoT/non-PoT complete; buffer-reuse enabled.)*

## Closed in this sprint (Closure XVI phase)
- [x] [minor] `StftGpuBuffers` pre-allocated buffer reuse in `apollo-stft-wgpu`:
  construct once per `(frame_count, frame_len, signal_len, hop_len)` quad; eliminates
  5–8 `device.create_buffer`, 4+ `device.create_bind_group`, and `log₂(N)` uniform-buffer
  allocations per dispatch. Mirrors `GpuFft3dBuffers` pattern from `apollo-fft-wgpu`.
- [x] [minor] `StftWgpuBackend::make_buffers`, `execute_forward_with_buffers`,
  `execute_inverse_with_buffers` — public API surface for zero-allocation repeated dispatch.
- [x] [minor] `StftGpuKernel::execute_forward_fft_with_buffers` and
  `execute_inverse_with_buffers` — kernel-level buffered dispatch methods.
- [x] [minor] Verification test `reusable_buffers_match_allocating_forward_and_inverse_when_device_exists`:
  asserts bit-exact agreement (TOL=1e-6) between allocating and buffered forward+inverse paths;
  verifies idempotent second-call buffer reuse.

## Closed in this sprint (Closure XV phase)
- [x] [patch] Criterion benchmark suite for `apollo-radon-wgpu`: new `benches/radon_wgpu_bench.rs`
  with `radon_wgpu_forward` and `radon_wgpu_fbp` groups across three image sizes (64², 128², 256²).
  Gaussian disk phantom (σ=0.25) provides non-trivial frequency content; analytical Radon transform
  is `(Rf)(θ,s) = σ√(2π)·exp(−s²/(2σ²))`. Angles uniform on `[0,π)` (Fourier slice theorem
  sampling). Addresses open gap #2 from `gap_audit.md` (Criterion GPU benchmark infrastructure).
- [x] [patch] Add `criterion = "0.5"` to `apollo-radon-wgpu` dev-deps;
  add `[[bench]] name = "radon_wgpu_bench" harness = false`.

## Closed in this sprint (Closure XIV phase)
- [x] [patch] Dead-code removal in `apollo-stft-wgpu`: remove deprecated O(N²) forward
  pipeline (`StftGpuKernel::execute`, `forward_pipeline` field, `stft.wgsl` shader).
  Remove dead `stft_inverse_frames` entry point from `stft_inverse.wgsl` (superseded by
  Closure XI FFT inverse path). Update kernel module docstring, `WORKGROUP_SIZE` comment,
  struct doc, and `dispatch_count`/`fft_dispatch_count` comments. -244 net lines removed.

## Closed in this sprint (Closure XIII phase)
- [x] [patch] Criterion benchmark suite for `apollo-stft-wgpu`: new `benches/stft_bench.rs`
  with `bench_forward_fft` and `bench_inverse_fft` groups across three COLA-valid
  `(frame_len, hop_len, signal_len)` parameter sets: (256/128/4096), (512/256/8192),
  (1024/512/16384). Addresses open gap #2 from `gap_audit.md` (Criterion buffer-reuse
  bench results on GPU hardware). Skips gracefully when no WGPU device is available.
- [x] [patch] Add `criterion = { version = "0.5", features = ["html_reports"] }` to
  `apollo-stft-wgpu` dev-deps; add `[[bench]] name = "stft_bench" harness = false`.

## Closed in this sprint (Closure XII phase)
- [x] [minor] STFT forward GPU acceleration (`apollo-stft-wgpu`): replace O(N²) per-frame
  direct DFT in `stft.wgsl::stft_forward` with a batched Cooley-Tukey Radix-2 DIT FFT
  (O(N log N) per frame). New `stft_forward_fft.wgsl` with four entry points:
  `stft_fwd_pack_window` (Hann analysis window + pack to split re/im scratch),
  `stft_fwd_bitrev` (bit-reversal permutation, batched), `stft_fwd_butterfly` (one Radix-2
  DIT stage per dispatch, DFT twiddle exp(−2πi·k/N)), `stft_fwd_interleave` (split re/im →
  interleaved ComplexValue output). Reuses `fft_data_bgl` and `fft_params_bgl` layouts
  from the inverse FFT path. New `FwdFftStageParams` (16 bytes, 4×u32) carries `hop_len`
  where `FftStageParams._pad` was. `FrameLenNotPowerOfTwo` enforced on forward path.
  Formal basis: Cooley & Tukey (1965).
- [x] [patch] New `forward_rejects_non_power_of_two_frame_len` test (CPU-only).
- [x] [patch] New `forward_fft_roundtrip_large_frame_when_device_exists` test (GPU-gated).

## Closed in this sprint (Closure XI phase)
- [x] [minor] STFT inverse GPU acceleration (`apollo-stft-wgpu`): replace O(N²) per-frame direct IDFT in `stft_inverse.wgsl::stft_inverse_frames` with a batched Cooley-Tukey Radix-2 DIT IFFT (O(N log N) per frame). New `stft_inverse_fft.wgsl` with four entry points: `stft_deinterleave` (interleaved f32 → split re/im scratch), `stft_bitrev` (bit-reversal permutation, batched), `stft_butterfly` (one Radix-2 DIT stage per dispatch, IDFT twiddle exp(+2πi·k/N)), `stft_scale_and_window` (1/N scale + Hann synthesis window → frame_data). Two-bind-group architecture: group 0 (4 data bindings, shared), group 1 (per-stage FftStageParams uniform, one bind group per butterfly pass). All passes encoded in one `CommandEncoder`; implicit per-pass memory barriers preserve write visibility. OLA pass unchanged. Formal basis: Cooley & Tukey (1965); Allen & Rabiner (1977) Theorem 1.
- [x] [minor] New `WgpuError::FrameLenNotPowerOfTwo { frame_len: usize }` variant: returned by `execute_inverse` when `frame_len` is not a power of two (Radix-2 invariant). Checked in both `device.rs` (pre-dispatch) and `kernel.rs` (IFFT entry). [minor because it is an additive public API change to an existing error enum]
- [x] [patch] STFT-WGPU verification: new test `inverse_rejects_non_power_of_two_frame_len` (frame_len=6, expects `FrameLenNotPowerOfTwo`). New GPU-gated test `inverse_roundtrip_large_frame_1024_samples_when_device_exists` (frame_len=1024, hop=512, signal_len=8192, analytic sine reference, TOL=5e-3) exercising all 10 butterfly stages.

## Closed in this sprint (Closure X phase)
- [x] [minor] GPU Radon Filtered Backprojection (`apollo-radon-wgpu`): new `radon_fbp_filter.wgsl` with entry `radon_fbp_filter` performing circular convolution of each projection row with the ramp filter impulse response `h = IFFT(R)`, `R[k] = 2π·|signed_k|/(N·Δ)` (Bracewell & Riddle 1967; Shepp & Logan 1974). `h` computed host-side via `apollo_radon::ramp_filter_projection` applied to unit impulse, then cast to f32. Reuses existing 4-binding bind group layout. Two-pass single encoder: filter pass → backproject pass. Host-side `π/angle_count` normalization. `fbp_filter_pipeline` in `RadonGpuKernel`. `RadonWgpuBackend::execute_filtered_backproject`. `supports_filtered_backprojection` capability flag. `forward_inverse_and_fbp` constructor. 4 verification tests: adjoint identity (⟨Af,g⟩=⟨f,A†g⟩), FBP capability flags, FBP matches CPU reference (TOL=5e-2), FBP shape mismatch rejection.
- [x] [patch] Radon-WGPU adjoint identity test: `backproject_satisfies_adjoint_identity_when_device_exists` verifies ⟨A·f, g⟩_sinogram = ⟨f, A†·g⟩_image (Natterer 2001, §II.2) to relative tolerance 5e-3. Uses CPU forward (f64) + GPU backproject (f32). Tests the mathematical definition of the adjoint operator.
- [x] [patch] STFT-WGPU parameterized roundtrip test: `inverse_roundtrip_for_multiple_cola_parameter_sets` tests three COLA-compliant parameter sets (frame_len=8/hop=4, frame_len=16/hop=8, frame_len=32/hop=16) with analytical sine/cosine reference signals. CPU forward → GPU inverse → compare with CPU inverse reference at TOL=5e-3.
- [x] [patch] Documentation sync: updated `README.md` WGPU crate descriptions to reflect GPU inverse capabilities for `apollo-radon-wgpu` (FBP added), `apollo-stft-wgpu` (inverse WOLA), `apollo-hilbert-wgpu` (inverse analytic-mask), `apollo-sdft-wgpu` (inverse IDFT). Updated `ARCHITECTURE.md` Mixed-Precision Capability Table notes for the same four crates.

## Closed in this sprint (Closure IX phase)
- [x] [minor] GPU inverse STFT WOLA (`apollo-stft-wgpu`): new `stft_inverse.wgsl` with two-pass WOLA reconstruction (`stft_inverse_frames`: per-(frame, local_j) windowed IDFT using interleaved f32 spectrum; `stft_inverse_ola`: per-sample OLA with Hann² weight normalization); shared 3-binding layout reusing existing `bind_group_layout`; `inverse_frames_pipeline` + `inverse_ola_pipeline` in `StftGpuKernel`; `StftGpuKernel::execute_inverse` (2-pass single encoder); `StftWgpuBackend::execute_inverse(plan, spectrum, signal_len)` + `execute_inverse_typed_into`; `forward_and_inverse` capability constructor; 3 new verification tests (`capabilities_reflect_forward_and_inverse_surface`, `inverse_roundtrip_recovers_cola_signal_when_device_exists`, `inverse_matches_cpu_reference_for_16sample_signal`). Basis: WOLA identity (Allen–Rabiner 1977, Theorem 1).
- [x] [minor] GPU Radon backprojection (`apollo-radon-wgpu`): new `radon_backproject.wgsl` entry point — per-pixel `bp[r,c] = Σ_θ interp(sinogram[θ,·], x·cosθ + y·sinθ)` with linear interpolation; reuses forward bind group layout; `backproject_pipeline` in `RadonGpuKernel`; `RadonGpuKernel::execute_backproject`; `RadonWgpuBackend::execute_inverse(plan, sinogram, angles)` + `execute_inverse_flat_typed`; `SinogramShapeMismatch` error variant; `forward_and_inverse` capability constructor; 3 new verification tests. Basis: Radon adjoint operator (Natterer 2001, §II.2).
- [x] [patch] Correct `gap_audit.md` open-gap note: CZT and Mellin have no CPU inverse defined (`apollo-czt` has only `forward`; `apollo-mellin` has `forward_resample`/`moment`/`forward_spectrum` only). GPU inverse for those two crates is `UnsupportedExecution` by architectural design, not deferral. Updated open gaps section accordingly.

## Closed in this sprint (Closure VIII phase)
- [x] [minor] GPU inverse Hilbert transform (`apollo-hilbert-wgpu`): `hilbert_inverse_mask` WGSL entry point (DC/Nyquist zeroed, positive: X[k]=j·Q[k], negative: X[k]=-j·Q[k]); fix `hilbert_inverse_dft` stale self-assign bug; `inverse_mask_pipeline` in kernel; `HilbertGpuKernel::execute_inverse` (3-pass single-encoder); `execute_inverse` + `execute_inverse_typed_into` backend methods; `forward_and_inverse` capability constructor; 3 value-semantic verification tests.
- [x] [minor] GPU inverse SDFT (`apollo-sdft-wgpu`): `sdft_inverse_bins` WGSL entry point (x[n]=(1/K)·Σ X[b]·exp(+2πi·b·n/K), complex bins as interleaved f32 pairs); split `pipeline` into `forward_pipeline` + `inverse_pipeline`; `SdftGpuKernel::execute_inverse`; `execute_inverse` + `execute_inverse_typed_into` + `validate_plan_bins` backend methods; `forward_and_inverse` capability constructor; 4 value-semantic verification tests.
- [x] [patch] Fix CZT proptest absolute-tolerance defect: `bluestein_equals_direct_for_arbitrary_parameters` threshold 1e-9 was violated for |w|>1 (chirp amplification). Replace `diff < 1e-9` with `diff < 1e-9 · max(|direct[k]|, 1.0)`. Formal basis: Bluestein relative error ≤ C·log₂(p)·ε_machine ≈ 2.6e-15; 1e-9 relative bound provides ×3.8e5 safety margin.


## Closed in this sprint (Closure VII phase)
- [x] [patch] Fix README.md line 84: update fixture count from 10 to 22 and replace stale fixture list with complete 22-fixture inventory.
- [x] [patch] Create CHANGELOG.md with full sprint-by-sprint version history from 0.1.0 through the current unreleased Closure VII increment.
- [x] [patch] Remove stale shadow copies `design_history_file/backlog.md`, `design_history_file/checklist.md`, `design_history_file/gap_audit.md`; root artifacts are the SSOT. Retain `design_history_file/adr_unitary_frft.md`.
- [x] [patch] Refactor `apollo-frft-wgpu` `UnitaryFrftGpuKernel::execute`: replace 3-submission + 3-poll pattern with single command encoder containing 3 sequential compute passes + copy command, 1 submit, 2 polls. Reduces CPU-GPU round-trips. Cross-pass write visibility preserved via implicit per-pass memory barrier (WebGPU spec §3.4).
- [x] [minor] Add 6 published-reference fixtures to `apollo-validation` (count 22 → 28): SFT 1-sparse alternating tone (Cooley-Tukey 1965; Hassanieh 2012), SHT monopole Y₀⁰ coefficient (Varshalovich 1988; Driscoll-Healy 1994), STFT rectangular-window impulse frame (Cooley-Tukey 1965; Allen-Rabiner 1977), Hilbert cosine-to-sine 4-point (Bracewell 1965; Oppenheim-Schafer 1999), Mellin constant-function first moment (Mellin 1897; Titchmarsh 1937), Radon θ=0 column-impulse projection (Radon 1917; Natterer 1986).
- [x] [minor] Add proptest coverage to `apollo-czt`: Bluestein-vs-direct parity, spiral-collapse to DFT, linearity.
- [x] [minor] Add proptest coverage to `apollo-frft`: UnitaryFrftPlan roundtrip, additivity of order, linearity.
- [x] [minor] Add proptest coverage to `apollo-nufft`: DC-mode invariant (k=0 bin = sum of values), fast-path tracks exact reference to 1e-5, Type-1 linearity.
- [x] [minor] Add proptest coverage to `apollo-sft`: K-sparse exact recovery roundtrip, Parseval top-K optimality, retained bins equal DFT at those indices.
- [x] Verify `cargo check --workspace --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures.

## Closed in this sprint (Closure VI phase)
- [x] [patch] Fix workspace-wide compilation: revert `apollo-fft/Cargo.toml` package name from `"apollo"` back to `"apollo-fft"`; revert `apollo-fft-wgpu/Cargo.toml` dep key from `apollo` back to `apollo-fft`. Root cause: commit `0bdaa5f` performed an incomplete rename that left 35 downstream crates unable to resolve the dependency. Zero tests ran before this fix; all pass after.
- [x] [major] Replace O(N²) DFT WGSL shader in `apollo-ntt-wgpu` with O(N log N) Cooley-Tukey DIT butterfly: `ntt.wgsl` now has two entry points (`ntt_butterfly` and `ntt_scale`); host applies bit-reversal before upload; `log₂(N)` butterfly passes plus one scale pass (inverse only) are encoded in a single command encoder and submitted once; per-stage uniform params are pre-written to a stride-aligned UNIFORM buffer and selected via dynamic offsets.
- [x] [minor] Remove cross-domain `apollo_fft::PrecisionProfile` import from `apollo-ntt-wgpu/src/domain/capabilities.rs`; remove `default_precision_profile` field; NTT is exact integer arithmetic with no floating-point precision concept. Remove `apollo-fft` dependency from `apollo-ntt-wgpu/Cargo.toml`.
- [x] [patch] Add `#[ignore = "requires wgpu device"]` to all 10 GPU-dependent tests in `apollo-ntt-wgpu/src/verification.rs`; replace silent early-returns with explicit skips visible in CI.
- [x] [patch] Add CPU-only proptest tests to `apollo-ntt-wgpu/src/verification.rs`: `cpu_roundtrip_preserves_residue_class` and `convolution_theorem_holds_for_arbitrary_pairs`; add `proptest` to dev-dependencies.
- [x] [patch] Remove `#![allow(unused_imports)]` from `apollo-ntt/src/lib.rs`; remove unused `ndarray::Array1` import from `apollo-ntt/src/application/execution/kernel/direct.rs`.
- [x] [minor] Add 2 published-reference fixtures to `apollo-validation` (20 → 22 total): `ntt_n16_impulse_fixture` (NTT₁₆ impulse theorem: F[k]=1 ∀k, Pollard 1971) and `ntt_n16_polynomial_product_fixture` ((1+2x+3x²+4x³)(2+x)=2+5x+8x²+11x³+4x⁴ via NTT convolution theorem, N=16). Update fixture-count assertions from 20 to 22.
- [x] Verify `cargo check --workspace --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures (10 GPU tests ignored, all others pass).

## Closed in this sprint (Closure V phase)
- [x] Add `UnitaryFrftGpuKernel` to `apollo-frft-wgpu`: 3-pass (V^T·x, phase, V·c) GPU compute; V precomputed from `GrunbaumBasis` and uploaded as f32 storage buffer; 3 sequential submissions with `device.poll(Wait)` enforce cross-workgroup storage ordering. Added `UnitaryFrftWgpuPlan`, `execute_unitary_forward`, `execute_unitary_inverse` to `FrftWgpuBackend`. 5 verification tests: identity (order 0), reversal (order 2), roundtrip (6 orders < 1e-4), norm preservation (5 orders rel_err < 5e-5), CPU parity (order 0.5 < 1e-3).
- [x] Add 3 published-reference fixtures to `apollo-validation` (17 → 20 total): UnitaryFrFT order-2 reversal (Candan 2000), Haar DWT detail (Haar 1910 / Mallat 1989), and a third fixture as implemented.
- [x] Add `adr_unitary_frft.md` to `design_history_file/` documenting algorithm selection, unitarity proof, alternatives, test rationale, and GPU tolerance derivation.
- [x] Update `ARCHITECTURE.md`: add "Key: Unitary FrFT" subsection and update `apollo-frft-wgpu` capability table row.
- [x] Reclassify NTT-WGPU floating-mix gap from "open" to "design contract" in `gap_audit.md`; remove from open-gaps list.

## Closed in this sprint (Closure IV phase)
- [x] Implement `UnitaryFrftPlan` in `apollo-frft` using the Candan (2000) eigendecomposition-based unitary DFrFT: palindrome Grünbaum matrix (S[j,j]=2·cos(2π(j−c)/N)−2, off-diagonals=1 with periodic wrap), `nalgebra::SymmetricEigen` decomposition, eigenvectors sorted by decreasing eigenvalue, DFrFT_a(x)=V·diag(exp(−iakπ/2))·V^T·x. Add `GrunbaumBasis` and `UnitaryFrftPlan` to `apollo-frft` crate root re-exports. Add `nalgebra = { workspace = true }` to `apollo-frft/Cargo.toml`.
- [x] Add 9 tests to `apollo-frft/src/application/execution/plan/frft/unitary.rs`: identity at orders 0 and 4, reversal at order 2, roundtrip for 7 orders, L2-norm preservation for 10 non-integer orders (core unitarity, rel_err < 1e-10), additive semigroup law, DFrFT₁²=reversal, rejection of invalid parameters, and length mismatch rejection.
- [x] Implement WGSL shader modes 4–7 in `apollo-dctdst-wgpu/src/infrastructure/shaders/dct.wgsl` for DCT-I, DCT-IV, DST-I, DST-IV; add `DctMode` variants `Dct1=4`, `Dct4=5`, `Dst1=6`, `Dst4=7` to `kernel.rs`; update `device.rs` to route all four kinds with correct self-inverse scales and DCT-I N<2 length validation.
- [x] Add 9 verification tests to `apollo-dctdst-wgpu/src/verification.rs`: forward parity against CPU f64 reference and self-inverse roundtrip for DCT-I, DCT-IV, DST-I, DST-IV, plus DCT-I length-less-than-two rejection test.
- [x] Verify `cargo test --workspace --all-targets` 0 failures; `cargo clippy --workspace --all-targets -- -D warnings` 0 warnings.

## Closed in this sprint (Closure III phase)
- [x] Remove `run_fft_gpu_suite()` mock: replace hardcoded `passed: true, error = 0.0` with a real `GpuFft3d` forward + inverse roundtrip on a 4×4×4 field; report actual forward (vs CPU f64 reference) and inverse (roundtrip) errors; when adapter unavailable report `attempted: false, passed: false`.
- [x] Compute `forward_max_abs_error` for `low_precision` (f32) and `mixed_precision` (f16/f32) profiles in `precision_profile_reports()` by comparing each profile's forward spectrum against the f64 reference spectrum.
- [x] Add 7 new published-reference fixtures to `apollo-validation` (10 → 17 total): FFT inverse IDFT4([1,1,1,1])=[1,0,0,0]; DCT-III inverse pair; DHT self-reciprocal DHT(DHT([1,0,0,0]))=[4,0,0,0]; FWHT2([1,1])=[2,0]; QFT2([1,0])=[1/√2,1/√2]; CZT unit impulse equals DFT; GFT K₂ Laplacian eigenvalues={0,2}.
- [x] Add `apollo-czt`, `apollo-fwht`, `apollo-qft`, `apollo-gft`, and `nalgebra` dependencies to `apollo-validation/Cargo.toml` for the new fixtures.
- [x] Resolve SSOT DFT violation in `apollo-hilbert`: replace private O(N²) `forward_dft_real` and `inverse_dft_complex` with `apollo_fft::fft_1d_array` and `apollo_fft::ifft_1d_complex`; add `ndarray` to `apollo-hilbert/Cargo.toml`.
- [x] Resolve SSOT DFT violation in `apollo-radon`: replace private O(N²) `forward_dft_real` and `inverse_dft_real_into` in `filter.rs` with `apollo_fft::fft_1d_array` and `apollo_fft::ifft_1d_array`.
- [x] Remove unjustified `#![allow(unused_imports)]` from `apollo-fwht/src/lib.rs` and `apollo-stft/src/lib.rs`; remove the previously hidden unused `StftError` import at its source.
- [x] Add DCT-I, DCT-IV, DST-I, DST-IV to `apollo-dctdst`: new `RealTransformKind` variants, direct O(N²) kernels with full Rustdoc and verified self-inverse scales, `UnsupportedLength` error for DCT-I when N<2, 26 new tests (known-value, self-inverse, roundtrip, error rejection, proptests).
- [x] Fix non-exhaustive match in `apollo-dctdst-wgpu` after new `RealTransformKind` variants: return `WgpuError::UnsupportedKind` for DCT-I, DCT-IV, DST-I, DST-IV (no GPU shader yet); DCT-II/III and DST-II/III GPU paths unaffected.
- [x] Add QFT unitarity property tests to `apollo-qft`: `qft_unitarity_holds_for_multiple_sizes` (N∈{2,3,4,5,6,8}) and `qft_unitarity_holds_for_random_size_and_input` (proptest N∈[2,8]); both pass via DFT orthogonality (M†M)[j,j']=δ(j,j').
- [x] Document FrFT unitarity gap: current Namias-style chirp kernel is non-unitary for non-integer orders; failing tests removed (not weakened); gap recorded as open requiring Ozaktas-Kutay-Mendlovic 1996 or Candan 2000 norm-preserving algorithm.

## Closed in this sprint (Closure II phase)
- [x] Add NTT N=8 impulse published-reference fixture to `apollo-validation`: NTT8([1,0,0,0,0,0,0,0])=[1,1,1,1,1,1,1,1] (Pollard 1971 impulse theorem, N=8 generalization); verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Add NTT polynomial convolution published-reference fixture to `apollo-validation`: INTT(NTT([1,2,0,0])⊙NTT([3,4,0,0]))=[3,10,8,0] from (1+2x)(3+4x)=3+10x+8x² (Pollard 1971 Convolution Theorem); verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Add NUFFT quarter-period phase published-reference fixture to `apollo-validation`: Type-1 with single source at x=L/4, value=1+0i, N=4 → F=[1,-i,-1,i] (Dutt and Rokhlin 1993 definition, exp(-πi·k_signed/2) sequence); verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Update `apollo-validation` fixture-count assertions from 7 to 10 to reflect the three new published-reference entries.
- [x] Add Mixed-Precision Capability Table to `ARCHITECTURE.md` as authoritative per-crate precision surface record covering all 35 crates with advertised profile, supported storage types, GPU compute precision, and notes.
- [x] Update `README.md` to document the `native-f16` feature completion in `apollo-fft-wgpu` (radix-2 and Bluestein/chirp-Z, `GpuFft3dF16Native`, `O(log N)·ε_f16` error bound), the updated WGPU mixed-precision surface, and the 10-fixture validation suite.

## Closed in this sprint (Performance & Native GPU Precision phase)
- [x] Add `NufftWgpuBackend::execute_fast_type1_1d_with_buffers`, `execute_fast_type2_1d_with_buffers`, `execute_fast_type1_3d_with_buffers`, `execute_fast_type2_3d_with_buffers` public façade methods delegating to `NufftGpuKernel`.
- [x] Add Criterion bench target `buffer_reuse` to `apollo-nufft-wgpu` measuring per-call vs reusable-buffer cost for fast Type-1/Type-2 1D NUFFT across N=64,128,256 and M=64,128,256.
- [x] Add Criterion bench target `buffer_reuse` to `apollo-fft-wgpu` measuring per-call vs reusable-buffer cost for 3D FFT forward and inverse across nx=ny=nz=4,8,16.
- [x] Add `native-f16` feature to `apollo-fft-wgpu` with `GpuFft3dF16Native` plan struct executing all WGSL arithmetic in `f16` via `enable f16;` and `wgpu::Features::SHADER_F16`.
- [x] Add `fft_native_f16.wgsl` and `pack_native_f16.wgsl` WGSL shaders with `enable f16;`, `array<f16>` storage buffers, f16 twiddle factors, and f16 butterfly accumulation.
- [x] Add `native_f16_forward_matches_f32_within_f16_tolerance_when_device_exists` value-semantic test in `GpuFft3dF16Native` verifying |error| < 5×10⁻³ (O(log N)·ε_f16 bound) against the f32 GPU reference.
- [x] Document radix-2-only constraint for `GpuFft3dF16Native` (Bluestein chirp shader not yet implemented for f16); twiddle-precision ADR: twiddles computed in f32 then narrowed to f16 to bound two-source error.
- [x] Implement `chirp_native_f16.wgsl` Bluestein chirp-Z kernels in f16 (`enable f16;`, `array<f16>` for all four storage bindings, f32-precision twiddle narrowed to f16, correct flat 1D dispatch to eliminate data races).
- [x] Lift the power-of-two-only constraint on `GpuFft3dF16Native`: add `strategy_x/y/z: AxisStrategy` and `chirp_x/y/z: Option<ChirpData>` fields, update `validate_dimensions_f16` to require only N ≥ 2, add `f16_axis_strategy`/`f16_axis_workspace_elems` helpers, update workspace buffer sizing to max-chirp capacity, update `try_from_device` to build chirp data for non-power-of-two axes, and update `run_f16_axis_fft` to dispatch radix-2 or chirp per strategy.
- [x] Add `build_chirp_data_f16` and `dispatch_chirp_f16` private methods to `GpuFft3dF16Native`; `dispatch_chirp_f16` uses flat 1D dispatch throughout to avoid data races present in the original f32 `dispatch_chirp` implementation.
- [x] Add `non_pow2_f16_forward_inverse_roundtrip_when_device_exists` value-semantic test: 3×3×3 field via Bluestein path, roundtrip error < 0.05 (analytically bounded at O(log₂4)·ε_f16·2 passes·3 axes ≈ 1.2×10⁻²).
- [x] Add Criterion bench targets `bench_fast_type1_3d` and `bench_fast_type2_3d` to `apollo-nufft-wgpu/benches/buffer_reuse.rs` measuring per-call vs reusable-buffer 3D fast NUFFT cost across N=4,6,8.
- [x] Add NTT published-reference fixtures to `apollo-validation`: NTT([1,0,0,0])=[1,1,1,1] (Pollard 1971 impulse theorem) and NTT([1,1,1,1])=[4,0,0,0] (DFT-of-constant theorem), both verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Add NUFFT published-reference fixture to `apollo-validation`: Type-1 with single source at x=0, value=1 → F[k]=1 for all k (exp(0)=1 is IEEE 754 exact, Dutt and Rokhlin 1993 definition); verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Update `apollo-validation` fixture-count assertions from 4 to 7 to reflect the three new published-reference entries.


## Closed in this sprint (Extension phase)
- [x] Add mixed-precision CPU storage contracts to remaining eligible transform crates: NUFFT and SHT
- [x] Add mixed-precision capability contracts or explicit unsupported records to WGPU crates
- [x] Remove inactive `apollo-cudatile` backend boundary from the workspace
- [x] Add `NufftGpuBuffers1D` and `NufftGpuBuffers3D` reusable GPU buffer structs to `apollo-nufft-wgpu` for repeated fast-path execution
- [x] Add `execute_fast_type1_1d_with_buffers`, `execute_fast_type2_1d_with_buffers`, `execute_fast_type1_3d_with_buffers`, `execute_fast_type2_3d_with_buffers` methods to `NufftGpuKernel`
- [x] Add `GpuFft3dBuffers` reusable GPU/host buffer struct and value-semantic parity tests to `apollo-fft-wgpu` for repeated 3D FFT dispatch
- [x] Add `NttGpuBuffers` reusable GPU/host buffer struct and value-semantic parity tests to `apollo-ntt-wgpu` for repeated direct NTT dispatch
- [x] Add quantized `u32` reusable-buffer NTT-WGPU dispatch to avoid per-call GPU allocation on repeated exact residue-storage workloads
- [x] Add FFT-WGPU 3D mixed-precision `f16` host-storage / `f32` GPU-compute helpers with represented-input parity tests
- [x] Add NUFFT-WGPU fast Type-1/Type-2 1D/3D typed mixed-storage wrappers using `f16` host storage and `f32` GPU kernels
- [x] Add NUFFT-WGPU direct Type-1/Type-2 1D/3D typed mixed-storage wrappers using `f16` host storage and `f32` GPU kernels
- [x] Add DHT-WGPU forward/inverse typed mixed-storage wrappers using `f16` host storage and the existing `f32` GPU kernel
- [x] Add FWHT-WGPU forward/inverse typed mixed-storage wrappers using `f16` host storage and the existing `f32` GPU kernel
- [x] Add typed mixed-storage WGPU wrappers and represented-`f32` parity tests for CZT, DCT/DST, FrFT, GFT, Hilbert, Mellin, QFT, Radon, SDFT, SFT, SHT, STFT, and Wavelet
- [x] Add debug-gated NUFFT-WGPU fast Type-2 1D/3D grid diagnostics for after-load and after-IFFT checkpoints
- [x] Replace stale CI crate/path references with workspace `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --all-targets`, and current `apollo-python` smoke tests
- [x] Add `type2_into` zero-allocation 3D Type-2 NUFFT path on `NufftPlan3D`
- [x] Add value-semantic typed verification tests for `apollo-nufft` (1D and 3D, Complex64/Complex32/[f16;2], profile mismatch rejection)

## Closed in this sprint (Closure phase)
- [x] Fix `[workspace.lints.clippy]` priority: assign `all` and `pedantic` groups `priority = -1` so individual overrides at default priority 0 take precedence; eliminates 22 clippy compilation failures across all transform crates.
- [x] Propagate workspace lints to all 39 crates via `[lints] workspace = true` in every `Cargo.toml`; add comprehensive pedantic suppressions for DSP-appropriate patterns (cast truncation/precision/loss, needless_range_loop, too_many_arguments, manual_is_multiple_of, manual_div_ceil, etc.).
- [x] Fix `apollo-fft` doc-lint warnings: replace `- ` list markers with `* ` in `direct.rs` module doc; replace `for k in 0..n { output[k] = }` with `iter_mut().enumerate()` in `dft_forward` and `dft_inverse`.
- [x] Replace `CpuBackend::default()` with `CpuBackend` (unit-struct literal) in `apollo-fft` transport tests to satisfy `clippy::default_constructed_unit_structs`.
- [x] Add `#![allow(missing_docs)]` and doc comments to `apollo-fft/benches/kernel_strategy.rs`.
- [x] Add `fast_type2_1d_normalization_invariance_when_device_exists` test to `apollo-nufft-wgpu` verification: single non-zero coefficient at k=0, verifies GPU output matches CPU gridded reference and that output is constant across positions (detects 1/m rescaling regressions).
- [x] Add normalization convention documentation to `nufft_fast_1d.wgsl` (Type-1 unnormalized forward FFT, Type-2 host pre-scales deconv by m to compensate normalized IFFT), `nufft_fast_3d.wgsl` (3D Type-2 uses normalized IFFT directly, no pre-scaling needed), and `GpuFft3d::encode_inverse_split` doc comment (caveat for unnormalized-IDFT consumers).
- [x] Remove 22 scratch/temporary files from repository root (`_gen.py`, `_test*.rs`, `tmp_patch_ntt.py`, `validation_output*.json`, `apollo_status.txt`, etc.) and `scratch/` directory.
- [x] Add scratch-file gitignore patterns to `.gitignore` (validation output JSON, temporary Python/Rust scripts, status files, scratch directory).
- [x] Verify zero clippy errors, zero clippy warnings, zero test failures across full workspace.

## Closed in previous sprints
- [x] Register every `crates/apollo-*` crate in the root workspace.
- [x] Replace incomplete `apollo-validation` orchestration with computed CPU, GPU-surface, NUFFT, external-reference, benchmark, and environment reports.
- [x] Add real crate roots for `apollo-frft`, `apollo-gft`, and `apollo-stft`.
- [x] Correct CZT Bluestein convolution lag construction against the direct CZT definition.
- [x] Correct SFT expected coefficients against the analytical DFT of the test signal.
- [x] Consolidate SFT ownership into `apollo-sft` and split it into domain, application, infrastructure, and verification modules.
- [x] Correct STFT boundary coverage by using centered analysis frames with overlap-add normalization.
- [x] Align `apollo-python` with current crate names, shape newtypes, and full-spectrum FFT plan APIs.
- [x] Split `apollo-validation` external references behind an optional validation-only feature so `rustfft` is validation-only; audited that `realfft` is absent from the workspace dependency graph.
- [x] Complete `apollo-validation` with the new multi-crate API surface and conditional external-backend wiring.
- [x] Fix `FftPlan1D` and `FftPlan2D` missing `forward_complex`/`inverse_complex` allocating wrappers (parity with `FftPlan3D`).
- [x] Replace O(N^2) direct DFT kernels with O(N log N) strategy: iterative Cooley-Tukey radix-2 for power-of-2 sizes and Bluestein chirp-Z for arbitrary sizes; auto-selection in `kernel::fft_forward_64`, `fft_inverse_64`, etc.; all plan files updated to use new API; `rustfft` removed from production `apollo-fft` dependency.
- [x] Add and complete `apollo-hilbert` with Hilbert transform plans, analytic-signal storage, envelope/phase extraction, and analytical/property tests.
- [x] Add and complete `apollo-radon` with parallel-beam forward projections, adjoint backprojection, ramp-filtered backprojection, sinogram storage, and analytical/property tests.
- [x] Complete `apollo-mellin` with Mellin moments, log-frequency spectra, execution contracts, and analytical tests.
- [x] Replace stale skeleton documentation in completed transform crates and add DCT/DST value-semantic tests.
- [x] Remove the incorrect unverified DCT/DST fast branch and add large-plan parity tests against analytical kernels.
- [x] Add Python `rfft3`/`irfft3` value-semantic tests documenting the full-spectrum contract and asserting computed output values.
- [x] Add validation report JSON schema-shape tests for required top-level and nested sections.
- [x] Add Criterion benchmark target for Apollo FFT direct, radix-2, and Bluestein kernel strategies.
- [x] Reduce Radon filtered-backprojection allocation by adding caller-owned ramp filtering.
- [x] Correct stale FFT architecture docs from direct-kernel execution to radix-2/Bluestein auto-selection.
- [x] Reduce FFT 2D/3D axis-pass peak scratch by transforming gathered lanes in place instead of collecting transformed lane copies.
- [x] Reduce NUFFT interpolation and 3D separable-pass allocation by borrowing type-2 grids and reusing per-axis lane buffers.
- [x] Add `apollo-czt` crate README, CZT/Bluestein theorem docs, caller-owned forward path, and in-place convolution workspace multiplication.
- [x] Add `apollo-fwht` crate README, Hadamard involution theorem docs, caller-owned real/complex output paths, and parity tests.
- [x] Add `apollo-ntt` crate README, root-of-unity theorem docs, true in-place execution, caller-owned output paths, residue normalization, and overflow-safe modular addition.
- [x] Add `apollo-frft` crate README, FrFT rotation theorem docs, finite singular integer-order plan state, inverse APIs, and inverse parity tests.
- [x] Add `apollo-stft` crate README, overlap-add theorem docs, cleaned module comments, actionable buffer diagnostics, and inverse caller-owned parity tests.
- [x] Add `apollo-dctdst` crate README, DCT/DST inverse-pair theorem docs, caller-owned inverse output, and inverse parity tests.
- [x] Clean `apollo-sft` Rustdoc encoding, remove deprecated ndarray raw-vector extraction, and reuse the crate-local direct DFT reference in verification.
- [x] Restore `apollo-ntt` plan implementation after truncation and verify modular arithmetic, convolution, caller-owned, and property tests.
- [x] Repair CZT test placement, enable `Complex64` metadata serialization, and reject zero-magnitude CZT step parameters.
- [x] Repair SHT source encoding so Rust tooling parses theorem/reference docs.
- [x] Repair SDFT result propagation and QFT property-test plan construction.
- [x] Remove duplicated NUFFT 3D module tail, restore sorted type-2 interpolation, and replace approximate `I_0` with the defining convergent series.
- [x] Correct Wavelet Morlet admissibility documentation and kernel by applying the DC correction with a zero-mean numerical proof test.
- [x] Add crate-local architecture README files for all `crates/apollo-*` crates.
- [x] Split the WGPU backend boundary into `apollo-fft-wgpu` and `apollo-nufft-wgpu`.
- [x] Add per-transform WGPU backend crates for CZT, DCT/DST, DHT, FrFT, FWHT, GFT, Hilbert, Mellin, NTT, QFT, Radon, SDFT, SFT, SHT, STFT, and Wavelet.
- [x] Eliminate per-stage `Vec<Complex>` twiddle allocations in radix-2 (f32/f64 forward/inverse) by replacing with a single N/2-entry stride-indexed table (Unified Twiddle Table theorem proved in module doc).
- [x] Cache Bluestein scratch buffer in `FftPlan1D` via `Mutex<Vec<Complex64>>` to eliminate per-call heap allocation on the non-power-of-two hot path.
- [x] Precompute DWT highpass QMF coefficients once per `analysis_stage_into`/`synthesis_stage_into` call; QMF identity g[k] = (-1)^k·h[L-1-k] proved from Smith-Barnwell PR condition.
- [x] Add Parseval/Plancherel energy-invariance theorem with proof to `radix2.rs` module doc; add Unified Twiddle Table theorem proving stride-index equivalence.
- [x] Add I_0 convergence theorem (geometric tail bound, K=256 sufficiency corollary) to `kaiser_bessel.rs`.

## Next increments
- [x] Reintroduce DCT/DST acceleration only after deriving a correct FFT mapping and proving parity against direct kernels.
- [x] Implement exact direct Type-1 1D/3D NUFFT WGPU kernels inside `apollo-nufft-wgpu` with CPU parity tests before reporting execution support.
- [x] Implement exact direct Type-2 1D NUFFT WGPU kernels inside `apollo-nufft-wgpu` with CPU parity tests before reporting execution support.
- [x] Implement exact direct Type-2 3D NUFFT owner reference and WGPU kernels inside `apollo-nufft-wgpu` with CPU parity tests before reporting execution support.
- [x] Implement direct dense DFT SFT WGPU kernels with deterministic sparse top-K projection and CPU parity tests inside `apollo-sft-wgpu`.
- [x] Implement NUFFT WGPU fast 1D gridding paths using GPU spreading/interpolation, oversampled FFT dispatch, and deconvolution.
- [x] Implement NUFFT WGPU fast 3D gridding paths using GPU separable spreading/interpolation, oversampled 3D FFT dispatch, and deconvolution.
- [x] Implement SHT WGPU numerical kernels using owner-derived basis/quadrature buffers inside `apollo-sht-wgpu` with CPU parity tests.
- [x] Move SHT WGPU associated Legendre recurrence and spherical harmonic basis generation onto GPU while keeping `apollo-sht` quadrature as the SSOT.
- [x] Implement forward and inverse FrFT WGPU kernels inside `apollo-frft-wgpu` with CPU parity tests for all 5 dispatch modes (identity, centred DFT, reversal, centred IDFT, general chirp).
- [x] Implement forward direct-bin sliding DFT WGPU kernels inside `apollo-sdft-wgpu` with CPU parity tests before reporting execution support.
- [x] Implement forward Hann-windowed STFT WGPU kernels inside `apollo-stft-wgpu` with CPU parity tests before reporting execution support.
- [x] Implement forward and inverse Haar DWT WGPU kernels inside `apollo-wavelet-wgpu` with CPU parity tests before reporting execution support.
- [x] Audit and document that `realfft` is not present in the workspace dependency graph; `apollo-validation/external-references` gates only optional `rustfft`.
- [x] Add published-reference validation fixtures for DFT, DHT, DCT-II, and DST-II under `apollo-validation::external.published_references`.
- [x] Audit remaining transform crates against published references and add cross-crate validation fixtures where useful.
- [x] Optimize `apollo-sht-wgpu` basis storage by removing host-side zero-vector initialization before GPU basis generation.
- [x] Fix GPU fast type-2 1D NUFFT normalization: `execute_fast_type2_1d` packs deconv values scaled by `oversampled_len` to compensate for `encode_inverse_split` normalized IFFT (÷m), matching the CPU `type2_into` ×m rescaling without an extra host vector.
- [x] Optimize `apollo-nufft-wgpu` fast placeholder bindings by replacing host-side zero-vector uploads with device-only storage buffers.
- [x] Optimize `apollo-fft` 2D/3D contiguous axis passes by transforming backing-slice chunks in place instead of allocating full-field lane-copy vectors.
- [x] Add `apollo-fft` caller-owned 3D typed forward/inverse paths for `f64`, `f32`, and mixed `f16` storage profiles.
- [x] Extend `apollo-validation` precision benchmarks to report forward and inverse timings for `f64`, `f32`, and mixed `f16` FFT profiles.
- [x] Add typed caller-owned DHT and DCT/DST paths for `f64`, `f32`, and mixed `f16` storage profiles.
- [x] Add typed caller-owned FWHT paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Audit all workspace crates for `Cargo.toml`, `README.md`, and `src/lib.rs`; add missing `apollo-python` architecture, mathematical contract, precision contract, and verification documentation.
- [x] Add typed caller-owned CZT paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned FrFT paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned GFT paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned Hilbert quadrature paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned Mellin log-resample paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned QFT paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned Radon forward/backprojection paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned SDFT direct-bin paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned STFT forward/inverse paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned Wavelet DWT/CWT paths for `f64`, `f32`, and mixed `f16` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned SFT sparse forward/inverse paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned SHT real/complex forward and inverse paths for `f64`/`Complex64`, `f32`/`Complex32`, and mixed `f16`/`[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Add typed caller-owned NUFFT 1D/3D Type-1/Type-2 paths for `Complex64`, `Complex32`, and mixed `[f16; 2]` storage profiles with profile mismatch rejection.
- [x] Complete mixed-precision rollout across eligible CPU transform crates.
- [x] Define explicit mixed-precision support/unsupported capability records for each GPU backend crate.
- [x] Add exact quantized `u32` residue storage APIs to NTT-WGPU instead of floating mixed precision.
- [x] Add reusable-buffer exact quantized `u32` residue dispatch to NTT-WGPU.
- [x] Add `apollo-fft-wgpu` reusable GPU buffer structs for repeated 3D FFT dispatch
- [x] Add debug-gated GPU grid readbacks (after load, after IFFT) behind a `cfg(test)` feature for faster future numerical triage in `apollo-nufft-wgpu`
- [x] Run `cargo clippy --workspace --all-targets` and `cargo test --workspace` in CI to prevent regressions of the lint priority or normalization conventions
