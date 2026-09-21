# Apollo Checklist

## claude/fable — current execution
- [ ] [Lane order inverse](backlog.md#apollo-planar-lane-order-inverse): PR enqueued; merged; PR 363 rebased and re-verified.
- [ ] [Rectangular odd powers](backlog.md#apollo-planar-rectangular-odd-powers): ADR 0060 Accepted; PR enqueued; merged, lease discharged.

- [x] [Stage sweeps](backlog.md#apollo-planar-stage-sweeps): PR 354 merged; lease discharged.
- [x] [Seam lane order](backlog.md#apollo-planar-seam-lane-order): hermes in-lane unpack landed (PR 161); plane layout, seams, transpose and combine on the lane (`3adf12b1`).
- [x] [Seam lane order](backlog.md#apollo-planar-seam-lane-order): paired measurement, ADR 0057 Accepted, board updated.
- [x] [Seam lane order](backlog.md#apollo-planar-seam-lane-order): PR 357 merged; lease discharged.

## codex/root — current execution

- [ ] [FourStep movement](backlog.md#apollo-four-step-square-movement): verify current Leto/Moirai 0.6 adoption against the accepted `3f1c0db7` evidence, then complete provider-first delivery.

## ATLAS-APOLLO-REAL-HALF-THROUGHPUT-2026-09-01 [perf] — Codex

- [x] Attribute cached-plan acquisition, pair packing, half-length FFT, and
      untangling at every census length with an exact-revision phase probe.
- [x] Retain only a production change whose mechanism addresses the measured
      bound and improves two unchanged 100-sample paired runs.
- [x] Preserve direct-DFT values, conjugate symmetry, DC/Nyquist behavior,
      f32/f64 parity, caller-owned output, and zero warmed allocation.
- [x] Pass 512/512 debug and release value tests, warning-denied host/AArch64
      gates, Rustdoc, doctest, standalone lock, formatting, docs, and exact-diff
      review. Record `cargo asm` as unavailable rather than claiming codegen.
- [ ] Commit, publish, obtain independent review, pass hosted gates, merge, and
      discharge the source and PM leases.
## ATLAS-APOLLO-MELLIN-REAL-COMPLEX-DOT-2026-09-01 [perf] — Codex

- [x] Commit the unchanged public Mellin forward-spectrum benchmark and record
      exact N = 64 control plus N = 128/256 candidate baselines.
- [x] Implement one generic Hermes real-by-interleaved-complex dot with exact
      preflight, scalar/native tails, differential values, and isolated timing.
- [x] Delete Apollo's retained real-lane materialization only if two unchanged
      complete-path comparisons improve with a neutral control; prove the 16N
      retained-byte removal and warm allocation contract.
- [x] Merge provider head `59c89431` through Hermes PR #113 as `2e993503`; pass
      exact-Git consumer Clippy, 27/27 debug and release Nextest, Rustdoc,
      doctest, benchmark smoke, and the 36-source shadow-lock check.
- [ ] Integrate the canonical Apollo lock after its peer lease discharges, pass
      exact Apollo review, and merge without squash; otherwise remove the
      candidate and retain the negative instrument.

## ATLAS-APOLLO-STFT-WINDOW-INTERLEAVE-2026-09-01 [perf] — Codex

- [x] Commit the unchanged CPU STFT complete-plan benchmark and record an exact
      pre-adoption baseline with frame-length and signal-length controls.
- [x] Reject the consumer because two adjacent complete-path comparisons do not
      separate target movement from the scalar control; remove all experimental
      source and retain the benchmark instrument only.
- [x] Record the source-equivalent value, warm-allocation, and analytical
      16,384-byte-per-active-worker evidence with its revision-attribution limit.
- [ ] Pass exact benchmark smoke, warning-denied all-target Clippy, format/diff,
      independent review, and merge the instrument without squash.

## ATLAS-APOLLO-RADER-59-VARIANCE-2026-08-29 [perf] — Codex

- [x] Reproduce the apparent 59-point variance with the retained 100-sample
      native strategy instrument and distinguish route cost from noise.
- [x] Enumerate every dynamic prime below the next non-smooth boundary and
      compare full-cyclic, half-cyclic, Bluestein, and automatic routes for
      f32 and f64 without changing the timed closure.
- [x] Retain only the common 59/83/107 route win; preserve the incumbent route
      at 167 and above where replications or precisions do not agree.
- [x] Verify f32/f64 output against independent direct DFTs and assert the
      shape-only selector boundary.
- [x] Prove warm execution allocates zero times through both Apollo's global
      allocator and direct Mnemosyne hooks.
- [x] Complete warning-denied package gates, exact-lock validation, and lease
      discharge at exact provider `0ba9c504`.
- [ ] Obtain independent artifact review, pass hosted verification, and merge.

## ATLAS-APOLLO-BASE-KERNEL-LANE-WIDTH-2026-08-29 [perf] — Codex

- [x] Reproduce the current standalone locked f32/f64 power-of-two disparity
      with the unchanged clone-inclusive comparison instrument at `c08ddf86`.
- [x] Attribute the gap to native-width code generation and implement one
      canonical eight-lane four-byte base variant without changing the f64
      route or public surface.
- [x] Prove forward/inverse value semantics, no-mutation decline, and warm
      allocation parity at 64/128/256/512 in debug and release profiles.
- [x] Run warning-denied diagnostics and inspect release code generation for
      in-loop probes, calls, spills, and scalar fallback.
- [x] Retain only a paired benchmark improvement with stable f64 controls;
      synchronize the gap audit, backlog, CHANGELOG, and exact evidence.
- [ ] Obtain independent artifact review, publish, pass hosted gates, merge,
      and discharge the lease.

## APOLLO-FFT-HEPHAESTUS-CUTOVER-2026-08-28 [major] [arch] [perf] — Codex

- [x] Delete Apollo's duplicate dense WGPU FFT implementation, shaders,
      benchmark, feature, and public exports while retaining CUDA FFT.
- [x] Move all in-repository dense WGPU consumers to prepared rank-generic
      Hephaestus plans and retain NUFFT plans, buffers, and host capacity.
- [x] Prepare and bind all four reusable NUFFT domain stages once, update only
      retained parameter buffers, and encode each operation as one grouped
      Hephaestus sequence with a retained fixed-capacity interpolation grid.
- [x] Prove value correctness when one workspace executes maximum-capacity then
      shorter logical-sample requests across 1-D/3-D Type-1/Type-2 paths.
- [x] Prove zero Apollo-owned warm allocations for host conversion, readback,
      and non-contiguous 3-D coefficient staging while preserving capacity.
- [x] Reject foreign-device reusable workspaces before every host-to-device
      write and prove the typed error leaves all four caller outputs unchanged.
- [x] Repair benchmark-mode selection and complete the bounded retained-versus-
      per-call GPU measurement without reducing statistical observations.
- [x] Correct base-128 split inverse normalization and repair the hosted book
      and package-owned benchmark-smoke gates without extending their budgets.
- [x] Pass the local feature matrix, value-semantic Nextest suites, doctests,
      warning-denied Rustdoc, residue checks, and SemVer classification.
- [x] Commit source candidate `f981908f` and documentation correction
      `7c063e27`; provider PR #234 is merged at `44754cd1`.
- [x] Obtain independent GREEN review of the exact source and documentation
      candidates.
- [ ] Deliver PR #176 through hosted gates, merge, synchronize closure
      revisions, and discharge the lease.

## ATLAS-APOLLO-BRANCH-DEBT-2026-08-27 [patch] — Codex

- [x] Remove `codex/fix-atlas-sha` after proving its Hermes 0.6 requirement is
      superseded by main's Hermes 0.7 requirement and `08306b94` lock pin.
- [x] Reconcile `fix/apollo-fft-workspace-buffers` against current main and
      either integrate its unique value with focused gates or prove it obsolete.
- [ ] Process the remaining stale branches one complete increment at a time;
      keep the board inventory and branch count exact.

Evidence: `dd0dad47` changed only `Cargo.toml` and `Cargo.lock` from Hermes 0.5
to 0.6. Main requires Hermes 0.7 and locks `08306b94`; the superseded local and
remote `codex/fix-atlas-sha` refs were deleted on 2026-08-27.

Workspace-buffer salvage merged through PR #150 as `0536c9c8`. Retained host
staging covers the four f64/half Leto GPU entry points, and Leto `fb70cb6`
initializes final Mnemosyne-backed outputs once. Code revision `14993ed9`
passed locked check, strict Clippy, 482/482 Nextest, 2/2 doctests,
warning-denied Rustdoc, provider audit, 223/223 interop patch SemVer,
196/196 FFT minor SemVer, and bounded GPU benchmark smoke. The stale local and
remote refs are deleted; their Winograd blob remains on two separately tracked
stale branches.

## ATLAS-APOLLO-BOOK-TEST-2026-08-20 [patch]

- [x] Diagnose PR #108 job 96546609469: package build passed; mdBook
      compilation failed only because included examples lacked extern crate
      declarations for staged crates.
- [x] Add the Apollo, Eunomia, and Leto declarations to the two included
      examples and repin the shared workflow to Atlas 20c9398.
- [x] Pass local formatting, mdBook build, strict links, and diff checks.
- [ ] Collect the exact-head hosted rerun, then merge and verify the default.

## ATLAS-ORPHAN-MODULES-096-APOLLO [patch]

- [x] Audit all three detector-reported Apollo orphan paths against the actual
      compiler inputs and caller graph.
- [x] Delete the unreferenced `winograd/composite/large.rs` duplicate; live
      medium-composite ownership already covers the overlapping codelets.
- [x] Classify the two `include!` inputs as detector false positives and retain
      their single-source ownership.
- [ ] Coordinate the shared Atlas detector correction for `include!` edges;
      the root script is peer-edited in this integration cycle.

Residual evidence: after the source deletion, the provider-local detector
count is expected to remain two until the shared script recognizes compiled
`include!` inputs. No generated code is retained solely to satisfy an
existence-based detector.

## Rust crate publication aliases [patch]

- [x] Bind Mnemosyne and Moirai to their collision-free registry packages.
- [x] Regenerate and verify the clean standalone lockfile.
- [x] Pass the clean-checkout focused package gates.
- [x] Keep historical benchmark sources resolvable under the candidate provider
      graph without changing their production code.
- [x] Remove sibling-source includes from the `apollo-fft` package archive.
- [ ] Pass exact locked dry runs for `apollo-fft-macros` and `apollo-fft`.
- [ ] Publish reusable crates in dependency order and verify the sparse index.
- [ ] Register crates.io Trusted Publishers for the release workflow.

## Closure XLII — Apollo vs RustFFT f32 N=4096 Performance Disparity [patch]
Sprint target version: 0.13.3

- [x] Re-run f32 N=4096 focused Criterion probes for Apollo and RustFFT.
- [x] Reject disabling the f32 N=4096 radix-16 quad suffix. Same-session
  Criterion with the quad predicate disabled measured Apollo 6.5098 µs vs
  RustFFT 3.7433 µs, so the spilled quad leaf remains faster than the fallback
  schedule.
- [x] Restore benchmark compilation after current API drift by adding the local
  RustFFT dev-dependency, registering `vs_rustfft`, repairing Winograd typed
  entry points, and routing the untracked benchmark through current mixed-radix
  precomputed-twiddle APIs.
- [x] Record current residual disparity: focused f32 N=4096 precomputed-twiddle
  median is Apollo 22.790 µs vs RustFFT 3.5969 µs. This row is not comparable
  with the prior plan-scratch row because the plan-scratch API is absent in this
  checkout.
- [x] `cargo check -p apollo-fft --benches`: passed with warnings.
- [x] `cargo test -p apollo-fft dft7 --lib -- --test-threads=1`: 5 passed.
- [x] Add `stockham` to the kernel module tree and route f32 power-of-two
  lengths >=1024 through `<f32 as StockhamKernel>::forward_with_scratch` using
  thread-local reusable scratch.
- [x] Restore Stockham test-local twiddle builders to current radix2 table
  builders and restore inverse scratch trait coverage needed by Stockham tests.
- [x] Reject the initial production `hybrid_radix8x512_32_avx_fma` dispatch
  probe for N=4096: Criterion regressed Apollo zero-alloc reused to 10.707 µs
  and caller-twiddle reused to 12.101 µs on the then-current route.
- [x] Reject direct no-argument mixed-radix micro-dispatch: Criterion measured
  Apollo zero-alloc reused 8.1406 µs vs RustFFT 6.2656 µs.
- [x] Final retained f32 N=4096 Criterion: Apollo zero-alloc reused 7.0463 µs,
  Apollo caller-twiddle reused 8.9737 µs, RustFFT reused 6.2814 µs.
- [x] Re-test f32 N=4096 Stockham suffix scheduling on the current retained
  path. Retain disabled quad suffix because longer Criterion measured Apollo
  caller-twiddle reused 6.0315 µs versus 8.9737 µs with the quad suffix.
- [x] Reject triple-only N=4096 schedule. Longer Criterion measured Apollo
  zero-alloc reused 7.6359 µs vs RustFFT 4.9184 µs.
- [x] Add single-entry thread-local f32 forward-twiddle fast cache for the
  public zero-allocation path; it borrows the cached table and avoids the
  per-call `Arc` clone on the Stockham route.
- [x] Final longer f32 N=4096 Criterion after cache/schedule changes: Apollo
  zero-alloc reused 6.3347 µs, Apollo caller-twiddle reused 6.0315 µs,
  RustFFT reused 4.2974 µs.
- [x] Reject and remove the terminal groups=1 in-place Stockham stage after
  auditing the layout contract: groups=1 reads interleaved pairs
  (`src[2j]`, `src[2j+1]`), so a direct in-place final stage overwrites future
  source elements.
- [x] Consolidate f32 Stockham public-path scratch and cached twiddle state into
  one thread-local workspace.
- [x] Add `#[inline(always)]` to the f32 public dispatch chain
  (`FftPrecision for Complex32`, `fft_forward_32`, `forward_inplace_32`,
  `forward_stockham_cached_32`).
- [x] Reject direct concrete f32 benchmark calls: Criterion measured Apollo
  public 8.9688 µs, caller-twiddle 8.0722 µs, RustFFT 6.2812 µs.
- [x] Reject shortened public branch in `fft_forward_32`: Criterion measured
  Apollo public 9.0138 µs, caller-twiddle 8.1278 µs, RustFFT 6.3773 µs.
- [x] Reject zero-copy generic scheduler flip for N=4096: it passed roundtrip
  but violated the scheduler assertion and regressed caller-twiddle to
  14.413 µs versus RustFFT 9.3559 µs on the same run.
- [x] Reject the promoted f32 8x512 N=4096 production route: with the branch
  removed, generic Stockham measured Apollo public 8.5731 µs and caller-twiddle
  7.6865 µs versus the promoted 8x512 route at 12.891 µs public and
  11.188 µs caller-twiddle.
- [x] Reject split scratch/twiddle public cache route: Criterion regressed
  Apollo public to 12.110 µs and caller-twiddle to 12.146 µs, and introduced
  dead-code warnings.
- [x] Reject contiguous-output transpose in the 8x512 helper: Criterion showed
  no caller-twiddle improvement and public noise/regression.
- [x] Test and supersede the f32 N=4096 single/pair/single copyback-free tail:
  enabling the existing radix-16 groups=8 leaf only at `(stride=256, n=4096,
  source=scratch)` is faster and still ends in `data`.
- [x] Supersede the radix-16 tail with the radix-8/radix-8 tail schedule after
  repeated Criterion showed the caller-twiddle row near 4.8-4.9 µs and public
  dispatch near the caller-twiddle row.
- [x] Add `#[inline(always)]` to `forward_inplace_32_with_twiddles` and
  `with_stockham_scratch_32`; reject `#[inline(always)]` on the target-feature
  AVX wrapper because rustc rejects the attribute combination, and reject
  forcing the f32 `StockhamKernel` trait method to `#[inline(always)]` because
  Criterion regressed the retained rows.
- [x] Replace the combined f32 Stockham scratch/twiddle workspace with split
  scratch and twiddle caches after the radix-8/radix-8 tail made the split path
  faster; remove the dead combined-workspace code and warnings.
- [x] Reject extending the f32 low-live threshold from 32 KiB to 64 KiB: focused
  Criterion did not produce a stable arithmetic-path improvement.
- [x] Reject a single-entry f32 Stockham twiddle cache separate from scratch:
  the public-path result was not stable and the caller-twiddle row did not
  improve.
- [x] Reject direct N=4096 four-pass specialization and unchecked twiddle
  subslices: the direct route did not hold up in repeat measurement and the
  unchecked subslice variant regressed both Apollo rows.
- [x] Reject stride-64 radix-16 fusion for f32 N=4096: correctness held, but
  focused Criterion regressed Apollo public zero-alloc reused to 9.7711 µs and
  caller-twiddle reused to 9.3225 µs versus RustFFT 3.7232 µs.
- [x] Reject forced Stockham monomorphization annotations at this boundary:
  rustc rejects `#[inline(always)]` on `#[target_feature]` functions, and the
  valid trait/cache inlining probes did not retain a repeatable improvement
  after focused Criterion measurement.
- [x] Reject paired 128-bit stores in
  `stage_triple32_quarter_groups_one_avx_fma`: the reduced store count added
  shuffles and regressed Apollo public zero-alloc reused to 7.1908 µs and
  caller-twiddle reused to 6.1711 µs versus RustFFT 3.8321 µs.
- [x] Reject even-radix tail monomorphization for the same suffix: first run
  reached Apollo caller-twiddle 5.3101 µs, but repeat moved to 5.6971 µs and
  did not establish a retained improvement.
- [x] Reject const-generic radix-1 quarter-turn signs: correctness held, but
  focused Criterion regressed Apollo public zero-alloc reused to 8.1940 µs and
  did not produce a statistically significant caller-twiddle gain.
- [x] Generate release assembly with
  `cargo rustc -p apollo-fft --release --lib -- --emit=asm` and identify the
  f32 Stockham codelet call-boundary cost: the default Windows ABI emits
  XMM6-XMM15 save/restore in the separate quarter-groups-one suffix.
- [x] Reject private raw-pointer `sysv64` ABI for the f32 radix-1 and
  quarter-groups-one codelets: assembly improved the suffix prologue, but
  focused Criterion did not retain a repeatable kernel-row improvement and the
  probe was reverted.
- [x] Audit GhostCell fit for the retained f32 N=4096 path: no graph or shared
  interior-mutability topology exists in the hot route; scratch is thread-local
  and lexically borrowed, so GhostCell would add no performance contract here.
- [x] Add SWAR-adjacent scalar cleanup for non-Stockham routes: replace
  division/modulo in shared power-of-two digit reversal with shift/mask digit
  extraction.
- [x] Benchmark affected f32 N=256 radix-4 route after shift/mask permutation:
  repeat Criterion measured Apollo public 983.67 ns, Apollo caller-twiddle
  991.61 ns, and RustFFT 137.65 ns. The change is correctness-preserving and
  neutral; the N=256 gap remains in radix-4 butterflies/scheduling.
- [x] Expand f32 forward Stockham/autosort dispatch from lengths >=1024 to
  lengths >=256. This routes N=256 through caller-scratch Stockham and removes
  the radix-4 digit-reversal route for that size.
- [x] Benchmark retained N=256 autosort expansion: focused Criterion repeat
  measured Apollo public 197.50 ns, Apollo caller-twiddle 218.36 ns, and
  RustFFT 113.96 ns.
- [x] Reject N=64 autosort expansion: threshold 64 measured Apollo public
  64.969 ns and caller-twiddle 45.621 ns, while the public row regressed and
  caller-twiddle was neutral. Restore threshold to 256.
- [x] Add f32 inverse zero-allocation rows to `vs_rustfft` so inverse Stockham
  integration has a Criterion gate.
- [x] Route f32 inverse power-of-two lengths >=256 through Stockham with inverse
  twiddles; normalized inverse uses the unnormalized Stockham route followed by
  explicit `1/N` scaling.
- [x] Verify f32 Stockham forward+normalized-inverse roundtrip at N=256.
- [x] Benchmark inverse route against old digit-reversal baseline: old inverse
  path measured Apollo 963.10 ns at N=256 and 23.104 µs at N=4096; retained
  Stockham inverse measured Apollo 230.60 ns at N=256 and 5.5408 µs at N=4096.
- [x] Final current-tree Criterion after rejected probes were reverted: Apollo
  public zero-alloc reused 5.4298 µs, Apollo caller-twiddle reused 5.2661 µs,
  RustFFT reused 3.6958 µs. Earlier same-state best retained run measured
  Apollo public 4.8645 µs and caller-twiddle 4.7913 µs; the spread is recorded
  as benchmark variance, not a new retained optimization.
- [x] Current retained run after the latest rejected probes were reverted:
  Apollo public zero-alloc reused 5.4895 µs, Apollo caller-twiddle reused
  5.4176 µs, RustFFT reused 4.3328 µs.
- [x] Reject static N=4096 f32 twiddle specialization: Criterion regressed
  Apollo public zero-alloc reused to 5.4357 µs and caller-twiddle reused to
  5.7335 µs.
- [x] Add f64 Stockham/autosort dispatch for forward and inverse power-of-two
  lengths >=256, reusing thread-local scratch and inverse twiddles for
  unnormalized inverse.
- [x] Add f64 inverse zero-allocation rows to `vs_rustfft`.
- [x] Verify f64 Stockham forward+normalized-inverse roundtrip at N=256.
- [x] Benchmark f64 Stockham against the prior digit-reversal baseline:
  retained Stockham measured N=256 forward 315.24 ns and inverse 257.88 ns
  versus old 830.23 ns and 778.38 ns; N=4096 forward 10.050 µs and inverse
  10.731 µs versus old 25.456 µs and 32.167 µs.
- [x] Reject f64 N=64 autosort expansion: threshold 64 measured Apollo public
  82.748 ns and caller-twiddle 92.935 ns, a Criterion regression versus the
  existing radix route, so the f64 threshold remains 256.
- [x] Remove production dispatch to the f64 N=256/N=512 fixed single-pass
  kernels so those sizes use the fused generic AVX scheduler. Focused Criterion
  improved f64 N=256 to Apollo public 255.90 ns, caller-twiddle 228.16 ns,
  and inverse 225.37 ns; f64 N=512 measured Apollo public 591.36 ns and
  caller-twiddle 581.33 ns.
- [x] Remove production dispatch to the f32 N=512 fixed single-pass kernel so
  N=512 uses the fused generic AVX scheduler. Focused Criterion measured Apollo
  public 366.39 ns, caller-twiddle 346.71 ns, inverse 328.85 ns, RustFFT
  forward 329.96 ns, and RustFFT inverse 356.70 ns.
- [x] Keep old f64/f32 N=512 fixed kernels test-only for hybrid-radix probe
  equivalence; delete the now-unused f64 N=256 fixed kernel.
- [x] Add f32 and f64 N=512 mixed-radix forward+normalized-inverse roundtrip
  tests for the retained fused-scheduler route.
- [x] Add a static f32 N=4096 four-triple Stockham schedule that directly
  invokes the retained radix-8 fused stages at strides 1, 8, 64, and 512.
- [x] Verify the static f32 N=4096 route with a public forward+normalized-
  inverse roundtrip test using tolerance `8*N*f32::EPSILON`.
- [x] Benchmark static f32 N=4096 route: Apollo caller-twiddle forward improved
  to 5.4670 µs and inverse to 5.1970 µs; RustFFT still measured 3.7807 µs
  forward and 3.7765 µs inverse on that run.
- [x] Reject static f64 N=4096 schedule because focused Criterion regressed
  Apollo caller-twiddle forward to 11.264 µs.
- [x] Reject f32 N=512 no-copy tail schedule because focused Criterion
  regressed Apollo caller-twiddle forward to 440.90 ns and inverse to
  570.83 ns.
- [x] Reject production f32 8x512 row-Stockham decomposition. It preserved
  N=4096 correctness but regressed Criterion to 11.792 µs forward and
  11.786 µs inverse.
- [x] Reject contiguous-store transpose variant of the f32 8x512 row-Stockham
  decomposition. It improved the failed probe to 9.9378 µs forward and
  9.9228 µs inverse but remained slower than the retained four-triple schedule.
- [x] Implement f32 Butterfly512-style 8x64 production candidate with radix-8
  column pass, mixed twiddles, fixed 64-point row butterflies, and transpose.
- [x] Reject f32 Butterfly512-style 8x64 production candidate: correctness held
  but Criterion regressed N=512 forward to 546.25 ns and inverse to 573.94 ns.
- [x] Reject vectorized mixed-twiddle variant of the f32 Butterfly512 candidate
  because forward regressed further to 773.36 ns.
- [x] Audit the RustFFT `Butterfly512Avx` pathway instead of treating the prior
  8x64 candidate as the complete design. The required base-kernel contract is
  16 column rows by 32 columns, 120 f32 packed mixed-twiddle vectors, fused
  twiddle+4x4 transpose chunks, then 32-point row butterflies.
- [x] Add executable f32/f64 packed Butterfly512 twiddle-layout tests in
  `stockham.rs`. These pin Apollo's next fused kernel to the separated-column
  contract before production dispatch changes.
- [x] Benchmark current open zero-allocation rows for f32/f64 N=256/N=512/N=4096.
  Current repeated f32 N=4096 forward is Apollo 9.4509 µs versus RustFFT
  6.3698 µs; f64 N=4096 forward baseline was Apollo 17.686 µs versus RustFFT
  12.225 µs.
- [x] Reject restoring production f32/f64 N=512 fixed single-pass leaves:
  focused Criterion regressed f64 N=512 forward/inverse to 1.4856 µs /
  1.3834 µs and f32 N=512 forward/inverse to 685.78 ns / 683.37 ns.
- [x] Retain f64 N=4096 forward-only static four-triple schedule selected by
  the mathematically defined twiddle sign. It improved forward from the current
  17.686 µs baseline to 15.844 µs; inverse remains on the generic schedule
  because the static route regressed inverse.
- [x] Remove per-row `Vec<Complex64>` allocation from 3D R2C/C2R Z-axis
  split passes by reusing caller-owned half-spectrum rows and mutable C2R
  scratch rows.
- [x] Remove unused f32 R2C/C2R future-reservation plan fields and their
  `Arc`/twiddle/scratch allocations from `FftPlan3D`.
- [x] Verify retained 3D R2C/C2R memory cleanup: `cargo test -p apollo-fft r2c
  --lib -- --test-threads=1` passed 7/7.
- [x] Reject closure-borrowed thread-local twiddle-cache probe: focused f32
  N=4096 public zero-allocation Criterion regressed to 8.4200 µs median.
- [x] Restore retained twiddle-cache route and re-run focused f32 N=4096 public
  zero-allocation Criterion: 7.0245 µs median in this session.
- [x] Remove unreachable `Vec<Vec<Complex64>>` and `Vec<Vec<Complex32>>`
  fallback materialization from 2D FFT axis dispatch.
- [x] Verify 2D FFT after fallback removal:
  `cargo test -p apollo-fft dimension_2d --lib -- --test-threads=1`.
- [x] Correct generic DFT-8 forward/inverse twiddle signs in the monomorphized
  Winograd helper used by composite-radix stages.
- [x] Verify the exposed composite-radix correction:
  `cargo test -p apollo-fft dft8 --lib -- --test-threads=1`;
  `cargo test -p apollo-fft composite --lib -- --test-threads=1`;
  `cargo test -p apollo-fft --lib -- --test-threads=1`.
- [x] Remove deprecated FFT forwarding aliases:
  `FftPlan1D/2D/3D::{forward_into,inverse_into}` and `ProcessorFft3d`.
- [x] Update Python wrappers to use canonical caller-owned 3D FFT APIs.
- [x] Verify alias cleanup:
  `cargo check -p apollo-fft --benches`;
  `cargo check -p apollo-python`;
  `cargo test -p apollo-fft --lib -- --test-threads=1`;
  `rg -n "Compatibility alias|ProcessorFft3d|forward_into\(|inverse_into\(|deprecated|Deprecated|#\[deprecated\]|allow\(dead_code\)|dead_code" crates/apollo-fft crates/apollo-python --glob '*.rs'`
  returned no matches.
- [ ] Surpass RustFFT in every zero-allocation benchmark row. Latest focused
  matrix still shows open gaps at f64 N=256, f64 N=4096, f32 N=512, and
  f32 N=4096.
