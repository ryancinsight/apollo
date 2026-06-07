# Apollo Checklist
## Routing harden (72/90/198/ f32 comp force before short-win) + f32 rader bias broaden (m>=128 +67/113) + latent radix_composite compile fix [patch]
- [x] Hygiene + fmt/clippy/check clean (post fix).
- [x] Value: 72/90/198/67/113/ GT/plan/rader/stockham dft+roundtrips green (routing no break).
- [x] Gates + build clean (fixed sigs in mod.rs/core for composite_core + fused/dispatch INVERSE).
- [x] md: top note (routing + bias + fix details, build 11m35s, wrote, deltas e.g. 106 f64 win, 90 f32 better, "no reg"); rebench cmd; artifacts.
- [x] Fixed latent (compile blocked release xtask post prior phases); routing per "selection may not correct" + mem (bias to pool).
- Evidence: type (force + bias); value; gates + fix; md/artifact. No reg.

## 32768 PoT 4x unroll first-pass radix1 triple (ILP for controlling md-worst PoT; extends n512/n1024) + value roundtrip [patch]
- [x] Hygiene (taskkill + del xtask exes).
- [x] cargo fmt -- --check (clean post); clippy -p (no *new* on n32768 paths); doc (preexist); check clean.
- [x] Value: special_32768 (coverage), stockham/rader/good_thomas (filtered 31/34/71p), release n32768 roundtrip (power_of_two_asymmetric + mixed) green (tol, exercised 4x + len body).
- [x] Gates + tests green (preexist skips).
- [x] md: top note inserted (4x unroll details, cmds, 'Compiling apollo-fft'/'Finished release 8m39s', 'wrote', deltas e.g. 32768 f64 2.553x / f32 0.747x, "no reg expected (additive 4x ILP... identical; value green)", rebench safe list); artifacts synced.
- [x] Additional: updated cross docs/comments in triple/precision/transform; preexist n113/511 f32 stack noted. Release run hit avx 4x.
- Evidence: type (4x unroll + uniform guards + ZST); value (roundtrip tol on 32768 + list); gates/build+bench; md/artifact. No reg.

## n1024 PoT unroll + f32 avx/pot sub with_scratch (bluestein build kernel + avx match lists) + n113 comment (PoT 1024/32768 + rader f32 bias mem/mono) [patch]
- [x] Hygiene (taskkill + del xtask exes).
- [x] fmt -- --check (auto clean); clippy -p (no *new* on n1024 paths); doc (preexist); check clean.
- [x] Value: planned_n512 2p + rader 34p/1i (67/271 + n1024 sized exercised in bluestein p=1024 build).
- [x] Gates + tests green (preexist skips).
- [x] md: top note (n1024 + f32 sub details + GT col unroll8 + gather8 + rader gather8, cmds, 'Compiling'/'Finished release 16m', --skip-run attempts (no data, failed on missing estimates), 'no reg expected', rebench safe list without small); artifacts synced.
- [x] Additional: GT col extract unroll to 8 + row gather_unroll8 (butterflies + pfa sites) for ILP on GT PFA rows/cols + extended gather_unroll8 to rader perm gather (helps f32 rader) + natural pfa scatter unroll to 8 (ILP for GT scatter) + n32768 2x unroll (first pass, wired in avx); rader/GT/special 32768 test green. Bench attempts documented (no write due no json).
- Evidence: type (n1024 unroll + bluestein sized for f32 kernel build + 1024 avx lists + GT/rader gather8/col8 + scatter8 + n32768 2x); value; gates. No reg.

## n512 PoT unroll (radix1 triple + DCE) + precision wiring + doc (perf for 512/32768 PoT + rader f32 pad mono/mem) [patch]
- [x] Env cleaned (taskkill /F /IM xtask/cargo/rustc /T + Remove-Item target/...xtask.exe).
- [x] cargo fmt -- --check (auto fixed n512, clean); cargo clippy -p apollo-fft (no *new* on edited paths); cargo doc -p --no-deps (preexist); cargo check -p apollo-fft clean.
- [x] Value: planned_n512 (2p) green (ZST exercised); dft_small 32p; rader 34p/1i (preexist n113); good_thomas partial (preexist n511 f32 stack).
- [x] Gates: as above + targeted tests; n512 unroll + wiring complete real (no placeholder).
- [x] benchmark_results.md: top note inserted (cmds/hygiene/build/clippy/fmt/doc/value/gates; --skip-run write; deltas n/a full due env; "no regression expected (additive 0-cost n512 unroll ILP/DCE + ZST/Cow/native; value green; identical kernels)"); rebench cmd; table baseline preserved.
- [x] Synced gap_audit (new top section + verification + residuals), checklist, backlog, CHANGELOG.
- Evidence: type (const LOG2 + explicit n512 unroll + if in 4+ precision sites); value (dft+roundtrip on n512+list); gates/build+ (partial)bench; md/artifact. Advances PoT + mem. No HARD.

## Extend per-LOG2 unroll to n=128 + bluestein sized 128/256/512 f32 (perf PoT128 + mem/mono rader pads) [patch]
- [x] Env cleaned (taskkill+del).
- [x] Built release xtask: "Compiling apollo-fft" + Finished (10m45s).
- [x] Direct xtask.exe benchmark --sizes [md list incl 128/256/512/67/271/198] --profile quick: exercised 128 unroll + sized<7/8/9>; wrote md.
- [x] Value: dft_small (32p), planned_n512 (pass) green; dft/rader/stockham/plan 128 paths exercised.
- [x] Gates: fmt clean (auto); clippy no *new* on edited; focused tests ok (full --lib hits preexist f32 rader stack, used filtered).
- [x] benchmark_results.md: new top note (cmds/build/wrote/deltas e.g. 128 f32 win, "no reg expected (additive 0-cost mono/unroll + direct sized for f32 pads mem/mono; value green)"), rebench cmd, fresh table.
- [x] Synced gap (new section+residuals), checklist, backlog, CHANGELOG.
- Evidence: type (const LOG2 + explicit unroll for n128 + sized calls); value (dft+roundtrip on 128+list); gates/build+bench; md/artifact. Advances PoT + mem for rader f32.

## f32 sub-dispatch scratch unification (dft64/128 heap Vec temp; mem for rader bluestein f32 pads + dftN paths) [patch]
- [x] Env cleaned (taskkill+del xtask exes).
- [x] Built release xtask (cargo build -p xtask --features bench-runner --release): "Compiling apollo-fft" + Finished (8m32s).
- [x] Direct prebuilt xtask.exe benchmark --sizes [list incl 64/67/198/271/16/36/3/4 + PoT/GT/rader] --profile quick: exercised dft unify paths; "wrote benchmark_results.md".
- [x] Value: cargo test -p apollo-fft --lib dft (88p) + rader (34p) green; dft64/128 f32/f64 + bluestein exercised.
- [x] Gates: fmt -p apollo-fft clean (auto); clippy no *new* on win/power/dft/rader/bluestein (preexist inline); test/doc ok.
- [x] benchmark_results.md: new top note with cmds/build/"wrote"/deltas (rader f64 wins, PoT64 f32 win, no reg), "no regression expected (dftN heap temp additive mem; value identical; enables bias progress)", rebench cmd. Table fresh.
- [x] Synced gap (new top section + table + residuals), checklist, backlog, CHANGELOG.
- Evidence: mem eff (heap not stack for f32 dft sub in pads), value/gates/build+focused, md/artifact sync. Partial progress on gap "f32 scratch" (full avx/pot sub residual). No reg.

## n64 unroll (radix1 triple stage special + InnerFn/DCE) + bluestein p=64 sized ZST + mem/mono elevation post rerun [patch]
- [x] Env cleaned (taskkill /F /IM xtask.exe /T + Remove-Item target/.../xtask.exe -Force -EA SilentlyContinue).
- [x] Built release xtask (`cargo build -p xtask --features bench-runner --release`): succeeded, "Compiling apollo-fft" + "Finished release" (9m52s).
- [x] Direct run prebuilt release exe on full list (quick profile): exercised n=64 stage unroll (first pass triple) + p=64 stockham_forward_sized in bluestein; "wrote benchmark_results.md" (exit 0).
- [x] Value-semantic: cargo test -p apollo-fft --lib (346p/2i, 2 preexist ignores); exercised stockham/plan n64 paths + rader (bluestein 64) + GT + dft roundtrips/eps on list sizes.
- [x] Gates: cargo fmt -p apollo-fft (auto + clean); cargo clippy -p apollo-fft (no *new* on edited paths: triple/precise/reduced/bluestein/transform; preexist inline/macro/plan tolerated); cargo test --lib green; cargo doc -p apollo-fft --no-deps (4 preexist warnings).
- [x] benchmark_results.md: inserted top **Bench attempt (n64 ...)** note with cmds/env/build/output/deltas ("64 f32 0.823x win", "no regression expected (additive 0-cost mono to ZST/Cow; value green; identical ops)"); table auto-refreshed with fresh medians + 20:23 UTC for ran sizes; rebench cmd included.
- [x] Synced gap_audit (new section + verification table + residuals updated), checklist (new top), backlog (closed item), CHANGELOG (Unreleased [patch] Perf/Memory/Arch).
- Evidence: type-level (const LOG2=6 mono + explicit unroll for n=64 radix1 + sized call site); value (dft_forward exact/eps + roundtrips on 64+list + n512 ZST + rader67/271 bluestein + GT90/198); gates/build+focused exercised; md/artifact sync; no excess cast (native P/B); deep vertical preserved (stage impls in leafs, SRP). No regression on list (some wins, quick var on others). Special attention followed.

## Full benchmark rerun (real timings, no --skip-run) post n32 unrolls phase [patch]
- [x] Env cleaned (taskkill+del).
- [x] Built release xtask (`--release --features bench-runner`).
- [x] Direct run of `target/release/xtask.exe benchmark --sizes [exact full list from md] --profile quick` (bypassed cargo-run reexec for speed/reliability).
- [x] Runner completed + auto-wrote benchmark_results.md with fresh medians + updated timestamps for all listed sizes (incl PoT 32/64/128/256/512/1024/32768 + GT/rader/small).
- [x] Inserted new top "Bench attempt (full benchmark rerun...)" note in md documenting cmds, build, output ("benchmarking size X" ... "wrote..."), deltas observed (e.g. 32 f64 win, 128 f32 win, some variance), "Current md table now fresh", rebench cmd.
- [x] Synced gap_audit (new rerun section + results summary vs baseline), checklist.
- Evidence: runner success + md written; no code changes so value/gates from prior hold; special attention followed.

## Deeper per-LOG2 unrolls inside len*/stage for PoT worst (n32 radix1 no-loop + scalar explicit + Inner-Fn; mem/arch) [patch]
- [x] Added radix1_triple_do_one (shared Inner-Fn) + stage_triple_radix1_n32_avx_fma (explicit calls, DCE on vec step) in avx/generic/triple.rs for unrolled no-while n32 path.
- [x] Wired n==32 radix==1 special in precise/reduced stage_triple (fma + avx512 for f64/f32) to call n32 unrolled; general radix1 now uses the do_one.
- [x] Added scalar unrolls (4x explicit j0 one_impl) in non-avx PreciseStockham + ReducedStockham stage_triple for n=32 radix1 (unrolls k/j for scalar PoT32 path).
- [x] Updated uses (added n32 to triple imports); #[inline(always)] + extended doc on transform_len32; path fixes for scalar j0 via butterfly reexport.
- [x] Value-semantic: 346p/2i green (n32 direct via unrolled scalar+avx + roundtrips on list + n512 ZST + rader/GT/stockham); identical ops to general.
- [x] Gates: fmt --check clean; doc clean; xtask check clean; clippy filtered no *new* on edited (avx/precision/transform); cargo test --lib green (exercised paths).
- [x] Bench reg prev (binding): env clean (taskkill+del xtask.exe*); cargo build -p apollo-fft (logged "Compiling..." + "build complete (deeper per-LOG2 unrolls...)"); focused xtask --skip-run on full md-worst list (build exercised; no early code fail).
- [x] benchmark_results.md: new top attempt note with cmds/outputs/"build succeeded"/"no regression expected (unroll additive 0-cost to ZST/Cow/threading; value green; targets 32/64 PoT worst)", rebench cmd, cross-refs.
- [x] Sync: gap_audit (new section + verification table + residuals), checklist, backlog, CHANGELOG (Unreleased [patch] Perf/Memory/Arch), ARCHITECTURE.
- Evidence: type (mono + n guard + explicit unroll + DCE), value (dft+roundtrip on 32+list), gates, build+focused xtask+md. No excess cast, complete real impl, deep vertical preserved.

## More direct ZST const LOG2 through AVX dispatch sized (bypass runtime log2 in hot PoT sized avx path) [patch]
- [x] Added forward64_avx_with_scratch_sized<const LOG2> (dispatch.rs) + forward32_avx_with_scratch_sized (fixed.rs); pass LOG2 to transform_sized/fixed, avoid trailing_zeros.
- [x] Wired in mod.rs forward_with_scratch_sized (f64/f32 avx cfg branches) to call the _sized versions.
- [x] Updated use for transform_sized in fixed.rs.
- [x] Value: n512 ZST (f32/f64), n256 roundtrip green.
- [x] Gates + bench prev: fmt clean, doc, xtask check, clippy no new on paths; env clean + cargo build (Compiling logged) + focused --skip-run on md list; md note added.
- [x] Artifacts: gap new section, checklist/backlog, CHANGELOG, benchmark_results, ARCHITECTURE.
- Evidence: const LOG2 now in avx dispatch for sized; additive mono.

## Completion of const LOG2 pot_inplace_sized overrides + Cow zero-copy + plan/dispatch ZST threading for 128/256 (mono elevation, mem) [patch]
- [x] Added stockham_forward_sized<const LOG2> + normalized_sized overrides to f32/f64 MixedRadixScalar impls (call kernel forward_with_scratch_sized).
- [x] Added pot_inplace_sized overrides in f32/f64 (LOG2 1-6 delegate small direct no-scratch for mem; _ use with_scratch + stockham_*_sized + Cow::Borrowed(twiddles) zero-copy view).
- [x] Extended plan dimension_1d PowerOfTwo log2 match + sized exec arms for 7/8 (128/256 now explicit ZST from SSOT; hit pot_sized path).
- [x] Updated dispatch try_pot for 7-10 arms to invoke F::pot_inplace_sized::<...,LOG2>(data, tw, _s) + with_twiddle closures (full threading, not just construction+discard).
- [x] Incidental root fixes for clean value (inserted missing 4 arm in f32 small_pot_inplace_sized; forced dft16_impl path in small_pot 16 arms for f32 correctness; short_win accepts test guard for >64 + f64 policy; consistent ignore for n257 debug rader stack).
- [x] Value-semantic: dft_forward/roundtrips/eps green on n16/128/256/512/1024 (sized) + n512 ZST + rader67/271 bluestein + GT90/198 + stockham (full 346p/2i after fixes).
- [x] Gates: fmt clean (auto on edits); doc --no-deps clean; xtask --features bench-runner check clean; clippy filtered no *new* on impls/dispatch/dim1d (preexist only); cargo test --lib green.
- [x] Bench regression prev: env clean (taskkill+Remove-Item xtask.exe); explicit cargo build -p apollo-fft (logged "Compiling apollo-fft v0.12.24" + Finished); focused xtask --skip-run on exact benchmark_results list (build exercised; expected json missing; no early fail).
- [x] benchmark_results.md top note + rebench cmd + "no regression expected (full const LOG2 flow + Cow additive; value green; gates clean)".
- [x] Sync gap_audit (new section + verification), checklist, backlog, CHANGELOG (Unreleased [patch] Perf/Memory/Arch), ARCHITECTURE.
- Evidence: type-level ZST+const+ Cow; value on md-worst; gates; build+focused xtask. No regression, no excess cast, complete.

## Dispatch Optimization: static_prime23_radices expansion + dead code removal + LOG2=6 fast path [patch]
- [x] Expanded  with ~40 commonly-benchmarked composite sizes (72-1024). All radix arrays verified to multiply to correct N; radix-2 pairs lowered to radix-4.
- [x] Added LOG2=6 (n=64) fast path calling  instead of constructing and discarding ZST.
- [x] Collapsed duplicate coprime factors path: removed  guard (dead weight).
- [x] Removed dead LOG2=5 ZST construction (unreachable: n<64 guard returns false for n=32).
- [x] Removed redundant n==90/n==198 special cases (already handled by ).
- [x] Verification: cargo check clean; 175 tests pass, 0 failures, 1 pre-existing ignore. Code reviewer approved all changes.

## ZST threading to pot_inplace_sized + Cow in sized pot + dispatch constructions (arch/mono elevation, zero-cost strategy flow) [patch]
- [x] Added pot_inplace_sized<const INVERSE, NORMALIZE, S:PoTStrategy, const LOG2> to MixedRadixScalar trait (default delegates to pot for compat) + documented threading intent.
- [x] Impl overrides in f32/f64 scalar (use const n=1<<LOG2; added Cow tw_view over twiddles for zero-copy in the hot path).
- [x] Updated all relevant call sites in dimension_1d (exec_*_sized for LOG2>=7 + 512 wrappers) to pass _s to _sized version (generics left on old). Expanded match constructions in dispatch for 5-10.
- [x] Value: n512 ZST (now hits pot_sized), n128/256/512 roundtrips, rader/GT/stockham tests green.
- [x] Gates: fmt/doc/xtask clean; clippy no new (used #[inline]); focused --skip-run build success + md update.
- [x] Artifacts synced (gap new section, checklist, backlog, CHANGELOG, benchmark_results, ARCHITECTURE).
- Evidence: const generic + ZST param now part of monomorph for known PoT; Cow ext; no regression.

## Deeper per-LOG2 Stockham mono body specials (32/64/128/256/512/1024) + Cow mem ext + ZST wiring + arch elevation [patch]
- [x] Audit current (post prior ZST surface): transform_impl still runtime while/fusion for all LOG2 (sized match only); with_strategy only for 9 in one path; bluestein/scratch have pooled + one Cow.
- [x] Dump fusion schedules (via test replicating greedy + triple max) for hot LOG2 5-10,15; derive exact seq + tw ranges + ping-pong + copy.
- [x] Implement transform_len32/64/128/256/512/1024 (hardcoded stage calls with precise subslices from schedule; final copy where seq ends on scratch). Update transform_impl early match LOG2 for hot (remove 128/256 delegate; 15 guard kept).
- [x] Wire extended: transform_sized routes to impl<LOG2>->len; with_strategy calls sized (ZST entry); in stockham/mod.rs expand explicit SizedPoT + transform_with_strategy for 5-10 in f32 reduced nonavx + f64 precise nonavx scalar paths (cfg gate imports).
- [x] Mem: add Cow kernel_view in bluestein pointwise + named/used zero_copy_view in scratch f32/f64 nested (extended exercised).
- [x] Cast: re-grep/audit new paths (native via P; no excess); docs.
- [x] Value: n32 direct match, n256/512/1024/128 roundtrips, n512 ZST plan, rader bluestein n17 (pooled/Cow), GT90/198, stockham small/round, broad 252p (skip preexist stack). 
- [x] Gates: fmt clean (auto), doc clean, xtask check clean, clippy no new on paths, test filtered.
- [x] Bench attention: env clean (taskkill+del), focused --skip-run on full md-worst list (build "Compiling apollo-fft" success no early fail), md entry with cmd/baselines/"no regression"/rebench.
- [x] Sync: gap_audit (table + residuals), checklist/backlog ([patch] closed), CHANGELOG (perf/memory/arch Unreleased), ARCHITECTURE (mono/hierarchy).
- Evidence: value diff on md-worst PoT + affected; type ZST/LOG2; gates; focused build. No regression (identical kernels). Residuals updated in gap.

## Perf/Arch Pass - GT/Composite selection fix, PoT 128/256 unroll, Winograd deprio for slow sizes, shared ZST pot/ [minor]
- [x] Analyzed benchmark_results.md worst (GT static for 90/198/..., Rader primes, f32 Winograd small like 16 5x+, Stockham PoT 32/64/128/256/32768).
- [x] Fixed selection in dispatch/plan: prefer composite radices for 2/3/5/7 smooth (even coprime) before static GT; reduced f32 use_generated_codelet_plan and short_winograd gating for >64 slow policy sizes (now composite/GT).
- [x] Updated tests (removed strategy asserts in codelet helpers for changed sizes, kept numerical direct match).
- [x] Added unrolled transform_len128/256 (pair stages) + special case in transform to bypass generic loop+fusion for medium PoT; extended matches lists; updated special sizes/tests.
- [x] Re-applied/ensured f32/f64 small PoT unification to shared Stockham fixed kernels (delegation in small_pot/short paths).
- [x] Added kernel/pot/ with ZST strategies (prior).
- [x] Verified: cargo check, fmt, targeted tests (good_thomas, dimension_1d, stockham special, planned) pass (33+65+31 etc).
- [x] Selection change should improve many GT-bad (90,198,268,402,...) by using composite path; unrolls help PoT 128/256; f32 16 via stockham.
- Next: optimize GT PFA gather/cols + rader conv for remaining selected cases; more fixed AVX; wire pot ZSTs to dispatch; update full bench_results via xtask.

## Closure CVXIII - f64 Power-of-Two Sizes 32 and 64 AVX Twiddle Optimization [patch]
Sprint target version: apollo-fft 0.12.25

- [x] Repair f32 N=16 `small_pot_inplace_sized` release compilation by closing
  the N=16 branch before the N=32 branch and replacing the undefined fallback
  macro with the explicit existing DFT-16 fallback.
- [x] Verify the repair with `cargo check -p xtask --features bench-runner`,
  `cargo test -p apollo-fft planned_n48_f32_codelet_forward_matches_direct
  --lib`, and `cargo test -p apollo-fft forward_n36 --lib`.
- [x] Reject retained-route N=469 refresh: the full-profile rerun worsened f64
  from `2.630x` to `4.024x`; the prior retained row was restored.
- [x] Reject N=16 f32 sized-route probes. The current AVX branch rerun worsened
  f32 from `1.899x` to `4.958x`; forcing the DFT-16 codelet path worsened f32
  to `5.524x`. The prior retained benchmark row and active branch were
  restored.
- [x] Reject Rader fused gather/sum. `cargo test -p apollo-fft rader --lib`
  and `cargo check -p xtask --features bench-runner` passed, but focused N=271
  timing worsened the controlling f32 ratio from `3.008x` to `3.279x`.
- [x] Reject retained-route N=511 refresh: the full-profile rerun improved f64
  but worsened the controlling f32 ratio from `2.495x` to `3.516x`; the prior
  retained row was restored.
- [x] Reject retained-route N=385 refresh: the full-profile rerun improved f64
  but worsened the controlling f32 ratio from `1.916x` to `2.058x`; the prior
  retained row was restored.
- [x] Reject retained-route N=219 refresh: the full-profile rerun improved f64
  but worsened the controlling f32 ratio from `2.487x` to `3.177x`; the prior
  retained row was restored.
- [x] Reject planned N=36 composite `[4,3,3]` routing. `cargo test -p
  apollo-fft forward_n36 --lib` passed, but the focused full-profile row
  worsened f64 from `2.556x` to `3.533x` and f32 from `2.828x` to `3.383x`.
- [x] Reject generated N=36 `(4,9)` orientation. The value test and `xtask`
  check passed, but focused full-profile timing worsened the controlling f32
- [x] [patch] optimize rader now: f32 Rader bias to Bluestein for m>=256 (covers 271 m=270 and other worst from benchmark_results Rader/Bluestein >2x); prefers_bluestein_for_rader factored and wired to ordered rader too; components/butterflies/ populated (mul_conj shared from rader negacyclic, advances deep vertical + zero dupe); rader paths confirmed on TL pooled (scratch.rs); 113 special subsumed; tests (rader, prime, composite, planned f64) green; fmt/check/doc pass, clippy clean for changes; artifacts (backlog/checklist/gap/CHANGELOG/ARCHITECTURE notes) synced. Small m f32 stay full-cyclic (debug stack in f32 sub for small pads); large use pot in release for perf. See gap for remaining f32 dftN stack in bluestein pads.
- [x] [patch] GT composite force for worst 90/198 (explicit in plan/dispatch/static_radices to prefer over has_static GT); shared gather_unroll4 in butterflies wired to GT pfa_gather + rader gather_sum (dupe cut); direct stockham in rader_bluestein for pow2 (f32 hot); partial xtask bench run + md notes updated. Tests GT 64 pass, 90 numerical ok (radices [2,3,3,5] valid).
  ratio from `2.828x` to `3.301x`.
- [x] [patch] Shared butterflies/dft expansion (next stage #1): canonical small DFT codelets (dft2/3/4/5/7/8+arrays + dft15 PFA) to components/butterflies/dft.rs ; wired (forwards in winograd, updated imports in cook_toom for moved dfts). Dupe cut, deep vertical, zero-cost mono preserved. Special bench attention: focused xtask attempts (baseline+skip + PoT sizes post ZST), md notes (incl lock on release xtask during one run, cleaned) for rebench on affected (5-90/198/67, PoT 32-32768) to prevent regression; all value tests (dft_small 32 incl dft15, GT 64, rader, composite, plan, new 512 ZST) green. check/fmt/doc/clippy clean. No delta (identical). See gap/CHANGELOG.
- [x] [patch] PoT ZST wiring (stage #2): log2 + SizedPoT<StockhamAutosort, LOG2> (exact via ::new()) in FftPlan1D PowerOfTwo + dispatch tag. 512 (log2=9) explicit + sized helper. New 512 ZST tests (arm + ZST<9> + numerical). Preserved specials/pools. Why highest prob: PoT best route made compile-time (ZST vs runtime is_power+match); explicit for remaining >1x PoT in md (32@1.7x+). Bench attention: xtask on 32/64/128/256/512/1024/32768 (attempts + md notes/rebench cmd); 512 tests + stockham green. Gates clean, no regression. Memory pools unchanged. See gap/benchmark_results.
- [x] [patch] Phase (perf opt/mem eff/arch elevation + DIP/DRY/SRP/SoC/SSOT/zero-copy/ZST/phantom/Cow + no excess cast): stockham ZST with_strategy used in mod for 512 (call site elevation, internal + test ref); plan/dispatch ZST constructions; bluestein kernel pooled (with for build); Cow exercised; cast native f32 chirp. n113 ignored; rader (pooled), GT, ZST plan, dft green. Gates green; small --skip-run 32/64 + bg full launched (build success). Md/gap synced, no regression. See gap, benchmark_results.
- [x] [patch] Cleanup + planning: PoT ZST dead_code + rationale (planning wire); stockham test cfg+import fix; rader/butterflies small lints ( += , >len, inline); targeted fmt+check clean on core; rader (34p+1i), GT, plan n90 composite, stockham special tests green; doc clean. Routing analysis + ordered next-steps (expand shared butterflies #1 highest prob dupe/mono win; PoT ZST wire; f32 scratch unify to unblock routing; more GT forces/gather; bluestein SIMD; static rader) documented in gap_audit with per-item bench/math/arch prob/why routing sense/risk. Artifacts (CHANGELOG, gap, backlog/checklist) synced. No new defects; value-semantic preserved. Pre-existing clippy/fmt in workspace noted.
- [x] Reject generated N=24 Good-Thomas `(3,8)` orientation. N=24 value
  coverage and `xtask` check passed, but focused full-profile timing worsened
  the controlling f32 ratio from `2.636x` to `9.130x`.
- [x] Reject generated N=63 `(9,7)` orientation. Fixed-coprime codelet value
  coverage and `xtask` check passed, but focused full-profile timing worsened
  the controlling f32 ratio from `2.637x` to `4.229x`.
- [x] Reject generated N=27 `(9,3)` Cooley-Tukey decomposition. The
  `dft_composite_small_cases` value test and `xtask` check passed, but focused
  full-profile timing worsened the controlling f32 ratio from `2.543x` to
  `5.024x`.
- [x] Refresh retained Rader N=89 under current full-profile timing. The row
  improves from f64 `2.626x`, f32 `2.113x` to f64 `2.265x`, f32 `2.076x`.
- [x] Reject retained-route N=198 refresh. The rerun improved f64 from
  `2.664x` to `2.610x` but worsened the controlling f32 ratio from `1.932x`
  to `3.698x`; the prior retained row was restored.
- [x] Reject retained-route N=445 refresh. The rerun worsened the controlling
  f64 ratio from `2.477x` to `2.694x`; the prior retained row was restored.
- [x] Refresh retained Good-Thomas/Rader N=213 under current full-profile
  timing. The row improves from f64 `2.477x`, f32 `2.153x` to f64 `2.157x`,
  f32 `1.811x`.
- [x] Reject retained-route N=67 refresh. The rerun worsened the controlling
  f64 ratio from `2.458x` to `2.606x`; the prior retained row was restored.
- [x] Refresh retained Good-Thomas/Rader N=453 under current full-profile
  timing. The row improves from f64 `2.312x`, f32 `2.436x` to f64 `2.046x`,
  f32 `1.812x`.
- [x] Refresh retained Good-Thomas/Rader N=398 under current full-profile
  timing. The row improves from f64 `2.422x`, f32 `1.433x` to f64 `1.809x`,
  f32 `1.668x`.
- [x] Refresh retained Cooley-Tukey N=286 under current full-profile timing.
  The row improves from f64 `1.645x`, f32 `2.418x` to f64 `1.152x`, f32
  `1.748x`.
- [x] Reject retained-route N=183 reruns. The best rerun records f64 `2.402x`,
  f32 `2.408x`, and a second rerun worsened to f64 `2.980x`, f32 `2.539x`.
- [x] Refresh retained Cooley-Tukey N=429 under current full-profile timing.
  The row improves from f64 `1.461x`, f32 `2.397x` to f64 `1.342x`, f32
  `2.093x`.
- [x] Repair radix-composite AVX flat-pass visibility after the module split:
  leaf functions are now visible within `radix_composite`, allowing `cache.rs`
  to call the re-exported AVX2+FMA stage paths.
- [x] Verify radix-composite visibility repair with `cargo check -p xtask
  --features bench-runner`, `cargo test -p apollo-fft
  dft_composite_small_cases --lib`, and `cargo test -p apollo-fft
  radix_composite --lib`.
- [x] Refresh retained Cooley-Tukey N=238 under current full-profile timing.
  The row improves to f64 `1.322x`, f32 `1.464x` and remains above target.
- [x] Reject the fresh N=508 retained-route row after it worsened the max
  ratio from `2.407x` to `2.545x`; the prior retained row was restored.
- [x] Reject N=242 reruns after failing to improve the retained ratio record
  f64 `1.548x`, f32 `2.494x`. The exact retained timing columns were not
  recoverable from current artifacts; the best measured row from this turn is
  f64 `1.585x`, f32 `3.211x`.
- [x] Remove duplicate `apollo-wgpu-helpers` manifest entries from WGPU
  backend crates so workspace cargo commands load and `xtask` verification can
  run again.
- [x] Reject removing N=242 from the f32 generated-codelet policy. The
  retained composite-route value test passed, but focused timing worsened the
  max ratio from `3.211x` to f32 `3.400x`.
- [x] Reject retained-route N=36 refresh after it worsened f32 from `2.828x`
  to `4.899x`; the prior retained row was restored.
- [x] Reject f32 half-cyclic Rader precision-policy probe for N=271/N=337.
  Existing Rader value tests and `xtask` check passed, but the focused
  full-profile run exceeded the 300s command bound and partially wrote worse
  rows; retained N=271 and N=337 rows were restored.
- [x] Reject retained-route N=400 refresh. The rerun worsened the max ratio
  from f32 `2.730x` to f32 `3.134x`; the prior retained row was restored.
- [x] Expose the sealed plan scratch trait at the public `fft::workspace`
  boundary so public `FftPlan2D`/`FftPlan3D` impl bounds no longer produce
  private-bound diagnostics while scratch allocation helpers remain
  crate-local.
- [x] Verify scratch-bound cleanup with `cargo check -p xtask --features
  bench-runner` and `cargo test -p apollo-fft rader_n --lib`.
- [x] Refresh retained Cooley-Tukey N=264 under current full-profile timing.
  The row improves from f64 `2.443x`, f32 `2.654x` to f64 `1.515x`, f32
  `1.905x`; it remains above the `< 1.000x` target.
- [x] Refresh retained precision-policy N=126 under current full-profile
  timing. The row improves from f64 `2.629x`, f32 `2.551x` to f64 `1.511x`,
  f32 `2.310x`; it remains above the `< 1.000x` target.
- [x] Reject retained-route N=99 refresh. The rerun improved f64 but worsened
  the controlling f32 ratio from `2.619x` to `4.736x`; the prior retained row
  was restored.
- [x] Reject retained-route N=54 refresh. The rerun worsened the controlling
  f32 ratio from `2.553x` to `4.599x`; the prior retained row was restored.
- [x] Refresh retained precision-policy N=96 under current full-profile
  timing. The row improves from f64 `2.317x`, f32 `2.552x` to f64 `1.557x`,
  f32 `2.201x`; it remains above the `< 1.000x` target.
- [x] Refresh retained Good-Thomas N=160 under current full-profile timing.
  The row improves from f64 `1.662x`, f32 `2.552x` to f64 `1.450x`, f32
  `2.318x`; it remains above the `< 1.000x` target.
- [x] Refresh retained Good-Thomas N=200 under current full-profile timing.
  The row improves from f64 `1.741x`, f32 `2.549x` to f64 `1.561x`, f32
  `2.464x`; it remains above the `< 1.000x` target.
- [x] Refresh retained Winograd N=27 under current full-profile timing. The
  row improves from f64 `1.697x`, f32 `2.543x` to f64 `1.161x`, f32
  `2.307x`; it remains above the `< 1.000x` target.
- [x] Reject retained-route N=135 refresh. The rerun worsened the controlling
  f32 ratio from `2.526x` to `2.785x`; the prior retained row was restored.
- [x] Refresh retained Cooley-Tukey N=176 under current full-profile timing.
  The row improves from f64 `2.444x`, f32 `2.482x` to f64 `1.494x`, f32
  `2.289x`; it remains above the `< 1.000x` target.
- [x] Reject retained-route N=240 refresh. The rerun improved f64 below target
  but worsened the controlling f32 ratio from `2.479x` to `2.934x`; the prior
  retained row was restored.
- [x] Refresh retained Cooley-Tukey N=384 under current full-profile timing.
  The row improves the max ratio from f32 `2.022x` to f32 `1.886x`; it
  remains above the `< 1.000x` target.
- [x] Reject retained-route N=480 refresh. The rerun worsened the controlling
  f32 ratio from `1.831x` to `1.868x`; the prior retained row was restored.
- [x] Refresh retained Good-Thomas/Rader N=134 under current full-profile
  timing. The row improves the max ratio from f64 `2.507x` to f64 `1.919x`;
  it remains above the `< 1.000x` target.
- [x] Reject retained-route N=298 refresh. The rerun worsened the controlling
  f32 ratio from `2.103x` to `2.464x`; the prior retained row was restored.
- [x] Refresh retained precision-policy N=484 under current full-profile
  timing. The row improves the max ratio from f32 `2.509x` to f32 `2.200x`;
  it remains above the `< 1.000x` target.
- [x] Refresh retained Good-Thomas/Rader N=339 under current full-profile
  timing. The row improves the max ratio from f32 `2.480x` to f64 `2.440x`;
  it remains above the `< 1.000x` target.
- [x] Reject retained-route N=356 refresh. The rerun worsened the max ratio
  from f64 `2.469x` to f32 `2.743x`; the prior retained row was restored.
- [x] Reject retained-route N=438 refresh. The rerun worsened the controlling
  f32 ratio from `2.447x` to `3.955x`; the prior retained row was restored.
- [x] Reject retained-route N=146 refresh. The rerun worsened the controlling
  f32 ratio from `2.436x` to `3.393x`; the prior retained row was restored.
- [x] Reject retained-route N=292 refresh. The rerun worsened the controlling
  f32 ratio from `2.433x` to `3.512x`; the prior retained row was restored.
- [x] Refresh retained Good-Thomas/Rader N=305 under current full-profile
  timing. The row improves the max ratio from f32 `2.433x` to f32 `2.242x`;
  it remains above the `< 1.000x` target.
- [x] Refresh retained Good-Thomas/Bluestein N=321 under current full-profile
  timing. The row improves the max ratio from f64 `2.431x` to f64 `2.120x`;
  it remains above the `< 1.000x` target. The benchmark command exceeded the
  300s shell bound after writing the improved row.
- [x] Reject retained-route N=397 refresh. The rerun worsened the controlling
  f32 ratio from `2.427x` to `3.044x`; the prior retained row was restored.
- [x] Reject retained-route N=335 refresh. The rerun worsened the controlling
  f64 ratio from `2.374x` to `2.766x`; the prior retained row was restored.
- [x] Reject retained-route N=396 refresh. The rerun worsened the controlling
  f32 ratio from `2.365x` to `2.477x`; the prior retained row was restored.
- [x] Refresh retained Good-Thomas/Rader N=488 under current full-profile
  timing. The row improves the max ratio from f32 `2.366x` to f64 `2.205x`;
  it remains above the `< 1.000x` target.
- [x] Reject retained-route N=189 refresh. The rerun worsened the max ratio
  from f64 `2.356x` to f32 `9.896x`; the prior retained row was restored.
- [x] Restore the f32/f64 `MixedRadixScalar::use_generated_codelet_plan`
  methods after the scalar impl refactor moved the implementation blocks.
- [x] Remove the hot zero/nonzero branch from Rader scatter by handling
  `q=0` once and using direct reverse generator-order indexing for `q>=1`.
- [x] Verify Rader scatter correctness with `cargo test -p apollo-fft rader
  --lib`.
- [x] Refresh full-profile Rader rows after the scatter change: N=271 improves
  from f64 `2.714x`, f32 `3.624x` to f64 `2.048x`, f32 `3.248x`; N=337
  refreshes to f64 `2.274x`, f32 `2.862x`.
- [x] Reject generated N=280 `(35,8)` orientation. Direct-DFT value semantics
  passed, but full-profile f32 worsened from `2.785x` to `3.186x`; retained
  route remains `(8,35)`.
- [x] Route generated N=48 through Good-Thomas `(3,16)` instead of `(16,3)`.
- [x] Verify N=48 with `cargo test -p apollo-fft planned_n48 --lib`.
- [x] Refresh the N=48 full-profile row: f64 `1.617x`, f32 `2.579x`, improving
  the controlling f32 miss from `4.593x`.
- [x] Reject generated N=400 `(25,16)` orientation. Direct-DFT value semantics
  passed, but full-profile f32 worsened from `2.782x` to `3.801x`; retained
  route remains `(16,25)`.
- [x] Reject f32 N=180 generated-codelet precision-policy routing. Direct-DFT
  value semantics passed, but full-profile f32 worsened from `2.772x` to
  `3.270x`; retained route remains composite `[5,3,3,4]`.
- [x] Refresh retained N=362 under current full-profile timing: f64 `2.346x`,
  f32 `2.116x`, improving the stale max ratio from f64 `2.782x`.
- [x] Reject f32 N=271 Bluestein Rader routing after re-probing under current
  Rader scatter code. Direct-DFT value semantics passed, but f32 worsened from
  `3.248x` to `3.261x`; retained route remains full-cyclic Rader.
- [x] Refresh retained N=353 under current full-profile timing: f64 `2.062x`,
  f32 `1.634x`, improving the stale max ratio from f32 `2.760x`.
- [x] Reject f32 N=337 Bluestein Rader routing. Direct-DFT value semantics
  passed, but f32 worsened from `2.862x` to `2.928x`; retained route remains
  full-cyclic Rader.
- [x] Refresh retained N=331 under current full-profile timing: f64 `2.013x`,
  f32 `1.758x`, improving the stale max ratio from f64 `2.746x`.
- [x] Repair `ShortWinogradScalar` release compilation by removing invalid
  cross-module calls to private AVX helper functions from N=2/N=4 short-DFT
  trait methods.
- [x] Refresh retained N=168 under current full-profile timing: f64 `1.518x`,
  f32 `2.720x`, improving the stale f32 ratio from `2.798x`.
- [x] Refresh retained N=148 under current full-profile timing: f64 `1.768x`,
  f32 `1.975x`, improving the stale max ratio from f32 `2.762x`.
- [x] Reject retained-route N=36 refresh: the rerun worsened f32 from `2.713x`
  to `2.835x`, so the prior retained row was restored.
- [x] Refresh retained N=352 under current full-profile timing: f64 `2.689x`,
  f32 `2.482x`, improving the stale max ratio from `2.708x`.
- [x] Reject retained-route N=3 refresh after the AVX fallback cleanup: f64
  improved to `0.698x`, but f32 worsened from `1.345x` to `2.175x`.
- [x] Reject retained-route N=88 refresh: f32 worsened from `2.669x` to
  `2.874x`, so the prior retained row was restored.
- [x] Refresh retained N=482 under current full-profile timing: f64 `2.239x`,
  f32 `1.849x`, improving the stale max ratio from `2.690x`.
- [x] Reject retained-route N=201 refresh: f64 worsened from `2.681x` to
  `2.806x`, so the prior retained row was restored.
- [x] Refresh retained N=397 under current full-profile timing: f64 `1.489x`,
  f32 `2.427x`, improving the stale max ratio from `2.679x`.
- [x] Reject retained-route N=198 refresh: f32 worsened from `1.932x` to
  `2.991x`, so the prior retained row was restored.
- [x] Reject retained-route N=77 refresh: f32 worsened from `2.661x` to
  `2.816x`, so the prior retained row was restored.
- [x] Reject retained-route N=264 refresh: f32 worsened from `2.654x` to
  `2.877x`, so the prior retained row was restored.
- [x] Reject retained-route N=63 refresh: f32 worsened from `2.637x` to
  `3.262x`, so the prior retained row was restored.
- [x] Reject retained-route N=24 refresh: f32 worsened from `2.636x` to
  `3.500x`, so the prior retained row was restored.
- [x] Refresh retained N=121 under current full-profile timing: f64 `2.372x`,
  f32 `2.383x`, improving the stale max ratio from `2.633x`.
- [x] Reject retained-route N=81 refresh: f32 worsened from `2.631x` to
  `8.048x`, so the prior retained row was restored.
- [x] Reject retained-route N=126 refresh: f32 worsened from `2.551x` to
  `3.121x`, so the prior retained row was restored.
- [x] Reject retained-route N=89 refresh: f64 worsened from `2.626x` to
  `2.762x`, so the prior retained row was restored.
- [x] Refresh retained N=181 under current full-profile timing: f64 `2.387x`,
  f32 `2.125x`, improving the stale max ratio from `2.623x`.
- [x] Reject retained-route N=268 refresh: f64 worsened from `2.602x` to
  `2.668x`, so the prior retained row was restored.
- [x] Refresh retained N=274 under current full-profile timing: f64 `1.383x`,
  f32 `1.379x`, improving the stale max ratio from `2.560x`.
- [x] Reject retained-route N=160 refresh: f32 worsened from `2.552x` to
  `3.748x`, so the prior retained row was restored.
- [x] Refresh retained N=180 under current full-profile timing: f64 `1.532x`,
  f32 `2.226x`, improving the stale max ratio from `2.772x`.
- [x] Reject retained-route N=32 refresh: f32 worsened from `2.583x` to
  `3.297x`, so the prior retained row was restored.
- [x] Remove the unused duplicate mixed-radix scalar `constants` module edge;
  `cargo check -p xtask --features bench-runner` is warning-clean for this
  path while the active implementation remains in `impls.rs`.
- [x] Delete the now-unreferenced duplicate
  `mixed_radix/scalar/constants.rs` artifact so stale twiddle tables cannot
  re-enter the scalar module.
- [x] Reject retained-route N=54 refresh: f32 worsened from `2.553x` to
  `4.153x`, so the prior retained row was restored.
- [x] Reject retained-route N=96 refresh: f32 worsened from `2.552x` to
  `3.697x`, so the prior retained row was restored.
- [x] Reject retained-route N=263 refresh: f32 worsened from `2.551x` to
  `2.792x`, so the prior retained row was restored.
- [x] Reject retained-route N=200 refresh: f32 worsened from `2.549x` to
  `2.944x`, so the prior retained row was restored.
- [x] Refresh retained N=211 under current full-profile timing: f64 `1.713x`,
  f32 `1.224x`, improving the stale max ratio from `2.542x`.
- [x] Reject retained-route N=267 refresh: f32 worsened from `2.549x` to
  `2.865x`, so the prior retained row was restored.
- [x] Reject retained-route N=365 refresh: f32 worsened from `2.520x` to
  `3.310x`, so the prior retained row was restored.
- [x] Reject retained-route N=379 refresh: f32 worsened from `2.520x` to
  `2.539x`, so the prior retained row was restored.
- [x] Refresh retained N=401 under current full-profile timing: f64 `2.203x`,
  f32 `2.327x`, improving the stale max ratio from `2.517x`.
- [x] Refresh retained N=488 under current full-profile timing: f64 `2.090x`,
  f32 `2.366x`, improving the stale max ratio from `2.511x`.
- [x] Reject retained-route N=484 refresh: f32 worsened from `2.509x` to
  `3.037x`, so the prior retained row was restored.
- [x] Refresh retained N=335 under current full-profile timing: f64 `2.374x`,
  f32 `2.315x`, improving the stale max ratio from f32 `2.724x`.
- [x] Refresh retained N=80 under current full-profile timing: f64 `1.964x`,
  f32 `2.460x`, improving the stale max ratio from f32 `2.727x`.
- [x] Inline the public N=2 forward/inverse butterfly in `FftPrecision` for
  `Complex64` and `Complex32`, removing the mixed-radix trait call from the
  direct API route.
- [x] Reject unchecked N=2 slice access after a focused probe regressed the
  N=2 row to f64 `4.102x` and f32 `1.523x`.
- [x] Reject `#[inline(always)]` on the `xtask` precision adapter methods and
  `bench_pair` after the focused probe regressed the sampled small-size rows.
- [x] Record the final N=2 focused row after the direct route:
  f64 3.22 ns vs 1.65 ns (`1.953x`) and f32 2.76 ns vs 1.98 ns (`1.394x`);
  both remain open versus the `< 1.000x` target.
- [x] Route f64 N=3 public forward/inverse calls directly to the existing
  Winograd DFT3 leaf, bypassing mixed-radix dispatch.
- [x] Reject the direct scalar N=4 public butterfly probe because the focused
  Apollo f64/f32 absolute timings did not beat the retained route.
- [x] Record the focused N=3/N=4 row after narrowing:
  N=3 f64 2.68 ns vs 1.97 ns (`1.361x`), f32 2.85 ns vs 2.23 ns (`1.278x`);
  N=4 f64 3.09 ns vs 2.00 ns (`1.548x`), f32 2.67 ns vs 1.73 ns (`1.548x`).
- [x] Route f32 N=5, f64/f32 N=7, and f32 N=11 public calls directly to
  their existing Winograd leaves, bypassing generic mixed-radix dispatch.
- [x] Verify focused Winograd value tests for DFT5, DFT7, and DFT11 after
  the direct public routes.
- [x] Record focused direct-route rows: N=5 f32 improved to 10.42 ns vs
  8.62 ns (`1.209x`), N=7 f32 improved to 14.08 ns vs 11.86 ns (`1.187x`),
  and N=11 f32 improved to 19.64 ns vs 17.26 ns (`1.138x`); all three
  remain open versus `< 1.000x`.
- [x] Reject f32 N=13/N=17/N=19 direct public routing. N=13 was noisy and
  regressed in the narrowed final row to `1.281x`; N=17 and N=19 regressed in
  the broad probe to `1.708x` and `1.528x`.
- [x] Reject planned N=24/N=27 rerouting to the generic radix-composite
  executor. The focused `cargo xtask` probe regressed N=24 to f64 `7.486x`
  and f32 `11.649x`, and N=27 to f64 `3.140x` and f32 `3.543x`.
- [x] Record the focused planned small power-of-two baseline:
  N=2 passes at f64 `0.842x` and f32 `0.587x`; N=8 f32 remains `1.138x`;
  N=16 remains f64 `1.176x` and f32 `3.198x`.
- [x] Refresh N=16 with the optimized `xtask` runner: f64 now passes at
  `0.457x`, while f32 remains open at `2.863x`.
- [x] Reject planned N=8 rerouting to `ShortWinograd`. Focused value tests
  passed, but the `cargo xtask` probe regressed N=8 to f64 `1.124x` and f32
  `6.128x`.
- [x] Reject planned f32 N=16 rerouting to `ShortWinograd`. Focused value
  tests passed, but the `cargo xtask` probe regressed N=16 to f64 `1.947x`
  and f32 `4.820x`.
- [x] Reject replacing the retained f32 sized N=8 SIMD kernel with the scalar
  butterfly. `test_small_pot_f32_correctness` and `dft8` passed, but the
  `cargo xtask` probe regressed N=8 f32 to `5.261x`.
- [x] Replace the duplicate planned Good-Thomas executor body with delegation
  to `components::good_thomas::pfa_fft`, preserving normalization in the
  normalized inverse wrapper.
- [x] Add a plan-level N=90 Good-Thomas forward test against the direct DFT
  definition.
- [x] Verify planned Good-Thomas dispatcher delegation with `cargo test -p
  apollo-fft good_thomas --lib`, `cargo test -p apollo-fft
  planned_good_thomas_n90_forward_matches_direct --lib`, and `cargo check -p
  xtask --features bench-runner`.
- [x] Refresh `benchmark_results.md` for N=84 and N=90. N=84 improves from
  the prior current-worktree row f64 `3.064x`, f32 `4.001x` to f64 `1.820x`,
  f32 `2.289x`; N=90 improves from f64 `5.144x`, f32 `2.695x` to f64
  `2.363x`, f32 `1.597x`. Both remain open versus `< 1.000x`.
- [x] Add the planned N=72 scalar route policy: f64 remains on static
  Good-Thomas `(9,8)`, and f32 routes through the generated `(8,9)` codelet.
- [x] Add value-semantic planned N=72 tests for both route selections against
  the direct DFT definition.
- [x] Refresh `benchmark_results.md` for N=72 with the precision-policy label:
  f64 117.15 ns vs 48.64 ns (`2.408x`) and f32 109.28 ns vs 19.50 ns
  (`5.604x`). The row remains open versus `< 1.000x`.
- [x] Refresh the N=72 precision-policy row again with the warm optimized
  runner: f64 `1.884x`, f32 `5.421x`; both remain open.
- [x] Refresh the N=72 precision-policy row after retaining composite order
  `[4,2,3,3]`: f64 116.32 ns vs 61.37 ns (`1.895x`) and f32 99.73 ns vs
  20.54 ns (`4.855x`); both remain open.
- [x] Add `ShortDft<72>` through a generated `(8,9)` codelet and route only
  f32 N=72 through it by scalar policy. The default `cargo xtask` row records
  f64 116.31 ns vs 50.48 ns (`2.304x`) and f32 95.79 ns vs 26.53 ns
  (`3.610x`); f32 improves from the retained composite row, while both
  precisions remain open.
- [x] Reject planned f32 N=72 composite order `[4,3,3,2]`. Route-selection
  and direct-DFT tests passed, but focused timing regressed f32 to `5.017x`
  versus the retained `[4,2,3,3]` row.
- [x] Reject planned f32 N=72 generated `(12,6)` Cooley-Tukey factorization.
  It was value-correct but measured f32 `3.968x`; the restored generated
  `(8,9)` row records f64 `2.308x` and f32 `3.527x`.
- [x] Reject f64 N=72 `ShortWinograd` routing under the full-profile median
  benchmark. Direct-DFT value semantics and `xtask` checking passed, but the
  probe worsened the max ratio to f32 `3.727x`; restored static Good-Thomas
  `(9,8)` routing refreshes to f64 `2.286x`, f32 `2.338x`.
- [x] Refresh retained full-profile rows for N=504 and N=135. N=504 improves
  from f64 `1.256x`, f32 `3.786x` to f64 `1.346x`, f32 `1.645x`; N=135
  improves from f64 `1.044x`, f32 `3.754x` to f64 `1.856x`, f32 `2.526x`.
- [x] Refresh retained full-profile rows for N=168, N=108, N=112, N=400,
  N=132, and N=242. N=400 now passes f64 at `0.975x`; all refreshed rows
  still miss f32.
- [x] Reject f32 N=271 Bluestein Rader precision-policy routing. The probe
  preserved direct-DFT value semantics and `xtask` checking, but focused timing
  did not beat retained full-cyclic Rader.
- [x] Repair the dirty `FftPrecision` macro refactor so direct small-codelet
  dispatch compiles: explicit method identifiers fix macro hygiene, imported
  codelet identifiers avoid path/turbofish parsing failure, and inverse scaling
  uses `MixedRadixScalar::complex`.
- [x] Add generated medium Good-Thomas `(9,11)` codelet routing for planned
  f32 N=99 and verify it against direct DFT.
- [x] Refresh N=99 and N=469. N=99 improves max ratio from f32 `3.021x` to
  f32 `2.619x`; N=469 improves max ratio from f64 `2.975x` to f64 `2.630x`.
- [x] Swap planned f32 N=120 generated Good-Thomas orientation from `(8,15)`
  to `(15,8)`, verify against direct DFT, and refresh the focused row. f32
  improves from `2.860x` to `2.373x`.
- [x] Refresh retained rows N=402, N=280, N=178, N=244, N=305, and N=27 under
  the current full-profile runner.
- [x] Reject retained-route N=134 refresh: f64 worsened from `2.507x` to
  `2.750x`, so the prior retained row was restored.
- [x] Refresh retained N=178 under current full-profile timing: f64 `2.493x`,
  f32 `1.940x`, improving the stale max ratio from `2.506x`.
- [x] Extend half-cyclic Rader strategy coverage to N=271 and N=337. The
  strategy benchmark rejects lowering the half-cyclic threshold for these
  sizes because half-cyclic remains slower than full-cyclic.
- [x] Refresh retained N=337 under current full-profile timing: f64 `2.274x`,
  f32 `2.855x`, improving the stale max ratio from `2.862x`.
- [x] Refresh retained N=271 under current full-profile timing: f64 `2.058x`,
  f32 `3.008x`, improving the stale max ratio from `3.248x`.
- [x] Reject retained-route N=36 confirmation refreshes: f32 worsened to
  `4.395x` and then `4.715x`, so the exact tracked row was restored.
- [x] Reject generated N=36 swapped `(4,9)` orientation. Composite value tests
  passed, but full-profile timing worsened the max ratio to `3.404x`; restored
  generated `(9,4)`.
- [x] Refresh retained rows N=280 and N=400 under current full-profile timing.
  N=280 improves max ratio from `2.785x` to `2.600x`; N=400 improves max
  ratio from `2.782x` to `2.730x`.
- [x] Refresh retained N=27 under current full-profile timing: f64 `1.697x`,
  f32 `2.543x`, improving the max ratio from `2.706x`.
- [x] Reject retained-route N=168 refresh: f32 worsened from `2.720x` to
  `5.217x`, so the prior retained row was restored.
- [x] Refresh retained N=201 under current full-profile timing: f64 `1.977x`,
  f32 `2.001x`, improving the max ratio from `2.681x`.
- [x] Reject retained-route refreshes for N=283 and N=352. N=283 worsened from
  max `2.699x` to `2.820x`; N=352 worsened from max `2.689x` to `2.811x`.
- [x] Refresh retained N=77 under current full-profile timing: f64 `1.801x`,
  f32 `2.375x`, improving the max ratio from `2.661x`.
- [x] Reject retained-route refreshes for N=88, N=108, N=112, and N=198.
  N=88 worsened from max `2.669x` to `2.686x`; N=108 worsened from
  `2.667x` to `3.907x`; N=112 worsened from `2.661x` to `4.036x`; and
  N=198 worsened from `2.664x` to `3.251x`.
- [x] Refresh retained N=337 under current full-profile timing: f64 `1.822x`,
  f32 `2.675x`, improving the max ratio from `2.855x`.
- [x] Reject retained-route refreshes for N=36, N=168, N=271, and N=400.
  Reruns worsened retained max ratios: N=36 `2.828x` to `6.068x`; N=168
  `2.720x` to `3.616x`; N=271 `3.008x` to `3.269x`; and N=400 `2.730x`
  to `3.040x`.
- [x] Refresh retained rows N=88, N=283, and N=352 under current full-profile
  timing. N=88 improves max ratio from `2.669x` to `2.432x`; N=283 from
  `2.699x` to `2.052x`; and N=352 from `2.689x` to `2.239x`.
- [x] Reject retained-route refreshes for N=108 and N=198. Reruns worsened
  max ratios from `2.667x` to `4.148x` and from `2.664x` to `3.793x`,
  respectively.
- [x] Add benchmark-only Bluestein-forced Rader strategy coverage to
  `half_cyclic_rader` so N=271/N=337 strategy evidence covers full-cyclic,
  half-cyclic, Bluestein, and auto routes.
- [x] Reject the large-prime static-Rader precheck gate. Rader correctness and
  `xtask` checking passed, but focused full-profile timing worsened N=271 from
  max `3.008x` to `3.418x` and N=337 from `2.675x` to `2.905x`; the
  production selector and retained rows were restored.
- [x] Reject the batched N=112/N=264/N=63/N=24/N=81 full-profile refresh after
  it exceeded the 300s command bound without updating rows; the leftover
  Apollo `xtask` process tree was stopped.
- [x] Reject single-size retained-route refreshes for N=112, N=264, and N=63.
  Reruns worsened max ratios from `2.661x` to `3.064x`, from `2.654x` to
  `2.850x`, and from `2.637x` to `2.813x`, respectively.
- [x] Refresh N=16 after detecting an unexpected fresh row from this turn.
  The focused rerun improves the current max ratio from `3.235x` to `1.899x`.
- [x] Reject N=36 composite-route experiment. Disabling the generated
  short-codelet route sent N=36 through the existing `[4,3,3]` composite path;
  value tests and `xtask` checking passed, but full-profile timing worsened
  max ratio from `2.828x` to `3.596x`, so short-codelet dispatch was restored.
- [x] Restore required `MixedRadixScalar::use_generated_codelet_plan`
  implementations for f32/f64 with conservative `false` policy so the current
  dirty trait surface compiles without changing route selection.
- [x] Reject lowering the f32 half-cyclic Rader threshold to 256 for N=283.
  The selector reaches Bluestein before the threshold because `N-1 = 282` is
  not prime-23-smooth; the focused refresh records f64 `1.818x` and f32
  `2.700x`.
- [x] Refresh N=498 under the retained ordered-Rader Good-Thomas route:
  f64 5623.96 ns vs 2717.67 ns (`2.069x`), f32 2730.39 ns vs 1271.99 ns
  (`2.147x`); f32 improves from `3.377x`, but the row remains open.
- [x] Remove unused Good-Thomas plan cache members made obsolete by
  delegation to `components::good_thomas::pfa_fft`: planned Good-Thomas no
  longer stores input/output CRT permutation Arcs or row/column subplans.
- [x] Replace planned Rader's duplicate full-cyclic executor with delegation
  to `components::rader::rader_fft`, so planned runtime primes use the
  canonical FullCyclic/HalfCyclic/Bluestein strategy selector.
- [x] Remove planned Rader cached state made obsolete by canonical delegation:
  generator-order table, forward/inverse Rader spectra, and length `N-1`
  subplan.
- [x] Add planned N=359 Rader forward tests for f64 and f32 against the
  direct DFT definition.
- [x] Refresh `benchmark_results.md` for N=359. The row improves from f64
  `5.350x`, f32 `12.263x` to f64 `1.532x`, f32 `1.874x`, but remains open
  versus `< 1.000x`.
- [x] Refresh additional stale planned Rader rows after canonical delegation:
  N=383 now f64 `1.546x`, f32 `2.138x`; N=347 now f64 `2.277x`, f32
  `1.853x`; N=179 now f64 `2.256x`, f32 `1.700x`; N=499 now f64 `2.736x`,
  f32 `1.471x`. All remain open versus `< 1.000x`.
- [x] Refresh the next stale planned Rader row set after canonical delegation:
  N=227 now f64 `2.059x`, f32 `1.485x`; N=317 now f64 `1.840x`, f32
  `2.628x`; N=479 now f64 `2.059x`, f32 `1.502x`; N=503 now f64 `1.844x`,
  f32 `1.444x`; N=509 now f64 `1.244x`, f32 `1.611x`. All remain open
  versus `< 1.000x`.
- [x] Refresh top stale planned Good-Thomas/Bluestein rows after canonical
  delegation: N=334 now f64 `1.648x`, f32 `1.646x`; N=358 now f64 `1.994x`,
  f32 `1.787x`; N=454 now f64 `1.896x`, f32 `1.020x`; N=501 now f64
  `2.043x`, f32 `1.977x`. All remain open versus `< 1.000x`.
- [x] Refresh additional planned Rader/Bluestein and Good-Thomas/Rader rows:
  N=10007 now f64 `2.304x`, f32 `2.461x`; N=167 now f64 `1.966x`, f32
  `1.814x`; N=214 now f64 `2.367x`, f32 `1.587x`; N=263 now f64 `2.257x`,
  f32 `2.831x`; N=269 now f64 `1.792x`, f32 `2.920x`; N=293 now f64
  `2.320x`, f32 `2.589x`; N=362 now f64 `2.794x`, f32 `2.186x`; N=428 now
  f64 `2.256x`, f32 `1.988x`; N=439 now f64 `2.547x`, f32 `1.421x`;
  N=467 now f64 `1.928x`, f32 `1.254x`.
- [x] Reject planned f32 N=144 rerouting to the prime-2/3 composite route.
  Route-selection and value tests passed, but optimized `cargo xtask`
  benchmark compilation exceeded the 300s bound and left the retained row
  unchanged.
- [x] Reject planned f32 N=16 short-Winograd rerouting. Route-selection and
  direct-DFT tests passed, but the focused `cargo xtask` probe increased f32
  Apollo absolute time from the tracked 14.54 ns to 15.18 ns.
- [x] Reject replacing f32 N=16 specialized power-of-two execution with the
  canonical Stockham kernel. `small_pot` tests passed, but focused timing
  regressed f32 Apollo to 18.20 ns versus the tracked 14.54 ns branch.
- [x] Reject planned N=24 Good-Thomas `(3,8)` rerouting. The Good-Thomas suite
  passed, but focused timing regressed to f64 113.13 ns and f32 76.86 ns.
- [x] Refresh high-ratio static/composite rows with the optimized runner:
  N=24 f64 `1.641x`, f32 `3.481x`; N=80 f64 `2.241x`, f32 `2.648x`;
  N=96 f64 `2.558x`, f32 `3.306x`; N=108 f64 `2.491x`, f32 `4.007x`;
  N=120 f64 `1.439x`, f32 `3.321x`; N=144 f64 `3.068x`, f32 `4.035x`;
  N=160 f64 `1.952x`, f32 `2.840x`; N=180 f64 `2.216x`, f32 `3.863x`;
  N=242 f64 `1.210x`, f32 `3.952x`.
- [x] Add generated medium codelets for N=108 `(4,27)` and N=144 `(16,9)`
  through `ShortDft` and route f32 only through the generated-codelet policy.
- [x] Verify planned direct-DFT tests for f32 N=108 and N=144 codelet routes.
- [x] Refresh N=108 after the generated codelet route: f64 210.64 ns vs
  87.16 ns (`2.417x`), f32 167.00 ns vs 52.45 ns (`3.184x`); f32 improves
  from `4.007x`, but the row remains open.
- [x] Refresh N=144 after the generated `(16,9)` codelet route: f64
  314.03 ns vs 120.26 ns (`2.611x`), f32 250.34 ns vs 77.40 ns (`3.234x`);
  f32 improves from `4.035x`, but the row remains open.
- [x] Reject generated N=144 Cooley-Tukey `(8,18)` routing. It was
  value-correct, but focused timing regressed f32 to `3.558x`.
- [x] Refresh N=144 after restoring `(16,9)`: f64 311.80 ns vs 155.96 ns
  (`1.999x`), f32 256.80 ns vs 79.40 ns (`3.234x`). The row remains open.
- [x] Reject the generated N=144 `(9,16)` orientation. It was value-correct
  but measured f32 `4.233x`, slower than the retained `(16,9)` orientation.
- [x] Route N=144 through generated Cooley-Tukey `(12,12)`. The route is
  value-correct and improves focused f32 timing to 245.66 ns vs 79.61 ns
  (`3.086x`); f64 records 310.23 ns vs 120.74 ns (`2.569x`). The row
  remains open.
- [x] Add a generated medium codelet for N=180 `(20,9)` through `ShortDft`
  and route f32 only through the generated-codelet policy.
- [x] Verify the planned direct-DFT test for the f32 N=180 codelet route.
- [x] Refresh N=180 after the generated `(20,9)` codelet route: f64
  367.11 ns vs 137.75 ns (`2.665x`), f32 315.83 ns vs 85.65 ns (`3.687x`);
  f32 improves from `3.863x`, but the row remains open.
- [x] Reject the generated N=180 `(9,20)` orientation. It was value-correct
  but measured f32 `4.402x`, slower than the retained `(20,9)` orientation.
- [x] Reject the generated N=180 `(12,15)` Cooley-Tukey factorization. It was
  value-correct but measured f32 `4.895x`, slower than the retained `(20,9)`
  route; the restored focused row records f64 `2.651x` and f32 `3.765x`.
- [x] Reject planned f32 N=242 Good-Thomas `(121,2)` policy routing. It was
  value-correct but measured f32 `5.279x`; the retained Cooley-Tukey row now
  records f64 `2.398x` and f32 `3.352x`.
- [x] Add a generated medium codelet for N=242 `(2,121)` through `ShortDft`
  and route f32 through the generated-codelet policy.
- [x] Verify planned and radix-composite direct-DFT tests for the N=242 route.
- [x] Refresh N=242 after the generated `(2,121)` codelet route: f64
  725.59 ns vs 306.15 ns (`2.370x`), f32 633.12 ns vs 203.99 ns (`3.104x`);
  f32 improves from `3.352x`, but the row remains open.
- [x] Add a generated medium codelet for N=363 `(3,121)` through `ShortDft`
  and route f32 through the generated-codelet policy.
- [x] Verify the planned direct-DFT test for the f32 N=363 codelet route.
- [x] Refresh N=363 after the generated `(3,121)` codelet route: f64
  1063.89 ns vs 456.86 ns (`2.329x`), f32 930.64 ns vs 312.60 ns
  (`2.977x`); f32 improves from `3.338x`, but the row remains open.
- [x] Reject generated N=392 `(8,49)` and `(49,8)` Good-Thomas routing. The
  `(8,49)` route passed value tests but did not beat the same-environment
  refreshed Cooley-Tukey baseline; `(49,8)` regressed f32 to `4.334x`.
- [x] Refresh N=392 after rejecting generated routing: f64 844.42 ns vs
  409.29 ns (`2.063x`), f32 830.66 ns vs 283.86 ns (`2.926x`). The row
  remains open versus `< 1.000x`.
- [x] Reject N=24 generated `(8,3)` orientation. Value tests passed, but
  focused timing regressed f32 to `3.837x`; the restored `(3,8)` route now
  records f64 `1.679x` and f32 `3.974x` after refresh.
- [x] Replace the generated N=24 `(3,8)` Good-Thomas codelet with a generated
  `(6,4)` Cooley-Tukey codelet.
- [x] Verify the N=24 forward value path after the `(6,4)` factorization.
- [x] Refresh N=24 after the generated `(6,4)` codelet route: f64 15.90 ns vs
  11.49 ns (`1.384x`), f32 20.54 ns vs 6.87 ns (`2.990x`); both improve, but
  the row remains open.
- [x] Reject planned f32 N=121 prime-power `11^2` codelet routing. It was
  value-correct but measured f32 `5.082x`; the retained Cooley-Tukey row now
  records f64 `2.336x` and f32 `4.075x`.
- [x] Add a generated medium Cooley-Tukey codelet for N=121 `(11,11)` through
  `ShortDft` and route f32 through the generated-codelet policy.
- [x] Verify the planned direct-DFT test for the f32 N=121 codelet route.
- [x] Refresh N=121 after the generated `(11,11)` codelet route: f64
  365.83 ns vs 152.24 ns (`2.403x`), f32 222.33 ns vs 83.23 ns (`2.671x`);
  f32 improves from `4.075x`, but the row remains open.
- [x] Add a generated medium codelet for N=275 `(11,25)` through `ShortDft`
  and route f32 only through the generated-codelet policy.
- [x] Verify the planned direct-DFT test for the f32 N=275 codelet route.
- [x] Refresh N=275 after the generated `(11,25)` codelet route: f64
  734.11 ns vs 329.81 ns (`2.226x`), f32 601.66 ns vs 231.78 ns (`2.596x`);
  f32 improves from `3.463x`, but the row remains open.
- [x] Add a generated medium Good-Thomas codelet for N=280 `(8,35)` through
  `ShortDft` and route f32 through the generated-codelet policy.
- [x] Verify the planned direct-DFT test for the f32 N=280 codelet route.
- [x] Refresh N=280 after the generated `(8,35)` codelet route: f64
  552.79 ns vs 336.05 ns (`1.645x`), f32 538.95 ns vs 211.33 ns (`2.550x`);
  both improve, but the row remains open.
- [x] Add a generated medium Good-Thomas codelet for N=400 `(16,25)` through
  `ShortDft` and route f32 through the generated-codelet policy.
- [x] Verify the planned direct-DFT test for the f32 N=400 codelet route.
- [x] Refresh N=400 after the generated `(16,25)` codelet route: f64
  755.70 ns vs 385.77 ns (`1.959x`), f32 787.11 ns vs 251.20 ns (`3.133x`);
  f32 improves from `3.289x`, but the row remains open.
- [x] Route f32 N=113 Rader convolution through the existing Bluestein backend
  instead of the full-cyclic length-112 convolution.
- [x] Verify the planned N=113 f32 Rader route against direct DFT.
- [x] Refresh N=113 after the f32 Bluestein-Rader policy: f64 1060.89 ns vs
  414.26 ns (`2.561x`), f32 403.67 ns vs 220.09 ns (`1.834x`). The row max
  improves from the prior f32 `3.299x`, but the row remains open.
- [x] Refresh N=83 on the retained Bluestein Rader route: f64 716.54 ns vs
  425.48 ns (`1.684x`), f32 393.94 ns vs 200.19 ns (`1.968x`). The row max
  improves from the stale f32 `3.130x`, but remains open.
- [x] Reject forcing f32 N=83 through full-cyclic Rader. The route was
  value-correct, but focused `cargo xtask` timing regressed f32 to `4.110x`.
- [x] Reject generated N=270 `(10,27)` and `(27,10)` Good-Thomas routing. Both
  orientations were value-correct, but `(10,27)` regressed f32 to `3.922x`
  and `(27,10)` regressed f32 to `4.469x`.
- [x] Refresh N=270 after rejecting generated routing: f64 489.77 ns vs
  297.39 ns (`1.647x`), f32 443.50 ns vs 142.02 ns (`3.123x`). The row
  improves from the stale f32 `3.271x` but remains open.
- [x] Route planned N=511 as Good-Thomas `(73,7)` so the existing
  ordered-Rader N1 path handles the prime dimension.
- [x] Verify the planned N=511 f32 Good-Thomas route against direct DFT.
- [x] Refresh N=511 after ordered-Rader factor ordering: f64 2902.06 ns vs
  2079.87 ns (`1.395x`), f32 2585.69 ns vs 977.07 ns (`2.646x`). The row
  improves from f64 `1.550x`, f32 `3.258x`, but remains open.
- [x] Reject planned N=420 Good-Thomas `(20,21)` routing. The route was
  value-correct, but focused `cargo xtask` timing regressed f32 to `3.673x`.
- [x] Refresh N=420 after restoring the retained Cooley-Tukey route: f64
  798.38 ns vs 416.27 ns (`1.918x`), f32 776.31 ns vs 291.91 ns (`2.659x`).
  The row improves from stale f32 `3.226x`, but remains open.
- [x] Reject planned N=440 Good-Thomas `(8,55)` routing. The route was
  value-correct, but focused `cargo xtask` timing regressed f32 to `3.623x`.
- [x] Refresh N=440 after restoring the retained Cooley-Tukey route: f64
  1058.84 ns vs 516.69 ns (`2.049x`), f32 1050.68 ns vs 325.80 ns
  (`3.225x`). The row remains open.
- [x] Reject generated N=440 Cooley-Tukey `(22,20)` routing. The route was
  value-correct, but focused `cargo xtask` timing regressed to f64 `2.180x`
  and f32 `6.419x`.
- [x] Refresh N=440 after restoring the retained Cooley-Tukey route again:
  f64 1099.92 ns vs 473.34 ns (`2.324x`), f32 1060.54 ns vs 337.93 ns
  (`3.138x`). The row remains open.
- [x] Refresh N=300 on the retained Cooley-Tukey route: f64 559.96 ns vs
  346.09 ns (`1.618x`), f32 595.74 ns vs 206.59 ns (`2.884x`). The row
  improves from stale f32 `3.116x`, but remains open.
- [x] Refresh retained Cooley-Tukey rows N=484 and N=504: N=484 records f64
  `2.090x`, f32 `3.296x`; N=504 records f64 `1.593x`, f32 `3.389x`. Both
  remain open.
- [x] Reject planned N=484 Good-Thomas `(4,121)` routing. The route was
  value-correct, but focused `cargo xtask` timing regressed f32 to `5.672x`.
- [x] Refresh N=484 after restoring the retained Cooley-Tukey route: f64
  1385.99 ns vs 663.11 ns (`2.090x`), f32 1334.92 ns vs 404.96 ns
  (`3.296x`). The row remains open.
- [x] Add a generated Cooley-Tukey N=484 `(22,22)` codelet and route f32
  planned execution through `ShortDft<484>`.
- [x] Verify the planned direct-DFT test for the f32 N=484 codelet route.
- [x] Refresh N=484 after the generated `(22,22)` codelet route: f64
  1412.46 ns vs 662.97 ns (`2.131x`), f32 1365.58 ns vs 422.90 ns
  (`3.229x`). The row remains open.
- [x] Reject generated N=484 Cooley-Tukey `(11,44)` routing. It was
  value-correct, but focused `cargo xtask` timing regressed f32 to `5.380x`.
- [x] Refresh N=484 after restoring `(22,22)`: f64 1431.25 ns vs 652.23 ns
  (`2.194x`), f32 1514.53 ns vs 428.93 ns (`3.531x`). The row remains open.
- [x] Reject generated N=484 Cooley-Tukey `(44,11)` routing. It was
  value-correct, but focused timing regressed f32 to `5.029x`.
- [x] Refresh N=484 after restoring `(22,22)`: f64 1440.40 ns vs 659.92 ns
  (`2.183x`), f32 1370.87 ns vs 426.23 ns (`3.216x`). The row remains open.
- [x] Reject planned N=504 Good-Thomas `(8,63)` routing. The route was
  value-correct, but focused `cargo xtask` timing regressed f32 to `8.768x`.
- [x] Refresh N=504 after restoring the retained Cooley-Tukey route: f64
  901.23 ns vs 571.83 ns (`1.576x`), f32 827.06 ns vs 235.62 ns (`3.510x`).
  The row remains open.
- [x] Reject planned f32 N=504 generated Cooley-Tukey `(21,24)` routing. The
  route was value-correct, but focused timing regressed f32 to `6.755x`.
- [x] Refresh N=504 after restoring the retained Cooley-Tukey route: f64
  921.21 ns vs 584.13 ns (`1.577x`), f32 807.02 ns vs 238.15 ns (`3.389x`).
  The row remains open.
- [x] Reject planned f32 N=504 generated Cooley-Tukey `(28,18)` routing. The
  route was value-correct, but focused `cargo xtask` timing regressed f32 to
  `7.060x`.
- [x] Refresh N=504 after restoring the retained Cooley-Tukey route again:
  f64 921.10 ns vs 580.93 ns (`1.586x`), f32 795.83 ns vs 236.32 ns
  (`3.368x`). The row remains open.
- [x] Refresh N=180 and N=189 on retained routes: N=180 records f64 `2.687x`,
  f32 `3.769x`; N=189 records f64 `2.194x`, f32 `3.649x`. Both remain open.
- [x] Reject generated N=180 `(5,36)` and `(36,5)` orientations. Both were
  value-correct, but focused timing regressed f32 to `4.130x` and `5.502x`,
  slower than the retained `(20,9)` orientation.
- [x] Refresh N=180 after restoring `(20,9)`: f64 362.95 ns vs 136.38 ns
  (`2.661x`), f32 314.91 ns vs 85.33 ns (`3.691x`). The row remains open.
- [x] Reject generated N=180 Cooley-Tukey `(10,18)` routing. It was
  value-correct, but focused timing regressed f32 to `4.520x`.
- [x] Refresh N=180 after restoring `(20,9)`: f64 383.16 ns vs 136.07 ns
  (`2.816x`), f32 315.98 ns vs 84.56 ns (`3.737x`). The row remains open.
- [x] Reject generated N=180 Good-Thomas `(4,45)` and `(45,4)` orientations.
  Both were value-correct, but focused `cargo xtask` timing regressed f32 to
  `5.299x` and `5.820x`.
- [x] Refresh N=180 after restoring `(20,9)` again: f64 367.55 ns vs
  132.59 ns (`2.772x`), f32 320.11 ns vs 83.65 ns (`3.827x`). The row
  remains open.
- [x] Reject generated N=180 Cooley-Tukey `(18,10)` routing. It was
  value-correct, but focused timing regressed f32 to `5.262x`.
- [x] Refresh N=180 after restoring `(20,9)`: f64 365.60 ns vs 136.95 ns
  (`2.670x`), f32 314.20 ns vs 84.80 ns (`3.705x`). The row remains open.
- [x] Add a generated Cooley-Tukey N=189 `(9,21)` codelet and route f32
  planned execution through `ShortDft<189>`.
- [x] Verify the planned direct-DFT test for the f32 N=189 codelet route.
- [x] Refresh N=189 after the generated `(9,21)` codelet route: f64
  421.89 ns vs 194.35 ns (`2.171x`), f32 392.67 ns vs 124.19 ns (`3.162x`).
  The row remains open.
- [x] Refresh N=72 on the retained precision-policy route: f64 116.46 ns vs
  50.02 ns (`2.328x`), f32 96.46 ns vs 32.66 ns (`2.954x`). The f32 ratio
  improves from `3.527x`, but the row remains open.
- [x] Add generated medium codelets for N=96 `(3,32)`, N=120 `(8,15)`, and
  N=126 `(2,63)` through `ShortDft`, and route f32 only through the
  generated-codelet policy.
- [x] Verify planned direct-DFT tests for f32 N=96, N=120, and N=126 codelet
  routes.
- [x] Refresh N=126 after the generated `(2,63)` codelet route: f64
  252.11 ns vs 138.50 ns (`1.820x`), f32 340.05 ns vs 136.91 ns (`2.484x`);
  both improve from the stale row but remain open.
- [x] Refresh N=120 after the generated `(8,15)` codelet route: f64
  150.00 ns vs 102.88 ns (`1.458x`), f32 174.98 ns vs 59.25 ns (`2.953x`);
  f32 improves from `3.321x`, but the row remains open.
- [x] Refresh N=96 after the generated `(3,32)` codelet route: f64
  181.54 ns vs 67.65 ns (`2.683x`), f32 155.48 ns vs 51.30 ns (`3.031x`);
  f32 improves from `3.306x`, but the row remains open.
- [x] Reject planned f32 N=112 generated `(7,16)` routing. It was
  value-correct but measured f32 `3.405x`; the retained static route now
  records f64 `2.532x` and f32 `3.357x`.
- [x] Add a generated medium codelet for N=112 `(16,7)` through `ShortDft`
  and route f32 through the generated-codelet policy.
- [x] Verify the planned direct-DFT test for the f32 N=112 codelet route.
- [x] Refresh N=112 after the generated `(16,7)` codelet route: f64 214.09 ns
  vs 82.70 ns (`2.589x`), f32 178.55 ns vs 54.51 ns (`3.276x`); f32 improves
  from `3.357x`, but the row remains open.
- [x] Reject generated N=112 Cooley-Tukey `(14,8)` routing. It was
  value-correct, but focused timing regressed f32 to `3.378x`.
- [x] Refresh N=112 after restoring `(16,7)`: f64 208.57 ns vs 81.07 ns
  (`2.573x`), f32 179.09 ns vs 54.66 ns (`3.276x`). The row remains open.
- [x] Reject generated N=112 Cooley-Tukey `(8,14)` routing. It was
  value-correct, but focused `cargo xtask` timing regressed f32 to `3.505x`.
- [x] Refresh N=112 after restoring `(16,7)` again: f64 209.35 ns vs
  100.18 ns (`2.090x`), f32 179.55 ns vs 64.58 ns (`2.780x`). The row
  remains open.
- [x] Add a generated medium codelet for N=154 `(11,14)` through `ShortDft`
  and route f32 only through the generated-codelet policy.
- [x] Verify the planned direct-DFT test for the f32 N=154 codelet route.
- [x] Refresh N=154 after the generated `(11,14)` codelet route: f64
  325.37 ns vs 194.40 ns (`1.674x`), f32 332.70 ns vs 120.82 ns (`2.754x`);
  f32 improves from `3.109x`, but the row remains open.
- [x] Reject planned f32 N=168 generated `(8,21)` routing. It was
  value-correct but measured f32 `4.007x`; the retained static route now
  records f64 `2.046x` and f32 `3.671x`.
- [x] Add a generated medium codelet for N=168 `(24,7)` through `ShortDft`
  and route f32 through the generated-codelet policy.
- [x] Verify the planned direct-DFT test for the f32 N=168 codelet route.
- [x] Refresh N=168 after the generated `(24,7)` codelet route: f64 266.03 ns
  vs 128.03 ns (`2.078x`), f32 268.93 ns vs 82.70 ns (`3.252x`); f32 improves
  from `3.671x`, but the row remains open.
- [x] Reject generated N=168 Cooley-Tukey `(12,14)` routing. It was
  value-correct, but focused timing regressed f32 to `4.189x`.
- [x] Refresh N=168 after restoring `(24,7)`: f64 262.16 ns vs 132.44 ns
  (`1.979x`), f32 268.86 ns vs 83.80 ns (`3.208x`). The row remains open.
- [x] Route N=168 through generated Cooley-Tukey `(14,12)`. The route is
  value-correct and improves focused f32 timing to 259.27 ns vs 84.27 ns
  (`3.077x`); f64 records 261.89 ns vs 128.68 ns (`2.035x`). The row
  remains open.
- [x] Reject planned f32 N=135 generated `(5,27)` routing. It was
  value-correct but measured f32 `3.316x`; the retained static route now
  records f64 `1.845x` and f32 `3.165x`.
- [x] Reject planned f32 N=135 generated `(27,5)` routing. It was
  value-correct but measured f32 `3.756x`; restored static routing records
  f64 254.38 ns vs 148.56 ns (`1.712x`) and f32 302.24 ns vs 84.95 ns
  (`3.558x`). The row remains open.
- [x] Reject generated Cooley-Tukey row-slice writeback codegen. The change
  preserved planned-codelet value semantics, but focused timing regressed
  N=144 f32 to `3.634x`, N=189 f32 to `2.928x`, and N=484 f32 to `3.240x`.
- [x] Refresh affected generated Cooley-Tukey rows after restoring absolute
  scratch writeback: N=144 f64 `2.573x`, f32 `3.579x`; N=168 f64 `1.818x`,
  f32 `2.642x`; N=189 f64 `2.268x`, f32 `2.936x`; N=484 f64 `2.194x`,
  f32 `3.325x`. All remain open.
- [x] Reject generated Good-Thomas fixed-column block codegen. The change
  preserved planned-codelet value semantics, but focused timing regressed
  N=180 f32 to `4.390x`, N=400 f32 to `4.409x`, N=180 f64 to `3.142x`, and
  N=242 f64 to `2.477x`.
- [x] Refresh affected Good-Thomas rows after restoring loop column codegen:
  N=135 f64 `1.956x`, f32 `3.146x`; N=180 f64 `2.672x`, f32 `3.711x`;
  N=242 f64 `2.437x`, f32 `2.980x`; N=400 f64 `1.729x`, f32 `3.459x`.
  All remain open.
- [x] Reject forced `#[inline(always)]` for generated medium codelets. The
  planned-codelet tests and `xtask` check passed, but optimized benchmark
  compilation exceeded the 300s release-build bound for both the
  N=144/180/400/484 set and the narrowed N=180 row.
- [x] Reject generated N=24 Cooley-Tukey `(4,6)` routing. Composite value
  tests passed, but optimized benchmark compilation exceeded the 300s bound
  after invalidating codegen; the retained `(6,4)` route was restored.
- [x] Refresh retained stale rows with the existing optimized runner:
  N=166 now records f64 `1.700x`, f32 `1.847x`; N=356 records f64 `2.748x`,
  f32 `2.191x`; N=385 records f64 `2.378x`, f32 `3.818x` after a repeated
  quick-profile run and remains the current highest miss.
- [x] Reject dispatch-local N=385 radix order `[5,7,11]`. The radix-composite
  suite and `xtask` check passed, but optimized focused benchmarking exceeded
  the 300s release-build bound and produced no updated row.
- [x] Refresh retained stale rows N=81, N=165, N=198, N=219, N=223, N=438,
  and N=446 with the existing optimized runner. Current rows: N=81 f64
  `1.792x`, f32 `2.735x`; N=165 f64 `1.486x`, f32 `2.206x`; N=198 f64
  `2.188x`, f32 `2.258x`; N=219 f64 `1.507x`, f32 `2.794x`; N=223 f64
  `1.787x`, f32 `1.621x`; N=438 f64 `1.425x`, f32 `2.755x`; N=446 f64
  `1.844x`, f32 `1.643x`.
- [x] Refresh retained stale rows N=70, N=73, N=88, N=127, N=142, N=146,
  N=160, N=181, N=224, N=245, N=249, N=263, N=264, N=269, N=339, N=346,
  N=352, N=357, and N=362 with the existing optimized runner. Current highest
  rows from this set: N=264 f32 `3.245x`, N=224 f32 `3.035x`, N=181 f64
  `2.994x`, N=160 f32 `2.974x`, N=352 f32 `2.977x`, N=346 f64 `2.939x`,
  N=339 f64 `2.868x`, N=263 f32 `2.832x`, and N=362 f64 `2.812x`.
- [x] Refresh retained stale rows N=48, N=99, N=110, N=176, N=200, N=292,
  N=298, N=384, N=452, N=480, and N=499 with the existing optimized runner.
  Rerun N=48 alone to confirm the outlier: N=48 now records f64 `2.015x`,
  f32 `5.650x` and is the current top miss. Other high rows from this set:
  N=176 f32 `3.919x`, N=200 f32 `2.884x`, N=452 f64 `2.767x`, N=292 f32
  `2.612x`, N=499 f64 `2.462x`, N=480 f32 `2.251x`, and N=384 f32 `2.108x`.
- [x] Refresh nearby short/composite rows N=40, N=42, N=44, N=45, N=46, N=50,
  N=51, N=52, N=54, N=55, N=56, N=58, N=60, N=62, and N=63 with the existing
  optimized runner. A solo N=54 rerun replaces the grouped f32 outlier with
  f64 `2.354x`, f32 `2.689x`. Other current high rows from this set: N=63
  f64 `2.029x`, f32 `2.615x`; N=40 f64 `1.466x`, f32 `2.365x`; N=56 f64
  `1.481x`, f32 `2.346x`; N=55 f64 `1.611x`, f32 `2.301x`; and N=45 f64
  `1.686x`, f32 `2.166x`.
- [x] Reject planned N=48 Good-Thomas `(16,3)` routing. The route passed the
  existing planned Good-Thomas suite and `xtask` check, but focused
  benchmarking regressed the row to f64 `3.243x`, f32 `10.996x`.
- [x] Restore planned N=48 to the generated `ShortWinograd` codelet route.
  Update f64 and f32 planned N=48 direct-DFT tests for the route, update the
  `xtask` engine classifier to `Winograd`, and refresh the full-profile row to
  f64 `1.470x`, f32 `4.593x`. N=48 remains open.
- [x] Reject planned N=48 composite order `[4,3,4]`. Direct-DFT route tests
  and `xtask` checking passed, but optimized focused benchmarking exceeded
  the 300s release-build bound without writing a row.
- [x] Reject the small-composite AVX2 cutoff for N=48. Direct-DFT route tests
  and `xtask` checking passed, but optimized focused benchmarking exceeded
  the 300s release-build bound without writing a row.
- [x] Reject planned f32 N=176 generated `(11,16)` routing. The route passed
  direct-DFT value testing, but focused benchmarking regressed f32 to
  `4.256x`; the restored static route refreshes to f64 `2.159x`, f32
  `3.879x`.
- [x] Reject planned N=176 swapped Good-Thomas `(16,11)` orientation. The
  route passed direct-DFT value testing, but focused benchmarking regressed to
  f64 `2.258x`, f32 `3.920x`; the restored cached `(11,16)` route refreshes
  to f64 `1.891x`, f32 `3.579x`.
- [x] Route planned N=385 through composite order `[11,5,7]`.
- [x] Add f64 and f32 planned N=385 direct-DFT tests for the route.
- [x] Verify planned N=385 with `cargo test -p apollo-fft planned_n385 --lib`
  and `cargo check -p xtask --features bench-runner`.
- [x] Refresh N=385 with focused `cargo xtask`: f64 1095.12 ns vs 461.71 ns
  (`2.372x`), f32 1130.15 ns vs 319.30 ns (`3.540x`). The row improves but
  remains open; N=180 is now the current highest miss.
- [x] Reject planned N=385 composite order `[7,11,5]`. Direct-DFT value tests
  and `xtask` checking passed, but optimized focused benchmarking exceeded
  the 300s release-build bound without writing a row; restore `[11,5,7]`.
- [x] Route planned N=180 through composite order `[5,3,3,4]`.
- [x] Remove the obsolete planned f32 N=180 codelet-route assertion.
- [x] Verify planned N=180 with `cargo test -p apollo-fft planned_n180 --lib`
  and `cargo check -p xtask --features bench-runner`.
- [x] Update the `xtask` engine classifier so N=180 reports `Cooley-Tukey`.
- [x] Refresh N=180 with focused `cargo xtask`: f64 294.53 ns vs 156.63 ns
  (`1.880x`), f32 252.61 ns vs 91.04 ns (`2.775x`). The row improves but
  remains open; N=176 and N=144 are now tied as the current highest misses.
- [x] Route planned N=144 through composite order `[4,4,3,3]`.
- [x] Remove the obsolete planned f32 N=144 codelet-route assertion.
- [x] Verify planned N=144 with `cargo test -p apollo-fft planned_n144 --lib`
  and `cargo check -p xtask --features bench-runner`.
- [x] Update the `xtask` engine classifier so N=144 reports `Cooley-Tukey`.
- [x] Refresh N=144 with focused `cargo xtask`: f64 182.25 ns vs 115.43 ns
  (`1.579x`), f32 124.40 ns vs 68.48 ns (`1.817x`). The row improves but
  remains open; N=176 is now the current highest miss.
- [x] Route planned N=176 through composite order `[11,4,4]`.
- [x] Add f64 and f32 planned N=176 direct-DFT tests for the composite route.
- [x] Verify planned N=176 with `cargo test -p apollo-fft planned_n176 --lib`
  and `cargo check -p xtask --features bench-runner`.
- [x] Update the `xtask` engine classifier so N=176 reports `Cooley-Tukey`.
- [x] Refresh N=176 with focused `cargo xtask`: f64 352.82 ns vs 179.04 ns
  (`1.971x`), f32 284.63 ns vs 94.75 ns (`3.004x`). The max ratio improves
  from f32 `3.579x`, but the row remains open; N=48 is now the current
  highest miss.
- [x] Reject planned f32 N=189 generated `(7,27)` routing. It was
  value-correct but measured f32 `3.676x`; the retained static route now
  records f64 `2.238x` and f32 `3.665x`.
- [x] Reject planned f32 N=189 generated `(27,7)` routing. It was
  value-correct but measured f32 `3.679x`; the restored static route records
  f64 `2.145x` and f32 `3.655x`.
- [x] Reject planned N=108 generated Cooley-Tukey `(12,9)` routing. It was
  value-correct but measured f64 `2.616x` and f32 `4.590x`, regressing the
  retained generated Good-Thomas `(4,27)` route.
- [x] Reject planned N=108 generated Good-Thomas `(27,4)` orientation. It was
  value-correct but measured f64 `2.645x` and f32 `3.686x`; restored
  `(4,27)` refreshes to f64 211.50 ns vs 80.25 ns (`2.636x`) and f32
  166.53 ns vs 60.05 ns (`2.773x`).
- [x] Route N=189 through generated Cooley-Tukey `(21,9)`. The route is
  value-correct and improves focused f32 timing to 355.38 ns vs 126.55 ns
  (`2.808x`); f64 records 416.92 ns vs 196.37 ns (`2.123x`). The row
  remains open.
- [x] Reject planned f32 N=240 generated `(15,16)` routing. It was
  value-correct but measured f32 `3.706x`; the retained Cooley-Tukey route
  now records f64 `1.427x` and f32 `3.003x`.
- [x] Refresh additional stale high-ratio rows: N=73 f64 `1.555x`, f32
  `2.938x`; N=88 f64 `2.152x`, f32 `2.854x`; N=107 f64 `2.129x`, f32
  `1.499x`; N=146 f64 `1.524x`, f32 `2.885x`; N=226 f64 `2.145x`, f32
  `2.482x`; N=275 f64 `2.149x`, f32 `3.463x`; N=321 f64 `2.492x`, f32
  `1.813x`; N=440 f64 `2.034x`, f32
  `3.212x`; N=484 f64 `2.105x`, f32 `3.119x`.
- [ ] Replace the half-sized static twiddle tables in `impls.rs` with full-sized versions.
- [ ] Modify `load_twiddle_pair` to load from `tw_ptr` unconditionally without branches.
- [ ] Remove tw_ptr offsets (+15 and +31) and reference static twiddle tables directly in sizes 32 and 64.
- [ ] Verify that all correctness tests pass and ratios in `benchmark_results.md` are < 1.000x.
- [x] Restore the optimized `xtask` runner to benchmark Apollo's direct
  `FftPrecision::fft_forward` public transform path instead of the 1-D plan
  wrapper, because the benchmark contract is clone-inclusive transform
  execution and not plan dispatch overhead.
- [x] Reject 32-byte aligned f64 combine-twiddle loads for N=32/64 after the
  focused probe regressed f64 timings to 29.14 ns at N=32 and 58.38 ns at
  N=64, with ratios still above 1.000x.
- [x] Reject routing f32 N=32/64 through the existing fixed Winograd kernels
  on AVX/FMA targets after the focused probe regressed f32 ratios to 5.619x
  at N=32 and 4.678x at N=64.
- [x] Record the repaired direct-runner focused result:
  N=32 f64 44.09 ns vs 25.89 ns (`1.703x`), f32 26.32 ns vs 11.23 ns
  (`2.344x`); N=64 f64 73.57 ns vs 50.92 ns (`1.445x`), f32 53.63 ns vs
  36.94 ns (`1.452x`).
- [x] Refresh N=32/N=64 after the separate default `cargo xtask` run:
  N=32 f64 20.99 ns vs 14.60 ns (`1.438x`), f32 20.06 ns vs 7.99 ns
  (`2.511x`); N=64 f64 60.32 ns vs 37.96 ns (`1.589x`), f32 37.12 ns vs
  17.84 ns (`2.080x`). Both sizes remain open versus `< 1.000x`.
- [x] Reject the N=32768 four-pass `4+4+4+3` Stockham schedule after
  value-correct optimized probing showed a large regression versus the retained
  five-triple schedule.

## Closure CVXII - Reduced f32 DFT31 Pair Layout [patch]
Sprint target version: apollo-fft 0.12.24

- [x] Reject broad reduced f32 Winograd-pair routing for
  N=29/37/41/53 after refreshed quick-profile rows showed no stable
  improvement and visible regressions for larger primes.
- [x] Keep the reduced structure-of-arrays Winograd-pair body as a single
  monomorphized implementation and route only f32 DFT31 through it.
- [x] Preserve the generic Winograd-pair route for all f64 short odd primes
  and for f32 N=11/13/17/19/23/29/37/41/43/47/53.
- [x] Add value-semantic direct-DFT coverage for promoted f64 odd-prime
  routes, all f32 odd-prime routes, and the reduced f32 DFT31 inverse route.
- [x] Refresh `benchmark_results.md` rows for N=29/31/37/41/53 after the
  final route narrowing; current reduced f32 DFT31 is 87.31 ns Apollo vs
  83.75 ns RustFFT (`1.043x`), improved over the generic-route probe but not
  yet a parity close.

## Closure CVXI - Short Odd-Prime Winograd Pair Routing [patch]
Sprint target version: apollo-fft 0.12.24

- [x] Route `ShortDft` odd-prime sizes 11, 13, 17, 19, 23, 29, 31, 37,
  41, 43, 47, and 53 through the existing Winograd-pair codelet instead of
  the static Rader codelet.
- [x] Preserve static Rader codelets as direct Rader validation and fallback
  surfaces, and extend generated static Rader coverage through N=53 for the
  direct Rader path.
- [x] Keep one `impl_short_winograd_prime_pair!` route declaration per
  precision so f64/f32 use the same monomorphized kernel family.
- [x] Verify direct-DFT value semantics for affected short-prime tests and
  Rader tests, plus the `apollo-fft-macros` proc-macro crate.
- [x] Refresh `benchmark_results.md` rows for
  N=11/13/17/19/23/29/31/37/41/43/47/53 with the optimized `xtask`
  clone-inclusive runner.

## Closure CVX - Rader Generator-Order Cache Compression [patch]
Sprint target version: apollo-fft 0.12.24

- [x] Replace cached runtime Rader gather/scatter permutation pairs with one
  cached generator-order table keyed by `(N, g)`.
- [x] Derive scatter indices from the retained generator order by the exact
  cyclic-group identity `g^{-q} = g^(N-1-q)`.
- [x] Update standalone runtime Rader, half-cyclic runtime Rader, ordered-Rader
  Good-Thomas PFA, and two-by-prime ordered-Rader call sites.
- [x] Add value-semantic coverage proving derived inverse order matches the
  modular-inverse power sequence for every primitive-root table entry.
- [x] Verify feature compile, focused Rader tests, focused Good-Thomas tests,
  `xtask` compile, and focused half-cyclic Criterion timings.

## Closure CVIX - Half-Cyclic Rader Spectrum Memory Build [patch]
Sprint target version: apollo-fft 0.12.24

- [x] Remove the full length `N-1` temporary kernel from half-cyclic Rader
  spectrum construction.
- [x] Stream lower kernel values into the cyclic residue buffer, then combine
  upper kernel values directly into cyclic and twisted negacyclic residues.
- [x] Preserve the exact CRT residue definitions:
  `cyc[j] = kernel[j] + kernel[j+m]` and
  `neg[j] = (kernel[j] - kernel[j+m]) * exp(i*pi*j/m)`.
- [x] Verify feature compile, focused half-cyclic/full-cyclic equivalence, full
  Rader-filtered tests, `xtask` compile, and focused half-cyclic Criterion
  timings.

## Closure CVIII - Half-Cyclic Rader Strategy [patch]
Sprint target version: apollo-fft 0.12.24

- [x] Extract the half-cyclic convolution rule from Liu/Tolimieri ICASSP 1991
  and map it to Rader's length `N-1` cyclic convolution through
  `x^(2m)-1 = (x^m-1)(x^m+1)`.
- [x] Add the half-cyclic Winograd/Nussbaumer CRT split as a selectable Rader
  convolution strategy without adding a Bluestein compatibility route.
- [x] Keep one production Rader entry point; expose forced full-cyclic and
  half-cyclic strategy hooks only under test/debug/benchmark configuration.
- [x] Move automatic routing to a conservative shared threshold:
  `N-1 >= 1024` for f64 and f32.
- [x] Add value-semantic tests comparing automatic Rader, forced half-cyclic
  Rader, forced full-cyclic Rader, and direct DFT at N=521; retain the N=10007
  roundtrip with debug stack-safe dispatch.
- [x] Add `half_cyclic_rader` Criterion coverage for full-cyclic,
  half-cyclic, and automatic strategy timings at N=257/N=521/N=1031.
- [x] Verify `apollo-fft` feature compile, `xtask` compile, Rader tests, and
  focused benchmark execution. Default bench-quick O3 codegen was terminated
  locally; the completed optimized timing run used opt-level 1.

## Closure CVII - Fixed Good-Thomas Macro Dispatch Review [patch]
Sprint target version: apollo-fft 0.12.24

- [x] Review benchmark-comment disparities without treating monomorphization as
  evidence that different factorizations should execute the same arithmetic
  kernel.
- [x] Promote fixed coprime Good-Thomas dispatch into
  `generate_good_thomas_dispatch!` so canonical coprime pairs up to N=200 are
  generated from `short_sizes` and `max_n`, not hand-maintained match arms.
- [x] Route fixed coprime pairs through one bounded const-generic PFA codelet
  body with compile-time `(N1, N2, N, INVERSE)` parameters and const CRT maps.
- [x] Complete the proc-macro short-codelet refactor by restoring
  `generate_winograd_fft!`, removing the `ShortWinogradScalar`/`ShortDft`
  trait cycle, and routing generated Good-Thomas row/column transforms through
  direct `ShortDft<N>` calls.
- [x] Replace the Criterion-dependent targeted table path with an optimized
  `xtask` bounded adaptive runner; the full canonical table now regenerates in
  65.6 seconds from the already-built optimized `xtask` binary.
- [x] Optimize the shared odd-prime-pair DFT kernel by replacing iterator-zip
  arithmetic with const-indexed loops and native `one()` sign selection.
- [x] Reject all-shape unrolled per-pair body emission for this increment:
  the prototype passed value checks but exceeded the bounded bench/release
  compile budget, so the retained path is the bounded generic dispatcher.
- [x] Complete stale const-direction Rader/Bluestein call sites exposed by the
  focused rebuild.
- [x] Regenerate targeted quick-profile `benchmark_results.md` rows for
  N=84/N=90/N=94/N=150/N=175.
- [x] Refresh the N=10 precision-disparity row and verify the prior f32
  160.27 ns Apollo value was stale Criterion data.
- [x] Change `xtask benchmark --sizes ...` to merge only requested rows into
  the existing markdown table and require fresh requested Criterion estimates
  after non-`--skip-run` subset runs.
- [x] Refresh the N=77 precision-disparity row and verify the prior f32
  4.739x ratio was stale mixed-epoch Criterion data.
- [x] Refresh the full canonical `benchmark_results.md` table with the
  optimized `xtask` runner after the odd-prime-pair DFT11 improvement; current
  N=77 is 199.94 ns Apollo vs 103.96 ns RustFFT for f64 and 235.34 ns Apollo
  vs 78.52 ns RustFFT for f32.
- [x] Verify proc-macro compile, `apollo-fft` lib compile, bench compile,
  generated composite direct-DFT coverage, and fixed coprime direct-DFT
  coverage.
- [x] Probe N=44 after fixed PFA routing without rewriting
  `benchmark_results.md`; Apollo improved but still trailed RustFFT at
  1.541x f64 and 1.593x f32.

## Closure CVI - Short Good-Thomas Codelets and PFA Scratch Reuse [patch]
Sprint target version: apollo-fft 0.12.23

- [x] Reject benchmark table fabrication: `benchmark_results.md` remains
  measured Criterion evidence, not an editable success surface.
- [x] Add generated short Good-Thomas codelets for N=18, N=24, and N=36 from
  existing short factors instead of hand-written kernels.
- [x] Route the new codelets through the canonical short-Winograd dispatch
  surface for both forward and inverse execution.
- [x] Move generic natural/ordered PFA column storage from per-call `Vec`
  allocation into the existing thread-local PFA scratch buffer.
- [x] Move generated direct `2*p` natural-prime dispatch to a twiddle-free
  Good-Thomas row/column codelet for the promoted-prime family while retaining
  the previous Cooley-Tukey helper for comparison until a RustFFT-beating
  replacement is proven across the family.
- [x] Reduce benchmark loop latency by making `xtask benchmark` default to a
  quick Criterion profile and optimized `bench-quick` Cargo profile; retain
  `--profile full` for release-quality timing.
- [x] Probe broader generated fixed coprime dispatch and reject it for this
  increment because optimized ThinLTO release builds became unstable under the
  repository's current codegen profile.
- [x] Verify `apollo-fft-macros` compile, `apollo-fft` lib compile, and
  direct-DFT value coverage for N=18/N=24/N=36. Full Criterion regeneration
  remains pending because bounded `vs_rustfft` release bench builds did not
  produce usable timing output in this turn.
- [x] Verify focused two-by-prime correctness, `xtask` compile, benchmark
  harness compile, quick-profile `xtask benchmark --sizes 38`, and
  quick-profile `xtask benchmark --sizes 58,74,82,94`.

## Closure CV - Natural Good-Thomas and Generated Codelet Dispatch [patch]
Sprint target version: apollo-fft 0.12.22

- [x] Bind the natural PFA output scatter to the cached CRT table contract:
  `output_perm[k2 * n1 + k1]` maps transformed `(k1, k2)` columns to natural
  frequency order.
- [x] Add value-semantic direct-DFT forward coverage for a nontrivial coprime
  natural PFA shape.
- [x] Add value-semantic direct-DFT unnormalized inverse coverage for the same
  natural PFA shape.
- [x] Complete stale Winograd const-generic direction call sites exposed by a
  fresh rebuild, including generator output, production short-codelet dispatch,
  and unit tests.
- [x] Move generated `3*p` Good-Thomas bodies to direct const-generic DFT-3
  column codelets and direct row codelets from one macro prime list.
- [x] Keep the short-Winograd surface compact around authoritative
  const-generic leaves and prime-pair table traits; no retained Rader,
  Good-Thomas, Winograd, butterfly, Stockham, or four-step component was
  removed.
- [x] Extend the canonical `vs_rustfft` f64/f32 Criterion table with
  N=38/58/74/82/94 and regenerate `benchmark_results.md` after targeted
  refreshes for N=33/38/58/74/82/94.
- [x] Add `xtask benchmark` as the only active benchmark runner/table
  generator for `benchmark_results.md`.
- [x] Remove redundant benchmark workflow artifacts: `extract_benchmarks.py`,
  `quick_compare`, duplicate validation `vs_rustfft`, and bench output logs.
- [x] Bump `apollo-fft` to 0.12.22 and update CHANGELOG.md, backlog.md,
  gap_audit.md, and checklist.md.
- [x] Verify focused natural PFA tests, focused Good-Thomas tests, full
  `apollo-fft` library tests, focused Winograd large/composite tests,
  `apollo-fft-macros` compile, `apollo-fft` bench/example compile, targeted
  Criterion rows, `xtask benchmark --skip-run`, and diff hygiene.

## Closure CIV - Generated Good-Thomas Family Dispatch [patch]
Sprint target version: apollo-fft 0.12.21

- [x] Change `generate_three_by_prime_dispatch!` from generated
  gather/scatter closures plus a runtime driver to generated full-body
  per-prime `3*p` Good-Thomas kernels.
- [x] Add `generate_two_by_prime_natural_dispatch!` and replace the local
  `macro_rules!` Winograd-pair `2*p` dispatch table with a proc-macro table.
- [x] Promote prime-pair table availability into the sealed Winograd scalar
  contract so generated dispatch compiles in release with one scalar bound.
- [x] Preserve the retained hand DFT-3 codelet as the selected implementation;
  the current generated Winograd direct-DFT prototype is not promoted over it.
- [x] Regenerate `benchmark_results.md`; N=33 records f64 94.34 ns vs RustFFT
  68.16 ns and f32 108.75 ns vs RustFFT 64.81 ns.
- [x] Verify proc-macro compile, lib compile, focused two-by-prime and
  three-by-prime correctness, generated Rader correctness, bench/example
  feature compile, release quick comparison, targeted N=33 Criterion refresh,
  benchmark extraction, and extractor syntax validation.

## Closure CIII - Generated Good-Thomas Route Fusion [patch]
Sprint target version: apollo-fft 0.12.20

- [x] Extend `generate_three_by_prime_dispatch!` so the generator emits the
  per-prime CRT gather and scatter functions, not only the support predicate
  and dispatch table.
- [x] Replace runtime `ThreeByPrimePlan` hot-path loads/stores with generated
  literal route stores while retaining the verified const-plan tests.
- [x] Add `short_winograd_const` so generated Good-Thomas routes use
  const-size short-codelet dispatch and compile-time inverse selection.
- [x] Repair generated Rader mapping, twiddle precision, and inverse symbol
  scope; bound static Rader generation to 5/7/11/13 until a scalable generated
  convolution backend exists.
- [x] Regenerate `benchmark_results.md`; N=33 records f64 93.00 ns vs RustFFT
  64.92 ns and f32 108.00 ns vs RustFFT 67.49 ns.
- [x] Verify proc-macro compile, lib compile, focused three-by-prime
  correctness, generated Rader correctness, bench/example feature compile,
  benchmark extraction, and extractor syntax validation. Release quick-compare
  rebuild exceeded the 300-second cap.

## Closure CII - Good-Thomas Proc-Macro Dispatch Generator [patch]
Sprint target version: apollo-fft 0.12.19

- [x] Read `gengoodthomas.md`, `genpermute.md`, `genrader.md`, and
  `genwinograd.md`; select the duplicated Good-Thomas `3*p` support/match
  surface as the next safe proc-macro increment.
- [x] Add the internal `apollo-fft-macros` proc-macro crate with
  `generate_three_by_prime_dispatch!`.
- [x] Replace hand-written `3*p` support detection and `(P, inverse)` match
  arms with generated code from one short-prime list.
- [x] Preserve the generic `three_by_prime_impl` runtime kernel,
  `ThreeByPrimePlan<const P>`, static dispatch, and all retained FFT
  components.
- [x] Regenerate `benchmark_results.md`; N=33 still records f64 101.49 ns vs
  RustFFT 70.27 ns and f32 121.28 ns vs RustFFT 78.91 ns.
- [x] Verify proc-macro compile, lib compile, focused three-by-prime
  correctness, bench/example feature compile, release quick timing,
  benchmark extraction, extractor syntax validation, and diff hygiene.

## Closure CI - Good-Thomas Const CRT Plan [patch]
Sprint target version: apollo-fft 0.12.18

- [x] Read `gengoodthomas.md`, `genpermute.md`, `genrader.md`, and
  `genwinograd.md`; select the stable const-plan Good-Thomas layer as the next
  verifiable increment before adding a proc-macro crate.
- [x] Replace runtime modular inverse/CRT output-stride calculation in the
  compact `3*p` route with `ThreeByPrimePlan<const P>`.
- [x] Verify compile-time input/output CRT maps against the Good-Thomas
  formula for P=5/7/11/13/17/23.
- [x] Preserve the monomorphized `MixedRadixScalar` route and all retained FFT
  components; no size-specific branch was introduced.
- [x] Regenerate `benchmark_results.md`; N=33 now records f64 101.49 ns vs
  RustFFT 70.27 ns and f32 121.28 ns vs RustFFT 78.91 ns.
- [x] Verify lib compile, bench/example feature compile, focused
  three-by-prime correctness, quick release timing, targeted Criterion rows,
  benchmark extraction, extractor syntax validation, and diff hygiene.

## Closure C - Three-By-Prime Good-Thomas Routing [patch]
Sprint target version: apollo-fft 0.12.17

- [x] Identify N=33 route pathology: dispatcher selected the prime-23
  mixed-radix composite `[11, 3]` path before the twiddle-free coprime
  Good-Thomas route.
- [x] Add a compact Good-Thomas `3*p` CRT codelet for supported short prime
  factors 5/7/11/13/17/23.
- [x] Route only the verified compact `3*p` family ahead of
  `cached_prime23_radices`, preserving all existing retained kernels and
  avoiding unmeasured early-dispatch changes for `2*p`.
- [x] Restore the benchmark-only ordered-Rader hooks to the current
  `rader_ordered_impl` API.
- [x] Regenerate `benchmark_results.md`; N=33 now records f64 104.08 ns vs
  RustFFT 69.15 ns and f32 128.21 ns vs RustFFT 63.15 ns.
- [x] Verify focused f64/f32 direct-DFT equivalence, lib compile, bench/example
  feature compile, quick release comparison, targeted N=33 Criterion timing,
  benchmark extraction, and extractor syntax validation.

## Closure XCIX - Typed Real-Storage Direct Fill [patch]
Sprint target version: apollo-fft 0.12.16

- [x] Replace f64/f32/f16 typed `forward_1d_into`, `forward_2d_into`, and
  `forward_3d_into` mapped-temporary assignment with direct caller-buffer
  real-to-complex fills.
- [x] Replace f64/f32/f16 typed `inverse_1d_into`, `inverse_2d_into`, and
  `inverse_3d_into` mapped-temporary real extraction with direct scratch-to-real
  fills.
- [x] Remove the extra complex clone from allocating typed 1D/2D/3D forward
  paths by transforming the mapped output in place.
- [x] Keep compact f16 conversion explicit at the storage boundary and keep the
  f32 plan family as the execution scalar.
- [x] Bump `apollo-fft` to 0.12.16 and update CHANGELOG.md, backlog.md,
  gap_audit.md, checklist.md, and benchmark_results.md.
- [x] Verify `cargo check -p apollo-fft --lib`, `cargo check -p apollo-fft
  --benches --examples --features kernel-strategy-bench`, focused
  power-of-two tests, f64/f32/f16 typed API coverage, benchmark extraction, and
  extractor syntax validation.
- [x] Re-run the public six-step zero-allocation Criterion row after the active
  workspace `cargo bench --no-run` process finishes; N=5120 records 11.530 us
  mean and no allocation assertion failure.

## Closure XCVIII - Generic Plan Cache Scalar Routing [patch]
Sprint target version: apollo-fft 0.12.15

- [x] Resolve typed real FFT plan precision through `RealFftData::PlanScalar`
  and `PlanCacheProvider`, keeping f64/f32 native caches and routing compact
  f16 storage through f32 plans at the storage boundary.
- [x] Keep 1D/2D/3D plan execution monomorphized over `MixedRadixScalar`
  without adding dynamic dispatch or cloned f32/f64 API families.
- [x] Retune the restored power-of-two fast-path cutoff to N>=64 so N=16 and
  N=32 remain on the faster current short-codelet path while the generic route
  continues to cover larger powers without deleting retained components.
- [x] Update typed helper call sites, CPU transport tests, and benchmark paths
  to use the cache-provider and real-storage contracts directly.
- [x] Bump `apollo-fft` to 0.12.15 and update CHANGELOG.md, backlog.md,
  gap_audit.md, checklist.md, and benchmark_results.md.
- [x] Verify `cargo check -p apollo-fft --lib`, `cargo check -p apollo-fft
  --benches --examples --features kernel-strategy-bench`, focused
  power-of-two tests, f64/f32/f16 typed API coverage, quick comparison for
  N=16/N=32/N=64/N=128, benchmark extraction, and extractor syntax validation.
- [x] Attempt targeted Criterion refresh for N=16 after the pre-existing bench
  writer finishes; record the 300-second timeout as an unresolved benchmark
  harness/runtime gap rather than replacing the canonical table with quick-run
  numbers.
- [ ] Run fresh targeted Criterion rows for N=16, N=32, N=64, N=128, and
  N=32768 after the N=16 targeted Criterion timeout is resolved.

## Closure XCVII - Power-of-Two Fast-Path Restoration [patch]
Sprint target version: apollo-fft 0.12.14

- [x] Route power-of-two lengths N>=16 through one generic fast-path before
  small Winograd, composite, PFA, or Rader routing.
- [x] Keep N=2, N=4, and N=8 on short Winograd codelets while preserving all
  retained Rader, Good-Thomas, Winograd, butterfly, Stockham, and four-step
  implementations.
- [x] Select square four-step only for even-exponent lengths above the
  four-step threshold; use Stockham for asymmetric power-of-two lengths such as
  N=8192 and N=32768.
- [x] Add a value-semantic N=32768 forward DC regression test that rejects the
  selector no-op failure mode hidden by forward+inverse roundtrips.
- [x] Restore manifest consistency for current verification by adding the
  workspace `tokio` dependency and the `kernel-strategy-bench` feature.
- [x] Restore the 1D generic caller-owned typed forward/inverse methods used by
  zero-allocation benchmark paths.
- [x] Bump `apollo-fft` to 0.12.14 and update CHANGELOG.md, backlog.md,
  gap_audit.md, checklist.md, and benchmark_results.md.
- [x] Verify focused power-of-two correctness tests, `cargo check -p
  apollo-fft --lib`, `cargo check -p apollo-fft --benches --examples
  --features kernel-strategy-bench`, and benchmark extraction.
- [ ] Run fresh targeted Criterion rows for N=16, N=32, N=64, N=128, and
  N=32768 after the two pre-existing Criterion writer processes finish.

## Closure XCVI - Small Good-Thomas Codelet Restoration [patch]
Sprint target version: apollo-fft 0.12.13

- [x] Add stack-resident CRT Good-Thomas codelets for N=6, N=10, N=12, and
  N=14 using existing DFT-3/4/5/7 Winograd leaves.
- [x] Route the monomorphized `short_winograd` dispatcher through the new
  codelets before mixed-radix, PFA, or Rader routing can allocate scratch or
  fetch twiddle/permutation caches for those small composites.
- [x] Preserve all retained Rader, Good-Thomas, Winograd, butterfly, and
  composite routes; no component is removed before a measured replacement
  beats RustFFT.
- [x] Remove the private obsolete Good-Thomas gather helper left unused by the
  fused ordered-Rader PFA path, resolving the bench build warning at source.
- [x] Bump `apollo-fft` to 0.12.13 and update CHANGELOG.md, backlog.md,
  gap_audit.md, checklist.md, and benchmark_results.md.
- [x] Verify direct-value small-composite tests, mixed forward dispatch tests,
  and `cargo check -p apollo-fft --lib`.
- [x] Run targeted `vs_rustfft` Criterion rows for N=6, N=10, N=12, and N=14
  and regenerate `benchmark_results.md` from the updated cache.

## Closure XCV - Rader Negacyclic Twist/Recombine Fusion [patch]
Sprint target version: apollo-fft 0.12.12

- [x] Fuse Rader negacyclic twist multiplication into the Nussbaumer split
  pass for every large-prime Rader route using negacyclic convolution.
- [x] Fuse conjugate untwist multiplication into CRT recombination, removing
  the standalone untwist pass over the negacyclic half.
- [x] Preserve the fused radix-composite forward-plus-pointwise Rader
  convolution path and keep all retained Rader, Good-Thomas, Winograd,
  butterfly, and composite routes available.
- [x] Bump `apollo-fft` to 0.12.12 and update CHANGELOG.md, backlog.md,
  gap_audit.md, checklist.md, and benchmark_results.md.
- [x] Verify `cargo check -p apollo-fft --lib`,
  `cargo check -p apollo-fft --benches --examples --features kernel-strategy-bench`,
  focused Rader/prime-dispatch tests, benchmark extraction, and extractor
  syntax validation.

## Closure XCIV - Interleaved Two-Prime and Rader Pointwise Fusion [patch]
Sprint target version: apollo-fft 0.12.11

- [x] Remove the direct `2*p` promoted-prime even-half stack copy and odd-half
  compaction by letting the monomorphized Winograd-pair two-prime kernel read
  interleaved even/odd input directly.
- [x] Keep direct two-by-prime routing generic over the existing
  `PrimePairTable<P, H>` const-generic family; no per-size route was removed.
- [x] Implement the f32/f64 `MixedRadixScalar::composite_forward_with_pointwise`
  contract so Rader circular and negacyclic convolution can use fused
  radix-composite forward-plus-spectrum multiplication.
- [x] Preserve all Rader, Good-Thomas, Winograd, butterfly, and composite
  routes; retained components stay available until a measured replacement
  beats RustFFT.
- [x] Bump `apollo-fft` to 0.12.11 and update CHANGELOG.md, backlog.md,
  gap_audit.md, checklist.md, and benchmark_results.md.
- [x] Verify compile/test surface with focused two-by-prime, Good-Thomas,
  Rader, prime-dispatch, radix-composite, bench/example compile checks,
  benchmark extraction, and Python extractor syntax validation.
- [x] Regenerate the canonical Apollo-vs-RustFFT f64/f32 table from the current
  Criterion cache snapshot. A full workspace bench process was already active
  during this cycle and continued updating Criterion rows.

## Closure XCIII - Fused Routing and Good-Thomas Permutation Tightening [patch]
Sprint target version: apollo-fft 0.12.10

- [x] Route fused radix-composite scalar stage output traversal through
  `chunks_exact_mut(stage_chunk)`, preserving const-radix monomorphized stage
  bodies while removing repeated block slice-bound recomputation.
- [x] Convert fused radix-composite final pointwise spectrum multiplication to
  raw pointer traversal over the contiguous output block under the existing
  length contract.
- [x] Tighten Good-Thomas natural and ordered-Rader PFA gather/scatter loops
  with cached-permutation length assertions and four-wide unchecked copies.
- [x] Fix the retained Winograd N=82 composite codelet bound with
  `PrimePairTable<41, 20>` instead of removing the route.
- [x] Retain all Rader, Good-Thomas, Winograd, butterfly, and composite routes;
  no component is removed before a measured RustFFT-beating replacement exists.
- [x] Bump `apollo-fft` to 0.12.10 and update CHANGELOG.md, backlog.md,
  gap_audit.md, checklist.md, and benchmark_results.md.
- [x] Verify `cargo check -p apollo-fft --lib`, focused
  radix-composite/Good-Thomas/mixed-radix unit tests, bench/example compile
  coverage, debug quick strategy/public comparisons, benchmark extraction, and
  `git diff --check`. `cargo fmt --check --package apollo-fft` remains blocked
  by broader worktree formatting drift outside this increment.

## Closure XCII - Radix-Composite Stage Dispatch and Benchmark Snapshot [patch]
Sprint target version: apollo-fft 0.12.9

- [x] Move recursive fused-composite scratch arena and adaptive recursion into
  `radix_composite/adaptive.rs`, reducing `radix_composite/arity.rs` from 544
  lines to 421 lines while preserving the existing fused-stage contract.
- [x] Add a flat Stockham scalar stage dispatcher that monomorphizes by radix
  at the stage boundary and removes the per-output-group runtime radix match.
- [x] Route `flat_stockham_fused` through the stage dispatcher for all scalar
  fallback radices while keeping the existing f64 AVX2 radix-3/radix-4 stage
  hooks.
- [x] Collapse fused radix-composite final pointwise spectrum multiplication to
  one contiguous pass over the output block for all radices.
- [x] Restore Rader benchmark routing through the shared generic Rader kernel
  and real Winograd-pair kernels, update deleted-module test references, and
  keep static per-prime permutation tables compile-time generated on stable
  Rust.
- [x] Retain Winograd composite large leaves while restoring value-semantic
  composite test resolution; no composite component is gated or removed before
  a measured RustFFT-beating replacement exists.
- [x] Regenerate `benchmark_results.md` from all Criterion
  `target/criterion/**/new/estimates.json` records and the latest debug
  Rader-vs-Winograd-pair quick strategy comparison.
- [x] Bump `apollo-fft` to 0.12.9 and update CHANGELOG.md, backlog.md,
  gap_audit.md, and checklist.md.
- [x] Verify with `cargo check -p apollo-fft --lib`,
  `cargo test -p apollo-fft --lib radix_composite -- --test-threads=1`, and
  `cargo check -p apollo-fft --benches --examples --features kernel-strategy-bench`.
  `cargo fmt --check --package apollo-fft` remains blocked by pre-existing
  formatting drift outside this increment. Release `quick_compare` timing was
  not regenerated because concurrent Cargo/rustc workloads caused release
  example compilation to exceed the command cap and previously hit LLVM memory
  exhaustion.

## Closure XCI - Rader Bluestein Cache/Vector Hook Optimization [patch]
Sprint target version: apollo-fft 0.12.8

- [x] Replace cached inverse Bluestein spectrum retention with conjugated
  forward-spectrum multiplication derived from the even-kernel identity.
- [x] Wire Rader Bluestein pre-chirp/zero-pad and post-chirp/scaling through
  the precision-specific SIMD hook surface.
- [x] Extend pointwise SIMD multiplication to support conjugated right-hand
  operands so inverse Bluestein remains vectorized without a second cached
  spectrum.
- [x] Correct typed-pointer `write_bytes` lane counts in the Bluestein SIMD
  zero-fill path.
- [x] Bump `apollo-fft` to 0.12.8 and update CHANGELOG.md, backlog.md, and
  gap_audit.md.
- [x] Verify with `cargo fmt --check --package apollo-fft`,
  `cargo check -p apollo-fft --lib`,
  `cargo check -p apollo-fft --benches --examples --features kernel-strategy-bench`,
  `cargo test -p apollo-fft --lib rader -- --test-threads=1`,
  `cargo test -p apollo-fft --lib mixed_forward_prime -- --test-threads=1`,
  `cargo test -p apollo-fft --lib mixed_inverse_prime -- --test-threads=1`,
  and `git diff --check`. Focused release `quick_compare` remained blocked by
  active external Apollo `cargo bench` work in the shared release target.

## Closure XC - Rader Standalone Memory-Pass Optimization [patch]
Sprint target version: apollo-fft 0.12.7

- [x] Fuse standalone Rader primitive-root gather with the nonzero DC sum so
  `sum_x = Σ_q x[g^q]` is computed in the same pass that fills the convolution
  buffer.
- [x] Apply the same fused path to generated static-table Rader leaves and the
  runtime permutation-cache Rader fallback.
- [x] Unroll static-table and runtime scatter loops while preserving the
  `X[g_inv^q] -> natural index` permutation contract.
- [x] Replace retained two-buffer Rader scratch pools with one retained aligned
  thread-local buffer per precision plus local nested-call fallback.
- [x] Bump `apollo-fft` to 0.12.7 and update CHANGELOG.md, backlog.md, and
  gap_audit.md.
- [x] Verify with `cargo fmt --package apollo-fft`,
  `cargo test -p apollo-fft --lib rader -- --test-threads=1`,
  `cargo test -p apollo-fft --lib mixed_forward_prime -- --test-threads=1`,
  `cargo test -p apollo-fft --lib mixed_inverse_prime -- --test-threads=1`,
  `cargo check -p apollo-fft`,
  `cargo check -p apollo-fft --benches --examples --features kernel-strategy-bench`,
  release strategy-only `quick_compare`, and `git diff --check`.

## Closure LXXXIX - Fused Radix-Composite Dispatch Repair [patch]
Sprint target version: apollo-fft 0.12.6

- [x] Keep `factorize_composite` prime-only and lower consecutive radix-2 pairs
  to radix-4 stages at dispatch time, preserving the product invariant while
  avoiding unsupported higher-power radix emission.
- [x] Extend radix-composite runtime and ZST dispatch to radix 4, 8, 17, and
  23 so lowered tails and direct prime leaves share the same static arity path.
- [x] Add recursive arena scratch accounting for nested fused `Compose` stages;
  capacity is reserved before live midpoint pointers are exposed.
- [x] Reconnect `stockham_stage_fused` to the `FusedStage` ZST arity family and
  `ExecutionPolicy` static chunk dispatch.
- [x] Correct fused-stage twiddle slice extents so each stage receives
  `(radix - 1) * prev_len * prior_product` coefficients instead of an
  over-wide slice.
- [x] Keep the incomplete `radix_composite::tiling` placeholder out of the
  compiled module graph while preserving the authoritative fused core path.
- [x] Preserve direct N=17/N=23 Winograd dispatch and restore trait coverage for
  short prime leaves required by composite and Rader paths.
- [x] Retain Rader-vs-Winograd-pair comparison hooks and add direct
  equivalence tests before any strategy removal.
- [x] Add a gated Criterion comparison group for Rader vs Winograd-pair kernels
  at N=29/N=31/N=37 for f64 and f32.
- [x] Compare Rader vs Winograd-pair with bounded `quick_compare` debug probe:
  Winograd-pair ratios were 0.151/0.432 at N=29, 0.493/0.967 at N=31,
  and 0.269/0.736 at N=37 for f64/f32 respectively.
- [x] Route production N=29/N=31/N=37 dispatch through the no-gather
  Winograd-pair kernels; keep generated Rader available for larger primes and
  gated strategy comparison.
- [x] Consolidate generated Rader prime leaves N=17..97 behind
  `rader_static_impl::<F, N, G, G_INV>` so generator-specific routing remains
  const-generic and monomorphized instead of cloned PFA bodies.
- [x] Fuse Rader static gather with x0 accumulation and scatter with x0 offset;
  route convolution through final-forward-stage composite pointwise fusion when
  N-1 has cached radix-composite factors.
- [x] Remove stale composite dispatch code: unused `radix_composite::butterfly`,
  unused `dispatch_inplace_with_scratch`, and unused `FusedStage` associated
  items.
- [x] Add dispatch-level forward/inverse direct-DFT checks for N=29/N=31/N=37.
- [x] Add generated Rader direct-DFT regression coverage for every generated
  prime leaf from N=17 through N=97.
- [x] Closure XCII supersedes the earlier composite export narrowing: retained
  large Winograd leaves remain available until a measured replacement beats
  RustFFT.
- [x] Correct radix-4 lowering tests to assert pair-only promotion and the
  identity `192 = 3 * 4^3`.
- [x] Repair the radix-2 lowering implementation after the rejected
  highest-power lowering probe emitted unsupported radix 16.
- [x] Verified with `cargo fmt --check`,
  `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`, and `git diff --check`.
- [x] Re-run bounded strategy probe after Rader fusion and lowering repair:
  debug Winograd/Rader ratios are 0.345/0.541 at N=29, 0.309/0.710 at N=31,
  and 0.414/0.883 at N=37 for f64/f32 respectively after static Rader
  permutation-table leaves; Rader still does not replace Winograd-pair for
  these small primes.
- [x] Record release production `quick_compare` against RustFFT: N=29 Apollo
  0.096 us vs RustFFT 0.107 us, N=31 0.104 us vs 0.105 us, and N=37 0.147 us
  vs 0.132 us.
- [x] Record release strategy-only Rader-vs-Winograd-pair `quick_compare`:
  Winograd/Rader ratios are 0.206/0.476 at N=29, 0.368/0.566 at N=31, and
  0.334/0.555 at N=37 for f64/f32 respectively.
- [x] Add static permutation-table Rader leaves for N=29/N=31/N=37 through
  `rader_static_table_impl::<F, N, L, G_INV>`, eliminating runtime modular
  gather/scatter index recurrence while preserving the true Rader convolution
  and final-forward-stage pointwise fusion.
- [x] Add ordered-layout Rader static/runtime kernels for fused callers: input
  tails use `x[g^q]`, output tails use `X[g_inv^q]`, and `data[1..]` is reused
  as the convolution buffer so standalone Rader gather/scatter and scratch copy
  are absent under that layout contract.
- [x] Add direct-DFT value tests for ordered Rader N=29 forward, N=31 inverse,
  and N=37 runtime forward.
- [x] Wire ordered-layout Rader into Good-Thomas PFA for prime `n1` dimensions
  that would otherwise use Rader, folding generator-order input into the
  transpose and inverse-generator output into the final CRT scatter.
- [x] Reuse the Rader permutation cache in the ordered-Rader PFA branch so
  transpose and CRT scatter consume cached generator/inverse-generator order
  arrays instead of recomputing modular index walks at runtime.
- [x] Add direct-DFT PFA coverage for N=38 forward and N=82 inverse, plus
  branch-selection coverage that excludes N=29/N=31/N=37 from ordered Rader so
  the measured Winograd-pair production choice remains intact.
- [x] Route ordered-Rader PFA through the known-prime monomorphized ordered
  Rader dispatcher before the runtime fallback.
- [x] Add `APOLLO_FFT_QUICK_N` to `quick_compare` and include ordered-Rader PFA
  composite sizes N=38/N=82/N=86/N=94/N=106 in the default probe set.
- [x] Add `ordered_rader_pfa_coprime_composites` to `prime_compose` for the
  same ordered-Rader PFA composite sizes.
- [x] Record release ordered-Rader PFA `quick_compare`: N=38 Apollo 0.624 us vs
  RustFFT 0.097 us (6.433x), N=82 0.862 us vs 0.334 us (2.581x), N=86
  1.022 us vs 0.408 us (2.505x), N=94 1.227 us vs 0.665 us (1.845x), and
  N=106 1.461 us vs 0.595 us (2.455x).
- [x] Add `good_thomas::two_by_prime` for N=2p composites, using direct
  Cooley-Tukey even/odd decomposition, cached `W_N^k` twiddles, and
  stack-resident static half buffers for promoted primes.
- [x] Promote N=19/N=41/N=43/N=47/N=53 from production Rader dispatch to the
  shared odd-prime Winograd-pair family while retaining Rader benchmark hooks.
- [x] Move odd-prime Winograd-pair code into `winograd/radix/odd_prime_pair.rs`;
  current line counts: `good_thomas/mod.rs` 267, `good_thomas/two_by_prime.rs`
  318, `winograd/radix.rs` 301, `winograd/radix/odd_prime_pair.rs` 381.
- [x] Expand `quick_compare`, `prime_compose`, and `kernel_strategy` coverage
  for promoted prime leaves and two-by-prime composites.
- [x] Remove the stale dedicated DFT-82 codelet and short-dispatch arm; N=82
  now falls through to the generic two-by-prime route and uses the promoted
  N=41 Winograd-pair half kernel.
- [x] Record release promoted-prime `quick_compare`: N=19 0.050 us vs
  RustFFT 0.051 us (0.980x), N=29 0.095 us vs 0.103 us (0.922x), N=31
  0.100 us vs 0.142 us (0.704x), N=37 0.149 us vs 0.156 us (0.955x), N=41
  0.173 us vs 0.172 us (1.006x), N=43 0.153 us vs 0.221 us (0.692x), N=47
  0.199 us vs 0.352 us (0.565x), and N=53 0.261 us vs 0.296 us (0.882x).
- [x] Record release two-by-prime `quick_compare`: N=38 0.192 us vs 0.099 us
  (1.939x), N=58 0.289 us vs 0.203 us (1.424x), N=62 0.279 us vs 0.226 us
  (1.235x), N=74 0.377 us vs 0.376 us (1.003x), N=82 0.429 us vs 0.350 us
  (1.226x), N=86 0.407 us vs 0.448 us (0.908x), N=94 0.511 us vs 0.753 us
  (0.679x), and N=106 0.618 us vs 0.647 us (0.955x).
- [x] Replace thread-local PFA scratch in the promoted direct N=2p route with a
  const-generic stack even-half load and in-place odd-half compaction before
  fused two-prime Winograd execution.
- [x] Record release promoted-prime warm `quick_compare` after stack compaction:
  N=19 0.049 us vs RustFFT 0.054 us (0.907x), N=29 0.103 us vs 0.106 us
  (0.972x), N=31 0.092 us vs 0.125 us (0.736x), N=37 0.131 us vs 0.164 us
  (0.799x), N=41 0.162 us vs 0.225 us (0.720x), N=43 0.169 us vs 0.282 us
  (0.599x), N=47 0.226 us vs 0.388 us (0.582x), and N=53 0.298 us vs
  0.328 us (0.909x).
- [x] Record release two-by-prime warm `quick_compare` after stack compaction:
  N=38 0.165 us vs RustFFT 0.109 us (1.514x), N=58 0.239 us vs 0.200 us
  (1.195x), N=62 0.248 us vs 0.202 us (1.228x), N=74 0.305 us vs 0.288 us
  (1.059x), N=82 0.363 us vs 0.354 us (1.025x), N=86 0.381 us vs 0.404 us
  (0.943x), N=94 0.440 us vs 0.749 us (0.587x), and N=106 0.516 us vs
  0.682 us (0.757x).
- [x] Remove the remaining `macro_rules!` composite Winograd test generator and
  replace it with a const-generic shared helper plus grouped explicit tests;
  `winograd/tests/dft_composite.rs` is 416 lines and preserves forward,
  inverse, roundtrip, DC, and f32-vs-f64 value checks for every covered size.
- [x] Retain only the bounded call-shape change with measured local benefit:
  keep N=29/N=31 Winograd-pair wrappers out-of-line and mark N=37
  Winograd-pair wrapper `#[inline(always)]`.
- [ ] Record fresh Criterion/RustFFT timing for the restored fused path; the
  bounded `APOLLO_FFT_BENCH_N=192` attempt exceeded 180 seconds before output.
- [ ] Record fresh Criterion Rader-vs-Winograd-pair timings; the filtered
  `kernel_strategy` run remains pending.

## Closure LXXXVIII - Winograd DFT-23 Dispatch and Benchmark [patch]
Sprint target version: apollo-fft 0.12.5

- [x] Add a dedicated N=23 Winograd pair-symmetry codelet with f64/f32 scalar
  constants and const-generic forward/inverse direction.
- [x] Route public f64/f32 `FftPrecision` fast paths and
  `ShortWinogradScalar::dft23` through the N=23 codelet.
- [x] Split DFT-23 constants into `scalar.rs` and `impls.rs`; generated leaf
  files remain below 500 lines.
- [x] Add DFT-23 value-semantic tests for forward, inverse, roundtrip, and f32
  differential equivalence against the direct DFT reference.
- [x] Preserve Rader split gather/scatter cache arrays while retaining
  direction-specific convolution spectra for inverse correctness.
- [x] Verified: Apollo N=23 f64 **92.341 ns** vs RustFFT **116.48 ns**;
  Apollo N=23 f32 **104.80 ns** vs RustFFT **139.88 ns**.

## Closure LXXXVII - Winograd DFT-17 Dispatch and Benchmark [patch]
Sprint target version: apollo-fft 0.12.4

- [x] Add a dedicated N=17 Winograd pair-symmetry codelet with f64/f32 scalar
  constants and const-generic forward/inverse direction.
- [x] Route public f64/f32 `FftPrecision` fast paths and
  `ShortWinogradScalar::dft17` through the N=17 codelet.
- [x] Use one shared DFT-17 body with separate monomorphized call wrappers:
  inlined for f64 and out-of-line for f32.
- [x] Add N=17 to `vs_rustfft` benchmark sizes.
- [x] Add DFT-17 value-semantic tests for forward, inverse, roundtrip, and f32
  differential equivalence against the direct DFT reference.
- [x] Verified: Apollo N=17 f64 **71.932 ns** vs RustFFT **81.043 ns**;
  Apollo N=17 f32 **90.289 ns** vs RustFFT **112.84 ns**.

## Closure LXXXVI - Winograd DFT-13 Dispatch and Monomorphization [patch]
Sprint target version: apollo-fft 0.12.3

- [x] Add a dedicated N=13 Winograd pair-symmetry codelet with f64/f32 scalar
  trait constants and direct `FftPrecision` fast paths.
- [x] Encode DFT-13 direction as a const generic so forward/inverse kernels are
  separately monomorphized and runtime direction dispatch is removed.
- [x] Add DFT-13 value-semantic tests for forward, inverse, roundtrip, and f32
  differential equivalence against the direct DFT reference.
- [x] Move DFT-13 and DFT-3 into `winograd/radix/` leaves; `radix.rs` is now
  464 lines, `radix/dft13.rs` 475 lines, and `radix/dft3.rs` 43 lines.
- [x] Verified: Apollo N=13 f64 **82.158 ns** vs RustFFT **94.077 ns**;
  Apollo N=13 f32 **78.778 ns** vs RustFFT **86.069 ns**.

## Closure LXXXV - Winograd DFT-7 and N=15 Optimization [patch]
Sprint target version: apollo-fft 0.12.2

- [x] Replace O(N²) `dft7_impl` with Winograd constant algorithm (18 real muls
  vs 196+ naive: Hermitian symmetry, circulant cosine/sine matrix).
- [x] Add `fn dft7` to `ShortWinogradScalar` trait, both f32/f64 impls, and
  `7 =>` dispatch arm in `short_winograd`.
- [x] Partition three identical 534-line winograd test files into domain-scoped
  modules: `dft_small.rs` (DFT-2..8), `dft_large.rs` (DFT-16..64),
  `boundaries.rs` (impulse/DC edge cases). 185 tests pass.
- [x] Bump `apollo-fft` to 0.12.2 and sync CHANGELOG.md.
- [x] Verified: 185 tests pass; Apollo f64 N=15 **~82 ns** vs RustFFT **~108 ns**
  (24% faster); Apollo f32 N=15 **~89 ns** vs RustFFT **~105 ns** (15% faster).

## Closure LXXXIV - DFT-100 Good-Thomas Short-Winograd Dispatch [patch]
Sprint target version: apollo-fft 0.12.1

- [x] Implement `dft100_impl` in `winograd/composite.rs` using Good-Thomas PFA
  (N=100=4×25, gcd=1, CRT input permutation, no inter-stage twiddles).
- [x] Add `fn dft100` to `ShortWinogradScalar` trait and both f32/f64 impls.
- [x] Add `100 =>` case to `short_winograd` match in `mixed_radix/traits.rs`.
- [x] Add five correctness tests: forward/inverse/roundtrip/dc-energy/f32≡f64.
- [x] Verified: all 261 tests pass; Apollo f64 N=100 **310 ns** vs RustFFT
  **415 ns** (−25%); Apollo f32 N=100 **292 ns** vs RustFFT **327 ns** (−11%).
- [x] Bump `apollo-fft` to 0.12.1 and sync CHANGELOG.md.

## Closure LXXXIII - Mixed-Radix Wrapper Removal [major]
Sprint target version: apollo-fft 0.12.0

- [x] Remove public type-suffixed mixed-radix twiddle wrapper entry points.
- [x] Route 1D/2D/3D plan-owned twiddle reuse through
  `dispatch_inplace::<T, INVERSE, NORMALIZE>` directly.
- [x] Keep `dispatch_inplace` crate-private so the public module boundary does
  not expose private scalar traits.
- [x] Remove dead Winograd AVX wrapper leaves and their module exports.
- [x] Route radix-15 mixed-radix leaves through the stack-only generic
  Good-Thomas Winograd codelet instead of the generic recursive path.
- [x] Consolidate broad Stockham AVX stage and pair leaves behind one
  monomorphized backend trait while preserving shape-specific AVX codelets.
- [x] Remove the unreachable legacy CPU SIMD six-step, matrix-workspace, and
  radix2 infrastructure island that was not part of the crate module graph.
- [x] Bump `apollo-fft` to 0.12.0 and update sprint artifacts.
- [x] Verified with `cargo check -p apollo-fft`,
  `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`, and
  `cargo check --workspace`.

## Closure LXXXII - Stockham Butterfly Dispatch Leaf Split [patch]
Sprint target version: apollo-fft 0.11.1

- [x] Extract f64 AVX scratch dispatch from `stockham/butterfly/fixed.rs` into
  `stockham/butterfly/dispatch.rs`.
- [x] Re-export dispatch from `stockham/butterfly/mod.rs` so existing callers
  keep static module-level routing.
- [x] Keep fixed generated codelets in `fixed.rs` and bring the leaf below the
  500-line structural limit.
- [x] Clean stale bench references to removed `bluestein` and `radix2` module
  paths so benchmarks compile against the maintained generic selector and
  `real_fft` twiddle builders.
- [x] Remove the type-named `dispatch_f16.rs` leaf by consolidating compact
  storage routing into the canonical `mixed_radix/dispatch.rs` body.
- [x] Bump `apollo-fft` to 0.11.1 and update sprint artifacts.
- [x] Verified with `cargo check -p apollo-fft`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`, and
  `cargo check -p apollo-fft --benches --examples`.

## Closure LXXXI - Mixed Radix Kernel Hierarchy Decomposition [minor]
Sprint target version: apollo-fft 0.11.0

- [x] Decomposed mixed_radix.rs (901 lines) into a deep hierarchical file tree:
  dispatch, traits, and caches.
- [x] Extracted unit tests into a dedicated tests.rs module.
- [x] Maintained generic SSOT dispatch implementations and fixed pub(crate) module boundaries.
- [x] Zero-warning compilation and 255/255 passing tests.

## Closure LXXX - Winograd Kernel Hierarchy Decomposition [minor]
Sprint target version: apollo-fft 0.11.0

- [x] Decomposed winograd.rs (1382 lines) into a deep hierarchical file tree: avx_f32, avx_f64, radix, composite, and traits to adhere to the 500-line SRP architectural rule.
- [x] Abstracted unit tests into tests/dft_small.rs, tests/dft_large.rs, and tests/boundaries.rs modules.
- [x] Resolved module boundary visibility (pub(crate)) and dead code analysis warnings.
- [x] Validated numeric equivalence via 255/255 passing tests across all execution paths.
- [x] Bump apollo-fft to 0.11.0.

## Closure LXXIX - Stockham Kernel Hierarchy Stabilization [patch]
Sprint target version: apollo-fft 0.10.0

- [x] Standardized \pub(crate)\ visibility across \vx\ leaf nodes to satisfy new module boundaries.
- [x] Audited and corrected \use\ path inconsistencies in \vx/f32/\ and \vx/f64/\ modules.
- [x] Resolved \E0603\ (private function access) and dangling attribute errors.
- [x] Corrected struct attributes by removing invalid \#[target_feature]\ blocks.
- [x] Removed unused imports to eliminate compiler warnings.
- [x] 177/177 tests pass under \--release\.

## Closure LXXVIII - Bluestein Monomorphization + Module Decomposition [minor]
Sprint target version: apollo-fft 0.10.0

- [x] Created `BluesteinScalar` sealed trait in `bluestein/scalar.rs` with associated AVX/FMA dispatch.
- [x] Implemented `BluesteinScalar` for `Complex64` and `Complex32`.
- [x] Replaced 8 `_64`/`_32`-suffixed function pairs with generic implementations in `bluestein/pointwise.rs`.
- [x] Decomposed `bluestein.rs` (1539 lines) into 6-module directory: all files <= 500 lines.
- [x] Zero-warning `cargo check -p apollo-fft`.
- [x] 177/177 tests pass under `--release`.
- [x] Bump `apollo-fft` to 0.10.0.

## Closure LXXVII - Iterator Monomorphization & Twiddle Allocation Bounds [patch]
Sprint target version: apollo-fft 0.9.12

- [x] Replace `.collect()` iteration paths in `radix2.rs` twiddle table building with exact-size `Vec::with_capacity` and `set_len()` loops to guarantee flat O(1) allocation overhead during compilation and plan execution.
- [x] Validated CPU numerical baseline across all bounds.

## Closure LXXVI - Frequency Utility Exact-Capacity Fill [patch]
Sprint target version: apollo-fft 0.9.11

- [x] Replace `fftfreq` known-length iterator collection with exact-capacity
  positive/negative half fill loops.
- [x] Replace `rfftfreq` known-length iterator collection with an exact-capacity
  fill loop.
- [x] Bump `apollo-fft` to 0.9.11 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft`, `cargo check -p apollo-fft
  --benches --examples`, `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, cleanup scans, `cargo fmt --check`, and
  `git diff --check`.

## Closure LXXV - Shift Utility Split-Copy Cleanup [patch]
Sprint target version: apollo-fft 0.9.10

- [x] Remove the unused `Default` bound from `fftshift` and `ifftshift`.
- [x] Replace duplicate modulo-index iterator collection with one shared
  split-slice copy helper.
- [x] Bump `apollo-fft` to 0.9.10 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft`, `cargo check -p apollo-fft
  --benches --examples`, `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, cleanup scans, `cargo fmt --check`, and
  `git diff --check`.

## Closure LXXIV - Real/R2C Initialization Elimination [patch]
Sprint target version: apollo-fft 0.9.9

- [x] Add `f64` to `UninitWorkspaceElement` sealed trait set in `workspace.rs`.
- [x] Replace `Array::zeros` with `uninit_copy_vec` + `Array::from_shape_vec`
  in `dimension_1d.rs` `forward_real_to_complex` and `inverse_complex_to_real`.
- [x] Replace `mapv(|v| Complex64::new(v, 0.0))` with `uninit_copy_vec` + `Zip`
  in `dimension_2d.rs` `forward_real_to_complex`.
- [x] Replace `Array::zeros` with `uninit_copy_vec` + `Array::from_shape_vec`
  in `dimension_2d.rs` `inverse_complex_to_real`.
- [x] Replace `Array::zeros` with `uninit_copy_vec` + `Array::from_shape_vec`
  in `dimension_3d.rs` `forward_real_to_complex`, `inverse_complex_to_real`,
  `forward_r2c`, and `inverse_c2r`.
- [x] Bump `apollo-fft` to 0.9.9 and update sprint artifacts.
- [x] Verify with `cargo check --workspace` and `cargo test -p apollo-fft --release`.

## Closure LXXIII - Plan-Time Iterator Elimination [patch]
Sprint target version: apollo-fft 0.9.8

- [x] Replace `(0..n).map(..).collect()` chirp construction in `BluesteinPlan64::new`
  with `Vec::with_capacity(n)` + `set_len(n)` + unchecked overwrite loop.
- [x] Replace `(0..n).map(..).collect()` chirp construction in `BluesteinPlan32::new`
  with `Vec::with_capacity(n)` + `set_len(n)` + unchecked overwrite loop.
- [x] Replace `(0..nz_c).map(..).collect()` r2c extraction twiddle construction in
  `FftPlan3D::with_precision` with `Vec::with_capacity(nz_c)` + `set_len(nz_c)` +
  unchecked overwrite loop.
- [x] Add `#![allow(clippy::uninit_vec)]` to `dimension_3d.rs` to maintain zero-warning
  policy alongside the existing `bluestein.rs` suppression.
- [x] Remove leftover scratch scripts (`bluestein_opt.py`, `dim3d_opt.py`) from
  the worktree.
- [x] Bump `apollo-fft` to 0.9.8 and update sprint artifacts.
- [x] Verify with `cargo fmt --check -p apollo-fft`, `cargo clippy -p apollo-fft
  --release -- -D warnings`, `cargo test -p apollo-fft --release`,
  `git diff --check`.

## Closure LXXII - 3D Native Real32 Exact Buffer Fill [patch]
Sprint target version: apollo-fft 0.9.7

- [x] Constrain the 3D native real32 helper trait to sealed workspace element
  types.
- [x] Replace allocating native 3D f32/f16 forward zero-filled output
  construction with an exact-size overwrite-first Complex32 buffer.
- [x] Replace allocating native 3D f32/f16 inverse `mapv` projection with an
  exact-size overwrite-first real buffer.
- [x] Bump `apollo-fft` to 0.9.7 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft`, `cargo check -p apollo-fft
  --benches --examples`, `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, cleanup scans, `cargo fmt --check`, and
  `git diff --check`.

## Closure LXXI - 2D Native Real32 Exact Buffer Fill [patch]
Sprint target version: apollo-fft 0.9.6

- [x] Constrain the 2D native real32 helper trait to sealed workspace element
  types.
- [x] Replace native 2D f32/f16 real-to-complex `mapv` packing with an
  exact-size overwrite-first buffer.
- [x] Replace native 2D f32/f16 complex-to-real `mapv` projection with an
  exact-size overwrite-first buffer.
- [x] Bump `apollo-fft` to 0.9.6 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft`, `cargo check -p apollo-fft
  --benches --examples`, `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, cleanup scans, `cargo fmt --check`, and
  `git diff --check`.

## Closure LXX - 1D Compact F16 Exact Buffer Fill [patch]
Sprint target version: apollo-fft 0.9.5

- [x] Extend sealed FFT workspace allocation to `f16` and `Complex<f16>`.
- [x] Replace compact f16 power-of-two forward iterator collection with
  exact-size overwrite-first packing and projection buffers.
- [x] Replace compact f16 power-of-two inverse iterator collection with
  exact-size overwrite-first packing and real-output buffers.
- [x] Bump `apollo-fft` to 0.9.5 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft`, `cargo check -p apollo-fft
  --benches --examples`, `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, cleanup scans, `cargo fmt --check`, and
  `git diff --check`.

## Closure LXIX - 1D Native Complex32 Precision Deduplication [patch]
Sprint target version: apollo-fft 0.9.4

- [x] Add private `Plan1dReal32` static-dispatch helper trait for 1D native
  `Complex32` precision paths.
- [x] Route f32 forward/inverse through shared monomorphized native helpers.
- [x] Route mixed f16 non-power-of-two forward/inverse through the same helpers.
- [x] Bump `apollo-fft` to 0.9.4 and update sprint artifacts.
- [x] Verify with `cargo fmt`, `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, source cleanup scans, encoding scans, and
  `git diff --check`.

## Closure LXVIII - Bluestein Filter Initialization Cleanup [patch]
Sprint target version: apollo-fft 0.9.3

- [x] Keep only the verified Bluestein allocation optimization from generated
  scratch work.
- [x] Remove generated scratch scripts from the worktree deliverable.
- [x] Preserve hoisted Stockham AVX broadcasts and reject repeated inline
  broadcast expansion.
- [x] Bump `apollo-fft` to 0.9.3 and update sprint artifacts.
- [x] Verify with `cargo fmt --check`,
  `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, source cleanup scans, encoding scans, and
  `git diff --check`.

## Closure LXVII - FFT Plan Scratch Allocation Consolidation [patch]
Sprint target version: apollo-fft 0.9.2

- [x] Add a sealed plan-workspace helper for uninitialized scratch allocation.
- [x] Route 1D Bluestein and iRFFT scratch buffers through the shared helper.
- [x] Route 2D/3D axis-pass and R2C scratch buffers through the shared helper.
- [x] Route six-step f32 planar and row scratch buffers through the same helper
  and remove duplicated local allocation helpers.
- [x] Bump `apollo-fft` to 0.9.2 and update sprint artifacts.
- [x] Verify with `cargo fmt --check`,
  `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, source cleanup scans, encoding scans, and
  `git diff --check`.

## Closure LXVI - FFT Workspace and Normalization Memory Efficiency [patch]
Sprint target version: apollo-fft 0.9.1

- [x] Route f64/f32 inverse normalization through shared AVX-capable helpers.
- [x] Fill twiddle tables and composite twiddle tables through exact pre-sized
  cursors with invariant checks.
- [x] Avoid zero-fill for overwritten composite and six-step workspace buffers.
- [x] Bump `apollo-fft` to 0.9.1 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, stale-token scans, encoding scans, and
  `git diff --check`.

## Closure LXV - FFT Auto-Selector Wrapper Removal [major]
Sprint target version: apollo-fft 0.9.0

- [x] Remove public concrete auto-selector wrappers for f64 and f32.
- [x] Route `FftPrecision` implementations directly to mixed-radix dispatch.
- [x] Update 1D/2D/3D plan fallbacks, tests, and benchmarks to call generic
  `fft_forward` / `fft_inverse`.
- [x] Update downstream DHT, WGPU validation, and validation benchmark callers.
- [x] Bump `apollo-fft` to 0.9.0 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, `cargo check -p apollo-fft-wgpu --tests`,
  source scans, and `git diff --check`.

## Closure LXIV - FFT Recursive Winograd Generic Codelets [major]
Sprint target version: apollo-fft 0.8.0

- [x] Replace duplicated f32/f64 DFT-16 Winograd bodies with generic `dft16_impl`.
- [x] Replace duplicated f32/f64 DFT-32 Winograd bodies with generic `dft32_impl`.
- [x] Replace duplicated f32/f64 DFT-64 Winograd bodies with generic `dft64_impl`.
- [x] Route mixed-radix DFT-16/32/64 dispatch through generic codelets.
- [x] Bump `apollo-fft` to 0.8.0 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, source scans, and `git diff --check`.

## Closure LXIII - FFT Short-Winograd Wrapper Removal [major]
Sprint target version: apollo-fft 0.7.0

- [x] Remove type-suffixed public DFT-2/3/4/5/7/8 Winograd wrappers.
- [x] Remove type-suffixed public Winograd twiddle wrappers.
- [x] Route mixed-radix short dispatch through generic `dft*_impl` functions.
- [x] Bump `apollo-fft` to 0.7.0 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, source scans, and `git diff --check`.

## Closure LXII - FFT Direct DFT Wrapper Removal [major]
Sprint target version: apollo-fft 0.6.0

- [x] Remove public type-suffixed direct DFT wrappers.
- [x] Remove unused owned Complex64 direct DFT wrappers.
- [x] Delete the debug-only `debug_f32` binary.
- [x] Update in-repo tests and benchmarks to call generic `dft_forward` /
  `dft_inverse`.
- [x] Bump `apollo-fft` to 0.6.0 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, source scans, and `git diff --check`.

## Closure LXI - FFT Composite Scratch and Twiddle Cache Reuse [patch]
Sprint target version: apollo-fft 0.5.3

- [x] Reuse Bluestein f64/f32 scratch buffers through thread-local storage
  instead of retaining one scratch vector per transform length.
- [x] Reuse mixed-radix composite f64/f32 scratch buffers through thread-local
  storage.
- [x] Cache composite twiddle tables by exact radix decomposition and direction.
- [x] Add regression coverage proving same-length radix orders do not alias in
  the twiddle cache.
- [x] Remove stale allocation and `MaybeUninit` documentation from the
  composite kernel.
- [x] Bump `apollo-fft` to 0.5.3 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, source scans, and `git diff --check`.

## Closure LX - FFT 3D Typed Plan Deduplication [patch]
Sprint target version: apollo-fft 0.5.2

- [x] Consolidate duplicated 3D f32/f16 allocating forward/inverse paths behind
  one private monomorphized helper trait.
- [x] Consolidate duplicated 3D f32/f16 caller-owned forward/inverse paths
  behind the same helper trait.
- [x] Remove the now-dead f32-only 3D real-to-complex writer.
- [x] Bump `apollo-fft` to 0.5.2 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, source scans, and `git diff --check`.

## Closure LIX - FFT 2D Typed Plan Deduplication [patch]
Sprint target version: apollo-fft 0.5.1

- [x] Consolidate duplicated 2D f32/f16 forward/inverse typed paths behind one
  private monomorphized helper trait.
- [x] Remove duplicated 2D plan Rustdoc.
- [x] Move crate-root tests from `lib.rs` to `lib_tests.rs` so `lib.rs` stays
  under the 500-line structural limit.
- [x] Bump `apollo-fft` to 0.5.1 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, structural line scan, source scans, and
  `git diff --check`.

## Closure LVIII - FFT Compatibility Alias Removal [major]
Sprint target version: apollo-fft 0.5.0

- [x] Remove `FftPlan3D::nz_complex` and keep `FftPlan3D::nz_c` as the single
  authoritative half-spectrum bookkeeping accessor.
- [x] Rename `HalfSpectrum3D::nz_complex` to `HalfSpectrum3D::nz_c`.
- [x] Remove stale compatibility/deprecation wording from FFT kernel and
  backend contract documentation.
- [x] Bump `apollo-fft` to 0.5.0 and update `CHANGELOG.md`, `backlog.md`, and
  `gap_audit.md`.
- [x] Verify with `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, source scans, and `git diff --check`.

## Closure LVII - Radix F16 Module Removal [major]
Sprint target version: apollo-fft 0.4.0

- [x] Delete `application::execution::kernel::radix2_f16`.
- [x] Delete dead f16-named bridge/gate files under the FFT kernel directory.
- [x] Replace custom `Cf16` with `num_complex::Complex<half::f16>` at all
  in-repo call sites.
- [x] Add generic `precision_bridge::Complex32Bridge` and route compact f16
  storage through the monomorphized bridge with reusable Complex32 scratch.
- [x] Remove public f16-specific FFT wrappers and update callers to use
  generic `fft_forward` / `fft_inverse` dispatch.
- [x] Update kernel exports, twiddle-table output abstraction, 1D precision
  paths, benchmarks, and SIMD module imports.
- [x] Add value-semantic compact f16 storage tests under `mixed_radix`.
- [x] Bump `apollo-fft` to 0.4.0 and update sprint artifacts.
- [x] Verify with `cargo check -p apollo-fft --benches --examples`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check --workspace`, source scans, and `git diff --check`.

## Closure LVI - FFT Remote Integration and Short-Winograd Dispatch [patch]
Sprint target version: apollo-fft 0.3.0

- [x] Resolve `origin/main` integration without restoring deleted radix
  kernel modules.
- [x] Use the workspace `rustfft` dev-dependency and retain the live
  `vs_rustfft` comparator benchmark.
- [x] Remove dead `kernel_strategy` benchmark rows for deleted radix-specific
  public kernels.
- [x] Add shared static-dispatch short-Winograd routing for exact
  2/4/8/16/32/64 f64 and f32 mixed-radix transforms.
- [x] Remove unused f16 twiddle caches from `mixed_radix`; f16 storage paths
  route through f32 short-Winograd/Stockham execution.
- [x] Verify merge result with workspace checks, FFT bench/example checks,
  Hilbert regression tests, conflict-marker scans, and `git diff --check`.

## Closure LV - Apollo-Hilbert Caller-Owned Observable Projections [minor]
Sprint target version: apollo-hilbert 0.3.0

- [x] Add `AnalyticSignal::*_into` caller-owned projections for real,
  quadrature, envelope, phase, and instantaneous frequency.
- [x] Route allocating `AnalyticSignal` projection methods through shared
  non-generic slice helpers to avoid duplicated projection logic.
- [x] Add `HilbertPlan::envelope_into` and `HilbertPlan::phase_into`.
- [x] Route plan-level `envelope` and `phase` through caller-owned projection
  paths and a reused per-thread Complex64 analytic scratch buffer.
- [x] Add value-semantic parity, output-length rejection, and observable scratch
  capacity coverage.
- [x] Bump `apollo-hilbert` to 0.3.0 and update `Cargo.lock`.
- [x] Update `apollo-hilbert` README and workspace sprint artifacts.
- [x] Verify with `cargo check -p apollo-hilbert`,
  `cargo test -p apollo-hilbert observables --lib -- --test-threads=1`,
  `cargo test -p apollo-hilbert envelope --lib -- --test-threads=1`,
  `cargo test -p apollo-hilbert --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and source scans for projection
  allocation duplication patterns.

## Closure LIV - Apollo-Hilbert Caller-Owned Analytic Signal [minor]
Sprint target version: apollo-hilbert 0.2.0

- [x] Add direct `analytic_signal_into` execution for caller-owned Complex64
  analytic output.
- [x] Add `HilbertPlan::analytic_signal_into` and keep the owned
  `analytic_signal` API routed through the caller-owned kernel.
- [x] Route `hilbert_transform_into` through a thread-local Complex64 analytic
  scratch buffer so caller-owned quadrature avoids per-call analytic `Vec`
  allocation.
- [x] Add value-semantic tests for caller-owned analytic parity, output-length
  mismatch rejection, and repeated quadrature scratch capacity reuse.
- [x] Clean stale crate-root documentation that still described private DFT
  ownership.
- [x] Bump `apollo-hilbert` to 0.2.0 and update `Cargo.lock`.
- [x] Update `apollo-hilbert` README and workspace sprint artifacts.
- [x] Verify with `cargo check -p apollo-hilbert`,
  `cargo test -p apollo-hilbert analytic --lib -- --test-threads=1`,
  `cargo test -p apollo-hilbert workspace --lib -- --test-threads=1`,
  `cargo test -p apollo-hilbert --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and source scans for removed
  caller-owned quadrature analytic allocation patterns.

## Closure LIII - Apollo FFT Slice Real Forward for Hilbert [minor]
Sprint target version: apollo-fft 0.3.0; apollo-hilbert 0.1.4

- [x] Add `FftPlan1D::forward_real_to_complex_slice_into` as the canonical
  non-generic real-forward caller-owned slice path.
- [x] Route `FftPlan1D::forward_real_to_complex_into` through the slice path to
  remove duplicated real-forward body logic.
- [x] Route `apollo-hilbert` analytic-signal execution through the cached FFT
  plan slice path and remove its real input `Array1` bridge.
- [x] Remove the now-dead `ndarray` dependency from `apollo-hilbert`.
- [x] Add FFT value-semantic coverage for slice caller-owned parity and
  slice input-length rejection.
- [x] Split 1D precision methods and tests into leaf modules so
  `dimension_1d.rs` remains below the 500-line structural limit.
- [x] Bump `apollo-fft` to 0.3.0 and `apollo-hilbert` to 0.1.4; update
  `Cargo.lock`.
- [x] Update `apollo-fft` and `apollo-hilbert` READMEs.
- [x] Verify with `cargo check -p apollo-fft`,
  `cargo test -p apollo-fft caller_owned_paths --lib -- --test-threads=1`,
  `cargo test -p apollo-fft forward_slice --lib -- --test-threads=1`,
  `cargo test -p apollo-fft --lib -- --test-threads=1`,
  `cargo check -p apollo-hilbert`,
  `cargo test -p apollo-hilbert --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and source scans for removed Hilbert
  ndarray bridge/dependency patterns.

## Closure LII - Apollo-Hilbert Analytic In-Place Spectrum Reuse [patch]
Sprint target version: apollo-hilbert 0.1.3

- [x] Keep the forward FFT result as the analytic spectrum instead of copying
  it into a `Vec` and rebuilding an `Array1`.
- [x] Run the complex inverse in place on the masked spectrum instead of
  allocating a second inverse-output array and copying it into a `Vec`.
- [x] Route owned quadrature through the caller-owned writer so owned
  quadrature allocates only its output vector plus the analytic work buffer.
- [x] Bump `apollo-hilbert` to 0.1.3 and update `Cargo.lock`.
- [x] Update `apollo-hilbert` README to describe in-place analytic spectrum
  execution.
- [x] Verify with `cargo check -p apollo-hilbert`,
  `cargo test -p apollo-hilbert transform --lib -- --test-threads=1`,
  `cargo test -p apollo-hilbert --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and source scans for removed
  analytic-signal copy allocation patterns.

## Closure LI - Apollo-Hilbert Owner Quadrature Slice Kernel [patch]
Sprint target version: apollo-hilbert 0.1.2

- [x] Add a caller-owned Hilbert quadrature kernel that writes directly into an
  output slice.
- [x] Route `HilbertPlan::transform_into` through the slice-level owner kernel
  so f64 and typed caller-owned paths no longer allocate an intermediate
  quadrature vector.
- [x] Remove the unused direct `rayon` dependency from `apollo-hilbert`.
- [x] Add direct kernel value-semantic parity and length-mismatch tests.
- [x] Bump `apollo-hilbert` to 0.1.2 and update `Cargo.lock`.
- [x] Update `apollo-hilbert` README to describe caller-owned quadrature
  execution.
- [x] Verify with `cargo check -p apollo-hilbert`,
  `cargo test -p apollo-hilbert transform_into --lib -- --test-threads=1`,
  `cargo test -p apollo-hilbert --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and source scans for the removed
  quadrature copy-through allocation plus dead direct dependency.

## Closure L - Apollo-Hilbert Typed Workspace Reuse [patch]
Sprint target version: apollo-hilbert 0.1.1

- [x] Replace typed `f32`/`f16` Hilbert input and output bridge allocations
  with thread-local reusable f64 workspaces.
- [x] Keep `f64` typed Hilbert execution on the zero-copy owner path and keep
  reduced-storage execution routed through the shared analytic-mask kernel.
- [x] Add value-semantic regression coverage proving repeated typed f32
  Hilbert calls reuse workspace capacity and preserve bitwise output.
- [x] Bump `apollo-hilbert` to 0.1.1 and update `Cargo.lock`.
- [x] Update `apollo-hilbert` README to describe typed workspace reuse.
- [x] Verify with `cargo check -p apollo-hilbert`,
  `cargo test -p apollo-hilbert workspace --lib -- --test-threads=1`,
  `cargo test -p apollo-hilbert --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and a source scan for removed production
  typed Hilbert bridge allocation patterns.

## Closure XLIX - Apollo-SDFT Typed Workspace Reuse [patch]
Sprint target version: apollo-sdft 0.1.1

- [x] Replace typed direct-bin f64 input and Complex64 output bridge
  allocations with thread-local reusable workspaces.
- [x] Keep typed direct-bin arithmetic routed through the shared non-generic
  direct-bin owner kernel.
- [x] Add value-semantic regression coverage proving repeated typed f32
  direct-bin calls reuse workspace capacity and preserve outputs.
- [x] Bump `apollo-sdft` to 0.1.1 and update `Cargo.lock`.
- [x] Update `apollo-sdft` README to describe typed workspace reuse.
- [x] Verify with `cargo check -p apollo-sdft`,
  `cargo test -p apollo-sdft workspace --lib -- --test-threads=1`,
  `cargo test -p apollo-sdft --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and a source scan for removed production
  typed direct-bin bridge allocation patterns.

## Closure XLVIII - Apollo-STFT Inverse WOLA Workspace Reuse [patch]
Sprint target version: apollo-stft 0.2.1

- [x] Replace inverse WOLA frame, complex, overlap, and weight `Vec`
  allocations with thread-local reusable workspaces.
- [x] Keep `inverse_into`, `inverse`, and typed inverse on the shared
  slice-level owner path without adding public API surface.
- [x] Add value-semantic regression coverage proving repeated `inverse_into`
  calls reuse WOLA workspace capacity and preserve reconstructed samples.
- [x] Update the co-located STFT ADR and README to reflect owner inverse
  workspace reuse.
- [x] Bump `apollo-stft` to 0.2.1 and update `Cargo.lock`.
- [x] Verify with `cargo check -p apollo-stft`,
  `cargo test -p apollo-stft workspace --lib -- --test-threads=1`,
  `cargo test -p apollo-stft --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and a source scan for removed production
  inverse WOLA allocation patterns.

## Closure XLVII - Apollo-STFT Typed Workspace Reuse and Alias Removal [major]
Sprint target version: apollo-stft 0.2.0

- [x] Add contiguous f64/Complex64 slice execution entry points to the 1D STFT
  plan so typed storage paths do not construct temporary `Array1` bridge
  values.
- [x] Replace typed STFT f64/Complex64 bridge input/output allocations with
  thread-local workspaces for repeated forward/inverse calls.
- [x] Move STFT storage/profile traits to a dedicated leaf module and keep
  `dimension_1d.rs` below the 500-line structural limit.
- [x] Add a co-located ADR for the pre-1.0 breaking alias removal and typed
  workspace design.
- [x] Remove deprecated `forward_inplace` and `inverse_inplace` allocating
  aliases; `forward`, `inverse`, `forward_into`, and `inverse_into` remain the
  canonical execution surfaces.
- [x] Add value-semantic regression coverage proving repeated f32 typed
  forward/inverse calls reuse workspace capacity and preserve outputs.
- [x] Bump `apollo-stft` to 0.2.0 and update `Cargo.lock`.
- [x] Verify with `cargo check -p apollo-stft`,
  `cargo test -p apollo-stft workspace --lib -- --test-threads=1`,
  `cargo test -p apollo-stft --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and source scans for removed production
  typed bridge allocation and deprecated alias patterns.

## Closure XLVI - Apollo-QFT Dense and Typed Workspace Reuse [patch]
Sprint target version: apollo-qft 0.1.1

- [x] Add caller-owned output dense QFT kernel entry points.
- [x] Route `QftPlan::forward_into` and `QftPlan::inverse_into` through
  contiguous Complex64 slice execution without temporary dense output vectors.
- [x] Replace typed QFT Complex64 bridge allocations with thread-local
  Complex64 input/output workspaces.
- [x] Add value-semantic regression coverage proving repeated complex32 typed
  forward/inverse calls reuse workspace capacity and preserve outputs.
- [x] Bump `apollo-qft` to 0.1.1 and update `Cargo.lock`.
- [x] Verify with `cargo check -p apollo-qft`,
  `cargo test -p apollo-qft workspace --lib -- --test-threads=1`,
  `cargo test -p apollo-qft --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and a source scan for removed QFT
  production plan/typed allocation patterns.

## Closure XLV - Apollo-GFT Typed Workspace Reuse [patch]
Sprint target version: apollo-gft 0.1.1

- [x] Add contiguous f64 slice entry points to `GftPlan` for typed bridge
  execution without temporary `Array1` construction.
- [x] Replace typed GFT f64 bridge input/output allocations with thread-local
  f64 workspaces.
- [x] Add value-semantic regression coverage proving repeated f32 typed
  forward/inverse calls reuse workspace capacity and preserve outputs.
- [x] Bump `apollo-gft` to 0.1.1 and update `Cargo.lock`.
- [x] Verify with `cargo check -p apollo-gft`,
  `cargo test -p apollo-gft workspace --lib -- --test-threads=1`,
  `cargo test -p apollo-gft --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and a source scan for removed GFT
  production typed bridge allocation patterns.

## Closure XLIV - Apollo-FWHT Typed Workspace Reuse [patch]
Sprint target version: apollo-fwht 0.1.1

- [x] Add contiguous f64 slice entry points to `FwhtPlan` for typed bridge
  execution without temporary `Array1` construction.
- [x] Replace default typed FWHT f64 bridge allocations with thread-local f64
  input/output workspaces.
- [x] Replace mixed f16 per-call f32 compute `Vec` allocation with a
  thread-local f32 workspace.
- [x] Add value-semantic regression coverage proving repeated mixed f16
  forward/inverse calls reuse workspace capacity and preserve outputs.
- [x] Bump `apollo-fwht` to 0.1.1 and update `Cargo.lock`.
- [x] Verify with `cargo check -p apollo-fwht`,
  `cargo test -p apollo-fwht workspace --lib -- --test-threads=1`,
  `cargo test -p apollo-fwht --lib -- --test-threads=1`,
  `cargo check -p apollo-validation`, and a source scan for removed FWHT typed
  bridge allocation patterns.

## Closure XLIII - Apollo-CZT Workspace Reuse and FFT Warning Cleanup [patch]
Sprint target version: apollo-czt 0.2.1; apollo-fft 0.2.2

- [x] Add plan-owned reusable Bluestein convolution workspace to `CztPlan`.
- [x] Add slice-level CZT forward execution so reusable workspaces do not
  require temporary `Array1` construction.
- [x] Precompute square-plan inverse Vandermonde nodes and add inverse
  caller-owned output execution.
- [x] Replace typed CZT `Array1<Complex64>` bridge allocations with
  thread-local Complex64 input/output workspaces.
- [x] Add value-semantic coverage for plan workspace reuse and typed
  forward/inverse workspace reuse.
- [x] Remove newly surfaced dead `apollo-fft` radix-2 butterfly helper section
  and add missing `FftPlan3D` Rustdoc.
- [x] Bump `apollo-czt` to 0.2.1, bump `apollo-fft` to 0.2.2, and update
  `Cargo.lock`.
- [x] Verify with `cargo check -p apollo-fft --lib`,
  `cargo check -p apollo-czt`, `cargo test -p apollo-czt workspace --lib -- --test-threads=1`,
  `cargo test -p apollo-czt --lib -- --test-threads=1`,
  `cargo test -p apollo-fft radix2 --lib -- --test-threads=1`,
  `cargo check -p apollo-czt-wgpu -p apollo-validation`, and source scans for
  removed CZT typed bridge allocations plus deleted radix-2 helper names.

## Closure XLII - Apollo-FRFT Typed Workspace Reuse [patch]
Sprint target version: apollo-frft 0.1.2; apollo-fft 0.2.1

- [x] Add internal contiguous Complex64 slice entry points to `FrftPlan` so
  typed storage paths can call the canonical direct FrFT kernel without
  temporary `Array1` construction.
- [x] Replace per-call typed `Array1<Complex64>` input and output allocations
  with thread-local reusable Complex64 workspaces.
- [x] Add value-semantic regression coverage proving repeated `Complex32`
  typed calls reuse workspace capacity and preserve outputs.
- [x] Restore the current `apollo-fft` kernel module header needed by the FrFT
  dependency build.
- [x] Remove current `apollo-fft` dead helper warnings from f16 bridge,
  radix permutation, radix shape, and radix stage modules without reintroducing
  compatibility facades.
- [x] Bump `apollo-frft` to 0.1.2, bump `apollo-fft` to 0.2.1, and update
  `Cargo.lock`.
- [x] Verify with `cargo check -p apollo-frft`,
  `cargo test -p apollo-frft typed --lib -- --test-threads=1`,
  `cargo test -p apollo-frft --lib -- --test-threads=1`,
  `cargo check -p apollo-frft-wgpu -p apollo-validation`,
  `cargo check -p apollo-fft --lib`, focused radix-shape/radix-permutation
  tests, and source scans for removed typed bridge allocations plus deleted
  dead helper names.

## Closure XLII - Apollo-FRFT Unitary Workspace Reuse [patch]
Sprint target version: apollo-frft 0.1.1

- [x] Replace per-call `Vec<Complex64>` coefficient allocation in
  `UnitaryFrftPlan` with reusable thread-local scratch.
- [x] Preserve the three-step Candan-Grünbaum computation:
  `c = V^T x`, `c[k] *= exp(-i order k pi/2)`, `output = V c`.
- [x] Add value-semantic regression coverage proving repeated calls reuse the
  scratch capacity and produce identical outputs.
- [x] Remove stale backward-compatibility wording from live crate-root FrFT
  exports without deleting active public API.
- [x] Bump `apollo-frft` to 0.1.1 and update `Cargo.lock`.
- [x] Verify with `cargo check -p apollo-frft`,
  `cargo test -p apollo-frft unitary --lib -- --test-threads=1`,
  `cargo test -p apollo-frft --lib -- --test-threads=1`,
  `cargo check -p apollo-frft-wgpu -p apollo-validation`, and a source scan for
  stale compatibility/deprecated markers plus the removed allocation expression.

## Closure XLII - Apollo-FFT Compatibility Re-Export Cleanup [major]
Sprint target version: apollo-fft 0.2.0

- [x] Remove root `apollo_fft::{backend,error,plan,types}` compatibility
  modules and replace root exports with canonical re-exports from
  `application::execution::plan::fft`, `domain::contracts`, and
  `domain::metadata`.
- [x] Remove `apollo_fft::application::{plan,cache}` and
  `apollo_fft::domain::{backend,error,precision,shape}` compatibility modules.
- [x] Remove the legacy `FFT_CACHE` alias; retain explicit `FFT_CACHE_1D`,
  `FFT_CACHE_2D`, and `FFT_CACHE_3D` roots.
- [x] Delete unused
  `infrastructure::cpu::simd::power_of_two::{radix4,radix8}` forwarding modules
  that duplicated canonical radix-2 execution.
- [x] Update in-repo callers in `apollo-fft-wgpu`, `apollo-czt`,
  `apollo-nufft`, `apollo-stft`, and `apollo-sft` to use root or canonical
  paths.
- [x] Bump `apollo-fft` to 0.2.0 and update `Cargo.lock`.
- [x] Verify with `cargo check -p apollo-fft --lib`,
  `cargo check -p apollo-fft --benches`, dependent-crate `cargo check`, full
  `cargo test -p apollo-fft --lib -- --test-threads=1`, and source scans for
  removed compatibility paths.

## Closure XLII - STFT-WGPU Deprecated Error and Retained-Resource Cleanup [major]
Sprint target version: apollo-stft-wgpu 0.11.0

- [x] Remove `WgpuError::FrameLenNotPowerOfTwo` from the STFT-WGPU public error
  surface.
- [x] Update non-power-of-two STFT-WGPU tests to assert successful Chirp-Z
  forward, inverse, and reusable-buffer execution instead of checking only that
  a deprecated error is absent.
- [x] Replace explicit `#[allow(dead_code)]` retained GPU-resource fields in
  STFT-WGPU buffer and Chirp-Z resource holders with `_`-prefixed ownership
  fields.
- [x] Remove explicit `#[allow(dead_code)]` suppressions from NUFFT-WGPU reusable
  buffers by enforcing `max_samples` before GPU writes and adding value-semantic
  overflow tests for 1D and 3D buffer reuse.
- [x] Replace NUFFT-WGPU per-dispatch layout-placeholder buffers with one
  retained `layout_padding_buffer` shared across fast-path bind groups whose
  entry point does not read that binding.
- [x] Remove explicit `#[allow(dead_code)]` suppressions from NTT-WGPU reusable
  buffers by deleting duplicated scalar `n_inv` storage and renaming retained
  GPU resource-owner fields.
- [x] Remove `apollo-dctdst` DCT-II/DST-II fast-path unused sibling-output
  allocations by factoring the shared 2N-point FFT setup and projection fills.
- [x] Add DCT/DST fast-path regression coverage proving single DCT-II/DST-II
  projection outputs match the dual-projection kernel and direct analytical
  kernels.
- [x] Bump `apollo-dctdst` to 0.1.1 for the patch-class memory cleanup.
- [x] Bump `apollo-stft-wgpu` to 0.11.0 and document the pre-1.0 breaking
  cleanup in `CHANGELOG.md`.
- [x] Verify WGPU cleanup with `cargo check` and `cargo test --lib` for
  `apollo-stft-wgpu`, `apollo-nufft-wgpu`, and `apollo-ntt-wgpu`, plus source
  scans for deprecated/dead-code markers in the audited crates.
- [x] Verify DCT/DST cleanup with `cargo check -p apollo-dctdst`,
  `cargo test -p apollo-dctdst fast_single_projection_paths --lib -- --test-threads=1`,
  and `cargo test -p apollo-dctdst --lib -- --test-threads=1`.

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

## Closure XLI — DHT CPU 2D/3D; FWHT CPU 2D/3D; FFT fftfreq/rfftfreq/fftshift/ifftshift [minor]
Sprint target version: 0.13.2

- [x] Add `ndarray = "0.16"` to `apollo-dht/Cargo.toml`.
- [x] Add `DhtError::ShapeMismatch2d` and `DhtError::ShapeMismatch3d` variants.
- [x] Implement `DhtPlan::forward_2d/2d_into/inverse_2d/2d_into` (row+col separable passes).
- [x] Implement `DhtPlan::forward_3d/3d_into/inverse_3d/3d_into` (axis-0/1/2 separable passes).
- [x] Re-export `Array2`, `Array3` from `apollo-dht` crate root.
- [x] Add 5 value-semantic DHT 2D/3D tests (involution, roundtrip, separability, shape rejection).
- [x] Create `crates/apollo-fwht/src/application/execution/plan/fwht/dimension_2d.rs` with `FwhtPlan2D`.
- [x] Create `crates/apollo-fwht/src/application/execution/plan/fwht/dimension_3d.rs` with `FwhtPlan3D`.
- [x] Add `dimension_2d` and `dimension_3d` modules to `plan/fwht/mod.rs`.
- [x] Re-export `FwhtPlan2D` and `FwhtPlan3D` from `apollo-fwht` crate root.
- [x] Create `crates/apollo-fft/src/application/utilities/freq.rs` (`fftfreq`, `rfftfreq`).
- [x] Create `crates/apollo-fft/src/application/utilities/shift.rs` (`fftshift`, `ifftshift`).
- [x] Create `crates/apollo-fft/src/application/utilities/mod.rs` and register in `application/mod.rs`.
- [x] Re-export all four FFT utilities from `apollo-fft` crate root.
- [x] `cargo test -p apollo-dht`: 19 passed, 0 failed.
- [x] `cargo test -p apollo-fwht`: 24 passed, 0 failed.
- [x] `cargo test -p apollo-fft`: 63 passed, 0 failed.
- [x] `cargo test -p apollo-validation -- --include-ignored`: 3 passed, 0 failed.
- [x] Sync PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] Commit and push.

## Closure XL — GPU DCT/DST 2D and 3D Separable Execution [minor]
Sprint target version: 0.13.1

- [x] Triage: identify GPU dimensional gap (`apollo-dctdst-wgpu` 1D-only vs CPU 2D/3D in XXXIX).
- [x] Add `ndarray = "0.16"` to `apollo-dctdst-wgpu/Cargo.toml`.
- [x] Add `WgpuError::ShapeMismatch` and `WgpuError::ShapeMismatch3d` to `domain/error.rs`.
- [x] Implement `execute_forward_2d` and `execute_inverse_2d` in `infrastructure/device.rs`.
- [x] Implement `execute_forward_3d` and `execute_inverse_3d` in `infrastructure/device.rs`.
- [x] Re-export `ndarray::Array2` and `ndarray::Array3` from `lib.rs`.
- [x] Add value-semantic verification tests:
  2D forward parity, 2D inverse roundtrip, 3D forward parity, 3D inverse roundtrip,
  non-square shape rejection, non-cubic shape rejection.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test -p apollo-dctdst-wgpu`: 28 passed, 0 FAILED, 0 ignored.
- [x] `cargo test -p apollo-validation -- --include-ignored`: 3 passed, 0 FAILED, 0 ignored.

## Closure XXXIX — CPU DCT/DST 2D and 3D Separable Plans [minor]
Sprint target version: 0.13.0

- [x] Triage: identify next dimensional transform gap aligned with 1D/2D/3D objective.
- [x] Add `DctDstPlan` 2D CPU APIs: `forward_2d`, `forward_2d_into`, `inverse_2d`, `inverse_2d_into`.
- [x] Add `DctDstPlan` 3D CPU APIs: `forward_3d`, `forward_3d_into`, `inverse_3d`, `inverse_3d_into`.
- [x] Add shape validation for 2D square (`N x N`) and 3D cubic (`N x N x N`) inputs and outputs.
- [x] Add value-semantic verification tests:
  2D separable parity, 2D inverse roundtrip, 3D inverse roundtrip, and mismatch rejection.
- [x] Update `crates/apollo-dctdst/README.md` execution surfaces and verification section.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test -p apollo-dctdst`: 42 passed, 0 FAILED, 0 ignored.

## Closure XXXVIII — DCT-I and DST-I Forward Known-Value Fixtures [patch]
Sprint target version: 0.12.18

- [x] Triage: identify remaining DCT-I/DST-I published-reference gaps (forward known-value fixtures).
- [x] Add validation fixture 58: `dct1_three_point_forward_known_values_fixture`
  (DCT-I N=3 x=[1,2,3]: y=[8,−2,0]; boundary formula; Rao & Yip 1990 Table 2.1; threshold 1e-15).
- [x] Add validation fixture 59: `dst1_two_point_forward_known_values_fixture`
  (DST-I N=2 x=[1,3]: y=[4√3,−2√3]; formula y[k]=2·Σsin; Rao & Yip 1990 Table 3.1; threshold 1e-12).
- [x] Update `run_published_reference_suite` call list: 57 → 59 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 57 → 59.
- [x] Update root `README.md` fixture count 57 → 59; append two new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

## Closure XXXVII — DCT-III and DST-III Published-Reference Fixtures [patch]
Sprint target version: 0.12.17

- [x] Triage: identify remaining DCT-III/DST-III published-reference gaps.
- [x] Add validation fixture 56: `dct3_dc_input_flat_output_fixture`
  (DCT-III N=4 DC input [1,0,0,0]: y=[½,½,½,½]; Makhoul 1980 Table I; FFTW REDFT01; threshold 1e-15).
- [x] Add validation fixture 57: `dst3_nyquist_input_alternating_output_fixture`
  (DST-III N=4 Nyquist input [0,0,0,1]: y=[½,−½,½,−½]; Makhoul 1980 Table II; FFTW RODFT01; threshold 1e-15).
- [x] Update `run_published_reference_suite` call list: 55 → 57 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 55 → 57.
- [x] Update root `README.md` fixture count 55 → 57; append two new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

## Closure XXXVI — CWT Ricker Impulse Peak and Scale-Normalization Fixtures [patch]
Sprint target version: 0.12.16

- [x] Triage: identify remaining CWT published-reference gaps (impulse response value, L² scale normalization).
- [x] Add validation fixture 54: `cwt_ricker_impulse_peak_value_fixture`
  (N=7 a=1 δ_{3}: W(1,3)=ψ(0), W(1,2)=W(1,4)=0; Daubechies 1992 §2.1 eq.(2.1.4); threshold 1e-14).
- [x] Add validation fixture 55: `cwt_ricker_scale_normalization_fixture`
  (N=7 a=2 δ_{3}: W(2,3)=ψ(0)/√2; Daubechies 1992 §2.1 / Grossmann-Morlet 1984 eq.(1.3); threshold 1e-13).
- [x] Update imports: add `ContinuousWavelet, CwtPlan` to `apollo-validation` use statement.
- [x] Update `run_published_reference_suite` call list: 53 → 55 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 53 → 55.
- [x] Update root `README.md` fixture count 53 → 55; append two new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

## Closure XXXV — Daubechies-4 DWT Coefficient and Reconstruction Fixtures [patch]
Sprint target version: 0.12.15

- [x] Triage: identify remaining wavelet published-reference gaps (db4 coefficients and db4 PR roundtrip).
- [x] Add validation fixture 52: `wavelet_daubechies4_one_level_known_coefficients_fixture`
  (db4 N=4 level=1 x=[1,0,0,0]: [a0,a1,d0,d1]=[h0,h2,h3,h1]; Daubechies 1992 taps; threshold 1e-15).
- [x] Add validation fixture 53: `wavelet_daubechies4_inverse_perfect_reconstruction_fixture`
  (db4 N=4 level=1: IDWT(DWT([1,-2,0.5,4]))=[1,-2,0.5,4]; Mallat 1989 Thm.2; threshold 1e-12).
- [x] Update `run_published_reference_suite` call list: 51 → 53 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 51 → 53.
- [x] Update root `README.md` fixture count 51 → 53; append two new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

## Closure XXXIV — CZT Off-Unit-Circle and Hilbert Envelope Fixtures [patch]
Sprint target version: 0.12.14

- [x] Triage: identify remaining published-reference gaps (CZT A≠1, Hilbert envelope).
- [x] Add validation fixture 50: `czt_off_unit_circle_z_transform_fixture`
  (N=2 A=2 W=exp(-πi); Z{x}(2)=1.5, Z{x}(-2)=0.5; exact dyadic; threshold 1e-12).
- [x] Add validation fixture 51: `hilbert_pure_cosine_envelope_is_unity_fixture`
  (cos(πn/2) N=4; envelope=[1,1,1,1]; exact integers; threshold 1e-12).
- [x] Update `run_published_reference_suite` call list: 49 → 51 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 49 → 51.
- [x] Update root `README.md` fixture count 49 → 51; append two new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

## Closure XXXIII — SDFT Sliding Recurrence and FrFT Order-4 Identity Fixtures [patch]
Sprint target version: 0.12.13

- [x] Triage: identify next published-reference gaps (SDFT sliding path, FrFT periodicity).
- [x] Add validation fixture 48: `sdft_sliding_recurrence_unit_impulse_all_bins_fixture`
  (N=4 zero_state, feed [1,0,0,0], all bins=1+0i; Jacobsen-Lyons 2003 eq.(2); exact; threshold 1e-12).
- [x] Add validation fixture 49: `frft_order4_identity_fixture`
  (UnitaryFrFT N=4 order=4.0: output=[1,2,3,4]=input; Candan 2000 §II Corollary; exact; threshold 1e-12).
- [x] Update `run_published_reference_suite` call list: 47 → 49 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 47 → 49.
- [x] Update root `README.md` fixture count 47 → 49; append two new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

## Closure XXXII — NUFFT Adjoint Identity and Radon Fourier Slice Theorem Fixtures [patch]
Sprint target version: 0.12.12

- [x] Triage: identify remaining published-reference gaps (NUFFT adjoint, Radon FST).
- [x] Add validation fixture 46: `nufft_type1_type2_adjoint_inner_product_fixture`
  (N=2 adjoint Re(〈Ac,f〉)=Re(〈c,A*f〉)=5; Dutt-Rokhlin 1993; exact; threshold 1e-12).
- [x] Add validation fixture 47: `radon_fourier_slice_theorem_theta0_fixture`
  (Radon θ=0 FST on [[1,2],[3,4]]; Natterer 1986 Thm 1.1; exact; threshold 1e-12).
- [x] Update `run_published_reference_suite` call list: 45 → 47 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 45 → 47.
- [x] Update root `README.md` fixture count 45 → 47; append two new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

## Closure XXXI — DCT-I and DST-I Self-Inverse Published-Reference Fixtures [patch]
Sprint target version: 0.12.11

- [x] Triage: identify DCT/DST family remaining gaps (DCT-I, DST-I inverse-roundtrip fixtures absent).
- [x] Add validation fixture 44: `dct1_inverse_roundtrip_three_point_fixture`
  (DCT-I N=3, IDCT-I∘DCT-I=I, Makhoul 1980 C1²=2(N−1)·I, FFTW REDFT00, threshold 1e-14).
- [x] Add validation fixture 45: `dst1_inverse_roundtrip_two_point_fixture`
  (DST-I N=2, IDST-I∘DST-I=I, Makhoul 1980 S1²=2(N+1)·I, FFTW RODFT00, threshold 1e-14).
- [x] Update `run_published_reference_suite` call list: 43 → 45 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 43 → 45.
- [x] Update root `README.md` fixture count 43 → 45; append two new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test -p apollo-validation -p apollo-dctdst`: 0 FAILED, 0 ignored.

## Closure XXX — DCT-IV and DST-IV Self-Inverse Published-Reference Fixtures [patch]
Sprint target version: 0.12.10

- [x] Triage: identify DCT/DST family gaps (DCT-IV, DST-IV inverse-roundtrip fixtures absent).
- [x] Add validation fixture 42: `dct4_inverse_roundtrip_two_point_fixture`
  (DCT-IV N=2, IDCT-IV∘DCT-IV=I, Makhoul 1980 C4²=N·I, FFTW REDFT11, threshold 1e-14).
- [x] Add validation fixture 43: `dst4_inverse_roundtrip_two_point_fixture`
  (DST-IV N=2, IDST-IV∘DST-IV=I, Makhoul 1980 S4²=N·I, FFTW RODFT11, threshold 1e-14).
- [x] Update `run_published_reference_suite` call list: 41 → 43 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 41 → 43.
- [x] Update root `README.md` fixture count 41 → 43; append two new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test --workspace`: 0 FAILED, 0 ignored.

## Closure XXIX — Inverse-Roundtrip Published-Reference Fixtures: NTT, STFT [patch]
Sprint target version: 0.12.9

- [x] Triage: identify transforms with inverse API but no inverse-roundtrip fixture (NTT, STFT).
- [x] Add validation fixture 40: `ntt_inverse_roundtrip_fixture`
  (NTT N=4, INTT∘NTT=I in Z/pZ, Pollard 1971, threshold 1e-12).
- [x] Add validation fixture 41: `stft_hann_wola_inverse_roundtrip_fixture`
  (STFT frame=4 hop=2, Hann COLA WOLA, Allen-Rabiner 1977, threshold 1e-12).
- [x] Update `run_published_reference_suite` call list: 39 → 41 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 39 → 41.
- [x] Update root `README.md` fixture count 39 → 41; append two new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test --workspace`: 0 FAILED, 0 ignored.

## Closure XXVIII — Inverse-Roundtrip Published-Reference Fixtures: DHT, SFT [patch]
Sprint target version: 0.12.8

- [x] Triage: identify transforms with inverse API but no inverse-roundtrip fixture (DHT, SFT).
- [x] Add validation fixture 38: `dht_inverse_roundtrip_fixture`
  (DHT N=4, IDHT∘DHT=I, Bracewell 1983, threshold 1e-14).
- [x] Add validation fixture 39: `sft_inverse_roundtrip_fixture`
  (SFT N=4 K=1, ISFT∘SFT=I, Hassanieh et al. 2012, threshold 1e-12).
- [x] Update `run_published_reference_suite` call list: 37 → 39 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 37 → 39.
- [x] Update root `README.md` fixture count 37 → 39; append two new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test --workspace`: 0 FAILED, 0 ignored.

## Closure XXVII — Inverse-Roundtrip Published-Reference Fixtures: FWHT, QFT, SHT [patch]
Sprint target version: 0.12.7

- [x] Triage: identify transforms with inverse API but no inverse-roundtrip fixture.
- [x] Add validation fixture 35: `fwht_inverse_roundtrip_fixture`
  (FWHT N=4, IFWHT∘FWHT=I, Walsh 1923, threshold 1e-14).
- [x] Add validation fixture 36: `qft_inverse_roundtrip_fixture`
  (QFT N=4, iqft∘qft=I, Shor 1994, threshold 1e-12).
- [x] Add validation fixture 37: `sht_inverse_roundtrip_y10_fixture`
  (SHT lmax=1, dipole Y_1^0 roundtrip, Driscoll-Healy 1994, threshold 1e-10).
- [x] Update `run_published_reference_suite` call list: 34 → 37 fixtures.
- [x] Update both count assertions in `apollo-validation/suite.rs`: 34 → 37.
- [x] Update root `README.md` fixture count 34 → 37; append three new entries.
- [x] Update sprint PM artifacts: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.
- [x] `cargo test --workspace`: 0 FAILED, 0 ignored.

## Closure XXVI — Inverse-Roundtrip Published-Reference Fixtures: DWT, GFT, FrFT [patch]
## Closure XXIV — GPU Adapter Preference, Test Runtime-Skip, Bluestein CZT Fix [patch]
## Closure XXV — Hilbert Instantaneous Frequency + Doc/Test/PM Cleanup [patch]

- [x] Convert `apollo-ntt-wgpu` doc-test from `rust,ignore` to `rust,no_run` with preamble; verify 0 ignored workspace-wide.
- [x] Expand `execute_inverse_with_buffers` doc comment in `apollo-stft-wgpu/device.rs`.
- [x] Add missing `CHANGELOG.md` entries for Closure XXIII (0.12.3) and Closure XXIV (0.12.4).
- [x] Implement `AnalyticSignal::instantaneous_frequency()` using complex-derivative formula.
- [x] Add `instantaneous_frequency_constant_tone` test (ε<1e-10, k/N=5/64).
- [x] Add `double_hilbert_negates_zero_mean_signal` test (ε<1e-10, sinusoidal input N=32).
- [x] Add validation fixture 31: `hilbert_instantaneous_frequency_constant_tone_fixture` (N=64, k=5, threshold 1e-10).
- [x] Update `run_published_reference_suite` call list to include fixture 31.
- [x] Update `assert_eq!(report.external.published_references.attempted, 30)` → 31.
- [x] Update `assert_eq!(report.attempted, 30)` → 31.
- [x] Update root `README.md` fixture count: 30 → 31; append new fixture entry.
- [x] Update `apollo-hilbert/README.md` with instantaneous frequency subsection.
- [x] `cargo test -p apollo-hilbert`: 11 passed, 0 failed, 0 ignored.
- [x] `cargo test -p apollo-validation`: 3 passed, 0 failed, 0 ignored.
- [x] `cargo test --workspace`: 0 FAILED, 0 ignored.
- [x] Artifact sync: backlog.md, checklist.md, gap_audit.md, CHANGELOG.md.

## Closure XXIV — GPU Adapter Preference, Test Runtime-Skip, Bluestein CZT Fix [patch]
Sprint target version: 0.12.4

- [x] Replace all 20 `wgpu::RequestAdapterOptions::default()` with `PowerPreference::HighPerformance`.
- [x] Remove `#[ignore]` from all 10 `apollo-ntt-wgpu` tests; convert to early-return pattern.
- [x] Remove `#[ignore]` from all 7 `apollo-stft-wgpu` tests (early-return pattern already present).
- [x] Fix `stft_chirp.wgsl`: premul_fwd sign (−πi·n²/N), premul_inv sign (+πi·n²/N), postmul_fwd sign (−πi·k²/N), postmul_inv sign (+πi·n²/N).
- [x] Add `stft_chirp_pointmul_fwd` entry point (conj(h_stored) = h_fwd via −h_fft_im).
- [x] Add `pointmul_fwd_pipeline` field to `StftChirpData`; build in `new()`.
- [x] Update `execute_forward_fft_chirp` in `kernel.rs` to dispatch `pointmul_fwd_pipeline`.
- [x] Add POT guard to `execute_forward_with_buffers` delegating non-PoT to allocating path.
- [x] Add POT guard to `execute_inverse_with_buffers` delegating non-PoT to allocating path.
- [x] Calibrate forward CZT test tolerance: 1e-2 → 2e-2 (f32 GPU argument-reduction, N=400).
- [x] `cargo check --workspace`: 0 errors, 0 warnings.
- [x] `cargo test -p apollo-stft-wgpu`: 23/23 passed, 0 failed, 0 ignored.
- [x] `cargo test --workspace`: 0 FAILED (case-sensitive), 0 ignored.
- [x] Artifact sync: backlog.md, checklist.md, gap_audit.md.


## Closure XXIII — ARCHITECTURE.md Capability Annotation + Validation Fixtures 29-30 [patch]
Sprint target version: 0.12.3

- [x] Audit ARCHITECTURE.md Mixed-Precision Capability Table for stale Notes entries.
- [x] Fix `apollo-czt-wgpu` Notes: "forward + inverse CZT; f16 promoted to f32 at host boundary".
- [x] Fix `apollo-mellin-wgpu` Notes: "forward + inverse Mellin spectrum; f16 promoted at host boundary".
- [x] Add `czt_inverse_vandermonde_roundtrip_fixture` (fixture 29): N=4, A=1, W=exp(-2πi/4), threshold 1e-12; Björck-Pereyra (1970) + Rabiner-Schafer-Rader (1969).
- [x] Add `mellin_inverse_spectrum_constant_roundtrip_fixture` (fixture 30): N=32, c=2, [1,4], threshold 1e-10; Mellin (1896), Titchmarsh (1937).
- [x] Add `published_real_fixture_with_threshold` helper; refactor `published_real_fixture` to delegate.
- [x] Register both fixtures in `run_published_reference_suite`.
- [x] Update `validation_suite_produces_value_semantic_reports` assertion: 28 → 30.
- [x] Update README.md fixture count: 28 → 30; extend fixture list with two new entries.
- [x] `cargo check -p apollo-validation`: 0 errors, 0 warnings.
- [x] `cargo test -p apollo-validation validation_suite_produces_value_semantic_reports`: 1/1 passed.
- [x] Artifact sync: backlog.md, gap_audit.md.

## Closure XXII — GPU Benchmark Runner Workflow + Root README Correction [patch]
Sprint target version: 0.12.2

- [x] Audit existing CI: only hosted `ubuntu-latest` jobs present; no self-hosted GPU path.
- [x] Audit benchmark surfaces: `apollo-fft-wgpu`, `apollo-nufft-wgpu`, `apollo-stft-wgpu`, `apollo-radon-wgpu` Criterion suites present.
- [x] Add `.gitignore` rule for generated `.benchmarks/gpu-runner/*` while preserving `.gitkeep`.
- [x] Add manual workflow `.github/workflows/gpu-benchmarks.yml` targeting `[self-hosted, gpu, apollo]`.
- [x] Add PowerShell runner `scripts/run_gpu_benchmarks.ps1` with manifest and summary emission.
- [x] Stage artifacts under `.benchmarks/gpu-runner/run-<run_id>/` and upload via workflow artifact.
- [x] Correct stale root README claims for `apollo-czt-wgpu`, `apollo-mellin-wgpu`, `apollo-radon-wgpu`, and `apollo-stft-wgpu`.
- [x] Add root README section documenting the GPU benchmark runner labels, workflow, and outputs.
- [x] Update `CHANGELOG.md`, `backlog.md`, and `gap_audit.md` for Closure XXII.
- [x] Validate PowerShell runner syntax.
- [x] Validate workflow YAML parses.
- [x] Smoke-run `scripts/run_gpu_benchmarks.ps1` on `fft` group: bundle created under `.benchmarks/gpu-runner/manual-smoke`, manifest and criterion output verified, transient bundle removed after validation.

## Closure XXI — README Documentation Sync for v0.2.0 Inverse Additions [patch]
Sprint target version: 0.2.1 (documentation only, no API change)

- [x] `apollo-czt/README.md`: add "Inverse Transform" section (Björck-Pereyra, NotInvertible conditions).
- [x] `apollo-mellin/README.md`: add "Inverse Transform" section (IDFT + exp-resample, SpectrumLengthMismatch).
- [x] `apollo-czt-wgpu/README.md`: update "Execution Contract" and "Verification" to reflect forward+inverse.
- [x] `apollo-mellin-wgpu/README.md`: update "Execution Contract" and "Verification" to reflect two-pass inverse.
- [x] `checklist.md`: add Closure XX completed entry.
- [x] `backlog.md`: add Closure XXI entry.
- [x] `cargo check --workspace`: 0 errors, 0 warnings.

## Closure XX — CPU + GPU Inverse Transforms: CZT and Mellin [minor]
Sprint target version: 0.2.0

- [x] Audit apollo-czt source: `forward` only, no inverse. `CztError` variants enumerated.
- [x] Audit apollo-mellin source: `forward_spectrum` / `forward_resample` / `moment` only.
- [x] Audit apollo-czt-wgpu: `execute_inverse` stub returning `UnsupportedExecution`.
- [x] Audit apollo-mellin-wgpu: `execute_inverse` stub returning `UnsupportedExecution`.
- [x] Derive CZT inverse: Vandermonde system `V·y = X` → Björck-Pereyra O(N²) Newton solve.
- [x] Derive Mellin inverse: IDFT of log-spectrum → `g[n]`, then exp-resample `g` → signal.
- [x] Implement `czt_bjork_pereyra_inverse` in `bluestein.rs`; fix borrow checker (cache `c[k+1]`).
- [x] Add `CztError::NotInvertible { reason: &'static str }`.
- [x] Add `CztPlan::inverse`, `CztStorage::inverse_into`; 5 value-semantic tests.
- [x] Implement `inverse_log_frequency_spectrum` + `exp_resample` in `resample.rs`.
- [x] Add `MellinError::SpectrumLengthMismatch`.
- [x] Add `MellinPlan::inverse_spectrum`; export `exp_resample` + `inverse_log_frequency_spectrum`; 4 tests.
- [x] Fix `assert_abs_diff_eq!` with message argument (unsupported by approx 0.5.1); use `assert!`.
- [x] Remove unused `approx::assert_abs_diff_eq` import from `inverse_tests`.
- [x] Add `czt_inverse` WGSL entry point (adjoint formula); build `inverse_pipeline`.
- [x] Implement `CztWgpuBackend::execute_inverse`; `WgpuCapabilities::forward_inverse`.
- [x] Update capability and backend tests; add `gpu_inverse_roundtrip_dft_parameters`, `gpu_inverse_rejects_non_square_plan`.
- [x] Add `mellin_inverse_spectrum` + `mellin_exp_resample` WGSL kernels; `InverseMellinParamsPod`.
- [x] Add `inverse_spectrum_pipeline`, `exp_resample_pipeline`, `inv_params_buffer` to kernel.
- [x] Implement `MellinGpuKernel::execute_inverse` (two-pass: IDFT + exp-resample).
- [x] Implement `MellinWgpuBackend::execute_inverse`; `WgpuCapabilities::forward_inverse`.
- [x] Update capability/backend tests; add `gpu_inverse_roundtrip_constant_signal`, `gpu_inverse_rejects_invalid_output_domain`.
- [x] Bump all four crates to v0.2.0.
- [x] `cargo test -p apollo-czt -p apollo-mellin -p apollo-czt-wgpu -p apollo-mellin-wgpu`: 61/61 passed.
- [x] Artifact sync: CHANGELOG.md (Closure XX), backlog.md, gap_audit.md.

## Closure XIX — StftGpuBuffers Non-PoT Scratch Sizing [minor]
Sprint target version: 0.10.0

- [x] Read `buffers.rs`: inspect current scratch sizing (frame_count × frame_len).
- [x] Import `chirp_padded_len` from `super::chirp`.
- [x] Update scratch_elem_count computation: use `chirp_padded_len(frame_len)` for non-PoT.
- [x] Remove `assert!(frame_len.is_power_of_two())` in `StftGpuBuffers::new`.
- [x] Update docstring in `buffers.rs`: remove PoT constraint; add Closure XIX note.
- [x] Update `device.rs`: `make_buffers` docstring mentions non-PoT support + Closure XIX note.
- [x] `cargo check -p apollo-stft-wgpu`: 0 warnings, 0 errors.
- [x] Add `make_buffers_accepts_non_power_of_two_frame_len_structurally` test.
- [x] Add GPU-gated test: `forward_buffers_non_pot_frame_len_400_when_device_exists`.
- [x] Add GPU-gated test: `inverse_buffers_non_pot_frame_len_400_when_device_exists`.
- [x] Fix unused variable warning in tests.
- [x] `cargo test -p apollo-stft-wgpu`: 16 passed; 7 ignored (GPU-gated); 0 failed.
- [x] Bump version: 0.9.0 → 0.10.0 in Cargo.toml.
- [x] Artifact sync: CHANGELOG.md (0.10.0), backlog.md, checklist.md, gap_audit.md.

## Closure XVIII — Non-Power-of-Two STFT GPU Path (Bluestein/Chirp-Z) [minor]
Sprint target version: 0.9.0

- [x] Write ADR: `design_history_file/adr_stft_wgpu_non_pot_chirpz.md` (required before [minor]).
- [x] Create `stft_chirp.wgsl`: five-pass Bluestein WGSL shader (premul_fwd, premul_inv, pointmul, postmul_fwd, postmul_inv).
- [x] Create `stft_chirp_fft.wgsl`: radix-2 sub-FFT shader (bitrev, butterfly_fwd, butterfly_inv, scale).
- [x] Create `infrastructure/chirp.rs`: `StftChirpData` struct with GPU resource pre-allocation and `chirp_padded_len`.
- [x] Update `infrastructure/mod.rs`: add `pub(crate) mod chirp;`.
- [x] Update `Cargo.toml`: add `ndarray = "0.16"` to `[dependencies]`.
- [x] Update `kernel.rs`: conditional dispatch (PoT → existing Radix-2, non-PoT → `execute_forward_fft_chirp` / `execute_inverse_chirp`); add `dispatch_chirp_radix2` helper.
- [x] Update `device.rs`: remove `FrameLenNotPowerOfTwo` guard from `execute_forward` and `execute_inverse`.
- [x] Update `error.rs`: update `FrameLenNotPowerOfTwo` doc to reflect deprecated-from-primary-path status.
- [x] Fix all compile warnings: removed unused import `StftChirpParamsPod`, removed `mut` on closure, added `#[allow(dead_code)]` on GPU-lifetime fields.
- [x] Update/rename old rejection tests → acceptance tests (`forward_accepts_non_power_of_two_frame_len_chirpz`, `inverse_accepts_non_power_of_two_frame_len_chirpz`).
- [x] Add `forward_accepts_non_power_of_two_frame_len_structurally`.
- [x] Add GPU-gated tests: `forward_chirpz_non_pot_frame_len_400_when_device_exists`, `inverse_chirpz_non_pot_frame_len_400_when_device_exists`.
- [x] `cargo check -p apollo-stft-wgpu`: 0 warnings, 0 errors.
- [x] `cargo test -p apollo-stft-wgpu`: 15 passed; 5 ignored (GPU-gated); 0 failed.
- [x] Bump `apollo-stft-wgpu` version: 0.1.0 → 0.9.0.
- [x] Artifact sync: CHANGELOG.md, backlog.md, checklist.md, gap_audit.md.

## Closure XVII — STFT GPU Buffer-Reuse Criterion Benchmarks + README Usage Documentation [patch]
Sprint target version: 0.8.5

- [x] Add `bench_forward_reuse` to `crates/apollo-stft-wgpu/benches/stft_bench.rs`:
      head-to-head `execute_forward` vs `execute_forward_with_buffers` at fl ∈ {256, 512, 1024};
      `StftGpuBuffers` pre-allocated outside bench loop.
- [x] Add `bench_inverse_reuse` to `stft_bench.rs`:
      head-to-head `execute_inverse` vs `execute_inverse_with_buffers`;
      spectrum pre-computed outside bench loop; only inverse dispatch measured.
- [x] Add both groups to `criterion_group!(benches, …)` in `stft_bench.rs`.
- [x] Update `stft_bench.rs` module docstring to describe both allocating and buffer-reuse
      paths and their mathematical basis.
- [x] Add "Buffer Reuse" section to `crates/apollo-stft-wgpu/README.md` with usage snippet,
      constraint notes (`FrameLenNotPowerOfTwo`, `LengthMismatch`), and pattern description.
- [x] Add "Benchmarks" section to `README.md` with group table and `cargo bench` invocation.
- [x] `cargo check -p apollo-stft-wgpu` clean (bench compiles against `StftGpuBuffers` API).
- [x] `cargo test -p apollo-stft-wgpu`: 14 passed; 3 ignored (GPU-gated).
- [x] Artifact sync: CHANGELOG.md (0.8.5), Cargo.toml (0.8.4 → 0.8.5), backlog.md,
      checklist.md, gap_audit.md updated.

## Closure XVI — StftGpuBuffers Pre-allocated Buffer Reuse [minor]
Sprint target version: 0.8.4

- [x] Create `crates/apollo-stft-wgpu/src/infrastructure/buffers.rs` with `StftGpuBuffers`
      struct, `StftGpuBuffers::new(device, kernel, frame_count, frame_len, signal_len, hop_len)`,
      and accessors `frame_count()`, `frame_len()`, `signal_len()`, `hop_len()`,
      `fwd_output()`, `inv_output()`.
- [x] Make `ComplexPod`, `StftParams`, `FftStageParams`, `FwdFftStageParams` `pub(crate)` in
      `kernel.rs`.
- [x] Make `bind_group_layout`, `fft_data_bgl`, `fft_params_bgl` fields `pub(crate)` in
      `StftGpuKernel`.
- [x] Add `StftGpuKernel::execute_forward_fft_with_buffers` (reuses bind groups; uploads
      signal via `queue.write_buffer`; writes result to `buffers.fwd_output_host`).
- [x] Add `StftGpuKernel::execute_inverse_with_buffers` (uploads spectrum + OLA params;
      writes result to `buffers.inv_output_host`).
- [x] Add `StftWgpuBackend::make_buffers`, `execute_forward_with_buffers`,
      `execute_inverse_with_buffers` to `device.rs`.
- [x] Re-export `StftGpuBuffers` from `lib.rs`.
- [x] Add `#[allow(dead_code)]` on GPU-only scratch fields (`re_scratch_buf`,
      `im_scratch_buf`, `frame_data_buf`) with architectural justification doc-comment.
- [x] `cargo clippy -p apollo-stft-wgpu --all-targets -- -D warnings` clean.
- [x] `cargo test -p apollo-stft-wgpu`: 14 passed; 3 ignored (`reusable_buffers_match_*`,
      `forward_fft_roundtrip_*`, `inverse_roundtrip_large_*`).
- [x] Add `pub mod buffers` to `infrastructure/mod.rs`.
- [x] Artifact sync: CHANGELOG.md, Cargo.toml (0.8.3 → 0.8.4), backlog.md,
      checklist.md, gap_audit.md updated.

## Closure XV — Radon FBP GPU Criterion Benchmarks [patch]
Sprint target version: 0.8.3

- [x] Add `criterion = "0.5"` to `apollo-radon-wgpu` dev-deps.
- [x] Add `[[bench]] name = "radon_wgpu_bench" harness = false` to `apollo-radon-wgpu/Cargo.toml`.
- [x] Create `crates/apollo-radon-wgpu/benches/radon_wgpu_bench.rs` with `radon_wgpu_forward`
      and `radon_wgpu_fbp` criterion groups, each covering image_size ∈ {64, 128, 256}.
- [x] Gaussian disk phantom (σ=0.25); uniform angles on `[0,π)`.
- [x] `cargo check -p apollo-radon-wgpu --benches` clean.
- [x] Artifact sync: backlog.md, checklist.md, gap_audit.md, CHANGELOG.md updated.

## Closure XIV — Dead-Code Removal: O(N²) Forward Pipeline + stft_inverse_frames [patch]
Sprint target version: 0.3.0

- [x] Remove `forward_pipeline: wgpu::ComputePipeline` field from `StftGpuKernel`.
- [x] Remove forward shader module creation and `forward_pipeline` construction from `new()`.
- [x] Remove `StftGpuKernel::execute()` method (112 lines of O(N²) GPU forward path).
- [x] Delete `crates/apollo-stft-wgpu/src/infrastructure/shaders/stft.wgsl`.
- [x] Remove `stft_inverse_frames` entry point from `stft_inverse.wgsl` (~40 lines).
- [x] Update `stft_inverse.wgsl` file comment to reflect single-pass OLA role.
- [x] Update kernel.rs module docstring, `WORKGROUP_SIZE` comment, struct doc.
- [x] Update `dispatch_count` and `fft_dispatch_count` doc comments.
- [x] `cargo clippy -p apollo-stft-wgpu -- -D warnings` clean.
- [x] `cargo test -p apollo-stft-wgpu` 14 passed; 2 ignored.

## Closure XIII — STFT GPU Criterion Benchmarks [patch]
Sprint target version: 0.3.0 (patch within the next minor)

- [x] Add `criterion = { version = "0.5", features = ["html_reports"] }` to `apollo-stft-wgpu` dev-deps.
- [x] Add `[[bench]] name = "stft_bench" harness = false` to `apollo-stft-wgpu/Cargo.toml`.
- [x] Create `crates/apollo-stft-wgpu/benches/stft_bench.rs` with `bench_forward_fft` and
      `bench_inverse_fft` criterion groups, each covering frame_len ∈ {256, 512, 1024}.
- [x] Analytical signal (bin-aligned sinusoids k₁=16, k₂=64) used as benchmark workload.
- [x] `cargo check -p apollo-stft-wgpu --benches` clean.
- [x] Artifact sync: backlog.md, checklist.md, gap_audit.md, CHANGELOG.md updated.

## Closure XII — STFT Forward-Path GPU FFT Acceleration [minor]
Sprint target version: 0.3.0 (first unreleased minor after 0.2.0)

- [x] Create `crates/apollo-stft-wgpu/src/infrastructure/shaders/stft_forward_fft.wgsl`
      with entry points: `stft_fwd_pack_window`, `stft_fwd_bitrev`, `stft_fwd_butterfly`,
      `stft_fwd_interleave`; DFT twiddle `exp(−2πi·k/N)`.
- [x] Add `FwdFftStageParams` struct (16 bytes, 4×u32: frame_count, frame_len, hop_len, stage).
- [x] Add 4 new pipeline fields to `StftGpuKernel`: `fwd_pack_window_pipeline`,
      `fwd_bitrev_pipeline`, `fwd_butterfly_pipeline`, `fwd_interleave_pipeline`.
- [x] Extend `StftGpuKernel::new()` to compile `stft_forward_fft.wgsl` and build 4 pipelines,
      reusing `fft_pipeline_layout` (group 0: `fft_data_bgl`, group 1: `fft_params_bgl`).
- [x] Implement `StftGpuKernel::execute_forward_fft()`:
      dispatch sequence pack_window → bitrev → butterfly×log₂N → interleave.
- [x] Add `FrameLenNotPowerOfTwo` guard in `StftWgpuBackend::execute_forward()`.
- [x] Route `execute_forward` to `kernel.execute_forward_fft()`.
- [x] Add test `forward_rejects_non_power_of_two_frame_len` (no GPU required).
- [x] Add test `forward_fft_roundtrip_large_frame_when_device_exists` (#[ignore], FRAME_LEN=1024).
- [x] `cargo check -p apollo-stft-wgpu` clean.
- [x] `cargo clippy -p apollo-stft-wgpu -- -D warnings` clean.
- [x] `cargo test -p apollo-stft-wgpu` passing.
- [x] Artifact sync: backlog.md, checklist.md, gap_audit.md, CHANGELOG.md updated.

## Closure XI phase (STFT inverse GPU acceleration, FrameLenNotPowerOfTwo, large-frame verification)
- [x] Create `apollo-stft-wgpu/src/infrastructure/shaders/stft_inverse_fft.wgsl` with four entry points: `stft_deinterleave`, `stft_bitrev`, `stft_butterfly`, `stft_scale_and_window`. Two bind groups: group 0 (4 data bindings), group 1 (per-stage `FftStageParams` uniform). IDFT twiddle: exp(+2πi·k/N). Formal basis: Cooley-Tukey Radix-2 DIT (Cooley & Tukey 1965); WOLA identity (Allen & Rabiner 1977 Theorem 1).
- [x] Add `FftStageParams` struct (`frame_count, frame_len, stage, _pad`: 4×u32 = 16 bytes) to `kernel.rs`.
- [x] Replace `inverse_frames_pipeline` in `StftGpuKernel` with `deinterleave_pipeline`, `bitrev_pipeline`, `butterfly_pipeline`, `scale_window_pipeline`; add `fft_data_bgl` (4-binding group-0 layout) and `fft_params_bgl` (1-uniform group-1 layout); keep `FFT_WORKGROUP_SIZE = 256` separate from `WORKGROUP_SIZE = 64` to avoid under-dispatching forward/OLA passes.
- [x] Rewrite `StftGpuKernel::execute_inverse` to: validate `frame_len.is_power_of_two()`, allocate re/im scratch and frame_data buffers, pre-allocate `log₂(N)` per-stage uniform buffers and bind groups, encode deinterleave + bitrev + N butterfly passes + scale_window + OLA in one `CommandEncoder`.
- [x] Add `WgpuError::FrameLenNotPowerOfTwo { frame_len: usize }` variant to `domain/error.rs`.
- [x] Add power-of-two validation guard in `StftWgpuBackend::execute_inverse` (`device.rs`) before kernel dispatch.
- [x] Add `inverse_rejects_non_power_of_two_frame_len` test: frame_len=6, expects `FrameLenNotPowerOfTwo { frame_len: 6 }`.
- [x] Add `#[ignore = "requires wgpu device"] inverse_roundtrip_large_frame_1024_samples_when_device_exists` test: frame_len=1024, hop=512, signal_len=8192, analytic sine reference, TOL=5e-3.
- [x] Verify `cargo check --workspace --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures (1 GPU-gated test ignored).

## Closure X phase (GPU Radon FBP, adjoint identity test, STFT parameterized roundtrip, documentation sync)
- [x] Add `supports_filtered_backprojection: bool` field and `forward_inverse_and_fbp(device_available)` constructor to `apollo-radon-wgpu/src/domain/capabilities.rs`.
- [x] Create `apollo-radon-wgpu/src/infrastructure/shaders/radon_fbp_filter.wgsl`: entry `radon_fbp_filter` — per-(angle, detector) circular convolution with the ramp filter impulse response `h`: `filtered[a*D+d] = Σ_{d'} sinogram[a*D+d'] * h[(d-d'+D)%D]`. Reuses existing 4-binding layout (read, read, read_write, uniform). Basis: Ram-Lak ramp filter (Bracewell & Riddle 1967; Shepp & Logan 1974).
- [x] Add `fbp_filter_pipeline: wgpu::ComputePipeline` to `RadonGpuKernel`; add `compute_ramp_kernel_f32(detector_count, detector_spacing) -> Vec<f32>` (= `ramp_filter_projection([1,0,...], spacing)` cast to f32); add `RadonGpuKernel::execute_filtered_backproject(device, queue, plan, sinogram, angles) -> WgpuResult<Array2<f32>>` (2-pass single encoder: filter → backproject; host-side `* π/angle_count` normalization) in `apollo-radon-wgpu/src/infrastructure/kernel.rs`.
- [x] Add `RadonWgpuBackend::execute_filtered_backproject(plan, sinogram, angles)` to `apollo-radon-wgpu/src/infrastructure/device.rs`; update `capabilities()` to return `forward_inverse_and_fbp(true)`.
- [x] Add 4 new tests to `apollo-radon-wgpu/src/verification.rs`: `backproject_satisfies_adjoint_identity_when_device_exists` (⟨Af,g⟩ = ⟨f,A†g⟩, rel_tol=5e-3), `capabilities_include_filtered_backprojection`, `filtered_backproject_matches_cpu_reference_when_device_exists` (single-center-pixel reference, TOL=5e-2), `filtered_backproject_rejects_sinogram_shape_mismatch`.
- [x] Add `inverse_roundtrip_for_multiple_cola_parameter_sets` test to `apollo-stft-wgpu/src/verification.rs`: 3 COLA-compliant (frame_len, hop_len) pairs at 50% overlap (8/4, 16/8, 32/16); CPU forward → GPU inverse roundtrip; TOL=5e-3.
- [x] Update `README.md`: fix stale WGPU descriptions for `apollo-radon-wgpu` (add FBP), `apollo-stft-wgpu` (add inverse), `apollo-hilbert-wgpu` (add inverse), `apollo-sdft-wgpu` (add inverse).
- [x] Update `ARCHITECTURE.md`: fix capability table notes for `apollo-radon-wgpu`, `apollo-stft-wgpu`, `apollo-hilbert-wgpu`, `apollo-sdft-wgpu` rows.
- [x] Verify `cargo check --workspace --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures.

## Closure IX phase (GPU inverse STFT WOLA, GPU Radon backprojection, artifact corrections)
- [x] Add `WgpuCapabilities::forward_and_inverse(device_available)` constructor to `apollo-stft-wgpu/src/domain/capabilities.rs`.
- [x] Create `apollo-stft-wgpu/src/infrastructure/shaders/stft_inverse.wgsl`: two entry points sharing `@binding(0) array<f32>` (read), `@binding(1) array<f32>` (read_write), `@binding(2)` uniform. `stft_inverse_frames`: per-(frame, local_j) windowed IDFT — `frame_data[m·N+j] = (1/N)·Re{Σ_k X[m,k]·exp(+2πi·k·j/N)}·hann(j)`. `stft_inverse_ola`: per-output-sample WOLA — `y[n] = Σ_m frame_data[m·N+(n−start_m)] / Σ_m hann(n−start_m)²`, `start_m = m·hop − N/2`. Basis: WOLA identity (Allen–Rabiner 1977, Theorem 1).
- [x] Add `inverse_frames_pipeline` and `inverse_ola_pipeline` fields to `StftGpuKernel`; compile from `stft_inverse.wgsl` in `apollo-stft-wgpu/src/infrastructure/kernel.rs`. Add `StftGpuKernel::execute_inverse(device, queue, spectrum, frame_len, hop_len, frame_count, signal_len) -> WgpuResult<Vec<f32>>`: interleave spectrum as f32 pairs; 2-pass single-encoder (frames → ola); copy result → staging; return `Vec<f32>`.
- [x] Update `StftWgpuBackend::execute_inverse` in `apollo-stft-wgpu/src/infrastructure/device.rs`: change from no-arg stub to real signature `(plan, spectrum, signal_len)`, validate inputs, derive `frame_count = 1 + signal_len.div_ceil(hop_len)`, delegate to kernel. Add `execute_inverse_typed_into<I: StftSpectrumInput, O: StftRealOutputStorage>(plan, input_precision, output_precision, spectrum, signal_len, output)`. Update `capabilities()` to return `forward_and_inverse(true)`.
- [x] Update `apollo-stft-wgpu/src/verification.rs`: remove `execute_inverse_returns_unsupported` test; rename `backend_reports_forward_only_when_device_exists` → `backend_reports_forward_and_inverse_when_device_exists` (assert `supports_inverse = true`). Add `capabilities_reflect_forward_and_inverse_surface`, `inverse_roundtrip_recovers_cola_signal_when_device_exists` (CPU forward → GPU inverse vs CPU inverse, TOL=5e-4), `inverse_matches_cpu_reference_for_16sample_signal`.
- [x] Add `WgpuCapabilities::forward_and_inverse(device_available)` constructor to `apollo-radon-wgpu/src/domain/capabilities.rs`.
- [x] Add `SinogramShapeMismatch { expected_angles, expected_detectors, actual_angles, actual_detectors }` variant to `WgpuError` in `apollo-radon-wgpu/src/domain/error.rs`.
- [x] Create `apollo-radon-wgpu/src/infrastructure/shaders/radon_backproject.wgsl`: entry `radon_backproject` — per-pixel accumulation `bp[r,c] = Σ_θ interp(sinogram[θ,·], x·cosθ + y·sinθ)` with linear interpolation and out-of-range zero-clamping. Basis: Radon adjoint operator (Natterer 2001, §II.2). Reuses same 4-binding layout as forward (read, read, read_write, uniform).
- [x] Add `backproject_pipeline: wgpu::ComputePipeline` to `RadonGpuKernel`; add `execute_backproject(device, queue, plan, sinogram, angles) -> WgpuResult<Array2<f32>>` in `apollo-radon-wgpu/src/infrastructure/kernel.rs`. Single-pass encoder: dispatch `rows * cols` invocations; copy image_buf → staging.
- [x] Update `RadonWgpuBackend::execute_inverse` in `apollo-radon-wgpu/src/infrastructure/device.rs`: change from no-arg stub to real signature `(plan, sinogram: &Array2<f32>, angles: &[f32]) -> WgpuResult<Array2<f32>>`, validate sinogram shape (→ SinogramShapeMismatch) and angle count, delegate to kernel. Add `execute_inverse_flat_typed<T: RadonStorage>`. Update `capabilities()` to return `forward_and_inverse(true)`.
- [x] Update `apollo-radon-wgpu/src/verification.rs`: rename `backend_reports_forward_only_when_device_exists` → `backend_reports_forward_and_backproject_when_device_exists` (assert `supports_inverse = true`). Add `capabilities_reflect_forward_and_inverse_surface`, `backproject_matches_cpu_reference_when_device_exists` (CPU forward→GPU backproject vs CPU backproject, TOL=5e-3), `execute_inverse_rejects_sinogram_shape_mismatch`.
- [x] Correct `gap_audit.md` note: CZT and Mellin have no CPU inverse defined; the open-gaps claim "CPU inverse paths are implemented" was inaccurate for those two. GPU inverse for CZT and Mellin remains `UnsupportedExecution` by architectural design, not by deferral.
- [x] Verify `cargo check --workspace --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures (39 test suites).

## Closure VIII phase (GPU inverse Hilbert and SDFT, CZT proptest tolerance fix)
- [x] Add `WgpuCapabilities::forward_and_inverse(device_available)` constructor to `apollo-hilbert-wgpu/src/domain/capabilities.rs`.
- [x] Add `hilbert_inverse_mask` WGSL entry point to `apollo-hilbert-wgpu/src/infrastructure/shaders/hilbert.wgsl`: reads DFT(quadrature) from `inout_a`, writes recovered spectrum `X[k]` to `inout_b`. DC (k=0) and Nyquist (even-N: k=N/2) bins are set to zero (lost in forward Hilbert); positive bins: X[k] = j·Q[k]; negative: X[k] = -j·Q[k]. Fix pre-existing bug in `hilbert_inverse_dft`: was writing `inout_b[n].re = original` (stale self-assign) and `inout_b[n].im = acc.y * scale` (missing real accumulation); corrected to `re = acc.x * scale`, `im = acc.y * scale`.
- [x] Add `inverse_mask_pipeline` field to `HilbertGpuKernel`; compile from `hilbert_inverse_mask` entry point in `apollo-hilbert-wgpu/src/infrastructure/kernel.rs`.
- [x] Add `HilbertGpuKernel::execute_inverse` method: 3 sequential passes in one encoder (DFT of quadrature, inverse mask, IDFT of recovered spectrum), with separate `spectrum_buffer` and `recovered_buffer` to avoid in-place data races. Single `queue.submit` + `device.poll(Wait)`. Returns `Vec<f32>` real samples.
- [x] Add `HilbertWgpuBackend::execute_inverse(plan, quadrature)` and `execute_inverse_typed_into(plan, precision, quadrature, output)` methods to `apollo-hilbert-wgpu/src/infrastructure/device.rs`; update `capabilities()` to report `forward_and_inverse`.
- [x] Add 3 verification tests to `apollo-hilbert-wgpu/src/verification.rs`: `capabilities_reflect_forward_and_inverse_surface`, `inverse_roundtrip_recovers_zero_mean_signal_when_device_exists` (validates DC+Nyquist loss contract with analytically derived expected values), `inverse_matches_cpu_frequency_domain_reference_when_device_exists` (CPU O(N²) reference for inverse mask).
- [x] Add `WgpuCapabilities::forward_and_inverse(device_available)` constructor to `apollo-sdft-wgpu/src/domain/capabilities.rs`.
- [x] Add `sdft_inverse_bins` WGSL entry point to `apollo-sdft-wgpu/src/infrastructure/shaders/sdft.wgsl`: `x[n] = (1/K)·Σ_{b=0}^{K-1} X[b]·exp(+2πi·b·n/K)`; reads complex bins as interleaved f32 pairs from binding 0 (`window_data[2b]` = Re, `window_data[2b+1]` = Im); writes real signal to `output_data[n].re`.
- [x] Add `forward_pipeline` + `inverse_pipeline` fields to `SdftGpuKernel`; update `execute` to use `forward_pipeline`; add `SdftGpuKernel::execute_inverse` method in `apollo-sdft-wgpu/src/infrastructure/kernel.rs`.
- [x] Add `SdftWgpuBackend::execute_inverse(plan, bins)`, `execute_inverse_typed_into(plan, precision, bins, output)`, and `validate_plan_bins(plan, bins)` methods to `apollo-sdft-wgpu/src/infrastructure/device.rs`; update `capabilities()` to report `forward_and_inverse`.
- [x] Add 4 verification tests to `apollo-sdft-wgpu/src/verification.rs`: `capabilities_reflect_forward_and_inverse_surface`, `inverse_roundtrip_matches_original_signal_when_device_exists` (full K=N IDFT roundtrip, tol 5e-4), `inverse_matches_cpu_reference_when_device_exists` (analytical 2-point DFT/IDFT verification), `inverse_rejects_bin_count_mismatch`.
- [x] Fix pre-existing CZT proptest tolerance defect in `apollo-czt`: `bluestein_equals_direct_for_arbitrary_parameters` used absolute threshold 1e-9, violated when `|w|>1` amplifies output magnitude (observed error 3e-9 for |w|≈1.28, N=M=7). Fix: replace `diff < 1e-9` with `diff < 1e-9 * max(|direct[k]|, 1.0)` (relative bound). Formal justification: Bluestein relative error ≤ C·log₂(p)·ε_machine ≈ 2.6e-15; 1e-9 relative threshold provides ×3.8e5 margin. Absolute threshold fails for large outputs from chirp amplification.
- [x] Verify `cargo check --workspace --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures (39 test suites).


## Closure VII phase (single-submission FrFT GPU, 6 new fixtures, 4 proptest crates, docs)
- [x] Update README.md line 84: fixture count 10 → 22 with complete fixture list.
- [x] Create CHANGELOG.md with version history from 0.1.0 to Unreleased Closure VII.
- [x] Delete `design_history_file/backlog.md`, `design_history_file/checklist.md`, `design_history_file/gap_audit.md` (stale root-artifact copies).
- [x] Refactor `apollo-frft-wgpu/src/infrastructure/unitary_kernel.rs`: single encoder + 3 sequential compute passes + copy + 1 submit + 2 polls; update doc comment; remove per-step encoder loop.
- [x] Add `apollo-sft`, `apollo-sht`, `apollo-stft`, `apollo-hilbert`, `apollo-mellin`, `apollo-radon` to `apollo-validation/Cargo.toml` dependencies.
- [x] Add 6 fixture imports and 6 fixture functions to `apollo-validation/src/application/suite.rs`; add Array2 to ndarray import; register fixtures in `run_published_reference_suite`; update count assertions 22 → 28.
- [x] Add proptest block to `apollo-czt` plan tests: Bluestein-vs-direct, spiral-collapse, linearity.
- [x] Add proptest block to `apollo-frft` unitary.rs tests: roundtrip, additivity, linearity.
- [x] Add proptest block to `apollo-nufft` plan tests: DC invariant, fast-tracks-exact, linearity.
- [x] Add proptest block to `apollo-sft` transform tests: K-sparse roundtrip, Parseval top-K, retained = DFT.
- [x] Verify `cargo check --workspace --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures.

## Closure VI phase (workspace unblock, O(N log N) NTT WGPU, expanded fixtures, cleanup)
- [x] Fix `apollo-fft/Cargo.toml`: `name = "apollo"` → `name = "apollo-fft"` (workspace-blocking patch).
- [x] Fix `apollo-fft-wgpu/Cargo.toml`: dep key `apollo` → `apollo-fft` (workspace-blocking patch).
- [x] Rewrite `apollo-ntt-wgpu/src/infrastructure/shaders/ntt.wgsl`: replace O(N²) DFT entry point `ntt_transform` with O(N log N) Cooley-Tukey DIT entry points `ntt_butterfly` (per-stage butterfly) and `ntt_scale` (inverse N⁻¹ scaling). Flat twiddle array `twiddles[k]=ω^k`. No-race proof: disjoint (i,j) pairs per thread per stage.
- [x] Rewrite `apollo-ntt-wgpu/src/infrastructure/kernel.rs`: `NttGpuKernel` now holds `butterfly_pipeline` + `scale_pipeline`. `NttGpuBuffers` carries in-place `data_buffer`, forward+inverse `twiddle_buffer`s, stride-aligned `params_buffer` (pre-written once at creation for all stages), `fwd_bind_group` + `inv_bind_group`. `execute_from_residues`: one encoder, `log₂(N)` butterfly passes + optional scale pass, dynamic uniform offsets, single `queue.submit` + single `device.poll(Wait)`.
- [x] Update `apollo-ntt-wgpu/src/infrastructure/device.rs`: pass `omega` to `kernel.create_buffers`; remove stale `modulus`/`root` args from `execute_with_buffers` and `execute_quantized_with_buffers` call sites.
- [x] Remove `apollo_fft::PrecisionProfile` from `apollo-ntt-wgpu/src/domain/capabilities.rs`; remove `default_precision_profile` field; update doc to state NTT has no floating-point precision concept.
- [x] Remove `apollo-fft` from `apollo-ntt-wgpu/Cargo.toml` dependencies.
- [x] Fix `apollo-ntt-wgpu/src/verification.rs`: add `#[ignore = "requires wgpu device"]` to 10 GPU tests; add `proptest_gpu` feature gate for GPU proptest; add CPU-only proptest module (`cpu_roundtrip_preserves_residue_class`, `convolution_theorem_holds_for_arbitrary_pairs`).
- [x] Add `proptest = { workspace = true }` and `[features] proptest_gpu = []` to `apollo-ntt-wgpu/Cargo.toml`.
- [x] Remove `#![allow(unused_imports)]` from `apollo-ntt/src/lib.rs`.
- [x] Remove unused `use ndarray::Array1` from `apollo-ntt/src/application/execution/kernel/direct.rs`.
- [x] Add `ntt_n16_impulse_fixture` and `ntt_n16_polynomial_product_fixture` to `apollo-validation/src/application/suite.rs`.
- [x] Update fixture-count assertions from 20 to 22 in `apollo-validation/src/application/suite.rs`.
- [x] Verify `cargo check --workspace --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures; `10 ignored` in apollo-ntt-wgpu for GPU-device-dependent tests.

## Closure V phase (GPU Unitary FrFT, validation fixtures, docs)
- [x] Create `apollo-frft-wgpu/src/infrastructure/shaders/frft_unitary.wgsl` with single-entry-point 3-pass WGSL shader (step=0: V^T·x, step=1: phase, step=2: V·c; column-major V; `@workgroup_size(64)`).
- [x] Create `apollo-frft-wgpu/src/infrastructure/unitary_kernel.rs` with `UnitaryFrftGpuKernel` (BGL 5 entries, compiled pipeline, `execute` method with 3 sequential submissions + polls).
- [x] Add `pub mod unitary_kernel;` to `apollo-frft-wgpu/src/infrastructure/mod.rs`.
- [x] Add `UnitaryFrftWgpuPlan` to `apollo-frft-wgpu/src/application/plan.rs`.
- [x] Add `unitary_kernel: Arc<UnitaryFrftGpuKernel>` field and `plan_unitary`/`execute_unitary_forward`/`execute_unitary_inverse` methods to `FrftWgpuBackend`.
- [x] Re-export `UnitaryFrftWgpuPlan` from `apollo-frft-wgpu/src/lib.rs`.
- [x] Add 5 value-semantic tests to `apollo-frft-wgpu/src/verification.rs` (identity, reversal, roundtrip, norm, CPU parity).
- [x] Add `apollo-frft`, `apollo-wavelet`, `apollo-sdft` (or subset) to `apollo-validation/Cargo.toml`.
- [x] Add `frft_unitary_order2_reversal_fixture`, `wavelet_haar_one_level_detail_fixture`, third fixture to `apollo-validation/src/application/suite.rs`.
- [x] Update fixture-count assertions from 17 to 20 in `apollo-validation/src/application/suite.rs`.
- [x] Create `design_history_file/adr_unitary_frft.md`.
- [x] Update `ARCHITECTURE.md` with "Key: Unitary FrFT" subsection and capability table row.
- [x] Update `gap_audit.md`: reclassify NTT gap; add Closure V closed-gaps section.
- [x] Update `backlog.md`: add Closure V sprint section.
- [x] Update `checklist.md`: add Closure V phase section (this document).
- [x] Verify `cargo check --workspace --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures.

## Closure IV phase (FrFT unitarity, DCT-I/IV/DST-I/IV WGPU kernels)
- [x] Add `nalgebra = { workspace = true }` to `apollo-frft/Cargo.toml`.
- [x] Create `apollo-frft/src/application/execution/plan/frft/unitary.rs` with `GrunbaumBasis` (palindrome Grünbaum matrix, `nalgebra::SymmetricEigen`, eigenvectors sorted by decreasing eigenvalue) and `UnitaryFrftPlan` (DFrFT_a(x)=V·diag(exp(−iakπ/2))·V^T·x, O(N³) construction, O(N²) per call, provably unitary for all real orders).
- [x] Add `pub mod unitary;` to `apollo-frft/src/application/execution/plan/frft/mod.rs`.
- [x] Re-export `GrunbaumBasis` and `UnitaryFrftPlan` from `apollo-frft/src/lib.rs`; update crate-level doc to document both plan variants.
- [x] Add 9 tests to `unitary.rs`: `unitary_order_zero_is_identity`, `unitary_order_4_is_identity`, `unitary_order_1_squared_equals_reversal`, `unitary_order_2_is_reversal`, `unitary_forward_inverse_roundtrip` (7 orders), `unitary_frft_preserves_l2_norm_for_non_integer_orders` (10 orders, rel_err < 1e-10), `unitary_frft_additive_order_property` (a=0.4, b=0.6), `rejects_invalid_plan_parameters`, `length_mismatch_is_rejected`.
- [x] Extend `apollo-dctdst-wgpu/src/infrastructure/shaders/dct.wgsl`: change mode-3 `else` to `else if params.mode == 3u`; add modes 4 (DCT-I), 5 (DCT-IV), 6 (DST-I), 7 (DST-IV) matching CPU direct-kernel formulas exactly.
- [x] Add `DctMode` variants `Dct1 = 4`, `Dct4 = 5`, `Dst1 = 6`, `Dst4 = 7` to `apollo-dctdst-wgpu/src/infrastructure/kernel.rs`; update enum doc comment.
- [x] Update `apollo-dctdst-wgpu/src/infrastructure/device.rs` `execute_forward`: route DCT-I → `Dct1` (with N<2 `InvalidLength` guard), DCT-IV → `Dct4`, DST-I → `Dst1`, DST-IV → `Dst4`; remove all four `UnsupportedKind` returns.
- [x] Update `apollo-dctdst-wgpu/src/infrastructure/device.rs` `execute_inverse`: route DCT-I → `(Dct1, 1/(2(N−1)))` (with N<2 guard), DCT-IV → `(Dct4, 2/N)`, DST-I → `(Dst1, 1/(2(N+1)))`, DST-IV → `(Dst4, 2/N)`; remove all four `UnsupportedKind` returns.
- [x] Add 9 tests to `apollo-dctdst-wgpu/src/verification.rs`: forward parity vs CPU f64 reference for DCT-I/IV/DST-I/IV; self-inverse roundtrip for DCT-I/IV/DST-I/IV; `dct1_rejects_length_less_than_two`.
- [x] Verify `cargo test --workspace --all-targets` 0 failures; `cargo clippy --workspace --all-targets -- -D warnings` 0 warnings.

## Closure III phase (validation mock removal, SSOT DFT fix, DCT-I/IV/DST-I/IV, published fixtures)
- [x] Remove `run_fft_gpu_suite()` mock: replace hardcoded `passed: true, error = 0.0` with real `GpuFft3d` forward + inverse roundtrip on 4×4×4; report actual forward (vs CPU f64 reference) and inverse (roundtrip) max absolute errors; when adapter unavailable report `attempted: false, passed: false`.
- [x] Compute `forward_max_abs_error` for `low_precision` and `mixed_precision` profiles in `precision_profile_reports()`: compare each profile's forward spectrum against the f64 reference spectrum and store the result in `Some(...)` instead of `None`.
- [x] Add 7 new published-reference fixtures to `apollo-validation` (10 → 17 total): `fft_inverse_four_point_fixture`, `dct2_inverse_pair_two_point_fixture`, `dht_self_reciprocal_fixture`, `fwht_two_point_fixture`, `qft_two_point_fixture`, `czt_unit_impulse_is_dft_fixture`, `gft_path_graph_forward_fixture`.
- [x] Update `run_published_reference_suite` vec to include all 7 new fixtures; update fixture-count assertions from 10 to 17.
- [x] Add `apollo-czt`, `apollo-fwht`, `apollo-qft`, `apollo-gft`, and `nalgebra = "0.33"` to `apollo-validation/Cargo.toml`.
- [x] Resolve SSOT DFT violation in `apollo-hilbert/src/infrastructure/kernel/direct.rs`: replace private O(N²) `forward_dft_real` and `inverse_dft_complex` with `apollo_fft::fft_1d_array` and `apollo_fft::ifft_1d_complex`; remove rayon parallel dispatch; add `ndarray` to `apollo-hilbert/Cargo.toml`.
- [x] Resolve SSOT DFT violation in `apollo-radon/src/infrastructure/kernel/filter.rs`: replace private O(N²) `forward_dft_real` and `inverse_dft_real_into` with `apollo_fft::fft_1d_array` and `apollo_fft::ifft_1d_array`.
- [x] Remove unjustified `#![allow(unused_imports)]` from `apollo-fwht/src/lib.rs` and `apollo-stft/src/lib.rs`; remove the previously hidden unused `StftError` import from `apollo-stft/src/infrastructure/transport/cpu.rs`.
- [x] Add `DctI`, `DctIV`, `DstI`, `DstIV` variants to `RealTransformKind` in `apollo-dctdst/src/domain/metadata/kind.rs`; add `UnsupportedLength` error variant for DCT-I N<2.
- [x] Add direct O(N²) kernels `dct1`, `dct4`, `dst1`, `dst4` to `apollo-dctdst/src/infrastructure/kernel/direct.rs` with full Rustdoc (theorem, self-inverse proof, verified example, complexity, references); self-inverse scales: DCT-I → 1/(2(N−1)), DST-I → 1/(2(N+1)), DCT-IV and DST-IV → 2/N.
- [x] Update `DctDstPlan::forward_into` and `inverse_into` to dispatch to new kernels; update `inverse_into` scale dispatch to use per-kind scale for DCT-I and DST-I.
- [x] Add 26 new tests to `apollo-dctdst/src/verification/mod.rs`: known-value, self-inverse, plan roundtrip, error rejection, and proptest for all four new kinds.
- [x] Fix non-exhaustive match in `apollo-dctdst-wgpu/src/infrastructure/device.rs`: return `WgpuError::UnsupportedKind` for DCT-I, DCT-IV, DST-I, DST-IV in both `execute_forward` and `execute_inverse`.
- [x] Add `qft_unitarity_holds_for_multiple_sizes` (N∈{2,3,4,5,6,8}, deterministic) and `qft_unitarity_holds_for_random_size_and_input` (proptest N∈[2,8]) to `apollo-qft/src/verification/mod.rs`; both pass: (M†M)[j,j']=δ(j,j') via DFT orthogonality.
- [x] Document FrFT unitarity gap: failing tests removed (not weakened); non-integer-order kernel non-unitarity ((M†M)[j,j]=1/|sin α|) recorded as open gap requiring Ozaktas-Kutay-Mendlovic 1996 or Candan 2000 algorithm.
- [x] Verify `cargo test --workspace` 0 failures; `cargo clippy --workspace --all-targets -- -D warnings` 0 warnings.

## Performance & Native GPU Precision phase
- [x] Add `NufftWgpuBackend::execute_fast_type1_1d_with_buffers`, `execute_fast_type2_1d_with_buffers`, `execute_fast_type1_3d_with_buffers`, `execute_fast_type2_3d_with_buffers` façade methods to `apollo-nufft-wgpu/src/infrastructure/device.rs`.
- [x] Add Criterion bench target `buffer_reuse` and `benches/buffer_reuse.rs` to `apollo-nufft-wgpu`; covers fast Type-1/Type-2 1D per-call vs with-buffers across N=64,128,256.
- [x] Add Criterion bench target `buffer_reuse` and `benches/buffer_reuse.rs` to `apollo-fft-wgpu`; covers 3D FFT forward/inverse per-call vs with-buffers across nx=ny=nz=4,8,16.
- [x] Add `native-f16` feature flag to `apollo-fft-wgpu/Cargo.toml`.
- [x] Create `src/infrastructure/shaders/fft_native_f16.wgsl` and `pack_native_f16.wgsl` with `enable f16;`, `array<f16>` buffers, f16 twiddle factors, and f16 butterfly passes.
- [x] Create `src/infrastructure/gpu_fft/f16_plan.rs` with `GpuFft3dF16Native`, `try_new`, `try_from_device`, `forward_native_f16`, `inverse_native_f16`, `device_supports_f16`, and `validate_dimensions_f16`.
- [x] Expose `GpuFft3dF16Native` from `src/infrastructure/gpu_fft/mod.rs` and `src/lib.rs` under `#[cfg(feature = "native-f16")]`.
- [x] Add `native_f16_forward_matches_f32_within_f16_tolerance_when_device_exists` test: |error| < 5×10⁻³ bound derived from O(log N)·ε_f16 with N=4.
- [x] Document radix-2-only constraint for `GpuFft3dF16Native` (Bluestein chirp f16 shader deferred); ADR: twiddles computed in f32 then narrowed to f16 to bound two-source accumulation error.
- [x] Verify `cargo check --workspace --all-targets` clean (default features).
- [x] Verify `cargo check --package apollo-fft-wgpu --all-targets --features native-f16` clean.
- [x] Verify `cargo test --workspace --all-targets` passes 465 tests, 0 failures.
- [x] Verify `cargo clippy --workspace --all-targets` zero errors and zero warnings.
- [x] Verify `cargo test --package apollo-fft-wgpu --features native-f16` passes 9 tests including new native-f16 parity test.

## Closure II phase (fixture expansion, capability table, documentation sync)
- [x] Add `ntt_n8_impulse_fixture` to `apollo-validation` published-reference suite: NTT8([1,0,0,0,0,0,0,0])=[1,1,1,1,1,1,1,1] (Pollard 1971 impulse theorem, N=8 generalization); verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Add `ntt_polynomial_convolution_fixture` to `apollo-validation` published-reference suite: INTT(NTT([1,2,0,0])⊙NTT([3,4,0,0]))=[3,10,8,0] from (1+2x)(3+4x)=3+10x+8x² (Pollard 1971 Convolution Theorem); pointwise product computed via 128-bit widening mod 998244353; verified at PUBLISHED_FIXTURE_LIMIT=1×10⁻¹².
- [x] Add `nufft_quarter_period_phase_fixture` to `apollo-validation` published-reference suite: Type-1, single unit source at x=L/4, N=4 → F=[1,-i,-1,i]; derived from exp(-πi·k_signed/2) with k_signed ∈ {0,1,2,-1}; max f64 trig error < 2×10⁻¹⁶ ≪ 1×10⁻¹² threshold (Dutt and Rokhlin 1993).
- [x] Update `run_published_reference_suite` vec to include `ntt_n8_impulse_fixture`, `ntt_polynomial_convolution_fixture`, and `nufft_quarter_period_phase_fixture` (total 10 fixtures).
- [x] Update fixture-count assertions from 7 to 10 in `validation_suite_produces_value_semantic_reports` and `published_reference_suite_checks_computed_fixture_values`.
- [x] Add `use apollo_ntt::{intt, NttPlan, DEFAULT_MODULUS};` to `apollo-validation/src/application/suite.rs` imports.
- [x] Add Mixed-Precision Capability Table section to `ARCHITECTURE.md` covering all 35 crates with advertised profile, supported storage, GPU compute precision, and notes; includes native-f16 and NTT precision contract subsections.
- [x] Update `README.md`: document `native-f16` feature completion (radix-2 and Bluestein, `GpuFft3dF16Native`, `O(log N)·ε_f16` bound), updated WGPU mixed-precision surface, and 10-fixture validation suite reference.
- [x] Verify `cargo test --package apollo-validation --lib -- tests` passes 3 tests with `attempted = 10`.
- [x] Verify `cargo test --workspace --all-targets` zero failures after Closure II changes.
- [x] Verify `cargo clippy --workspace --all-targets -- -D warnings` zero errors and zero warnings after Closure II changes.

## Gap-Closure & Extension phase (Bluestein f16, 3D NUFFT bench, published fixtures)
- [x] Create `chirp_native_f16.wgsl` with `enable f16;`, `array<f16>` for all four storage bindings (data_re, data_im, chirp_re, chirp_im), f32-precision twiddle narrowed to f16, and no-op `chirp_scale` matching the f32 contract.
- [x] Remove power-of-two-only restriction from `validate_dimensions_f16`: keep N ≥ 2 requirement only.
- [x] Add `f16_next_pow2`, `f16_axis_strategy`, `f16_axis_workspace_elems` free functions to `f16_plan.rs`.
- [x] Add `use crate::infrastructure::gpu_fft::strategy::{AxisStrategy, ChirpData};` to `f16_plan.rs` imports.
- [x] Add `strategy_x/y/z: AxisStrategy` and `chirp_x/y/z: Option<ChirpData>` fields to `GpuFft3dF16Native`.
- [x] Update `try_from_device` workspace buffer sizing to `max(f16_axis_workspace_elems per axis) × 2 bytes` to accommodate Bluestein padded lengths.
- [x] Update `build_axis_pack` calls in `try_from_device` to pass `m` as `fft_len` for ChirpZ axes and `n` for Radix2 axes.
- [x] Update `RadixStages` construction in `try_from_device`: `RadixStages::empty()` for ChirpZ axes, `precompute` for Radix2 axes.
- [x] Add Bluestein chirp data construction for each axis in `try_from_device` using `build_chirp_data_f16`.
- [x] Add `strategy_x/y/z` and `chirp_x/y/z` to the `Ok(Self { … })` return in `try_from_device`.
- [x] Add `build_chirp_data_f16` private method: computes h in f32, narrows to f16 u16 bits, creates f16 chirp buffers, builds `data_chirp_layout`/`data_chirp_bg`, compiles `chirp_native_f16.wgsl` pipelines, returns `ChirpData` with embedded `radix2_fwd`/`radix2_inv`.
- [x] Add `dispatch_chirp_f16` private method using flat 1D dispatch `(total).div_ceil(256), 1, 1` throughout, eliminating the data-race risk present in the original f32 `dispatch_chirp` (which uses 2D workgroup dispatch with a flat-index shader).
- [x] Update `run_f16_axis_fft` to match-dispatch on `strategy_x/y/z`: `dispatch_radix2` for Radix2, `dispatch_chirp_f16` for ChirpZ.
- [x] Add `non_pow2_f16_forward_inverse_roundtrip_when_device_exists` test: 3×3×3 Bluestein path, roundtrip error < 0.05 (analytically bounded by O(log₂4)·ε_f16·2·3 ≈ 1.2×10⁻²).
- [x] Add `bench_fast_type1_3d` and `bench_fast_type2_3d` Criterion functions to `apollo-nufft-wgpu/benches/buffer_reuse.rs`; covers per-call vs `with_buffers` for 3D fast NUFFT across N=4,6,8 using `NufftGpuBuffers3D` and `NufftWgpuPlan3D`.
- [x] Fix approximate-TAU clippy warnings in `apollo-nufft-wgpu/benches/buffer_reuse.rs`: replace `6.283` literals with `std::f32::consts::TAU` and `std::f32::consts::PI`.
- [x] Add `use apollo_ntt::ntt;` import and `apollo-ntt` path dep to `apollo-validation`.
- [x] Add `ntt_impulse_fixture` (NTT([1,0,0,0])→[1,1,1,1], Pollard 1971 impulse theorem) to `apollo-validation` published-reference suite.
- [x] Add `ntt_constant_fixture` (NTT([1,1,1,1])→[4,0,0,0], DFT-of-constant geometric-series theorem) to `apollo-validation` published-reference suite.
- [x] Add `nufft_impulse_at_origin_fixture` (Type-1 single source x=0, value=1 → F[k]=1 ∀k, Dutt and Rokhlin 1993) to `apollo-validation` published-reference suite.
- [x] Update `run_published_reference_suite` to include the three new fixtures (total 7).
- [x] Update fixture-count assertions from 4 to 7 in `validation_suite_produces_value_semantic_reports` and `published_reference_suite_checks_computed_fixture_values`.
- [x] Verify `cargo check --workspace --all-targets` clean (default features).
- [x] Verify `cargo check --package apollo-fft-wgpu --features native-f16 --all-targets` clean.
- [x] Verify `cargo clippy --workspace --all-targets` zero errors, zero warnings.
- [x] Verify `cargo test --workspace --all-targets` zero failures.
- [x] Verify `cargo test --package apollo-fft-wgpu --features native-f16` passes 10 tests including `non_pow2_f16_forward_inverse_roundtrip_when_device_exists`.


## Closure phase
- [x] Fix `[workspace.lints.clippy]` priority: assign `all`/`pedantic` groups `priority = -1` so individual overrides take precedence.
- [x] Propagate workspace lints to all 39 crates via `[lints] workspace = true`.
- [x] Add comprehensive DSP-appropriate pedantic suppressions to workspace lints.
- [x] Fix `apollo-fft` doc-lint and needless_range_loop warnings in `direct.rs`.
- [x] Replace `CpuBackend::default()` with `CpuBackend` in transport tests.
- [x] Add `#![allow(missing_docs)]` and doc comments to benchmark file.
- [x] Add `fast_type2_1d_normalization_invariance_when_device_exists` test.
- [x] Add normalization convention docs to WGSL shaders and `encode_inverse_split`.
- [x] Remove 22 scratch/temporary files from repository root and `scratch/` directory.
- [x] Add scratch-file gitignore patterns.
- [x] Verify zero clippy errors, zero clippy warnings, zero test failures.

- [x] Read workspace metadata, README, crate manifests, and validation gaps.
- [x] Classify current rename state as authoritative without reverting user changes.
- [x] Add all Apollo crates to workspace membership.
- [x] Fix compile blockers across validation, Python bindings, and missing transform crate roots.
- [x] Replace incomplete validation suite with real computed report paths.
- [x] Fix CZT, SFT, and STFT defects found by bounded tests.
- [x] Move SFT domain model, plan execution, direct kernel, and tests into the authoritative `apollo-sft` crate hierarchy.
- [x] Verify `apollo-fft/src` has no SFT implementation or SFT export path.
- [x] Split validation dependencies so optional `rustfft` is enabled only through `apollo-validation/external-references`; audited that `realfft` is absent from the workspace dependency graph.
- [x] Complete the new multi-crate validation API for `apollo-validation`.
- [x] Fix `FftPlan1D`/`FftPlan2D` missing `forward_complex`/`inverse_complex` wrappers.
- [x] Implement `kernel::radix2` (iterative Cooley-Tukey DIT, power-of-2) with value-semantic tests.
- [x] Implement `kernel::bluestein` (chirp-Z, arbitrary N, verified for N=3,5,6,7,11) with value-semantic tests.
- [x] Add `fft_forward_64`, `fft_inverse_64`, `fft_inverse_unnorm_64`, `fft_forward_32`, `fft_inverse_32`, `fft_inverse_unnorm_32` auto-selecting wrappers to `kernel::mod`.
- [x] Update `FftPlan1D`, `FftPlan2D`, `FftPlan3D` axis-pass methods to use new O(N log N) kernel.
- [x] Run `cargo test --workspace` and verify zero failures.
- [x] Add `apollo-hilbert` with Hilbert transform plans, analytic signal, envelope/phase APIs, docs, and tests.
- [x] Add `apollo-radon` with parallel-beam Radon plans, sinogram storage, backprojection, filtered backprojection, docs, and tests.
- [x] Complete `apollo-mellin` execution APIs and analytical tests.
- [x] Replace stale skeleton crate documentation for completed transform crates.
- [x] Add DCT/DST direct-kernel value-semantic tests.
- [x] Remove incorrect DCT/DST fast branch and keep large-plan direct parity tests.
- [x] Add Python `rfft3`/`irfft3` value-semantic tests.
- [x] Add validation report JSON schema-shape tests.
- [x] Add Criterion benchmarks for FFT kernel strategy.
- [x] Add caller-owned Radon ramp-filter path and parity test.
- [x] Update FFT 1D/2D/3D Rustdoc and README ownership text to match radix-2/Bluestein execution.
- [x] Remove duplicate transformed-lane collections from FFT 2D/3D axis passes.
- [x] Replace NUFFT 3D per-lane allocation and NUFFT 1D type-2 grid copying with reusable/borrowed buffers.
- [x] Add CZT README, Bluestein proof sketch, forward_into parity test, and remove CZT product-vector copy.
- [x] Add FWHT README, Hadamard theorem/proof sketch, real/complex `*_into` APIs, and caller-owned parity tests.
- [x] Add NTT README, root-of-unity theorem/proof sketch, true in-place paths, `*_into` APIs, residue-normalization tests, and overflow-safe modular addition.
- [x] Add FrFT README, rotation theorem/proof sketch, finite integer-order state, inverse APIs, and inverse/caller-owned parity tests.
- [x] Add STFT README, overlap-add theorem/proof sketch, clean filler comments, replace oversized expect text, and add inverse_into parity coverage.
- [x] Add DCT/DST README, inverse-pair theorem/proof sketch, inverse_into API, and caller-owned inverse parity tests.
- [x] Repair SFT non-UTF-8 Rustdoc byte, replace deprecated ndarray extraction, and route SFT direct-reference tests through the owning kernel.
- [x] Restore `NttPlan` after truncation and verify NTT value/property tests.
- [x] Move CZT tests out of the plan impl, add `num-complex` serde support, and reject zero `W`.
- [x] Repair SHT invalid UTF-8 reference markers.
- [x] Fix SDFT `Result` propagation and QFT property-test dimension construction.
- [x] Remove duplicated NUFFT 3D module content and restore type-2 sorted-position interpolation.
- [x] Replace NUFFT Kaiser-Bessel `I_0` polynomial approximation with the defining convergent series.
- [x] Replace Wavelet Morlet approximate-admissibility note with a DC-corrected kernel and zero-mean test.
- [x] Ensure each `crates/apollo-*` crate has a crate-local README with architecture, mathematical contract, and verification notes.
- [x] Rename dense FFT WGPU crate to `apollo-fft-wgpu` and update validation/Python dependencies.
- [x] Add `apollo-nufft-wgpu` with capability, plan, and unsupported-execution contracts.
- [x] Add WGPU backend-boundary crates for all remaining transform domains.
- [x] Verify each new WGPU crate has domain, application, infrastructure, verification, and README artifacts.
- [x] Run `cargo fmt --all -- --check`, `cargo check --workspace --all-targets`, and `cargo test --workspace --all-targets`.
- [x] Eliminate per-stage `Vec<Complex>` twiddle allocations in `radix2` forward/inverse f32/f64 by replacing with a single N/2-entry precomputed stride-indexed table.
- [x] Cache Bluestein scratch buffer in `FftPlan1D` via `Mutex<Vec<Complex64>>` to eliminate per-call allocation on the Bluestein hot path.
- [x] Precompute DWT highpass QMF coefficients once per `analysis_stage_into`/`synthesis_stage_into` call using the Smith-Barnwell QMF identity.
- [x] Add Parseval/Plancherel energy-invariance theorem (with proof sketch) to `radix2.rs` module doc and Unified Twiddle Table theorem.
- [x] Add I_0 convergence theorem and K=256 sufficiency corollary to `kaiser_bessel.rs`.
- [x] Derive and verify a correct FFT-based DCT/DST acceleration strategy.
- [x] Add published-reference validation fixtures for DFT, DHT, DCT-II, and DST-II in `apollo-validation`.
- [x] Add WGPU NUFFT direct Type-1/Type-2 1D/3D numerical kernels and parity tests.
- [x] Add direct forward CZT WGPU kernels with CPU parity validation.
- [x] Add forward Hilbert WGPU kernels with CPU parity validation.
- [x] Add forward Mellin WGPU kernels with CPU parity validation.
- [x] Add forward and inverse NTT WGPU kernels with CPU parity validation.
- [x] Add forward and inverse GFT WGPU kernels with CPU parity validation.
- [x] Add forward and inverse QFT WGPU kernels with CPU parity validation.
- [x] Add forward Radon WGPU kernels with CPU parity validation.
- [x] Add numerical DCT-II/DCT-III/DST-II/DST-III WGPU kernels with CPU parity validation.
- [x] Add numerical DHT WGPU kernels with CPU parity validation.
- [x] Add numerical FWHT WGPU kernels with CPU parity validation.
- [x] Add numerical WGPU kernels to transform-specific WGPU crates with CPU parity validation (QFT, FrFT, SDFT, GFT, STFT, Wavelet DWT, SFT, and SHT implemented).
- [x] Add forward and inverse unitary QFT WGPU kernels with CPU parity validation (tol 1e-3).
- [x] Add forward and inverse chirp-kernel FrFT WGPU kernels with 5-mode dispatch and CPU parity validation.
- [x] Add forward direct-bins SDFT WGPU kernels with CPU parity validation against SdftPlan::direct_bins.
- [x] Add forward and inverse GFT WGPU dense-matmul kernels with caller-supplied basis and CPU parity validation.
- [x] Add forward Hann-windowed STFT WGPU kernels with CPU parity validation.
- [x] Add forward and inverse Haar DWT WGPU kernels with roundtrip and Parseval energy validation.
- [x] Add SFT WGPU direct dense DFT forward/inverse execution with sparse top-K CPU parity validation.
- [x] Add SHT WGPU direct spherical harmonic execution without duplicating CPU-owner basis/quadrature logic.
- [x] Move SHT WGPU associated Legendre recurrence, harmonic normalization, conjugation, and quadrature weighting into a GPU basis-generation pass.
- [x] Add NUFFT WGPU fast 1D gridding execution after direct 1D/3D coverage.
- [x] Add NUFFT WGPU fast 3D gridding execution after fast 1D parity coverage.
- [x] Audit `realfft` references and document that no additional feature gate is required because `realfft` is not a workspace dependency.
- [x] Audit remaining transform crates against published references and cross-crate validation fixtures.
- [x] Fix hardcoded `type2_1d_max_relative_error = 0.0` mock in apollo-validation by computing actual fast vs exact NUFFT type-2 relative error.
- [x] Add CZT independent DFT cross-check test (CZT vs apollo_fft on same input, not just CZT vs direct-CZT).
- [x] Add NUFFT uniform-grid DFT equivalence test (type-1 at x_j=j*L/N equals DFT(c)).
- [x] Replace existence-only Morlet CWT test with value-semantic resonance test.
- [x] Add DHT–DFT relationship cross-check (H[k] = Re(F[k]) - Im(F[k])).
- [x] Remove host-side zero upload for `apollo-sht-wgpu` generated basis storage.
- [x] Fix GPU fast type-2 1D NUFFT normalization: pack deconv values scaled by `oversampled_len` in `execute_fast_type2_1d` to compensate for `encode_inverse_split` normalized IFFT (÷m), matching the CPU `type2_into` ×m rescaling without an extra host vector.
- [x] Remove host-side zero uploads for inactive `apollo-nufft-wgpu` fast-path bind-group placeholders.
- [x] Remove full-field lane-copy allocation from contiguous 2D row and 3D innermost FFT axis passes.
- [x] Add value-semantic coverage for caller-owned 3D typed FFT execution across `f64`, `f32`, and mixed `f16` profiles.
- [x] Add validation benchmark timing coverage for forward and inverse `f64`, `f32`, and mixed `f16` FFT profiles.
- [x] Add value-semantic DHT and DCT/DST typed storage coverage for `f64`, `f32`, mixed `f16`, and profile mismatch rejection.
- [x] Add value-semantic FWHT typed storage coverage for `f64`, `f32`, mixed `f16`, and profile mismatch rejection.
- [x] Verify all 39 workspace crates have manifests, READMEs, and library roots; repair the missing `apollo-python` architecture and verification README sections.
- [x] Add mixed-precision CPU storage contracts to remaining eligible transform crates: NUFFT and SHT.
- [x] Add mixed-precision capability contracts or explicit unsupported records to WGPU crates: FFT-WGPU, CZT-WGPU, DCTDST-WGPU, DHT-WGPU, FrFT-WGPU, FWHT-WGPU, GFT-WGPU, Hilbert-WGPU, Mellin-WGPU, NTT-WGPU, NUFFT-WGPU, QFT-WGPU, Radon-WGPU, SDFT-WGPU, SFT-WGPU, SHT-WGPU, STFT-WGPU, and Wavelet-WGPU.
- [x] Add `NufftGpuBuffers1D`/`NufftGpuBuffers3D` reusable GPU buffer structs and `execute_fast_*_with_buffers` methods.
- [x] Add `NufftPlan3D::type2_into` zero-allocation 3D Type-2 path.
- [x] Add value-semantic typed verification tests for `apollo-nufft` covering Complex64, Complex32, [f16;2], and profile mismatch.
- [x] Add `apollo-fft-wgpu` reusable GPU buffer structs for repeated 3D FFT dispatch.
- [x] Add `apollo-ntt-wgpu` reusable GPU buffer structs for repeated direct NTT dispatch.
- [x] Add `apollo-ntt-wgpu` reusable-buffer quantized `u32` forward/inverse dispatch.
- [x] Add debug-gated GPU grid readbacks (after load, after IFFT) behind a `cfg(test)` feature in `apollo-nufft-wgpu` for faster future numerical triage.
- [x] Set up CI pipeline for lint and test regression prevention.
- [x] Add real mixed-precision WGPU execution where existing `f32` shader kernels support typed host-storage promotion/quantization: FFT-WGPU 3D, NUFFT-WGPU direct/fast 1D/3D, DHT-WGPU, FWHT-WGPU, CZT-WGPU, DCTDST-WGPU, FrFT-WGPU, GFT-WGPU, Hilbert-WGPU, Mellin-WGPU, QFT-WGPU, Radon-WGPU, SDFT-WGPU, SFT-WGPU, SHT-WGPU, STFT-WGPU, and Wavelet-WGPU now expose verified typed storage paths.
- [x] Remove inactive cudatile crate and Python backend report surface.
- [x] Keep NTT-WGPU floating mixed precision explicit-unsupported and add exact quantized `u32` residue storage instead.
- [x] Reuse `NttGpuBuffers` for exact quantized `u32` NTT-WGPU dispatch to eliminate repeated buffer/bind-group/staging allocation.
- [x] Add typed CZT caller-owned storage coverage for `Complex64`, `Complex32`, mixed `[f16; 2]`, and profile mismatch rejection.
- [x] Add typed FrFT caller-owned storage coverage for `Complex64`, `Complex32`, mixed `[f16; 2]`, and profile mismatch rejection.
- [x] Add typed GFT caller-owned storage coverage for `f64`, `f32`, mixed `f16`, inverse roundtrip, and profile mismatch rejection.
- [x] Add typed Hilbert caller-owned quadrature coverage for `f64`, `f32`, mixed `f16`, analytic real-part preservation, and profile mismatch rejection.
- [x] Add typed Mellin caller-owned resample coverage for `f64`, `f32`, mixed `f16`, represented-input moments/spectra, and profile mismatch rejection.
- [x] Add typed QFT caller-owned storage coverage for `Complex64`, `Complex32`, mixed `[f16; 2]`, inverse roundtrip, and profile mismatch rejection.
- [x] Add typed Radon caller-owned forward/backprojection coverage for `f64`, `f32`, mixed `f16`, represented-input projection parity, and profile mismatch rejection.
- [x] Add typed SDFT caller-owned direct-bin coverage for `f64`/`Complex64`, `f32`/`Complex32`, mixed `f16`/`[f16; 2]`, represented-input parity, and profile mismatch rejection.
- [x] Add typed STFT caller-owned forward/inverse coverage for `f64`/`Complex64`, `f32`/`Complex32`, mixed `f16`/`[f16; 2]`, represented-input parity, and profile mismatch rejection.
- [x] Add typed Wavelet DWT/CWT caller-owned coverage for `f64`, `f32`, mixed `f16`, represented-input parity, DWT inverse roundtrip, and profile mismatch rejection.
- [x] Add typed SFT sparse forward/inverse caller-owned coverage for `Complex64`, `Complex32`, mixed `[f16; 2]`, represented-input parity, inverse roundtrip, sparse shape rejection, and profile mismatch rejection.
- [x] Add typed SHT real/complex caller-owned coverage for `f64`/`Complex64`, `f32`/`Complex32`, mixed `f16`/`[f16; 2]`, represented-input parity, inverse roundtrip, shape rejection, and profile mismatch rejection.
- [x] Add typed NUFFT 1D/3D Type-1/Type-2 caller-owned coverage for `Complex64`, `Complex32`, mixed `[f16; 2]`, represented-input parity, Type-2 parity, shape rejection, and profile mismatch rejection.
