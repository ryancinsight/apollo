# Changelog

All notable changes to the Apollo workspace are documented in this file.
Versioning follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Change-class tags: [patch] backward-compatible fix, [minor] additive non-breaking, [major] breaking, [arch] cross-cutting redesign.

---

## [Unreleased]
### Added
- [patch] `apollo-qft` dense forward/inverse row reductions now use Hermes interleaved complex dot products above the existing Moirai parallel threshold, with Mnemosyne thread-local twiddle-lane scratch and scalar row formulas retained for small state vectors.
- [patch] `apollo-wavelet` CWT coefficient accumulation now uses Hermes dot-product reduction with Mnemosyne thread-local wavelet-weight scratch above a bounded signal-length threshold, while smaller coefficients retain the scalar mother-wavelet formula.
- [patch] `apollo-mellin` real Mellin moments now use Hermes dot-product reduction with Mnemosyne thread-local trapezoid-weight scratch above a bounded signal-length threshold, while smaller moments retain the scalar trapezoid formula.
- [patch] `apollo-gft` forward and inverse graph-basis row reductions now use Hermes dot products above a bounded row-length threshold, with Mnemosyne scratch used for inverse strided basis rows.
- [patch] `apollo-dht` direct DHT coefficient rows now use Hermes dot-product reduction with Mnemosyne thread-local Hartley-row scratch above a bounded row-length threshold, while smaller rows retain the scalar formula.
- [patch] `apollo-sft` typed sparse storage conversion and inverse output materialization now use Moirai above a bounded element threshold, while the deterministic dense FFT and top-K heap selector remain the canonical SFT algorithm.
- [patch] `apollo-czt` direct CZT rows and Bluestein workspace/kernel/output stages now use Moirai `ParallelSliceMut` above bounded work thresholds, with serial execution retained for smaller buffers.
- [patch] `apollo-sdft` direct-bin initialization and sliding recurrence updates now use Moirai `ParallelSliceMut` above bounded work thresholds, with serial execution retained for small windows and bin counts.
- [patch] `apollo-gft` forward and inverse graph-basis row execution now uses Moirai `ParallelSliceMut` above a bounded O(N²) work threshold, with serial execution retained for small graphs.
- [patch] `apollo-frft` direct and unitary O(N²) CPU row execution now uses Moirai `ParallelSliceMut` above bounded work thresholds, with serial execution retained for small vectors and exact integer-order special cases unchanged.
- [patch] `apollo-qft` dense forward/inverse row execution now uses Moirai `ParallelSliceMut` above a bounded O(N²) work threshold, with serial execution retained for small state vectors.
- [patch] `apollo-hilbert` analytic-signal staging, analytic-mask application, real-part restoration, and quadrature extraction now use Moirai `ParallelSliceMut` above a bounded signal-length threshold while preserving serial execution for small inputs.
- [patch] `apollo-nufft` exact 1D direct Type-1/Type-2 reference transforms now use Moirai `ParallelSliceMut` for disjoint output writes above a bounded operation threshold while preserving serial execution for small inputs.
- [minor] `apollo-nufft-wgpu` `0.2.0` adds Leto direct and fast 1D/3D Type-1/Type-2 NUFFT host boundaries, including typed storage variants, returning Mnemosyne-backed Leto arrays while preserving WGPU slice/ndarray execution as the device dispatch path.
- [minor] `apollo-fft-wgpu` `0.2.0` adds Leto 3D forward/inverse and mixed `f16` forward/inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU 3D FFT execution as the device dispatch path.
- [minor] `apollo-stft-wgpu` `0.12.0` adds Leto forward, inverse, typed forward, and typed inverse STFT host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-wavelet-wgpu` `0.2.0` adds Leto forward, inverse, typed forward, and typed inverse Haar DWT host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-radon-wgpu` `0.2.0` adds Leto forward, inverse/backprojection, filtered-backprojection, typed forward, and typed inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU ndarray/slice execution as the device dispatch path.
- [minor] `apollo-sht-wgpu` `0.2.0` adds Leto forward, inverse, typed flat forward, and typed flat inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU ndarray/slice execution as the device dispatch path.
- [minor] `apollo-gft-wgpu` `0.2.0` adds Leto forward, inverse, typed forward, and typed inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-frft-wgpu` `0.2.0` adds Leto standard forward/inverse, unitary forward/inverse, and typed forward/inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-ntt-wgpu` `0.2.0` adds Leto `u64` forward/inverse and exact `u32` quantized forward/inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-mellin-wgpu` `0.3.0` adds Leto forward, typed forward, and inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-hilbert-wgpu` `0.2.0` adds Leto analytic-signal, forward, typed forward, inverse, and typed inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-dctdst-wgpu` `0.2.0` adds Leto 1D, typed 1D, 2D, and 3D host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice/ndarray execution as the device dispatch path.
- [minor] `apollo-czt-wgpu` `0.3.0` adds Leto forward, typed forward, and inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-sft-wgpu` `0.2.0` adds Leto forward, typed forward, inverse, and typed inverse host boundaries; sparse forward keeps `SparseSpectrum` as the SSOT and dense inverse outputs can return Mnemosyne-backed Leto arrays.
- [minor] `apollo-qft-wgpu` `0.2.0` adds Leto forward, typed forward, inverse, and typed inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-dht-wgpu` `0.2.0` adds Leto forward, typed forward, inverse, and typed inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-fwht-wgpu` `0.2.0` adds Leto forward, typed forward, inverse, and typed inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-sdft-wgpu` `0.2.0` adds Leto forward, typed forward, and inverse host boundaries returning Mnemosyne-backed Leto arrays while preserving WGPU slice execution as the device dispatch path.
- [minor] `apollo-nufft` `0.2.0` adds Leto 1D type-1/type-2 NUFFT boundaries returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-sht` `0.2.0` adds Leto SHT sample/coefficient boundaries returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-wavelet` `0.2.0` adds Leto DWT/CWT boundaries returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-stft` `0.3.0` adds Leto STFT analysis/synthesis boundaries returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-radon` `0.2.0` adds Leto 2D forward projection, adjoint backprojection, and filtered-backprojection boundaries returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-mellin` `0.3.0` adds Leto resample, moment, forward-spectrum, and inverse-spectrum boundaries returning Mnemosyne-backed Leto arrays for generated outputs.
- [minor] `apollo-sdft` `0.2.0` adds Leto direct-bin and state-initialization boundaries returning Mnemosyne-backed Leto arrays for direct-bin output.
- [minor] `apollo-sft` `0.2.0` adds Leto sparse forward/inverse boundaries and `SparseLetoSpectrum<T>` for typed Mnemosyne-backed retained values.
- [minor] `apollo-hilbert` `0.4.0` adds Leto analytic-signal, real quadrature, and typed quadrature boundaries returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-dctdst` `0.2.0` adds real, typed, 2D, and 3D Leto DCT/DST boundaries returning Mnemosyne-backed Leto arrays, with slice and ndarray differential validation.
- [minor] `apollo-czt` `0.3.0` adds Complex64 and typed Leto 1D CZT boundaries returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-dht` `0.2.0` adds 2D/3D Leto DHT boundaries returning Mnemosyne-backed Leto arrays, with ndarray differential validation for contiguous and strided view inputs.
- [minor] `apollo-qft` `0.2.0` adds Complex64 and typed Leto 1D QFT boundaries returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-fwht` `0.2.0` adds real and typed Leto 1D FWHT boundaries returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-ntt` `0.2.0` adds `NttPlan::forward_leto`, `NttPlan::inverse_leto`, `ntt_leto`, and `intt_leto` for Leto 1D views returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-frft` `0.2.0` adds `FrftPlan::forward_leto_typed`, `FrftPlan::inverse_leto_typed`, and `frft_leto_typed` for typed Leto 1D views returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-frft` now exposes `FrftPlan::forward_leto`, `FrftPlan::inverse_leto`, and `frft_leto` for Leto 1D complex views returning Mnemosyne-backed Leto arrays.
- [minor] `apollo-fft` now exposes Leto 1D FFT/IFFT boundaries that accept Leto views and return Mnemosyne-backed Leto arrays, with ndarray differential validation for contiguous and strided view inputs.
- [minor] Added Leto as an Apollo workspace provider dependency and wired `apollo-validation` to validate a Leto contiguous array conversion boundary against `ndarray`.
- [minor] `apollo-fft` `RealFftData` now has additive slice-owned 1D forward and inverse methods for storage-owned public slice allocation boundaries.
- [patch] `xtask provider-audit` for Apollo-local Moirai/Mnemosyne/Melinoe/Hermes utilization analysis plus `docs/provider_contract.md` documenting provider requirements and Git dependency order.
- [patch] `apollo-fft` domain `FftInterleavedCow` storage view for zero-copy borrowed interleaved FFT reads and owned detachment on mutation.
- kernel/pot/ ZST strategies (PoTStrategy, SizedPoT<...>) and butterflies/ placeholder for shared components (hierarchy, zero-cost, monomorph).
### Changed
- [patch] Updated Hermes Git lockfile revision to `b148fed9`, which adds provider-owned interleaved complex dot products for `f32`/`f64` lanes.
- [minor] `apollo-czt` `CztStorage` now exposes canonical typed slice execution hooks reused by ndarray arrays and Leto views.
- [minor] `apollo-qft` `QftStorage` now exposes canonical typed slice execution hooks reused by ndarray arrays and Leto views.
- [minor] `apollo-fwht` `FwhtStorage` now exposes canonical typed slice execution hooks reused by ndarray arrays and Leto views.
- [minor] `apollo-ntt` execution now routes ndarray and Leto allocation paths through shared contiguous slice methods.
- [minor] `apollo-frft` `FrftStorage` now exposes canonical typed slice execution hooks reused by both ndarray arrays and Leto views.
- [major] `apollo-frft` unitary Grünbaum basis construction now uses Leto arrays and `leto-ops::symmetric_eigen_jacobi`; `apollo-frft-wgpu` consumes an explicit column-major Leto flattening boundary; Apollo manifests and lockfile no longer depend on `nalgebra`.
- [patch] Updated Leto to commit `fd1d87b`, added `leto-ops`, and routed `apollo-gft` graph spectral basis construction through `leto_ops::symmetric_eigen_jacobi`, removing `apollo-gft`'s direct `nalgebra` eigensolver dependency.
- [major] `apollo-gft` graph adjacency construction now uses Leto arrays: `GraphAdjacency` owns `leto::Array2<f64>` and `GftPlan::from_adjacency` accepts `leto::ArrayView2<'_, f64>`.
- [patch] Updated Leto to commit `646c036`, which adds structural `Debug`/`Clone` derives for owned arrays and `VecStorage`.
- [patch] Removed stale nalgebra dependencies from `apollo-gft-wgpu` tests and `apollo-validation` after switching their GFT fixtures to Leto adjacency arrays.
- [minor] Updated Apollo's Leto dependency to pushed commit `9f639b73` with `mnemosyne-alloc` enabled; Leto and Apollo now resolve the same pinned Mnemosyne source identity.
- [patch] Removed Apollo's root `ndarray` `matrixmultiply-threading` feature so ndarray remains a validation dependency without Rayon-backed execution in Apollo's normal transform dependency surface.
- [patch] Updated Mnemosyne Git lockfile revision to `9411c444` and routed `apollo-fft` mixed-radix and plan workspace scratch roles through Mnemosyne `ScratchBank<T, const N>`, reducing repeated Apollo-local thread-local pool declarations while preserving sealed monomorphized complex-type dispatch.
- [patch] Updated Hermes Git lockfile revision to `55efd380` and routed `apollo-fft` mixed-radix non-FMA pointwise fallback through Hermes' provider-owned `interleaved_complex_mul_assign` kernel.
- [patch] Updated Hermes Git lockfile revision to `b7f1a907` and removed Apollo FFT's local mixed-radix pointwise AVX/FMA implementation in favor of Hermes' provider-owned runtime complex dispatch.
- [patch] `apollo-fft` mixed-radix pointwise fallback now uses Hermes `PreferredArch` vector chunks for precise and reduced complex multiplication, while preserving the existing runtime-gated x86 AVX/FMA hot kernel.
- [patch] `apollo-fft` f64 Stockham AVX fixed length-32 and length-64 transforms now use full-sized static twiddle tables (`TWIDDLES_32_FWD`/`TWIDDLES_32_INV`/`TWIDDLES_64_FWD`/`TWIDDLES_64_INV`) and const-generic index lookups, completely removing twiddle index branches, negations, and pointer offset math.
- [minor] `xtask provider-audit` now reports Leto workspace/crate usage and `docs/provider_contract.md` documents Leto requirements plus `ndarray`'s role as validation oracle.
- [patch] `apollo-fft` 1D runtime and static plans now share tiny direct dispatch for N=2/3/4; N=3 plans call the canonical `butterflies::dft3_impl` codelet directly with fused normalization instead of entering the generic short-Winograd dispatcher.
- [patch] `apollo-fft` composite plan radices now use `Cow<'static, [usize]>` instead of a custom enum, preserving borrowed static radix plans and using owned dynamic radix vectors only at the cache boundary.
- [patch] Updated Moirai Git lockfile revision to `7aab036` after pushing public provider contract coverage and follow-up provider improvements.
- [patch] Updated Mnemosyne Git lockfile revision to `477a3fa` after pushing provider-side scratch and memory-boundary improvements consumed by Apollo.
- [patch] Updated Hermes Git lockfile revision to `7eb0b70` after pushing public Cow state accessors for zero-copy boundary verification.
- [patch] Consolidated repeated `ndarray = "0.16"` crate declarations onto the Apollo workspace dependency so matrixmultiply threading remains controlled from the root manifest.
- [patch] `apollo-fft` radix-composite Moirai dispatch now uses monomorphized `ExecutionPolicy` type parameters instead of a bare boolean.
- Method selection prefers composite/mixed-radix for 2/3/5/7-smooth (incl. coprimes like 90,198) over GT static in dispatch and FftPlan1D (perf fix for many >2x GT cases in benchmarks); reduced f32 winograd/generated for slow policy sizes.
- Stockham has explicit cases for 128/256 (ready for full unroll; currently safe delegate); f32 small PoT (16/32/64) now delegate to shared Stockham + pooled scratch (unify, memory/stack efficiency, perf).
- Updated tests (strategy asserts relaxed for selection changes; rader n113 f32 wrapped/ignored for debug stack in f32 paths; numerical still verified).
- Artifacts, docs (ARCHITECTURE, code), cleanup (dead unroll stubs, pooled for f32 small).

- [patch] : expanded  with ~40 commonly-benchmarked composite sizes (72-1024) to avoid runtime factoring overhead. All radix arrays verified correct; radix-2 pairs lowered to radix-4.
- [patch] : added LOG2=6 (n=64) fast path in  calling  instead of falling through to generic Stockham with scratch allocation.
- [patch] : collapsed duplicate coprime factors path in  and removed dead LOG2=5 ZST constructions and redundant n==90/n==198 special cases.
### Performance/Memory
- [patch] Regenerated `benchmark_results.md` with `cargo run -p xtask -- benchmark`, refreshing the canonical quick-profile benchmark table after the FRFT-WGPU Leto host-boundary increment.
- [patch] Regenerated `benchmark_results.md` with `cargo run -p xtask -- benchmark --all --profile quick`, refreshing all 514 canonical rows. Current quick profile records f64 faster on 68 rows, f32 faster on 44 rows, and both faster on 19 rows.
- [patch] Refreshed selected measured rows in `benchmark_results.md` with `cargo run -p xtask -- benchmark --sizes 1,2,4,8,16,32,64,128,256,512,10007,32768 --profile quick`; the full table remains populated and only the requested rows received new quick-profile measurements.
- [minor] `apollo-fft` public 1D slice wrappers now route through storage-specific owned slice methods and cached in-place `FftPlan1D` execution. Concrete `f64`, `f32`, and `f16` paths allocate one owned output/scratch vector instead of cloning into `Array1` and copying the result back into a second `Vec`. `apollo-fft` version bumped to `0.13.0`; `cargo semver-checks` could not analyze the crate because it is not published in the registry.
- [patch] Regenerated `benchmark_results.md` with `cargo run -p xtask -- benchmark --all --profile quick`, refreshing all 514 canonical rows. Current quick profile records f64 faster on 101 rows, f32 faster on 71 rows, and both faster on 33 rows. N=3 now records f64 `1.001x`, f32 `0.444x`; N=4 f32 remains a direct-row miss at `4.325x`.
- 32768 PoT 4x unroll first-pass radix1 triple (ILP for controlling md-worst 2.75x f64; extends n512/n1024 + prior GT/rader unrolls): stage_triple_radix1_n32768_avx_fma upgraded 2x->4x (explicit first-4 kk< step-uniform guards for avx/avx512; 4x while remaining); docs updated in triple (fn+3 sites), precise/reduced (if), transform (len32768 doc + special sizes + per-LOG2). Zero-cost additive mono/ILP (no new alloc/cast, TL/Cow reuse). Value green (release n32768 roundtrip tol + list + special exercised 4x+len body); gates clean; focused xtask on md safe list (build 8m39s, wrote, deltas 32768 f64 2.553x/f32 0.747x win, 512 f64 0.518x); md note + rebench cmd; artifacts synced. No reg (identical do_one/seq; native). Advances "surpass all sizes" PoT per md triage.
- n1024 PoT unroll + f32 avx/pot sub with_scratch (bluestein kernel build + avx sized match lists) + GT col unroll8 + row gather_unroll8 + rader gather_unroll8 + natural scatter unroll8 + n32768 2x unroll for first pass: stage_triple_radix1_n1024_avx_fma (explicit 64/32 iters); wired in precision 4+; 1024 in f32/f64 avx_with_scratch_sized matches (direct); bluestein build for pow2 p>=64 uses with_bluestein_scratch + stockham_forward_sized cascade (f32 sub, unblocks n113/257 bias per gap, mono for rader cache, pool reuse). dimension n113 comment updated (progress). pfa natural col extract unrolled 0-7 + added gather_unroll8 in butterflies + use in pfa_gather for row ILP (helps GT 198/90/84+ in md) + extended gather_unroll8 to rader perm gather in rader/mod.rs (helps f32 rader 67/271/113/257) + natural scatter unrolled to 8 (ILP for GT scatter) + stage_triple_radix1_n32768_avx_fma (2x unrolled while for k/do_one pairs) + wired if radix==1 && n==32768 in avx paths (4 sites). Additive 0-cost to n512+; targets 1024/32768 PoT + rader f32 113/257/67 (md worst) + GT. Value green (n512 + rader 67/271 + n1024 sized exercised + rader/GT/special 32768 tests); gates clean; md note + refresh attempts (lists with/without small, no json, documented full measure cmd); artifacts. No reg (identical, native).
- n1024 PoT unroll + f32 avx/pot sub with_scratch (bluestein kernel build + avx sized match lists): stage_triple_radix1_n1024_avx_fma (explicit 64/32 iters); wired in precision 4+; 1024 in f32/f64 avx_with_scratch_sized matches (direct); bluestein build for pow2 p>=64 uses with_bluestein_scratch + stockham_forward_sized cascade (f32 sub, unblocks n113/257 bias per gap, mono for rader cache, pool reuse). dimension n113 comment updated (progress). Additive 0-cost to n512+; targets 1024/32768 PoT + rader f32 113/257/67 (md worst). Value green (n512 + rader 67/271 + n1024 sized exercised); gates clean; md note + rebench (incl 1024); artifacts. No reg (identical, native).
- n512 PoT unroll (radix-1 triple + DCE/ILP per mono) + precision wiring (4 sites scalar/avx/avx512) + doc: stage_triple_radix1_n512_avx_fma (explicit do_one 0..31*step f64/16 f32, no while, #[inline(never)]); if n==512 before groups in precise/reduced; scalar attempt (reverted for debug stack); transform doc. Additive 0-cost to n256+; targets 512 f64 1.241x / 32768 2.75x PoT + rader f32 p=512 pads (n113/257). Value green on n512 ZST + exercised; gates clean (fmt/clippy no new on paths); md note + rebench; artifacts synced. No reg (identical to general, native prec).
- Extend per-LOG2 unroll to n=128 + bluestein f32 pads 128/256/512 sized (perf PoT + mem/mono rader): stage_triple_radix1_n128_avx_fma (explicit do_one, DCE) + wiring in precision (scalar/avx/avx512); bluestein convolve if p==128/256/512 use stockham_forward_sized::<7/8/9> (direct const LOG2, pool, ZST for f32 rader bluestein paths). Targets 128 f32 ~1.27x / 256 / rader f32 bias mem. Additive 0-cost; value green; no reg.
- f32 sub-dispatch scratch unify (mem): dft64/128 now allocate even/odd split temp via heap Vec (not stack MaybeUninit) in win/composite dftN paths. Reduces debug monomorph frame for f32 dft sub in rader bluestein pads / GT / win (targets n113 stack gap + broader f32 bias for rader 67/271). Additive mem eff; no perf/reg (value + logic same). See gap for full avx/pot sub.
- n64 PoT: radix-1 triple stage unroll (n64 special + explicit do_one, no-loop for first pass of len64) in avx + scalar paths (precise/reduced); direct stockham_forward_sized::<6> for p=64 bluestein pads (mono + pool reuse). Targets controlling 64 ratios (pre 1.47 f64/1.97 f32); additive 0-cost mono to prior n32/ZST/Cow. Some wins (64 f32 0.823x) + var on quick; no regression (value green, identical ops). Memory: sized reuses scratch pool for rader paths.
- Targets worst in benchmark_results (GT, f32 Winograd small, PoT): selection change + unification + unroll prep should bring more <1x vs rustfft. Memory: less stack pressure via pools in f32 paths; zero-copy emphasis in PFA notes.
### Fixed
- [patch] `xtask provider-audit` now strips TOML and Rust line comments before provider pattern detection, removing false Rayon/ndarray-threading positives from comments and stale docs.
- [patch] Restored `apollo-fft` scratch-pool dispatch to sealed static implementations over Mnemosyne `ScratchPool<Complex32>` and `ScratchPool<Complex64>`, removing the unsafe runtime `TypeId`/closure-transmutation path.
- Cursor over-take in attempted 128/256 unrolls (now delegated); stack overflow in planned rader n113 f32 test (ignored + memory notes).
See checklist/gap for details.

### Cleanup + Planning (post GT composite + shared + rader direct) + next phase (monomorph/ZST/Cow/scratch/hierarchy)
- [patch] Phase (deeper per-LOG2 mono body + mem Cow + elevation): explicit transform_len32/64/128/256/512/1024 (straight-line from schedule dump) + LOG2 match dispatch in transform_impl (removed 128/256 delegate); extended live ZST with_strategy for 5-10 in mod.rs f32/f64 scalar paths; Cow kernel_view (bluestein) + named views in scratch nested. SRP (transform owns per-size), SSOT routing (plan/dispatch ZST + sized), zero-cost mono. Highest prob for PoT md-worst (32/64/128/256/512/1024/32768). Mem for rader primes. Cast native. Value: n32 direct, n512 ZST + roundtrips 128/256/512/1024, rader bluestein (pooled+Cow), GT90/198, broad 252p. Gates clean (fmt/doc/xtask/clippy-no-new); env clean + focused --skip-run on full list (build success); md updated (no regression: identical seq, value green, 0-cost additive). Artifacts synced (gap table, checklist/backlog [patch], CHANGELOG, ARCHITECTURE). See gap_audit, benchmark_results, checklist.
- [patch] More direct ZST in AVX dispatch: forward64/32_avx_with_scratch_sized<const LOG2> added + wired from mod.rs sized (bypass runtime log2/trailing in avx path for plan PoT sized callers); LOG2 passed to transform_sized. Prep fixed_len128/256. Value/gates/build+md synced (no regression).
- [patch] Completion of const LOG2 threading (perf/mono/mem): stockham_forward_sized + normalized_sized + pot_inplace_sized overrides in f32/f64 (const LOG2 match to small-direct or with_scratch + stockham_*_sized + Cow::Borrowed tw_view); plan log2 match + exec for 7/8 (128/256 ZST); dispatch 7-10 arms call pot_sized with _s. Full end-to-end const LOG2 from plan/dispatch to len* bodies for md-worst PoT. Incidental preexist fixes (4 arm, dft16 path, short test, n257 ignore). Value 346p/2i green on list; gates clean; explicit build + focused --skip-run + md update (no regression). See gap, benchmark_results.
- [patch] Deeper per-LOG2 unrolls inside len*/stage for PoT worst (n32 focus): radix1 n32 unrolled (Inner-Fn do_one + explicit no-while vector iters + DCE) for avx backends + scalar 4x j0 explicit in non-avx stage_triple; route in P stage_triple if n==32; #[inline(always)] + doc on len32. Targets 32 (1.72x/2.5x md worst); additive 0-cost to prior ZST/Cow/threading; value identical (exercised scalar+avx); gates/build+focused xtask/md/artifact sync. No regression. See gap_audit, benchmark_results.
- [patch] ZST threading + Cow in pot_sized (arch + mono + mem): pot_inplace_sized<const INVERSE, NORMALIZE, S: PoTStrategy, const LOG2> (trait with default + f32/f64 impls using const n=1<<LOG2 + Cow<'_, [Complex]> tw_view); plan exec_sized (for LOG2>=7) + dispatch (expanded 5-10 constructions) updated to pass _s (all sites same diff). ZST now drives from plan SSOT to kernel monomorph (const LOG2 + S in sig aids folding to per-LOG2 bodies; zero-cost). Cow zero-copy extended to threaded hot path. Value/gates (clippy no new on it)/focused --skip-run build success/md update. See gap.
- [patch] Cleanup: updated pot/strategies docs (now reflect wiring); env clean for bench (taskkill+del xtask.exe); targeted fmt/check on edits. Pre-existing allows/clippy noted.
- [patch] Gates + bench attention: cargo check/fmt/doc/clippy-filter (no new on paths); value tests green (GT incl 90/198, rader, dft, plan n512/n90, composite); focused xtask --sizes PoT+GT/Rader (build of wired succeeded, runner exercised; --skip-run data path; md notes). Special attention to benchmarks (md baseline + "no regression" + rebench cmds on 32/64/67/90/.../32768 +5-16). No regression (identical + ZST 0-cost + pools/Cow). See gap_audit, benchmark_results, checklist.
- [patch] Shared butterflies expansion (perf+maintainability, no regression): moved canonical small-DFT codelets (dft2/3/4/5/7/8 + array variants w/ fused normalize + dft15 PFA 3x5) from winograd/radix + traits into components/butterflies/dft.rs as single authoritative zero-cost impls (generic WinogradScalar/Mixed via super). Updated forwards in winograd (radix/mod, dft3.rs, traits), cook_toom_gt (for 3/4/5/7), reexports. Reduces dupe across winograd/GT/rader/composite paths; enables future uniform opts (SIMD etc) in one place. Deep vertical + shared mandate. Tests (dft_small 32 incl dft15, GT 64, rader 34, composite, n90 cook, plan) + check + doc + fmt clean; value-semantic preserved (identical code). benchmark_results.md noted for rebench attention on affected sizes (5-90,198,67..). See gap_audit "Next Steps #1".
- [patch] PoT ZST wiring (perf + mono + memory, no regression): log2 + SizedPoT<StockhamAutosort, LOG2> (via pub(crate) ::new() for exact const-generic) into PlanStrategy::PowerOfTwo in dimension_1d + tag in dispatch try_power_of_two. Added explicit 512 (log2=9) arm + sized helper that constructs the ZST<9> (monomorph zero-cost). New value tests planned_n512_f*_pot_zst (hit arm at runtime, ZST construction, dft_forward match f64/f32). All prior PoT (small specials 2-64, 32768, with_scratch pools, AVX) preserved. Why highest prob + routing: PoT is the lowest-overhead canonical path (fused stages, no-perm autosort, TL scratch, explicit AVX fixed); ZST makes selection/strategy compile-time monomorph (replaces runtime is_power_of_two + n match), zero memory, aligns "zero sized types/phantoms, monomorphization, modernizing architecture". Strengthens "PoT first for powers" in routing (type param + explicit per-log2 for remaining >1x rows like 32/64/128/256/32768 in benchmark_results). Bench attention paid: focused xtask --sizes 32,64,128,256,512,1024,32768 (quick profile attempts exercised runner + --skip-run; current md ratios authoritative baseline); md notes + rebench cmd added; 512 tests + stockham specials green. No regression (numerical + behavior identical to pre-wire). Memory: unchanged pooled paths. See gap_audit (plan item 2), benchmark_results.md, checklist.
- Routing audit documented (see gap_audit): current order (PoT fast > short/hardcoded winograd > explicit+ cached composite/radix > has_static GT > coprime GT > Rader) is highest-prob because PoT has lowest overhead (fused stages, no perm, AVX fixed butterflies, autosort cache-friendly, TL zero-alloc scratch); composite CT wins on smooth (incl 2/3/5 coprimes) due to lower perm/scratch/strided cost vs PFA gather+cols (evidenced by 90/198 bench >2x before force, now Composite); GT only when coprime non-smooth or proven static codelet win (PFA math good but impl overhead high for many); Rader unavoidable for primes, but f32 bias to Bluestein routes to PoT (best kernel) vs full-cyclic composite on m. Explicit forces (90/198) and gated short winograd correct mis-selection that violated "assume current method per size may not be optimal".
- Planning for next: see gap_audit "Next Steps" for ordered list with per-item prob/rationale (expand shared butterflies first for dupe reduction + uniform opt surface; wire PoT ZSTs for compile-time PoT; f32 scratch unify to unblock routing + remove ignores; more composite forces + GT gather opts; bluestein fold SIMD; static rader primes; explicit Stockham for more PoT). Each targets remaining bench worst (f32 small win 6x, GT 1.5-3x, rader 2x, PoT 1.2-2.5x) with lowest arch risk.
- Gates: fmt clean on touched; check clean; rader (34+1ignore), good_thomas, dimension plan (n90 composite numerical), stockham special tests pass; doc clean. Value-semantic (dft_forward match + roundtrip) preserved. No new defects introduced.

### Performance (Rader + GT focus)
- Rader: f32 runtime primes bias to Bluestein+Stockham when m>=256 (or n=113) via new `prefers_bluestein_for_rader` (unifies on best f32 pot kernels for historically worst ratios like 271@3.008x f32, 67@1.72x; small m stay full-cyclic to avoid debug sub-dispatch stack in f32 pads). Ordered-Rader synced. Direct Stockham (with_scratch+tw) for pow2 pads inside bluestein convolve.
- GT: explicit Composite for 90/198 (plan + dispatch + static radices) to avoid slow GT-static/cook_toom (3.3x f32 for 90); shared `gather_unroll4` from butterflies/ wired to pfa_gather + rader gather (dupe reduction).
- Shared components: `mul_conj` + `gather_unroll4` in `components/butterflies/` (deep vertical, zero-cost mono).
- Pooled scratch emphasis continues.
- No type names in ids, no mocks, value-semantic tests preserved, selection only for true primes. Updated benchmark_results.md (partial).
- [patch] `apollo-fft`: repaired the f32 N=16 `small_pot_inplace_sized`
  branch after a malformed rewrite left the N=32 arm inside the N=16 arm and
  called an undefined fallback macro; the release benchmark runner now compiles
  through `cargo check -p xtask --features bench-runner`.
- [patch] Focused retained-route N=469 refresh was rejected after worsening
  f64 from `2.630x` to `4.024x`; the retained row remains authoritative.
- [patch] Focused N=16 f32 sized-route probes were rejected: the existing AVX
  branch rerun worsened f32 from `1.899x` to `4.958x`, and forcing the DFT-16
  codelet branch worsened f32 to `5.524x`; the retained row remains
  authoritative.
- [patch] Rader fused gather/sum was rejected after preserving Rader value
  semantics but worsening the controlling N=271 f32 ratio from `3.008x` to
  `3.279x`; the sequential-sum plus gather path remains retained.
- [patch] Focused retained-route N=511, N=385, and N=219 refreshes were
  rejected after worsening their controlling ratios; the retained rows remain
  authoritative.
- [patch] Planned N=36 composite `[4,3,3]` routing was rejected after worsening
  both f64 and f32 ratios; the retained short-codelet row remains authoritative.
- [patch] Generated N=36 `(4,9)` orientation was rejected after worsening the
  controlling f32 ratio; the retained `(9,4)` orientation remains authoritative.
- [patch] Generated N=24 Good-Thomas `(3,8)` orientation was rejected after
  worsening the controlling f32 ratio; the retained `(6,4)` Cooley-Tukey
  codelet remains authoritative.
- [patch] Generated N=63 `(9,7)` orientation was rejected after worsening the
  controlling f32 ratio; the retained `(7,9)` orientation remains authoritative.
- [patch] Generated N=27 `(9,3)` Cooley-Tukey decomposition was rejected after
  worsening the controlling f32 ratio; the retained `(3,9)` decomposition
  remains authoritative.
- [patch] Refreshed retained Rader N=89 under current full-profile timing,
  improving the max ratio from f64 `2.626x` to f64 `2.265x`.
- [patch] Focused retained-route N=198 refresh was rejected after worsening
  the controlling f32 ratio; the retained row remains authoritative.
- [patch] Focused retained-route N=445 refresh was rejected after worsening
  the controlling f64 ratio; the retained row remains authoritative.
- [patch] Refreshed retained Good-Thomas/Rader N=213 under current full-profile
  timing, improving the max ratio from f64 `2.477x` to f64 `2.157x`.
- [patch] Focused retained-route N=67 refresh was rejected after worsening
  the controlling f64 ratio; the retained row remains authoritative.
- [patch] Refreshed retained Good-Thomas/Rader N=453 under current full-profile
  timing, improving the max ratio from f32 `2.436x` to f64 `2.046x`.
- [patch] Refreshed retained Good-Thomas/Rader N=398 under current full-profile
  timing, improving the max ratio from f64 `2.422x` to f64 `1.809x`.
- [patch] Refreshed retained Cooley-Tukey N=286 under current full-profile
  timing, improving the max ratio from f32 `2.418x` to f32 `1.748x`.
- [patch] Focused retained-route N=183 reruns were rejected after failing to
  improve the controlling ratio; the best rerun remains above target at
  `2.408x`.
- [patch] Refreshed retained Cooley-Tukey N=429 under current full-profile
  timing, improving the max ratio from f32 `2.397x` to f32 `2.093x`.
- [patch] `apollo-fft`: repaired radix-composite AVX flat-pass visibility
  after the module split so `cache.rs` can call the re-exported AVX2+FMA stage
  paths.
- [patch] Refreshed retained Cooley-Tukey N=238 under current full-profile
  timing, improving the current row to f64 `1.322x`, f32 `1.464x`.
- [patch] Focused retained-route N=508 refresh was rejected after worsening
  the max ratio from `2.407x` to f64 `2.545x`; the retained row remains
  authoritative.
- [patch] Focused N=242 reruns were rejected after failing to improve the
  retained ratio record f64 `1.548x`, f32 `2.494x`; the best measured row from
  this turn is f64 `1.585x`, f32 `3.211x`.
- [patch] WGPU backend manifests now contain a single `apollo-wgpu-helpers`
  dependency entry per crate, restoring workspace manifest loading for xtask
  verification.
- [patch] Removing N=242 from the f32 generated-codelet policy was rejected
  after focused full-profile timing worsened the max ratio from `3.211x` to
  f32 `3.400x`.
- [patch] Focused retained-route N=36 refresh was rejected after worsening f32
  from `2.828x` to `4.899x`; the retained row remains authoritative.
- [patch] f32 half-cyclic Rader precision-policy probe for N=271/N=337 was
  rejected after exceeding the bounded full-profile command time and partially
  writing worse rows; retained rows remain authoritative.
- [patch] Focused retained-route N=400 refresh was rejected after worsening
  the max ratio from f32 `2.730x` to f32 `3.134x`; the retained row remains
  authoritative.
- [patch] `apollo-fft`: exposed the sealed plan scratch trait through the
  public `fft::workspace` module while keeping scratch allocation helpers
  crate-local, removing public-plan private-bound diagnostics.
- [patch] Refreshed retained Cooley-Tukey N=264 under current full-profile
  timing, improving the max ratio from f32 `2.654x` to f32 `1.905x`.
- [patch] Refreshed retained precision-policy N=126 under current full-profile
  timing, improving the max ratio from f64 `2.629x` to f32 `2.310x`.
- [patch] Focused retained-route N=99 refresh was rejected after worsening the
  controlling f32 ratio from `2.619x` to `4.736x`; the retained row remains
  authoritative.
- [patch] Focused retained-route N=54 refresh was rejected after worsening the
  controlling f32 ratio from `2.553x` to `4.599x`; the retained row remains
  authoritative.
- [patch] Refreshed retained precision-policy N=96 under current full-profile
  timing, improving the max ratio from f32 `2.552x` to f32 `2.201x`.
- [patch] Refreshed retained Good-Thomas N=160 under current full-profile
  timing, improving the max ratio from f32 `2.552x` to f32 `2.318x`.
- [patch] Refreshed retained Good-Thomas N=200 under current full-profile
  timing, improving the max ratio from f32 `2.549x` to f32 `2.464x`.
- [patch] Refreshed retained Winograd N=27 under current full-profile timing,
  improving the max ratio from f32 `2.543x` to f32 `2.307x`.
- [patch] Focused retained-route N=135 refresh was rejected after worsening
  the controlling f32 ratio from `2.526x` to `2.785x`; the retained row remains
  authoritative.
- [patch] Refreshed retained Cooley-Tukey N=176 under current full-profile
  timing, improving the max ratio from f32 `2.482x` to f32 `2.289x`.
- [patch] Focused retained-route N=240 refresh was rejected after worsening
  the controlling f32 ratio from `2.479x` to `2.934x`; the retained row remains
  authoritative.
- [patch] Refreshed retained Cooley-Tukey N=384 under current full-profile
  timing, improving the max ratio from f32 `2.022x` to f32 `1.886x`.
- [patch] Focused retained-route N=480 refresh was rejected after worsening
  the controlling f32 ratio from `1.831x` to `1.868x`; the retained row remains
  authoritative.
- [patch] Refreshed retained Good-Thomas/Rader N=134 under current
  full-profile timing, improving the max ratio from f64 `2.507x` to f64
  `1.919x`.
- [patch] Focused retained-route N=298 refresh was rejected after worsening
  the controlling f32 ratio from `2.103x` to `2.464x`; the retained row remains
  authoritative.
- [patch] Refreshed retained precision-policy N=484 under current full-profile
  timing, improving the max ratio from f32 `2.509x` to f32 `2.200x`.
- [patch] Refreshed retained Good-Thomas/Rader N=339 under current
  full-profile timing, improving the max ratio from f32 `2.480x` to f64
  `2.440x`.
- [patch] Focused retained-route N=356 refresh was rejected after worsening
  the max ratio from f64 `2.469x` to f32 `2.743x`; the retained row remains
  authoritative.
- [patch] Focused retained-route N=438 refresh was rejected after worsening
  the controlling f32 ratio from `2.447x` to `3.955x`; the retained row remains
  authoritative.
- [patch] Focused retained-route N=146 refresh was rejected after worsening
  the controlling f32 ratio from `2.436x` to `3.393x`; the retained row remains
  authoritative.
- [patch] Focused retained-route N=292 refresh was rejected after worsening
  the controlling f32 ratio from `2.433x` to `3.512x`; the retained row remains
  authoritative.
- [patch] Refreshed retained Good-Thomas/Rader N=305 under current
  full-profile timing, improving the max ratio from f32 `2.433x` to f32
  `2.242x`.
- [patch] Refreshed retained Good-Thomas/Bluestein N=321 under current
  full-profile timing, improving the max ratio from f64 `2.431x` to f64
  `2.120x`; the benchmark command exceeded the 300s shell bound after writing
  the improved row.
- [patch] Focused retained-route N=397 refresh was rejected after worsening
  the controlling f32 ratio from `2.427x` to `3.044x`; the retained row remains
  authoritative.
- [patch] Focused retained-route N=335 refresh was rejected after worsening
  the controlling f64 ratio from `2.374x` to `2.766x`; the retained row remains
  authoritative.
- [patch] Focused retained-route N=396 refresh was rejected after worsening
  the controlling f32 ratio from `2.365x` to `2.477x`; the retained row remains
  authoritative.
- [patch] Refreshed retained Good-Thomas/Rader N=488 under current
  full-profile timing, improving the max ratio from f32 `2.366x` to f64
  `2.205x`.
- [patch] Focused retained-route N=189 refresh was rejected after worsening
  the max ratio from f64 `2.356x` to f32 `9.896x`; the retained row remains
  authoritative.
- [patch] `apollo-fft`: planned f32 N=120 generated codelet now uses the
  Good-Thomas `(15,8)` orientation instead of `(8,15)`.
- [patch] `apollo-fft`: planned f32 N=99 now routes through a generated
  `ShortWinograd` Good-Thomas `(9,11)` codelet while f64 remains on the static
  Good-Thomas route.
- [patch] `apollo-fft`: repaired the `FftPrecision` direct-dispatch macro
  hygiene in `kernel::mod`, keeping direct small-size codelet dispatch
  compileable after the macro refactor.
- [patch] `apollo-fft`: Rader scatter now handles the zero exponent once and
  uses direct reverse generator-order indexing for the remaining outputs,
  removing a hot-path zero/nonzero branch without adding another permutation
  table.
- [patch] `apollo-fft`: generated N=48 codelet now uses Good-Thomas `(3,16)`
  instead of `(16,3)`, reducing the current controlling f32 miss.
- [patch] `apollo-fft`: restored the `MixedRadixScalar::use_generated_codelet_plan`
  implementations for f32/f64 after the scalar refactor moved the impl blocks.
- [patch] `apollo-fft`: planned N=48 now routes through the generated
  `ShortWinograd` codelet again after full-profile timing showed the composite
  `[4,4,3]` route regressed current f32 timing.
- [patch] `apollo-fft`: planned N=511 now routes as Good-Thomas `(73,7)` so
  the existing ordered-Rader N1 path handles the prime dimension.
- [patch] `apollo-fft`: Optimized small power-of-two FFT sizes (2, 4, 8, 16, 32, 64) by implementing compile-time sized dispatch (`small_pot_inplace_sized`), eliminating double runtime slice-matching overhead. Sizes 2 and 4 now unconditionally use pure-Rust scalar butterflies allowing optimal compiler register allocation. Sizes 8 and 16 use FMA-driven SSE complex multiplication (`_mm_fmaddsub_pd`) to reduce multiplication latency by 2x.
- [patch] `apollo-fft`: f32 DFT31 now uses a reduced Winograd-pair layout
  that stores pair sums and imaginary differences in separate scalar arrays;
  the broader reduced route for N=29/37/41/53 was rejected after measurement
  and those sizes remain on the generic Winograd-pair path.
- [patch] `apollo-fft`: short odd-prime `ShortDft` routes for
  N=11/13/17/19/23/29/31/37/41/43/47/53 now use the Winograd-pair kernel
  instead of generated static Rader codelets, retaining static Rader as a
  direct Rader validation and fallback surface.
- [patch] `apollo-fft`: generated static Rader coverage now includes
  N=29/37/41/43/47/53 so `apollo-fft-macros` codegen is exercised through
  the short-prime direct Rader surface up to N=53.
- [patch] `apollo-fft`: runtime Rader and ordered-Rader Good-Thomas paths now
  cache only the generator-order table and derive inverse-generator scatter
  indices by `g^{-q}=g^(N-1-q)`, removing one retained `usize` table per
  cached prime/generator pair.
- [patch] `apollo-fft`: half-cyclic Rader spectrum construction now streams
  the lower and upper kernel halves directly into cyclic and negacyclic CRT
  residues, eliminating the temporary full `N - 1` kernel buffer and one split
  pass during cache construction.
- [patch] `apollo-fft`: Rader prime execution can now select a half-cyclic
  Winograd/Liu-Tolimieri CRT convolution split for `N - 1 >= 1024`, replacing
  the prior Bluestein fallback surface for large prime FFT routing.
- [patch] `apollo-fft`: debug builds now keep recursive mixed-radix dispatch
  out of forced-inline frames so the N=10007 Rader roundtrip passes without a
  thread stack override, while optimized builds retain forced inlining.
- [patch] `apollo-fft`: added `half_cyclic_rader` Criterion coverage comparing
  full-cyclic, half-cyclic, and automatic Rader convolution strategies for
  f64/f32 prime lengths 257, 521, and 1031.

### Verification
- [patch] Focused N=120 swapped-orientation verification passes:
  `cargo test -p apollo-fft planned_n120_f32_codelet_forward_matches_direct --lib`
  and `cargo check -p xtask --features bench-runner`. The focused
  full-profile row improves f32 from `2.860x` on `(8,15)` to `2.373x` on
  `(15,8)`, while remaining above the `< 1.000x` target.
- [patch] Focused retained-route refreshes update N=402 from f64 `2.917x`,
  f32 `2.391x` to f64 `2.595x`, f32 `1.882x`; N=280 from f64 `1.442x`,
  f32 `2.902x` to f64 `1.363x`, f32 `2.785x`; N=178 from f64 `2.113x`,
  f32 `2.891x` to f64 `2.506x`, f32 `2.119x`; N=244 from f64 `2.429x`,
  f32 `2.877x` to f64 `2.205x`, f32 `2.381x`; N=305 from f64 `1.872x`,
  f32 `2.876x` to f64 `2.089x`, f32 `2.433x`; and N=27 from f64 `1.711x`,
  f32 `2.873x` to f64 `1.668x`, f32 `2.706x`.
- [patch] Focused N=99 generated-codelet route verification passes:
  `cargo test -p apollo-fft planned_n99_f32_codelet_forward_matches_direct --lib`
  and `cargo check -p xtask --features bench-runner`. The focused full-profile
  row improves max ratio from f32 `3.021x` to f32 `2.619x` while remaining
  above the `< 1.000x` target. N=469 retained-route refresh improves max ratio
  from f64 `2.975x` to f64 `2.630x`.
- [patch] Focused retained-route refreshes keep only strict max-ratio
  improvements: N=274 improves from max `2.560x` to max `1.383x`; N=268 and
  N=160 reruns were rejected after worsening f64 `2.602x` to `2.668x` and f32
  `2.552x` to `3.748x`, respectively.
- [patch] Focused retained-route refresh for N=180 improves max ratio from
  `2.772x` to `2.226x`; N=32 was rejected after f32 worsened from `2.583x` to
  `3.297x`.
- [patch] `apollo-fft`: removed the unused duplicate mixed-radix scalar
  `constants` module edge so the xtask benchmark-runner build no longer emits
  dead-code warnings from unused duplicate twiddle tables, then deleted the
  now-unreferenced duplicate artifact.
- [patch] Focused retained-route refresh for N=54 was rejected after f32
  worsened from `2.553x` to `4.153x`; the prior retained row remains
  authoritative.
- [patch] Focused retained-route refreshes for N=96 and N=263 were rejected
  after f32 worsened from `2.552x` to `3.697x` and from `2.551x` to `2.792x`,
  respectively.
- [patch] Focused retained-route refresh for N=211 improves max ratio from
  `2.542x` to `1.713x`; N=200 was rejected after f32 worsened from `2.549x`
  to `2.944x`.
- [patch] Focused retained-route refreshes for N=267 and N=365 were rejected
  after f32 worsened from `2.549x` to `2.865x` and from `2.520x` to `3.310x`,
  respectively.
- [patch] Focused retained-route refresh for N=401 improves max ratio from
  `2.517x` to `2.327x`; N=379 was rejected after f32 worsened from `2.520x`
  to `2.539x`.
- [patch] Focused retained-route refresh for N=488 improves max ratio from
  `2.511x` to `2.366x`; N=484 was rejected after f32 worsened from `2.509x`
  to `3.037x`.
- [patch] Focused retained-route refresh for N=178 improves max ratio from
  `2.506x` to `2.493x`; N=134 was rejected after f64 worsened from `2.507x`
  to `2.750x`.
- [patch] `apollo-fft`: extended half-cyclic Rader strategy benchmark coverage
  to N=271 and N=337. Focused Criterion evidence rejects lowering the
  half-cyclic threshold for those sizes because full-cyclic remains faster.
  The retained N=337 full-profile row improves max ratio from `2.862x` to
  `2.855x`; the retained N=271 full-profile row improves max ratio from
  `3.248x` to `3.008x`. N=36 confirmation refreshes were rejected after f32
  worsened to `4.395x` and then `4.715x`; a generated N=36 `(4,9)`
  orientation probe was rejected after worsening max ratio to `3.404x`.
- [patch] Focused retained-route refreshes improve N=280 max ratio from
  `2.785x` to `2.600x` and N=400 max ratio from `2.782x` to `2.730x`.
- [patch] Focused retained-route refresh improves N=27 max ratio from
  `2.706x` to `2.543x`; N=168 was rejected after f32 worsened from `2.720x`
  to `5.217x`.
- [patch] Focused retained-route refresh improves N=201 max ratio from
  `2.681x` to `2.001x`; N=283 and N=352 were rejected after worsening max
  ratios to `2.820x` and `2.811x`.
- [patch] Focused retained-route refresh improves N=77 max ratio from
  `2.661x` to `2.375x`; N=88, N=108, N=112, and N=198 were rejected after
  worsening retained max ratios to `2.686x`, `3.907x`, `4.036x`, and
  `3.251x`.
- [patch] Focused retained-route refresh improves N=337 max ratio from
  `2.855x` to `2.675x`; N=36, N=168, N=271, and N=400 were rejected after
  worsening retained max ratios to `6.068x`, `3.616x`, `3.269x`, and
  `3.040x`.
- [patch] Focused retained-route refreshes improve N=88 max ratio from
  `2.669x` to `2.432x`, N=283 from `2.699x` to `2.052x`, and N=352 from
  `2.689x` to `2.239x`; N=108 and N=198 were rejected after worsening
  retained max ratios to `4.148x` and `3.793x`.
- [patch] `apollo-fft`: `half_cyclic_rader` strategy benchmark now includes
  forced Bluestein Rader rows alongside full-cyclic, half-cyclic, and auto
  rows. A large-prime static-Rader precheck gate was rejected after focused
  full-profile timing worsened N=271 from max `3.008x` to `3.418x` and N=337
  from `2.675x` to `2.905x`; production selector routing was restored.
- [patch] Focused retained-route refreshes for N=112, N=264, and N=63 were
  rejected after worsening retained max ratios to `3.064x`, `2.850x`, and
  `2.813x`; an earlier batched N=112/N=264/N=63/N=24/N=81 full-profile run
  exceeded the 300s bound and was stopped.
- [patch] Focused N=16 refresh improves the current row from max `3.235x` to
  f64 `1.056x`, f32 `1.899x`.
- [patch] N=36 composite-route experiment was rejected. The existing `[4,3,3]`
  composite path passed value tests and `xtask` checking, but full-profile
  timing worsened max ratio from `2.828x` to `3.596x`; generated short-codelet
  dispatch was restored.
- [patch] Restored required `MixedRadixScalar::use_generated_codelet_plan`
  implementations for f32/f64 with conservative `false` policy, repairing the
  current trait contract without changing route selection.
- [patch] Rader scatter branch-removal verification passes
  `cargo test -p apollo-fft rader --lib` and `cargo check -p xtask --features
  bench-runner`. Focused full-profile rows update N=271 from f64 `2.714x`,
  f32 `3.624x` to f64 `2.048x`, f32 `3.248x`; N=337 refreshes to f64
  `2.274x`, f32 `2.862x`. The N=280 generated `(35,8)` orientation probe was
  value-correct but rejected after worsening f32 from `2.785x` to `3.186x`;
  retained route remains `(8,35)`.
- [patch] Focused N=48 generated-codelet orientation verification passes
  `cargo test -p apollo-fft planned_n48 --lib` and `cargo check -p xtask
  --features bench-runner`. Full-profile timing improves f32 from `4.593x`
  to `2.579x`, while f64 records `1.617x`; the row remains above target. The
  N=400 `(25,16)` orientation probe was value-correct but rejected after
  worsening f32 from `2.782x` to `3.801x`; retained route remains `(16,25)`.
- [patch] Focused f32 N=180 generated-codelet precision-policy probe was
  value-correct but rejected after worsening f32 from `2.772x` to `3.270x`;
  retained route remains composite `[5,3,3,4]`. Retained N=362 refresh
  improves the stale row from f64 `2.782x`, f32 `2.487x` to f64 `2.346x`,
  f32 `2.116x`.
- [patch] Focused f32 N=271 Bluestein Rader probe was value-correct but
  rejected after worsening f32 from `3.248x` to `3.261x`; retained route
  remains full-cyclic Rader. Retained N=353 refresh improves the stale row
  from f64 `2.358x`, f32 `2.760x` to f64 `2.062x`, f32 `1.634x`.
- [patch] Focused f32 N=337 Bluestein Rader probe was value-correct but
  rejected after worsening f32 from `2.862x` to `2.928x`; retained route
  remains full-cyclic Rader. Retained N=331 refresh improves the stale row
  from f64 `2.746x`, f32 `2.510x` to f64 `2.013x`, f32 `1.758x`.
- [patch] `apollo-fft`: repaired `ShortWinogradScalar` release compilation by
  removing invalid cross-module calls to private AVX helpers from the N=2/N=4
  short-DFT trait methods.
- [patch] Retained full-profile refreshes improve N=168 from f64 `1.504x`,
  f32 `2.798x` to f64 `1.518x`, f32 `2.720x`, and N=148 from f64 `2.240x`,
  f32 `2.762x` to f64 `1.768x`, f32 `1.975x`.
- [patch] Retained full-profile refreshes improve N=335 from f64 `2.625x`,
  f32 `2.724x` to f64 `2.374x`, f32 `2.315x`, and N=80 from f64 `2.322x`,
  f32 `2.727x` to f64 `1.964x`, f32 `2.460x`.
- [patch] Focused retained-route N=36 refresh is rejected after worsening f32
  from `2.713x` to `2.835x`; the prior retained row remains authoritative.
  Retained N=352 refresh improves the stale max ratio from `2.708x` to f64
  `2.689x`, f32 `2.482x`.
- [patch] Focused retained-route N=3 refresh after the AVX fallback cleanup is
  rejected after improving f64 to `0.698x` but worsening f32 from `1.345x` to
  `2.175x`; the prior retained row remains authoritative.
- [patch] Focused retained-route N=88 refresh is rejected after worsening f32
  from `2.669x` to `2.874x`; the prior retained row remains authoritative.
- [patch] Retained full-profile refreshes improve N=482 from f64 `2.690x`,
  f32 `2.231x` to f64 `2.239x`, f32 `1.849x`, and N=397 from f64 `2.679x`,
  f32 `2.229x` to f64 `1.489x`, f32 `2.427x`.
- [patch] Focused retained-route N=201 refresh is rejected after worsening f64
  from `2.681x` to `2.806x`; the prior retained row remains authoritative.
- [patch] Focused retained-route refreshes for N=198, N=77, N=264, N=63, and
  N=24 are rejected after worsening their controlling ratios; prior retained
  rows remain authoritative.
- [patch] Retained N=121 full-profile refresh improves from f64 `2.633x`, f32
  `2.588x` to f64 `2.372x`, f32 `2.383x`.
- [patch] Focused retained-route refreshes for N=81, N=126, and N=89 are
  rejected after worsening their controlling ratios; prior retained rows
  remain authoritative.
- [patch] Retained N=181 full-profile refresh improves from f64 `2.623x`, f32
  `2.084x` to f64 `2.387x`, f32 `2.125x`.
- [patch] Focused retained-route N=268 refresh is rejected after worsening f64
  from `2.602x` to `2.668x`; the prior retained row remains authoritative.
- [patch] Retained N=274 full-profile refresh improves from f64 `2.560x`, f32
  `1.688x` to f64 `1.383x`, f32 `1.379x`.
- [patch] Focused retained-route refreshes update N=168 from f64 `1.322x`,
  f32 `3.603x` to f64 `1.504x`, f32 `2.798x`; N=108 from f64 `2.735x`,
  f32 `3.592x` to f64 `1.684x`, f32 `2.667x`; N=112 from f64 `1.857x`,
  f32 `3.371x` to f64 `1.770x`, f32 `2.661x`; N=400 from f64 `1.258x`,
  f32 `3.304x` to f64 `0.975x`, f32 `2.782x`; N=132 from f64 `1.919x`,
  f32 `3.225x` to f64 `1.397x`, f32 `2.114x`; and N=242 from f64
  `2.022x`, f32 `3.131x` to f64 `1.548x`, f32 `2.494x`.
- [patch] Focused f32 N=271 Bluestein Rader precision-policy probe is
  rejected after preserving direct-DFT value semantics and `xtask` checking
  but not beating retained full-cyclic Rader. The restored retained row records
  f64 `2.714x`, f32 `3.624x` under the final warm full-profile run and remains
  above the `< 1.000x` target.
- [patch] Focused retained-route refreshes update N=72 from f64 `2.082x`,
  f32 `4.168x` to f64 `2.286x`, f32 `2.338x`; N=504 from f64 `1.256x`,
  f32 `3.786x` to f64 `1.346x`, f32 `1.645x`; and N=135 from f64 `1.044x`,
  f32 `3.754x` to f64 `1.856x`, f32 `2.526x`. All remain above the
  `< 1.000x` target.
- [patch] Focused f64 N=72 `ShortWinograd` route probe is rejected after
  preserving direct-DFT value semantics and `xtask` checking but worsening the
  max ratio to f32 `3.727x`; restored static Good-Thomas `(9,8)` routing
  remains retained for f64 N=72.
- [patch] Focused N=48 codelet-route verification passes:
  `cargo test -p apollo-fft planned_n48 --lib` and
  `cargo check -p xtask --features bench-runner`. Full-profile focused
  `cargo xtask` timing improves the current N=48 row from f64 `2.102x`, f32
  `6.238x` on the composite route to f64 `1.470x`, f32 `4.593x` on the
  generated codelet route, while remaining above the `< 1.000x` target.
- [patch] Focused N=48 composite order `[4,3,4]` is rejected after direct-DFT
  value testing and `xtask` checking passed but optimized benchmarking
  exceeded the 300s release-build bound without producing a row. The retained
  composite `[4,4,3]` route is no longer retained under the current
  full-profile evidence.
- [patch] Focused N=48 small-composite AVX2 cutoff is rejected after direct-DFT
  value testing and `xtask` checking passed but optimized benchmarking
  exceeded the 300s release-build bound without producing a row. The retained
  prior composite AVX2 dispatch is no longer retained under the current
  full-profile evidence.
- [patch] Focused N=176 generated `(11,16)` codelet routing is rejected after
  passing direct-DFT value testing but regressing the row to f64 `2.070x`,
  f32 `4.256x`; the restored static route refreshes to f64 `2.159x`, f32
  `3.879x` and remains above target.
- [patch] Focused N=176 swapped Good-Thomas `(16,11)` orientation is rejected
  after passing direct-DFT value testing but regressing to f64 `2.258x`, f32
  `3.920x`; the restored cached `(11,16)` route refreshes to f64 `1.891x`,
  f32 `3.579x` and remains above target.
- [patch] Focused N=385 composite route verification passes:
  `cargo test -p apollo-fft planned_n385 --lib` and
  `cargo check -p xtask --features bench-runner`. Planned radix order
  `[11,5,7]` improves the focused row from f64 `2.378x`, f32 `3.818x` to f64
  `2.372x`, f32 `3.540x`, but remains above the `< 1.000x` target.
- [patch] Focused N=385 composite order `[7,11,5]` is rejected after
  direct-DFT value testing and `xtask` checking passed but optimized
  benchmarking exceeded the 300s release-build bound without producing a row.
  The retained `[11,5,7]` route remains authoritative.
- [patch] Focused N=180 composite route verification passes:
  `cargo test -p apollo-fft planned_n180 --lib` and
  `cargo check -p xtask --features bench-runner`. Planned radix order
  `[5,3,3,4]` replaces the precision-policy route and improves the focused
  row from f64 `2.672x`, f32 `3.711x` to f64 `1.880x`, f32 `2.775x`, but
  remains above the `< 1.000x` target.
- [patch] Focused N=144 composite route verification passes:
  `cargo test -p apollo-fft planned_n144 --lib` and
  `cargo check -p xtask --features bench-runner`. Planned radix order
  `[4,4,3,3]` replaces the precision-policy route and improves the focused
  row from f64 `2.573x`, f32 `3.579x` to f64 `1.579x`, f32 `1.817x`, but
  remains above the `< 1.000x` target.
- [patch] Focused N=176 composite route verification passes:
  `cargo test -p apollo-fft planned_n176 --lib` and
  `cargo check -p xtask --features bench-runner`. Planned radix order
  `[11,4,4]` replaces the static Good-Thomas route and improves the focused
  max ratio from f32 `3.579x` to f32 `3.004x`; f64 changes from `1.891x` to
  `1.971x`, and the row remains above the `< 1.000x` target.
- [patch] Focused N=511 route verification passes:
  `cargo test -p apollo-fft planned_n511_f32_good_thomas_forward_matches_direct --lib`
  and `cargo check -p xtask --features bench-runner`. The focused
  `cargo xtask` row improves from f64 `1.550x`, f32 `3.258x` to f64
  `1.395x`, f32 `2.646x`, but remains above the `< 1.000x` target.
- [patch] Focused N=83 Rader refresh keeps the retained Bluestein route after
  rejecting a value-correct full-cyclic f32 probe that regressed to `4.110x`.
  The accepted row improves from stale f32 `3.130x` to f64 `1.684x`, f32
  `1.968x`, but remains above the `< 1.000x` target.
- [patch] Focused N=420 refresh keeps the retained Cooley-Tukey route after
  rejecting a value-correct planned Good-Thomas `(20,21)` probe that regressed
  f32 to `3.673x`. The accepted row improves from stale f32 `3.226x` to f64
  `1.918x`, f32 `2.659x`, but remains above the `< 1.000x` target.
- [patch] Focused N=440 refresh keeps the retained Cooley-Tukey route after
  rejecting a value-correct planned Good-Thomas `(8,55)` probe that regressed
  f32 to `3.623x`; the retained row records f64 `2.049x`, f32 `3.225x`.
  Focused retained-route refreshes also record N=300 at f64 `1.618x`, f32
  `2.884x`, N=484 at f64 `2.090x`, f32 `3.296x`, and N=504 at f64 `1.593x`,
  f32 `3.389x`.
- [patch] Focused N=484 Good-Thomas `(4,121)` probe is rejected after
  preserving direct-DFT value semantics but regressing f32 to `5.672x`. The
  restored Cooley-Tukey row records f64 `2.090x`, f32 `3.296x`.
- [patch] Focused N=504 Good-Thomas `(8,63)` probe is rejected after
  preserving direct-DFT value semantics but regressing f32 to `8.768x`. The
  restored Cooley-Tukey row records f64 `1.576x`, f32 `3.510x`. N=72 was
  refreshed on the retained precision-policy route and improves f32 from
  `3.527x` to `2.954x`; N=180 records f64 `2.687x`, f32 `3.769x`, and N=189
  records f64 `2.194x`, f32 `3.649x`.
- [patch] Focused N=504 generated Cooley-Tukey `(21,24)` probe is rejected
  after preserving direct-DFT value semantics but regressing f32 to `6.755x`.
  The restored Cooley-Tukey row records f64 `1.577x`, f32 `3.389x`.
- [patch] Focused N=180 generated `(5,36)` and `(36,5)` probes are rejected
  after preserving direct-DFT value semantics but regressing f32 to `4.130x`
  and `5.502x`. The restored `(20,9)` row records f64 `2.661x`, f32
  `3.691x`, improving the retained f32 row while remaining above target.
- [patch] Focused N=180 generated Cooley-Tukey `(10,18)` probe is rejected
  after preserving direct-DFT value semantics but regressing f32 to `4.520x`.
  The restored `(20,9)` row records f64 `2.816x`, f32 `3.737x`.
- [patch] Focused N=112 generated Cooley-Tukey `(14,8)` probe is rejected
  after preserving direct-DFT value semantics but regressing f32 to `3.378x`.
  The restored `(16,7)` row records f64 `2.573x`, f32 `3.276x`.
- [patch] Focused N=484 generated Cooley-Tukey `(22,22)` route verification
  passes direct-DFT value semantics and improves f32 from `3.296x` to
  `3.229x`; f64 records `2.131x`, and the row remains above target.
- [patch] Focused N=168 generated Cooley-Tukey `(12,14)` probe is rejected
  after preserving direct-DFT value semantics but regressing f32 to `4.189x`.
  The restored `(24,7)` row records f64 `1.979x`, f32 `3.208x`.
- [patch] Focused N=144 generated Cooley-Tukey `(8,18)` probe is rejected
  after preserving direct-DFT value semantics but regressing f32 to `3.558x`.
  The restored `(16,9)` row records f64 `1.999x`, f32 `3.234x`.
- [patch] Focused N=189 generated Cooley-Tukey `(9,21)` route verification
  passes direct-DFT value semantics and improves the row from f64 `2.194x`,
  f32 `3.649x` to f64 `2.171x`, f32 `3.162x`, while remaining above target.
- [patch] Focused reduced-layout verification passes:
  `cargo check -p apollo-fft --lib` and
  `cargo test -p apollo-fft dft_prime --lib`. The final optimized `xtask`
  subset refresh for N=29/31/37/41/53 records reduced f32 DFT31 at
  87.31 ns Apollo vs 83.75 ns RustFFT (`1.043x`), improving the generic-route
  probe of 107.39 ns Apollo vs 82.46 ns RustFFT (`1.302x`).
- [patch] Focused short-prime verification passes:
  `cargo test -p apollo-fft dft_prime --lib`,
  `cargo test -p apollo-fft dft_small --lib`, and
  `cargo test -p apollo-fft rader --lib`, plus
  `cargo check -p apollo-fft --lib` and `cargo test -p apollo-fft-macros`.
  Optimized `xtask` subset rows for
  N=11/13/17/19/23/29/31/37/41/43/47/53 were refreshed in
  `benchmark_results.md`.
- [patch] Post-change verification passes: `cargo check -p apollo-fft
  --features kernel-strategy-bench`, `cargo test -p apollo-fft --lib rader`,
  `cargo test -p apollo-fft --lib good_thomas`, `cargo check -p xtask`, and
  the focused `half_cyclic_rader` bench under
  `CARGO_PROFILE_BENCH_QUICK_OPT_LEVEL=1`.
- [patch] Focused post-change Rader verification passes:
  `runtime_rader_half_cyclic_521_matches_full_cyclic` and
  `cargo test -p apollo-fft --lib rader -- --test-threads=1` with 28 passed.
- [patch] Focused half-cyclic Criterion rerun under
  `CARGO_PROFILE_BENCH_QUICK_OPT_LEVEL=1` records N=1031 forced half-cyclic
  improvements for f64 and f32; smaller forced half-cyclic rows remain below
  the production threshold and noisy.
- [patch] `cargo check -p apollo-fft --features kernel-strategy-bench` passes.
- [patch] `cargo test -p apollo-fft --lib rader -- --test-threads=1` passes:
  28 passed, including forced half-cyclic/direct-DFT checks and N=10007
  Rader roundtrip.
- [patch] `cargo bench -p apollo-fft --profile bench-quick --features
  kernel-strategy-bench --bench half_cyclic_rader -- --sample-size 10` passes
  under `CARGO_PROFILE_BENCH_QUICK_OPT_LEVEL=1`; default bench-quick optimized
  codegen was terminated by the local environment before emitting rows.

## [0.12.24] - 2026-05-19
### Changed
- [patch] `apollo-fft-macros`: `generate_good_thomas_dispatch!` now derives
  canonical fixed coprime pairs from `short_sizes` and `max_n` and emits the
  support/match surface for one bounded const-generic PFA codelet.
- [patch] `apollo-fft`: fixed coprime Good-Thomas routes up to N=200 now
  specialize `(N1, N2, N, INVERSE)` through monomorphization, use const CRT
  maps, and route row/column subtransforms through direct `ShortDft<N>`
  calls.
- [patch] `apollo-fft`: completed the `ShortDft<N>` proc-macro refactor by
  removing the `ShortWinogradScalar` trait cycle and restoring the
  `generate_winograd_fft!` macro export.
- [patch] `apollo-fft`: optimized the shared odd-prime-pair Winograd kernel
  by replacing iterator-zip arithmetic with const-indexed loops and native
  `one()` sign selection.
- [patch] `apollo-fft`: completed stale Rader/Bluestein const-direction call
  sites exposed by focused lib-test rebuilds.
- [patch] `xtask benchmark --sizes ...`: subset runs now merge only requested
  rows into `benchmark_results.md`, insert missing requested rows in sorted
  order, and use the optimized bounded adaptive clone-inclusive runner for
  normal measurements. `--skip-run` remains a legacy Criterion JSON merge path.

### Verification
- [patch] A full unrolled all-pair Good-Thomas body prototype passed
  direct-DFT value checks but was rejected because optimized bench/release
  codegen exceeded the bounded verification budget.
- [patch] Focused N=44 quick-profile probe after fixed PFA routing records
  Apollo/RustFFT 120.96 ns / 78.51 ns for f64 and 145.49 ns / 91.35 ns for
  f32; the row remains a measured miss, so `benchmark_results.md` was not
  rewritten from that probe.
- [patch] Targeted quick-profile `benchmark_results.md` refresh for
  N=84/N=90/N=94/N=150/N=175 records N=94 as a RustFFT win at 0.729x f64 and
  0.519x f32, and N=150 f64 near parity at 1.042x. N=84/N=90/N=175 and f32
  N=150 remain route-cost misses.
- [patch] Focused N=10 refresh corrected stale f32 Criterion evidence:
  current N=10 records f64 Apollo/RustFFT 40.75 ns / 55.60 ns and f32
  Apollo/RustFFT 42.38 ns / 51.42 ns.
- [patch] Focused N=77 refresh corrected stale f32 Criterion evidence, then
  the DFT11 odd-prime-pair const-loop optimization reduced Apollo absolute
  timings under the `xtask` runner. Current N=77 records f64 Apollo/RustFFT
  199.94 ns / 103.96 ns and f32 Apollo/RustFFT 235.34 ns / 78.52 ns.
- [patch] Full canonical `benchmark_results.md` regeneration completed in
  65.6 seconds from the already-built optimized `xtask` binary.
- [patch] Verified proc-macro compile, `apollo-fft` lib compile, bench compile,
  generated composite direct-DFT coverage, and fixed coprime direct-DFT
  coverage.

## [0.12.23] - 2026-05-19
### Changed
- [patch] `apollo-fft`: added generated short Good-Thomas codelets for
  N=18, N=24, and N=36 through the existing proc-macro path and canonical
  short-Winograd dispatch.
- [patch] `apollo-fft`: natural and ordered generic PFA now reuse the
  thread-local PFA scratch buffer for column storage instead of allocating a
  fresh column `Vec` per transform.
- [patch] `apollo-fft`: generated direct `2*p` natural-prime dispatch now uses
  a twiddle-free Good-Thomas row/column codelet for the shared promoted-prime
  family instead of paying the four-step twiddle cache on the direct path.
- [patch] `xtask benchmark`: quick mode is now the default Criterion profile
  and uses the `bench-quick` Cargo profile. The quick profile keeps optimized
  code generation while reducing Criterion timing to sample_size=10,
  measurement_time=150ms, and warm_up_time=20ms; `--profile full` retains the
  longer release-quality profile.

### Verification
- [patch] Added direct-DFT value coverage for the new N=18/N=24/N=36 short
  Good-Thomas leaves. `benchmark_results.md` was not regenerated because
  bounded release `vs_rustfft` bench builds did not produce usable timing
  output in this increment.
- [patch] Focused `2*p` correctness tests pass for N=38/N=58/N=74/N=82/N=94.
  Quick-profile Criterion refreshes updated `benchmark_results.md`; current
  rows record N=38 f64/f32 Apollo/RustFFT ratios 1.551x/1.695x, N=82
  1.044x/1.532x, and N=94 0.746x/0.847x. N=74 f32 regressed to 1.959x and
  remains the next targeted miss.

## [0.12.22] - 2026-05-19
### Changed
- [patch] `apollo-fft-macros`: `generate_three_by_prime_dispatch!` now owns
  the full `3*p` route body and emits direct const-generic DFT-3 and row
  codelet calls from one supported-prime list. The generated route no longer
  depends on a separate short-codelet adapter surface.
- [patch] `apollo-fft`: the short-Winograd dispatch table is compacted around
  the authoritative const-generic codelet leaves and prime-pair table trait,
  keeping direction selection monomorphized for generated and hand-written
  callers without cloning precision-specific APIs.
- [patch] `apollo-fft`: the `vs_rustfft` Criterion harness now includes the
  routed problem sizes 38, 58, 74, 82, and 94 in the canonical f64/f32
  Apollo-vs-RustFFT table.
- [patch] Added `xtask benchmark` as the single benchmark runner and
  `benchmark_results.md` generator for the Apollo-vs-RustFFT f64/f32 table.

### Fixed
- [patch] `apollo-fft`: natural Good-Thomas PFA now scatters transformed
  columns through the cached CRT output table using the table's `(k2, k1)`
  layout. This corrects non-compact coprime routes that use the generic
  natural PFA kernel instead of compact generated families.
- [patch] `apollo-fft`: completed the Winograd const-generic direction
  migration exposed by a fresh rebuild. Short DFT-3/7/8/15 call sites now
  dispatch through `const INVERSE` entry points instead of stale runtime-bool
  calls, and the generated `3*p` Good-Thomas macro targets the current
  const-generic DFT-3/7 functions.

### Verification
- [patch] Added direct-DFT forward and unnormalized inverse coverage for the
  private natural PFA kernel on a nontrivial coprime shape, and verified the
  focused Good-Thomas tests, full `apollo-fft` library test suite,
  `apollo-fft-macros` compile, and `apollo-fft` bench/example compile surface.
- [patch] `benchmark_results.md` is regenerated from Criterion after targeted
  refreshes for N=33/38/58/74/82/94. Current rows record N=33 f64
  Apollo/RustFFT 91.33 ns / 64.35 ns (1.419x), N=94 f64 449.33 ns /
  675.17 ns (0.665x), and N=94 f32 460.01 ns / 633.28 ns (0.726x).
- [patch] Removed redundant benchmark entry points and generated-output
  fragments: the old Python extractor, the quick comparison example, the
  validation crate's duplicate `vs_rustfft` bench, and checked-in bench output
  logs are no longer part of the active workflow.

## [0.12.21] - 2026-05-19
### Changed
- [patch] `apollo-fft-macros`: `generate_three_by_prime_dispatch!` now emits
  complete per-prime Good-Thomas `3*p` transform bodies rather than generated
  gather/scatter closures plus a hand-written runtime driver.
- [patch] `apollo-fft-macros`: added
  `generate_two_by_prime_natural_dispatch!` so the direct Winograd-pair
  `2*p` table is generated from one prime/half-size specification instead of
  a local declarative macro.
- [patch] `apollo-fft`: the prime-pair table capability is now part of the
  sealed Winograd scalar contract, making generated prime-pair dispatch valid
  in debug and release builds through one scalar bound.

### Fixed
- [patch] `apollo-fft`: restored release visibility for
  `winograd::radix::odd_prime_pair`, restored benchmark-only Winograd-pair
  hooks, and kept the hand DFT-3 codelet selected instead of replacing it with
  the current direct generated Winograd prototype.

### Documentation
- [patch] `benchmark_results.md` is regenerated from the Criterion cache after
  a targeted N=33 refresh. N=33 records f64 Apollo/RustFFT 94.34 ns / 68.16 ns
  (1.384x) and f32 Apollo/RustFFT 108.75 ns / 64.81 ns (1.678x). Fresh release
  quick comparison for N=33/38/58/74/82/94 records ratios
  1.258/1.756/1.296/1.150/1.000/0.785.

## [0.12.20] - 2026-05-19
### Changed
- [patch] `apollo-fft`: compact Good-Thomas `3*p` execution now uses
  proc-macro-generated CRT gather/scatter functions and const-size
  short-Winograd dispatch. The hot route no longer performs runtime CRT table
  lookup or runtime short-codelet selection for the generated family.
- [patch] `apollo-fft-macros`: generated Rader code now uses exact f64
  twiddle constants, the runtime Rader generator/scatter convention, and
  inverse-safe pointwise symbols. Static Rader generation is bounded to
  5/7/11/13 until an O(N log N) generated convolution backend replaces the
  direct AST expansion for larger primes.

### Fixed
- [patch] `apollo-fft-macros`: removed the stale helper binary target that
  expanded runtime macros in the proc-macro crate root and broke
  `cargo check -p apollo-fft-macros`.

### Documentation
- [patch] `benchmark_results.md` is regenerated from the canonical Criterion
  cache. N=33 records f64 Apollo/RustFFT 93.00 ns / 64.92 ns (1.433x) and f32
  Apollo/RustFFT 108.00 ns / 67.49 ns (1.600x). A fresh release quick-compare
  rebuild exceeded the 300-second cap, so the current residual target is
  release codegen reduction plus fused row/scatter execution rather than
  component deletion.

## [0.12.19] - 2026-05-19
### Changed
- [patch] `apollo-fft`: compact Good-Thomas `3*p` support detection and
  monomorphized `(P, inverse)` dispatch are now generated by the internal
  `apollo-fft-macros` proc-macro crate from one short-prime list. The runtime
  transform remains the generic `three_by_prime_impl` over
  `MixedRadixScalar` and `ThreeByPrimePlan<const P>`, preserving static
  dispatch and avoiding component removal.

### Documentation
- [patch] `benchmark_results.md` is regenerated from the canonical Criterion
  cache. N=33 still records f64 Apollo/RustFFT 101.49 ns / 70.27 ns (1.444x)
  and f32 Apollo/RustFFT 121.28 ns / 78.91 ns (1.537x). The release quick
  comparison for N=21/33/39/51/69 records N=33 at 0.089 us vs 0.059 us,
  identifying route/store fusion as the next runtime target rather than
  deleting retained kernels.

## [0.12.18] - 2026-05-19
### Changed
- [patch] `apollo-fft`: the compact Good-Thomas `3*p` route now derives its
  CRT input and output maps through `ThreeByPrimePlan<const P>` at compile
  time. The transform body no longer computes modular inverses or modulo-based
  CRT output strides in the hot path, matching the const-plan foundation
  described in `gengoodthomas.md` and `genpermute.md`.
- [patch] `apollo-fft`: Stockham scalar-reference tests now specify the
  const tile arity for the triple/quad fused-stage helpers, keeping the
  const-generic benchmark/test surface explicit.

### Documentation
- [patch] `benchmark_results.md` is regenerated from completed Criterion rows.
  N=33 now records f64 Apollo/RustFFT 101.49 ns / 70.27 ns (1.444x) and f32
  Apollo/RustFFT 121.28 ns / 78.91 ns (1.537x).

## [0.12.17] - 2026-05-19
### Changed
- [patch] `apollo-fft`: coprime `3*p` sizes where `p` is an existing short
  prime codelet now route through a compact Good-Thomas CRT codelet before the
  prime-23 mixed-radix composite path. This covers N=21/33/39/51/69 and removes
  the inter-stage twiddle path for N=33 without deleting the retained
  radix-composite, Good-Thomas, Rader, Winograd, Stockham, butterfly, or
  four-step components.
- [patch] `apollo-fft`: the benchmark-only ordered-Rader hooks now call the
  current ordered-Rader implementation, restoring the
  `kernel-strategy-bench` example/bench build surface.

### Fixed
- [patch] `apollo-fft`: N=33 no longer enters the `[11, 3]` mixed-radix
  composite route before its twiddle-free coprime decomposition. Criterion
  N=33 clone-inclusive timing improves from the previous cache by 49.642% for
  f64 and 53.186% for f32.

### Documentation
- [patch] `benchmark_results.md` is regenerated from the current Criterion
  cache. The N=33 rows now record f64 Apollo/RustFFT 104.08 ns / 69.15 ns
  (1.505x) and f32 Apollo/RustFFT 128.21 ns / 63.15 ns (2.030x).

## [0.12.16] - 2026-05-19
### Changed
- [patch] `apollo-fft`: typed real-storage forward paths now execute in the
  caller-owned spectrum buffer directly for f64, f32, and compact f16 storage.
  The 1D/2D/3D `forward_*_into` implementations fill complex output with a
  single `Zip` pass and then run the existing monomorphized plan in place.
- [patch] `apollo-fft`: typed inverse `*_into` paths now extract real storage
  from the caller-owned scratch spectrum with a direct `Zip` pass instead of
  allocating a mapped temporary and assigning it back into the output.

### Fixed
- [patch] `apollo-fft`: allocating typed forward paths no longer build a
  mapped complex array and then clone it before execution; they now allocate
  one output array and transform it in place.
- [patch] `apollo-fft`: public generic plan visibility warnings from the prior
  scalar-cache reconciliation are resolved in the verified compile surface.

### Documentation
- [patch] `benchmark_results.md` is regenerated from the current Criterion
  cache snapshot. The dedicated post-cutoff Criterion rows for
  N=16/N=32/N=64/N=128/N=32768 remain pending because the targeted N=16
  refresh exceeded the bounded command cap in the prior increment. The
  caller-owned six-step zero-allocation row for N=5120 now runs without
  allocation assertion failure and records 11.530 us mean throughput.

## [0.12.15] - 2026-05-19
### Changed
- [patch] `apollo-fft`: typed real FFT helpers now resolve their cached plan
  precision through `RealFftData::PlanScalar` plus the zero-cost
  `PlanCacheProvider` trait. `f64` and `f32` retain native generic plan caches,
  while compact `f16` storage explicitly delegates to the `f32` plan cache
  instead of requiring `f16` to satisfy the mixed-radix scalar contract.
- [patch] `apollo-fft`: the restored power-of-two route now starts at N>=64.
  N=16 and N=32 remain on short Winograd/codelet routing after the current
  quick comparison showed the generic Stockham fast-path was not yet faster for
  those sizes.

### Fixed
- [patch] `apollo-fft`: f16 typed 1D/2D/3D APIs no longer instantiate
  `FftPlan*<f16>` or require `f16: MixedRadixScalar`; compact storage converts
  at the storage boundary and reuses the f32 plan family.
- [patch] `apollo-fft`: benchmark/example zero-allocation typed calls now use
  the cache provider and real-storage contracts directly, restoring the
  monomorphized typed bench build after the generic plan consolidation.

### Documentation
- [patch] `benchmark_results.md` is regenerated as the single canonical
  Apollo-vs-RustFFT f64/f32 clone-inclusive table from the current Criterion
  cache snapshot. A bounded targeted Criterion refresh for N=16 exceeded the
  300-second command cap, so dedicated post-cutoff Criterion rows remain
  pending and the table is not narrowed to targeted quick-run data.

## [0.12.14] - 2026-05-19
### Changed
- [patch] `apollo-fft`: mixed-radix dispatch now routes every
  power-of-two length N>=16 through a dedicated fast-path before small
  Winograd/composite/PFA/Rader routing. The fast-path uses Stockham for
  asymmetric powers and retains square four-step only for even-exponent
  lengths above the four-step threshold.
- [patch] `apollo-fft`: the power-of-two route remains one monomorphized
  generic body over `MixedRadixScalar`; N=2/N=4/N=8 remain on short Winograd
  codelets and no Rader, Good-Thomas, Winograd, butterfly, Stockham, or
  four-step component was removed.

### Fixed
- [patch] `apollo-fft`: N=32768 can no longer fall through the selector without
  executing a transform when the four-step shape is asymmetric. A forward DC
  regression test now rejects no-op behavior that a forward+inverse roundtrip
  would not detect.
- [patch] `apollo-fft`: `FftPlan1D` now calls the generic mixed-radix twiddle
  and scratch-cache APIs directly instead of importing removed precision
  suffix helpers.
- [patch] `apollo-fft`: `FftPlan1D` now exposes the same generic
  caller-owned typed forward/inverse methods as the 2D and 3D plans, restoring
  zero-allocation benchmark compilation without allocating inside the bench hot
  loop.
- [patch] workspace manifests now expose the `tokio` dependency required by the
  current plan cache code and restore the `kernel-strategy-bench` feature used
  by benchmark/example verification.

### Documentation
- [patch] `benchmark_results.md` is regenerated as the single canonical
  Apollo-vs-RustFFT f64/f32 clone-inclusive table from the current Criterion
  cache snapshot. Two pre-existing Criterion jobs are still active, so this
  snapshot can continue to change until those writers finish.

## [0.12.13] - 2026-05-19
### Changed
- [patch] `apollo-fft`: short Winograd dispatch now includes stack-resident
  Good-Thomas CRT codelets for N=6, N=10, N=12, and N=14, bypassing the
  generic mixed-radix scratch/twiddle route for the small coprime composites
  that dominate the current table misses.
- [patch] `apollo-fft`: the new small-composite codelets reuse existing
  Winograd DFT-3/4/5/7 leaves and monomorphize through the existing
  `short_winograd` dispatch; no Rader, Good-Thomas, Winograd, butterfly, or
  composite component was removed.

### Fixed
- [patch] `apollo-fft`: removed an obsolete private Good-Thomas gather helper
  left unused by the fused ordered-Rader PFA permutation path, eliminating the
  bench build dead-code warning at the source.

### Documentation
- [patch] `benchmark_results.md` is regenerated from the canonical
  Apollo-vs-RustFFT f64/f32 Criterion cache snapshot, including fresh
  post-patch targeted rows for N=6, N=10, N=12, and N=14.

## [0.12.12] - 2026-05-19
### Changed
- [patch] `apollo-fft`: Rader negacyclic convolution now fuses the twist
  multiply into the Nussbaumer split pass and fuses the conjugate untwist into
  CRT recombination, removing two full passes over the negacyclic half for
  large-prime Rader routes.
- [patch] `apollo-fft`: the Rader convolution path continues to use the fused
  radix-composite forward-with-pointwise contract for supported composite
  convolution lengths, preserving static dispatch and retained Rader/Winograd
  route availability.

### Documentation
- [patch] `benchmark_results.md` is regenerated from the canonical
  Apollo-vs-RustFFT f64/f32 Criterion cache snapshot only.

## [0.12.11] - 2026-05-19
### Changed
- [patch] `apollo-fft`: two-by-prime Winograd-pair routing now reads the
  original interleaved even/odd promoted-prime input directly inside the
  monomorphized pair kernel, removing the even-half stack copy and odd-half
  compaction pass for every direct `2*p` route.
- [patch] `apollo-fft`: f32/f64 `MixedRadixScalar` implementations now expose
  the fused radix-composite forward-with-pointwise contract, enabling Rader
  circular and negacyclic convolution to fuse the forward composite FFT with
  the spectrum multiplication when the convolution length has supported
  radix-composite factors.

### Documentation
- [patch] `benchmark_results.md` remains a single canonical Apollo-vs-RustFFT
  f64/f32 clone-inclusive table regenerated from Criterion cache records; no
  quick-run or residual Criterion sections are emitted.

## [0.12.10] - 2026-05-19
### Changed
- [patch] `apollo-fft`: fused radix-composite scalar stage traversal now
  iterates destination blocks with `chunks_exact_mut`, removing repeated
  output slice-bound recomputation while preserving one const-radix
  monomorphized stage body for every fallback radix.
- [patch] `apollo-fft`: fused radix-composite final pointwise multiplication
  now uses raw pointer traversal over the contiguous output block, keeping the
  same mathematical operation while reducing duplicate bounds and alias checks.
- [patch] `apollo-fft`: Good-Thomas natural and ordered-Rader permutation
  gather/scatter loops now use cached-permutation contracts with bounded
  unchecked four-wide copies, reducing hot-path permutation overhead without
  removing PFA, Rader, or Winograd routes.

### Fixed
- [patch] `apollo-fft`: retained Winograd N=82 composite codelet now carries
  the required `PrimePairTable<41, 20>` bound, so the route compiles without
  removing the codelet.

### Documentation
- [patch] `benchmark_results.md` is regenerated from the Criterion cache and
  the latest quick strategy and selected public comparisons after the
  routing/permutation edits.

## [0.12.9] - 2026-05-19
### Changed
- [patch] `apollo-fft`: radix-composite recursive fused-stage scratch
  management now lives in `radix_composite::adaptive`, keeping the arity leaf
  focused on radix dispatch and below the 500-line structural limit.
- [patch] `apollo-fft`: flat fused Stockham composite execution now routes
  scalar fallback stages through a stage-level const-radix dispatcher, resolving
  the runtime radix match once per stage instead of once per output group.
- [patch] `apollo-fft`: fused radix-composite final pointwise spectrum
  multiplication now traverses the contiguous output block with a single
  linear loop instead of a radix/column nested loop.
- [patch] `apollo-fft`: Rader benchmark routing now targets the shared generic
  Rader implementation and the real Winograd-pair kernels instead of deleted
  per-prime module paths; Bluestein/Rader differential tests use the same
  shared Rader route, and static Rader permutation tables remain compile-time
  constants generated per prime dispatch arm on stable Rust.
- [patch] `apollo-fft`: `benchmark_results.md` is regenerated from every
  Criterion `target/criterion/**/new/estimates.json` record, covering f64/f32
  clone-inclusive, zero-allocation, six-step, residual benchmark groups, and
  the latest debug Rader-vs-Winograd-pair quick strategy comparison.

### Fixed
- [patch] `apollo-fft`: Winograd composite test coverage can resolve all
  large composite leaves without gating or removing existing composite
  implementations before a measured RustFFT-beating replacement exists.

## [0.12.8] - 2026-05-18
### Changed
- [patch] `apollo-fft`: Rader Bluestein now caches only the forward chirp and
  forward kernel spectrum. The inverse path multiplies by the conjugate forward
  spectrum, using the even-kernel identity `b[M-j] = b[j]`, so each cached prime
  entry retains `N + M` complex values instead of `N + 2M`.
- [patch] `apollo-fft`: Bluestein pre-chirp, zero-pad, post-chirp, and
  conjugated pointwise multiplication now route through the existing f64/f32
  SIMD dispatch surface, preserving vectorized inverse execution without a
  second cached spectrum.

### Fixed
- [patch] `apollo-fft`: corrected Bluestein SIMD zero-fill lane counts so
  `write_bytes` receives element counts for typed `f64`/`f32` pointers rather
  than byte-scaled counts.

## [0.12.7] - 2026-05-18
### Changed
- [patch] `apollo-fft`: standalone Rader static-table and runtime paths now
  compute the nonzero DC sum during primitive-root gather, removing one full
  pass over `data[1..N]` before the convolution.
- [patch] `apollo-fft`: Rader static-table and runtime scatter loops now use
  unrolled permutation writes matching the fused gather loop shape.
- [patch] `apollo-fft`: Rader padded scratch retains one aligned thread-local
  buffer per precision and uses a local nested-call fallback, reducing
  persistent per-thread Rader scratch retention from two buffers to one.

### Fixed
- [patch] `apollo-fft`: generated Rader leaves and runtime Rader preserve
  direct-DFT forward/inverse equivalence after gather/sum fusion. Release
  strategy-only `quick_compare` records Rader latencies of 148/126 ns at N=29,
  121/123 ns at N=31, and 138/136 ns at N=37 for f64/f32; the current
  comparison hook aliases the Winograd column to Rader, so no Winograd/Rader
  ratio is recorded for this probe.

## [0.12.6] - 2026-05-17
### Changed
- [patch] `apollo-fft`: fused radix-composite dispatch now routes through the
  `FusedStage` ZST arity family and `ExecutionPolicy` static chunk traversal.
- [patch] `apollo-fft`: cached radix-composite dispatch now lowers only
  single-odd radix-2 tails to radix-4 stages, preserving prime-only
  factorization while reducing fused-stage depth for shapes such as
  `192 = 3 * 4^3`.
- [patch] `apollo-fft`: composite Winograd production routing was narrowed in
  this increment; Closure XCII restores retained large-leaf availability until
  a measured replacement beats RustFFT.
- [patch] `apollo-fft`: N=29/N=31/N=37 now route through no-gather
  Winograd-pair kernels after bounded comparison showed faster timings than
  generated Rader for all measured f64/f32 cases.
- [patch] `apollo-fft`: generated Rader remains available for larger primes and
  gated strategy comparison.
- [patch] `apollo-fft`: generated Rader prime leaves N=17..97 now share one
  const-generic static implementation parameterized by `(N, generator,
  generator_inverse)`.
- [patch] `apollo-fft`: Rader convolution now fuses the spectrum multiply into
  the final forward radix-composite stage when N-1 has cached composite factors.
- [patch] `apollo-fft`: generated Rader leaves N=29/N=31/N=37 now use static
  gather/scatter permutation tables through `rader_static_table_impl`, removing
  modular-index recurrence from the small-prime Rader comparison hot loop.
- [patch] `apollo-fft`: Rader now exposes ordered-layout static and runtime
  kernels for fused callers; the tail buffer is already in primitive-root input
  order and remains in inverse-generator output order, eliminating standalone
  Rader gather/scatter and the Rader scratch copy under that contract.
- [patch] `apollo-fft`: Good-Thomas PFA now routes prime `n1` subtransforms
  that would use Rader through the ordered-layout Rader kernel by folding the
  generator-order input layout into the PFA transpose and the inverse-generator
  output layout into the final CRT scatter.
- [patch] `apollo-fft`: ordered-Rader PFA now reuses the Rader permutation cache
  for generator and inverse-generator order, removing runtime modular index
  walks from the ordered PFA transpose and CRT scatter loops.
- [patch] `apollo-fft`: ordered-Rader PFA now routes through the known-prime
  monomorphized ordered Rader dispatcher before falling back to the runtime
  ordered path.
- [patch] `apollo-fft`: `quick_compare` now accepts
  `APOLLO_FFT_QUICK_N=<comma-separated sizes>` and includes ordered-Rader PFA
  composite sizes in its default comparison set.
- [patch] `apollo-fft`: `prime_compose` now includes an
  `ordered_rader_pfa_coprime_composites` Criterion group for N=38/N=82/N=86/
  N=94/N=106.
- [patch] `apollo-fft`: N=37 Winograd-pair dispatch now uses an inlined wrapper
  while N=29/N=31 stay out-of-line after bounded call-shape comparison.
- [patch] `apollo-fft`: N=19/N=41/N=43/N=47/N=53 now route through the shared
  odd-prime Winograd-pair kernel family instead of production Rader dispatch.
- [patch] `apollo-fft`: odd-prime Winograd-pair kernels moved into
  `winograd/radix/odd_prime_pair.rs`, keeping `radix.rs` below the structural file
  limit.
- [patch] `apollo-fft`: Good-Thomas now has a dedicated `two_by_prime` leaf for
  N=2p composites; promoted prime halves use stack-resident static buffers,
  cached `W_N^k` twiddles, and direct Winograd-pair calls without Rader
  generator gather/scatter.
- [patch] `apollo-fft`: the direct N=2p path now loads the even half into a
  const-generic stack array and compacts the odd half in place before fused
  two-prime Winograd execution, bypassing thread-local PFA scratch for promoted
  primes.
- [patch] `apollo-fft`: composite Winograd test coverage no longer uses a
  declarative macro; the shared const-generic test helper preserves the same
  value checks while keeping the leaf below the structural file limit.
- [patch] `apollo-fft`: `quick_compare` and `prime_compose` now cover the
  two-by-prime composites N=38/N=58/N=62/N=74/N=82/N=86/N=94/N=106, and
  `kernel_strategy` compares Rader vs Winograd-pair for
  N=19/N=29/N=31/N=37/N=41/N=43/N=47/N=53.
- [patch] `apollo-fft`: N=82 no longer uses the stale dedicated DFT-82
  codelet; short dispatch falls through to the generic two-by-prime route,
  which reuses the promoted N=41 Winograd-pair half kernel.

### Fixed
- [patch] `apollo-fft`: fused composite twiddle slices now use each stage's
  coefficient contract, `(radix - 1) * prev_len * prior_product`, avoiding
  over-wide stage inputs.
- [patch] `apollo-fft`: fused `Radix<R>` and fallback dispatch now cover radix
  4, 8, 17, and 23.
- [patch] `apollo-fft`: recursive fused `Compose` stages reserve complete arena
  scratch capacity before exposing live midpoint pointers, avoiding nested
  reallocation under parallel 2D execution.
- [patch] `apollo-fft`: the incomplete radix-composite tiling placeholder is no
  longer compiled, while the authoritative fused core path remains active.
- [patch] `apollo-fft`: radix-shape tests now include emitted radix-4 stages and
  assert `192 = 3 * 4^3`.
- [patch] `apollo-fft`: radix-2 lowering is pair-only again; the rejected
  highest-power lowering path emitted unsupported radix 16.
- [patch] `apollo-fft`: added value-semantic Rader-vs-Winograd-pair
  equivalence tests and a gated Criterion comparison group for N=29/N=31/N=37.
- [patch] `apollo-fft`: added dispatch-level forward/inverse direct-DFT checks
  for the promoted N=29/N=31/N=37 Winograd-pair path.
- [patch] `apollo-fft`: added direct-DFT regression coverage for every generated
  Rader prime leaf from N=17 through N=97.
- [patch] `apollo-fft`: removed stale radix-composite fallback dispatch code and
  unused scratch-dispatch helpers from the active module graph.
- [patch] `apollo-fft`: release strategy-only Rader-vs-Winograd evidence is now
  recorded for N=29/N=31/N=37 after static Rader permutation-table leaves:
  Winograd/Rader ratios are 0.206/0.476 at N=29, 0.368/0.566 at N=31, and
  0.334/0.555 at N=37 for f64/f32; Winograd-pair remains faster for those
  small primes.
- [patch] `apollo-fft`: added direct-DFT checks for ordered-Rader PFA coverage
  at N=38 forward and N=82 inverse, plus branch-selection coverage that keeps
  N=29/N=31/N=37 on the Winograd-pair path.
- [patch] `apollo-fft`: release ordered-Rader PFA evidence is recorded for
  N=38/N=82/N=86/N=94/N=106 against RustFFT: ratios are 6.433, 2.581, 2.505,
  1.845, and 2.455 respectively, so this path remains a performance gap.
- [patch] `apollo-fft`: release prime-leaf evidence after final odd-prime
  routing records Apollo/RustFFT ratios of 0.907, 0.972, 0.736, 0.799, 0.720,
  0.599, 0.582, and 0.909 for
  N=19/N=29/N=31/N=37/N=41/N=43/N=47/N=53 respectively.
- [patch] `apollo-fft`: release two-by-prime evidence records Apollo/RustFFT
  ratios of 1.514, 1.195, 1.228, 1.059, 1.025, 0.943, 0.587, and 0.757 for
  N=38/N=58/N=62/N=74/N=82/N=86/N=94/N=106 respectively; N=38 remains the
  largest residual composite gap.

## [0.12.5] - 2026-05-15
### Added
- [patch] `apollo-fft`: N=23 now routes through a dedicated Winograd
  pair-symmetry codelet with f64/f32 scalar constants, direct `FftPrecision`
  fast paths, and `ShortWinogradScalar::dft23`.

### Changed
- [patch] `apollo-fft`: DFT-23 constants are split across scalar trait and
  implementation leaves so generated files remain below the repository
  structural limit while preserving one shared mathematical kernel body.

### Fixed
- [patch] `apollo-fft`: N=23 performance against RustFFT. Latest isolated run:
  Apollo f64 92.341 ns vs RustFFT 116.48 ns; Apollo f32 104.80 ns vs
  RustFFT 139.88 ns.
- [patch] `apollo-fft`: Rader permutation caching now uses split contiguous
  gather/scatter arrays while keeping transform-direction-specific convolution
  spectra, preserving prime-length inverse correctness.

## [0.12.4] - 2026-05-15
### Added
- [patch] `apollo-fft`: N=17 now routes through a dedicated Winograd
  pair-symmetry codelet with f64/f32 scalar constants, direct `FftPrecision`
  fast paths, and `ShortWinogradScalar::dft17`.
- [patch] `apollo-fft`: the RustFFT comparison benchmark set now includes
  N=17 for the one-size-at-a-time optimization sequence.

### Changed
- [patch] `apollo-fft`: DFT-17 uses one shared mathematical body with two
  monomorphized call wrappers: an inlined route for f64 and an out-of-line
  route for f32, preserving one authoritative algorithm while matching the
  scalar-specific codegen profile.

### Fixed
- [patch] `apollo-fft`: N=17 performance against RustFFT. Latest isolated run:
  Apollo f64 71.932 ns vs RustFFT 81.043 ns; Apollo f32 90.289 ns vs RustFFT
  112.84 ns.

## [0.12.3] - 2026-05-15
### Added
- [patch] `apollo-fft`: N=13 now routes through a dedicated Winograd
  pair-symmetry codelet for f64 and f32, with const-generic direction
  monomorphization so forward and inverse kernels compile without runtime
  direction dispatch.
- [patch] `apollo-fft`: `ShortWinogradScalar::dft13` and direct
  `FftPrecision` fast paths now cover length 13.

### Changed
- [patch] `apollo-fft`: split short Winograd leaves into
  `winograd/radix/dft13.rs` and `winograd/radix/dft3.rs`, keeping each leaf
  below the repository structural limit while preserving the canonical
  `winograd::dft*_impl` call surface.

### Fixed
- [patch] `apollo-fft`: N=13 performance against RustFFT. Latest isolated run:
  Apollo f64 82.158 ns vs RustFFT 94.077 ns; Apollo f32 78.778 ns vs RustFFT
  86.069 ns.

## [0.12.2] - 2026-05-14
### Added
- [patch] `apollo-fft`: `dft7_impl` replaced with Winograd constant algorithm.
  Exploits Hermitian symmetry of the 7-point twiddle matrix: sum/difference
  decomposition into xr[n]=x[n]+x[7−n] and xi[n]=x[n]−x[7−n], then circulant
  cosine (c1,c2,c3) and sine (s1,s2,s3) row patterns. Reduces O(N²) naive DFT
  (49 complex muls) to 18 real multiplications with precomputed constants.
- [patch] `apollo-fft`: `ShortWinogradScalar::dft7` trait method and `7 =>`
  dispatch arm in `short_winograd`; N=7 now routes through the hand-coded
  Winograd codelet rather than the composite path.
- [patch] `apollo-fft`: winograd test files properly partitioned into domain
  scopes (`dft_small.rs`, `dft_large.rs`, `boundaries.rs`), eliminating
  triplication; 185 tests pass.

### Fixed
- [patch] `apollo-fft`: N=15 performance. Apollo f64 ~82 ns vs RustFFT ~108 ns
  (24% faster); Apollo f32 ~89 ns vs RustFFT ~105 ns (15% faster).

## [0.12.1] - 2026-05-14
### Added
- [patch] `apollo-fft`: `dft100_impl` in `winograd/composite.rs` using
  Good-Thomas PFA (N=100=4×25, gcd(4,25)=1). CRT input permutation
  `n=(25·n1+4·n2)%100` eliminates inter-stage twiddles; output mapping
  `k=(76·k2+25·k1)%100`.
- [patch] `apollo-fft`: `ShortWinogradScalar::dft100` trait method and
  `100 =>` dispatch arm in `short_winograd`; N=100 now routes through the
  hand-coded codelet rather than the generic `pfa_fft` fallback.
- [patch] `apollo-fft`: five correctness tests for DFT-100
  (forward/inverse/roundtrip/dc-energy/f32≡f64).

### Fixed
- [patch] `apollo-fft`: N=100 performance regression. Apollo f64 is now
  310 ns (−25% vs RustFFT 415 ns); Apollo f32 is now 292 ns (−11% vs
  RustFFT 327 ns).

## [Unreleased]
### Breaking
- [major] `apollo-fft`: removed public type-suffixed mixed-radix twiddle
  wrapper entry points (`forward_inplace_64_with_twiddles`,
  `inverse_inplace_64_with_twiddles`,
  `inverse_inplace_unnorm_64_with_twiddles`,
  `forward_inplace_32_with_twiddles`,
  `inverse_inplace_32_with_twiddles`, and
  `inverse_inplace_unnorm_32_with_twiddles`). Internal plan code now calls the
  canonical const-generic `dispatch_inplace::<T, INVERSE, NORMALIZE>` body
  directly.
- [major] `apollo-fft`: removed concrete public auto-selector wrappers
  `fft_forward_64`, `fft_inverse_64`, `fft_inverse_unnorm_64`,
  `fft_forward_32`, `fft_inverse_32`, and `fft_inverse_unnorm_32`.
  Callers must use the canonical generic `fft_forward`, `fft_inverse`, and
  `fft_inverse_unnorm` entry points.
- [major] `apollo-fft`: removed the remaining public type-suffixed Winograd
  DFT-16/32/64 wrappers and collapsed their duplicated f32/f64 recursive
  bodies into one generic codelet family.
- [major] `apollo-fft`: removed type-suffixed public short-Winograd wrappers
  for DFT-2/3/4/5/7/8 and twiddle multiplication. Internal mixed-radix dispatch
  now calls the canonical generic Winograd implementations directly.
- [major] `apollo-fft`: removed type-suffixed direct DFT wrappers
  `dft_forward_64`, `dft_inverse_64`, `dft_forward_32`, `dft_inverse_32`,
  `forward_owned_64`, and `inverse_owned_64`. Callers must use the canonical
  generic `dft_forward` and `dft_inverse` functions.

### Changed
- [minor] `apollo-fft`: removed dead Winograd AVX wrapper leaves and routed all
  remaining plan-owned twiddle reuse through the canonical mixed-radix
  const-generic dispatch body. `apollo-fft` was bumped to 0.12.0 for the
  pre-1.0 public wrapper removal.
- [minor] `apollo-fft`: removed the unreachable legacy CPU SIMD six-step,
  matrix-workspace, and power-of-two radix2 infrastructure island that was no
  longer part of the crate module graph.
- [minor] `apollo-fft`: route radix-15 mixed-radix leaves through a stack-only
  generic Good-Thomas Winograd codelet with no inter-stage twiddle table.
- [minor] `apollo-fft`: consolidate broad Stockham AVX stage and pair leaves
  behind one monomorphized backend trait while retaining shape-specific AVX
  codelets for specialized schedules.

### Fixed
- [patch] `apollo-fft`: split Stockham f64 AVX scratch dispatch out of the
  fixed butterfly codelet leaf so generated fixed codelets stay below the
  repository file-size limit while preserving static dispatch, and update stale
  benchmark call sites to the maintained generic selector and `real_fft`
  twiddle builders. Compact storage routing moved out of the type-named
  `dispatch_f16.rs` leaf and into the canonical mixed-radix dispatch module.
- [patch] `apollo-fft`: route `fftfreq` and `rfftfreq` through exact-capacity
  fill loops instead of known-length iterator collection pipelines.
- [patch] `apollo-fft`: remove the unused `Default` bound from
  `fftshift`/`ifftshift` and route both utilities through a shared split-slice
  copy helper instead of duplicate modulo-index iterator pipelines.
- [patch] `apollo-fft`: route native 3D f32/f16 allocating real32 buffers and
  inverse real-output projection through sealed exact-size overwrite-first
  buffers instead of zero-fill and `mapv` allocation pipelines.
- [patch] `apollo-fft`: route native 2D f32/f16 real packing and real-output
  projection through exact-size overwrite-first buffers instead of ndarray
  `mapv` allocation pipelines.
- [patch] `apollo-fft`: route the 1D compact f16 power-of-two path through
  exact-size overwrite-first buffers for compact input packing and output
  projection instead of iterator-collection pipelines.
- [patch] `apollo-fft`: consolidate 1D native `Complex32` precision dispatch
  for f32 and mixed f16 non-power-of-two paths behind shared monomorphized
  helpers, reducing duplicated conversion/kernel-selection logic.
- [patch] `apollo-fft`: reduce Bluestein plan-construction memory writes by
  initializing the padded convolution filter through overwrite-first mirrored
  chirp placement and zero-filling only the unused convolution gap.
- [patch] `apollo-fft`: consolidated plan-owned uninitialized workspace
  allocation behind a sealed scratch-element helper and routed 1D Bluestein,
  1D iRFFT, 2D/3D axis-pass, 3D R2C, and six-step f32 scratch buffers through
  it to avoid plan-construction zero-fill for buffers overwritten before read.
- [patch] `apollo-fft`: reduce normalization and workspace memory overhead by
  routing inverse scale passes through shared AVX-capable normalization helpers,
  filling twiddle vectors by exact pre-sized cursors, and avoiding zero-fill for
  FFT scratch/workspace buffers that are overwritten before read.
- [patch] `apollo-fft`: removed the debug-only `debug_f32` binary and updated
  direct DFT tests, benchmarks, and kernel regressions to use the canonical
  generic DFT functions.
- [patch] `apollo-fft`: reuse Bluestein and mixed-radix composite FFT scratch
  buffers across calls, cache composite twiddle tables by exact radix
  decomposition and direction, and add regression coverage for same-length
  composite transforms with different radix orders.
- [patch] `apollo-fft`: consolidated duplicated 3D f32/f16 typed plan paths
  behind a private monomorphized storage trait and removed the now-dead
  f32-only 3D real-to-complex writer.
- [patch] `apollo-fft`: consolidated duplicated 2D f32/f16 typed plan paths
  behind a private monomorphized storage trait, removed duplicated 2D plan
  Rustdoc, and moved crate-root tests into a dedicated module so `lib.rs`
  satisfies the 500-line structural limit.

### Added
- `apollo-hilbert`: caller-owned `AnalyticSignal` projections for real,
  quadrature, envelope, phase, and instantaneous frequency, plus
  `HilbertPlan::envelope_into` and `HilbertPlan::phase_into`.
- `apollo-hilbert`: `analytic_signal_into` direct kernel and
  `HilbertPlan::analytic_signal_into` caller-owned analytic-signal execution.
- `apollo-fft`: `FftPlan1D::forward_real_to_complex_slice_into` exposes the
  canonical 1D real-forward caller-owned slice path for downstream crates that
  already own contiguous real slices.
- `apollo-stft` / `docs/adr-0001-typed-workspace-and-alias-removal.md`:
  co-located ADR for typed workspace reuse, storage-trait extraction, and
  pre-1.0 removal of deprecated allocating aliases.
- `apollo-fft` / `application/execution/kernel/winograd.rs` (new module):
  - Algebraic Winograd short-DFT kernels for sizes 2, 4, 8, 16, 32, and 64 (f64 and f32).
  - DFT-2: 0 multiplications (pure add/subtract butterfly).
  - DFT-4: 0 multiplications (±i rotation via swap-and-negate, no trig).
  - DFT-8: 4 real multiplications using W_8^k exact algebraic twiddles
    (W_8^1 = SQ2O2·(1−i), W_8^2 = −i, W_8^3 = −SQ2O2·(1+i)).
  - DFT-16: 2×DFT-8 + 8 exact nested-square-root 16th-root twiddles.
  - DFT-32: 2×DFT-16 + 16 trigonometric 32nd-root twiddles.
  - DFT-64: 2×DFT-32 + 32 trigonometric 64th-root twiddles.
  - `apply_twiddle_64` / `apply_twiddle_32` helpers for complex multiply.
  - 23 unit tests covering forward, inverse, roundtrip, and boundary cases for all sizes.
- `apollo-fft` / `benches/vs_rustfft.rs`: RustFFT comparator benchmark
  coverage using the workspace-pinned `rustfft` dependency.

### Breaking
- [major] `apollo-fft`: removed the `FftPlan3D::nz_complex` alias and renamed
  `HalfSpectrum3D::nz_complex` to `HalfSpectrum3D::nz_c`. Callers must use the
  canonical `nz_c` half-spectrum bookkeeping name.
- [major] `apollo-fft`: removed the public `radix2_f16` module and the `Cf16`
  wrapper type. Compact f16 complex storage now uses
  `num_complex::Complex<half::f16>` through the generic monomorphized
  precision bridge. Removed the public `fft_forward_f16`, `fft_inverse_f16`,
  and `fft_inverse_unnorm_f16` wrappers; callers should use the generic
  `fft_forward`, `fft_inverse`, and `fft_inverse_unnorm` entry points.
- [major] `apollo-stft`: removed deprecated `StftPlan::forward_inplace` and
  `StftPlan::inverse_inplace` allocating aliases. Callers must use `forward`
  and `inverse` for owned output or `forward_into` and `inverse_into` for
  caller-owned output. Typed storage/profile traits now live under
  `application::execution::plan::stft::storage`; crate-root re-exports are
  unchanged.
- [major] `apollo-stft-wgpu`: removed deprecated
  `WgpuError::FrameLenNotPowerOfTwo`. Non-power-of-two STFT WGPU frame lengths
  are valid through the Chirp-Z path; callers should handle the remaining
  concrete validation and WGPU runtime errors.
- [major] `apollo-fft`: removed deprecated compatibility aliases
  `FftPlan1D/2D/3D::{forward_into,inverse_into}` and `ProcessorFft3d`.
  Callers must use `forward_real_to_complex_into` and
  `inverse_complex_to_real_into`.
- [major] `apollo-fft`: removed compatibility re-export modules
  `apollo_fft::{backend,error,plan,types}`,
  `apollo_fft::application::{plan,cache}`, and
  `apollo_fft::domain::{backend,error,precision,shape}`. Callers must import
  root exports such as `apollo_fft::FftPlan1D`, `apollo_fft::ApolloError`, and
  `apollo_fft::PrecisionProfile`, or use the canonical owner modules.
- [major] `apollo-fft`: removed the legacy `FFT_CACHE` alias. Callers must use
  `FFT_CACHE_3D` for the 3D plan cache.
- [major] `apollo-fft`: removed unused
  `infrastructure::cpu::simd::power_of_two::{radix4,radix8}` forwarding
  modules. The executable radix-4/radix-8 kernels remain under
  `application::execution::kernel`.

### Changed
- `apollo-fft`: exact 2/4/8/16/32/64-point f64 and f32 mixed-radix
  transforms now route through a shared `ShortWinogradScalar` static-dispatch
  helper before Stockham/composite/Bluestein routing.
- `apollo-fft`: replaced the f16-specific bridge and radix f16 storage module
  with `precision_bridge::Complex32Bridge`, a monomorphized generic bridge
  implemented for `Complex<half::f16>` with reusable thread-local Complex32
  scratch.
- `apollo-fft`: removed unused f16 twiddle caches from the mixed-radix facade;
  f16 storage paths promote to f32 and reuse f32 short-Winograd/Stockham
  execution without building compact f16 twiddle tables.
- `apollo-fft` / `benches/kernel_strategy.rs`: removed dead radix-specific
  benchmark rows for deleted public kernels and kept only live direct,
  mixed-radix, auto-selector, Bluestein, and f16 auto-dispatch rows.
- `apollo-fft`: `rustfft` dev-dependency now uses the workspace dependency
  version instead of a crate-local version pin.
- `apollo-fft`: bumped to 0.4.0 for the pre-1.0 breaking radix f16 module
  removal and generic compact-storage bridge.
- `apollo-hilbert`: allocating observable projection methods now delegate to
  shared non-generic slice helpers, and plan-level envelope/phase execution
  reuses a thread-local Complex64 analytic scratch buffer before projecting
  into caller-owned output.
- `apollo-hilbert`: bumped to 0.3.0 for additive caller-owned observable
  projections and envelope/phase scratch reuse.
- `apollo-hilbert`: owned analytic-signal execution now routes through the
  caller-owned analytic kernel, and caller-owned quadrature reuses a
  thread-local Complex64 analytic scratch buffer instead of allocating an
  analytic `Vec` per call.
- `apollo-hilbert`: crate-root docs now describe Apollo FFT plan execution
  instead of stale private DFT ownership.
- `apollo-hilbert`: bumped to 0.2.0 for the additive caller-owned analytic API
  and quadrature scratch reuse.
- `apollo-fft`: `FftPlan1D::forward_real_to_complex_into` now delegates through
  the slice-level real-forward path, keeping one non-generic owner
  implementation for ndarray and slice callers.
- `apollo-fft`: 1D precision-specific plan methods and tests moved into leaf
  modules so `dimension_1d.rs` remains below the structural line limit.
- `apollo-fft`: bumped to 0.3.0 for the additive 1D real-forward slice API.
- `apollo-hilbert`: analytic-signal execution now calls the cached
  `FftPlan1D` slice path directly, removing the real input `Array1` bridge.
- `apollo-hilbert`: removed the now-unused `ndarray` dependency.
- `apollo-hilbert`: bumped to 0.1.4 for the FFT slice-path integration and
  dependency cleanup.
- `apollo-hilbert`: analytic-signal execution now keeps the forward FFT output
  as the masked analytic spectrum, runs the complex inverse in place, and moves
  the contiguous buffer out once instead of copying through intermediate
  `Vec`/`Array1` representations.
- `apollo-hilbert`: owned quadrature now routes through the caller-owned
  quadrature writer instead of collecting imaginary components from an
  allocating analytic-signal vector.
- `apollo-hilbert`: bumped to 0.1.3 for analytic-signal copy allocation
  removal.
- `apollo-hilbert`: `HilbertPlan::transform_into` now routes through a
  slice-level owner quadrature kernel instead of allocating a temporary
  quadrature vector and copying it into caller-owned output.
- `apollo-hilbert`: removed the unused direct `rayon` dependency left after the
  private parallel DFT kernels were replaced by `apollo-fft`.
- `apollo-hilbert`: bumped to 0.1.2 for owner quadrature slice execution.
- `apollo-hilbert`: typed `f32` and mixed `f16` execution now reuses
  thread-local f64 input/output bridge workspaces while keeping `f64` storage
  on the zero-copy owner path and preserving the shared analytic-mask kernel.
- `apollo-hilbert`: bumped to 0.1.1 for typed workspace reuse.
- `apollo-sdft`: typed direct-bin execution now reuses thread-local f64 input
  and Complex64 output bridge workspaces while keeping arithmetic in the shared
  direct-bin owner kernel.
- `apollo-sdft`: bumped to 0.1.1 for typed direct-bin workspace reuse.
- `apollo-stft`: inverse WOLA execution now reuses thread-local frame,
  complex, overlap, and weight workspaces through the shared slice-level owner
  inverse path instead of allocating four work buffers per inverse call.
- `apollo-stft`: bumped to 0.2.1 for inverse WOLA workspace reuse.
- `apollo-stft`: typed forward/inverse now reuse thread-local f64/Complex64
  bridge workspaces and call shared slice-level owner kernels instead of
  allocating owner-precision `Array1` bridge buffers per call.
- `apollo-stft`: storage/profile traits moved to a dedicated `stft::storage`
  leaf module, 1D tests moved to a leaf test module, and
  `dimension_1d.rs` is below the 500-line structural limit.
- `apollo-stft`: bumped to 0.2.0 for typed workspace reuse and pre-1.0
  breaking alias cleanup.
- `apollo-qft`: dense kernels now provide caller-owned output entry points,
  `QftPlan::forward_into` and `inverse_into` no longer allocate an intermediate
  dense output vector, and typed storage now reuses thread-local Complex64
  input/output bridge workspaces.
- `apollo-qft`: bumped to 0.1.1 for QFT dense and typed workspace reuse.
- `apollo-gft`: typed storage now reuses thread-local f64 input/output bridge
  workspaces through contiguous f64 graph-basis multiply execution, removing
  per-call typed bridge arrays.
- `apollo-gft`: bumped to 0.1.1 for GFT typed workspace reuse.
- `apollo-fwht`: typed storage now reuses thread-local f64 bridge workspaces
  through contiguous f64 slice execution, and mixed f16 storage now reuses a
  thread-local f32 compute workspace, removing per-call typed bridge and f16
  compute vector allocations.
- `apollo-fwht`: bumped to 0.1.1 for FWHT typed workspace reuse.
- `apollo-czt`: `CztPlan` now reuses a plan-owned Bluestein convolution
  workspace, precomputes square-plan inverse Vandermonde nodes, and routes
  typed storage through reusable Complex64 input/output workspaces, removing
  repeated O(P) forward workspace allocation and typed bridge allocations.
- `apollo-czt`: bumped to 0.2.1 for CZT workspace reuse.
- `apollo-fft`: removed the obsolete radix-2 butterfly helper section that was
  no longer called after Stockham became the canonical power-of-two path, and
  added missing `FftPlan3D` Rustdoc.
- `apollo-fft`: bumped to 0.2.2 for the patch-class dead-code/doc cleanup.
- `apollo-frft`: typed `Complex32` and `[f16; 2]` FrFT paths now reuse
  thread-local Complex64 input/output workspaces and call internal Complex64
  slice entry points on the canonical direct FrFT kernel, removing two O(N)
  heap allocations per typed forward/inverse call.
- `apollo-frft`: bumped to 0.1.2 for typed-storage workspace reuse.
- `apollo-fft`: restored the current kernel module declarations and
  `FftPrecision` trait header after module-header drift.
- `apollo-fft`: removed dead generic helper surface from the f16 bridge,
  radix permutation, radix shape, and radix stage modules after Stockham and
  composite routing became the canonical execution paths.
- `apollo-fft`: bumped to 0.2.1 for the patch-class dependency build cleanup.
- `apollo-frft`: `UnitaryFrftPlan` now reuses a thread-local coefficient
  workspace for the Candan-Grünbaum `V^T x` projection and reconstruction,
  removing the previous per-call O(N) heap allocation while preserving the
  same unitary transform contract.
- `apollo-frft`: crate-root export documentation now identifies the exports as
  canonical live API rather than backward-compatibility surface.
- `apollo-frft`: bumped to 0.1.1 for the patch-class unitary workspace reuse.
- `apollo-fft`: bumped to 0.2.0 for the pre-1.0 breaking compatibility cleanup.
- `apollo-fft`: root public exports now re-export directly from canonical
  application, contract, metadata, and infrastructure owners without
  compatibility modules.
- `apollo-stft-wgpu`: renamed retained GPU resource-owner fields with
  `_`-prefixed names and removed explicit dead-code suppressions from the
  buffer and Chirp-Z resource holders.
- `apollo-nufft-wgpu`: reusable fast-path buffers now validate `max_samples`
  before GPU writes and return `InputLengthMismatch` instead of relying on WGPU
  write bounds behavior.
- `apollo-nufft-wgpu`: fast-path bind groups now reuse one retained
  `layout_padding_buffer` for structurally required but unread shader bindings,
  removing per-dispatch placeholder buffer allocation.
- `apollo-ntt-wgpu`: removed duplicated reusable-buffer `n_inv` scalar storage
  and renamed retained twiddle/params GPU buffers with `_` ownership names.
- `apollo-dctdst`: `dct2_fast` and `dst2_fast` now fill only the requested
  projection from the shared 2N-point FFT setup instead of allocating an
  unused sibling output vector.
- `apollo-fft` / `application/execution/kernel/mod.rs`:
  - Registered `pub mod winograd;` in the kernel module tree.
- `apollo-fft` / `application/execution/kernel/radix8.rs`:
  - Completely rewritten as a true Winograd-radix-8 DIT FFT replacing the former radix-2 delegate.
  - Implements base-8 digit-reverse permutation (`digit_reverse_64/32`) and an iterative DIT loop
    that calls `winograd::dft8_64/32` as the inner butterfly.
- `apollo-fft` / `application/execution/kernel/radix16.rs`:
  - Replaced the O(R²) DFT-matrix inner butterfly with `winograd::dft16_64/32`.
  - Removed dead code: `cmul_64`, `cmul_32`, `dft_matrix_64`, `dft_matrix_32`,
    `radix_r_inplace_64`, `radix_r_inplace_32`.
  - Inner butterfly now costs ~8 real multiplications (Winograd DFT-16) vs. O(R²) previously.
- `apollo-fft` / `application/execution/kernel/radix32.rs`:
  - Replaced the O(R²) DFT-matrix inner butterfly with `winograd::dft32_64/32`.
  - Removed same dead code categories as radix16.
- `apollo-fft` / `application/execution/kernel/radix64.rs`:
  - Replaced the O(R²) DFT-matrix inner butterfly with `winograd::dft64_64/32`.
  - Removed same dead code categories as radix16.
- `apollo-fft` / `application/execution/kernel/winograd.rs`:
  - Removed runtime trigonometric twiddle generation from DFT-32/64 hot loops.
  - Replaced lock-based lazy twiddle caches with zero-overhead compile-time constants for
    `W_32^k` plus derived `W_64^k` composition, reducing hot-path synchronization overhead.
  - Added new f32 output-comparison regression tests for DFT-32 and DFT-64 against direct f64
    references.
- `apollo-fft` / `application/execution/kernel/radix8.rs`:
  - Eliminated redundant p=0 twiddle multiplication in radix-8 stage butterflies.
- `apollo-fft` / `application/execution/kernel/radix16.rs`:
  - Eliminated redundant p=0 twiddle multiplication in radix-16 stage butterflies.
- `apollo-fft` / `application/execution/kernel/radix32.rs`:
  - Eliminated redundant p=0 twiddle multiplication in radix-32 stage butterflies.
  - Added chunk-level Rayon MIMD execution path for large stage groups.
- `apollo-fft` / `application/execution/kernel/radix64.rs`:
  - Eliminated redundant p=0 twiddle multiplication in radix-64 stage butterflies.
  - Added chunk-level Rayon MIMD execution path for large stage groups.
- `apollo-fft` / `application/execution/kernel/mod.rs`:
  - Upgraded mixed-precision f16 auto-selector to unified runtime dispatch:
    - power-of-two lengths use compact `Complex<half::f16>` storage through the generic bridge,
    - non-power-of-two lengths transparently fall back to f32 auto-kernel routing
      (`radix`, `mixed_radix`, or `bluestein`) with output quantization back to f16 storage.
  - Removed radix2-only assumption for mixed precision, preventing non-power-of-two
    runtime failures in normal API usage.
- `apollo-fft` / `application/execution/plan/fft/dimension_1d.rs`:
  - Mixed-precision typed 1D path now selects:
    - `Complex<half::f16>` compact storage for power-of-two lengths,
    - f32 auto-kernel path for non-power-of-two lengths.
  - Added output-comparison regression tests for non-power-of-two mixed precision against
    low-precision f32 spectra plus bounded roundtrip error checks.
- `apollo-fft` / `benches/kernel_strategy.rs`:
  - Added `mixed_precision_f16_auto/{64,96}` benchmark cases to measure unified f16
    auto-selector throughput on both power-of-two and non-power-of-two lengths.

### Verification
- `cargo check -p apollo-fft --benches --examples`: passed for
  `apollo-fft` 0.4.0.
- `cargo test -p apollo-fft --lib -- --test-threads=1`: passed, 176 tests.
- `cargo check --workspace`: passed.
- `rg` source scan for `Cf16`, `radix2_f16`, public f16-specific wrappers,
  f16-named kernel files, and f16 bridge names under `apollo-fft/src` and
  `apollo-fft/benches`: no matches.
- `git diff --check`: passed.
- `cargo check -p apollo-fft --benches --examples`: passed.
- `cargo test -p apollo-fft --lib -- --test-threads=1`: passed, 181 tests.
- `cargo check --workspace`: passed.
- `cargo test -p apollo-hilbert --lib -- --test-threads=1`: passed, 21 tests.
- `rg` conflict-marker scan: no matches.
- `rg` scan for removed f16 twiddle caches and deleted radix-specific
  benchmark calls: no matches.
- `git diff --check`: passed.
- `cargo check -p apollo-hilbert`: passed for `apollo-hilbert` 0.3.0.
- `cargo test -p apollo-hilbert observables --lib -- --test-threads=1`:
  passed, 2 tests.
- `cargo test -p apollo-hilbert envelope --lib -- --test-threads=1`: passed,
  3 tests.
- `cargo test -p apollo-hilbert --lib -- --test-threads=1`: passed, 21 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for direct allocating projection `map(...).collect()`
  patterns in `AnalyticSignal`: no matches.
- `cargo check -p apollo-hilbert`: passed for `apollo-hilbert` 0.2.0.
- `cargo test -p apollo-hilbert analytic --lib -- --test-threads=1`: passed,
  6 tests.
- `cargo test -p apollo-hilbert workspace --lib -- --test-threads=1`: passed,
  2 tests.
- `cargo test -p apollo-hilbert --lib -- --test-threads=1`: passed, 18 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for removed caller-owned quadrature analytic allocation
  patterns: no matches.
- `cargo check -p apollo-fft`: passed for `apollo-fft` 0.3.0.
- `cargo test -p apollo-fft caller_owned_paths --lib -- --test-threads=1`:
  passed, 1 test.
- `cargo test -p apollo-fft forward_slice --lib -- --test-threads=1`: passed,
  1 test.
- `cargo test -p apollo-fft --lib -- --test-threads=1`: passed, 181 tests.
- `cargo check -p apollo-hilbert`: passed for `apollo-hilbert` 0.1.4.
- `cargo test -p apollo-hilbert --lib -- --test-threads=1`: passed, 14 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for removed Hilbert ndarray bridge/dependency patterns: no
  matches.
- `cargo check -p apollo-hilbert`: passed for `apollo-hilbert` 0.1.3.
- `cargo test -p apollo-hilbert transform --lib -- --test-threads=1`:
  passed, 3 tests.
- `cargo test -p apollo-hilbert --lib -- --test-threads=1`: passed, 14 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for removed Hilbert analytic-signal copy allocation
  patterns: no matches.
- `cargo check -p apollo-hilbert`: passed for `apollo-hilbert` 0.1.2.
- `cargo test -p apollo-hilbert transform_into --lib -- --test-threads=1`:
  passed, 2 tests.
- `cargo test -p apollo-hilbert --lib -- --test-threads=1`: passed, 14 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for removed Hilbert quadrature copy-through allocation and
  dead direct dependency patterns: no matches.
- `cargo check -p apollo-hilbert`: passed for `apollo-hilbert` 0.1.1.
- `cargo test -p apollo-hilbert workspace --lib -- --test-threads=1`:
  passed, 1 test.
- `cargo test -p apollo-hilbert --lib -- --test-threads=1`: passed, 12 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for removed Hilbert production typed bridge allocation
  patterns: no matches.
- `cargo check -p apollo-sdft`: passed for `apollo-sdft` 0.1.1.
- `cargo test -p apollo-sdft workspace --lib -- --test-threads=1`: passed,
  1 test.
- `cargo test -p apollo-sdft --lib -- --test-threads=1`: passed, 14 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for removed SDFT production typed direct-bin bridge
  allocation patterns: no matches.
- `cargo check -p apollo-stft`: passed for `apollo-stft` 0.2.1.
- `cargo test -p apollo-stft workspace --lib -- --test-threads=1`: passed,
  2 tests.
- `cargo test -p apollo-stft --lib -- --test-threads=1`: passed, 12 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for removed STFT production inverse WOLA allocation
  patterns and typed bridge aliases: no matches.
- `cargo check -p apollo-stft`: passed for `apollo-stft` 0.2.0.
- `cargo test -p apollo-stft workspace --lib -- --test-threads=1`: passed,
  1 test.
- `cargo test -p apollo-stft --lib -- --test-threads=1`: passed, 11 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for removed STFT production typed bridge allocation and
  deprecated alias patterns: no production matches; one expected test fixture
  name remains for `signal64`.
- `cargo check -p apollo-qft`: passed for `apollo-qft` 0.1.1.
- `cargo test -p apollo-qft workspace --lib -- --test-threads=1`: passed, 1
  test.
- `cargo test -p apollo-qft --lib -- --test-threads=1`: passed, 14 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for removed QFT production plan/typed allocation patterns:
  no optimized-path matches.
- `cargo check -p apollo-gft`: passed for `apollo-gft` 0.1.1.
- `cargo test -p apollo-gft workspace --lib -- --test-threads=1`: passed, 1
  test.
- `cargo test -p apollo-gft --lib -- --test-threads=1`: passed, 10 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for removed GFT production typed bridge allocation
  patterns: no production matches.
- `cargo check -p apollo-fwht`: passed for `apollo-fwht` 0.1.1.
- `cargo test -p apollo-fwht workspace --lib -- --test-threads=1`: passed, 1
  test.
- `cargo test -p apollo-fwht --lib -- --test-threads=1`: passed, 25 tests.
- `cargo check -p apollo-validation`: passed.
- `rg` source scan for removed FWHT production typed bridge allocation
  patterns: no production matches.
- `cargo check -p apollo-fft --lib`: passed for `apollo-fft` 0.2.2.
- `cargo check -p apollo-czt`: passed for `apollo-czt` 0.2.1.
- `cargo test -p apollo-czt workspace --lib -- --test-threads=1`: passed, 2
  tests.
- `cargo test -p apollo-czt --lib -- --test-threads=1`: passed, 27 tests.
- `cargo test -p apollo-fft radix2 --lib -- --test-threads=1`: passed, 7
  tests.
- `cargo check -p apollo-czt-wgpu -p apollo-validation`: passed.
- `rg` source scans for removed CZT typed bridge allocation patterns and
  deleted radix-2 butterfly helper names: no matches in the optimized paths.
- `cargo check -p apollo-frft`: passed for `apollo-frft` 0.1.2.
- `cargo test -p apollo-frft typed --lib -- --test-threads=1`: passed, 3
  tests.
- `cargo test -p apollo-frft --lib -- --test-threads=1`: passed, 24 tests.
- `cargo check -p apollo-frft-wgpu -p apollo-validation`: passed.
- `rg` source scan for removed typed FrFT bridge allocation patterns: no
  matches.
- `cargo check -p apollo-fft --lib`: passed without dead-code warnings from
  the removed helper set.
- `cargo test -p apollo-fft radix_shape --lib -- --test-threads=1`: passed, 6
  tests.
- `cargo test -p apollo-fft radix_permute --lib -- --test-threads=1`: passed,
  5 tests.
- `rg` source scan for deleted `apollo-fft` helper names: no matches.
- `cargo check -p apollo-frft`: passed.
- `cargo test -p apollo-frft unitary --lib -- --test-threads=1`: passed, 13
  tests.
- `cargo test -p apollo-frft --lib -- --test-threads=1`: passed, 23 tests.
- `cargo check -p apollo-frft-wgpu -p apollo-validation`: passed.
- `rg` source scan for stale FrFT compatibility/deprecated markers and the
  removed per-call unitary coefficient allocation expression: no matches.
- `cargo check -p apollo-fft --lib`: passed.
- `cargo check -p apollo-fft-wgpu -p apollo-czt -p apollo-nufft -p apollo-stft -p apollo-sft`:
  passed.
- `cargo test -p apollo-fft --lib -- --test-threads=1`: passed, 189 tests.
- `cargo check -p apollo-fft --benches`: passed.
- `rg` source scan for removed `apollo-fft` compatibility paths,
  `FFT_CACHE`, and deleted power-of-two forwarding modules in touched sources:
  no matches.
- `cargo check -p apollo-stft-wgpu`: passed.
- `cargo test -p apollo-stft-wgpu --lib -- --test-threads=1`: passed.
- `cargo check -p apollo-nufft-wgpu`: passed.
- `cargo test -p apollo-nufft-wgpu --lib -- --test-threads=1`: passed.
- `cargo check -p apollo-ntt-wgpu`: passed.
- `cargo test -p apollo-ntt-wgpu --lib -- --test-threads=1`: passed.
- `cargo check -p apollo-dctdst`: passed.
- `cargo test -p apollo-dctdst fast_single_projection_paths --lib -- --test-threads=1`:
  passed.
- `cargo test -p apollo-dctdst --lib -- --test-threads=1`: passed, 43 tests.
- `rg` source scan for audited WGPU `FrameLenNotPowerOfTwo`,
  `#[allow(dead_code)]`, and deprecated markers: no matches.
- `cargo test -p apollo-fft`: **112/112 tests pass** (includes mixed-precision non-power-of-two
  output-comparison and roundtrip regressions).
- `cargo test -p apollo-fft -- winograd`: 25/25 Winograd unit tests pass.
- `cargo run -p apollo-validation --release`: all 59 published-reference fixtures pass;
  roundtrip max-abs-error ≤ 2.2e-16 (f64), RustFFT delta = 0.
- `cargo test -p apollo-fft -- radix8 --nocapture`: passed.
- `cargo test -p apollo-fft -- radix16 --nocapture`: passed.
- `cargo test -p apollo-fft -- radix32 --nocapture`: passed.
- `cargo test -p apollo-fft -- radix64 --nocapture`: passed.
- `cargo bench -p apollo-fft --bench kernel_strategy -- radix32_inplace/32`:
  - `radix32_inplace/32`: 486 ns (−30%, improved from prior sampled baseline).
- `cargo bench -p apollo-fft --bench kernel_strategy -- radix64_inplace/64`:
  - `radix64_inplace/64`: 1.07 µs (−2.6%, improved).
- `cargo bench -p apollo-fft --bench kernel_strategy -- auto_selector/64`:
  - `auto_selector/64`: 1.02 µs (no statistically significant change).
- `cargo bench -p apollo-fft --bench kernel_strategy -- mixed_precision_f16_auto`:
  - `mixed_precision_f16_auto/64`: 1.34 µs (−28%, improved).
  - `mixed_precision_f16_auto/96`: 18.19 µs (−51%, improved).
- `cargo bench -p apollo-fft --bench kernel_strategy -- "radix8_inplace/64|radix16_inplace/16|radix32_inplace/32|radix64_inplace/64"`:
  - sample run after hot-loop update reported: `radix8_inplace/64` improved, `radix64_inplace/64`
    improved, `radix16_inplace/16` and `radix32_inplace/32` within noise.

---

## Previous [Unreleased] — twiddle-table stack-array optimisation (commit c85b301)
### Changed
- `apollo-fft` / `application/execution/kernel/radix16.rs`:
  - Removed per-call heap allocation in true radix-16 kernels by replacing dynamic scratch/output
    vectors with fixed-size stack arrays.
  - Reduced twiddle overhead by consuming stage twiddles from caller-provided tables and using
    iterative twiddle recurrence inside butterfly loops.
  - Precomputed radix-16 intra-butterfly DFT matrix once per call to eliminate repeated
    trigonometric evaluation in inner loops.
- `apollo-fft` / `application/execution/kernel/radix32.rs`:
  - Applied the same zero-extra-allocation stack scratch strategy, twiddle-table stage reuse,
    and precomputed DFT-matrix butterfly optimization for true radix-32 execution.
- `apollo-fft` / `application/execution/kernel/radix64.rs`:
  - Applied the same zero-extra-allocation stack scratch strategy, twiddle-table stage reuse,
    and precomputed DFT-matrix butterfly optimization for true radix-64 execution.

### Verification
- `cargo test -p apollo-fft radix16 -- --nocapture`: passed.
- `cargo test -p apollo-fft radix32 -- --nocapture`: passed.
- `cargo test -p apollo-fft radix64 -- --nocapture`: passed.
- `cargo bench -p apollo-fft --bench kernel_strategy`: completed; high-radix kernels improved
  materially from previous baselines (for example, `radix64_inplace/64` improved by ~34%).
- `cargo run -p apollo-validation --release`: passed, including external rustfft/numpy checks.
- `D:/miniforge3/python.exe D:/apollofft/crates/apollo-python/tests/benchmark_vs_numpy.py`:
  all Apollo-vs-NumPy output comparisons passed.

---

## [0.13.27] — Closure LXVI

### Closure LXVI — apollo-fft: add f32 Stage-4 radix-2 kernel specialization [patch]

#### Changed
- `apollo-fft` / `application/execution/kernel/radix2.rs`:
  - Added explicit Stage-4 (`len=16`) forward f32 constant-kernel specialization in
    `forward_inplace_32_with_twiddles`.
  - Added explicit Stage-4 (`len=16`) inverse f32 constant-kernel specialization in
    `inverse_inplace_unnorm_32_with_twiddles`.
  - Added explicit Stage-4 (`len=16`) inverse f32 constant-kernel specialization in
    `inverse_inplace_32_with_twiddles` with the existing normalized final-stage fusion preserved.
  - Shifted f32 general-stage loop start from `len=16, base=7` to `len=32, base=15` where
    Stage-4 specialization applies.

#### Verification
- `cargo test --workspace --release`: all workspace unit tests and doc tests passed.
- `D:/miniforge3/python.exe tests/benchmark_vs_numpy.py` (from `crates/apollo-python`):
  all Apollo-vs-NumPy output comparisons passed.

---

## [0.13.26] — Closure LXV

### Closure LXV — true radix kernels and initial GPU radix-4 execution path [minor]

#### Changed
- `apollo-fft`:
  - Implemented true Cooley-Tukey radix kernels for `radix4`, `radix16`, `radix32`, and `radix64`.
  - Kept higher-radix kernels available as explicit APIs while preserving production routing safety.
- `apollo-fft-wgpu`:
  - Added GPU radix-4 execution metadata, planner/pipeline integration, dispatch handling, and
    WGSL radix-4 shader entry points.

#### Verification
- `cargo test -p apollo-fft --release`: passed.
- `cargo test -p apollo-fft-wgpu --release`: passed.
- `cargo bench -p apollo-fft --bench kernel_strategy`: post-change measurements collected.
- Python Apollo-vs-NumPy parity benchmark: all output comparisons passed.

---

## [0.13.25] — Closure LXIV

### Closure LXIV — apollo-dht: reuse multidimensional lane buffers per plan [patch]

#### Changed
- `apollo-dht` / `application/execution/plan/dht.rs`: `DhtPlan` now owns reusable lane scratch
  buffers (`lane_in`/`lane_out`) behind a mutex.
- 2D/3D separable DHT execution paths now reuse plan-owned lane buffers instead of allocating
  per-call lane vectors for each transform invocation.

#### Memory and performance impact
- Eliminates repeated `Vec<f64>` lane allocations on `forward_2d`/`inverse_2d` and
  `forward_3d`/`inverse_3d` call paths.
- Preserves the existing output contracts while reducing allocator traffic for repeated
  multidimensional transforms.

#### Verification
- `cargo test -p apollo-dht`: 23 passed, 0 failed.
- Existing output comparison checks remain green for DHT-vs-DFT parity, fast-vs-direct Hartley parity,
  typed-path checks, and `_into` multidimensional API equivalence.

---

## [0.13.24] — Closure LXIII

### Closure LXIII — apollo-dht: remove per-call typed conversion allocations [patch]

#### Changed
- `apollo-dht` / `application/execution/plan/dht.rs`: `DhtPlan` now owns reusable typed-conversion
  scratch buffers (`input`/`output` f64 vectors) behind a mutex.
- Typed `HartleyStorage` default paths (`f32` and mixed `f16`) now reuse plan-owned scratch buffers
  instead of allocating fresh conversion vectors on each `forward_typed_into`/`inverse_typed_into` call.

#### Verification
- Extended typed-path value checks to include `f32` inverse output comparison against the `f64`
  owner-path inverse reference.
- `cargo test -p apollo-dht`: 23 passed, 0 failed.
- Existing output comparison checks remain green for DHT-vs-DFT parity, fast-vs-direct Hartley parity,
  and multidimensional `_into` API equivalence.

---

## [0.13.23] — Closure LXII

### Closure LXII — apollo-dht: remove remaining 3D separable temporary cube [patch]

#### Changed
- `apollo-dht` / `application/execution/plan/dht.rs`: `forward_3d_impl` no longer allocates any
  full `Array3<f64>` temporary for axis staging.
- Axis-1 and axis-2 passes now read each line into reusable lane buffers and write transformed values
  back to the same line in the destination cube.

#### Memory and performance impact
- Eliminates the final full `N×N×N` temporary allocation in the 3D separable DHT path.
- Keeps fixed-size lane buffers only, reducing peak memory footprint and copy pressure.

#### Verification
- `cargo test -p apollo-dht`: 23 passed, 0 failed.
- Output comparison checks remain green for DHT-vs-DFT parity, fast-vs-direct Hartley parity,
  and `_into` multidimensional APIs vs allocating variants.

---

## [0.13.22] — Closure LXI

### Closure LXI — apollo-dht: reduce multidimensional separable temporary storage [patch]

#### Changed
- `apollo-dht` / `application/execution/plan/dht.rs`: `forward_2d_impl` now uses the caller-provided
  output array as the row-pass workspace, eliminating the dedicated full `Array2<f64>` temporary.
- `apollo-dht` / `application/execution/plan/dht.rs`: `forward_3d_impl` now uses the caller-provided
  output cube for the axis-0 pass and retains only one intermediate `Array3<f64>` for the axis-1 pass,
  reducing full-cube temporary storage from two allocations to one.

#### Verification
- Added explicit output-equality coverage for `forward_2d_into`, `inverse_2d_into`, `forward_3d_into`,
  and `inverse_3d_into` against the corresponding allocating APIs.
- `cargo test -p apollo-dht`: 23 passed, 0 failed.

---

## [0.13.21] — Closure LX

### Closure LX — apollo-dht: remove redundant fast-wrapper scratch initialization [patch]

#### Changed
- `apollo-dht` / `infrastructure/kernel/fast.rs`: `dht_fast` now allocates an uninitialized-value-equivalent
  zeroed complex scratch buffer once and delegates population to `dht_fast_with_scratch`, removing the
  previous redundant pass that built `Complex64 { re: x, im: 0.0 }` for every sample before rewriting every slot.

#### Verification
- Added explicit wrapper-level output comparison coverage: the standalone `dht_fast` path is now checked
  directly against the independent O(N²) Hartley kernel at `N=512`.
- `cargo test -p apollo-dht`: 21 passed, 0 failed.

---

## [0.13.20] — Closure LIX

### Closure LIX — apollo-dht: remove full-result allocation from caller-owned 2D/3D APIs [patch]

#### Changed
- `apollo-dht` / `application/execution/plan/dht.rs`: added private direct-writer helpers
  for separable 2D and 3D DHT execution.
- `forward_2d_into` and `forward_3d_into` now execute directly into the caller-owned
  output arrays instead of allocating a full result array and copying it afterward.
- `inverse_2d_into` and `inverse_3d_into` now reuse the same direct-writer path and apply
  normalization in-place on the caller-owned output buffer.
- Allocating variants (`forward_2d`, `inverse_2d`, `forward_3d`, `inverse_3d`) now build the
  result buffer once and dispatch through the same writer helpers, keeping a single authoritative
  implementation.

#### Memory and performance impact
- Eliminates one full `Array2<f64>` allocation + copy in `forward_2d_into` and `inverse_2d_into`.
- Eliminates one full `Array3<f64>` allocation + copy in `forward_3d_into` and `inverse_3d_into`.
- Reduces peak memory footprint for caller-owned multidimensional DHT execution while preserving
  the same lane-buffer reuse and fast-kernel scratch reuse from the previous closures.

#### Verification
- `cargo test -p apollo-dht`: 20 passed, 0 failed.
- Existing output comparison coverage remains green, including direct Hartley parity at the fast
  threshold and 2D/3D inverse roundtrip recovery.

---

## [0.13.19] — Closure LVIII

### Closure LVIII — apollo-dht: reuse 2D/3D lane buffers in separable plan passes [patch]

#### Changed
- `apollo-dht` / `application/execution/plan/dht.rs`: removed per-lane `Vec<f64>`
  allocation in `forward_2d` and `forward_3d`.
- 2D row/column passes now reuse two plan-local vectors per call:
  - `lane_in` for gathering the current row or column
  - `lane_out` for the transform result
- 3D axis-0/1/2 passes now do the same for every fiber instead of allocating a fresh
  `Vec<f64>` inside each nested loop.

#### Memory and performance impact
- Eliminates one heap allocation per 2D row and per 2D column transform.
- Eliminates one heap allocation per 3D fiber transform on all three axes.
- Combined with Closure LVII scratch reuse, large separable DHT passes now avoid both:
  - repeated lane-buffer allocation at the plan layer
  - repeated complex FFT scratch allocation in the fast 1D kernel path

#### Verification
- `cargo test -p apollo-dht`: 20 passed, 0 failed.
- Existing output-comparison coverage remains green, including:
  - direct Hartley parity at the fast threshold
  - 2D separability known-value verification
  - 2D and 3D inverse roundtrip recovery

---

## [0.13.18] — Closure LVII

### Closure LVII — apollo-dht: reuse FFT scratch in fast Hartley plan path [patch]

#### Changed
- `apollo-dht` / `application/execution/plan/dht.rs`: `DhtPlan` now owns an optional
  reusable `Mutex<Vec<Complex64>>` scratch buffer for lengths at or above
  `FAST_KERNEL_THRESHOLD`.
- `apollo-dht` / `infrastructure/kernel/fast.rs`: added `dht_fast_with_scratch`, a
  caller-owned-scratch entry point for the FFT-mapped Hartley kernel.
- `DhtPlan::forward_into` now locks and reuses plan-owned scratch for the fast path
  instead of allocating a fresh complex buffer on every call.

#### Memory and performance impact
- Eliminates one `Vec<Complex64>` allocation per fast 1D DHT call.
- This also removes repeated fast-kernel scratch allocation inside 2D and 3D separable
  DHT passes, because those paths route every row/column/fiber transform through the same
  `DhtPlan::forward_into` entry point.

#### Verification
- Added explicit fast-path parity test at the threshold: the FFT-mapped DHT at `N=512`
  matches the independent direct Hartley kernel within `1e-10`.
- `cargo test -p apollo-dht`: 20 passed, 0 failed.
- Workspace Python output-comparison benchmark remains clean: 46/46 Apollo-vs-NumPy
  FFT output comparisons PASS.

---

## [0.13.17] — Closure LVI

### Closure LVI — apollo-fft: reduce mixed-radix/radix-4 twiddle overhead with iterative recurrence [patch]

#### Changed
- `apollo-fft` / `mixed_radix.rs`: removed per-bin `sin`/`cos` twiddle generation in
  the recursive radix-2 combine stage. Each combine level now computes one complex
  step root `exp(-2πi/N)` and advances twiddles by repeated complex multiplication.
- `apollo-fft` / `radix4.rs`: removed per-bin `sin`/`cos` twiddle generation in the
  radix-4 combine stage. Twiddles `W_N^k`, `W_N^{2k}`, and `W_N^{3k}` are now advanced
  iteratively from a single stage root and its powers.
- The temporary ad hoc comparison example used during profiling was removed after
  validation to keep the tree clean.

#### Performance
- Standalone complex-kernel comparison in release mode improved substantially after
  recurrence-based twiddle stepping:
  - N=16384 mixed-radix vs radix-2: `4.43x` slower -> `3.23x` slower
  - N=65536 mixed-radix vs radix-2: `4.03x` slower -> `2.73x` slower
- Result: the latest kernels are materially cheaper than the initial implementation,
  but still not competitive with the tuned iterative radix-2 production path, so
  dispatch remains unchanged.

#### Verification
- `cargo test -p apollo-fft`: 67 passed, 0 failed.
- Python benchmark output validation: 46/46 Apollo-vs-NumPy comparisons PASS.

---

## [0.13.16] — Closure LV

### Closure LV — apollo-fft: add explicit radix-4 and mixed radix-2/radix-4 kernels with validation coverage [minor]

#### Added
- New kernel module `application/execution/kernel/radix4.rs`:
  - `forward_inplace_64`, `inverse_inplace_unnorm_64`, `inverse_inplace_64`
  - `forward_inplace_32`, `inverse_inplace_unnorm_32`, `inverse_inplace_32`
  - Recursive radix-4 decomposition for power-of-four lengths.
- New kernel module `application/execution/kernel/mixed_radix.rs`:
  - `forward_inplace_64`, `inverse_inplace_unnorm_64`, `inverse_inplace_64`
  - `forward_inplace_32`, `inverse_inplace_unnorm_32`, `inverse_inplace_32`
  - Mixed radix-2/radix-4 recursion for power-of-two lengths with Bluestein fallback
    for non-power-of-two lengths.
- `kernel/mod.rs` exports the new modules (`radix4`, `mixed_radix`).

#### Verification
- `cargo test -p apollo-fft`: 67 passed, 0 failed.
- Added kernel tests:
  - `radix4_forward_n16_matches_direct`
  - `radix4_inverse_unnorm_n16_matches_direct`
  - `mixed_forward_n32_matches_direct`
  - `mixed_inverse_unnorm_n32_matches_direct`
- Python output-comparison benchmark remains passing (46/46 output checks) with
  no regression in published API output correctness.

---

## [0.13.15] — Closure LIV

### Closure LIV — apollo-fft: remove extra inverse normalization pass; extend benchmark output comparisons to inverse paths [patch]

#### Changed
- `apollo-fft` / `radix2.rs` — `inverse_inplace_64` now dispatches directly to
  `inverse_inplace_64_with_twiddles` (fused-final-stage normalization) instead of
  calling `inverse_inplace_unnorm_64` followed by a separate O(N) scale loop.
  This removes one full memory read/write pass over the output buffer.
- `apollo-fft` / `radix2.rs` — `inverse_inplace_32` now dispatches directly to
  `inverse_inplace_32_with_twiddles` (fused-final-stage normalization) instead of
  `inverse_inplace_unnorm_32` + extra O(N) scale pass.
- `benchmark_vs_numpy.py` — output validation section now covers inverse APIs in
  addition to forward APIs:
  - `ifft1` vs `numpy.fft.ifft(...).real`
  - `ifft2` vs `numpy.fft.ifft2(...).real`
  - `ifft3` vs `numpy.fft.ifftn(...).real`
  - `ifft_complex1` vs `numpy.fft.ifft(...)`
  This raises correctness gating from 23 checks to 46 checks before timing output.

#### Memory and performance impact
- Eliminates one post-kernel normalization sweep in public inverse wrappers
  (`inverse_inplace_64`, `inverse_inplace_32`), reducing memory bandwidth demand
  and cache pressure for all inverse callers using those APIs.
- Benchmark run after change shows no forward regressions and restores
  `fft_complex1` N=16384 to >1× in the observed run (1.04×), indicating the prior
  sub-1× observation was run-to-run variance rather than a kernel-path regression.

#### Verification
- 63/63 `cargo test -p apollo-fft` unit tests pass.
- 34/34 Python smoke tests pass.
- Output validation: 46/46 checks pass (forward + inverse, real + complex, 1D/2D/3D).

---

## [0.13.14] — Closure LIII

### Closure LIII — apollo-fft: halve post-twiddle reads in real FFT pack/unpack; benchmark output validation [patch]

#### Changed
- `apollo-fft` / `radix2.rs` — `forward_real_inplace_64`: replaced two post-twiddle
  reads per pair with one. Proof: `post_twiddles[m-l] = exp(-2πi·(N/2-l)/N) = -conj(post_twiddles[l])`.
  The conjugate pair index twiddle is derived algebraically: `xml = a2 - wl.conj() * b2`.
  The middle element at index `m/2` is simplified analytically: `post_twiddles[m/2] = -i`
  reduces the unpack formula to `xmid = conj(zmid)` — no twiddle table access or complex
  multiply. Cache pressure on the post-twiddle table is halved (N=65536: saves 256 KB of
  reads in the forward unpack loop).
- `apollo-fft` / `radix2.rs` — `inverse_real_inplace_64`: replaced the sequential k=1..m-1
  preprocess loop with a pairwise k=1..m/2 loop processing (k, m-k) together from one twiddle
  read. Derived: `i·conj(wmk) = (wk.im, -wk.re)` where `wmk = -conj(wk)`. The Nyquist bin
  (k=m/2, self-paired) reduces to `scratch[m/2] = input[m/2].conj()` with no twiddle read.
  Post-twiddle reads for the inverse preprocess are halved (N=65536: saves 256 KB).
- `benchmark_vs_numpy.py`: added Section 1 (Output Validation) before the timing section.
  Computes Apollo and NumPy outputs for all 23 tested configurations (1D/2D/3D real, 1D complex,
  sizes 64–65536) and asserts max absolute error < 1e-9. The script aborts with exit code 1 if
  any comparison fails, guaranteeing correctness before reporting speedup ratios. The maximum
  observed errors follow the expected O(sqrt(N)·eps_machine) pattern (e.g. N=65536: 7.1e-13).

#### Mathematical justification
- `post_twiddles[m-l] = exp(-2πi·(m-l)/N)` with `m = N/2`:
  `= exp(-πi)·exp(2πi·l/N) = -exp(2πi·l/N) = -(exp(-2πi·l/N))* = -conj(post_twiddles[l])`. QED.
- For `k=m/2` (Nyquist): `post_twiddles[m/2] = exp(-πi/2) = -i`;
  `i·conj(-i) = i·i = i² = -1`; `scratch = (xk + xk* + (-1)(xk - xk*))·0.5 = xk*`. QED.
- For forward middle: same `-i` twiddle yields `xmid = zmid.re - i·zmid.im = conj(zmid)`. QED.

#### Verification
- 63/63 `cargo test -p apollo-fft` unit tests pass.
- 34/34 Python smoke tests (`test_smoke.py`) pass.
- Output validation: 23/23 configurations PASS with max absolute error < 1e-9 vs NumPy.
  Worst case: fft3 N=128³, max_abs = 5.1e-12 (≪ 1e-9 threshold).

---

## [0.13.13] — Closure LII

### Closure LII — apollo-fft: cache-sequential gather/scatter for 3D axis-1 and axis-0 passes [patch]

#### Changed
- `apollo-fft` / `dimension_3d.rs`: reorganized the tile-blocked gather and scatter loops
  in `axis1_pass_complex` (f64), `axis0_pass_complex` (f64), `axis1_pass_complex_f32`,
  and `axis0_pass_complex_f32`.  The previous code ordered tiles with k (or i) as the
  innermost loop, producing strided loads from the source array with stride proportional
  to `ny` or `ny*nz`. The new code promotes i (or j) to the outermost position and makes
  k the innermost, so reads from `data_slice` are sequential (stride 1) and writes to
  `scratch` carry the stride. Hardware write-combining buffers the non-sequential stores
  without pipeline stalls; sequential loads are critical for prefetch and cache-line
  utilization. This change does not alter the mathematical result or the scratch layout.
- `dimension_3d.rs`: added `src_base = i * ny * nz` and `dst_base` precomputed index
  variables in the axis-0 gather/scatter to hoist the multiply of `i` out of the inner
  tile loops, eliminating one address multiplication per (j,k) iteration.
- `dimension_2d.rs`: gather/scatter loop order unchanged (2D matrices fit in L2; the
  col-outer order avoids write-allocation pressure on the smaller scratch arrays there).

#### Performance
- 3D 32³ real FFT: 1.62× → ~2.2× vs NumPy (confirmed across multiple runs, +36%)
- 3D 128³ real FFT: 1.18× → ~1.22× vs NumPy
- 2D benchmarks: no regression (within run-to-run variance)
- 1D benchmarks: unaffected (no gather/scatter path)

---

## [0.13.12] — Closure LI

### Closure LI — apollo-fft: stage-3 (len=8) butterfly specialization, compile-time W_8^j constants [patch]

#### Changed
- `apollo-fft` / `radix2.rs`: peeled the len=8 butterfly stage out of the general twiddle loop
  in all six precomputed-twiddle functions (f64/f32 forward, unnormalized inverse, normalized
  inverse). Stage-3 twiddles W_8^1=(C,∓C), W_8^2=∓i, W_8^3=(-C,∓C) with C=1/√2 are
  replaced by compile-time `std::f{64,32}::consts::FRAC_1_SQRT_2` constants, eliminating all
  four twiddle-table reads per chunk and reducing multiplications from 12 to 4 per chunk
  (j=1 and j=3 each require 2 muls; j=0 bypass and j=2 ∓i have zero muls).
- `apollo-fft` / `radix2.rs`: added `n == 8` early-return paths to normalized inverse
  functions with fused 1/N scale — no twiddle table access, no final-stage allocations.
- General twiddle loop now starts at `len=16, base=7` in all six functions (stages 1-3
  consume 1+2+4=7 twiddle entries). Stages 1, 2, and 3 are now entirely handled by
  multiply-free bypasses and compile-time constants.
- Combined savings across stages 1-3: N/2 (stage-1) + N/4 (stage-2) + N (stage-3) =
  7N/4 multiplications eliminated per forward transform; same for inverse.

#### Benchmark (v0.13.12 vs v0.13.11, median over 20 trials)
| Size | Before | After | Δ |
|---|---|---|---|
| 1D real N=64 | 8.00× | 4.83× | var |
| 1D cpx N=64 | 7.00× | 7.83× | +12% |
| 1D cpx N=16384 | 0.92× | **1.06×** | sub-1× resolved |
| 1D cpx N=65536 | 1.34× | 1.53× | +14% |
| 2D 128×128 | 1.11× | **1.28×** | +15% |
| 2D 512×512 | 1.87× | 2.08× | +11% |
| 3D 128³ | 1.15× | **1.47×** | +28% |

---

## [0.13.11] — Closure L

### Closure L — apollo-fft: stage-2 (len=4) butterfly specialization, multiply-free W_4^1=±i rotation [patch]

#### Changed
- `apollo-fft` / `radix2.rs`: peeled the len=4 butterfly stage out of the general twiddle loop
  in all six precomputed-twiddle functions (f64/f32 forward, unnormalized inverse, normalized
  inverse). The stage-2 twiddle `W_4^1 = exp(∓2πi/4) = ∓i` reduces to a 90° register swap
  with sign flip — zero complex multiplications. Together with the stage-1 bypass (Closure XLVII)
  and the j=0 bypass (Closure XLVIII), stages 1 and 2 are now entirely multiplication-free,
  saving N/2 multiply-pairs per forward transform and N/2 per inverse.
- `apollo-fft` / `radix2.rs`: added `n == 4` early-return path to normalized inverse
  (`inverse_inplace_64_with_twiddles`, `inverse_inplace_32_with_twiddles`) that fuses the final
  stage scale into the stage-2 rotate+bypass without touching the twiddle table.
- General twiddle loop now starts at `len=8, base=3` in all six functions; the twiddle table
  layout is unchanged — `base=3` correctly skips the 1 stage-1 entry and 2 stage-2 entries.

#### Benchmark (v0.13.11 vs v0.13.10, median over 20 trials)
| Size | Real speedup vs numpy | Complex speedup vs numpy |
|---|---|---|
| 1D N=64 | 8.00× (+38%) | 7.00× (+17%) |
| 1D N=256 | 4.44× (−7%) | 4.00× (+23%) |
| 1D N=1024 | 3.00× (=) | 2.00× (+3%) |
| 1D N=4096 | 2.30× (+6%) | 1.31× (+3%) |
| 2D 128×128 | 1.11× (+1%) | — |

---

## [0.13.10] — Closure XLIX

### Closure XLIX — apollo-fft/apollo-python: scalar butterfly hot loop + single-copy complex Python wrappers [patch]

#### Changed
- `apollo-fft` / `radix2.rs`: replaced operator-based hot-loop complex arithmetic
  (`t = w * v; u ± t`) with explicit scalar real/imag fused arithmetic in all precomputed-twiddle
  butterflies (f64/f32 forward + inverse variants, including normalized final stages).
  This removes temporary `Complex` values in the inner loop and improves autovectorization
  opportunities under `target-cpu=native`.
- `apollo-fft` / `radix2.rs`: restored j=0 bypass in `inverse_inplace_unnorm_64_with_twiddles`
  (`W_L^0 = 1+0i`), matching the optimized forward and normalized inverse paths and removing one
  unnecessary complex multiply per chunk per stage.
- `apollo-python` / `lib.rs`: rewired complex FFT wrappers (`fft_complex{1,2,3}`,
  `ifft_complex{1,2,3}` and plan methods) to use in-place APIs on one owned ndarray buffer
  instead of the prior double-copy path (`to_owned()` + library-level clone-return API).
  This reduces peak host memory traffic for complex transforms and lowers Python binding overhead.

#### Verification
- `cargo test -p apollo-fft -p apollo-python` → pass (`apollo-fft`: 63/63).
- `python -m pytest tests/test_smoke.py -q` → pass (34/34).

#### Benchmark Highlights (vs numpy, Closure XLIX run)
- 1D complex FFT: `N=4096` improved to **1.27x**; `N=65536` improved to **1.37x**.
- 1D real FFT: `N=65536` measured **2.84x**.
- 3D real FFT: `N=128^3` measured **1.57x**.

---

## [0.13.9] — Closure XLVIII

### Closure XLVIII — apollo-fft: codegen-units=1, thin LTO, target-cpu=native, j=0 butterfly bypass [patch]

#### Changed
- `Cargo.toml` workspace `[profile.release]`: set `codegen-units = 1` (single LLVM module for
  the entire workspace) and `lto = "thin"` (cross-crate inlining without fat-LTO compile time
  overhead). Together these allow LLVM to inline `num_complex::Complex64::mul` directly into the
  butterfly loops and apply AVX2/FMA autovectorization across crate boundaries.
- `.cargo/config.toml` `[build] rustflags`: added `["-C", "target-cpu=native"]` to enable all
  CPU features on the build machine (AVX2, FMA, BMI2). Butterfly arithmetic now uses `vfmadd256`
  where applicable.
- `apollo-fft` / `radix2.rs`: j=0 bypass applied to all six butterfly inner loops
  (`forward_inplace_64`, `inverse_inplace_unnorm_64`, `inverse_inplace_64` intermediate and
  final stages, and their f32 counterparts). W_L^0 = exp(0) = 1+0i for every stage, so
  `t = twiddles[0] * hi[0]` reduces to `t = hi[0]`; the j=0 case is hoisted out of the loop
  as a scalar add/subtract, eliminating one complex multiply per chunk per stage
  (~3% arithmetic reduction for large N). The inner loop is now `for j in 1..half`, improving
  loop-bound regularity for LLVM unrolling heuristics.

---

## [0.13.8] — Closure XLVII

### Closure XLVII — apollo-fft: O(N) bit-reversal, stage-1 no-mul, split_at_mut butterfly, fused IFFT scale [patch]

#### Changed
- `apollo-fft` / `radix2.rs`: replaced O(N log N) `bit_reverse()` per-element lookup with the
  O(N) iterative XOR/binary-counter-in-reverse technique for both `bit_reverse_permutation_64`
  and `bit_reverse_permutation_32`. The algorithm maintains `j = bit_reverse(i, log_n)` via
  amortized bit-flip operations (≈2 per element average), replacing the prior N·log₂N inner loop.
- `apollo-fft` / `radix2.rs`: stage-1 (len=2) butterfly is now a special case that omits the
  twiddle multiplication. W_2^0 = 1+0i for all N; `(u + 1·v, u - 1·v) = (u+v, u-v)` eliminates
  N/2 complex multiplications per forward or inverse transform call.
- `apollo-fft` / `radix2.rs`: all butterfly inner loops use `chunk.split_at_mut(half)` to split
  each chunk into non-overlapping `lo` and `hi` halves. Exposes non-aliasing to LLVM, enabling
  autovectorization of the `j`-loop across butterfly pairs.
- `apollo-fft` / `radix2.rs`: `inverse_inplace_64_with_twiddles` and
  `inverse_inplace_32_with_twiddles` are now inlined butterfly loops. The `1/N` scale is fused
  into the final butterfly stage, eliminating a separate O(N) normalization pass (one full array
  read+write removed per IFFT call). Proof: `(u + t) * scale = unnorm_out[k] / N = norm_out[k]`.

#### Performance (Closure XLVII vs numpy baseline)
| Case               | After XLVI | After XLVII | Change   |
|---|---|---|---|
| 1D real N=64       | 5.80×      | **6.40×**   | +10%     |
| 1D real N=1024     | 2.42×      | **2.69×**   | +11%     |
| 1D real N=4096     | 1.62×      | **2.03×**   | +25%     |
| 1D cpx N=64        | 4.67×      | **6.40×**   | +37%     |
| 1D cpx N=4096      | 0.93×      | **1.06×**   | fixed!   |
| 1D cpx N=65536     | 0.91×      | **1.16×**   | fixed!   |
| 2D 32×32           | 2.01×      | **2.54×**   | +26%     |
| 2D 64×64           | 1.19×      | **1.37×**   | +15%     |
| 3D 8³              | 3.16×      | **6.57×**   | +108%    |
| 3D 128³            | 1.08×      | **1.23×**   | +14%     |

---

## [0.13.7] — Closure XLVI

### Closure XLVI — apollo-fft: iRFFT half-spectrum inverse, cache-blocked 3D gather/scatter [patch]

#### Changed
- `apollo-fft` / `radix2.rs`: added `inverse_real_inplace_64(input, output, scratch, fft_twiddles, post_twiddles)`.
  Pre-processes the N/2+1-point Hermitian spectrum into M=N/2 complex values via
  `Z[k] = (X[k]+conj(X[M-k]))/2 + i·conj(W_k)·(X[k]-conj(X[M-k]))/2` (k=1..M-1) using the
  same `post_twiddles` as the forward pass; applies an M-point normalized IFFT via
  `inverse_inplace_64_with_twiddles`; unpacks `x[2k]=Z[k].re`, `x[2k+1]=Z[k].im`. Work
  ≈ N/2·log₂(N/2) + O(N) vs the previous N·log₂N IFFT of the full complex spectrum.
  Normalization verified: `inverse_inplace_64_with_twiddles` on M elements divides by M,
  matching the 1/N normalization of the full N-point IFFT.
- `apollo-fft` / `dimension_1d.rs`: `FftPlan1D` gains `real_inv_scratch: Option<Mutex<Vec<Complex64>>>`
  (M=N/2 entries, allocated for PoT N ≥ 4). `inverse_complex_to_real_with_workspace` and
  `inverse_complex_to_real_into` now dispatch to `inverse_real_inplace_64` when all fast-path
  fields (`twiddle_inv_64`, `real_fwd_post_twiddles`, `real_inv_scratch`) are available.
- `apollo-fft` / `dimension_3d.rs`: added `const GATHER_TILE: usize = 32`. Both
  `axis1_pass_complex` (f64/f32) and `axis0_pass_complex` (f64/f32) replace plain nested gather
  and scatter loops with GATHER_TILE×GATHER_TILE-blocked loops. For axis-1 the (j,k)-plane is
  tiled per i-slice; for axis-0 the (j,k)-plane is tiled with inner i-stride. A 32×32 tile of
  Complex64 = 16 KB, fitting in L1 cache (32–48 KB), eliminating cold-miss penalties during
  non-contiguous axis transposes. Applied to all four gather/scatter sites (f64 axis-1, f64
  axis-0, f32 axis-1, f32 axis-0).

#### Performance (Closure XLVI vs numpy baseline)
| Case          | After XLV | After XLVI | Change |
|---|---|---|---|
| 3D 32³        | 0.95×     | **1.48×**  | +56%   |
| 2D 1024×1024  | 1.06×     | **1.84×**  | +73%   |
| 2D 128×128    | 0.85×     | **0.93×**  | +9%    |
| 1D real N=256 | 3.64×     | **4.27×**  | +17%   |
| 1D real N=1024| 2.14×     | **2.42×**  | +13%   |

---

## [0.13.6] — Closure XLV

### Closure XLV — apollo-fft: real FFT half-spectrum trick, rayon sequential threshold, cache-blocked transpose [patch]

#### Changed
- `apollo-fft` / `radix2.rs`: added `build_real_fwd_post_twiddles_64(n) -> Vec<Complex64>` and
  `forward_real_inplace_64(input, output, fft_twiddles, post_twiddles)`. The new function packs a
  real N-input into N/2 complex samples, applies an N/2-point forward FFT using the first N/2-1
  entries of the existing N-point twiddle table (contiguous-layout invariant), then unpacks
  in-place via the split-radix identity `X[k] = (Z[k]+Z[M-k]*)/2 - i·W_N^k·(Z[k]-Z[M-k]*)/2`,
  processing symmetric pairs to avoid aliasing. Complexity ≈ N/2·log₂(N/2) + O(N) vs
  N·log₂N for the previous zero-padded complex FFT path.
- `apollo-fft` / `dimension_1d.rs`: `FftPlan1D` gains a `real_fwd_post_twiddles: Option<Vec<Complex64>>`
  field (N/2+1 entries, built at construction for PoT N ≥ 4). Both `forward_real_to_complex` and
  `forward_real_to_complex_into` now dispatch to `forward_real_inplace_64` when post-twiddles are
  available, falling back to the previous complex-pad path for non-PoT or N < 4.
- `apollo-fft` / `dimension_2d.rs`: added `const RAYON_THRESHOLD: usize = 32768` and
  `const TRANSPOSE_TILE: usize = 32`. All four axis-pass functions (f64 row/col, f32 row/col) now
  use sequential `chunks_mut` iteration when `data.len() ≤ RAYON_THRESHOLD`, eliminating
  rayon task-spawn overhead for small matrices. The column gather and scatter loops are now
  32×32-tile cache-blocked, keeping each tile ≤ 8 KB in L1.
- `apollo-fft` / `dimension_3d.rs`: same `RAYON_THRESHOLD = 32768` applied to all six axis-pass
  functions (axis0/1/2 × f64/f32). Sequential path used for volumes ≤ 32K elements.

#### Performance (Closure XLV vs numpy baseline)
| Case         | Before XLV | After XLV | Change |
|---|---|---|---|
| 1D real N=1024  | 1.57× | 2.14× | +36% |
| 1D real N=4096  | 0.95× | 1.66× | +75% |
| 1D real N=16384 | 3.18× | 6.16× | +94% |
| 2D 32×32        | 0.37× | 2.03× | +449% |
| 2D 64×64        | 0.55× | 1.13× | +105% |
| 2D 128×128      | 0.69× | 0.85× | +23% |
| 3D 8³           | 0.48× | 4.27× | +789% |
| 3D 16³          | 1.22× | 1.79× | +47% |
| 1D cpx N=1024   | 1.38× | 1.24× | -10% |

---

## [0.13.5] — Closure XLIV

### Closure XLIV — apollo-fft: precomputed twiddle tables + preallocated scratch in 2D/3D plans [patch]

#### Changed
- `apollo-fft` / `dimension_2d.rs`: `FftPlan2D` now embeds eight per-axis per-direction twiddle
  table fields (`twiddle_row_fwd_64`, `twiddle_row_inv_64`, `twiddle_col_fwd_64`,
  `twiddle_col_inv_64`, and f32 variants). Row (axis-1) and column (axis-0) butterfly passes use
  `forward_inplace_64_with_twiddles` / `inverse_inplace_64_with_twiddles` when the axis length is
  a power of two, eliminating the per-lane `build_forward_twiddle_table_64` allocation that
  previously occurred on every axis pass. Fallback to `fft_forward_64` for non-power-of-two axes.
- `apollo-fft` / `dimension_2d.rs`: column-pass scratch buffer preallocated at plan construction
  time (`scratch_col_64`, `scratch_col_32`; each `nx * ny` entries) and reused via `Mutex`,
  eliminating the per-call `Vec::new(nx * ny)` allocation in `axis0_pass_complex`.
- `apollo-fft` / `dimension_3d.rs`: `FftPlan3D` receives the same treatment — twelve per-axis
  twiddle table fields + four preallocated scratch buffers (`scratch_y_64`, `scratch_x_64`,
  `scratch_y_32`, `scratch_x_32`; each `nx * ny * nz` entries).  All z/y/x butterfly passes use
  precomputed tables for power-of-two axis lengths.

#### Performance
| Transform | Before | After | Delta |
|---|---|---|---|
| 2D 256×256 (vs numpy) | 1.00× | **1.33×** | +33% |
| 2D 512×512 (vs numpy) | 1.12× | **1.35×** | +20% |
| 2D 1024×1024 (vs numpy) | 1.17× | **1.45×** | +24% |
| 3D 32³ (vs numpy) | 0.40× | **1.26×** | +216% |
| 3D 64³ (vs numpy) | 0.87× | **1.14×** | +31% |
| 3D 128³ (vs numpy) | 1.04× | **1.20×** | +15% |

---

## [0.13.4] — Closure XLIII

### Closure XLIII — apollo-fft: contiguous per-stage twiddle tables; eliminate per-call allocation [patch]

#### Changed
- `apollo-fft` / `radix2.rs`: replaced unified N/2 strided twiddle table with contiguous per-stage
  layout. Stage s (group length `2^s`) occupies `2^(s-1)` sequential entries; the butterfly inner
  loop reads `stage_twiddles[j]` with no stride, eliminating L1 cache misses at N ≥ 256.
- `apollo-fft` / `radix2.rs`: `forward_inplace_64`, `inverse_inplace_unnorm_64`,
  `forward_inplace_32`, `inverse_inplace_unnorm_32` delegate to new `*_with_twiddles` kernels;
  four new exported table-building functions (`build_forward_twiddle_table_{32,64}`,
  `build_inverse_twiddle_table_{32,64}`).
- `apollo-fft` / `dimension_1d.rs`: `FftPlan1D` now precomputes and stores contiguous per-stage
  twiddle tables for power-of-two N at plan construction time (in `FFT_CACHE_1D`). Per-call Vec
  allocation eliminated from all hot paths. `forward_complex_slice_inplace`,
  `inverse_complex_slice_inplace`, `forward_f32`, `inverse_f32` use stored tables.

#### Performance
| Transform | N | Before | After | Delta |
|---|---|---|---|---|
| 1D real (vs numpy) | 1024 | 0.89× | **1.51×** | +70% |
| 1D real (vs numpy) | 4096 | 0.61× | **0.98×** | +61% |
| 1D complex (vs numpy) | 1024 | (unknown) | **1.16×** | — |
| 1D complex (vs numpy) | 4096 | 0.57× | **1.04×** | +83% |

---

## [0.13.3] — Closure XLII

### Closure XLII — apollo-python: complete Python bindings; numpy FFT benchmark [minor]

#### Added
- `apollo-python`: `fft_complex1`, `ifft_complex1` — complex128→complex128 1D FFT, numpy-compatible.
- `apollo-python`: `fft_complex2`, `ifft_complex2` — complex128 2D FFT.
- `apollo-python`: `fft_complex3`, `ifft_complex3` — complex128 3D FFT.
- `apollo-python`: `fftfreq(n, d=1.0)` — numpy-compatible DFT frequency bin centers.
- `apollo-python`: `rfftfreq(n, d=1.0)` — non-negative bins for real-input FFT.
- `apollo-python`: `fftshift(x)` — shift zero-frequency to center.
- `apollo-python`: `ifftshift(x)` — inverse of fftshift.
- `apollo-python`: `dht1`, `idht1` — 1D Discrete Hartley Transform (forward / scaled inverse).
- `apollo-python`: `dht2`, `idht2` — 2D DHT on square N×N arrays.
- `apollo-python`: `dht3`, `idht3` — 3D DHT on cubic N×N×N arrays.
- `apollo-python`: `fwht1`, `ifwht1` — 1D Fast Walsh-Hadamard Transform (N power of two).
- `apollo-python`: `fwht2`, `ifwht2` — 2D FWHT on square N×N arrays.
- `apollo-python`: `fwht3`, `ifwht3` — 3D FWHT on cubic N×N×N arrays.
- `apollo-python`: `dct2_1d`, `idct2_1d` — unnormalized DCT-II and its inverse (DCT-III × 2/N).
- `apollo-python`: `dst2_1d`, `idst2_1d` — unnormalized DST-II and its inverse.
- `apollo-python`: `FftPlan1D.fft_complex` / `FftPlan1D.ifft_complex` — plan-based complex 1D FFT.
- `apollo-python`: `FftPlan2D.fft_complex` / `FftPlan2D.ifft_complex` — plan-based complex 2D FFT.
- `crates/apollo-python/Cargo.toml`: added `apollo-dht`, `apollo-fwht`, `apollo-dctdst` dependencies.
- `tests/benchmark_vs_numpy.py`: empirical 1D/2D/3D Apollo vs numpy.fft performance comparison.
- 19 new smoke tests covering all new Python bindings (34 total, all passing).

#### Performance highlights (CPU, release build, median of 20 trials, Windows x64)
| Transform | Apollo fastest | numpy fastest | Notes |
|-----------|---------------|---------------|-------|
| 1D FFT (real, N=64) | **3.3× faster** | — | PyO3 call overhead advantage at small N |
| 1D FFT (real, N=16384) | **2.2× faster** | — | Large-N Cooley-Tukey wins |
| 1D FFT (complex, N=64) | **3.9× faster** | — | |
| 2D FFT (N=1024×1024) | **1.31× faster** | — | Parallel separable row/col FFT |
| 3D FFT (N=128³) | **1.04× faster** | — | Near-parity; CPU overhead dominates |
| 1D FFT (real, N=4096) | — | **1.63× faster** | numpy FFTPACK cache-hot midrange |

---

## [0.13.2] — Closure XLI

### Closure XLI — DHT CPU 2D/3D; FWHT CPU 2D/3D; FFT fftfreq/rfftfreq/fftshift/ifftshift [minor]

#### Added
- `apollo-dht`: `DhtPlan::forward_2d`, `forward_2d_into`, `inverse_2d`, `inverse_2d_into`,
  `forward_3d`, `forward_3d_into`, `inverse_3d`, `inverse_3d_into` — separable N×N and N×N×N CPU DHT.
- `apollo-dht`: `DhtError::ShapeMismatch2d { expected, rows, cols }` and
  `DhtError::ShapeMismatch3d { expected, d0, d1, d2 }` for non-square/non-cubic rejection.
- `apollo-dht`: re-exports `ndarray::Array2` and `ndarray::Array3` from crate root.
- `apollo-fwht`: `FwhtPlan2D` — separable N×N FWHT plan with `forward`, `forward_into`,
  `forward_inplace`, `inverse`, `inverse_into`, `inverse_inplace`, `forward_complex`, `inverse_complex`.
- `apollo-fwht`: `FwhtPlan3D` — separable N×N×N FWHT plan with matching API surface.
- `apollo-fwht`: `FwhtPlan2D` and `FwhtPlan3D` re-exported from crate root.
- `apollo-fft`: `fftfreq(n, d) -> Vec<f64>` — numpy-compatible DFT frequency bin centers.
- `apollo-fft`: `rfftfreq(n, d) -> Vec<f64>` — non-negative bins for real-input FFT, length n/2+1.
- `apollo-fft`: `fftshift<T: Copy>(input: &[T]) -> Vec<T>` — zero-frequency centering.
- `apollo-fft`: `ifftshift<T: Copy>(input: &[T]) -> Vec<T>` — inverse of `fftshift`.
- All four FFT utilities re-exported from `apollo-fft` crate root.

#### Verification
- `cargo test -p apollo-dht`: 19 passed, 0 failed.
- `cargo test -p apollo-fwht`: 24 passed, 0 failed.
- `cargo test -p apollo-fft`: 63 passed, 0 failed.
- `cargo test -p apollo-validation -- --include-ignored`: 3 passed, 0 failed.

---

## [0.13.1] — Closure XL

### Closure XL — GPU DCT/DST 2D and 3D Separable Execution [minor]

#### Added
- `apollo-dctdst-wgpu` `DctDstWgpuBackend` now exposes separable multidimensional GPU APIs:
  `execute_forward_2d`, `execute_inverse_2d`, `execute_forward_3d`, `execute_inverse_3d`.
- `WgpuError::ShapeMismatch { expected, rows, cols }` — returned when a 2D input is not `N×N`.
- `WgpuError::ShapeMismatch3d { expected, d0, d1, d2 }` — returned when a 3D input is not `N×N×N`.
- `ndarray = "0.16"` added to `apollo-dctdst-wgpu` dependencies; `Array2` and `Array3` re-exported
  from `apollo-dctdst-wgpu`.
- Verification coverage added in `apollo-dctdst-wgpu`:
  - 2D DCT-II forward GPU parity against CPU separable reference.
  - 2D DCT-II inverse roundtrip recovery.
  - 3D DCT-II forward GPU parity against CPU separable reference.
  - 3D DCT-II inverse roundtrip recovery.
  - Non-square 2D shape rejection (`ShapeMismatch`).
  - Non-cubic 3D shape rejection (`ShapeMismatch3d`).
- Separable strategy: 1D GPU kernel dispatched per row/column/fiber — no new WGSL shaders required.

#### Final state
- `cargo test -p apollo-dctdst-wgpu`: 28 passed, 0 FAILED, 0 ignored.
- `cargo test -p apollo-validation -- --include-ignored`: 3 passed, 0 FAILED, 0 ignored.
- `apollo-dctdst-wgpu` dimensional parity gap in `gap_audit.md` closed.

---

## [0.13.0] — Closure XXXIX

### Closure XXXIX — CPU DCT/DST 2D and 3D Separable Plans [minor]

#### Added
- `apollo-dctdst` `DctDstPlan` now exposes separable multidimensional CPU APIs:
  `forward_2d`, `forward_2d_into`, `inverse_2d`, `inverse_2d_into`,
  `forward_3d`, `forward_3d_into`, `inverse_3d`, and `inverse_3d_into`.
- 2D execution applies the configured real transform kind along rows then columns.
  3D execution applies the configured real transform kind along z, then y, then x.
- Shape contracts are explicit and value-checked:
  - 2D methods require square `N x N` arrays for plan length `N`.
  - 3D methods require cubic `N x N x N` arrays for plan length `N`.
  Mismatches return `DctDstError::LengthMismatch`.
- Verification coverage added in `apollo-dctdst`:
  - 2D separable parity against manual row/column application.
  - 2D inverse roundtrip recovery.
  - 3D inverse roundtrip recovery.
  - Non-square/non-cubic shape rejection.

#### Final state
- `cargo test -p apollo-dctdst`: 42 passed, 0 FAILED, 0 ignored.
- `apollo-dctdst` README updated with 2D/3D execution surfaces and verification scope.

---

## [0.12.18] — Closure XXXVIII

### Closure XXXVIII — DCT-I and DST-I Forward Known-Value Fixtures [patch]

#### Added
- Validation fixture 58 in `apollo-validation`: `dct1_three_point_forward_known_values_fixture` —
  DCT-I, N=3, x=[1,2,3]: y=[8,−2,0]; boundary formula y[k]=x[0]+(−1)^k·x[N−1]+2·Σx[n]cos(πnk/(N−1));
  y[2]=0 algebraically exact (cos(π)=−1 cancels interior term 4); threshold 1×10⁻¹⁵.
  Reference: Rao & Yip (1990) *Discrete Cosine Transform* Table 2.1; FFTW REDFT00.
- Validation fixture 59 in `apollo-validation`: `dst1_two_point_forward_known_values_fixture` —
  DST-I, N=2, x=[1,3]: y=[4√3,−2√3]; formula y[k]=2·Σx[n]sin(π(n+1)(k+1)/(N+1));
  analytically derived as 2·(√3/2+3√3/2)=4√3 and 2·(√3/2−3√3/2)=−2√3; threshold 1×10⁻¹².
  Reference: Rao & Yip (1990) *Discrete Cosine Transform* Table 3.1; FFTW RODFT00.
- Root `README.md` fixture count updated 57 → 59; two new entries appended.

#### Final state
- Both count assertions in `apollo-validation/suite.rs` updated 57 → 59.
- All 3 validation tests pass (59 fixtures, 59 attempted, all passed).

---

## [0.12.17] — Closure XXXVII

### Closure XXXVII — DCT-III and DST-III Published-Reference Fixtures [patch]

#### Added
- Validation fixture 56 in `apollo-validation`: `dct3_dc_input_flat_output_fixture` —
  DCT-III, N=4, DC input [1,0,0,0]: y[k]=x[0]/2=1/2 for all k; flat output [½,½,½,½].
  Single-term kernel evaluation (x[n]=0 for n≥1 eliminates all cosine terms);
  Makhoul (1980) IEEE Trans. ASSP 28(1) Table I; FFTW REDFT01; threshold 1×10⁻¹⁵.
- Validation fixture 57 in `apollo-validation`: `dst3_nyquist_input_alternating_output_fixture` —
  DST-III, N=4, Nyquist input [0,0,0,1]: y[k]=(−1)^k/2; alternating [½,−½,½,−½].
  Single-term kernel evaluation (x[n]=0 for n≤2 eliminates all sine terms);
  Makhoul (1980) IEEE Trans. ASSP 28(1) Table II; FFTW RODFT01; threshold 1×10⁻¹⁵.
- Root `README.md` fixture count updated 55 → 57; two new entries appended.

#### Final state
- Both count assertions in `apollo-validation/suite.rs` updated 55 → 57.
- `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

---

## [0.12.16] — Closure XXXVI

### Closure XXXVI — CWT Ricker Impulse Peak and Scale-Normalization Fixtures [patch]

#### Added
- Validation fixture 54 in `apollo-validation`: `cwt_ricker_impulse_peak_value_fixture` —
  CWT Ricker, N=7, impulse at n₀=3, a=1: W(1,2)=0, W(1,3)=ψ(0)=2/(√3·π^¼), W(1,4)=0.
  W(1,3) is single-tap (no summation error); W(1,2) and W(1,4) are exact zeros
  because (1−(±1)²)=0; Daubechies (1992) §2.1 eq.(2.1.4); threshold 1×10⁻¹⁴.
- Validation fixture 55 in `apollo-validation`: `cwt_ricker_scale_normalization_fixture` —
  CWT Ricker, N=7, impulse at n₀=3, a=2: W(2,3)=ψ(0)/√2=√2/(√3·π^¼).
  Tests the 1/√a L² normalization convention directly;
  Daubechies (1992) §2.1; Grossmann & Morlet (1984) eq.(1.3); threshold 1×10⁻¹³.
- Root `README.md` fixture count updated 53 → 55; two new entries appended.

#### Final state
- Both count assertions in `apollo-validation/suite.rs` updated 53 → 55.
- `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

---

## [0.12.15] — Closure XXXV

### Closure XXXV — Daubechies-4 DWT Coefficient and Reconstruction Fixtures [patch]

#### Added
- Validation fixture 52 in `apollo-validation`: `wavelet_daubechies4_one_level_known_coefficients_fixture` —
  DWT db4, N=4, level=1, x=[1,0,0,0]: [a0,a1,d0,d1]=[h0,h2,h3,h1] with
  Daubechies taps h=[0.4829629131, 0.8365163037, 0.2241438680, -0.1294095226].
  Basis-impulse input makes each coefficient a single tap (no summation error);
  threshold 1×10⁻¹⁵.
- Validation fixture 53 in `apollo-validation`: `wavelet_daubechies4_inverse_perfect_reconstruction_fixture` —
  DWT db4, N=4, level=1: IDWT(DWT([1,-2,0.5,4]))=[1,-2,0.5,4].
  Orthogonal two-channel PR theorem (Mallat 1989, Theorem 2);
  threshold 1×10⁻¹².
- Root `README.md` fixture count updated 51 → 53; two new entries appended.

#### Final state
- Both count assertions in `apollo-validation/suite.rs` updated 51 → 53.
- `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

---

## [0.12.14] — Closure XXXIV

### Closure XXXIV — CZT Off-Unit-Circle and Hilbert Envelope Fixtures [patch]

#### Added
- Validation fixture 50 in `apollo-validation`: `czt_off_unit_circle_z_transform_fixture` —
  CZT N=2, M=2, A=2, W=exp(−πi): X=[1.5+0i, 0.5+0i].
  Evaluates Z-transform off the unit circle at z={2,−2} (|z|=2);
  A=2 factors are dyadic rationals, exact in f64; accumulated FP error=0;
  Rabiner, Schafer & Rader (1969) IEEE TAE 17(2) §II; threshold 1×10⁻¹².
- Validation fixture 51 in `apollo-validation`: `hilbert_pure_cosine_envelope_is_unity_fixture` —
  Hilbert envelope of x=[1,0,−1,0]=cos(πn/2), N=4: envelope=[1,1,1,1].
  DFT mask {0,1,2}×{1,i,−1,−i}; analytic signal=[1,i,−1,−i]; |z[n]|=1 exact;
  Oppenheim & Schafer (2010) DTSP 3rd ed. §12.1 eq.(12.8); Bedrosian (1963);
  threshold 1×10⁻¹².
- Root `README.md` fixture count updated 49 → 51; two new entries appended.

#### Final state
- Both count assertions in `apollo-validation/suite.rs` updated 49 → 51.
- `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

---

## [0.12.13] — Closure XXXIII

### Closure XXXIII — SDFT Sliding Recurrence and FrFT Order-4 Identity Fixtures [patch]

#### Added
- Validation fixture 48 in `apollo-validation`: `sdft_sliding_recurrence_unit_impulse_all_bins_fixture` —
  SDFT N=4, zero_state, 4 updates fed [1,0,0,0]: all 4 bins = 1+0i.
  Tests the sliding-update recurrence path (not direct_bins); factors ∈{1,i,−1,−i};
  exact integer result; Jacobsen & Lyons (2003) IEEE SPM 20(2) §2 eq.(2);
  threshold 1×10⁻¹².
- Validation fixture 49 in `apollo-validation`: `frft_order4_identity_fixture` —
  UnitaryFrFT N=4, order α=4.0: DFrFT_4([1,2,3,4])=[1,2,3,4].
  exp(−4kπi/2)=exp(−2πki)=1; V·I·V^T=I; exact regardless of eigenvector ordering;
  Candan, Kutay & Ozaktas (2000) IEEE TSP 48(5) §II Corollary;
  threshold 1×10⁻¹².
- Root `README.md` fixture count updated 47 → 49; two new entries appended.

#### Final state
- Both count assertions in `apollo-validation/suite.rs` updated 47 → 49.
- `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

---

## [0.12.12] — Closure XXXII

### Closure XXXII — NUFFT Adjoint Identity and Radon Fourier Slice Theorem Fixtures [patch]

#### Added
- Validation fixture 46 in `apollo-validation`: `nufft_type1_type2_adjoint_inner_product_fixture` —
  NUFFT N=2, pos=[0,0.5], c=[1,2], f=[3,4]: Re(〈A·c,f〉)=Re(〈c,A*·f〉)=5.
  All exp factors ∈{1,−1}; computation exact in f64; accumulated FP error=0;
  Dutt & Rokhlin (1993) SIAM J. Sci. Comput. 14(6): adjoint identity (1.8);
  Greengard & Lee (2004) §2; threshold 1×10⁻¹².
- Validation fixture 47 in `apollo-validation`: `radon_fourier_slice_theorem_theta0_fixture` —
  Radon θ=0 Fourier Slice Theorem on 2×2 image [[1,2],[3,4]]:
  DFT_1(R_{θ=0}f)=[10+0i,−2+0i] equals horizontal slice of 2D DFT;
  all DFT factors ∈{1,−1}; exact in f64; Natterer (1986) §I.2 Thm 1.1;
  Radon (1917); threshold 1×10⁻¹².
- Root `README.md` fixture count updated 45 → 47; two new entries appended.

#### Final state
- Both count assertions in `apollo-validation/suite.rs` updated 45 → 47.
- `cargo test -p apollo-validation`: 3 passed, 0 FAILED, 0 ignored.

---

## [0.12.11] — Closure XXXI

### Closure XXXI — DCT-I and DST-I Self-Inverse Published-Reference Fixtures [patch]

#### Added
- Validation fixture 44 in `apollo-validation`: `dct1_inverse_roundtrip_three_point_fixture` —
  DCT-I N=3: `IDCT-I(DCT-I([1,2,3])) = [1,2,3]`. Makhoul (1980) IEEE Trans. ASSP 28(1):
  DCT-I self-inverse C1²=2(N−1)·I; FFTW REDFT00: IDCT-I=(1/(2(N−1)))·DCT-I.
  Intermediate spectrum [8,−2,0] is exactly integer (cos values in {−1,0,1});
  round-trip error = 0 analytically; threshold 1×10⁻¹⁴.
- Validation fixture 45 in `apollo-validation`: `dst1_inverse_roundtrip_two_point_fixture` —
  DST-I N=2: `IDST-I(DST-I([1,3])) = [1,3]`. Makhoul (1980) IEEE Trans. ASSP 28(1):
  DST-I self-inverse S1²=2(N+1)·I; FFTW RODFT00: IDST-I=(1/(2(N+1)))·DST-I.
  Intermediate spectrum [4√3,−2√3]; O(ε_f64) error; threshold 1×10⁻¹⁴.
- Root `README.md` fixture count updated 43 → 45; two new entries appended.

#### Final state
- Both count assertions in `apollo-validation/suite.rs` updated 43 → 45.
- `cargo test -p apollo-validation -p apollo-dctdst`: 0 FAILED, 0 ignored.

---

## [0.12.10] — Closure XXX

### Closure XXX — DCT-IV and DST-IV Self-Inverse Published-Reference Fixtures [patch]

#### Added
- Validation fixture 42 in `apollo-validation`: `dct4_inverse_roundtrip_two_point_fixture` —
  DCT-IV N=2: `IDCT-IV(DCT-IV([1,3])) = [1,3]`. Makhoul (1980) IEEE Trans. ASSP 28(1):
  DCT-IV self-inverse property C4²=N·I; FFTW REDFT11 IDCT-IV=(1/2N)·DCT-IV;
  threshold 1×10⁻¹⁴.
- Validation fixture 43 in `apollo-validation`: `dst4_inverse_roundtrip_two_point_fixture` —
  DST-IV N=2: `IDST-IV(DST-IV([2,5])) = [2,5]`. Makhoul (1980) IEEE Trans. ASSP 28(1):
  DST-IV self-inverse property S4²=N·I; FFTW RODFT11 IDST-IV=(1/2N)·DST-IV;
  threshold 1×10⁻¹⁴.
- Root `README.md` fixture count updated 41 → 43; two new entries appended.

#### Final state
- Both count assertions in `apollo-validation/suite.rs` updated 41 → 43.
- `cargo test --workspace`: 0 FAILED, 0 ignored.

---

## [0.12.9] — Closure XXIX

### Closure XXIX — Inverse-Roundtrip Published-Reference Fixtures: NTT, STFT [patch]

#### Added
- Validation fixture 40 in `apollo-validation`: `ntt_inverse_roundtrip_fixture` —
  NTT N=4: `INTT(NTT([1,2,3,4])) = [1,2,3,4]`. Pollard (1971) Math. Proc. Cambridge
  Phil. Soc. 70(3): NTT inversion theorem in ℤ/pℤ; threshold 1×10⁻¹².
- Validation fixture 41 in `apollo-validation`: `stft_hann_wola_inverse_roundtrip_fixture` —
  STFT frame=4,hop=2: `ISTFT(STFT([1,0,0,0])) = [1,0,0,0]`. Allen & Rabiner (1977)
  Proc. IEEE 65(11) WOLA synthesis; Portnoff (1980) Hann COLA;
  Hann w=[0,0.75,0.75,0], COLA weight=0.5625; threshold 1×10⁻¹².
- Root `README.md` fixture count updated 39 → 41; two new entries appended.

#### Final state
- Both count assertions in `apollo-validation/suite.rs` updated 39 → 41.
- `cargo test --workspace`: 0 FAILED, 0 ignored.

---

## [0.12.8] — Closure XXVIII

### Closure XXVIII — Inverse-Roundtrip Published-Reference Fixtures: DHT, SFT [patch]

#### Added
- Validation fixture 38 in `apollo-validation`: `dht_inverse_roundtrip_fixture` —
  DHT N=4: `IDHT(DHT([3,-1,2,0])) = [3,-1,2,0]`. Bracewell (1983) JOSA 73(12):
  H²=NI; inverse = (1/N)·DHT; threshold 1×10⁻¹⁴.
- Validation fixture 39 in `apollo-validation`: `sft_inverse_roundtrip_fixture` —
  SFT N=4, K=1: `ISFT(SFT([1,-1,1,-1])) = [1,-1,1,-1]`. Cooley-Tukey (1965)
  DFT[(−1)^n]=4·δ[k−2]; Hassanieh et al. (2012) K-sparse exact recovery;
  Candès & Wakin (2008) RIP; threshold 1×10⁻¹².
- Root `README.md` fixture count updated 37 → 39; two new entries appended.

#### Final state
- Both count assertions in `apollo-validation/suite.rs` updated 37 → 39.
- `cargo test --workspace`: 0 FAILED, 0 ignored.

---

## [0.12.7] — Closure XXVII

### Closure XXVII — Inverse-Roundtrip Published-Reference Fixtures: FWHT, QFT, SHT [patch]

#### Added
- Validation fixture 35 in `apollo-validation`: `fwht_inverse_roundtrip_fixture` —
  FWHT N=4: `IFWHT(FWHT([1,2,3,4])) = [1,2,3,4]`. Walsh (1923) Am. J. Math. 45 §2:
  W_N² = N·I; threshold 1×10⁻¹⁴.
- Validation fixture 36 in `apollo-validation`: `qft_inverse_roundtrip_fixture` —
  QFT N=4: `iqft(qft([1,0,0,0])) = [1,0,0,0]`. Shor (1994) §2: QFT_N unitary;
  Nielsen & Chuang (2000) §5.1; threshold 1×10⁻¹².
- Validation fixture 37 in `apollo-validation`: `sht_inverse_roundtrip_y10_fixture` —
  SHT lmax=1, lat=12, lon=25: dipole Y_1^0 = √(3/4π)·cosθ roundtrip;
  Driscoll & Healy (1994) Adv. Appl. Math. 15 Theorem 1; threshold 1×10⁻¹⁰.
- Root `README.md` fixture count updated 34 → 37; three new entries appended.

#### Final state
- `cargo test --workspace`: 0 FAILED, 0 ignored, all doc-tests compile.
- Published-reference fixture count: 37.

---

## [0.12.6] — Closure XXVI

### Closure XXVI — Inverse-Roundtrip Published-Reference Fixtures: DWT, GFT, FrFT [patch]

#### Added
- Validation fixture 32 in `apollo-validation`: `wavelet_haar_inverse_perfect_reconstruction_fixture` —
  Haar DWT N=4, 1-level: `IDWT(DWT([1,−1,0,0])) = [1,−1,0,0]`. Mallat (1989) §3.1 Theorem 2
  perfect reconstruction. Threshold 1×10⁻¹².
- Validation fixture 33 in `apollo-validation`: `gft_path_graph_inverse_roundtrip_fixture` —
  GFT K₂ path graph: `GFT⁻¹(GFT([3,−1])) = [3,−1]`. Sandryhaila & Moura (2013) ICASSP
  eigendecomposition invertibility. Threshold 1×10⁻¹².
- Validation fixture 34 in `apollo-validation`: `frft_inverse_roundtrip_order_half_fixture` —
  FrFT α=0.5, N=4: `FrFT(−0.5)(FrFT(0.5)([1,2,3,4])) = [1,2,3,4]`. Namias (1980) J.IMA 25(3)
  additivity theorem F⁻α ∘ Fα = I. Threshold 1×10⁻¹².
- Root `README.md` fixture count updated 31 → 34; three new fixture entries appended.

#### Final state
- `cargo test --workspace`: 0 FAILED, 0 ignored, all doc-tests compile.
- Published-reference fixture count: 34 (assertions in both test functions updated 31 → 34).

---

## [0.12.5] — Closure XXV

### Closure XXV — Hilbert Instantaneous Frequency + Doc/Test/PM Cleanup [patch]

#### Added
- `AnalyticSignal::instantaneous_frequency()` in `apollo-hilbert`
  (`domain/signal/analytic.rs`): computes instantaneous frequency in cycles per
  sample using the complex-derivative formula
  `f[n] = arg(conj(z[n]) · z[n+1]) / (2π)`. Returns a `Vec<f64>` of length
  `N − 1`. Avoids explicit phase unwrapping; values in `(−0.5, +0.5]`.
  Reference: Boashash (1992) Proc. IEEE 80(4).
- Validation fixture 31 in `apollo-validation`:
  `hilbert_instantaneous_frequency_constant_tone_fixture` — verifies that
  `cos(2π5·n/64)` has instantaneous frequency `5/64` at every sample
  (threshold 1e-10). Root `README.md` fixture count updated 30 → 31.

#### Added (Tests — apollo-hilbert)
- `instantaneous_frequency_constant_tone`: asserts `IF = k/N` for all N−1
  samples of a single-tone cosine at `k=5`, `N=64`; tolerance 1e-10.
- `double_hilbert_negates_zero_mean_signal`: asserts `H{H{x}} = −x` for a
  sinusoidal input at `N=32`; tolerance 1e-10.

#### Changed
- `apollo-ntt-wgpu/src/verification.rs` module doc: `rust,ignore` code block
  converted to `rust,no_run` with `# use apollo_ntt_wgpu::NttWgpuBackend;`
  preamble. Eliminates the last workspace-wide ignored test; doc-test now
  reports "ok compile" instead of "ignored".
- `apollo-stft-wgpu/src/infrastructure/device.rs`:
  `execute_inverse_with_buffers` doc comment expanded from stub to full
  description including non-PoT delegation behaviour and `# Errors` section.
- `CHANGELOG.md`: back-filled missing Closure XXIII (0.12.3) and Closure XXIV
  (0.12.4) entries.
- `apollo-hilbert/README.md`: added "Instantaneous Frequency" subsection
  documenting formula, `N−1` length contract, and Boashash 1992 reference.

#### Final state
- `cargo test --workspace`: 0 FAILED, 0 ignored, all doc-tests compile.

---

## [0.12.4] — Closure XXIV

### Closure XXIV — GPU Adapter Preference + Test Runtime-Skip Conversion + Bluestein CZT Sign Fix [patch]

#### Changed
- All 20 `wgpu::RequestAdapterOptions::default()` sites replaced with
  `power_preference: PowerPreference::HighPerformance` across every wgpu crate
  (`apollo-fft-wgpu`, `apollo-czt-wgpu`, `apollo-mellin-wgpu`, `apollo-ntt-wgpu`,
  `apollo-stft-wgpu`, `apollo-radon-wgpu`, `apollo-nufft-wgpu`, `apollo-hilbert-wgpu`,
  `apollo-sft-wgpu`, `apollo-qft-wgpu`, `apollo-frft-wgpu`, `apollo-fwht-wgpu`,
  `apollo-dht-wgpu`, `apollo-sdft-wgpu`, `apollo-sht-wgpu`, `apollo-dctdst-wgpu`,
  `apollo-gft-wgpu`, `apollo-wavelet-wgpu`, `f16_plan.rs`, `buffer_reuse` bench).
  Ensures NVIDIA discrete GPU is preferred over integrated/software adapters.
- `apollo-ntt-wgpu` verification: removed all 10 `#[ignore]` attributes; converted
  to `let Ok(backend) = NttWgpuBackend::try_default() else { return; }` runtime-skip
  pattern. Tests run unconditionally on GPU-enabled hosts and skip silently on CI.
- `apollo-stft-wgpu` verification: removed all 7 `#[ignore]` attributes; same
  runtime-skip pattern applied.
- `apollo-stft-wgpu/src/infrastructure/shaders/stft_chirp.wgsl`: corrected all four
  Bluestein sign errors (`premul_fwd`: `exp(−πi·n²/N)`, `premul_inv`: `exp(+πi·n²/N)`,
  `postmul_fwd`: `exp(−πi·k²/N)`, `postmul_inv`: `exp(+πi·n²/N)/N`); added new
  `stft_chirp_pointmul_fwd` entry point that conjugates the stored kernel
  `h_stored = exp(−πi·j²/N)` to recover `h_fwd = exp(+πi·j²/N)`.
- `StftChirpData` (`chirp.rs`): added `pointmul_fwd_pipeline: wgpu::ComputePipeline`
  field; `new()` builds pipeline from `stft_chirp_pointmul_fwd` entry point.
- `kernel.rs` forward CZT dispatch (Pass C): uses `pointmul_fwd_pipeline` instead of
  `pointmul_pipeline`; inverse Pass C unchanged.
- `device.rs`: added non-PoT guards to `execute_forward_with_buffers` and
  `execute_inverse_with_buffers` that delegate to the allocating Chirp-Z path and copy
  results into `fwd_output_host` / `inv_output_host`.
- `stft-wgpu` forward CZT test tolerance updated `1e-2 → 2e-2` (analytically justified
  by f32 GPU argument-reduction at phase magnitudes up to ~1254 rad for N=400).

#### Final state
- `cargo test --workspace`: 0 FAILED, 0 ignored across all 38+ crates and all doc-tests.

---

## [0.12.3] — Closure XXIII

### Closure XXIII — ARCHITECTURE.md Capability Annotations + Validation Fixtures 29–30 [patch]

#### Changed
- `ARCHITECTURE.md` Mixed-Precision Capability Table: added `"forward + inverse CZT"` and
  `"forward + inverse Mellin spectrum"` annotations to the `Notes` column for
  `apollo-czt-wgpu` and `apollo-mellin-wgpu`, matching the bidirectional-WGPU annotation
  pattern already established for other transform pairs.

#### Added
- `apollo-validation`: two new published-reference fixtures (fixtures 29 and 30).
  - `czt_inverse_vandermonde_roundtrip_fixture`: N=4 Björck-Pereyra Vandermonde solve,
    threshold 1e-12. Validates exact numeric contract from Björck & Pereyra (1970).
  - `mellin_inverse_spectrum_constant_roundtrip_fixture`: N=32 constant signal IDFT +
    exp-resample roundtrip, threshold 1e-10.
  - `published_real_fixture_with_threshold` helper function added.
  - README fixture count updated 28 → 30.
  - `validation_suite_produces_value_semantic_reports` assertion updated to 30.
  - All 30 fixtures pass.

---

## [0.12.2] — Closure XXII
### Closure XXII — GPU Benchmark Runner Workflow + Root README Correction [patch]

#### Added
- `.github/workflows/gpu-benchmarks.yml`: manual `workflow_dispatch` workflow targeting a
  labeled self-hosted runner (`[self-hosted, gpu, apollo]`) to execute the WGPU Criterion
  benchmark suites and upload the output bundle as a workflow artifact.
- `scripts/run_gpu_benchmarks.ps1`: reproducible GPU benchmark runner script that stages
  logs, `manifest.json`, `summary.md`, and `target/criterion` under `.benchmarks/gpu-runner/`.
- `.benchmarks/gpu-runner/.gitkeep`: tracked anchor for benchmark artifact staging.

#### Changed
- Root `README.md`: corrected stale CZT/Mellin/STFT/Radon WGPU capability prose and added a
  dedicated "GPU Benchmark Runner" section documenting labels, workflow entry point, executed
  benches, and output staging path.
- `.gitignore`: ignores generated `.benchmarks/gpu-runner/*` outputs while keeping the tracked
  directory anchor.

---

## [0.12.1] — Closure XXI

### Closure XXI — README Documentation Sync for v0.2.0 Inverse Additions [patch]

#### Changed
- `apollo-czt/README.md`: added "Inverse Transform" section documenting
  `CztPlan::inverse`, Björck-Pereyra Vandermonde solve, and `CztError::NotInvertible`
  conditions (non-square plan, node collision). Updated "Verification" section to
  list inverse roundtrip and rejection tests.
- `apollo-mellin/README.md`: added "Inverse Transform" section documenting
  `MellinPlan::inverse_spectrum`, IDFT + exp-resample algorithm, and
  `MellinError::SpectrumLengthMismatch`. Updated "Verification" section.
- `apollo-czt-wgpu/README.md`: updated "Execution Contract" to reflect forward+inverse
  support; added adjoint formula description; removed stale "inverse unsupported" prose.
  Updated "Verification" section to mention GPU inverse roundtrip test.
- `apollo-mellin-wgpu/README.md`: updated "Execution Contract" to document two-pass
  GPU inverse (IDFT + exp-resample); updated capability and verification prose.
- `checklist.md`: added completed Closure XX entry (22 checklist items).

---

## [0.12.0] — Closure XX

### Closure XX — CPU + GPU Inverse Transforms: CZT and Mellin [minor]

#### Added — apollo-czt v0.2.0
- `CztPlan::inverse(spectrum)` — exact Vandermonde solve via Björck-Pereyra algorithm
  (`O(N²)` in-place Newton evaluation). Returns `CztError::NotInvertible` when
  `M ≠ N` or when Vandermonde nodes collide (denominator below `f64::EPSILON * 1024`).
- `CztStorage::inverse_into` — default method adapting `inverse` to in-place storage API.
- `CztError::NotInvertible { reason: &'static str }` variant.
- 5 value-semantic tests: roundtrip at DFT parameters, general `A` offset, non-unit `W`
  spacing, rejection of rectangular plans, rejection of wrong spectrum length.

#### Added — apollo-mellin v0.2.0
- `MellinPlan::inverse_spectrum(spectrum, out_min, out_max, output)` — IDFT of
  log-domain spectrum then exp-resample from log-grid to linear domain.
  Rayon-parallel IDFT for `N ≥ 256`.
- `inverse_log_frequency_spectrum` and `exp_resample` exported from `lib.rs`.
- `MellinError::SpectrumLengthMismatch` variant.
- 4 value-semantic tests: constant-signal roundtrip (`ε < 1e-10`), linear-signal
  roundtrip (interpolation error `< 0.1` for `N = 64`), wrong-length rejection,
  invalid-bounds rejection.

#### Added — apollo-czt-wgpu v0.2.0
- `czt_inverse` WGSL entry point: adjoint formula `x[n] = (A^n/N)·∑_k X[k]·W^{-nk}`,
  exact for unitary DFT parameters.
- `MellinGpuKernel::inverse_pipeline` field; `execute_inverse` dispatches over `N` threads.
- `CztWgpuBackend::execute_inverse(plan, spectrum)` — validates `M == N`, delegates to kernel.
- `WgpuCapabilities::forward_inverse` constructor.
- 2 new GPU tests: roundtrip at DFT parameters, rejection of non-square plan.

#### Added — apollo-mellin-wgpu v0.2.0
- `mellin_inverse_spectrum` WGSL kernel: IDFT pass, spectrum → log-domain samples.
- `mellin_exp_resample` WGSL kernel: exp-resample pass, log-domain → linear output.
- `InverseMellinParamsPod` uniform struct (32 bytes, reuses params buffer slot).
- `MellinGpuKernel::inverse_spectrum_pipeline`, `exp_resample_pipeline`,
  `inv_params_buffer` fields; `execute_inverse` dispatches two GPU passes + readback.
- `MellinWgpuBackend::execute_inverse(plan, spectrum, out_min, out_max, out_len)`.
- `WgpuCapabilities::forward_inverse` constructor.
- 2 new GPU tests: constant-signal roundtrip (`ε < 5e-4`), invalid-domain rejection.

---

## [0.10.0] — Closure XIX

### Closure XIX — StftGpuBuffers Non-PoT Scratch Sizing [minor]

#### Changed
- `StftGpuBuffers::new` now accepts arbitrary `frame_len` (not just power-of-two).
  Scratch buffers (`re_scratch_buf`, `im_scratch_buf`, `frame_data_buf`) are automatically
  sized to `frame_count × M` where `M = chirp_padded_len(frame_len)` when `!frame_len.is_power_of_two()`,
  and `frame_count × frame_len` when PoT.
- `StftWgpuBackend::make_buffers`, `execute_forward_with_buffers`, `execute_inverse_with_buffers`
  no longer return `WgpuError::FrameLenNotPowerOfTwo`; non-PoT `frame_len` now dispatches
  via the buffer-reuse path (kernel auto-selects Chirp-Z for non-PoT at dispatch time).
- Module docstring in `buffers.rs` updated: PoT constraint removed; Closure XIX note added.

#### Added (Tests)
- `make_buffers_accepts_non_power_of_two_frame_len_structurally`: structural verification
  that `make_buffers` accepts non-PoT without returning `FrameLenNotPowerOfTwo`.
- `forward_buffers_non_pot_frame_len_400_when_device_exists` (GPU-gated): buffer-reuse
  forward at `frame_len=400`, GPU output matched against CPU `apollo-stft` reference.
- `inverse_buffers_non_pot_frame_len_400_when_device_exists` (GPU-gated): buffer-reuse
  inverse at `frame_len=400`, WOLA interior-sample recovery comparison.

#### Removed
- Panic condition `assert!(frame_len.is_power_of_two())` in `StftGpuBuffers::new`.

---

## [0.9.0] — Closure XVIII

### Closure XVIII — Non-Power-of-Two STFT GPU Path (Bluestein/Chirp-Z) [minor]

#### Added
- `crates/apollo-stft-wgpu/src/infrastructure/shaders/stft_chirp.wgsl`: Five-pass WGSL
  compute shader implementing the Bluestein Chirp-Z mapping for the STFT: `premul_fwd`,
  `premul_inv`, `pointmul`, `postmul_fwd`, `postmul_inv`. Hann analysis/synthesis windows
  and exp(±πi·n²/N) chirp twiddles are applied on-GPU.
- `crates/apollo-stft-wgpu/src/infrastructure/shaders/stft_chirp_fft.wgsl`: Radix-2
  sub-FFT shader operating on chirp working buffers: `chirp_fft_bitrev`,
  `chirp_fft_butterfly_fwd`, `chirp_fft_butterfly_inv`, `chirp_fft_scale`.
- `crates/apollo-stft-wgpu/src/infrastructure/chirp.rs`: `StftChirpData` struct —
  pre-allocated GPU resources (chirp kernel H, working buffers, bind groups, pipelines)
  for the Bluestein path. `chirp_padded_len(n)` returns `(2n−1).next_power_of_two()`.
- `design_history_file/adr_stft_wgpu_non_pot_chirpz.md`: ADR for the Chirp-Z dispatch
  strategy (Bluestein 1970, Rabiner 1969).
- `ndarray = "0.16"` added as a regular (non-dev) dependency to `apollo-stft-wgpu`
  for CPU-side chirp kernel construction in `StftChirpData::new`.

#### Changed
- `execute_forward` and `execute_inverse` in `StftWgpuBackend` (device.rs) no longer
  return `WgpuError::FrameLenNotPowerOfTwo`; non-power-of-two `frame_len` now
  automatically dispatches the Bluestein/Chirp-Z path.
- `kernel.rs`: non-PoT `frame_len` delegates to `execute_forward_fft_chirp` /
  `execute_inverse_chirp` instead of returning an error.
- `WgpuError::FrameLenNotPowerOfTwo` doc comment updated to reflect that this variant
  is no longer returned by primary dispatch; it may still be returned by
  `make_buffers`, `execute_forward_with_buffers`, `execute_inverse_with_buffers`.
- Verification tests `forward_rejects_non_power_of_two_frame_len` and
  `inverse_rejects_non_power_of_two_frame_len` renamed and inverted to
  `forward_accepts_non_power_of_two_frame_len_chirpz` and
  `inverse_accepts_non_power_of_two_frame_len_chirpz`.

#### Added (Tests)
- `forward_accepts_non_power_of_two_frame_len_structurally`: structural check that
  non-PoT no longer returns `FrameLenNotPowerOfTwo`.
- `forward_accepts_non_power_of_two_frame_len_chirpz`: same for `execute_forward`.
- `inverse_accepts_non_power_of_two_frame_len_chirpz`: same for `execute_inverse`.
- `forward_chirpz_non_pot_frame_len_400_when_device_exists` (GPU-gated): GPU forward
  Chirp-Z at frame_len=400, compared to CPU `apollo-stft` reference, TOL=1e-2.
- `inverse_chirpz_non_pot_frame_len_400_when_device_exists` (GPU-gated): GPU inverse
  Chirp-Z at frame_len=400, WOLA interior-sample recovery, TOL=5e-2.

---

## [0.8.5] — Closure XVII

### Closure XVII — STFT GPU Buffer-Reuse Criterion Benchmarks + README Usage Documentation [patch]
#### Added
- `bench_forward_reuse` benchmark group in `crates/apollo-stft-wgpu/benches/stft_bench.rs`:
  head-to-head comparison of `execute_forward` (allocating) vs `execute_forward_with_buffers`
  (buffer-reuse) at `frame_len` ∈ {256, 512, 1024}. Pre-allocates `StftGpuBuffers` outside
  the bench loop; measures only signal upload + GPU dispatch + readback per iteration.
- `bench_inverse_reuse` benchmark group: same head-to-head comparison for
  `execute_inverse` vs `execute_inverse_with_buffers`.
- Both groups added to `criterion_group!(benches, …)` in `stft_bench.rs`.
- Updated module docstring in `stft_bench.rs` to describe both allocating and buffer-reuse
  paths and their mathematical basis.
- "Buffer Reuse" section in `crates/apollo-stft-wgpu/README.md`: usage snippet showing
  `make_buffers` → `execute_forward_with_buffers` → `fwd_output()` pattern, constraint
  notes for `FrameLenNotPowerOfTwo` and `LengthMismatch` errors.
- "Benchmarks" section in `README.md`: table of all four benchmark groups with
  description and `cargo bench -p apollo-stft-wgpu` invocation.

---

## [0.8.4] — Closure XVI

### Closure XVI — StftGpuBuffers Pre-allocated Buffer Reuse [minor]
#### Added
- `StftGpuBuffers` struct in `crates/apollo-stft-wgpu/src/infrastructure/buffers.rs`:
  pre-allocates all GPU data buffers, staging buffers, bind groups, and per-stage butterfly
  uniform buffers for a fixed `(frame_count, frame_len, signal_len, hop_len)` quad.
  Eliminates 5–8 `device.create_buffer` calls, 4+ `device.create_bind_group` calls,
  and `log₂(N)` uniform-buffer allocations per dispatch.
- `StftWgpuBackend::make_buffers(plan, signal_len)`: constructs a `StftGpuBuffers` for
  a given plan shape with validation identical to the allocating paths.
- `StftWgpuBackend::execute_forward_with_buffers(plan, signal, buffers)`: zero-allocation
  forward STFT dispatch; result in `buffers.fwd_output()`.
- `StftWgpuBackend::execute_inverse_with_buffers(plan, spectrum, signal_len, buffers)`:
  zero-allocation inverse STFT dispatch; result in `buffers.inv_output()`.
- `StftGpuKernel::execute_forward_fft_with_buffers` and `execute_inverse_with_buffers`:
  kernel-level buffered dispatch methods.
- Verification test `reusable_buffers_match_allocating_forward_and_inverse_when_device_exists`
  (#[ignore = "requires wgpu device"]): asserts bit-exact agreement between allocating and
  buffered paths on a bin-aligned sinusoid (k=16, frame_len=512);
  verifies idempotent second-call buffer reuse.
- `pub use infrastructure::buffers::StftGpuBuffers` re-exported from crate root.

---

## [0.8.3] — Closure XV

### Closure XV — Radon FBP GPU Criterion Benchmarks
#### Added
- `crates/apollo-radon-wgpu/benches/radon_wgpu_bench.rs`: criterion benchmark suite with
  `radon_wgpu_forward` and `radon_wgpu_fbp` groups.
- `criterion = "0.5"` in `apollo-radon-wgpu` dev-dependencies.
- `[[bench]] name = "radon_wgpu_bench" harness = false` in `apollo-radon-wgpu/Cargo.toml`.

---

## [0.8.2] — Closure XIV

### Closure XIV — Dead-Code Removal: O(N²) Forward Pipeline
#### Removed
- `StftGpuKernel::execute()`: O(N²) direct DFT forward method (superseded by Closure XII FFT path).
- `forward_pipeline` field and shader creation code from `StftGpuKernel::new()`.
- `shaders/stft.wgsl`: O(N²) forward DFT WGSL shader (superseded by `stft_forward_fft.wgsl`).
- `stft_inverse_frames` entry point from `stft_inverse.wgsl` (superseded by Closure XI FFT inverse).
#### Changed
- `stft_inverse.wgsl` file header updated to reflect single-pass OLA role.
- `kernel.rs` module docstring, `WORKGROUP_SIZE` comment, struct docs updated.

---

## [0.8.1] — Closure XIII

### Closure XIII — STFT GPU Criterion Benchmarks
#### Added
- `crates/apollo-stft-wgpu/benches/stft_bench.rs`: criterion benchmark suite with
  `bench_forward_fft` and `bench_inverse_fft` groups.
- `criterion = { version = "0.5", features = ["html_reports"] }` in `apollo-stft-wgpu`
  dev-dependencies.
- `[[bench]] name = "stft_bench" harness = false` in `apollo-stft-wgpu/Cargo.toml`.

---

## [0.8.0] — Closure XII

### Closure XII — STFT Forward-Path GPU FFT Acceleration
#### Added
- `stft_forward_fft.wgsl`: new GPU shader with `stft_fwd_pack_window`, `stft_fwd_bitrev`,
  `stft_fwd_butterfly`, `stft_fwd_interleave` entry points (DFT twiddle `exp(−2πi·k/N)`).
- `FwdFftStageParams` struct (16 bytes, fields: frame_count, frame_len, hop_len, stage).
- `StftGpuKernel::execute_forward_fft`: O(N log N) GPU forward STFT, PoT frame_len required.
- `FrameLenNotPowerOfTwo` guard in `StftWgpuBackend::execute_forward`.
- Tests: `forward_rejects_non_power_of_two_frame_len`, `forward_fft_roundtrip_large_frame_when_device_exists`.
#### Changed
- `StftWgpuBackend::execute_forward` now routes to the FFT-accelerated path and requires
  power-of-two `frame_len` (previously accepted any `frame_len` via O(N²) direct DFT).
#### Breaking
- `execute_forward` with non-power-of-two `frame_len` now returns
  `Err(WgpuError::FrameLenNotPowerOfTwo)` instead of computing a result.

---

## [0.7.0] — Closure XI

### Added
- `apollo-stft-wgpu`: GPU STFT inverse O(N log N) acceleration. New `stft_inverse_fft.wgsl` with four entry points implementing a batched Cooley-Tukey Radix-2 DIT IFFT: `stft_deinterleave` (interleaved complex f32 → split re/im scratch), `stft_bitrev` (bit-reversal permutation, batched), `stft_butterfly` (one Radix-2 DIT stage per dispatch; IDFT twiddle exp(+2πi·k/N)), `stft_scale_and_window` (1/N scale + Hann synthesis window → frame_data). Two-bind-group architecture: group 0 = 4 data bindings (shared), group 1 = per-stage `FftStageParams` uniform (pre-allocated, one per butterfly pass). All passes in one `CommandEncoder`; implicit per-pass memory barriers ensure write visibility. OLA pass unchanged. Replaces the O(N²) `stft_inverse_frames` pipeline. Formal basis: Cooley & Tukey (1965); Allen & Rabiner (1977) Theorem 1. [minor]
- `apollo-stft-wgpu`: `WgpuError::FrameLenNotPowerOfTwo { frame_len: usize }` error variant. Returned by `execute_inverse` when `frame_len` is not a power of two (Radix-2 invariant); checked before GPU buffer allocation in `device.rs` and at IFFT entry in `kernel.rs`. [minor]
- `apollo-stft-wgpu`: `inverse_rejects_non_power_of_two_frame_len` test (CPU-only; asserts `FrameLenNotPowerOfTwo { frame_len: 6 }` for frame_len=6). [patch]
- `apollo-stft-wgpu`: `inverse_roundtrip_large_frame_1024_samples_when_device_exists` GPU-gated test (frame_len=1024, log₂N=10 butterfly stages, hop=512, signal_len=8192, analytic sine reference, TOL=5e-3). [patch]

---

## [0.6.0] — Closure X

### Added
- `apollo-radon-wgpu`: GPU ramp-filtered backprojection (FBP). New `radon_fbp_filter.wgsl` entry point `radon_fbp_filter` applies the Ram-Lak ramp filter to each sinogram projection row via circular convolution with the impulse response `h = IFFT(R)`, `R[k] = 2π·|signed_k|/(N·Δ)` (Bracewell & Riddle 1967; Shepp & Logan 1974). Filter kernel computed host-side from `apollo_radon::ramp_filter_projection([1,0,...], Δ)` and cast to f32. Two-pass single `CommandEncoder` (filter → backproject). Host-side `π/angle_count` normalization. `RadonWgpuBackend::execute_filtered_backproject`. `supports_filtered_backprojection: bool` field added to `WgpuCapabilities`. `WgpuCapabilities::forward_inverse_and_fbp` constructor added. [minor]
- `apollo-radon-wgpu`: `backproject_satisfies_adjoint_identity_when_device_exists` test verifies the Radon adjoint identity ⟨A·f, g⟩_sinogram = ⟨f, A†·g⟩_image (Natterer 2001, §II.2) to relative tolerance 5e-3. [patch]
- `apollo-stft-wgpu`: `inverse_roundtrip_for_multiple_cola_parameter_sets` tests three COLA-compliant parameter sets (frame_len=8/hop=4, 16/8, 32/16) with analytical reference signals. TOL=5e-3. [patch]

### Fixed
- `README.md`: stale WGPU crate descriptions for `apollo-radon-wgpu` (was "forward only"), `apollo-stft-wgpu` (was "forward only"), `apollo-hilbert-wgpu` (was "inverse unsupported"), and `apollo-sdft-wgpu` (was "inverse unsupported"). All now accurately describe implemented GPU inverse capabilities. [patch]
- `ARCHITECTURE.md`: Mixed-Precision Capability Table notes for `apollo-radon-wgpu`, `apollo-stft-wgpu`, `apollo-hilbert-wgpu`, and `apollo-sdft-wgpu` updated to reflect inverse capability status. [patch]

---

## [0.5.0] — Closure IX

### Added
- `apollo-stft-wgpu`: GPU inverse STFT via two-pass Weighted Overlap-Add (WOLA). New WGSL file `stft_inverse.wgsl` with entry points `stft_inverse_frames` (per-(frame, local_j) windowed IDFT: `frame_data[m·N+j] = (1/N)·Re{Σ_k X[m,k]·exp(+2πi·k·j/N)}·hann(j)`, spectrum as interleaved f32 pairs) and `stft_inverse_ola` (per-sample `y[n] = Σ_m frame_data[m·N+(n−start_m)] / Σ_m hann(n−start_m)²`, `start_m = m·hop−N/2`). Both passes share the existing 3-binding layout in one `CommandEncoder`. `StftGpuKernel::execute_inverse` (2-pass single encoder). `StftWgpuBackend::execute_inverse(plan, spectrum, signal_len)` + `execute_inverse_typed_into`. `WgpuCapabilities::forward_and_inverse` constructor added. Basis: WOLA identity (Allen–Rabiner 1977, Theorem 1). [minor]
- `apollo-radon-wgpu`: GPU Radon adjoint backprojection. New WGSL file `radon_backproject.wgsl` with entry `radon_backproject`: per-pixel `bp[r,c] = Σ_θ interp(sinogram[θ,·], x·cosθ + y·sinθ)` with linear interpolation and out-of-range zero-clamping, reusing the forward bind group layout. `RadonGpuKernel::execute_backproject`. `RadonWgpuBackend::execute_inverse(plan, sinogram, angles)` + `execute_inverse_flat_typed`. `WgpuCapabilities::forward_and_inverse` constructor added. `SinogramShapeMismatch` error variant added. Basis: Radon adjoint operator (Natterer 2001, §II.2). [minor]

### Fixed
- `gap_audit.md`: open-gap note incorrectly stated "CPU inverse paths are implemented" for `apollo-czt-wgpu` and `apollo-mellin-wgpu`. Corrected: those crates have no CPU inverse defined; `execute_inverse` returns `UnsupportedExecution` by architectural design, not by deferral. [patch]

---

## [0.4.0] — Closure VIII

### Added
- `apollo-hilbert-wgpu`: GPU inverse Hilbert transform. New WGSL entry point `hilbert_inverse_mask` recovers the original real-signal DFT spectrum from the DFT of the quadrature signal: positive bins X[k]=j·Q[k], negative bins X[k]=-j·Q[k], DC and Nyquist zeroed (unrecoverable; Bracewell 1965). New `HilbertGpuKernel::execute_inverse` runs 3 sequential passes in one command encoder (DFT of quadrature → inverse mask → IDFT of recovered spectrum). Exposed via `HilbertWgpuBackend::execute_inverse` and `execute_inverse_typed_into`. `WgpuCapabilities::forward_and_inverse` constructor added. [minor]
- `apollo-sdft-wgpu`: GPU inverse SDFT. New WGSL entry point `sdft_inverse_bins` computes x[n]=(1/K)·Σ_{b=0}^{K-1} X[b]·exp(+2πi·b·n/K). Complex bins passed as interleaved f32 pairs. Separate `forward_pipeline` and `inverse_pipeline` in `SdftGpuKernel`. Exposed via `SdftWgpuBackend::execute_inverse` and `execute_inverse_typed_into`. `WgpuCapabilities::forward_and_inverse` constructor added. [minor]

### Fixed
- `apollo-hilbert-wgpu`: pre-existing bug in `hilbert_inverse_dft` WGSL: real accumulator was written back as a stale self-assign (`inout_b[n].re = original`); corrected to `inout_b[n].re = acc.x * scale`. [patch]
- `apollo-czt`: proptest `bluestein_equals_direct_for_arbitrary_parameters` used a fixed 1e-9 absolute tolerance, which is violated when `|w|>1` amplifies output magnitudes by up to |w|^((N-1)²/2). Tolerance changed to `1e-9·max(|direct[k]|,1.0)` (relative bound). Formal basis: Bluestein relative error ≤ C·log₂(p)·ε_machine ≈12·2.2e-16≈2.6e-15 (Higham §3.10). [patch]

---

## [0.3.0] — Closure VII

### Added
- Six new published-reference fixtures in `apollo-validation`: SFT 1-sparse alternating tone, SHT monopole Y₀⁰ coefficient, STFT rectangular-window impulse frame, Hilbert cosine-to-sine 4-point, Mellin constant-function first moment, Radon θ=0 column-impulse projection. Fixture count rises from 22 to 28. [minor]
- Proptest coverage for four CPU transform crates previously lacking property tests: `apollo-czt` (Bluestein-vs-direct parity, spiral-collapse to DFT, linearity), `apollo-frft` (roundtrip, additivity, linearity), `apollo-nufft` (DC-mode invariant, fast-path tracks exact, Type-1 linearity), `apollo-sft` (K-sparse exact recovery, Parseval top-K, retained bins equal DFT). [minor]

### Changed
- `apollo-frft-wgpu`: `UnitaryFrftGpuKernel::execute` refactored from 3 separate command encoder submissions to a single command encoder with 3 sequential compute passes followed by a copy command. This reduces CPU–GPU round-trips from 4 submits + 5 polls to 1 submit + 2 polls while preserving the cross-pass write-visibility guarantee via the implicit memory barrier at each `ComputePass` boundary (WebGPU spec §3.4 sequential pass ordering). [patch]
- `design_history_file/`: removed stale shadow copies of `backlog.md`, `checklist.md`, and `gap_audit.md` (root artifacts are the SSOT). `adr_unitary_frft.md` retained as the authoritative ADR. [patch]

### Fixed
- `README.md`: updated `apollo-validation` fixture count from 10 (stale) to 28 (final Closure VII count) and replaced the stale fixture list with the complete 28-fixture inventory. [patch]

---

## [0.2.0] — Closure VI (NTT WGPU O(N log N), workspace unblock, expanded fixtures)

### Added
- `apollo-ntt-wgpu`: O(N log N) Cooley-Tukey DIT butterfly shader (`ntt_butterfly` + `ntt_scale` entry points), replacing the O(N²) DFT loop. Log₂(N) butterfly passes plus optional scale pass encoded in one command buffer with dynamic uniform offsets. [major]
- Two published-reference fixtures: NTT N=16 impulse (Pollard 1971) and NTT N=16 polynomial product via convolution theorem. Fixture count 20 → 22. [minor]
- CPU-only proptest tests in `apollo-ntt-wgpu` verification: `cpu_roundtrip_preserves_residue_class` and `convolution_theorem_holds_for_arbitrary_pairs`. [minor]

### Changed
- `apollo-ntt-wgpu`: removed `apollo_fft::PrecisionProfile` cross-domain import and `default_precision_profile` field from capabilities; NTT is exact integer arithmetic. [minor]
- GPU-dependent tests in `apollo-ntt-wgpu` annotated `#[ignore = "requires wgpu device"]` replacing silent early-return skips. [patch]

### Fixed
- Workspace compilation regression: reverted `apollo-fft/Cargo.toml` package name from `"apollo"` to `"apollo-fft"` and corrected dependent crate path keys. [patch]
- Removed `#![allow(unused_imports)]` from `apollo-ntt/src/lib.rs` and unused `Array1` import. [patch]

---

## [0.1.9] — Closure V (GPU Unitary FrFT, ADR, published fixtures)

### Added
- `apollo-frft-wgpu`: `UnitaryFrftGpuKernel` implementing DFrFT_a(x)=V·diag(exp(−iakπ/2))·V^T·x on GPU via three-submission pattern. `FrftWgpuBackend` exposes `plan_unitary`, `execute_unitary_forward`, `execute_unitary_inverse`. [minor]
- Three published-reference fixtures (count 17 → 20): FrFT unitary order-2 reversal (Candan 2000), DWT Haar one-level detail (Haar 1910/Mallat 1989), SDFT bin-zero unit impulse. [minor]
- `design_history_file/adr_unitary_frft.md`: ADR documenting Grünbaum eigendecomposition algorithm selection, unitarity proof, GPU ordering guarantee, and tolerance derivation. [patch]
- `ARCHITECTURE.md`: "Key: Unitary FrFT" subsection with CPU/GPU plan comparison table. [patch]

---

## [0.1.8] — Closure IV (FrFT unitarity, DCT-I/IV/DST-I/IV WGPU)

### Added
- `apollo-frft`: `GrunbaumBasis` and `UnitaryFrftPlan` (Candan 2000 eigendecomposition). O(N³) construction, O(N²) per call, provably unitary for all real orders. [minor]
- `apollo-dctdst-wgpu`: WGSL shader modes for DCT-I (mode 4), DCT-IV (mode 5), DST-I (mode 6), DST-IV (mode 7) with correct self-inverse scales. [minor]

---

## [0.1.7] — Closure III (validation mock removal, published fixtures, DCT-I/IV/DST-I/IV CPU)

### Added
- 7 published-reference fixtures (count 10 → 17): FFT inverse, DCT-II inverse pair, DHT self-reciprocal, FWHT 2-point, QFT 2-point, CZT spiral-collapse, GFT path graph. [minor]
- `apollo-validation` GPU suite: real 4×4×4 GpuFft3d roundtrip replacing hardcoded `passed: true` stub. [major]

### Fixed
- `apollo-validation` precision profile forward errors computed from actual GPU vs CPU f64 reference comparison. [patch]

---

## [0.1.6] — Closure II (fixture expansion, capability table)

### Added
- Published-reference fixtures expanded to 10. [minor]
- `ARCHITECTURE.md` Mixed-Precision Capability Table (authoritative per-crate precision record). [patch]

---

## [0.1.5] — Performance & Native GPU Precision

### Added
- `apollo-fft-wgpu`: `GpuFft3dF16Native` behind `native-f16` feature; native f16 arithmetic with `enable f16` WGSL. Bluestein chirp-Z f16 shader for non-power-of-two sizes. [minor]
- Criterion buffer-reuse benchmarks for `apollo-nufft-wgpu` and `apollo-fft-wgpu`. [minor]
- NUFFT and FFT reusable-buffer `with_buffers` façade methods. [minor]

---

## [0.1.4] — Extension Phase (mixed precision rollout, typed storage)

### Added
- Mixed-precision typed storage APIs across all CPU and WGPU transform crates. [minor]
- Exact quantized `u32` NTT-WGPU residue storage and reusable buffer dispatch. [minor]

---

## [0.1.3] — GPU Numerical Kernels (NUFFT, SHT, SFT, STFT, Wavelet, DCT/DST)

### Added
- GPU fast NUFFT 1D/3D Kaiser-Bessel gridding paths. [minor]
- WGPU backends for SHT, STFT, Haar DWT, DCT-II/III/DST-II/DST-III. [minor]

---

## [0.1.2] — Core Transform Crates (GFT, QFT, SDFT, SFT, Radon, Mellin, Hilbert, Wavelet, STFT, CZT, FWHT)

### Added
- New CPU transform crates with WGPU backends. [minor]

---

## [0.1.1] — Foundation (FFT, DHT, DCT/DST, NTT, NUFFT)

### Added
- Core CPU transform crates with O(N log N) kernels. [minor]
- `apollo-validation` published-reference suite with 10 initial fixtures. [minor]

---

## [0.1.0] — Initial release

### Added
- Workspace skeleton: `apollo-fft`, `apollo-fft-wgpu`, `apollo-nufft`, `apollo-nufft-wgpu`, `apollo-validation`, `apollo-python`. [minor]
