# Apollo Architecture

## Dependency Rules

- `domain` defines the single source of truth for shapes, normalization, capabilities, device descriptors, and error contracts.
- `application` owns orchestration, cache policy, reusable plans, and zero-allocation execution paths.
- `infrastructure` owns concrete adapters such as CPU and WGPU.
- Public APIs are exposed from `lib.rs` and narrow facade modules only.

Allowed dependency direction:

`api/lib.rs -> application -> domain`

`api/lib.rs -> infrastructure -> application -> domain`

Infrastructure implementations may depend on shared traits and shared data contracts, but `domain` and `application` may not depend on a backend-specific module.

## SOC, SSOT, SRP, DIP, DRY

- Separation of concerns: numerical contracts, execution plans, and backend transport are split into separate modules.
- Single source of truth: shape metadata, normalization conventions, cache keys, and backend capability descriptors are defined once in `domain`.
- Single responsibility: each plan type owns exactly one dimensionality and one normalization convention.
- Dependency inversion: consumers program against `FftBackend`, not against CPU or GPU internals.
- Do not repeat yourself: helper constructors and shared validation live centrally and are reused by all crates.
- Recent (perf pass): introduced kernel/pot/ with ZST PoTStrategy/SizedPoT for monomorphized schedules; butterflies/ placeholder populated (mul_conj shared from rader negacyclic CRT; deep vertical, reduce redundancy across winograd/stockham/good_thomas/rader). Rader: f32 runtime primes (m>=256 +113) bias to Bluestein+Stockham PoT (targets worst bench Rader ratios) via factored prefers_bluestein_for_rader (ordered path synced); rader/bluestein on TL pooled scratch. Selection prefers composite for smooth over slow GT static (dispatch/plan). Stockham has dedicated cases for 128/256. f32 small PoT unified to pooled scratch + shared kernels for memory/stack + perf.
Cleanup/planning (this cycle): PoT ZSTs marked with rationale (pending wire); shared gather_unroll4 wired to GT/rader + mul_conj already; routing audit + documented next (shared butterflies expansion #1 for dupe/mono, ZST wire, f32 unify for routing safety, more GT/rader opts) with prob/why (bench ratios + math overhead + arch DIP/SoC) in gap_audit. Gates and value-semantic held.
This stage: butterflies/dft.rs hosts canonical dft2/3/4/5/7/8 (moved, wired from winograd+GT cook); refactor for shared, verified no regression (tests 64+ GT, 32+ dft, rader etc pass; benchmark note + rebench cmd for key sizes).
PoT ZST wiring (this sub-stage): log2 + SizedPoT<StockhamAutosort, LOG2> (exact via ::new()) into plan PowerOfTwo + dispatch; 512 explicit + sized helper + new value tests (ZST<9> + dft match). Preserved specials/pools. Bench attention (xtask PoT sizes attempts + md notes). Makes PoT routing stronger zero-cost mono. No regression.
Next phase (perf opt/mem eff/arch elev): stockham ZST with_strategy used (elev call site in mod for 512); plan/dispatch ZST constructions; bluestein kernel pooled (with for build temp, mem eff); Cow exercised in fold; cast native f32 chirp (no excess). n113 ignored; rader (pooled), GT, ZST plan, dft green. Gates; bg bench on full worst launched. Md synced (no regression). See gap, benchmark_results.
- Body unrolls for PoT worst (n32): per-LOG2 n32 radix1 unrolled (Inner-Fn + explicit no-loop in avx stage for vector iters; scalar 4x j0 explicit in non-avx) routed from len32 via P stage n==32 guard. Targets 32 ratios (highest in md); 0-cost additive mono/ILP; value preserved; artifacts updated.
This phase (deeper mono + elevation): explicit per-LOG2 Stockham bodies (len32/64/128/256/512/1024 straight-line stage seq from structural const LOG2 + schedule-derived subslices; dispatch in transform_impl; 128/256 now direct). Extended ZST with_strategy for 5-10 in f32/f64 scalar kernel entries (live from plan ZST tags). SRP: transform.rs owns the per-size (deep vertical). Mem: Cow extended (kernel_view bluestein + scratch views). Cast hygiene. Highest-prob for PoT md-worst (const seq bakes fusion/stage count). Value on 32 (direct), 128/256/512/1024 (round), n512 ZST, rader bluestein (Cow/pooled), GT90/198; 252p broad; gates clean; focused --skip-run build success + md "no regression" (identical to general + 0-cost additive). See gap_audit (verification table), benchmark_results, checklist. Residual: more unrolls inside lens, f32 scratch full, full rebench.
Follow-up (ZST threading + Cow): pot_inplace_sized<S: PoTStrategy, const LOG2> added (trait default + f32/f64 impls); plan sized execs + dispatch constructions now pass the _s (direct threading of strategy ZST + const LOG2 from plan SSOT into monomorphized kernel boundary for better const prop/folding to lenXXX; Cow tw_view in the sized pot impls). Arch: mono/ZST/zero-cost/DIP elevated. Value/gates/bench/md synced. See gap, benchmark_results.
- Completion (overrides + full wire for 128/256): stockham_*_sized + pot_inplace_sized overrides (const LOG2 + Cow); plan 7/8 arms + dispatch actual calls to pot_sized (end-to-end for 128/256/512/1024); incidental preexist fixes (4 arm, dft16, short guard). Enables const to all md PoT worst len bodies. Value/gates/build+focused/md updated. No regression. See gap, benchmark_results (latest attempt), checklist.
- More direct ZST (AVX dispatch): avx_with_scratch_sized<const LOG2> (f64/f32) + wiring in mod sized (bypass runtime log2 in avx path); LOG2 to transform_sized. Prep for fixed 128/256. See gap residual.

## Precision Model

- Precision is a domain-level contract, not an incidental backend detail.
- `PrecisionMode`, `StoragePrecision`, `ComputePrecision`, and `PrecisionProfile` are defined once
  in Apollo domain types and reused across Rust, Python, validation, and compatibility layers.
- Backends must advertise only the precision profiles they truly implement.
- Apollo never silently upgrades or downgrades a caller into mixed precision; lower-precision paths
  are explicit plan or API choices.
- Apollo currently defines `mixed_precision` for CPU FFT as `half::f16` storage with `f32` compute.

## Documentation Standard

All public types and methods must document:

- The algorithm family in use.
- The key theorem or invariant relied upon.
- A proof sketch or reasoning note.
- Complexity and allocation behavior.
- Normalization rules.
- Failure modes.

## Mixed-Precision Capability Table

The table below is the authoritative record of per-crate precision support. Each core transform crate now exposes its CPU backend by default, and its GPU backend when the optional `wgpu` feature is enabled.

| Crate | CPU Backend | GPU Backend (enabled via `wgpu` feature) | Notes |
|---|---|---|---|
| apollo-fft | HIGH_ACCURACY (f64/f32/f16 storage) | LOW_PRECISION_F32 (f32/f16 storage, f32 compute) | `native-f16` feature enables native GPU f16 compute |
| apollo-czt | HIGH_ACCURACY (Complex64/32/f16) | LOW_PRECISION_F32 (f32/[f16;2] storage, f32 compute) | Forward/inverse CZT; f16 promoted at host boundary |
| apollo-dctdst | HIGH_ACCURACY (f64/f32/f16 storage) | LOW_PRECISION_F32 (f32 storage, f32 compute) | Mixed f16 host path present |
| apollo-dht | HIGH_ACCURACY (f64/f32/f16 storage) | LOW_PRECISION_F32 (f32/f16 storage, f32 compute) | f16 promoted at host boundary |
| apollo-frft | HIGH_ACCURACY (Complex64/32/f16) | LOW_PRECISION_F32 (Complex32/[f16;2] storage, f32 compute) | Hephaestus typed direct/unitary kernels; Complex64 excluded from GPU |
| apollo-fwht | HIGH_ACCURACY (f64/f32/f16 storage) | LOW_PRECISION_F32 (f32/f16 storage, f32 compute) | f16 promoted at host boundary |
| apollo-gft | HIGH_ACCURACY (f64/f32/f16 storage) | LOW_PRECISION_F32 (f32/f16 storage, f32 compute) | f16 promoted at host boundary |
| apollo-hilbert | HIGH_ACCURACY (f64/f32/f16 storage) | LOW_PRECISION_F32 (f32/f16 storage, f32 compute) | Typed Hephaestus analytic/inverse masks; f16 promoted and f64 excluded at GPU boundary |
| apollo-mellin | HIGH_ACCURACY (f64/f32/f16 storage) | LOW_PRECISION_F32 (f32/f16 storage, f32 compute) | Typed Hephaestus log-grid forward/inverse; f16 promoted and f64 excluded at GPU boundary |
| apollo-ntt | exact u64 residues (u64 mod p) | exact u32 residues (u32 quantized, u32 modular) | Floating mixed-precision unsupported by design |
| apollo-nufft | HIGH_ACCURACY (Complex64/32/f16) | LOW_PRECISION_F32 (Complex32/[f16;2] storage, f32 compute) | f16 promoted at host boundary |
| apollo-qft | HIGH_ACCURACY (Complex64/32/f16) | LOW_PRECISION_F32 (Complex32/[f16;2] storage, f32 compute) | f16 promoted at host boundary |
| apollo-radon | HIGH_ACCURACY (f64/f32/f16 storage) | LOW_PRECISION_F32 (f32/f16 storage, f32 compute) | Forward/adjoint backprojection; f16 promoted at boundary |
| apollo-sdft | HIGH_ACCURACY (f64/f32/f16 storage) | LOW_PRECISION_F32 (f32/[f16;2] storage, f32 compute) | Forward/inverse direct-bins IDFT; f16 promoted at boundary |
| apollo-sft | HIGH_ACCURACY (Complex64/32/f16) | LOW_PRECISION_F32 (Complex32/[f16;2] storage, f32 compute) | f16 promoted at host boundary |
| apollo-sht | HIGH_ACCURACY (Complex64/32/f16) | LOW_PRECISION_F32 (Complex32/[f16;2] storage, f32 compute) | f16 promoted at host boundary |
| apollo-stft | HIGH_ACCURACY (Complex64/32/f16) | LOW_PRECISION_F32 (Complex32/[f16;2] storage, f32 compute) | Forward/inverse FFT-accelerated (Radix-2 DIT); f16 promoted |
| apollo-wavelet | HIGH_ACCURACY (f64/f32/f16 storage) | LOW_PRECISION_F32 (f32/f16 storage, f32 compute) | f16 promoted at host boundary |

### Key: native-f16 GPU (apollo-fft wgpu feature)

When the `native-f16` feature is enabled, `GpuFft3dF16Native` requires
`DeviceFeature::ShaderF16` from `hephaestus_wgpu::WgpuDevice` and executes all
butterfly arithmetic in `f16` inside the shader. The host boundary converts
`f32` input to half bit patterns before upload and half output to `f32` after
readback. Callers acquire the feature-qualified device directly from
Hephaestus and pass it to `GpuFft3dF16Native::try_from_device`. The sealed FFT
storage contract reuses the f32 typed descriptor and
command stream while selecting f16 WGSL and radix-two entries. Twiddle factors
are computed in `f32` then narrowed to `f16`. Non-power-of-two sizes use the
Bluestein chirp-Z shader (`chirp_native_f16.wgsl`); the 3×3×3 roundtrip test
uses the derived `γ_265·‖input‖₁` bound with half unit roundoff `u = 2⁻¹¹`.
No Apollo-owned WGPU API or provider-acquisition wrapper remains in this path.

### Key: NTT precision contract

`apollo-ntt` operates exclusively on exact modular residues. Floating-point mixed precision is architecturally unsupported because modular arithmetic requires exact integer representation. The GPU backend (with the `wgpu` feature enabled) uses `u32` residues (values mod p where p ≤ u32::MAX); the CPU backend uses `u64` residues for the default 998244353 modulus with 128-bit-widened intermediate products.

### Key: Unitary FrFT (apollo-frft wgpu feature)

Two plans coexist for the fractional Fourier transform:

| Plan | Construction | Per-call | Unitarity |
|---|---|---|---|
| `FrftPlan` | O(1) | O(N²) | Non-unitary for non-integer orders (‖M†M‖[j,j] = 1/|sin α|) |
| `UnitaryFrftPlan` | O(N³) | O(N²) | Provably unitary: ‖DFrFT_a(x)‖₂ = ‖x‖₂ for all real a |

`UnitaryFrftPlan` uses the Candan (2000) eigendecomposition of the Grünbaum commuting matrix.
Its eigenvector basis V satisfies V^T V = I and eigenvectors are symmetric or antisymmetric
under index reversal. DFrFT_a(x) = V · diag(exp(−iakπ/2)) · V^T · x.

The GPU backend (with the `wgpu` feature enabled) exposes `execute_unitary_forward` and `execute_unitary_inverse`
through a typed Hephaestus `UnitaryFrftKernel`. V is precomputed on CPU by Leto
(O(N³)) and uploaded as a column-major `f32` storage buffer. One Hephaestus
command stream encodes the projection, phase, and reconstruction as three
ordered passes; stream boundaries provide the inter-pass dependency ordering.

See `design_history_file/adr_unitary_frft.md` for the full algorithm selection record.
