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

The table below is the authoritative record of per-crate precision support. "Advertised profile" is the `default_precision_profile` value exposed in each crate's capability struct. "Supported storage" lists accepted host-storage types for typed/caller-owned APIs. "GPU compute" is the arithmetic precision used inside shaders or GPU kernels.

| Crate | Backend | Advertised profile | Supported storage | GPU compute | Notes |
|---|---|---|---|---|---|
| apollo-fft | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 at plan boundary |
| apollo-czt | CPU | HIGH_ACCURACY | Complex64, Complex32, [f16;2] | — | f16 promoted to Complex32 |
| apollo-dctdst | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-dht | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-frft | CPU | HIGH_ACCURACY | Complex64, Complex32, [f16;2] | — | f16 promoted to Complex32 |
| apollo-fwht | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-gft | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-hilbert | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-mellin | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-ntt | CPU | exact u64 residues | u64 mod p | — | floating mixed precision unsupported by design |
| apollo-nufft | CPU | HIGH_ACCURACY | Complex64, Complex32, [f16;2] | — | f16 promoted to Complex32 |
| apollo-qft | CPU | HIGH_ACCURACY | Complex64, Complex32, [f16;2] | — | f16 promoted to Complex32 |
| apollo-radon | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-sdft | CPU | HIGH_ACCURACY | f64/Complex64, f32/Complex32, f16/[f16;2] | — | f16 promoted to f32/Complex32 |
| apollo-sft | CPU | HIGH_ACCURACY | Complex64, Complex32, [f16;2] | — | f16 promoted to Complex32 |
| apollo-sht | CPU | HIGH_ACCURACY | f64/Complex64, f32/Complex32, f16/[f16;2] | — | f16 promoted to f32/Complex32 |
| apollo-stft | CPU | HIGH_ACCURACY | f64/Complex64, f32/Complex32, f16/[f16;2] | — | f16 promoted to f32/Complex32 |
| apollo-wavelet | CPU | HIGH_ACCURACY | f64, f32, half::f16 | — | f16 promoted to f32 |
| apollo-fft-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 (mixed) | f32 | native-f16 feature: arithmetic in f16 via SHADER_F16 |
| apollo-czt-wgpu | WGPU | LOW_PRECISION_F32 | f32, [f16;2] host (mixed) | f32 | forward + inverse CZT; f16 promoted to f32 at host boundary |
| apollo-dctdst-wgpu | WGPU | LOW_PRECISION_F32 | f32 | f32 | mixed f16 host path present |
| apollo-dht-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | f16 promoted at host boundary |
| apollo-frft-wgpu | WGPU | LOW_PRECISION_F32 | f32, [f16;2] host (mixed) | f32 | f16 promoted at host boundary; UnitaryFrftGpuKernel available |
| apollo-fwht-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | f16 promoted at host boundary |
| apollo-gft-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | f16 promoted at host boundary |
| apollo-hilbert-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | forward + inverse analytic-mask; f16 promoted at host boundary |
| apollo-mellin-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | forward + inverse Mellin spectrum; f16 promoted at host boundary |
| apollo-ntt-wgpu | WGPU | exact u32 residues | u32 quantized | u32 modular | floating mixed precision explicitly unsupported |
| apollo-nufft-wgpu | WGPU | LOW_PRECISION_F32 | Complex32, [f16;2] host (mixed) | f32 | f16 promoted at host boundary |
| apollo-qft-wgpu | WGPU | LOW_PRECISION_F32 | Complex32, [f16;2] host (mixed) | f32 | f16 promoted at host boundary |
| apollo-radon-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | forward + adjoint backprojection + FBP; f16 promoted at host boundary |
| apollo-sdft-wgpu | WGPU | LOW_PRECISION_F32 | f32, [f16;2] host (mixed) | f32 | forward + inverse direct-bins IDFT; f16 promoted at host boundary |
| apollo-sft-wgpu | WGPU | LOW_PRECISION_F32 | Complex32, [f16;2] host (mixed) | f32 | f16 promoted at host boundary |
| apollo-sht-wgpu | WGPU | LOW_PRECISION_F32 | Complex32, [f16;2] host (mixed) | f32 | f16 promoted at host boundary |
| apollo-stft-wgpu | WGPU | LOW_PRECISION_F32 | Complex32, [f16;2] host (mixed) | f32 | forward + inverse FFT-accelerated (Radix-2 DIT, O(N log N)); PoT frame_len required; f16 promoted at host boundary |
| apollo-wavelet-wgpu | WGPU | LOW_PRECISION_F32 | f32, half::f16 host (mixed) | f32 | f16 promoted at host boundary |

### Key: native-f16 GPU (apollo-fft-wgpu)

When the `native-f16` feature is enabled and the WGPU adapter exposes `wgpu::Features::SHADER_F16`, `GpuFft3dF16Native` executes all butterfly arithmetic in `f16` inside the shader. The host boundary converts `f32` input to `f16` before upload and `f16` output to `f32` after readback. Twiddle factors are computed in `f32` then narrowed to `f16` at plan build time to bound two-source error. Accumulation error is `O(log N)·ε_f16` where `ε_f16 ≈ 9.77×10⁻⁴`. Non-power-of-two sizes are supported via a Bluestein chirp-Z f16 shader (`chirp_native_f16.wgsl`).

### Key: NTT precision contract

`apollo-ntt` and `apollo-ntt-wgpu` operate exclusively on exact modular residues. Floating-point mixed precision is architecturally unsupported because modular arithmetic requires exact integer representation. The WGPU surface uses `u32` residues (values mod p where p ≤ u32::MAX); the CPU surface uses `u64` residues for the default 998244353 modulus with 128-bit-widened intermediate products.

### Key: Unitary FrFT (apollo-frft, apollo-frft-wgpu)

Two plans coexist for the fractional Fourier transform:

| Plan | Construction | Per-call | Unitarity |
|---|---|---|---|
| `FrftPlan` | O(1) | O(N²) | Non-unitary for non-integer orders (‖M†M‖[j,j] = 1/|sin α|) |
| `UnitaryFrftPlan` | O(N³) | O(N²) | Provably unitary: ‖DFrFT_a(x)‖₂ = ‖x‖₂ for all real a |

`UnitaryFrftPlan` uses the Candan (2000) eigendecomposition of the Grünbaum commuting matrix.
Its eigenvector basis V satisfies V^T V = I and eigenvectors are symmetric or antisymmetric
under index reversal. DFrFT_a(x) = V · diag(exp(−iakπ/2)) · V^T · x.

The GPU backend `apollo-frft-wgpu` exposes `execute_unitary_forward` and `execute_unitary_inverse`
via `UnitaryFrftGpuKernel`. V is precomputed on CPU (O(N³)) and uploaded as an f32 storage buffer.
Three sequential GPU submissions execute the 3-pass algorithm with `device.poll` barriers between
passes to guarantee cross-workgroup storage ordering.

See `design_history_file/adr_unitary_frft.md` for the full algorithm selection record.
